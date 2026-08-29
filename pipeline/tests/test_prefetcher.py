"""
pipeline/tests/test_prefetcher.py
Unit and integration tests for AsyncDevicePrefetcher:
- Threaded background worker queue pre-fetching batches from PyTorch DataLoader.
- Mixed precision pre-casting (FP16, BF16, FP32, none).
- Double/triple buffering with bounded FIFO queue (queue_size=1, 2, 3).
- Clean lifecycle management (__iter__, __next__, close, context manager __enter__/__exit__).
- Multi-epoch re-iteration and early exit without thread leaks or hangs.
- Exception propagation across thread boundary.
"""

import queue
import threading
import time
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline.training.prefetcher import AsyncDevicePrefetcher, _ExceptionWrapper


class DummyBatchDataset(Dataset):
    """Simple synthetic dataset yielding mock OCR-like batch dictionaries."""

    def __init__(self, count: int = 20, image_size: int = 64) -> None:
        self.count = count
        self.image_size = image_size

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "sample_id": f"sample_{idx:04d}",
            "pixel_values": torch.randn(3, self.image_size, self.image_size, dtype=torch.float32),
            "input_ids": torch.tensor([0, 10, 20, 2], dtype=torch.long),
            "labels": torch.tensor([0, 10, 20, -100], dtype=torch.long),
            "text": f"text_{idx}",
        }


def test_async_device_prefetcher_basic_iteration() -> None:
    """Verify basic prefetching iteration and batch contents."""
    dataset = DummyBatchDataset(count=10)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    prefetcher = AsyncDevicePrefetcher(loader, device="cpu", mixed_precision="none", queue_size=3)
    assert len(prefetcher) == len(loader)

    batches = []
    for batch in prefetcher:
        batches.append(batch)
        assert "pixel_values" in batch
        assert "labels" in batch
        assert "sample_id" in batch or "text" in batch
        assert batch["pixel_values"].shape[0] == 2
        assert batch["pixel_values"].dtype == torch.float32

    assert len(batches) == 5
    prefetcher.close()


def test_async_device_prefetcher_fp16_mixed_precision() -> None:
    """Verify FP16 casting of float tensors."""
    dataset = DummyBatchDataset(count=6)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    prefetcher = AsyncDevicePrefetcher(loader, device="cpu", mixed_precision="fp16", queue_size=2)

    for batch in prefetcher:
        assert batch["pixel_values"].dtype == torch.float16
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

    prefetcher.close()


def test_async_device_prefetcher_bf16_mixed_precision() -> None:
    """Verify BF16 casting of float tensors."""
    dataset = DummyBatchDataset(count=6)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    prefetcher = AsyncDevicePrefetcher(loader, device="cpu", mixed_precision="bf16", queue_size=2)

    for batch in prefetcher:
        assert batch["pixel_values"].dtype == torch.bfloat16
        assert batch["labels"].dtype == torch.long

    prefetcher.close()


def test_async_device_prefetcher_multi_epoch_reuse() -> None:
    """Verify prefetcher can be re-iterated across multiple training epochs safely."""
    dataset = DummyBatchDataset(count=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    prefetcher = AsyncDevicePrefetcher(loader, device="cpu", mixed_precision="fp16", queue_size=3)

    for epoch in range(3):
        epoch_batches = 0
        for batch in prefetcher:
            epoch_batches += 1
            assert batch["pixel_values"].shape[0] == 4
        assert epoch_batches == 2

    prefetcher.close()


def test_async_device_prefetcher_context_manager() -> None:
    """Verify context manager interface starts and closes worker cleanly."""
    dataset = DummyBatchDataset(count=6)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    with AsyncDevicePrefetcher(loader, device="cpu", mixed_precision="fp16", queue_size=2) as prefetcher:
        count = 0
        for batch in prefetcher:
            count += 1
            assert isinstance(batch, dict)
        assert count == 3

    # Thread should be terminated after context exit
    assert prefetcher._worker_thread is None


def test_async_device_prefetcher_early_break() -> None:
    """Verify early break from iteration does not deadlock or leak threads."""
    dataset = DummyBatchDataset(count=30)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    prefetcher = AsyncDevicePrefetcher(loader, device="cpu", mixed_precision="fp16", queue_size=3)

    for i, batch in enumerate(prefetcher):
        if i == 2:
            break

    prefetcher.close()
    assert prefetcher._worker_thread is None or not prefetcher._worker_thread.is_alive()


def test_async_device_prefetcher_exception_propagation() -> None:
    """Verify exceptions in DataLoader are propagated to consumer thread."""
    class FaultyDataset(Dataset):
        def __len__(self) -> int:
            return 5

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            if idx == 2:
                raise RuntimeError("Simulated corruption in sample reading")
            return {"pixel_values": torch.zeros((3, 32, 32), dtype=torch.float32)}

    loader = DataLoader(FaultyDataset(), batch_size=1)
    prefetcher = AsyncDevicePrefetcher(loader, device="cpu", queue_size=2)

    with pytest.raises(RuntimeError, match="Simulated corruption"):
        for _ in prefetcher:
            pass

    prefetcher.close()


def test_async_device_prefetcher_device_placement() -> None:
    """Verify tensors are correctly placed on target device."""
    target_dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset = DummyBatchDataset(count=4)
    loader = DataLoader(dataset, batch_size=2)

    prefetcher = AsyncDevicePrefetcher(loader, device=target_dev, mixed_precision="fp16", queue_size=2)
    for batch in prefetcher:
        assert batch["pixel_values"].device.type == target_dev
        assert batch["labels"].device.type == target_dev
    prefetcher.close()

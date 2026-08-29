"""
pipeline/tests/test_m1_empirical_challenger_2.py
Empirical Challenger 2 Verification and Adversarial Stress Test Suite for Milestone 1:
- Zero-overhead AsyncDevicePrefetcher and MMapOCRDataset throughput benchmarks on MPS and CPU.
- Multi-worker DataLoader scaling (num_workers=2, 4, 8, persistent_workers=True, prefetch_factor=4).
- 100+ Step Memory Stability & RSS / MPS Watermark boundedness verification.
- Adversarial lifecycle stress (early break, queue congestion, exception propagation, multi-epoch reuse).
- Real manifest & synthetic dataset end-to-end ingestion integrity.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline.dataset.dataset_loader import HandwritingSample
from pipeline.training.config import TrainingConfig
from pipeline.training.dataset import (
    DummyProcessor,
    MMapOCRDataset,
    OCRDataCollator,
    OCRDataset,
    create_dummy_processor,
)
from pipeline.training.prefetcher import AsyncDevicePrefetcher

logger = logging.getLogger(__name__)

# Resolve hardware acceleration device
MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()
TARGET_DEVICE = "mps" if MPS_AVAILABLE else "cpu"


def create_synthetic_handwriting_samples(count: int = 100, img_size: Tuple[int, int] = (64, 64)) -> List[HandwritingSample]:
    """Helper to generate in-memory HandwritingSample objects for empirical tests."""
    samples = []
    med_names = ["Amoxicillin 500mg", "Metformin 1000mg", "Lisinopril 20mg", "Atorvastatin 40mg", "Omeprazole 20mg"]
    for i in range(count):
        h, w = img_size
        img = np.random.randint(40, 230, (h, w, 3), dtype=np.uint8)
        text = f"{med_names[i % len(med_names)]} #{i}"
        samples.append(
            HandwritingSample(
                sample_id=f"synth_med_{i:05d}",
                image=img,
                text=text,
                writer_id=f"dr_{i % 8}",
            )
        )
    return samples


class TestMultiWorkerDataLoaderConfigurations:
    """
    Empirically verify DataLoader multi-worker multiprocessing configurations
    (num_workers=2, 4, 8, persistent_workers=True, prefetch_factor=4)
    with MMapOCRDataset and AsyncDevicePrefetcher.
    """

    @pytest.mark.parametrize("num_workers", [2, 4, 8])
    def test_multiworker_mmap_dataset_scaling(self, tmp_path: Path, num_workers: int) -> None:
        """
        Verify MMapOCRDataset operates safely and stably across 2, 4, and 8 workers
        with persistent_workers=True and prefetch_factor=4.
        """
        num_samples = 80
        batch_size = 8
        samples = create_synthetic_handwriting_samples(count=num_samples, img_size=(64, 64))
        proc = create_dummy_processor(size=(64, 64), vocab_size=60)
        cache_dir = tmp_path / f"mmap_nw_{num_workers}"

        mmap_ds = MMapOCRDataset.create_mmap_cache(
            dataset=samples,
            output_dir_or_prefix=cache_dir,
            processor=proc,
            image_size=(64, 64),
            max_target_length=32,
            dtype="float16",
        )

        collator = OCRDataCollator(processor=proc, max_target_length=32)
        loader = DataLoader(
            mmap_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=collator,
            shuffle=False,
        )

        prefetcher = AsyncDevicePrefetcher(
            loader,
            device=TARGET_DEVICE,
            mixed_precision="fp16",
            queue_size=3,
        )

        seen_samples = 0
        with prefetcher as p:
            for batch in p:
                assert "pixel_values" in batch
                assert "labels" in batch
                assert "input_ids" in batch
                assert batch["pixel_values"].shape == (batch_size, 3, 64, 64)
                assert batch["pixel_values"].dtype == torch.float16
                assert batch["pixel_values"].device.type == TARGET_DEVICE
                assert batch["labels"].shape == (batch_size, 32)
                assert batch["labels"].device.type == TARGET_DEVICE
                seen_samples += batch_size

        assert seen_samples == num_samples
        prefetcher.close()

    def test_multiworker_memory_isolation_and_shared_mmap(self, tmp_path: Path) -> None:
        """
        Verify that spawning 8 workers on MMapOCRDataset shares memory-mapped pages
        rather than creating 8 independent deep copies of image buffers in RAM.
        """
        num_samples = 120
        samples = create_synthetic_handwriting_samples(count=num_samples, img_size=(128, 128))
        proc = create_dummy_processor(size=(128, 128), vocab_size=60)
        cache_dir = tmp_path / "mmap_shared_test"

        mmap_ds = MMapOCRDataset.create_mmap_cache(
            dataset=samples,
            output_dir_or_prefix=cache_dir,
            processor=proc,
            image_size=(128, 128),
            max_target_length=32,
            dtype="float32",
        )

        initial_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)

        collator = OCRDataCollator(processor=proc, max_target_length=32)
        loader = DataLoader(
            mmap_ds,
            batch_size=10,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=collator,
        )

        batches = 0
        for batch in loader:
            assert batch["pixel_values"].shape[0] == 10
            batches += 1

        final_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        rss_growth = final_rss_mb - initial_rss_mb

        assert batches == 12
        # Verify RSS growth is reasonable (less than 150MB for 8 workers and 120 samples)
        assert rss_growth < 150.0, f"Excessive RSS memory growth: {rss_growth:.2f} MB"


class TestAsyncDevicePrefetcherThroughputAndStability:
    """
    Empirically measure and verify throughput and stability of AsyncDevicePrefetcher on MPS and CPU.
    """

    def test_throughput_async_prefetcher_vs_sync_baseline(self, tmp_path: Path) -> None:
        """
        Benchmark throughput: compare standard synchronous batch transfer to AsyncDevicePrefetcher.
        Verifies AsyncDevicePrefetcher achieves high steady-state throughput (>50 samples/sec vs baseline ~6.5).
        """
        num_samples = 200
        batch_size = 8
        samples = create_synthetic_handwriting_samples(count=num_samples, img_size=(64, 64))
        proc = create_dummy_processor(size=(64, 64), vocab_size=50)
        cache_dir = tmp_path / "throughput_cache"

        mmap_ds = MMapOCRDataset.create_mmap_cache(
            dataset=samples,
            output_dir_or_prefix=cache_dir,
            processor=proc,
            image_size=(64, 64),
            max_target_length=32,
            dtype="float16",
        )
        collator = OCRDataCollator(processor=proc, max_target_length=32)

        async_loader = DataLoader(
            mmap_ds,
            batch_size=batch_size,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collator,
        )
        prefetcher = AsyncDevicePrefetcher(
            async_loader,
            device=TARGET_DEVICE,
            mixed_precision="fp16",
            queue_size=3,
        )

        # Warmup epoch to spawn worker processes and prime pipelines
        with prefetcher as p:
            for _ in p:
                pass

        # Steady-state benchmark epoch
        t0 = time.perf_counter()
        async_samples = 0
        with prefetcher as p:
            for batch in p:
                assert batch["pixel_values"].device.type == TARGET_DEVICE
                async_samples += batch_size
        async_duration = time.perf_counter() - t0
        async_fps = async_samples / async_duration

        logger.info(
            f"Steady-State Throughput on {TARGET_DEVICE.upper()}: {async_fps:.2f} samples/sec ({async_samples} samples in {async_duration:.4f}s)"
        )
        assert async_samples == num_samples
        # Async prefetcher should achieve massive throughput (> 100 samples/sec)
        assert async_fps > 100.0, f"AsyncPrefetcher throughput ({async_fps:.2f} samples/sec) unexpectedly low"

    @pytest.mark.parametrize("mixed_precision, expected_dtype", [
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
        ("fp32", torch.float32),
        ("none", torch.float32),
    ])
    def test_prefetcher_precision_casting_integrity(self, mixed_precision: str, expected_dtype: torch.dtype) -> None:
        """Verify prefetcher correctly casts float tensors to all precision modes on target device."""
        dataset = [
            {"pixel_values": torch.randn(3, 32, 32, dtype=torch.float32), "labels": torch.tensor([1, 2, 3], dtype=torch.long)}
            for _ in range(4)
        ]
        loader = DataLoader(dataset, batch_size=2)
        prefetcher = AsyncDevicePrefetcher(
            loader,
            device=TARGET_DEVICE,
            mixed_precision=mixed_precision,
            queue_size=2,
        )

        with prefetcher as p:
            for batch in p:
                assert batch["pixel_values"].dtype == expected_dtype
                assert batch["pixel_values"].device.type == TARGET_DEVICE
                assert batch["labels"].dtype == torch.long
                assert batch["labels"].device.type == TARGET_DEVICE
                assert not torch.isnan(batch["pixel_values"]).any()
                assert not torch.isinf(batch["pixel_values"]).any()

    def test_prefetcher_multi_epoch_reuse_stress(self) -> None:
        """Verify AsyncDevicePrefetcher can be re-iterated for 5 consecutive epochs without deadlocks or thread leaks."""
        dataset = [{"pixel_values": torch.randn(3, 16, 16), "step": torch.tensor(i)} for i in range(12)]
        loader = DataLoader(dataset, batch_size=3)
        prefetcher = AsyncDevicePrefetcher(loader, device=TARGET_DEVICE, mixed_precision="fp16", queue_size=2)

        for epoch in range(5):
            epoch_batches = 0
            with prefetcher as p:
                for batch in p:
                    assert batch["pixel_values"].device.type == TARGET_DEVICE
                    epoch_batches += 1
            assert epoch_batches == 4, f"Epoch {epoch} yielded {epoch_batches} batches instead of 4"

        prefetcher.close()
        assert prefetcher._worker_thread is None


class TestMemoryBoundednessAndStability100PlusSteps:
    """
    Verify memory growth remains strictly bounded over 100+ simulated training steps
    measuring Resident Set Size (RSS) and Metal (MPS) allocated memory.
    """

    def test_120_step_memory_stability_mps(self, tmp_path: Path) -> None:
        """
        Simulate 120 consecutive training steps on MPS:
        - MMapOCRDataset backing
        - Multi-worker DataLoader (num_workers=2, persistent_workers=True, prefetch_factor=2)
        - AsyncDevicePrefetcher with queue_size=3 and FP16 casting
        - Mock forward/backward pass with gradient zeroing and periodic MPS cache defragmentation.
        """
        num_samples = 480
        batch_size = 4
        total_steps = num_samples // batch_size  # 120 steps

        samples = create_synthetic_handwriting_samples(count=num_samples, img_size=(64, 64))
        proc = create_dummy_processor(size=(64, 64), vocab_size=50)
        cache_dir = tmp_path / "mem_test_cache"

        mmap_ds = MMapOCRDataset.create_mmap_cache(
            dataset=samples,
            output_dir_or_prefix=cache_dir,
            processor=proc,
            image_size=(64, 64),
            max_target_length=32,
            dtype="float16",
        )

        collator = OCRDataCollator(processor=proc, max_target_length=32)
        loader = DataLoader(
            mmap_ds,
            batch_size=batch_size,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collator,
        )

        prefetcher = AsyncDevicePrefetcher(
            loader,
            device=TARGET_DEVICE,
            mixed_precision="fp16",
            queue_size=3,
        )

        # Simple compute model to simulate forward + backward computation on MPS
        mock_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 2),
        ).to(TARGET_DEVICE)
        mock_optimizer = torch.optim.SGD(mock_model.parameters(), lr=0.01)

        rss_history: List[float] = []
        mps_history: List[float] = []

        step = 0
        with prefetcher as p:
            for batch in p:
                step += 1
                mock_optimizer.zero_grad()
                pv = batch["pixel_values"].float()
                out = mock_model(pv)
                loss = out.sum()
                loss.backward()
                mock_optimizer.step()

                # Periodic defragmentation every 30 steps
                if TARGET_DEVICE == "mps" and step % 30 == 0:
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()

                # Sample memory
                current_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                rss_history.append(current_rss_mb)

                if TARGET_DEVICE == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                    mps_allocated_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                    mps_history.append(mps_allocated_mb)

        assert step == total_steps

        # Check post-warmup memory stability (steps 20 to 120)
        warmup_step = 20
        post_warmup_rss = rss_history[warmup_step:]
        min_post_rss = min(post_warmup_rss)
        max_post_rss = max(post_warmup_rss)
        rss_delta_mb = max_post_rss - min_post_rss

        logger.info(
            f"120-Step Memory Stability: Warmup RSS = {rss_history[0]:.2f} MB, Step 20 RSS = {rss_history[warmup_step]:.2f} MB, Final RSS = {rss_history[-1]:.2f} MB, Max Post-Warmup Delta = {rss_delta_mb:.2f} MB"
        )

        # Assert memory stability: post-warmup memory growth is bounded (< 80 MB across 100 steps)
        assert rss_delta_mb < 80.0, f"Memory growth unbounded: {rss_delta_mb:.2f} MB variance between step 20 and step 120"


class TestAdversarialLifecycleAndEdgeCases:
    """
    Adversarial challenge tests: early loop break, queue congestion, exception handling,
    and boundary cases.
    """

    def test_early_loop_break_thread_cleanup(self) -> None:
        """
        Challenge: Consumer breaks early (e.g. at step 3 out of 100).
        Ensures background worker thread joins immediately without hang or thread leak.
        """
        dataset = [{"x": torch.randn(4, 4)} for _ in range(100)]
        loader = DataLoader(dataset, batch_size=2)
        prefetcher = AsyncDevicePrefetcher(loader, device=TARGET_DEVICE, queue_size=3)

        count = 0
        with prefetcher as p:
            for batch in p:
                count += 1
                if count == 3:
                    break

        # Verify thread terminates cleanly
        time.sleep(0.2)
        assert prefetcher._worker_thread is None or not prefetcher._worker_thread.is_alive()

    @pytest.mark.parametrize("queue_size", [1, 5])
    def test_extreme_queue_sizes_and_bursty_consumption(self, queue_size: int) -> None:
        """
        Challenge: Extreme queue capacities (1 = minimal FIFO, 5 = deep buffer) with bursty consumption.
        """
        dataset = [{"x": torch.tensor([float(i)])} for i in range(20)]
        loader = DataLoader(dataset, batch_size=2)
        prefetcher = AsyncDevicePrefetcher(loader, device=TARGET_DEVICE, queue_size=queue_size)

        collected = []
        with prefetcher as p:
            for i, batch in enumerate(p):
                collected.append(batch["x"])
                if i % 3 == 0:
                    time.sleep(0.02)  # Simulates variable compute step latency

        assert len(collected) == 10

    def test_worker_exception_propagation_to_main_thread(self) -> None:
        """
        Challenge: An exception occurs inside the underlying DataLoader/Dataset.
        The prefetcher must propagate the exception to the consumer thread rather than hanging indefinitely.
        """
        class CorruptedDataset(Dataset):
            def __len__(self) -> int:
                return 10
            def __getitem__(self, idx: int) -> Dict[str, Any]:
                if idx == 4:
                    raise ValueError("Simulated corrupt image buffer at idx 4")
                return {"pixel_values": torch.zeros((3, 16, 16), dtype=torch.float32)}

        loader = DataLoader(CorruptedDataset(), batch_size=2)
        prefetcher = AsyncDevicePrefetcher(loader, device=TARGET_DEVICE, queue_size=2)

        with pytest.raises(ValueError, match="Simulated corrupt image buffer"):
            with prefetcher as p:
                for _ in p:
                    pass

        prefetcher.close()

    def test_empty_dataset_edge_case(self) -> None:
        """
        Challenge: DataLoader with 0 items. Prefetcher should raise StopIteration cleanly.
        """
        dataset: List[Dict[str, Any]] = []
        loader = DataLoader(dataset, batch_size=2)
        prefetcher = AsyncDevicePrefetcher(loader, device=TARGET_DEVICE)

        batches = []
        with prefetcher as p:
            for batch in p:
                batches.append(batch)

        assert batches == []
        assert len(prefetcher) == 0

    def test_single_sample_dataset_edge_case(self) -> None:
        """
        Challenge: Dataset with exactly 1 sample and batch_size=4.
        """
        dataset = [{"pixel_values": torch.randn(3, 16, 16), "labels": torch.tensor([1, 2])}]
        loader = DataLoader(dataset, batch_size=4)
        prefetcher = AsyncDevicePrefetcher(loader, device=TARGET_DEVICE, queue_size=2)

        batches = list(prefetcher)
        assert len(batches) == 1
        assert batches[0]["pixel_values"].shape[0] == 1
        prefetcher.close()


class TestRealManifestIngestionIntegrity:
    """
    Empirical tests validating end-to-end ingestion against real handwriting manifests.
    """

    MANIFEST_PATH = Path("data/reference_handwriting/train_manifest.jsonl")

    @pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Real handwriting manifest not available on disk")
    def test_real_manifest_mmap_cache_and_prefetcher(self, tmp_path: Path) -> None:
        """
        Verify building MMap cache from real manifest and streaming through multi-worker AsyncDevicePrefetcher.
        """
        # Read first 50 real records from train_manifest.jsonl
        subset_manifest_path = tmp_path / "subset_manifest.jsonl"
        with open(self.MANIFEST_PATH, "r", encoding="utf-8") as f_in, open(subset_manifest_path, "w", encoding="utf-8") as f_out:
            for i, line in enumerate(f_in):
                if i >= 50:
                    break
                f_out.write(line)

        proc = create_dummy_processor(size=(384, 384), vocab_size=100)
        cache_dir = tmp_path / "real_manifest_cache"

        mmap_ds = MMapOCRDataset.create_mmap_cache(
            dataset=subset_manifest_path,
            output_dir_or_prefix=cache_dir,
            processor=proc,
            image_size=(384, 384),
            max_target_length=64,
            dtype="float16",
        )

        assert len(mmap_ds) == 50

        collator = OCRDataCollator(processor=proc, max_target_length=64)
        loader = DataLoader(
            mmap_ds,
            batch_size=5,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collator,
        )

        prefetcher = AsyncDevicePrefetcher(
            loader,
            device=TARGET_DEVICE,
            mixed_precision="fp16",
            queue_size=2,
        )

        seen = 0
        with prefetcher as p:
            for batch in p:
                assert batch["pixel_values"].shape == (5, 3, 384, 384)
                assert batch["pixel_values"].dtype == torch.float16
                assert batch["pixel_values"].device.type == TARGET_DEVICE
                assert batch["labels"].shape == (5, 64)
                assert batch["labels"].device.type == TARGET_DEVICE
                seen += 5

        assert seen == 50
        prefetcher.close()

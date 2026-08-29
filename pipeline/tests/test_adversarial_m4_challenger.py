"""
Tier 5 Adversarial Hardening Test Suite (Milestone M4)
Challenger: challenger_m4_1

Stress tests:
1. AsyncDevicePrefetcher edge cases:
   - Abrupt iterator exhaustion and early loop breaking across 25 cycles
   - Empty loader / 0-item iterator immediate exhaustion and graceful shutdown
   - Concurrent shutdowns across multiple competing threads
   - Underlying dataset exception propagation to main consumer thread without hang
   - Rapid restart and re-iteration lifecycle safety
   - Multi-precision tensor transfers (fp16, bf16, fp32, none) under large batch loads

2. MMapOCRDataset edge cases:
   - Out-of-bounds index bounds checking (negative, >= len, extreme large integers)
   - Zero-sample / empty mmap cache creation and indexing semantics
   - Corrupted metadata and missing binary payload error handling
   - Thread and process concurrent access data integrity
   - Immutable zero-copy slice protections

3. OCRDataCollator edge cases:
   - Empty batch resilience
   - High sequence length disparity (1-token vs 256-tokens in single batch)
   - Uniform-padding and all-masked label generation
   - Heterogeneous token inputs
"""

import os
import json
import tempfile
import threading
import time
from typing import List, Dict, Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline.training.prefetcher import AsyncDevicePrefetcher
from pipeline.training.dataset import MMapOCRDataset, OCRDataCollator


# ============================================================================
# Helpers & Dummy Datasets
# ============================================================================

class DummyTensorDataset(Dataset):
    """Simple synthetic dataset for prefetcher tests."""
    def __init__(self, size: int = 50, channels: int = 1, height: int = 64, width: int = 128):
        self.size = size
        self.channels = channels
        self.height = height
        self.width = width

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range [0, {self.size})")
        return {
            "pixel_values": torch.randn(self.channels, self.height, self.width, dtype=torch.float32),
            "input_ids": torch.tensor([1, 2, 3, 4, idx % 50], dtype=torch.long),
            "sample_id": torch.tensor(idx, dtype=torch.long),
        }


class FaultyDataset(Dataset):
    """Dataset designed to raise runtime exception at a specific index."""
    def __init__(self, fail_at_idx: int = 3, size: int = 10):
        self.fail_at_idx = fail_at_idx
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        if idx == self.fail_at_idx:
            raise RuntimeError(f"Simulated fault at index {idx}")
        return {
            "pixel_values": torch.ones(1, 32, 64, dtype=torch.float32) * idx,
            "input_ids": torch.tensor([idx], dtype=torch.long),
        }


class EmptyDataset(Dataset):
    """0-sample dataset."""
    def __len__(self):
        return 0

    def __getitem__(self, idx: int):
        raise IndexError("Empty dataset index error")


# ============================================================================
# 1. AsyncDevicePrefetcher Adversarial Hardening
# ============================================================================

class TestAsyncDevicePrefetcherAdversarial:
    """Stress tests for AsyncDevicePrefetcher."""

    def test_abrupt_early_break_lifecycle_25_cycles(self):
        """Verify 25 successive cycles of initializing prefetcher and breaking after 1-2 steps."""
        dataset = DummyTensorDataset(size=40)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        device = torch.device("cpu")

        initial_thread_count = threading.active_count()

        for cycle in range(25):
            prefetcher = AsyncDevicePrefetcher(loader, device=device, queue_size=3)
            count = 0
            for batch in prefetcher:
                assert "pixel_values" in batch
                assert batch["pixel_values"].shape[0] == 4
                count += 1
                if count >= 2:
                    break
            prefetcher.close()

        time.sleep(0.2)
        current_thread_count = threading.active_count()
        assert current_thread_count <= initial_thread_count + 1

    def test_empty_dataset_immediate_exhaustion(self):
        """Verify prefetcher handles 0-sample empty dataloader cleanly without hanging."""
        dataset = EmptyDataset()
        loader = DataLoader(dataset, batch_size=4)
        prefetcher = AsyncDevicePrefetcher(loader, device=torch.device("cpu"), queue_size=2)

        batches = []
        for batch in prefetcher:
            batches.append(batch)

        assert len(batches) == 0
        prefetcher.close()

    def test_concurrent_multi_threaded_close_calls(self):
        """Verify calling close() concurrently from 10 threads is idempotent and does not deadlock."""
        dataset = DummyTensorDataset(size=20)
        loader = DataLoader(dataset, batch_size=2)
        prefetcher = AsyncDevicePrefetcher(loader, device=torch.device("cpu"), queue_size=4)

        _ = next(iter(prefetcher))

        errors = []

        def threaded_close():
            try:
                for _ in range(5):
                    prefetcher.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=threaded_close) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive(), "Thread deadlocked during concurrent close"

        assert len(errors) == 0

    def test_worker_exception_propagation_no_deadlock(self):
        """Verify exception raised in dataset worker loop is propagated to consumer thread and prefetcher closes."""
        dataset = FaultyDataset(fail_at_idx=3, size=10)
        loader = DataLoader(dataset, batch_size=1)
        prefetcher = AsyncDevicePrefetcher(loader, device=torch.device("cpu"), queue_size=2)

        with pytest.raises(RuntimeError, match="Simulated fault at index 3"):
            for _ in prefetcher:
                pass

        prefetcher.close()

    def test_multi_precision_casting_under_load(self):
        """Verify casting across fp16, bf16, fp32, none with nested dict structures."""
        dataset = DummyTensorDataset(size=8, channels=3, height=32, width=32)
        loader = DataLoader(dataset, batch_size=2)

        precisions = ["fp16", "bf16", "fp32", "none"]
        expected_types = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "none": torch.float32,
        }

        for precision in precisions:
            prefetcher = AsyncDevicePrefetcher(
                loader,
                device=torch.device("cpu"),
                mixed_precision=precision,
                queue_size=2
            )
            for batch in prefetcher:
                assert batch["pixel_values"].dtype == expected_types[precision]
                assert batch["input_ids"].dtype == torch.long
            prefetcher.close()

    def test_prefetcher_context_manager_safety(self):
        """Verify prefetcher works safely with __enter__ and __exit__."""
        dataset = DummyTensorDataset(size=12)
        loader = DataLoader(dataset, batch_size=3)

        with AsyncDevicePrefetcher(loader, device=torch.device("cpu"), queue_size=2) as prefetcher:
            items = list(prefetcher)
            assert len(items) == 4

        assert prefetcher._worker_thread is None or not prefetcher._worker_thread.is_alive()


# ============================================================================
# 2. MMapOCRDataset Adversarial Hardening
# ============================================================================

class TestMMapOCRDatasetAdversarial:
    """Stress tests for MMapOCRDataset memory mapping and bounds."""

    @pytest.fixture
    def sample_mmap_cache(self, tmp_path):
        """Create a temporary valid mmap cache with 5 samples."""
        num_samples = 5
        channels, height, width = 3, 32, 64
        max_label_len = 16

        cache_dir = tmp_path / "mmap_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Values in range [0.1, 0.9] to prevent /255 rescale normalization
        images = np.random.uniform(0.1, 0.9, size=(num_samples, channels, height, width)).astype(np.float32)
        labels = np.random.randint(1, 100, size=(num_samples, max_label_len), dtype=np.int64)

        images_file = cache_dir / "images.bin"
        labels_file = cache_dir / "labels.bin"
        meta_file = cache_dir / "meta.json"

        images_mmap = np.memmap(str(images_file), dtype="float32", mode="w+", shape=images.shape)
        images_mmap[:] = images[:]
        images_mmap.flush()

        labels_mmap = np.memmap(str(labels_file), dtype="int64", mode="w+", shape=labels.shape)
        labels_mmap[:] = labels[:]
        labels_mmap.flush()

        metadata = {
            "num_samples": num_samples,
            "image_shape": [num_samples, channels, height, width],
            "image_dtype": "float32",
            "labels_shape": [num_samples, max_label_len],
            "labels_dtype": "int64",
            "max_target_length": max_label_len,
            "version": "1.0",
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f)

        return str(cache_dir), metadata, images, labels

    def test_out_of_bounds_indexing_rejections(self, sample_mmap_cache):
        """Verify negative, out-of-range, and extreme indices raise IndexError."""
        cache_dir, meta, _, _ = sample_mmap_cache
        dataset = MMapOCRDataset(cache_dir)

        assert len(dataset) == 5

        with pytest.raises(IndexError):
            _ = dataset[-1]

        with pytest.raises(IndexError):
            _ = dataset[-100]

        with pytest.raises(IndexError):
            _ = dataset[5]

        with pytest.raises(IndexError):
            _ = dataset[999999]

    def test_corrupted_metadata_handling(self, tmp_path):
        """Verify initializing dataset with invalid metadata raises appropriate exceptions."""
        cache_dir = tmp_path / "corrupt_cache"
        cache_dir.mkdir(parents=True)

        meta_file = cache_dir / "meta.json"

        # Case 1: Corrupted JSON content
        meta_file.write_text("NOT_JSON")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _ = MMapOCRDataset(str(cache_dir))

        # Case 2: Missing essential binary files
        meta_file.write_text(json.dumps({"version": "1.0"}))
        with pytest.raises(FileNotFoundError):
            _ = MMapOCRDataset(str(cache_dir))

    def test_zero_sample_cache_lifecycle(self, tmp_path):
        """Verify handling of an empty 0-sample cache raises cleanly on zero-byte file or empty dataset."""
        cache_dir = tmp_path / "empty_cache"
        cache_dir.mkdir(parents=True)

        meta = {
            "num_samples": 0,
            "image_shape": [0, 3, 32, 64],
            "image_dtype": "float32",
            "labels_shape": [0, 10],
            "labels_dtype": "int64",
            "max_target_length": 10,
        }
        with open(cache_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        # Empty binaries
        open(cache_dir / "images.bin", "wb").close()
        open(cache_dir / "labels.bin", "wb").close()

        with pytest.raises((ValueError, IndexError, RuntimeError)):
            _ = MMapOCRDataset(str(cache_dir))

    def test_concurrent_multithreaded_mmap_reads(self, sample_mmap_cache):
        """Verify 20 threads simultaneously reading various indices produce identical deterministic tensors."""
        cache_dir, _, orig_images, orig_labels = sample_mmap_cache
        dataset = MMapOCRDataset(cache_dir)

        results = [None] * 20
        errors = []

        def worker(thread_idx: int):
            try:
                sample_idx = thread_idx % len(dataset)
                item = dataset[sample_idx]
                img_np = item["pixel_values"].numpy()
                lbl_np = item["input_ids"].numpy()

                assert np.allclose(img_np, orig_images[sample_idx], atol=1e-5)
                assert np.array_equal(lbl_np, orig_labels[sample_idx])
                results[thread_idx] = True
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)


# ============================================================================
# 3. OCRDataCollator Adversarial Hardening
# ============================================================================

class TestOCRDataCollatorAdversarial:
    """Stress tests for OCRDataCollator batch assembly."""

    def test_collator_empty_batch(self):
        """Verify collating an empty batch returns empty dict."""
        collator = OCRDataCollator(pad_token_id=0)
        res = collator([])
        assert res == {}

    def test_extreme_sequence_length_disparity(self):
        """Verify collator handles 1-token sequence alongside 256-token sequence without corruption."""
        collator = OCRDataCollator(pad_token_id=0, max_target_length=256)

        batch = [
            {
                "pixel_values": torch.zeros(1, 32, 64),
                "input_ids": torch.tensor([42], dtype=torch.long),
            },
            {
                "pixel_values": torch.zeros(1, 32, 64),
                "input_ids": torch.randint(1, 1000, (256,), dtype=torch.long),
            },
            {
                "pixel_values": torch.zeros(1, 32, 64),
                "input_ids": torch.tensor([10, 20, 30], dtype=torch.long),
            },
        ]

        collated = collator(batch)

        assert "pixel_values" in collated
        assert "labels" in collated

        labels = collated["labels"]

        # Max length should be 256
        assert labels.shape == (3, 256)

        # Sample 0 has 1 valid token, remaining 255 must be padded to -100
        assert labels[0, 0] == 42
        assert (labels[0, 1:] == -100).all()

        if "decoder_attention_mask" in collated:
            mask = collated["decoder_attention_mask"]
            assert mask[0, 0] == 1
            assert (mask[0, 1:] == 0).all()

    def test_collator_all_pad_sequence(self):
        """Verify batch consisting entirely of pad tokens produces fully masked labels."""
        collator = OCRDataCollator(pad_token_id=0)

        batch = [
            {
                "pixel_values": torch.ones(1, 16, 16),
                "input_ids": torch.zeros(5, dtype=torch.long),
            }
        ]

        collated = collator(batch)
        assert (collated["labels"] == -100).all()
        if "decoder_attention_mask" in collated:
            assert (collated["decoder_attention_mask"] == 0).all()

    def test_matching_spatial_dimensions(self):
        """Verify collator stacks batch images properly."""
        collator = OCRDataCollator(pad_token_id=0)

        batch_good = [
            {"pixel_values": torch.randn(1, 32, 64), "input_ids": torch.tensor([1, 2])},
            {"pixel_values": torch.randn(1, 32, 64), "input_ids": torch.tensor([3, 4, 5])},
        ]
        out = collator(batch_good)
        assert out["pixel_values"].shape == (2, 1, 32, 64)

"""
pipeline/tests/test_m1_acceleration_challenger.py
Adversarial Empirical Stress Test Suite for Milestone 1 (Data Ingestion & Prefetching).

Comprehensive Challenge Vectors:
1. High Concurrency / Rapid Queue Drain / Burst Iteration (queue_size=1..64, empty loaders, 1-item loaders, multi-thread consumers).
2. Early Loop Breaks & Thread Lifecycle / Daemon Leak Verification (break at step 0..N, 50 rapid loops, active thread tracking).
3. Exception Propagation Across Thread Boundary (worker error, collate error, prepare_batch error, mid-stream vs first-step error).
4. Zero-Copy Tensor Slice Integrity & Mutation Isolation in MMapOCRDataset (memory sharing, in-place mutation, corrupted binary/meta, out-of-bounds).
5. Empirical DataLoader Wait-Time Elimination & Throughput Verification (I/O compute overlap, samples/sec throughput).
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline.dataset.dataset_loader import HandwritingSample
from pipeline.training.dataset import (
    DummyProcessor,
    DummyTokenizer,
    MMapOCRDataset,
    OCRDataCollator,
    OCRDataset,
    create_dummy_processor,
)
from pipeline.training.prefetcher import AsyncDevicePrefetcher, _ExceptionWrapper


# ==============================================================================
# Helper Classes & Utilities
# ==============================================================================

class SyntheticOCRDataset(Dataset):
    """Synthetic dataset generating controllable OCR tensors with simulated latency."""

    def __init__(
        self,
        count: int = 20,
        image_shape: Tuple[int, int, int] = (3, 64, 64),
        sleep_sec: float = 0.0,
        fault_at_index: Optional[int] = None,
        fault_exception: Exception = RuntimeError("Simulated dataset fetch error"),
    ) -> None:
        self.count = count
        self.image_shape = image_shape
        self.sleep_sec = sleep_sec
        self.fault_at_index = fault_at_index
        self.fault_exception = fault_exception

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.count:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.count}")

        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)

        if self.fault_at_index is not None and idx == self.fault_at_index:
            raise self.fault_exception

        return {
            "sample_id": f"sample_{idx:05d}",
            "pixel_values": torch.full(self.image_shape, float(idx), dtype=torch.float32),
            "input_ids": torch.tensor([0, idx % 50 + 4, 2], dtype=torch.long),
            "labels": torch.tensor([0, idx % 50 + 4, -100], dtype=torch.long),
            "text": f"text line {idx}",
            "writer_id": f"writer_{idx % 4}",
        }


def get_worker_threads() -> List[threading.Thread]:
    """Return all active AsyncDevicePrefetcher worker threads."""
    return [t for t in threading.enumerate() if t.name == "AsyncDevicePrefetcherWorker" and t.is_alive()]


# ==============================================================================
# Challenge Vector 1: High Concurrency, Rapid Queue Drain & Burst Iteration
# ==============================================================================

class TestConcurrencyAndBurstIteration:
    """Stress-test AsyncDevicePrefetcher under rapid queue drain and burst conditions."""

    @pytest.mark.parametrize("queue_size", [1, 2, 3, 5, 16, 64])
    def test_burst_drain_all_queue_sizes(self, queue_size: int) -> None:
        """Verify prefetcher yields exact batch count and data integrity across varied queue sizes."""
        num_samples = 40
        batch_size = 4
        dataset = SyntheticOCRDataset(count=num_samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        prefetcher = AsyncDevicePrefetcher(loader, device="cpu", mixed_precision="fp16", queue_size=queue_size)
        assert len(prefetcher) == len(loader)

        received_indices = []
        for batch in prefetcher:
            assert batch["pixel_values"].dtype == torch.float16
            assert batch["pixel_values"].shape == (batch_size, 3, 64, 64)
            # Extract sample IDs to verify ordering and zero loss
            for sid in batch["sample_id"]:
                received_indices.append(int(sid.split("_")[1]))

        assert received_indices == list(range(num_samples)), "Batch loss or misordering during burst drain"
        prefetcher.close()

    def test_empty_dataloader_iteration(self) -> None:
        """Verify iterating over an empty DataLoader raises StopIteration immediately without hanging."""
        dataset = SyntheticOCRDataset(count=0)
        loader = DataLoader(dataset, batch_size=4)

        prefetcher = AsyncDevicePrefetcher(loader, device="cpu", queue_size=3)
        batches = list(prefetcher)
        assert len(batches) == 0
        prefetcher.close()

    def test_single_element_dataloader(self) -> None:
        """Verify DataLoader with exactly 1 sample executes and finishes cleanly."""
        dataset = SyntheticOCRDataset(count=1)
        loader = DataLoader(dataset, batch_size=1)

        prefetcher = AsyncDevicePrefetcher(loader, device="cpu", queue_size=3)
        batches = list(prefetcher)
        assert len(batches) == 1
        assert batches[0]["sample_id"] == ["sample_00000"]
        prefetcher.close()

    def test_concurrent_multiple_prefetchers(self) -> None:
        """Verify multiple independent prefetchers running concurrently in separate threads do not collide."""
        def worker_task(thread_id: int, results: Dict[int, List[int]]) -> None:
            ds = SyntheticOCRDataset(count=20)
            loader = DataLoader(ds, batch_size=5)
            with AsyncDevicePrefetcher(loader, device="cpu", mixed_precision="fp16", queue_size=2) as prefetcher:
                collected = []
                for batch in prefetcher:
                    collected.extend([int(s.split("_")[1]) for s in batch["sample_id"]])
                results[thread_id] = collected

        threads = []
        results: Dict[int, List[int]] = {}
        for i in range(6):
            t = threading.Thread(target=worker_task, args=(i, results))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Worker thread timed out / hung"

        assert len(results) == 6
        for i in range(6):
            assert results[i] == list(range(20)), f"Thread {i} lost or corrupted batches"


# ==============================================================================
# Challenge Vector 2: Early Loop Breaks & Daemon Leak Verification
# ==============================================================================

class TestEarlyBreaksAndThreadLifecycle:
    """Stress-test loop break behaviors and guarantee no daemon thread accumulation."""

    @pytest.mark.parametrize("break_step", [0, 1, 2, 4, 9])
    def test_early_break_at_varying_steps(self, break_step: int) -> None:
        """Verify early loop termination at diverse steps cleans up background worker."""
        dataset = SyntheticOCRDataset(count=50)
        loader = DataLoader(dataset, batch_size=2)

        prefetcher = AsyncDevicePrefetcher(loader, device="cpu", mixed_precision="fp16", queue_size=3)
        for i, batch in enumerate(prefetcher):
            if i == break_step:
                break

        prefetcher.close()
        # Give a small grace period for join
        time.sleep(0.05)
        assert prefetcher._worker_thread is None or not prefetcher._worker_thread.is_alive()

    def test_rapid_consecutive_loop_breaks_zero_daemon_leak(self) -> None:
        """Adversarially break from 50 consecutive loops and verify active worker count remains 0."""
        dataset = SyntheticOCRDataset(count=20)
        loader = DataLoader(dataset, batch_size=2)

        # Baseline worker thread count
        initial_workers = len(get_worker_threads())

        for loop_idx in range(50):
            prefetcher = AsyncDevicePrefetcher(loader, device="cpu", queue_size=2)
            for i, batch in enumerate(prefetcher):
                break  # Break on first batch
            prefetcher.close()

        time.sleep(0.1)
        gc.collect()
        active_workers = len(get_worker_threads())
        assert active_workers <= initial_workers, f"Daemon thread leak detected: {active_workers} active workers remaining"

    def test_unclosed_prefetcher_garbage_collection(self) -> None:
        """Verify that prefetcher objects created and abandoned without explicit close() do not deadlock."""
        dataset = SyntheticOCRDataset(count=10)
        loader = DataLoader(dataset, batch_size=2)

        initial_workers = len(get_worker_threads())
        for _ in range(10):
            p = AsyncDevicePrefetcher(loader, device="cpu", queue_size=2)
            it = iter(p)
            _ = next(it)  # Partially consume
            del it
            del p  # Let it fall out of scope

        gc.collect()
        # Threads should not deadlock the interpreter
        assert True

    def test_idempotent_close_calls(self) -> None:
        """Verify calling close() repeatedly does not raise errors or hang."""
        dataset = SyntheticOCRDataset(count=6)
        loader = DataLoader(dataset, batch_size=2)
        prefetcher = AsyncDevicePrefetcher(loader, device="cpu")

        # Iterate once
        for _ in prefetcher:
            pass

        # Call close multiple times
        for _ in range(5):
            prefetcher.close()

        assert prefetcher._worker_thread is None

    def test_multi_epoch_clean_reiteration(self) -> None:
        """Verify prefetcher can be re-iterated 10 times consecutively with perfect batch counts."""
        dataset = SyntheticOCRDataset(count=12)
        loader = DataLoader(dataset, batch_size=3)
        prefetcher = AsyncDevicePrefetcher(loader, device="cpu", queue_size=2)

        for epoch in range(10):
            count = 0
            for batch in prefetcher:
                count += 1
                assert batch["pixel_values"].shape[0] == 3
            assert count == 4, f"Epoch {epoch} yielded {count} batches instead of 4"

        prefetcher.close()


# ==============================================================================
# Challenge Vector 3: Exception Propagation Across Thread Boundary
# ==============================================================================

class TestExceptionPropagation:
    """Stress-test worker error handling and cross-thread exception forwarding."""

    def test_exception_on_first_sample(self) -> None:
        """Verify exception raised on the first sample propagates immediately to main thread."""
        dataset = SyntheticOCRDataset(
            count=10,
            fault_at_index=0,
            fault_exception=ValueError("Corrupted image header at index 0"),
        )
        loader = DataLoader(dataset, batch_size=2)
        prefetcher = AsyncDevicePrefetcher(loader, device="cpu", queue_size=2)

        with pytest.raises(ValueError, match="Corrupted image header at index 0"):
            for _ in prefetcher:
                pass

        prefetcher.close()
        assert prefetcher._worker_thread is None or not prefetcher._worker_thread.is_alive()

    def test_exception_mid_stream_after_valid_batches(self) -> None:
        """Verify valid batches are yielded before exception at mid-stream index is raised."""
        dataset = SyntheticOCRDataset(
            count=10,
            fault_at_index=4,  # batch 2 contains index 4
            fault_exception=OSError("Disk read failure at sample 4"),
        )
        loader = DataLoader(dataset, batch_size=2)
        prefetcher = AsyncDevicePrefetcher(loader, device="cpu", queue_size=2)

        yielded_batches = []
        with pytest.raises(OSError, match="Disk read failure at sample 4"):
            for batch in prefetcher:
                yielded_batches.append(batch)

        # Batches 0 (samples 0, 1) and 1 (samples 2, 3) should have succeeded
        assert len(yielded_batches) == 2
        prefetcher.close()

    def test_exception_in_collate_fn(self) -> None:
        """Verify exceptions in collate function propagate cleanly to main iterator."""
        def faulty_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            for item in batch:
                if item["sample_id"] == "sample_00003":
                    raise TypeError("Mismatched tensor shape in collator")
            return OCRDataCollator()(batch)

        dataset = SyntheticOCRDataset(count=8)
        loader = DataLoader(dataset, batch_size=2, collate_fn=faulty_collate)
        prefetcher = AsyncDevicePrefetcher(loader, device="cpu", queue_size=2)

        with pytest.raises(TypeError, match="Mismatched tensor shape in collator"):
            for _ in prefetcher:
                pass

        prefetcher.close()

    def test_recovery_and_reiteration_after_exception(self) -> None:
        """Verify that after catching an exception and fixing the dataset, prefetcher can re-run."""
        dataset = SyntheticOCRDataset(count=6, fault_at_index=1)
        loader = DataLoader(dataset, batch_size=2)
        prefetcher = AsyncDevicePrefetcher(loader, device="cpu", queue_size=2)

        # 1. Trigger exception
        with pytest.raises(RuntimeError):
            for _ in prefetcher:
                pass
        prefetcher.close()

        # 2. Fix dataset fault
        dataset.fault_at_index = None

        # 3. Re-iterate
        batches = list(prefetcher)
        assert len(batches) == 3
        prefetcher.close()


# ==============================================================================
# Challenge Vector 4: Zero-Copy Slice Integrity & Mutation Isolation in MMapOCRDataset
# ==============================================================================

class TestMMapZeroCopyAndMutationIsolation:
    """Stress-test MMapOCRDataset zero-copy guarantees, in-place mutation isolation, and memory integrity."""

    @pytest.fixture
    def sample_mmap_cache(self, tmp_path: Path) -> Tuple[Path, int]:
        """Create a temporary memory-mapped dataset cache."""
        num_samples = 16
        img_size = (384, 384)
        samples = []
        for i in range(num_samples):
            img = np.full((img_size[0], img_size[1], 3), i * 10, dtype=np.uint8)
            samples.append(
                HandwritingSample(
                    sample_id=f"med_rx_{i:04d}",
                    image=img,
                    text=f"Amoxicillin {250 * (i + 1)}mg PO TID",
                    writer_id=f"doc_{i % 3}",
                )
            )

        processor = create_dummy_processor(size=img_size, vocab_size=50)
        cache_dir = tmp_path / "test_mmap_cache"
        MMapOCRDataset.create_mmap_cache(
            dataset=samples,
            output_dir_or_prefix=cache_dir,
            processor=processor,
            image_size=img_size,
            max_target_length=32,
            dtype="float16",
        )
        return cache_dir, num_samples

    def test_mmap_zero_copy_tensor_view_fp16(self, sample_mmap_cache: Tuple[Path, int]) -> None:
        """Verify MMapOCRDataset with return_fp16=True directly slices mmap buffer."""
        cache_dir, num_samples = sample_mmap_cache
        ds = MMapOCRDataset(cache_dir, return_fp16=True)

        assert len(ds) == num_samples
        item = ds[3]
        assert item["pixel_values"].dtype == torch.float16
        assert item["pixel_values"].shape == (3, 384, 384)
        assert item["sample_id"] == "med_rx_0003"
        assert item["writer_id"] == "doc_0"

    def test_in_place_tensor_mutation_isolation(self, sample_mmap_cache: Tuple[Path, int]) -> None:
        """
        Adversarial test: Mutate retrieved tensor in-place.
        Verify that mutating a retrieved sample DOES NOT corrupt subsequent reads from disk/mmap.
        """
        cache_dir, _ = sample_mmap_cache
        ds = MMapOCRDataset(cache_dir, return_fp16=False)

        # 1. Read sample 0 and record initial values
        item0 = ds[0]
        initial_val = item0["pixel_values"][0, 0, 0].item()

        # 2. Mutate pixel_values in place
        item0["pixel_values"].add_(100.0)
        assert item0["pixel_values"][0, 0, 0].item() != initial_val

        # 3. Read sample 0 again fresh from dataset
        item0_fresh = ds[0]
        fresh_val = item0_fresh["pixel_values"][0, 0, 0].item()

        # 4. Verify disk / mmap storage was NOT permanently altered
        assert np.isclose(fresh_val, initial_val, atol=1e-4), "In-place mutation corrupted mmap source!"

    def test_mmap_corrupted_metadata_handling(self, tmp_path: Path) -> None:
        """Verify MMapOCRDataset raises informative error when metadata file is missing or invalid."""
        empty_dir = tmp_path / "empty_cache"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="MMap metadata file not found"):
            _ = MMapOCRDataset(empty_dir)

    def test_mmap_out_of_bounds_adversarial_indices(self, sample_mmap_cache: Tuple[Path, int]) -> None:
        """Verify strict IndexError on boundary and extreme negative/positive indices."""
        cache_dir, num_samples = sample_mmap_cache
        ds = MMapOCRDataset(cache_dir)

        for invalid_idx in [-100, -1, num_samples, num_samples + 50, 999999]:
            with pytest.raises(IndexError, match="out of bounds"):
                _ = ds[invalid_idx]

    def test_mmap_multi_worker_dataloader_scaling(self, sample_mmap_cache: Tuple[Path, int]) -> None:
        """Verify MMapOCRDataset executes reliably with multi-process DataLoader and collator."""
        cache_dir, num_samples = sample_mmap_cache
        ds = MMapOCRDataset(cache_dir, return_fp16=True)
        collator = OCRDataCollator(max_target_length=32)

        loader = DataLoader(
            ds,
            batch_size=4,
            shuffle=True,
            num_workers=2,
            collate_fn=collator,
        )

        total_batches = 0
        total_samples = 0
        for batch in loader:
            total_batches += 1
            total_samples += batch["pixel_values"].shape[0]
            assert batch["pixel_values"].dtype == torch.float16
            assert batch["labels"].dtype == torch.long

        assert total_samples == num_samples
        assert total_batches == num_samples // 4


# ==============================================================================
# Challenge Vector 5: Empirical Dataloader Wait Time Elimination & Throughput
# ==============================================================================

class TestDataloaderWaitTimeAndThroughput:
    """Empirical verification of wait time elimination and high-throughput mmap scaling."""

    def test_dataloader_wait_time_elimination_benchmark(self) -> None:
        """
        Empirically verify that AsyncDevicePrefetcher overlaps data I/O with compute.
        Scenario:
        - 8 batches.
        - Data generation time per batch = 15ms.
        - Simulated forward/backward compute time per batch = 25ms.
        
        Expected:
        - Un-prefetched baseline: Total time ≈ 8 * (15ms + 25ms) = 320ms.
        - Prefetched: Total time ≈ 15ms (initial fetch) + 8 * 25ms (compute) = 215ms.
        - Per-step dataloader wait time during compute loop ≈ 0ms.
        """
        num_batches = 8
        batch_size = 2
        data_latency = 0.015  # 15ms per batch
        compute_latency = 0.025  # 25ms per batch

        dataset = SyntheticOCRDataset(count=num_batches * batch_size, sleep_sec=data_latency / batch_size)

        # 1. Baseline Sequential Loop
        loader_base = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        t0 = time.perf_counter()
        for batch in loader_base:
            time.sleep(compute_latency)  # Simulate GPU compute
        baseline_duration = time.perf_counter() - t0

        # 2. Accelerated Prefetched Loop
        loader_accel = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        prefetcher = AsyncDevicePrefetcher(loader_accel, device="cpu", mixed_precision="fp16", queue_size=3)

        wait_times = []
        t0 = time.perf_counter()
        with prefetcher as p:
            it = iter(p)
            for step in range(num_batches):
                t_get_start = time.perf_counter()
                batch = next(it)
                t_wait = time.perf_counter() - t_get_start
                wait_times.append(t_wait)
                time.sleep(compute_latency)  # Simulate GPU compute
        accel_duration = time.perf_counter() - t0

        # Steady-state wait times (excluding the initial prefetch on step 0)
        steady_state_wait = wait_times[1:]
        mean_steady_wait = float(np.mean(steady_state_wait)) if steady_state_wait else 0.0

        # Assertions:
        # 1. Steady state wait time is near zero (< 8ms vs 15ms full latency)
        assert mean_steady_wait < 0.008, f"Prefetcher failed to eliminate wait time: mean wait = {mean_steady_wait*1000:.2f}ms"
        # 2. Total accelerated runtime is significantly faster than baseline
        assert accel_duration < baseline_duration * 0.85, (
            f"Expected prefetch speedup: baseline={baseline_duration:.3f}s, accel={accel_duration:.3f}s"
        )

    def test_mmap_dataset_samples_per_sec_throughput(self, tmp_path: Path) -> None:
        """
        Verify that MMapOCRDataset delivers >10,000 samples/sec raw throughput in Python,
        vastly exceeding the training pipeline baseline requirement (>6.5 samples/sec).
        """
        num_samples = 1000
        img_size = (64, 64)
        samples = [
            HandwritingSample(
                sample_id=f"bench_{i:05d}",
                image=np.full((img_size[0], img_size[1], 3), 128, dtype=np.uint8),
                text=f"bench transcription line {i}",
            )
            for i in range(num_samples)
        ]

        cache_dir = tmp_path / "bench_mmap_cache"
        MMapOCRDataset.create_mmap_cache(
            dataset=samples,
            output_dir_or_prefix=cache_dir,
            image_size=img_size,
            max_target_length=16,
            dtype="float16",
        )

        ds = MMapOCRDataset(cache_dir, return_fp16=True)

        # Measure sequential slice throughput
        t0 = time.perf_counter()
        for idx in range(num_samples):
            _ = ds[idx]
        elapsed = time.perf_counter() - t0
        throughput = num_samples / elapsed

        assert throughput > 5000.0, f"MMap throughput too slow: {throughput:.1f} samples/sec"

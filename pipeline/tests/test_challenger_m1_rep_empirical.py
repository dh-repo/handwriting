"""
pipeline/tests/test_challenger_m1_rep_empirical.py
Empirical Verification & Stress Test Suite for Milestone M1 (Data Ingestion & Prefetching).
Replacement Challenger 2 Suite.

Empirically tests:
1. Multi-worker DataLoader scaling (num_workers=2, 4, 8, persistent_workers=True, prefetch_factor=4)
   on Apple Silicon Metal (MPS) and CPU with AsyncDevicePrefetcher and MMapOCRDataset.
2. 100+ Step Memory Stability & Boundedness on MPS and CPU (monitoring RSS and MPS allocations).
3. Real & Synthetic Manifest Ingestion Throughput across worker topologies.
4. Multi-Epoch Lifecycle & Early-Break Robustness on Apple Silicon MPS.
5. In-place Mutation Isolation and Zero-Copy Concurrency Integrity.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import psutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline.training.config import get_default_num_workers
from pipeline.training.dataset import (
    DummyProcessor,
    DummyTokenizer,
    MMapOCRDataset,
    OCRDataCollator,
    OCRDataset,
    create_dummy_processor,
)
from pipeline.training.prefetcher import AsyncDevicePrefetcher, _ExceptionWrapper


def get_current_rss_mb() -> float:
    """Return resident set size (RSS) memory in MB for the current process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def get_mps_allocated_mb() -> float:
    """Return MPS allocated memory in MB if MPS is available, else 0.0."""
    if torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
        except Exception:
            return 0.0
    return 0.0


@pytest.fixture(scope="module")
def synthetic_mmap_cache(tmp_path_factory) -> str:
    """Create a temporary memory-mapped dataset cache of 400 samples."""
    tmp_dir = tmp_path_factory.mktemp("mmap_cache_empirical")
    num_samples = 400
    image_shape = (num_samples, 3, 384, 384)
    max_target_length = 64

    # Create dummy images (FP16)
    meta_path = tmp_dir / "meta.json"
    images_path = tmp_dir / "images.bin"
    labels_path = tmp_dir / "labels.bin"

    images_mmap = np.memmap(str(images_path), dtype=np.float16, mode="w+", shape=image_shape)
    # Populate with synthetic gradient data
    for i in range(num_samples):
        val = (i % 256) / 255.0
        images_mmap[i] = np.full((3, 384, 384), val, dtype=np.float16)
    images_mmap.flush()

    labels_mmap = np.memmap(str(labels_path), dtype=np.int32, mode="w+", shape=(num_samples, max_target_length))
    sample_ids = []
    texts = []
    writer_ids = []
    for i in range(num_samples):
        labels_mmap[i, :4] = [0, (i % 100) + 4, 15, 2]
        labels_mmap[i, 4:] = 1  # pad token
        sample_ids.append(f"synthetic_{i:05d}")
        texts.append(f"Synthetic sample text {i}")
        writer_ids.append(f"writer_{i % 8}")
    labels_mmap.flush()

    metadata = {
        "num_samples": num_samples,
        "image_shape": list(image_shape),
        "image_dtype": "float16",
        "max_target_length": max_target_length,
        "labels_dtype": "int32",
        "labels_shape": [num_samples, max_target_length],
        "sample_ids": sample_ids,
        "texts": texts,
        "writer_ids": writer_ids,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return str(tmp_dir)


# ==============================================================================
# 1. Multi-Worker Scaling (num_workers=2, 4, 8) on MPS & CPU
# ==============================================================================

class TestMultiWorkerScalingEmpirical:
    """Empirically test multi-worker configurations with AsyncDevicePrefetcher."""

    @pytest.mark.parametrize("num_workers", [2, 4, 8])
    @pytest.mark.parametrize("device_str", ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"])
    def test_multi_worker_mmap_prefetcher_scaling(
        self, synthetic_mmap_cache: str, num_workers: int, device_str: str
    ) -> None:
        """Verify MMapOCRDataset + DataLoader + AsyncDevicePrefetcher across worker counts."""
        dataset = MMapOCRDataset(synthetic_mmap_cache, return_fp16=True)
        assert len(dataset) == 400

        collator = OCRDataCollator(processor=None, max_target_length=64)
        batch_size = 16

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=collator,
            pin_memory=False,
        )

        device = torch.device(device_str)
        prefetcher = AsyncDevicePrefetcher(
            loader=loader,
            device=device,
            mixed_precision="fp16",
            queue_size=3,
        )

        t0 = None
        total_samples = 0
        batches_read = 0

        with prefetcher as p_iter:
            for batch in p_iter:
                if t0 is None:
                    t0 = time.perf_counter()
                assert isinstance(batch, dict)
                assert "pixel_values" in batch
                assert "labels" in batch

                pv = batch["pixel_values"]
                assert pv.device.type == device.type
                assert pv.dtype == torch.float16
                assert not torch.isnan(pv).any()

                total_samples += pv.shape[0]
                batches_read += 1

        elapsed = time.perf_counter() - (t0 or time.perf_counter())
        samples_per_sec = total_samples / max(elapsed, 1e-6)

        assert total_samples == 400
        assert batches_read == 25
        # MMap dataset + prefetcher should deliver huge throughput
        assert samples_per_sec > 100.0, f"Throughput too low: {samples_per_sec:.1f} samples/sec"

    def test_real_manifest_multi_worker_ingestion(self) -> None:
        """Test real dataset manifest ingestion with 4 workers and AsyncDevicePrefetcher."""
        manifest_path = Path("data/reference_handwriting/val_manifest.jsonl")
        if not manifest_path.exists():
            pytest.skip("Real val_manifest.jsonl not found, skipping real manifest test")

        # Load first 128 items from manifest for rapid empirical validation
        lines = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 128:
                    break
                lines.append(json.loads(line))

        processor = create_dummy_processor()
        dataset = OCRDataset(samples=lines, processor=processor, max_target_length=64)
        collator = OCRDataCollator(processor=processor, max_target_length=64)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            persistent_workers=False,
            collate_fn=collator,
        )

        prefetcher = AsyncDevicePrefetcher(
            loader=loader,
            device=device,
            mixed_precision="fp16",
            queue_size=3,
        )

        total_samples = 0
        with prefetcher as p_iter:
            for batch in p_iter:
                pv = batch["pixel_values"]
                assert pv.device.type == device.type
                assert pv.dtype == torch.float16
                assert pv.shape[0] <= 8
                total_samples += pv.shape[0]

        assert total_samples == 128


# ==============================================================================
# 2. 100+ Step Memory Stability & Boundedness Stress Test
# ==============================================================================

class TestBoundedMemory100PlusSteps:
    """Verify memory remains strictly bounded over 100+ simulated training steps."""

    @pytest.mark.parametrize("device_str", ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"])
    def test_120_steps_bounded_memory_growth(self, synthetic_mmap_cache: str, device_str: str) -> None:
        """Simulate 120 continuous batch steps and measure RSS and MPS memory."""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        dataset = MMapOCRDataset(synthetic_mmap_cache, return_fp16=True)
        collator = OCRDataCollator(processor=None, max_target_length=64)

        # Batch size 4 with 400 samples gives 100 batches per epoch. We iterate across epochs to reach 120 steps.
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=collator,
        )

        device = torch.device(device_str)
        initial_rss = get_current_rss_mb()
        initial_mps = get_mps_allocated_mb()

        memory_checkpoints: Dict[int, Dict[str, float]] = {}
        steps_executed = 0
        target_steps = 120

        # Run 2 epochs to execute 120 steps
        for epoch in range(2):
            if steps_executed >= target_steps:
                break
            prefetcher = AsyncDevicePrefetcher(
                loader=loader,
                device=device,
                mixed_precision="fp16",
                queue_size=3,
            )
            with prefetcher as p_iter:
                for batch in p_iter:
                    # Simulate lightweight tensor computation on device
                    pv = batch["pixel_values"]
                    lbl = batch["labels"]
                    _ = pv.sum() + lbl.float().sum()

                    steps_executed += 1
                    if steps_executed in (10, 30, 60, 90, 120):
                        rss_mb = get_current_rss_mb()
                        mps_mb = get_mps_allocated_mb()
                        memory_checkpoints[steps_executed] = {
                            "rss_mb": rss_mb,
                            "mps_mb": mps_mb,
                        }

                    if steps_executed >= target_steps:
                        break

        assert steps_executed == target_steps

        # Check memory delta between steady-state step 30 and step 120 (90 steps later)
        rss_30 = memory_checkpoints[30]["rss_mb"]
        rss_120 = memory_checkpoints[120]["rss_mb"]
        rss_delta = rss_120 - rss_30

        mps_30 = memory_checkpoints[30]["mps_mb"]
        mps_120 = memory_checkpoints[120]["mps_mb"]
        mps_delta = mps_120 - mps_30

        # Assert memory growth is bounded: steady state growth should be negligible (< 25 MB RSS growth)
        assert rss_delta < 25.0, (
            f"Unbounded RSS memory growth detected over 90 steps: "
            f"Step 30: {rss_30:.2f} MB -> Step 120: {rss_120:.2f} MB (Delta: {rss_delta:.2f} MB)"
        )

        if device_str == "mps":
            assert abs(mps_delta) < 10.0, (
                f"Unbounded MPS memory growth detected over 90 steps: "
                f"Step 30: {mps_30:.2f} MB -> Step 120: {mps_120:.2f} MB (Delta: {mps_delta:.2f} MB)"
            )


# ==============================================================================
# 3. Precision Casting & Queue Pressure on Metal MPS
# ==============================================================================

class TestAsyncPrefetcherMPSPrecisionAndQueuePressure:
    """Stress-test mixed precision casting and queue pressure on Apple Silicon Metal."""

    @pytest.mark.parametrize("precision", ["fp16", "bf16", "fp32", "none"])
    def test_precision_modes_on_device(self, synthetic_mmap_cache: str, precision: str) -> None:
        """Verify all precision modes cast floating point tensors correctly on target device."""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dataset = MMapOCRDataset(synthetic_mmap_cache, return_fp16=(precision == "fp16"))
        collator = OCRDataCollator(processor=None, max_target_length=64)

        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, prefetch_factor=2, collate_fn=collator)
        prefetcher = AsyncDevicePrefetcher(
            loader=loader,
            device=device,
            mixed_precision=precision,
            queue_size=2,
        )

        with prefetcher as p_iter:
            batch = next(p_iter)
            pv = batch["pixel_values"]
            assert pv.device.type == device.type

            if precision == "fp16":
                assert pv.dtype == torch.float16
            elif precision == "bf16":
                assert pv.dtype == torch.bfloat16
            elif precision in ("fp32", "none"):
                assert pv.dtype == torch.float32

    @pytest.mark.parametrize("queue_size", [1, 2, 3, 8])
    def test_queue_sizes_under_consumer_lag(self, synthetic_mmap_cache: str, queue_size: int) -> None:
        """Test bounded queues when consumer intentionally simulates 5ms delay per batch."""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dataset = MMapOCRDataset(synthetic_mmap_cache, return_fp16=True)
        collator = OCRDataCollator(processor=None, max_target_length=64)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, prefetch_factor=2, collate_fn=collator)

        prefetcher = AsyncDevicePrefetcher(
            loader=loader,
            device=device,
            mixed_precision="fp16",
            queue_size=queue_size,
        )

        count = 0
        with prefetcher as p_iter:
            for batch in p_iter:
                # Simulate consumer computation lag
                time.sleep(0.005)
                count += 1
                if count >= 10:
                    break

        assert count == 10


# ==============================================================================
# 4. Multi-Epoch Lifecycle & Early-Break Robustness
# ==============================================================================

class TestMultiEpochAndWorkerLifecycleUnderMPS:
    """Stress-test prefetcher multi-epoch re-initialization and early breaks."""

    def test_multi_epoch_reinitialization(self, synthetic_mmap_cache: str) -> None:
        """Run 3 sequential full epochs using persistent workers and verify complete retrieval."""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dataset = MMapOCRDataset(synthetic_mmap_cache, return_fp16=True)
        collator = OCRDataCollator(processor=None, max_target_length=64)

        loader = DataLoader(
            dataset,
            batch_size=20,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collator,
        )

        for epoch in range(3):
            prefetcher = AsyncDevicePrefetcher(loader, device=device, mixed_precision="fp16", queue_size=3)
            samples = 0
            with prefetcher as p_iter:
                for batch in p_iter:
                    samples += batch["pixel_values"].shape[0]
            assert samples == 400, f"Epoch {epoch} failed to yield all 400 samples"

    def test_early_breaks_mps_resource_cleanup(self, synthetic_mmap_cache: str) -> None:
        """Rapidly break out of iteration after 3 steps, 10 times in a row, on MPS."""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dataset = MMapOCRDataset(synthetic_mmap_cache, return_fp16=True)
        collator = OCRDataCollator(processor=None, max_target_length=64)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, persistent_workers=True, prefetch_factor=2, collate_fn=collator)

        for run in range(10):
            prefetcher = AsyncDevicePrefetcher(loader, device=device, mixed_precision="fp16", queue_size=3)
            steps = 0
            with prefetcher as p_iter:
                for batch in p_iter:
                    steps += 1
                    if steps >= 3:
                        break
            # Ensure worker thread cleaned up
            assert prefetcher._worker_thread is None or not prefetcher._worker_thread.is_alive()


# ==============================================================================
# 5. In-Place Mutation Isolation and Zero-Copy Concurrency
# ==============================================================================

class TestZeroCopyMMapSliceSafetyUnderMultiWorkerMPS:
    """Stress-test in-place mutations and concurrent slices from MMapOCRDataset."""

    def test_in_place_mutation_isolation_across_workers(self, synthetic_mmap_cache: str) -> None:
        """Mutate retrieved tensors in-place and verify underlying dataset cache remains intact."""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dataset = MMapOCRDataset(synthetic_mmap_cache, return_fp16=True)
        collator = OCRDataCollator(processor=None, max_target_length=64)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, prefetch_factor=2, collate_fn=collator)

        prefetcher = AsyncDevicePrefetcher(loader, device=device, mixed_precision="fp16", queue_size=3)

        with prefetcher as p_iter:
            for batch in p_iter:
                # In-place add 999.0
                pv = batch["pixel_values"]
                pv.add_(999.0)

        # Re-read raw mmap to ensure underlying disk memory was not corrupted
        fresh_ds = MMapOCRDataset(synthetic_mmap_cache, return_fp16=True)
        sample_0 = fresh_ds[0]
        raw_pv = sample_0["pixel_values"]
        assert raw_pv.max() <= 1.0, f"Underlying disk mmap was corrupted by in-place mutation: max={raw_pv.max()}"

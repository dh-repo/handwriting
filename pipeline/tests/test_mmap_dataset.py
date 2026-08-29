"""
pipeline/tests/test_mmap_dataset.py
Unit and integration tests for MMapOCRDataset:
- High-performance zero-copy tensor slicing from memory-mapped numpy binary arrays.
- Memory map cache export (create_mmap_cache) from samples, OCRDataset, and manifest files.
- Support for FP16, FP32, and uint8 storage formats.
- Pre-tokenized input_ids retrieval and OCRDataCollator compatibility.
- Seamless integration with PyTorch DataLoader and AsyncDevicePrefetcher.
- Robust boundary condition checks and error handling.
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pipeline.dataset.dataset_loader import HandwritingSample
from pipeline.training.dataset import (
    DummyProcessor,
    MMapOCRDataset,
    OCRDataCollator,
    OCRDataset,
    create_dummy_processor,
)
from pipeline.training.prefetcher import AsyncDevicePrefetcher


def create_sample_dataset(count: int = 10, img_size: int = 64) -> List[HandwritingSample]:
    """Helper to create test handwriting sample objects."""
    samples = []
    for i in range(count):
        img = np.random.randint(50, 200, (img_size, img_size, 3), dtype=np.uint8)
        sample = HandwritingSample(
            sample_id=f"med_sample_{i:04d}",
            image=img,
            text=f"Amoxicillin 500mg #{i}",
            writer_id=f"doc_{i % 3}",
        )
        samples.append(sample)
    return samples


def test_mmap_ocr_dataset_creation_and_slicing(tmp_path: Path) -> None:
    """Verify creation of mmap cache and zero-copy slicing of samples."""
    samples = create_sample_dataset(count=12, img_size=64)
    processor = create_dummy_processor(size=(64, 64), vocab_size=50)

    cache_dir = tmp_path / "mmap_cache"
    mmap_ds = MMapOCRDataset.create_mmap_cache(
        dataset=samples,
        output_dir_or_prefix=cache_dir,
        processor=processor,
        image_size=(64, 64),
        max_target_length=32,
        dtype="float16",
    )

    assert len(mmap_ds) == 12
    item0 = mmap_ds[0]
    assert item0["sample_id"] == "med_sample_0000"
    assert item0["writer_id"] == "doc_0"
    assert item0["text"] == "Amoxicillin 500mg #0"
    assert item0["pixel_values"].shape == (3, 64, 64)
    assert item0["pixel_values"].dtype == torch.float32  # Converted on retrieve when return_fp16=False
    assert "input_ids" in item0
    assert item0["input_ids"].shape == (32,)

    # Test return_fp16=True
    fp16_ds = MMapOCRDataset(cache_dir, return_fp16=True)
    item0_fp16 = fp16_ds[0]
    assert item0_fp16["pixel_values"].dtype == torch.float16


def test_mmap_ocr_dataset_uint8_format(tmp_path: Path) -> None:
    """Verify uint8 storage format and on-the-fly normalization."""
    samples = create_sample_dataset(count=6, img_size=48)
    processor = create_dummy_processor(size=(48, 48), vocab_size=50)

    cache_dir = tmp_path / "mmap_uint8"
    mmap_ds = MMapOCRDataset.create_mmap_cache(
        dataset=samples,
        output_dir_or_prefix=cache_dir,
        processor=processor,
        image_size=(48, 48),
        max_target_length=20,
        dtype="uint8",
    )

    assert len(mmap_ds) == 6
    item = mmap_ds[2]
    assert item["pixel_values"].shape == (3, 48, 48)
    assert item["pixel_values"].dtype == torch.float32
    # Verify range is roughly [-1.0, 1.0]
    assert item["pixel_values"].min() >= -1.05
    assert item["pixel_values"].max() <= 1.05


def test_mmap_ocr_dataset_with_collator_and_prefetcher(tmp_path: Path) -> None:
    """Verify end-to-end integration with OCRDataCollator and AsyncDevicePrefetcher."""
    samples = create_sample_dataset(count=8, img_size=64)
    processor = create_dummy_processor(size=(64, 64), vocab_size=50)

    cache_dir = tmp_path / "mmap_collator_test"
    mmap_ds = MMapOCRDataset.create_mmap_cache(
        dataset=samples,
        output_dir_or_prefix=cache_dir,
        processor=processor,
        image_size=(64, 64),
        max_target_length=32,
        dtype="float16",
    )

    collator = OCRDataCollator(processor=processor, max_target_length=32)
    loader = DataLoader(mmap_ds, batch_size=4, shuffle=False, collate_fn=collator)

    prefetcher = AsyncDevicePrefetcher(loader, device="cpu", mixed_precision="fp16", queue_size=2)

    batches = list(prefetcher)
    assert len(batches) == 2
    b0 = batches[0]
    assert b0["pixel_values"].shape == (4, 3, 64, 64)
    assert b0["pixel_values"].dtype == torch.float16
    assert b0["labels"].shape[0] == 4
    # Check -100 label masking
    assert (b0["labels"] == -100).any()

    prefetcher.close()


def test_mmap_ocr_dataset_from_manifest(tmp_path: Path) -> None:
    """Verify loading and automatic cache generation from manifest JSONL."""
    manifest_p = tmp_path / "test_manifest.jsonl"
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(5):
        img_p = img_dir / f"img_{i}.png"
        img = np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(img_p), img)

        record = {
            "id": f"rec_{i}",
            "sample_id": f"sample_{i}",
            "image_path": str(img_p),
            "text": f"Prescription item {i}",
            "writer_id": f"writer_{i}",
        }
        records.append(record)

    with open(manifest_p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    processor = create_dummy_processor(size=(32, 64))
    cache_dir = tmp_path / "manifest_cache"

    mmap_ds = MMapOCRDataset.from_manifest(
        manifest_path=manifest_p,
        cache_dir=cache_dir,
        processor=processor,
        max_target_length=32,
    )

    assert len(mmap_ds) == 5
    assert mmap_ds[0]["sample_id"] == "sample_0"


def test_mmap_ocr_dataset_boundary_and_errors(tmp_path: Path) -> None:
    """Verify out-of-bounds index raises IndexError and missing files raise FileNotFoundError."""
    samples = create_sample_dataset(count=4, img_size=32)
    cache_dir = tmp_path / "mmap_errors"
    mmap_ds = MMapOCRDataset.create_mmap_cache(
        dataset=samples,
        output_dir_or_prefix=cache_dir,
        image_size=(32, 32),
        max_target_length=16,
    )

    with pytest.raises(IndexError):
        _ = mmap_ds[4]

    with pytest.raises(IndexError):
        _ = mmap_ds[-1]

    with pytest.raises(FileNotFoundError):
        _ = MMapOCRDataset(tmp_path / "non_existent_cache")

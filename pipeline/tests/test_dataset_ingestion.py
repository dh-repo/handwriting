"""
pipeline/tests/test_dataset_ingestion.py
Unit & integration tests verifying 50,000+ dataset ingestion, manifests, schemas, and writer isolation:
1. Manifest count thresholds (full >= 50,000, train >= 40,000, val >= 5,000, test >= 5,000)
2. JSONL schema compliance & required field types
3. Strict writer independence (0% writer overlap between train/val/test)
4. Disk image existence, non-zero file sizes, and readability
5. Dataset summary JSON consistency
"""

import json
from pathlib import Path
import random
import cv2
from PIL import Image
import pytest

DATA_DIR = Path("data/reference_handwriting")


@pytest.fixture(scope="module")
def manifest_paths():
    train_p = DATA_DIR / "train_manifest.jsonl"
    val_p = DATA_DIR / "val_manifest.jsonl"
    test_p = DATA_DIR / "test_manifest.jsonl"
    full_p = DATA_DIR / "full_manifest.jsonl"
    summary_p = DATA_DIR / "dataset_summary.json"

    assert train_p.exists(), f"Missing {train_p}"
    assert val_p.exists(), f"Missing {val_p}"
    assert test_p.exists(), f"Missing {test_p}"
    assert full_p.exists(), f"Missing {full_p}"
    assert summary_p.exists(), f"Missing {summary_p}"

    return {
        "train": train_p,
        "val": val_p,
        "test": test_p,
        "full": full_p,
        "summary": summary_p
    }


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def test_manifest_counts_and_splits(manifest_paths):
    train_records = load_jsonl(manifest_paths["train"])
    val_records = load_jsonl(manifest_paths["val"])
    test_records = load_jsonl(manifest_paths["test"])
    full_records = load_jsonl(manifest_paths["full"])

    assert len(train_records) >= 40000, f"Expected >= 40,000 train records, got {len(train_records)}"
    assert len(val_records) >= 5000, f"Expected >= 5,000 val records, got {len(val_records)}"
    assert len(test_records) >= 5000, f"Expected >= 5,000 test records, got {len(test_records)}"
    assert len(full_records) >= 50000, f"Expected >= 50,000 full records, got {len(full_records)}"
    assert len(train_records) + len(val_records) + len(test_records) == len(full_records)


def test_strict_writer_isolation(manifest_paths):
    train_records = load_jsonl(manifest_paths["train"])
    val_records = load_jsonl(manifest_paths["val"])
    test_records = load_jsonl(manifest_paths["test"])

    train_writers = set(r["writer_id"] for r in train_records)
    val_writers = set(r["writer_id"] for r in val_records)
    test_writers = set(r["writer_id"] for r in test_records)

    assert len(train_writers) >= 100, f"Too few train writers: {len(train_writers)}"
    assert len(val_writers) >= 10, f"Too few val writers: {len(val_writers)}"
    assert len(test_writers) >= 10, f"Too few test writers: {len(test_writers)}"

    # 0% writer overlap between all splits
    assert train_writers.isdisjoint(val_writers), f"Writer leakage Train/Val: {train_writers & val_writers}"
    assert train_writers.isdisjoint(test_writers), f"Writer leakage Train/Test: {train_writers & test_writers}"
    assert val_writers.isdisjoint(test_writers), f"Writer leakage Val/Test: {val_writers & test_writers}"


def test_jsonl_schema_compliance(manifest_paths):
    full_records = load_jsonl(manifest_paths["full"])

    # Sample 1000 records for comprehensive schema validation
    sample_records = random.sample(full_records, k=min(len(full_records), 1000))
    required_keys = {
        "id", "sample_id", "image_path", "relative_image_path",
        "transcription", "text", "writer_id", "dataset_source",
        "source", "category", "is_lasa", "therapeutic_class",
        "width", "height", "augmentation_params"
    }

    for r in sample_records:
        for k in required_keys:
            assert k in r, f"Missing key {k} in manifest record: {r}"

        assert isinstance(r["id"], str) and len(r["id"]) > 0
        assert isinstance(r["transcription"], str) and len(r["transcription"]) > 0
        assert isinstance(r["writer_id"], str) and len(r["writer_id"]) > 0
        assert isinstance(r["is_lasa"], bool)
        assert isinstance(r["width"], int) and r["width"] >= 32
        assert isinstance(r["height"], int) and r["height"] >= 16
        assert isinstance(r["augmentation_params"], dict)


def test_image_files_disk_readability(manifest_paths):
    full_records = load_jsonl(manifest_paths["full"])

    # Spot-check 200 random image files across splits
    sample_records = random.sample(full_records, k=min(len(full_records), 200))

    for r in sample_records:
        img_p = Path(r["image_path"])
        assert img_p.exists(), f"Image file not found on disk: {img_p}"
        assert img_p.stat().st_size > 100, f"Image file is empty or corrupted: {img_p}"

        # Open and check with PIL / OpenCV
        with Image.open(img_p) as pil_img:
            assert pil_img.format == "PNG"
            w, h = pil_img.size
            assert w == r["width"]
            assert h == r["height"]
            assert pil_img.mode in ("RGB", "L")


def test_dataset_summary_consistency(manifest_paths):
    with open(manifest_paths["summary"], "r", encoding="utf-8") as f:
        summary = json.load(f)

    full_records = load_jsonl(manifest_paths["full"])
    assert summary["total_samples"] == len(full_records)
    assert summary["splits"]["train"] >= 40000
    assert summary["splits"]["val"] >= 5000
    assert summary["splits"]["test"] >= 5000
    assert summary["unique_writers"]["writer_overlap_train_val"] == 0
    assert summary["unique_writers"]["writer_overlap_train_test"] == 0
    assert summary["unique_writers"]["writer_overlap_val_test"] == 0

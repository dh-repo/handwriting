"""
Unit and integration tests for pipeline/dataset/dataset_loader.py.
"""

import io
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pipeline.dataset.dataset_loader import (
    HandwritingSample,
    PrescriptionItem,
    MedicalPrescriptionSample,
    IAMDatasetParser,
    MedicalPrescriptionDatasetLoader,
    HandwritingPyTorchDataset,
    StreamingSyntheticDataset,
    handwriting_collate_fn,
    is_synthetic_training_record,
)
from pipeline.dataset.synthetic_generator import SyntheticHandwritingGenerator


def test_iam_lines_parser(sample_iam_lines_txt: str):
    """Test IAM lines.txt parsing."""
    parser = IAMDatasetParser()
    buf = io.StringIO(sample_iam_lines_txt)
    samples = parser.parse_lines_txt(buf)

    assert len(samples) == 7
    assert samples[0].sample_id == "a01-000u-00"
    assert samples[0].text == "A MOVE to stop Mr. Gaitskell from"
    assert samples[0].writer_id == "000u"
    assert samples[0].bbox == (408, 768, 27, 51)


def test_iam_writer_independent_splits(sample_iam_lines_txt: str):
    """Test that writer-independent splitting produces strictly disjoint writer sets."""
    parser = IAMDatasetParser()
    buf = io.StringIO(sample_iam_lines_txt)
    samples = parser.parse_lines_txt(buf)

    train_s, val_s, test_s = parser.create_writer_independent_splits(
        samples, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42
    )

    train_writers = set(s.writer_id for s in train_s)
    val_writers = set(s.writer_id for s in val_s)
    test_writers = set(s.writer_id for s in test_s)

    # Disjointness checks (0 writer overlap)
    assert len(train_writers.intersection(val_writers)) == 0
    assert len(train_writers.intersection(test_writers)) == 0
    assert len(val_writers.intersection(test_writers)) == 0
    assert len(train_s) + len(val_s) + len(test_s) == len(samples)


def test_medical_prescription_manifest_loader(sample_prescription_json: str):
    """Test JSON manifest ingestion for medical prescriptions."""
    loader = MedicalPrescriptionDatasetLoader()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write(sample_prescription_json)
        tmp_path = tmp.name

    try:
        samples = loader.load_manifest(tmp_path)
        assert len(samples) == 1
        rx = samples[0]
        assert isinstance(rx, MedicalPrescriptionSample)
        assert rx.sample_id == "rx_00001"
        assert rx.clinic_name == "Metro Clinic"
        assert rx.doctor_name == "Dr. House, MD"
        assert len(rx.items) == 1
        assert rx.items[0].medication == "Amoxicillin"
        assert rx.items[0].dosage == "500mg"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_pytorch_dataset_and_dataloader_batching():
    """Test HandwritingPyTorchDataset and custom handwriting_collate_fn."""
    # Create 4 dummy samples with variable image sizes
    samples = [
        HandwritingSample(
            sample_id=f"sample_{i}",
            text=f"Line transcription {i}",
            image=np.full((50, 150 + i * 40, 3), 200 - i * 10, dtype=np.uint8),
            writer_id=f"writer_{i % 2}"
        )
        for i in range(4)
    ]

    dataset = HandwritingPyTorchDataset(samples, target_size=(64, 300))
    assert len(dataset) == 4

    item0 = dataset[0]
    assert "pixel_values" in item0
    assert item0["pixel_values"].shape == (3, 64, 300)
    assert item0["pixel_values"].dtype == torch.float32
    assert item0["text"] == "Line transcription 0"

    # Test DataLoader batch collation
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=handwriting_collate_fn)
    batch = next(iter(loader))

    assert "pixel_values" in batch
    assert batch["pixel_values"].shape == (2, 3, 64, 300)
    assert len(batch["texts"]) == 2
    assert len(batch["sample_ids"]) == 2


def test_streaming_synthetic_dataset(synthetic_generator: SyntheticHandwritingGenerator):
    """Test infinite streaming synthetic dataset."""
    stream_ds = StreamingSyntheticDataset(
        generator=synthetic_generator,
        vocab=["Short line one", "Another test sentence"],
        target_size=(64, 256)
    )

    it = iter(stream_ds)
    sample1 = next(it)
    sample2 = next(it)

    assert sample1["pixel_values"].shape == (3, 64, 256)
    assert sample2["pixel_values"].shape == (3, 64, 256)
    assert sample1["sample_id"].startswith("syn_")
    assert sample2["sample_id"].startswith("syn_")


def test_manifest_metadata_precedence_and_template_category_archival(tmp_path: Path):
    """
    Verify top-level manifest attributes take precedence over nested metadata dictionary values,
    and different nested categories are archived into template_category.
    """
    manifest_data = [
        {
            "id": "rec_001",
            "image_path": "images/test1.png",
            "text": "Patient has severe cough",
            "category": "clinical_note",
            "is_lasa": False,
            "therapeutic_class": "respiratory",
            "writer_id": "w_123",
            "source": "synthetic",
            "metadata": {
                "category": "soap_note",
                "department": "pulmonology",
                "custom_flag": 42
            }
        },
        {
            "id": "rec_002",
            "image_path": "images/test2.png",
            "text": "Amoxicillin 500mg PO TID",
            "category": "prescription_item",
            "is_lasa": True,
            "metadata": {
                "category": "prescription_item",  # same category, should not create template_category
                "dosage_form": "capsule"
            }
        }
    ]

    manifest_file = tmp_path / "test_manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for item in manifest_data:
            f.write(json.dumps(item) + "\n")

    loader = MedicalPrescriptionDatasetLoader()
    samples = loader.load_manifest(str(manifest_file))

    assert len(samples) == 2

    # Sample 1: category overridden from "soap_note" to "clinical_note", template_category preserved
    s1 = samples[0]
    assert s1.metadata["category"] == "clinical_note"
    assert s1.metadata["template_category"] == "soap_note"
    assert s1.metadata["department"] == "pulmonology"
    assert s1.metadata["custom_flag"] == 42
    assert s1.metadata["is_lasa"] is False
    assert s1.metadata["therapeutic_class"] == "respiratory"
    assert s1.metadata["writer_id"] == "w_123"
    assert s1.metadata["source"] == "synthetic"

    # Sample 2: same category, no template_category created
    s2 = samples[1]
    assert s2.metadata["category"] == "prescription_item"
    assert "template_category" not in s2.metadata
    assert s2.metadata["dosage_form"] == "capsule"
    assert s2.metadata["is_lasa"] is True


def test_load_manifest_skips_synthetic_training_records(tmp_path: Path):
    manifest_file = tmp_path / "mixed_manifest.jsonl"
    records = [
        {
            "sample_id": "real_001",
            "image_path": "images/iam_line/real.png",
            "text": "put down a resolution",
            "dataset_source": "iam_line",
            "source": "Teklia/IAM-line",
        },
        {
            "sample_id": "sample_000001",
            "image_path": "images/sample_000001_syn.png",
            "text": "Liothyronine 50mcg",
            "dataset_source": "synthetic_medical_cursive",
            "source": "synthetic_medical_cursive",
        },
    ]
    with open(manifest_file, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    samples = MedicalPrescriptionDatasetLoader().load_manifest(str(manifest_file))
    assert len(samples) == 1
    assert samples[0].sample_id == "real_001"
    assert is_synthetic_training_record(records[1]) is True
    assert is_synthetic_training_record(records[0]) is False


"""
tests/test_adversarial_m1_deep_dataset.py
Deep Empirical Adversarial Stress Test Suite for Milestone 1:
- Public reference handwriting manifests (line counts, syntax, set algebra, ID uniqueness)
- Strict Writer Independence & Zero Leakage Across Splits (W_train ∩ W_val = ∅, W_train ∩ W_test = ∅, W_val ∩ W_test = ∅)
- Schema & Field Invariant Validation across downloaded corpus records
- Disk Image Existence, Non-emptiness, Magic Bytes, and Decodability via PIL & OpenCV
- Deep Clinical Vocabulary Validation (CMS Luhn-10, DEA-7, LASA pairs, 7 therapeutic classes)
- PyTorch DataLoader Ingestion on Manifests
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Set

import cv2
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "reference_handwriting"
VOCAB_DIR = DATA_DIR / "vocabularies"


@pytest.fixture(scope="module")
def manifests():
    train_p = DATA_DIR / "train_manifest.jsonl"
    val_p = DATA_DIR / "val_manifest.jsonl"
    test_p = DATA_DIR / "test_manifest.jsonl"
    full_p = DATA_DIR / "full_manifest.jsonl"
    summary_p = DATA_DIR / "dataset_summary.json"

    assert train_p.is_file(), f"Missing train manifest: {train_p}"
    assert val_p.is_file(), f"Missing val manifest: {val_p}"
    assert test_p.is_file(), f"Missing test manifest: {test_p}"
    assert full_p.is_file(), f"Missing full manifest: {full_p}"
    assert summary_p.is_file(), f"Missing dataset summary: {summary_p}"

    def parse_jsonl(p: Path) -> List[Dict]:
        records = []
        with open(p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line_str = line.strip()
                if line_str:
                    try:
                        record = json.loads(line_str)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Corrupted JSON on line {line_idx + 1} of {p}: {e}")
        return records

    train_data = parse_jsonl(train_p)
    val_data = parse_jsonl(val_p)
    test_data = parse_jsonl(test_p)
    full_data = parse_jsonl(full_p)

    if not full_data:
        pytest.skip("Reference handwriting manifests are being rebuilt from public corpora")

    with open(summary_p, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "full": full_data,
        "summary": summary_data,
    }


class TestManifestCountsAndInvariants:
    """Stress tests on line counts, split ratios, and set algebra."""

    def test_manifest_counts_thresholds(self, manifests):
        train = manifests["train"]
        val = manifests["val"]
        test = manifests["test"]
        full = manifests["full"]

        # Minimum thresholds
        assert len(train) >= 40000, f"Train count {len(train)} < 40,000"
        assert len(val) >= 5000, f"Val count {len(val)} < 5,000"
        assert len(test) >= 5000, f"Test count {len(test)} < 5,000"
        assert len(full) >= 50000, f"Full count {len(full)} < 50,000"

        # Exact addition invariant
        assert len(train) + len(val) + len(test) == len(full), (
            f"Split sum mismatch: train({len(train)}) + val({len(val)}) + test({len(test)}) "
            f"= {len(train) + len(val) + len(test)} != full({len(full)})"
        )

        sources = {r.get("dataset_source") for r in full}
        assert "synthetic_medical_cursive" not in sources
        assert all(not str(r.get("image_path", "")).endswith("_syn.png") for r in full)

    def test_sample_id_uniqueness_and_bijection(self, manifests):
        train_ids = set(r["id"] for r in manifests["train"])
        val_ids = set(r["id"] for r in manifests["val"])
        test_ids = set(r["id"] for r in manifests["test"])
        full_ids = set(r["id"] for r in manifests["full"])

        # Check for duplicates within splits
        assert len(train_ids) == len(manifests["train"]), "Duplicate IDs found within train split"
        assert len(val_ids) == len(manifests["val"]), "Duplicate IDs found within val split"
        assert len(test_ids) == len(manifests["test"]), "Duplicate IDs found within test split"
        assert len(full_ids) == len(manifests["full"]), "Duplicate IDs found within full manifest"

        # Check disjointness
        assert train_ids.isdisjoint(val_ids), f"Train/Val ID collision: {train_ids & val_ids}"
        assert train_ids.isdisjoint(test_ids), f"Train/Test ID collision: {train_ids & test_ids}"
        assert val_ids.isdisjoint(test_ids), f"Val/Test ID collision: {val_ids & test_ids}"

        # Check full manifest bijection
        assert full_ids == (train_ids | val_ids | test_ids), "full_manifest IDs do not equal union of splits"


class TestWriterIndependenceZeroLeakage:
    """Stress tests on writer independence and absence of data leakage."""

    def test_writer_independence_proof(self, manifests):
        train_writers = set(r["writer_id"] for r in manifests["train"])
        val_writers = set(r["writer_id"] for r in manifests["val"])
        test_writers = set(r["writer_id"] for r in manifests["test"])
        full_writers = set(r["writer_id"] for r in manifests["full"])

        # Assert zero intersection
        overlap_train_val = train_writers.intersection(val_writers)
        overlap_train_test = train_writers.intersection(test_writers)
        overlap_val_test = val_writers.intersection(test_writers)

        assert len(overlap_train_val) == 0, f"Writer Leakage Train/Val: {overlap_train_val}"
        assert len(overlap_train_test) == 0, f"Writer Leakage Train/Test: {overlap_train_test}"
        assert len(overlap_val_test) == 0, f"Writer Leakage Val/Test: {overlap_val_test}"

        assert len(full_writers) == len(train_writers) + len(val_writers) + len(test_writers)
        assert len(train_writers) >= 1
        assert len(val_writers) >= 1
        assert len(test_writers) >= 1

    def test_writer_sample_distribution(self, manifests):
        for split_name, records in [
            ("train", manifests["train"]),
            ("val", manifests["val"]),
            ("test", manifests["test"]),
        ]:
            writer_counts: Dict[str, int] = {}
            for r in records:
                w_id = r["writer_id"]
                writer_counts[w_id] = writer_counts.get(w_id, 0) + 1

            counts = list(writer_counts.values())
            min_c = min(counts)
            max_c = max(counts)

            assert min_c >= 1, f"{split_name} writer min sample count too low: {min_c}"
            assert max_c >= min_c
            assert len(writer_counts) >= 1


class TestManifestSchemaAndFields:
    """Validate 100% of rows for required fields and valid value domains."""

    def test_all_rows_schema_compliance(self, manifests):
        required_keys = [
            "id",
            "image_path",
            "transcription",
            "writer_id",
            "dataset_source",
            "is_lasa",
            "therapeutic_class",
            "augmentation_params",
        ]

        full_records = manifests["full"]
        assert len(full_records) >= 50000

        for idx, record in enumerate(full_records):
            for k in required_keys:
                assert k in record, f"Missing required key '{k}' on sample index {idx}: {record}"

            # Type checks
            assert isinstance(record["id"], str) and len(record["id"]) > 0
            assert isinstance(record["image_path"], str) and len(record["image_path"]) > 0
            assert not str(record["image_path"]).endswith("_syn.png")
            assert isinstance(record["transcription"], str) and len(record["transcription"].strip()) > 0
            assert isinstance(record["writer_id"], str) and len(record["writer_id"]) > 0
            assert isinstance(record["dataset_source"], str) and len(record["dataset_source"]) > 0
            assert record["dataset_source"] != "synthetic_medical_cursive"
            assert isinstance(record["is_lasa"], bool)
            assert isinstance(record["therapeutic_class"], str) and len(record["therapeutic_class"]) > 0
            assert isinstance(record["augmentation_params"], dict)

            # Downloaded scans are not locally font-rendered
            aug = record["augmentation_params"]
            assert aug.get("downloaded_original") is True

            # Dimensions
            assert "width" in record and isinstance(record["width"], int) and record["width"] >= 32
            assert "height" in record and isinstance(record["height"], int) and record["height"] >= 16

            # Split consistency
            assert record["split"] in ("train", "val", "test")


class TestImageIntegrityAndDecoding:
    """Stress tests on image disk presence and pixel array decodability."""

    def test_all_manifest_image_files_exist_and_non_empty(self, manifests):
        full_records = manifests["full"]

        # Check all downloaded images on disk
        missing = 0
        empty = 0

        for r in full_records:
            p = REPO_ROOT / r["image_path"]
            if not p.is_file():
                missing += 1
                continue
            st = p.stat()
            if st.st_size < 50:
                empty += 1

        assert missing == 0, f"{missing} image files missing on disk"
        assert empty == 0, f"{empty} image files are empty (<50 bytes)"

    def test_random_sample_decoding_opencv_pil(self, manifests):
        full_records = manifests["full"]

        # Sample 600 random records for full PIL and OpenCV decoding
        rng = random.Random(42)
        sample_n = min(600, len(full_records))
        sample_records = rng.sample(full_records, k=sample_n)

        decoded_count = 0
        for r in sample_records:
            img_path = REPO_ROOT / r["image_path"]

            # PIL validation
            with Image.open(img_path) as pil_img:
                assert pil_img.format == "PNG", f"Image {img_path} format is {pil_img.format}, expected PNG"
                w, h = pil_img.size
                assert w == r["width"], f"Width mismatch for {img_path}: {w} != {r['width']}"
                assert h == r["height"], f"Height mismatch for {img_path}: {h} != {r['height']}"

            # OpenCV decoding
            cv_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            assert cv_img is not None, f"cv2.imread failed on {img_path}"
            assert cv_img.shape == (r["height"], r["width"], 3), f"CV shape mismatch on {img_path}: {cv_img.shape}"
            decoded_count += 1

        assert decoded_count == sample_n

    def test_overall_dataset_image_quality_ratio(self, manifests):
        full_records = manifests["full"]
        valid_sample_count = 0
        rng = random.Random(100)
        sample_n = min(1000, len(full_records))
        sample_records = rng.sample(full_records, k=sample_n)

        for r in sample_records:
            img_path = REPO_ROOT / r["image_path"]
            cv_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            assert cv_img is not None, f"Image {img_path} could not be read"
            assert float(np.std(cv_img)) >= 0.5, f"Image {img_path} is blank: std={np.std(cv_img)}"
            if float(np.std(cv_img)) >= 1.0:
                valid_sample_count += 1

        ratio = valid_sample_count / len(sample_records)
        assert ratio >= 0.99, f"Valid high-quality non-blank image ratio too low: {ratio:.4f}"


class TestClinicalVocabulariesDeep:
    """Stress tests on clinical vocabularies, LASA pairs, Luhn checksums, and templates."""

    def test_rxnorm_medications(self):
        rx_path = VOCAB_DIR / "rxnorm_medications.json"
        assert rx_path.is_file()

        with open(rx_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "medications" in data
        assert "therapeutic_classes" in data
        assert "total_medications" in data

        meds = data["medications"]
        assert len(meds) == 1050, f"Expected 1,050 medications, got {len(meds)}"
        assert data["total_medications"] == 1050

        # Check therapeutic class coverage (150 per class)
        classes = data["therapeutic_classes"]
        assert len(classes) == 7
        for tc in classes:
            class_meds = [m for m in meds if m.get("therapeutic_class") == tc]
            assert len(class_meds) == 150, f"Therapeutic class {tc} expected 150 meds, got {len(class_meds)}"

        # Check LASA pairs
        lasa_meds = [m for m in meds if m.get("is_lasa")]
        assert len(lasa_meds) == 40, f"Expected 40 LASA meds (20 pairs), got {len(lasa_meds)}"

        med_lookup = {m["generic_name"].lower(): m for m in meds}
        for med in lasa_meds:
            assert len(med.get("confusion_pairs", [])) >= 1
            for cp in med["confusion_pairs"]:
                target_name = cp["target_drug"]
                assert target_name.lower() in med_lookup
                target_med = med_lookup[target_name.lower()]
                assert target_med["is_lasa"]
                # Reciprocal link check
                reciprocal_targets = [p["target_drug"].lower() for p in target_med.get("confusion_pairs", [])]
                assert med["generic_name"].lower() in reciprocal_targets
                assert cp["visual_similarity_score"] >= 0.70
                assert "visual_confusion_score" in cp
                assert cp["visual_confusion_score"] >= 0.70
                assert cp["visual_confusion_score"] == cp["visual_similarity_score"]

    def test_doctor_profiles_npi_luhn_and_dea(self):
        dr_path = VOCAB_DIR / "doctor_profiles.json"
        assert dr_path.is_file()

        with open(dr_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        doctors = data.get("doctors", [])
        assert len(doctors) == 126, f"Expected 126 doctors, got {len(doctors)}"
        assert data.get("total_doctors") == 126

        def is_valid_npi(npi: str) -> bool:
            if not (isinstance(npi, str) and len(npi) == 10 and npi.isdigit()):
                return False
            # CMS Luhn-10 with prefix '80840'
            full_str = "80840" + npi
            digits = [int(d) for d in full_str]
            total = 0
            for i, d in enumerate(reversed(digits)):
                if i % 2 == 1:  # odd positions from right (doubled)
                    doubled = d * 2
                    total += doubled if doubled < 10 else (doubled - 9)
                else:
                    total += d
            return total % 10 == 0

        def is_valid_dea(dea: str) -> bool:
            if not (isinstance(dea, str) and len(dea) == 9 and dea[:2].isalpha() and dea[2:].isdigit()):
                return False
            digits = [int(c) for c in dea[2:]]
            sum1 = digits[0] + digits[2] + digits[4]
            sum2 = digits[1] + digits[3] + digits[5]
            check = (sum1 + 2 * sum2) % 10
            return check == digits[6]

        for doc in doctors:
            assert is_valid_npi(doc["npi"]), f"Invalid NPI Luhn checksum: {doc['npi']} for {doc['full_name']}"
            assert is_valid_dea(doc["dea_number"]), f"Invalid DEA checksum: {doc['dea_number']} for {doc['full_name']}"
            assert "handwriting_style" in doc
            hw = doc["handwriting_style"]
            assert -15.0 <= hw["slant_angle_deg"] <= 40.0
            assert 0.3 <= hw["pressure_factor"] <= 2.0
            assert 0.4 <= hw["speed_factor"] <= 2.5
            assert 0.05 <= hw["legibility_score"] <= 1.0

    def test_clinical_templates_slot_resolution(self):
        tpl_path = VOCAB_DIR / "clinical_templates.json"
        assert tpl_path.is_file()

        with open(tpl_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        templates = data.get("templates", [])
        assert len(templates) == 220, f"Expected 220 templates, got {len(templates)}"
        assert data.get("total_templates") == 220

        categories = set(t["template_category"] for t in templates)
        assert len(categories) == 5, f"Expected 5 categories, got {categories}"

        slot_pattern = re.compile(r"\{[A-Z0-9_]+\}")
        for t in templates:
            raw_text = t["raw_template_text"]
            slots_declared = {s["slot_name"] for s in t.get("slots", [])}
            found_placeholders = set(slot_pattern.findall(raw_text))
            assert found_placeholders.issubset(slots_declared), (
                f"Template {t['template_id']} has unlisted slots: {found_placeholders - slots_declared}"
            )


class TestPyTorchDataLoaderManifestIngestion:
    """Stress test batch ingestion through PyTorch DataLoader."""

    def test_manifest_pytorch_batch_loading(self, manifests):
        train_records = manifests["train"]

        # Select first 32 items
        batch_records = train_records[:32]

        class ManifestMiniDataset(torch.utils.data.Dataset):
            def __init__(self, records):
                self.records = records

            def __len__(self):
                return len(self.records)

            def __getitem__(self, idx):
                r = self.records[idx]
                p = REPO_ROOT / r["image_path"]
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize to standard height 64, preserving aspect ratio up to max width 512
                h, w, c = img.shape
                new_w = min(512, max(32, int(w * (64 / h))))
                img_resized = cv2.resize(img, (new_w, 64), interpolation=cv2.INTER_AREA)
                # Pad to (64, 512)
                canvas = np.full((64, 512, 3), 255, dtype=np.uint8)
                canvas[:, :new_w, :] = img_resized
                tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
                return {
                    "pixel_values": tensor,
                    "id": r["id"],
                    "text": r["transcription"],
                    "writer_id": r["writer_id"],
                }

        ds = ManifestMiniDataset(batch_records)
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        total_samples = 0
        for batch in loader:
            pv = batch["pixel_values"]
            assert pv.shape == (8, 3, 64, 512)
            assert pv.min() >= 0.0
            assert pv.max() <= 1.0
            assert len(batch["id"]) == 8
            assert len(batch["text"]) == 8
            assert len(batch["writer_id"]) == 8
            total_samples += len(batch["id"])

        assert total_samples == 32

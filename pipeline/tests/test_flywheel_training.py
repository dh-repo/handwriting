"""
pipeline/tests/test_flywheel_training.py
Comprehensive test suite for Milestone 3 (Flywheel Training, Experience Replay & LASA Safety Gate).

Tests:
1. ExperienceReplayDataset & ReplayBatchSampler (data loading, ratio balance, edge cases).
2. load_lasa_catalog & audit_lasa_safety (20 bidirectional pairs, clean vs dangerous substitution detection).
3. check_cer_regression & decide_ship (extended CER regression gate & clinical zero-tolerance rule).
4. run_micro_tune (end-to-end micro-epoch execution, adapter save, weight merge).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tempfile
from typing import Any, List, Tuple

from PIL import Image
import pytest
import torch

from pipeline.training.dataset import create_dummy_processor
from pipeline.training.experience_replay import ExperienceReplayDataset, ReplayBatchSampler
from pipeline.training.lora_micro_tune import run_micro_tune
from pipeline.training.ship_gate import (
    LasaAuditResult,
    assert_shippable_checkpoint,
    audit_lasa_safety,
    check_cer_regression,
    decide_ship,
    load_lasa_catalog,
)
from pipeline.training.train import create_tiny_mock_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_feedback_and_anchor(tmp_path: Path):
    """Create a temporary feedback manifest and golden anchor dataset."""
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir(parents=True)

    # 1. Create 3 feedback samples
    manifest_lines = []
    for i in range(3):
        img_path = crop_dir / f"fb_{i:03d}.png"
        Image.new("RGB", (64, 64), color=(50 + i * 30, 80, 120)).save(img_path)
        manifest_lines.append(
            json.dumps(
                {
                    "feedback_id": f"fb_{i:03d}",
                    "line_crop": str(img_path),
                    "original_prediction": f"pred_err_{i}",
                    "operator_correction": f"correction_{i}",
                    "confidence": 0.65,
                    "document_id": "doc_001",
                    "timestamp": "2026-09-03T12:00:00Z",
                }
            )
        )

    # Add a base64-only feedback sample
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 200, 200)).save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    manifest_lines.append(
        json.dumps(
            {
                "feedback_id": "fb_b64_003",
                "line_crop": "",
                "line_crop_base64": b64_str,
                "operator_correction": "base64_correction",
            }
        )
    )

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    # 2. Create 6 golden anchor samples
    anchor_dir = tmp_path / "anchor"
    images_dir = anchor_dir / "images"
    images_dir.mkdir(parents=True)
    tsv_lines = []
    for j in range(6):
        fn = f"anchor_{j:03d}.png"
        Image.new("RGB", (64, 64), color=(150, 100 + j * 20, 50)).save(images_dir / fn)
        tsv_lines.append(f"{fn}\tanchor_text_{j}")

    (anchor_dir / "labels.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    return {
        "manifest_path": manifest_path,
        "anchor_dir": anchor_dir,
        "num_feedback": 4,
        "num_anchor": 6,
    }


# ---------------------------------------------------------------------------
# Experience Replay Dataset & Sampler Tests
# ---------------------------------------------------------------------------

def test_experience_replay_dataset_loading(mock_feedback_and_anchor):
    info = mock_feedback_and_anchor
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))

    ds = ExperienceReplayDataset(
        feedback_manifest_path=info["manifest_path"],
        anchor_dir=info["anchor_dir"],
        processor=processor,
        max_target_length=16,
    )

    assert len(ds) == info["num_feedback"] + info["num_anchor"]
    assert len(ds.feedback_indices) == info["num_feedback"]
    assert len(ds.anchor_indices) == info["num_anchor"]
    assert ds.feedback_indices == [0, 1, 2, 3]
    assert ds.anchor_indices == [4, 5, 6, 7, 8, 9]

    # Test feedback sample content
    fb_item = ds[0]
    assert fb_item["source"] == "feedback"
    assert fb_item["text"] == "correction_0"
    assert "pixel_values" in fb_item
    assert "input_ids" in fb_item
    assert fb_item["pixel_values"].shape == (3, 64, 64)

    # Test base64 feedback sample content
    b64_item = ds[3]
    assert b64_item["source"] == "feedback"
    assert b64_item["text"] == "base64_correction"
    assert b64_item["pixel_values"].shape == (3, 64, 64)

    # Test anchor sample content
    an_item = ds[4]
    assert an_item["source"] == "anchor"
    assert an_item["text"] == "anchor_text_0"
    assert an_item["pixel_values"].shape == (3, 64, 64)


def test_experience_replay_dataset_empty_fallbacks(mock_feedback_and_anchor, tmp_path: Path):
    info = mock_feedback_and_anchor
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))

    # 1. Empty feedback manifest (missing file)
    ds_no_fb = ExperienceReplayDataset(
        feedback_manifest_path=tmp_path / "non_existent.jsonl",
        anchor_dir=info["anchor_dir"],
        processor=processor,
    )
    assert len(ds_no_fb.feedback_indices) == 0
    assert len(ds_no_fb.anchor_indices) == info["num_anchor"]
    assert len(ds_no_fb) == info["num_anchor"]

    # 2. Empty anchor dir (missing labels.tsv)
    empty_anchor_dir = tmp_path / "empty_anchor"
    empty_anchor_dir.mkdir()
    ds_no_an = ExperienceReplayDataset(
        feedback_manifest_path=info["manifest_path"],
        anchor_dir=empty_anchor_dir,
        processor=processor,
    )
    assert len(ds_no_an.feedback_indices) == info["num_feedback"]
    assert len(ds_no_an.anchor_indices) == 0
    assert len(ds_no_an) == info["num_feedback"]


def test_replay_batch_sampler_ratio_balance():
    fb_indices = [0, 1, 2, 3]
    an_indices = list(range(4, 24))  # 20 anchor items
    batch_size = 4
    replay_ratio = 0.5  # 50% feedback, 50% anchor

    sampler = ReplayBatchSampler(
        feedback_indices=fb_indices,
        anchor_indices=an_indices,
        batch_size=batch_size,
        replay_ratio=replay_ratio,
        shuffle=True,
        seed=123,
    )

    batches = list(sampler)
    assert len(batches) > 0

    for b in batches:
        assert len(b) == batch_size
        fb_count = sum(1 for idx in b if idx in fb_indices)
        an_count = sum(1 for idx in b if idx in an_indices)
        assert fb_count == 2
        assert an_count == 2


def test_replay_batch_sampler_custom_ratio():
    fb_indices = [0, 1, 2]
    an_indices = list(range(3, 15))
    batch_size = 4
    replay_ratio = 0.25  # 1 feedback, 3 anchor

    sampler = ReplayBatchSampler(
        feedback_indices=fb_indices,
        anchor_indices=an_indices,
        batch_size=batch_size,
        replay_ratio=replay_ratio,
        shuffle=False,
    )

    for b in list(sampler):
        fb_count = sum(1 for idx in b if idx in fb_indices)
        an_count = sum(1 for idx in b if idx in an_indices)
        assert fb_count == 1
        assert an_count == 3


def test_replay_batch_sampler_empty_pools():
    # Only anchor available
    sampler_an_only = ReplayBatchSampler(
        feedback_indices=[],
        anchor_indices=[10, 11, 12, 13, 14, 15],
        batch_size=2,
        replay_ratio=0.5,
    )
    batches_an = list(sampler_an_only)
    assert len(batches_an) == 3
    for b in batches_an:
        assert len(b) == 2
        assert all(idx in [10, 11, 12, 13, 14, 15] for idx in b)

    # Only feedback available
    sampler_fb_only = ReplayBatchSampler(
        feedback_indices=[1, 2, 3, 4],
        anchor_indices=[],
        batch_size=2,
        replay_ratio=0.5,
    )
    batches_fb = list(sampler_fb_only)
    assert len(batches_fb) == 2
    for b in batches_fb:
        assert len(b) == 2
        assert all(idx in [1, 2, 3, 4] for idx in b)

    # Both empty
    sampler_empty = ReplayBatchSampler(feedback_indices=[], anchor_indices=[], batch_size=4)
    assert len(sampler_empty) == 0
    assert list(sampler_empty) == []


# ---------------------------------------------------------------------------
# LASA Catalog & Safety Audit Tests
# ---------------------------------------------------------------------------

def test_load_lasa_catalog():
    catalog = load_lasa_catalog()
    assert isinstance(catalog, list)
    assert len(catalog) == 20, f"Expected exactly 20 canonical LASA pairs, got {len(catalog)}"

    # Check key pairs
    pair_set = {tuple(sorted([a.lower(), b.lower()])) for a, b in catalog}
    assert ("hydralazine", "hydroxyzine") in pair_set
    assert ("adderall", "inderal") in pair_set
    assert ("amoxicillin", "ampicillin") in pair_set
    assert ("prednisolone", "prednisone") in pair_set
    assert ("zyprexa", "zyrtec") in pair_set


def test_audit_lasa_safety_clean():
    refs = [
        "Hydralazine 25mg PO BID",
        "Amoxicillin 500mg capsules TID",
        "Prednisone 10mg daily taper",
        "Normal saline 0.9% 1000ml IV",
    ]
    hyps = [
        "Hydralazine 25mg PO BID",
        "Amoxicillin 500mg capsules TID",
        "Prednisone 10mg daily taper",
        "Normal saline 0.9% 1000ml IV",
    ]

    result = audit_lasa_safety(refs, hyps)
    assert result.passed is True
    assert len(result.violations) == 0
    assert result.total_evaluated == 4


def test_audit_lasa_safety_benign_ocr_errors():
    # Inexact spelling errors that are NOT dangerous LASA substitutions
    refs = [
        "Hydralazine 25mg PO BID",
        "Amoxicillin 500mg TID",
    ]
    hyps = [
        "Hydralazin 25mg PO BID",  # minor truncation, but not Hydroxyzine
        "Amoxcilin 500mg TID",     # misspelling, but not Ampicillin
    ]

    result = audit_lasa_safety(refs, hyps)
    assert result.passed is True
    assert len(result.violations) == 0


def test_audit_lasa_safety_detects_dangerous_substitutions():
    refs = [
        "Patient prescribed Hydralazine 25mg PO BID for hypertension",
        "Amoxicillin 500mg PO TID for bacterial infection",
        "Metformin 1000mg BID for diabetes",
    ]
    hyps = [
        "Patient prescribed Hydroxyzine 25mg PO BID for hypertension",  # VIOLATION: Hydralazine -> Hydroxyzine
        "Ampicillin 500mg PO TID for bacterial infection",              # VIOLATION: Amoxicillin -> Ampicillin
        "Metformin 1000mg BID for diabetes",                            # Clean
    ]

    result = audit_lasa_safety(refs, hyps)
    assert result.passed is False
    assert len(result.violations) == 2
    assert result.total_evaluated == 3

    v1 = result.violations[0]
    assert v1["prescribed_drug"] == "Hydralazine" or v1["confused_drug"] == "Hydroxyzine"
    assert v1["index"] == 0

    v2 = result.violations[1]
    assert v2["prescribed_drug"] == "Amoxicillin" or v2["confused_drug"] == "Ampicillin"
    assert v2["index"] == 1


def test_audit_lasa_safety_bidirectional():
    # Reverse direction: Hydroxyzine -> Hydralazine
    refs = ["Hydroxyzine 50mg PO QHS for anxiety/itching"]
    hyps = ["Hydralazine 50mg PO QHS for anxiety/itching"]

    result = audit_lasa_safety(refs, hyps)
    assert result.passed is False
    assert len(result.violations) == 1
    assert result.violations[0]["prescribed_drug"] == "Hydroxyzine"
    assert result.violations[0]["confused_drug"] == "Hydralazine"


# ---------------------------------------------------------------------------
# CER Regression & Ship Gate Tests
# ---------------------------------------------------------------------------

def test_check_cer_regression():
    baseline = 0.0400  # 4.0% CER

    # Strict improvement
    assert check_cer_regression(0.0380, baseline, max_cer_regression=0.05) is True

    # Exact match
    assert check_cer_regression(0.0400, baseline, max_cer_regression=0.05) is True

    # 2.5% relative degradation (0.0410 <= 0.0420)
    assert check_cer_regression(0.0410, baseline, max_cer_regression=0.05) is True

    # Exactly 5.0% degradation (0.0420 <= 0.0420)
    assert check_cer_regression(0.0420, baseline, max_cer_regression=0.05) is True

    # 7.5% degradation (> 5.0% tolerance, 0.0430 > 0.0420)
    assert check_cer_regression(0.0430, baseline, max_cer_regression=0.05) is False


def test_decide_ship_with_lasa_and_cer_regression(tmp_path: Path):
    clean_audit = LasaAuditResult(total_evaluated=10, violations=[], passed=True)
    violation_audit = LasaAuditResult(
        total_evaluated=10,
        violations=[{"prescribed_drug": "Hydralazine", "confused_drug": "Hydroxyzine"}],
        passed=False,
    )

    report_good = {"checkpoint": "runs/test_model/best", "cer": 0.041, "num_beams": 1}
    report_bad_cer = {"checkpoint": "runs/test_model/best", "cer": 0.046, "num_beams": 1}

    # Case 1: Both CER within tolerance and LASA clean -> PROMOTE
    decision_pass = decide_ship(
        report_good,
        baseline_cer=0.040,
        max_cer_regression=0.05,
        lasa_audit=clean_audit,
        output_dir=tmp_path / "run_pass",
    )
    assert decision_pass["promote"] is True
    assert decision_pass["cer_passed"] is True
    assert decision_pass["lasa_passed"] is True
    assert (tmp_path / "run_pass" / "ship_decision.json").is_file()

    # Case 2: CER within tolerance but LASA violated -> REJECT
    decision_lasa_fail = decide_ship(
        report_good,
        baseline_cer=0.040,
        max_cer_regression=0.05,
        lasa_audit=violation_audit,
        output_dir=tmp_path / "run_lasa_fail",
    )
    assert decision_lasa_fail["promote"] is False
    assert decision_lasa_fail["cer_passed"] is True
    assert decision_lasa_fail["lasa_passed"] is False
    assert "LASA" in decision_lasa_fail["reason"]

    # Case 3: CER exceeds tolerance even if LASA clean -> REJECT
    decision_cer_fail = decide_ship(
        report_bad_cer,
        baseline_cer=0.040,
        max_cer_regression=0.05,
        lasa_audit=clean_audit,
        output_dir=tmp_path / "run_cer_fail",
    )
    assert decision_cer_fail["promote"] is False
    assert decision_cer_fail["cer_passed"] is False
    assert decision_cer_fail["lasa_passed"] is True
    assert "regressed" in decision_cer_fail["reason"]


def test_decide_ship_asserts_shippable_checkpoint():
    with pytest.raises(ValueError, match="stage1"):
        decide_ship({"checkpoint": "runs/base_iam_v1/best", "cer": 0.01})


# ---------------------------------------------------------------------------
# Background LoRA Micro-Tuning Integration Tests
# ---------------------------------------------------------------------------

def test_lora_micro_tune_execution(tmp_path: Path):
    """Verify run_micro_tune executes training steps, exports adapter, and merges weights."""
    fb_manifest = tmp_path / "manifest.jsonl"
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    crop_img = crop_dir / "fb1.png"
    Image.new("RGB", (64, 64), (100, 150, 200)).save(crop_img)
    fb_manifest.write_text(
        json.dumps({"line_crop": str(crop_img), "operator_correction": "sample correction"}) + "\n"
    )

    anchor_dir = tmp_path / "anchor"
    images_dir = anchor_dir / "images"
    images_dir.mkdir(parents=True)
    an_img = images_dir / "a1.png"
    Image.new("RGB", (64, 64), (200, 150, 100)).save(an_img)
    (anchor_dir / "labels.tsv").write_text("a1.png\tsample anchor\n", encoding="utf-8")

    proc = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    out_dir = tmp_path / "out_micro"

    res = run_micro_tune(
        feedback_manifest=fb_manifest,
        anchor_dir=anchor_dir,
        val_dir=None,
        output_dir=out_dir,
        steps=2,
        batch_size=2,
        replay_ratio=0.5,
        device="cpu",
        model=model,
        processor=proc,
        baseline_cer=0.05,
        max_cer_regression=0.05,
        merge_adapter=True,
    )

    assert res["steps"] == 2
    assert len(res["losses"]) == 2
    assert (out_dir / "adapter").exists()
    assert (out_dir / "adapter_weights.pt").exists()
    assert res["ship_decision"]["promote"] is True
    assert (out_dir / "best_model_merged").exists()


def test_lora_micro_tune_lasa_rejection(tmp_path: Path):
    """Verify run_micro_tune rejects checkpoints that trigger a LASA violation."""
    fb_manifest = tmp_path / "manifest.jsonl"
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    crop_img = crop_dir / "fb1.png"
    Image.new("RGB", (64, 64), (100, 150, 200)).save(crop_img)
    fb_manifest.write_text(
        json.dumps({"line_crop": str(crop_img), "operator_correction": "Hydralazine 25mg"}) + "\n"
    )

    anchor_dir = tmp_path / "anchor"
    (anchor_dir / "images").mkdir(parents=True)
    (anchor_dir / "labels.tsv").write_text("a1.png\tAmoxicillin 500mg\n", encoding="utf-8")
    Image.new("RGB", (64, 64), (200, 150, 100)).save(anchor_dir / "images" / "a1.png")

    proc = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    out_dir = tmp_path / "out_lasa_reject"

    # Inject explicit evaluation samples with a dangerous LASA error
    lasa_test_samples = [
        ("Hydralazine 25mg PO BID", "Hydroxyzine 25mg PO BID"),  # Substitution!
    ]

    res = run_micro_tune(
        feedback_manifest=fb_manifest,
        anchor_dir=anchor_dir,
        val_dir=None,
        output_dir=out_dir,
        steps=1,
        batch_size=2,
        device="cpu",
        model=model,
        processor=proc,
        baseline_cer=0.05,
        lasa_test_samples=lasa_test_samples,
        merge_adapter=True,
    )

    assert res["ship_decision"]["promote"] is False
    assert res["ship_decision"]["lasa_passed"] is False
    assert len(res["ship_decision"]["lasa_violations"]) == 1
    # Check that merge was aborted
    assert not (out_dir / "best_model_merged").exists()


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
    reason="MPS not available on this environment",
)
def test_lora_micro_tune_mps_execution(tmp_path: Path):
    """Verify run_micro_tune executes on Apple Silicon MPS with bf16 and periodic cache clearing."""
    fb_manifest = tmp_path / "manifest.jsonl"
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    crop_img = crop_dir / "fb1.png"
    Image.new("RGB", (64, 64), (100, 150, 200)).save(crop_img)
    fb_manifest.write_text(
        json.dumps({"line_crop": str(crop_img), "operator_correction": "mps correction"}) + "\n"
    )

    anchor_dir = tmp_path / "anchor"
    images_dir = anchor_dir / "images"
    images_dir.mkdir(parents=True)
    an_img = images_dir / "a1.png"
    Image.new("RGB", (64, 64), (200, 150, 100)).save(an_img)
    (anchor_dir / "labels.tsv").write_text("a1.png\tmps anchor\n", encoding="utf-8")

    proc = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    out_dir = tmp_path / "out_mps"

    res = run_micro_tune(
        feedback_manifest=fb_manifest,
        anchor_dir=anchor_dir,
        val_dir=None,
        output_dir=out_dir,
        steps=2,
        batch_size=2,
        replay_ratio=0.5,
        device="mps",
        empty_cache_steps=1,
        model=model,
        processor=proc,
        baseline_cer=0.05,
        max_cer_regression=0.05,
        merge_adapter=True,
    )

    assert res["steps"] == 2
    assert len(res["losses"]) == 2
    assert (out_dir / "adapter").exists()
    assert (out_dir / "adapter_weights.pt").exists()
    assert res["ship_decision"]["promote"] is True
    assert (out_dir / "best_model_merged").exists()


"""
pipeline/tests/test_challenger_m3_replay_stress.py
Empirical Challenger Test Suite for Milestone 3 (Experience Replay Dataset & Sampler).

Covers:
1. Ratio Invariant Stress:
   - 1 feedback vs 1000 anchors (150 batches)
   - 50 feedback vs 50 anchors (120 batches)
   - 200 feedback vs 10 anchors (150 batches)
   - Arbitrary replay ratios (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
   - Boundary batch sizes (batch_size=1, 2, 3, 5, 7, 8, 16)
2. Sampler Behavior:
   - drop_last=True vs drop_last=False
   - max_batches constraint
   - Empty feedback, empty anchor, both empty
   - Invalid arguments validation
3. Dataset Edge Cases & Resilience:
   - Non-existent and empty files
   - Corrupted JSON lines in manifest
   - Non-dict JSON records (empirical regression check)
   - Missing and corrupted image files on disk
   - Base64 encoding variations (data URLs, corrupted base64, non-image base64)
4. PyTorch DataLoader Integration:
   - num_workers=0 vs num_workers=2
   - Collation and tensor shape verification
   - Shuffle and seed determinism vs epoch progression
   - Multi-epoch restart stability
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, List

from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from pipeline.training.data import LineCropCollator
from pipeline.training.dataset import create_dummy_processor
from pipeline.training.experience_replay import ExperienceReplayDataset, ReplayBatchSampler


# ---------------------------------------------------------------------------
# 1. Ratio Invariant Stress Tests
# ---------------------------------------------------------------------------

def test_replay_ratio_invariant_1fb_1000anc():
    """
    Stress test extreme ratio imbalance: 1 feedback vs 1000 anchors.
    Verify EVERY batch out of 150 simulated batches contains exactly 4 feedback and 4 anchors.
    """
    fb_indices = [0]
    an_indices = list(range(1, 1001))
    batch_size = 8
    replay_ratio = 0.5
    num_test_batches = 150

    sampler = ReplayBatchSampler(
        feedback_indices=fb_indices,
        anchor_indices=an_indices,
        batch_size=batch_size,
        replay_ratio=replay_ratio,
        shuffle=False,
    )

    batches = []
    it = iter(sampler)
    for _ in range(num_test_batches):
        batches.append(next(it))

    assert len(batches) == num_test_batches

    for batch_idx, batch in enumerate(batches):
        assert len(batch) == batch_size, f"Batch {batch_idx} length {len(batch)} != {batch_size}"
        fb_in_batch = [idx for idx in batch if idx in fb_indices]
        an_in_batch = [idx for idx in batch if idx in an_indices]

        assert len(fb_in_batch) == 4, (
            f"Batch {batch_idx} has {len(fb_in_batch)} feedback samples, expected exactly 4"
        )
        assert len(an_in_batch) == 4, (
            f"Batch {batch_idx} has {len(an_in_batch)} anchor samples, expected exactly 4"
        )
        # Verify feedback sample 0 was repeated 4 times
        assert fb_in_batch == [0, 0, 0, 0]

    # Verify anchor index progression without premature looping in first 150 batches
    # (150 * 4 = 600 unique anchors should have been visited sequentially)
    all_anchors_seen = []
    for batch in batches:
        all_anchors_seen.extend([idx for idx in batch if idx in an_indices])
    expected_anchors = list(range(1, 601))
    assert all_anchors_seen == expected_anchors


def test_replay_ratio_invariant_50fb_50anc():
    """
    Stress test balanced ratio: 50 feedback vs 50 anchors.
    Verify EVERY batch out of 120 simulated batches across epochs contains exactly 4 feedback and 4 anchors.
    """
    fb_indices = list(range(50))
    an_indices = list(range(50, 100))
    batch_size = 8
    replay_ratio = 0.5
    num_test_batches = 120

    sampler = ReplayBatchSampler(
        feedback_indices=fb_indices,
        anchor_indices=an_indices,
        batch_size=batch_size,
        replay_ratio=replay_ratio,
        shuffle=False,
    )

    batches: List[List[int]] = []
    it = iter(sampler)
    while len(batches) < num_test_batches:
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(sampler)
            batches.append(next(it))

    assert len(batches) == num_test_batches

    for batch_idx, batch in enumerate(batches):
        assert len(batch) == batch_size
        fb_count = sum(1 for idx in batch if idx in fb_indices)
        an_count = sum(1 for idx in batch if idx in an_indices)
        assert fb_count == 4, f"Batch {batch_idx} fb_count={fb_count}, expected 4"
        assert an_count == 4, f"Batch {batch_idx} an_count={an_count}, expected 4"


def test_replay_ratio_invariant_200fb_10anc():
    """
    Stress test inverted ratio imbalance: 200 feedback vs 10 anchors.
    Verify EVERY batch out of 150 simulated batches across epochs contains exactly 4 feedback and 4 anchors.
    """
    fb_indices = list(range(200))
    an_indices = list(range(200, 210))
    batch_size = 8
    replay_ratio = 0.5
    num_test_batches = 150

    sampler = ReplayBatchSampler(
        feedback_indices=fb_indices,
        anchor_indices=an_indices,
        batch_size=batch_size,
        replay_ratio=replay_ratio,
        shuffle=False,
    )

    batches: List[List[int]] = []
    it = iter(sampler)
    while len(batches) < num_test_batches:
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(sampler)
            batches.append(next(it))

    assert len(batches) == num_test_batches

    for batch_idx, batch in enumerate(batches):
        assert len(batch) == batch_size
        fb_count = sum(1 for idx in batch if idx in fb_indices)
        an_count = sum(1 for idx in batch if idx in an_indices)
        assert fb_count == 4, f"Batch {batch_idx} fb_count={fb_count}, expected 4"
        assert an_count == 4, f"Batch {batch_idx} an_count={an_count}, expected 4"



@pytest.mark.parametrize(
    "ratio,expected_fb,expected_an",
    [
        (0.125, 1, 7),
        (0.25, 2, 6),
        (0.375, 3, 5),
        (0.50, 4, 4),
        (0.625, 5, 3),
        (0.75, 6, 2),
        (0.875, 7, 1),
    ],
)
def test_replay_ratio_various_proportions(ratio: float, expected_fb: int, expected_an: int):
    """
    Verify mini-batch proportion invariants across varied fractions of batch_size=8.
    Simulate 100 batches each.
    """
    fb_indices = list(range(30))
    an_indices = list(range(30, 100))
    batch_size = 8

    sampler = ReplayBatchSampler(
        feedback_indices=fb_indices,
        anchor_indices=an_indices,
        batch_size=batch_size,
        replay_ratio=ratio,
        shuffle=True,
        seed=42,
    )

    batches = list(sampler)
    assert len(batches) > 0

    for b_idx, batch in enumerate(batches):
        assert len(batch) == batch_size
        fb_count = sum(1 for idx in batch if idx in fb_indices)
        an_count = sum(1 for idx in batch if idx in an_indices)
        assert fb_count == expected_fb, (
            f"Ratio {ratio} Batch {b_idx}: got {fb_count} feedback, expected {expected_fb}"
        )
        assert an_count == expected_an, (
            f"Ratio {ratio} Batch {b_idx}: got {an_count} anchor, expected {expected_an}"
        )


def test_replay_ratio_extremes_0_and_1():
    """Verify boundary ratios 0.0 (all anchor) and 1.0 (all feedback)."""
    fb_indices = list(range(10))
    an_indices = list(range(10, 30))
    batch_size = 8

    # ratio = 0.0 -> 0 feedback, 8 anchor
    s_0 = ReplayBatchSampler(fb_indices, an_indices, batch_size=batch_size, replay_ratio=0.0)
    assert s_0.n_feedback == 0
    assert s_0.n_anchor == 8
    for batch in s_0:
        assert all(idx in an_indices for idx in batch)
        assert len(batch) == 8

    # ratio = 1.0 -> 8 feedback, 0 anchor
    s_1 = ReplayBatchSampler(fb_indices, an_indices, batch_size=batch_size, replay_ratio=1.0)
    assert s_1.n_feedback == 8
    assert s_1.n_anchor == 0
    for batch in s_1:
        assert all(idx in fb_indices for idx in batch)
        assert len(batch) == 8


@pytest.mark.parametrize(
    "batch_size,expected_fb,expected_an",
    [
        (2, 1, 1),
        (3, 2, 1),  # round(1.5) = 2
        (4, 2, 2),
        (5, 2, 3),  # round(2.5) = 2
        (6, 3, 3),
        (7, 4, 3),  # round(3.5) = 4
        (16, 8, 8),
    ],
)
def test_replay_batch_sizes_odd_and_even(batch_size: int, expected_fb: int, expected_an: int):
    """Verify exact sample counts for various odd and even batch sizes with replay_ratio=0.5."""
    fb_indices = list(range(20))
    an_indices = list(range(20, 50))

    sampler = ReplayBatchSampler(
        feedback_indices=fb_indices,
        anchor_indices=an_indices,
        batch_size=batch_size,
        replay_ratio=0.5,
        shuffle=False,
    )

    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        assert len(batch) == batch_size
        fb_count = sum(1 for idx in batch if idx in fb_indices)
        an_count = sum(1 for idx in batch if idx in an_indices)
        assert fb_count == expected_fb
        assert an_count == expected_an


def test_replay_batch_size_1_boundary():
    """
    Examine batch_size=1 boundary:
    With batch_size=1 and replay_ratio=0.5, a single sample cannot be split into halves.
    The sampler resolves this boundary to n_feedback=0, n_anchor=1.
    """
    fb_indices = list(range(5))
    an_indices = list(range(5, 15))

    sampler = ReplayBatchSampler(
        feedback_indices=fb_indices,
        anchor_indices=an_indices,
        batch_size=1,
        replay_ratio=0.5,
    )

    assert sampler.n_feedback == 0
    assert sampler.n_anchor == 1
    batches = list(sampler)
    assert len(batches) == len(an_indices)
    for b in batches:
        assert len(b) == 1
        assert b[0] in an_indices


# ---------------------------------------------------------------------------
# 2. Sampler Parameters & Edge Cases
# ---------------------------------------------------------------------------

def test_sampler_drop_last_true_vs_false():
    """Verify drop_last calculation when anchor pool does not divide evenly."""
    fb_indices = [0]
    an_indices = list(range(1, 11))  # 10 anchors, n_anchor=4 -> 2 full batches + 2 remainder
    batch_size = 8

    # drop_last=False: total batches should be 3
    sampler_keep = ReplayBatchSampler(
        fb_indices, an_indices, batch_size=batch_size, replay_ratio=0.5, drop_last=False
    )
    batches_keep = list(sampler_keep)
    assert len(sampler_keep) == 3
    assert len(batches_keep) == 3

    # drop_last=True: total batches should be 2
    sampler_drop = ReplayBatchSampler(
        fb_indices, an_indices, batch_size=batch_size, replay_ratio=0.5, drop_last=True
    )
    batches_drop = list(sampler_drop)
    assert len(sampler_drop) == 2
    assert len(batches_drop) == 2


def test_sampler_max_batches_constraint():
    """Verify max_batches truncates iteration properly."""
    fb_indices = list(range(10))
    an_indices = list(range(10, 500))  # would normally generate ~122 batches
    sampler = ReplayBatchSampler(
        fb_indices, an_indices, batch_size=8, replay_ratio=0.5, max_batches=15
    )

    assert len(sampler) == 15
    batches = list(sampler)
    assert len(batches) == 15


def test_sampler_invalid_arguments():
    """Verify ValueError is raised on invalid initialization parameters."""
    with pytest.raises(ValueError, match="batch_size"):
        ReplayBatchSampler([0], [1], batch_size=0)

    with pytest.raises(ValueError, match="batch_size"):
        ReplayBatchSampler([0], [1], batch_size=-4)

    with pytest.raises(ValueError, match="replay_ratio"):
        ReplayBatchSampler([0], [1], batch_size=4, replay_ratio=-0.1)

    with pytest.raises(ValueError, match="replay_ratio"):
        ReplayBatchSampler([0], [1], batch_size=4, replay_ratio=1.1)


# ---------------------------------------------------------------------------
# 3. Dataset Edge Cases & Resilience
# ---------------------------------------------------------------------------

def test_dataset_empty_and_corrupt_manifest(tmp_path: Path):
    """Verify dataset handling of empty and syntactically malformed JSONL files."""
    proc = create_dummy_processor(vocab_size=50, size=(64, 64))

    # 1. Manifest file with blank lines and broken JSON
    corrupt_manifest = tmp_path / "corrupt_manifest.jsonl"
    corrupt_manifest.write_text(
        "\n\n"
        "{bad json line\n"
        "   \n"
        '{"feedback_id": "ok_1", "operator_correction": "valid"}\n'
        "not even close to json\n",
        encoding="utf-8",
    )

    ds = ExperienceReplayDataset(feedback_manifest_path=corrupt_manifest, processor=proc)
    # Only 1 valid line should have been ingested
    assert len(ds.feedback_indices) == 1
    assert ds[0]["sample_id"] == "ok_1"
    assert ds[0]["text"] == "valid"


def test_dataset_missing_image_file_fallback(tmp_path: Path):
    """Verify that referencing a non-existent image path falls back to blank canvas without crashing."""
    proc = create_dummy_processor(vocab_size=50, size=(64, 64))

    manifest = tmp_path / "missing_img_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "feedback_id": "fb_missing",
                "line_crop": "/non/existent/path/to/missing_crop.png",
                "operator_correction": "missing image text",
            }
        )
        + "\n"
    )

    ds = ExperienceReplayDataset(feedback_manifest_path=manifest, processor=proc)
    assert len(ds) == 1
    item = ds[0]
    assert item["text"] == "missing image text"
    assert "pixel_values" in item
    assert item["pixel_values"].shape == (3, 64, 64)


def test_dataset_base64_payload_variations(tmp_path: Path):
    """Verify inline base64 image parsing, data URL prefix handling, and invalid base64 fallback."""
    proc = create_dummy_processor(vocab_size=50, size=(64, 64))

    # 1. Valid raw base64
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="PNG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 2. Data URL prefix base64
    data_url_b64 = f"data:image/png;base64,{raw_b64}"

    # 3. Invalid base64 string
    invalid_b64 = "this_is_not_valid_base64_!@#$"

    # 4. Valid base64 but NOT an image
    non_img_b64 = base64.b64encode(b"plain text payload not an image").decode("utf-8")

    manifest = tmp_path / "b64_manifest.jsonl"
    manifest.write_text(
        json.dumps({"feedback_id": "fb_raw", "line_crop_base64": raw_b64, "operator_correction": "raw"}) + "\n"
        + json.dumps({"feedback_id": "fb_url", "line_crop_base64": data_url_b64, "operator_correction": "url"}) + "\n"
        + json.dumps({"feedback_id": "fb_inv", "line_crop_base64": invalid_b64, "operator_correction": "inv"}) + "\n"
        + json.dumps({"feedback_id": "fb_non_img", "line_crop_base64": non_img_b64, "operator_correction": "non_img"}) + "\n"
    )

    ds = ExperienceReplayDataset(feedback_manifest_path=manifest, processor=proc)
    assert len(ds) == 4

    for i in range(4):
        item = ds[i]
        assert "pixel_values" in item
        assert item["pixel_values"].shape == (3, 64, 64)


def test_dataset_index_out_of_bounds(tmp_path: Path):
    """Verify IndexError is raised for out-of-bounds indices."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"operator_correction": "test"}) + "\n")
    ds = ExperienceReplayDataset(feedback_manifest_path=manifest)

    with pytest.raises(IndexError):
        _ = ds[-1]

    with pytest.raises(IndexError):
        _ = ds[1]


def test_manifest_non_dict_json_vulnerability(tmp_path: Path):
    """
    Vulnerability finding:
    When manifest.jsonl contains valid JSON that is not a dictionary (e.g. [1, 2, 3] or "hello"),
    _parse_feedback_manifest calls data.get(...) on a non-dict object, raising AttributeError
    instead of cleanly skipping the malformed line.
    """
    manifest = tmp_path / "non_dict_manifest.jsonl"
    manifest.write_text("[1, 2, 3]\n", encoding="utf-8")

    # Empirically reproduces the unhandled AttributeError crash
    with pytest.raises(AttributeError, match="'list' object has no attribute 'get'"):
        ExperienceReplayDataset(feedback_manifest_path=manifest)


def test_corrupt_image_logging_format_string_bug(tmp_path: Path, caplog):
    """
    Bug finding:
    In ExperienceReplayDataset._load_pil_image line 232:
    logger.warning("Failed opening image file %s: %exc", img_path, exc)
    uses '%exc' instead of '%s: %s'. Since '%e' is a float format specifier in Python logging,
    this raises TypeError: must be real number, not UnidentifiedImageError whenever
    a logging handler attempts to format the record.
    """
    corrupt_img = tmp_path / "corrupted.png"
    corrupt_img.write_bytes(b"NOT_A_VALID_IMAGE_HEADER_DATA")

    manifest = tmp_path / "corrupt_img_manifest.jsonl"
    manifest.write_text(
        json.dumps({"line_crop": str(corrupt_img), "operator_correction": "text"}) + "\n",
        encoding="utf-8",
    )

    ds = ExperienceReplayDataset(feedback_manifest_path=manifest)

    # Under pytest caplog (which formats log records), this triggers TypeError
    with pytest.raises(TypeError, match="must be real number"):
        _ = ds[0]



# ---------------------------------------------------------------------------
# 4. PyTorch DataLoader Integration Tests
# ---------------------------------------------------------------------------

def test_dataloader_num_workers_0_and_2(tmp_path: Path):
    """
    Verify PyTorch DataLoader integration with ReplayBatchSampler.
    Tests both num_workers=0 (main process) and num_workers=2 (multiprocessing spawn).
    """
    proc = create_dummy_processor(vocab_size=50, size=(64, 64))

    # Create 4 feedback samples
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    fb_lines = []
    for i in range(4):
        img_p = crop_dir / f"fb_{i}.png"
        Image.new("RGB", (64, 64), (10 * i, 20, 30)).save(img_p)
        fb_lines.append(json.dumps({"line_crop": str(img_p), "operator_correction": f"fb_corr_{i}"}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(fb_lines) + "\n")

    # Create 12 anchor samples
    anchor_dir = tmp_path / "anchor"
    images_dir = anchor_dir / "images"
    images_dir.mkdir(parents=True)
    tsv = []
    for j in range(12):
        img_p = images_dir / f"anc_{j}.png"
        Image.new("RGB", (64, 64), (50, 10 * j, 70)).save(img_p)
        tsv.append(f"anc_{j}.png\tanc_label_{j}")
    (anchor_dir / "labels.tsv").write_text("\n".join(tsv) + "\n")

    ds = ExperienceReplayDataset(
        feedback_manifest_path=manifest,
        anchor_dir=anchor_dir,
        processor=proc,
        max_target_length=16,
    )
    sampler = ReplayBatchSampler(
        ds.feedback_indices,
        ds.anchor_indices,
        batch_size=4,
        replay_ratio=0.5,
        shuffle=True,
        seed=123,
    )
    collator = LineCropCollator(processor=proc, max_target_length=16)

    # Test num_workers=0
    loader0 = DataLoader(ds, batch_sampler=sampler, collate_fn=collator, num_workers=0)
    batches0 = list(loader0)
    assert len(batches0) > 0
    for b in batches0:
        assert b["pixel_values"].shape[0] == 4
        assert b["pixel_values"].shape[1:] == (3, 64, 64)
        assert b["labels"].shape[0] == 4

    # Test num_workers=2
    loader2 = DataLoader(ds, batch_sampler=sampler, collate_fn=collator, num_workers=2)
    batches2 = list(loader2)
    assert len(batches2) == len(batches0)
    for b in batches2:
        assert b["pixel_values"].shape[0] == 4
        assert b["labels"].shape[0] == 4


def test_dataloader_seed_determinism_and_multi_epoch(tmp_path: Path):
    """Verify deterministic epoch repeats when seed is fixed, and multi-epoch progression."""
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    fb_lines = []
    for i in range(4):
        img_p = crop_dir / f"fb_{i}.png"
        Image.new("RGB", (64, 64)).save(img_p)
        fb_lines.append(json.dumps({"line_crop": str(img_p), "operator_correction": f"fb_{i}"}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(fb_lines) + "\n")

    anchor_dir = tmp_path / "anchor"
    images_dir = anchor_dir / "images"
    images_dir.mkdir(parents=True)
    tsv = []
    for j in range(8):
        img_p = images_dir / f"anc_{j}.png"
        Image.new("RGB", (64, 64)).save(img_p)
        tsv.append(f"anc_{j}.png\tanc_{j}")
    (anchor_dir / "labels.tsv").write_text("\n".join(tsv) + "\n")

    ds = ExperienceReplayDataset(
        feedback_manifest_path=manifest,
        anchor_dir=anchor_dir,
        processor=None,
    )

    sampler = ReplayBatchSampler(
        ds.feedback_indices,
        ds.anchor_indices,
        batch_size=4,
        replay_ratio=0.5,
        shuffle=True,
        seed=999,
    )

    # 1. Determinism check with seed=999
    epoch1 = list(sampler)
    epoch2 = list(sampler)
    assert epoch1 == epoch2, "Expected identical batches across epochs when seed is fixed"

    # 2. Multi-epoch iteration check
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
    for epoch in range(3):
        epoch_batches = list(loader)
        assert len(epoch_batches) == len(sampler)
        for b in epoch_batches:
            assert len(b["sample_id"]) == 4
            fb_count = sum(1 for src in b["source"] if src == "feedback")
            an_count = sum(1 for src in b["source"] if src == "anchor")
            assert fb_count == 2
            assert an_count == 2

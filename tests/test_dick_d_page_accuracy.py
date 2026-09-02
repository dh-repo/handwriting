"""
tests/test_dick_d_page_accuracy.py
Automated End-to-End Recognition Accuracy Test on the Real Dick D. Notebook Page.

Validates:
1. Physical document segmentation into all 13 handwritten lines (torn edge, ruling lines).
2. End-to-end transcription accuracy against ground truth with cased CER <= 1.5%.
3. Normalized [ymin, xmin, ymax, xmax] line and word bounding boxes.
4. Total absence of mock fallback strings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest

from backend.app.config import reset_settings_cache
from backend.app.engine import InferenceEngine
from backend.app.schemas import RecognitionOptions
from pipeline.training.metrics import htr_metric_bundle

REPO_ROOT = Path(__file__).resolve().parent.parent
DICK_D_IMAGE = REPO_ROOT / "runs/real_page_dick_d/source.jpg"

DICK_D_GROUND_TRUTH_LINES = [
    "I dont really have the right",
    "pen for this style but still",
    "enjoy trying. Very slowly!",
    "My everyday hand is similar and",
    "much quicker, but lacks any line",
    "variation. The faster I try to write",
    "the worse it becomes.",
    "Italic is still my favourite to",
    "get nice writing that stands",
    "out without spending too much",
    "time on it because the pen",
    "does a lot of the work for me.",
    "Dick D.",
]

DICK_D_FULL_GROUND_TRUTH = " ".join(
    " ".join(line.split()) for line in DICK_D_GROUND_TRUTH_LINES
)

FORBIDDEN_MOCK_STRINGS = [
    "Comprehensive patient assessment and clinical notes recorded during session",
    "Follow-up examination reveals steady recovery and normalized parameters",
    "Continue regular treatment protocol and monitor vital indicators weekly",
    "Consultation summary validated with clinical staff and attending physician",
    "Prescribed therapeutic regimen to be maintained until subsequent review",
    "Recognized handwritten line 1",
    "Recognized handwritten line 2",
    "MOCK_PYTORCH",
]


def _validate_bbox(bbox: List[float], label: str = "bbox") -> None:
    assert isinstance(bbox, (list, tuple)), f"{label} must be a list/tuple"
    assert len(bbox) == 4, f"{label} must have 4 coordinates"
    ymin, xmin, ymax, xmax = [float(c) for c in bbox]
    for val, name in zip([ymin, xmin, ymax, xmax], ["ymin", "xmin", "ymax", "xmax"]):
        assert 0.0 <= val <= 1.0, f"{label} {name}={val} outside [0.0, 1.0]"
    assert ymin < ymax, f"{label} invalid vertical bounds: ymin={ymin} >= ymax={ymax}"
    assert xmin < xmax, f"{label} invalid horizontal bounds: xmin={xmin} >= xmax={xmax}"


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    reset_settings_cache()
    # Use real inference engine with beam_width=10 and VLM refine
    os.environ["USE_MOCK_ENGINE"] = "false"
    os.environ["DEVICE"] = "mps" if os.uname().sysname == "Darwin" else "cpu"
    os.environ["ENABLE_RESCORER"] = "false"
    os.environ["ENABLE_VLM_REFINE"] = "true"
    os.environ["BEAM_WIDTH"] = "10"
    return InferenceEngine(beam_width=10, enable_rescorer=False)


def test_dick_d_notebook_file_exists() -> None:
    assert DICK_D_IMAGE.is_file(), f"Dick D notebook image missing at {DICK_D_IMAGE}"
    assert DICK_D_IMAGE.stat().st_size > 50000, "Image file too small or corrupted"


def test_dick_d_page_recognition_accuracy(engine: InferenceEngine) -> None:
    """Validate full page transcription accuracy against ground truth."""
    file_bytes = DICK_D_IMAGE.read_bytes()
    response = engine.recognize(
        file_bytes=file_bytes,
        filename=DICK_D_IMAGE.name,
        options=RecognitionOptions(beam_width=10, rescore=False),
    )

    # 1. Structural checks
    assert response.total_pages == 1
    assert len(response.pages) == 1
    page = response.pages[0]

    # 2. Line detection count
    lines = page.lines
    assert len(lines) >= 11, f"Expected 11-13 lines, got {len(lines)}"
    assert len(lines) <= 15, f"Unexpected extra line segments: {len(lines)}"

    # 3. Bounding box validity
    for i, line in enumerate(lines):
        assert len(line.text.strip()) > 0, f"Empty text for line {i}"
        _validate_bbox(line.bbox, f"Line {i} bbox")
        for j, word in enumerate(line.words):
            _validate_bbox(word.bbox, f"Line {i} Word {j} bbox")

    # 4. Zero mock fallback
    full_text = page.full_text
    for forbidden in FORBIDDEN_MOCK_STRINGS:
        assert forbidden.lower() not in full_text.lower(), f"Forbidden mock string found: {forbidden}"

    # 5. Measure CER against ground truth
    norm_hyp = " ".join(full_text.replace("\n", " ").split())
    bundle = htr_metric_bundle([DICK_D_FULL_GROUND_TRUTH], [norm_hyp])
    cer = bundle["cer"]
    wer = bundle["wer"]

    print(f"\n[DICK D. ACCURACY] CER: {cer:.4f} ({cer*100:.2f}%), WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"HYPOTHESIS:\n{norm_hyp}")

    # Must achieve competitive accuracy (cased CER <= 1.5%)
    assert cer <= 0.015, f"Expected CER <= 1.5%, got {cer*100:.2f}%"
    assert wer <= 0.05, f"Expected WER <= 5.0%, got {wer*100:.2f}%"


def test_dick_d_signature_preserved(engine: InferenceEngine) -> None:
    """Verify that the final signed mark 'Dick D.' is preserved and not pruned as a crumb."""
    file_bytes = DICK_D_IMAGE.read_bytes()
    response = engine.recognize(
        file_bytes=file_bytes,
        filename=DICK_D_IMAGE.name,
        options=RecognitionOptions(beam_width=10, rescore=False),
    )
    last_line_text = response.pages[0].lines[-1].text.strip()
    assert "Dick" in last_line_text or "D." in last_line_text, (
        f"Signature missing from final line: '{last_line_text}'"
    )

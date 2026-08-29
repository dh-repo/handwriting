"""
tests/e2e/utils/assertion_helpers.py
Authoritative Assertion Helpers, Metrics, and Invariant Validators for Opaque-Box E2E Testing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def compute_levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """
    Compute Levenshtein edit distance (insertions, deletions, substitutions) between two sequences.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1],  # Substitution
                )

    return dp[m][n]


def compute_cer(hypothesis: str, reference: str) -> float:
    """
    Compute Character Error Rate (CER) between hypothesis and reference strings.
    CER = Levenshtein_distance(hyp, ref) / max(len(ref), 1)
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    dist = compute_levenshtein_distance(list(hypothesis), list(reference))
    return float(dist / len(reference))


def compute_wer(hypothesis: str, reference: str) -> float:
    """
    Compute Word Error Rate (WER) between hypothesis and reference strings.
    WER = Levenshtein_distance(hyp_words, ref_words) / max(len(ref_words), 1)
    """
    hyp_words = hypothesis.strip().split()
    ref_words = reference.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = compute_levenshtein_distance(hyp_words, ref_words)
    return float(dist / len(ref_words))


def assert_valid_bbox(
    bbox: List[float] | Tuple[float, ...] | Sequence[float],
    name: str = "Bounding box",
) -> None:
    """
    Validate that bbox is [ymin, xmin, ymax, xmax] with coordinates in [0.0, 1.0],
    satisfying ymin < ymax and xmin < xmax.
    """
    assert isinstance(bbox, (list, tuple)), f"{name} must be a list/tuple, got {type(bbox)}"
    assert len(bbox) == 4, f"{name} must contain exactly 4 coordinates, got {len(bbox)}: {bbox}"

    ymin, xmin, ymax, xmax = bbox
    for val, coord_name in zip(bbox, ["ymin", "xmin", "ymax", "xmax"]):
        assert not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))), (
            f"{name} {coord_name}={val} contains NaN/Inf"
        )
        assert 0.0 <= val <= 1.0, f"{name} {coord_name}={val} is outside [0.0, 1.0]"

    assert ymin < ymax, f"{name} invalid vertical span: ymin ({ymin}) >= ymax ({ymax})"
    assert xmin < xmax, f"{name} invalid horizontal span: xmin ({xmin}) >= xmax ({xmax})"


def assert_bbox_iou(
    box1: List[float] | Tuple[float, ...] | Sequence[float],
    box2: List[float] | Tuple[float, ...] | Sequence[float],
    min_iou: float = 0.5,
) -> float:
    """Compute 2D Intersection-over-Union (IoU) and assert it meets min_iou threshold."""
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    inter_ymin = max(y1_min, y2_min)
    inter_xmin = max(x1_min, x2_min)
    inter_ymax = min(y1_max, y2_max)
    inter_xmax = min(x1_max, x2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (y1_max - y1_min) * (x1_max - x1_min))
    area2 = max(0.0, (y2_max - y2_min) * (x2_max - x2_min))
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    assert iou >= min_iou, f"Bounding box IoU {iou:.4f} is below minimum threshold {min_iou:.4f}"
    return iou


def assert_cer_below(
    hypothesis: str,
    reference: str,
    max_cer: float = 0.15,
) -> float:
    """Compute Character Error Rate (CER) and assert it is below max_cer."""
    cer = compute_cer(hypothesis, reference)
    assert cer <= max_cer, (
        f"Observed CER {cer:.4f} exceeds maximum threshold {max_cer:.4f}.\n"
        f"Hypothesis: {hypothesis!r}\n"
        f"Reference: {reference!r}"
    )
    return cer


def assert_wer_below(
    hypothesis: str,
    reference: str,
    max_wer: float = 0.25,
) -> float:
    """Compute Word Error Rate (WER) and assert it is below max_wer."""
    wer = compute_wer(hypothesis, reference)
    assert wer <= max_wer, (
        f"Observed WER {wer:.4f} exceeds maximum threshold {max_wer:.4f}.\n"
        f"Hypothesis: {hypothesis!r}\n"
        f"Reference: {reference!r}"
    )
    return wer


def assert_valid_recognition_response(
    response: Dict[str, Any] | Any,
) -> None:
    """Validate that recognition response adheres strictly to PROJECT.md interface contract."""
    data = response.model_dump() if hasattr(response, "model_dump") else response

    assert "document_id" in data and isinstance(data["document_id"], str) and data["document_id"], "Missing or empty document_id"
    assert "filename" in data and isinstance(data["filename"], str), "Missing filename"
    assert "total_pages" in data and isinstance(data["total_pages"], int) and data["total_pages"] >= 1, "total_pages must be >= 1"
    assert "pages" in data and isinstance(data["pages"], list) and len(data["pages"]) >= 1, "pages list must contain at least 1 page"
    assert "processing_time_ms" in data and data["processing_time_ms"] >= 0.0, "processing_time_ms must be >= 0.0"

    for page in data["pages"]:
        assert page["page_number"] >= 1, f"Invalid page_number {page['page_number']}"
        assert page["width"] > 0 and page["height"] > 0, f"Invalid page dimensions: {page['width']}x{page['height']}"
        assert isinstance(page["full_text"], str), "full_text must be a string"
        assert 0.0 <= page["mean_confidence"] <= 1.0, f"mean_confidence {page['mean_confidence']} outside [0.0, 1.0]"

        for line in page.get("lines", []):
            assert "line_id" in line and isinstance(line["line_id"], str), "Missing line_id"
            assert "text" in line and isinstance(line["text"], str), "Missing line text"
            assert 0.0 <= line["confidence"] <= 1.0, f"line confidence {line['confidence']} outside [0.0, 1.0]"
            assert_valid_bbox(line["bbox"], f"Line {line['line_id']} bbox")

            for word in line.get("words", []):
                assert "word_id" in word and isinstance(word["word_id"], str), "Missing word_id"
                assert "text" in word and isinstance(word["text"], str), "Missing word text"
                assert 0.0 <= word["confidence"] <= 1.0, f"word confidence {word['confidence']} outside [0.0, 1.0]"
                assert_valid_bbox(word["bbox"], f"Word {word['word_id']} bbox")


class AssertionHelpers:
    """Namespace container for custom assertions and metrics."""
    assert_valid_bbox = staticmethod(assert_valid_bbox)
    assert_bbox_iou = staticmethod(assert_bbox_iou)
    assert_cer_below = staticmethod(assert_cer_below)
    assert_wer_below = staticmethod(assert_wer_below)
    assert_valid_recognition_response = staticmethod(assert_valid_recognition_response)
    compute_cer = staticmethod(compute_cer)
    compute_wer = staticmethod(compute_wer)
    compute_levenshtein_distance = staticmethod(compute_levenshtein_distance)

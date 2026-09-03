"""
backend/tests/test_feedback_adversarial.py
Adversarial stress-test harness for POST /v1/feedback.
Challenger 2 Milestone 1 empirical verification.

Tests:
1. Extreme strings (10,000+ chars, unicode diacritics, Cyrillic/Arabic, emojis, newline injections).
2. Malformed base64 (invalid padding, non-base64 characters, truncated base64, fake PNG header with corrupt body, non-image data URL).
3. Boundary bounding boxes ([0,0,1,1], degenerate bboxes, inverted bboxes, out-of-bounds, subnormal floats, NaN/inf).
4. Missing and null fields (null document_id, empty string line_id, whitespace-only ids, boundary confidences).
5. Clean error handling (strictly 422 or 400, no 500, no crashes, no hangs).
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Generator

import numpy as np
from PIL import Image, ImageDraw
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, get_settings, reset_settings_cache
from backend.app.engine import reset_engine
from backend.app.main import create_app


@pytest.fixture
def adv_env(tmp_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Provide isolated environment for adversarial feedback testing."""
    fb_dir = tmp_path / "adv_feedback"
    manifest_path = fb_dir / "manifest.jsonl"
    crops_dir = fb_dir / "crops"

    fb_dir.mkdir(parents=True, exist_ok=True)

    test_settings = Settings(
        FEEDBACK_DIR=str(fb_dir),
        FEEDBACK_MANIFEST_PATH=str(manifest_path),
        FEEDBACK_CROPS_DIR=str(crops_dir),
        CONFUSION_LEARNING_RATE=0.10,
        USE_MOCK_ENGINE=True,
    )

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings

    yield {
        "client": TestClient(app),
        "fb_dir": fb_dir,
        "manifest_path": manifest_path,
        "crops_dir": crops_dir,
        "settings": test_settings,
    }

    app.dependency_overrides.clear()
    reset_settings_cache()
    reset_engine()


@pytest.fixture
def valid_png_b64() -> str:
    """Generate a minimal valid 10x10 PNG in raw base64."""
    img = Image.new("RGB", (10, 10), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def realistic_crop_png_b64() -> str:
    """Generate a realistic line crop PNG (400x80) with synthetic handwriting text."""
    img = Image.new("RGB", (400, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "Amoxicillin 500mg capsules PO TID", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ===========================================================================
# 1. EXTREME STRINGS & UNICODE ADVERSARIAL SUITE
# ===========================================================================

def test_extreme_large_string_payload(adv_env: Dict[str, Any], valid_png_b64: str) -> None:
    """Stress test with 10,000+ character strings in prediction and correction (rejected with 422)."""
    client: TestClient = adv_env["client"]
    manifest_path: Path = adv_env["manifest_path"]

    huge_prediction = "cydindamycfn " * 1000  # ~13,000 characters
    huge_correction = "clindamycin " * 1000   # ~12,000 characters

    payload = {
        "document_id": "doc_stress_large_str",
        "line_id": "line_huge_01",
        "original_prediction": huge_prediction,
        "operator_correction": huge_correction,
        "confidence": 0.42,
        "line_crop_base64": valid_png_b64,
    }

    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422, f"Expected 422 for payload exceeding max_length=500, got {resp.status_code}: {resp.text}"

    # Verify manifest was NOT written
    if manifest_path.exists():
        lines = [line for line in manifest_path.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
        assert len(lines) == 0


def test_unicode_zalgo_and_combining_diacritics(adv_env: Dict[str, Any]) -> None:
    """Adversarial Unicode strings with heavy combining diacritical marks (Zalgo text)."""
    client: TestClient = adv_env["client"]
    manifest_path: Path = adv_env["manifest_path"]

    # Complex combining accents stack
    zalgo_pred = "c\u0300\u0301\u0302\u0303\u0304\u0305\u0306\u0307\u0308y\u0310\u0311\u0312\u0313d"
    zalgo_corr = "c\u0314\u0315\u0316\u0317\u0318l\u0319\u031ai\u031b\u031cn"

    payload = {
        "document_id": "doc_zalgo_01",
        "line_id": "line_zalgo",
        "original_prediction": zalgo_pred,
        "operator_correction": zalgo_corr,
    }

    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200, f"Failed on Zalgo unicode: {resp.text}"
    record = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert record["original_prediction"] == zalgo_pred
    assert record["operator_correction"] == zalgo_corr


def test_unicode_multilingual_scripts_and_bidi(adv_env: Dict[str, Any]) -> None:
    """Adversarial Cyrillic, Arabic (RTL), Hebrew, CJK, and mixed BiDi strings."""
    client: TestClient = adv_env["client"]
    manifest_path: Path = adv_env["manifest_path"]

    test_cases = [
        ("Cyrillic", "Амоксициллин 500мг", "Амоксициллин 500 мг капсулы"),
        ("Arabic_RTL", "وصفة طبية: باراسيتامول ٥٠٠ ملغ", "وصفة طبية: باراسيتامول 500 ملغ"),
        ("Hebrew_RTL", "מרשם רופא: אקמול 500 מ״ג", "מרשם רופא: אקמול 500 מ\"ג"),
        ("CJK_Japanese", "抗生物質 アモキシシリン 250mg 毎食後", "抗生物質 アモキシシリン 250mg 毎食後服用"),
        ("Mixed_BiDi", "Rx #1234: دواء 50mg tablets [BID]", "Rx #1234: دواء 50mg tablets [TID]"),
    ]

    for idx, (label, orig, corr) in enumerate(test_cases):
        payload = {
            "document_id": f"doc_bidi_{idx}",
            "line_id": f"line_bidi_{label}",
            "original_prediction": orig,
            "operator_correction": corr,
        }
        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200, f"Failed on {label}: {resp.text}"

    # Verify all records persisted cleanly
    lines = [l for l in manifest_path.read_text(encoding="utf-8").split("\n") if l.strip()]
    assert len(lines) == len(test_cases)
    for line, (label, orig, corr) in zip(lines, test_cases):
        rec = json.loads(line)
        assert rec["original_prediction"] == orig
        assert rec["operator_correction"] == corr


def test_unicode_emojis_and_surrogate_pairs(adv_env: Dict[str, Any]) -> None:
    """Adversarial emojis, multi-character clusters, ZWJ sequences, and skin tones."""
    client: TestClient = adv_env["client"]
    manifest_path: Path = adv_env["manifest_path"]

    emoji_pred = "Rx: 💊💊 2x/day 👨‍⚕️ 🩺"
    emoji_corr = "Rx: 💊 1x/day 👩🏽‍⚕️ 🏥"

    payload = {
        "document_id": "doc_emoji_01",
        "line_id": "line_emoji",
        "original_prediction": emoji_pred,
        "operator_correction": emoji_corr,
    }

    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200, f"Failed on emoji payload: {resp.text}"
    rec = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert rec["original_prediction"] == emoji_pred
    assert rec["operator_correction"] == emoji_corr


def test_newline_and_injection_characters_in_text(adv_env: Dict[str, Any]) -> None:
    """
    Adversarial newline injections, carriage returns, tabs, null bytes, and JSON injection attacks.
    CRITICAL: Manifest is JSONL (newline-delimited). Injected newlines must NOT corrupt the manifest
    into multiple invalid lines or inject rogue JSON records.
    """
    client: TestClient = adv_env["client"]
    manifest_path: Path = adv_env["manifest_path"]

    injection_pred = "Line 1\nLine 2\r\nLine 3\rLine 4\tTabbed\x00NullByte"
    injection_corr = 'Line 1\n{"hacked": true, "role": "admin"}\nLine 3'

    payload = {
        "document_id": "doc_inject_nl",
        "line_id": "line_inject_nl",
        "original_prediction": injection_pred,
        "operator_correction": injection_corr,
    }

    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200, f"Failed on newline injection: {resp.text}"

    # Verify manifest JSONL integrity: must be EXACTLY 1 physical line
    content = manifest_path.read_text(encoding="utf-8")
    raw_lines = [l for l in content.split("\n") if l.strip()]
    assert len(raw_lines) == 1, f"Manifest corrupted into {len(raw_lines)} lines! Content: {content}"

    rec = json.loads(raw_lines[0])
    assert rec["original_prediction"] == injection_pred
    assert rec["operator_correction"] == injection_corr
    assert "hacked" not in rec  # JSON injection did not alter top-level keys


def test_empty_string_prediction_and_correction(adv_env: Dict[str, Any]) -> None:
    """Empty strings in predictions (insertion or deletion corrections) should be handled safely."""
    client: TestClient = adv_env["client"]
    manifest_path: Path = adv_env["manifest_path"]

    # Deletion: prediction had text, operator cleared it
    resp_del = client.post("/v1/feedback", json={
        "document_id": "doc_del",
        "line_id": "line_del",
        "original_prediction": "spurious noise mark",
        "operator_correction": "",
    })
    assert resp_del.status_code == 200, f"Expected 200 on deletion correction, got {resp_del.status_code}: {resp_del.text}"

    # Insertion: prediction was empty, operator transcribed handwritten line
    resp_ins = client.post("/v1/feedback", json={
        "document_id": "doc_ins",
        "line_id": "line_ins",
        "original_prediction": "",
        "operator_correction": "Aspirin 81mg",
    })
    assert resp_ins.status_code == 200, f"Expected 200 on insertion correction, got {resp_ins.status_code}: {resp_ins.text}"


# ===========================================================================
# 2. MALFORMED & ADVERSARIAL BASE64 SUITE
# ===========================================================================

@pytest.mark.parametrize(
    "label,bad_b64",
    [
        ("invalid_padding_1", "YWJj="),
        ("invalid_padding_2", "YWJ==="),
        ("only_padding", "===="),
        ("incomplete_octet", "a"),
        ("incomplete_octet_3", "abc"),
        ("non_b64_special_chars", "YWJj!@#$%^&*()_+"),
        ("non_b64_whitespace_embedded", "YW Jj"),
        ("non_b64_unicode_embedded", "YWJj🚀"),
        ("control_chars_embedded", "YWJj\x00\x08"),
    ],
)
def test_malformed_base64_encoding(
    adv_env: Dict[str, Any],
    label: str,
    bad_b64: str,
) -> None:
    """Base64 decoding syntax violations must return 422 cleanly without 500."""
    client: TestClient = adv_env["client"]
    payload = {
        "document_id": "doc_bad_b64",
        "line_id": "line_bad_b64",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "line_crop_base64": bad_b64,
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422, f"Expected 422 on {label}, got {resp.status_code}: {resp.text}"
    assert "detail" in resp.json()


def test_truncated_base64_valid_image(
    adv_env: Dict[str, Any],
    realistic_crop_png_b64: str,
) -> None:
    """
    Truncated realistic line crop image data must fail decompression/parsing
    and return 422 cleanly without crashing or returning 500.
    """
    client: TestClient = adv_env["client"]

    # Test truncations of a realistic 400x80 line crop at various percentages
    for pct in [0.25, 0.50, 0.75, 0.90, 0.95]:
        cutoff = max(4, int(len(realistic_crop_png_b64) * pct))
        # Sliced cleanly to multiple of 4 to isolate image corruption from base64 syntax
        truncated = realistic_crop_png_b64[: (cutoff // 4) * 4]
        payload = {
            "document_id": "doc_trunc_b64",
            "line_id": "line_trunc",
            "original_prediction": "pred",
            "operator_correction": "corr",
            "line_crop_base64": truncated,
        }
        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 422, (
            f"Expected 422 on truncated realistic image at pct={pct}, got {resp.status_code}: {resp.text}"
        )


def test_fake_png_header_corrupt_body(adv_env: Dict[str, Any]) -> None:
    """PNG magic bytes followed by garbage payload must return 422 cleanly."""
    client: TestClient = adv_env["client"]

    # 8-byte standard PNG signature followed by random garbage bytes
    png_signature = b"\x89PNG\r\n\x1a\n"
    corrupt_png_bytes = png_signature + b"\x00\x00\x00\rIHDR\xff\xff\xff\xff\x00\x00GARBAGE_DATA"
    b64_corrupt = base64.b64encode(corrupt_png_bytes).decode("utf-8")

    payload = {
        "document_id": "doc_fake_png",
        "line_id": "line_fake_png",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "line_crop_base64": b64_corrupt,
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422, f"Expected 422 on corrupt PNG body, got {resp.status_code}: {resp.text}"


@pytest.mark.parametrize(
    "mime_type,raw_content",
    [
        ("text/html", b"<html><body><script>alert('xss')</script></body></html>"),
        ("text/plain", b"Hello, this is just a plain text document."),
        ("application/pdf", b"%PDF-1.4\n%...\n%%EOF"),
        ("application/json", b'{"key": "value"}'),
        ("image/svg+xml", b'<svg xmlns="http://www.w3.org/2000/svg"><circle r="50"/></svg>'),
    ],
)
def test_non_image_data_urls(
    adv_env: Dict[str, Any],
    mime_type: str,
    raw_content: bytes,
) -> None:
    """Non-raster-image Data URLs (HTML, PDF, text, JSON, SVG) must be rejected with 422."""
    client: TestClient = adv_env["client"]
    b64_encoded = base64.b64encode(raw_content).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_encoded}"

    payload = {
        "document_id": "doc_non_img_url",
        "line_id": "line_non_img",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "line_crop_base64": data_url,
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422, f"Expected 422 for MIME {mime_type}, got {resp.status_code}: {resp.text}"


@pytest.mark.parametrize(
    "malformed_data_url",
    [
        "data:;base64,",
        "data:image/png;base64",
        "data:,",
        "data:image/png,",
        "data:text/plain;charset=utf-8,NotBase64Content",
    ],
)
def test_malformed_data_url_headers(
    adv_env: Dict[str, Any],
    malformed_data_url: str,
) -> None:
    """Malformed Data URL schemes must return 422 cleanly."""
    client: TestClient = adv_env["client"]
    payload = {
        "document_id": "doc_malformed_url",
        "line_id": "line_url",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "line_crop_base64": malformed_data_url,
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422, f"Expected 422 for '{malformed_data_url}', got {resp.status_code}: {resp.text}"


# ===========================================================================
# 3. BOUNDARY & ADVERSARIAL BOUNDING BOXES
# ===========================================================================

def test_bbox_exact_boundaries(adv_env: Dict[str, Any]) -> None:
    """Bbox with exact boundary coordinates [0.0, 0.0, 1.0, 1.0] must be accepted with 200 OK."""
    client: TestClient = adv_env["client"]
    payload = {
        "document_id": "doc_bbox_boundary",
        "line_id": "line_bbox_01",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "bbox": [0.0, 0.0, 1.0, 1.0],
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200, f"Expected 200 for [0, 0, 1, 1], got {resp.status_code}: {resp.text}"
    assert resp.json()["status"] == "persisted"


@pytest.mark.parametrize(
    "label,degenerate_bbox",
    [
        ("zero_height_mid", [0.5, 0.1, 0.5, 0.9]),
        ("zero_height_top", [0.0, 0.0, 0.0, 1.0]),
        ("zero_height_bottom", [1.0, 0.0, 1.0, 1.0]),
        ("zero_width_mid", [0.1, 0.5, 0.9, 0.5]),
        ("zero_width_left", [0.0, 0.0, 1.0, 0.0]),
        ("zero_width_right", [0.0, 1.0, 1.0, 1.0]),
        ("point_bbox", [0.5, 0.5, 0.5, 0.5]),
    ],
)
def test_bbox_degenerate_zero_area(
    adv_env: Dict[str, Any],
    label: str,
    degenerate_bbox: list,
) -> None:
    """Degenerate zero-height (ymin == ymax) or zero-width (xmin == xmax) bboxes must return 422."""
    client: TestClient = adv_env["client"]
    payload = {
        "document_id": "doc_degen_bbox",
        "line_id": "line_degen",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "bbox": degenerate_bbox,
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422, f"Expected 422 on {label} {degenerate_bbox}, got {resp.status_code}: {resp.text}"


@pytest.mark.parametrize(
    "label,inverted_bbox",
    [
        ("inverted_vertical", [0.9, 0.1, 0.2, 0.9]),
        ("inverted_horizontal", [0.1, 0.9, 0.8, 0.2]),
        ("inverted_both", [0.8, 0.8, 0.2, 0.2]),
    ],
)
def test_bbox_inverted_coordinates(
    adv_env: Dict[str, Any],
    label: str,
    inverted_bbox: list,
) -> None:
    """Inverted bounding box spans must return 422."""
    client: TestClient = adv_env["client"]
    payload = {
        "document_id": "doc_inv_bbox",
        "line_id": "line_inv",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "bbox": inverted_bbox,
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422, f"Expected 422 on {label}, got {resp.status_code}: {resp.text}"


@pytest.mark.parametrize(
    "label,oob_bbox",
    [
        ("ymin_negative", [-0.01, 0.0, 1.0, 1.0]),
        ("xmin_negative", [0.0, -0.01, 1.0, 1.0]),
        ("ymax_exceed_1", [0.0, 0.0, 1.01, 1.0]),
        ("xmax_exceed_1", [0.0, 0.0, 1.0, 1.01]),
        ("ymin_epsilon_neg", [-1e-7, 0.0, 1.0, 1.0]),
        ("xmax_epsilon_pos", [0.0, 0.0, 1.0, 1.0000001]),
        ("wildly_oob", [-100.0, -50.0, 200.0, 500.0]),
    ],
)
def test_bbox_out_of_bounds_coordinates(
    adv_env: Dict[str, Any],
    label: str,
    oob_bbox: list,
) -> None:
    """Out-of-bounds coordinates (< 0.0 or > 1.0) must return 422."""
    client: TestClient = adv_env["client"]
    payload = {
        "document_id": "doc_oob_bbox",
        "line_id": "line_oob",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "bbox": oob_bbox,
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422, f"Expected 422 on {label}, got {resp.status_code}: {resp.text}"


def test_bbox_subnormal_positive_floats(adv_env: Dict[str, Any]) -> None:
    """Extremely small subnormal positive float coordinates (> 0.0) must be handled safely."""
    client: TestClient = adv_env["client"]
    subnormal_val = 1e-300
    payload = {
        "document_id": "doc_subnormal",
        "line_id": "line_subnormal",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "bbox": [subnormal_val, subnormal_val, 0.95, 0.95],
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200, f"Expected 200 on positive subnormal float, got {resp.status_code}: {resp.text}"


def test_bbox_negative_subnormal_floats(adv_env: Dict[str, Any]) -> None:
    """Negative subnormal float coordinates (< 0.0) must trigger 422."""
    client: TestClient = adv_env["client"]
    neg_subnormal = -1e-300
    payload = {
        "document_id": "doc_neg_subnormal",
        "line_id": "line_neg_subnormal",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "bbox": [neg_subnormal, 0.0, 1.0, 1.0],
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422, f"Expected 422 on negative subnormal float, got {resp.status_code}: {resp.text}"


@pytest.mark.parametrize(
    "label,nan_inf_bbox",
    [
        ("nan_in_ymin", [float("nan"), 0.0, 1.0, 1.0]),
        ("nan_in_xmax", [0.0, 0.0, 1.0, float("nan")]),
        ("inf_in_ymax", [0.0, 0.0, float("inf"), 1.0]),
        ("neg_inf_in_ymin", [-float("inf"), 0.0, 1.0, 1.0]),
        ("string_nan", ["nan", 0.0, 1.0, 1.0]),
        ("string_inf", [0.0, "inf", 1.0, 1.0]),
    ],
)
def test_bbox_nan_and_infinity_rejection(
    adv_env: Dict[str, Any],
    label: str,
    nan_inf_bbox: list,
) -> None:
    """NaN and +/- Infinity in bounding box coordinates must return 422 without crashing."""
    client: TestClient = adv_env["client"]
    payload = {
        "document_id": "doc_nan_bbox",
        "line_id": "line_nan",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "bbox": nan_inf_bbox,
    }
    # Use allow_nan=True JSON encoding to send unquoted NaN/Inf over HTTP wire
    raw_json = json.dumps(payload, allow_nan=True).encode("utf-8")
    resp = client.post("/v1/feedback", content=raw_json, headers={"content-type": "application/json"})
    assert resp.status_code == 422, f"Expected 422 on {label}, got {resp.status_code}: {resp.text}"


@pytest.mark.parametrize(
    "label,bad_bbox_type",
    [
        ("string_bbox", "0.1, 0.2, 0.8, 0.9"),
        ("dict_bbox", {"ymin": 0.1, "xmin": 0.2, "ymax": 0.8, "xmax": 0.9}),
        ("int_bbox", 12345),
        ("bbox_3_coords", [0.1, 0.2, 0.8]),
        ("bbox_5_coords", [0.1, 0.2, 0.8, 0.9, 1.0]),
        ("bbox_nested_lists", [[0.1], [0.2], [0.8], [0.9]]),
    ],
)
def test_bbox_structural_type_errors(
    adv_env: Dict[str, Any],
    label: str,
    bad_bbox_type: Any,
) -> None:
    """Non-conforming bbox types and incorrect element lengths must return 422."""
    client: TestClient = adv_env["client"]
    payload = {
        "document_id": "doc_type_bbox",
        "line_id": "line_type",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "bbox": bad_bbox_type,
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422, f"Expected 422 on {label}, got {resp.status_code}: {resp.text}"


# ===========================================================================
# 4. MISSING, NULL, AND WHITESPACE FIELDS SUITE
# ===========================================================================

@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("document_id", None),
        ("document_id", ""),
        ("document_id", "   "),
        ("document_id", "\t\n  \r"),
        ("line_id", None),
        ("line_id", ""),
        ("line_id", "   "),
        ("line_id", "\t"),
        ("original_prediction", None),
        ("operator_correction", None),
        ("confidence", None),
        ("page_number", None),
    ],
)
def test_missing_and_whitespace_ids_rejection(
    adv_env: Dict[str, Any],
    field_name: str,
    bad_value: Any,
) -> None:
    """Null, empty string, or whitespace-only IDs must return 422."""
    client: TestClient = adv_env["client"]
    base_payload = {
        "document_id": "doc_ok",
        "line_id": "line_ok",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "confidence": 0.85,
    }
    base_payload[field_name] = bad_value

    resp = client.post("/v1/feedback", json=base_payload)
    assert resp.status_code == 422, f"Expected 422 on {field_name}={repr(bad_value)}, got {resp.status_code}: {resp.text}"
    assert "detail" in resp.json()


@pytest.mark.parametrize(
    "conf_val,expected_status",
    [
        (0.0, 200),
        (1.0, 200),
        (0.5, 200),
        (-0.0001, 422),
        (1.0001, 422),
        (float("nan"), 422),
        (float("inf"), 422),
        (-float("inf"), 422),
        ("invalid_str", 422),
    ],
)
def test_confidence_boundary_and_adversarial_values(
    adv_env: Dict[str, Any],
    conf_val: Any,
    expected_status: int,
) -> None:
    """Confidence score boundary values [0.0, 1.0] vs NaN, Inf, and out-of-range values."""
    client: TestClient = adv_env["client"]
    payload = {
        "document_id": "doc_conf_test",
        "line_id": "line_conf",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "confidence": conf_val,
    }
    # Encode with allow_nan=True so wire protocol carries IEEE specials directly
    raw_json = json.dumps(payload, allow_nan=True).encode("utf-8")
    resp = client.post("/v1/feedback", content=raw_json, headers={"content-type": "application/json"})
    assert resp.status_code == expected_status, (
        f"Expected {expected_status} for confidence={conf_val}, got {resp.status_code}: {resp.text}"
    )


def test_extra_arbitrary_keys_ignored_without_crash(adv_env: Dict[str, Any]) -> None:
    """Payload with extra unexpected JSON fields should be cleanly ignored without crash or 500."""
    client: TestClient = adv_env["client"]
    payload = {
        "document_id": "doc_extra_keys",
        "line_id": "line_extra",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "rogue_field_1": "attack",
        "rogue_nested_field": {"eval": "__import__('os').system('id')"},
        "sql_injection": "'; DROP TABLE feedback; --",
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200, f"Expected 200 with ignored extra fields, got {resp.status_code}: {resp.text}"
    assert resp.json()["status"] == "persisted"


def test_decompression_bomb_dos_protection(adv_env: Dict[str, Any]) -> None:
    """Oversized/decompression bomb images must be rejected with 422 rather than causing OOM or 500."""
    client: TestClient = adv_env["client"]
    # Temporarily set max pixels low to trigger decompression bomb check
    orig_max = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 500
        img = Image.new("RGB", (40, 40), color=(200, 200, 200))  # 1600 pixels > 500
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        bomb_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "document_id": "doc_bomb",
            "line_id": "line_bomb",
            "original_prediction": "pred",
            "operator_correction": "corr",
            "line_crop_base64": bomb_b64,
        }
        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 422, f"Expected 422 on decompression bomb, got {resp.status_code}: {resp.text}"
    finally:
        Image.MAX_IMAGE_PIXELS = orig_max


def test_dynamic_confusion_hook_exception_resilience(adv_env: Dict[str, Any]) -> None:
    """If dynamic confusion adaptation throws an unhandled exception, feedback must still persist."""
    from unittest.mock import MagicMock, patch
    client: TestClient = adv_env["client"]
    manifest_path: Path = adv_env["manifest_path"]

    mock_engine = MagicMock()
    mock_engine.rescorer.confusion_matrix.adapt_from_correction.side_effect = RuntimeError("DP alignment crashed")

    with patch("backend.app.routes.feedback.get_engine", return_value=mock_engine):
        payload = {
            "document_id": "doc_resilience",
            "line_id": "line_resilience",
            "original_prediction": "pred",
            "operator_correction": "corr",
            "sync_confusion_matrix": True,
        }
        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200, f"Expected 200 despite hook failure, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "persisted"
        assert data["confusion_pairs_updated"] == []

        # Confirm manifest still saved the entry
        assert manifest_path.exists()
        rec = json.loads(manifest_path.read_text(encoding="utf-8").strip())
        assert rec["document_id"] == "doc_resilience"


def test_path_traversal_ids_containment(adv_env: Dict[str, Any], valid_png_b64: str) -> None:
    """Document and line IDs containing directory traversal sequences (../..) must not escape crops dir."""
    client: TestClient = adv_env["client"]
    crops_dir: Path = adv_env["crops_dir"]
    fb_dir: Path = adv_env["fb_dir"]

    payload = {
        "document_id": "../../../etc/passwd",
        "line_id": "../../root_secret",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "line_crop_base64": valid_png_b64,
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    # Verify crop was stored ONLY within designated crops_dir
    crop_path = Path(data["crop_path"])
    assert crop_path.parent == crops_dir
    assert crop_path.exists()
    assert not (fb_dir.parent / "root_secret.png").exists()


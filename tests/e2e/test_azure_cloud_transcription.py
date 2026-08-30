"""
End-to-End Cloud Transcription Verification Test Suite for Microsoft Azure Deployment.

Validates:
1. Azure Frontend Health Proxy (/api/health) - status, models, rescorer.
2. Azure Frontend Homepage (/) - HTTP 200, HTML structure, security headers (X-Frame-Options, X-Content-Type-Options).
3. Single-Page Prescription Transcription - uploads sample_prescription.png, validates structured response,
   valid [ymin, xmin, ymax, xmax] line/word bounding boxes, and medical prescription tokens.
4. Clean Cursive Handwriting Transcription - uploads sample_clean_cursive.png, validates line extractions and bboxes.
5. Multi-Page PDF Transcription - uploads sample_multipage.pdf, validates 3 pages and line extractions on all pages.
6. Zero Mock Fallback Enforcement - verifies no mock generator sentences or synthetic markers appear in output.
7. Frontend API Proxy - uploads to /api/recognize and validates proxy response with structured layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import requests

# Live Azure frontend is the only public surface. Inference is proxied internally.
DEFAULT_FRONTEND_URL = "https://ca-frontend-playground.jollysand-1dc47ca9.eastus2.azurecontainerapps.io"

FRONTEND_URL = os.environ.get("AZURE_FRONTEND_URL", DEFAULT_FRONTEND_URL).rstrip("/")
HEALTH_URL = f"{FRONTEND_URL}/api/health"
RECOGNIZE_URL = f"{FRONTEND_URL}/api/recognize"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = REPO_ROOT / "samples"

# Request timeouts (cloud CPU inference for multi-page TrOCR can take 60-300s)
HEALTH_TIMEOUT = 15
INFERENCE_TIMEOUT = 480

# Known synthetic mock strings to disallow in real inference output
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


def pytest_configure(config: pytest.Config) -> None:
    """Register custom azure marker if not already registered."""
    config.addinivalue_line("markers", "azure: Tests against live Microsoft Azure deployment")


def _validate_bbox(bbox: List[float], label: str = "bbox") -> None:
    """Validate normalized [ymin, xmin, ymax, xmax] bounding box in [0.0, 1.0]."""
    assert isinstance(bbox, (list, tuple)), f"{label} must be a list/tuple, got {type(bbox)}"
    assert len(bbox) == 4, f"{label} must have 4 coordinates, got {len(bbox)}: {bbox}"
    ymin, xmin, ymax, xmax = [float(c) for c in bbox]
    for val, name in zip([ymin, xmin, ymax, xmax], ["ymin", "xmin", "ymax", "xmax"]):
        assert 0.0 <= val <= 1.0, f"{label} {name}={val} outside [0.0, 1.0]"
    assert ymin < ymax, f"{label} invalid vertical bounds: ymin={ymin} >= ymax={ymax}"
    assert xmin < xmax, f"{label} invalid horizontal bounds: xmin={xmin} >= xmax={xmax}"


@pytest.mark.azure
def test_azure_backend_health() -> None:
    """Validate Azure backend /v1/health probe returns HTTP 200, healthy status, and active models."""
    url = HEALTH_URL
    response = requests.get(url, timeout=HEALTH_TIMEOUT)
    assert response.status_code == 200, f"Expected HTTP 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data.get("status") == "healthy", f"Expected status 'healthy', got {data.get('status')}"
    assert data.get("rescorer_active") is True, f"Expected rescorer_active True, got {data.get('rescorer_active')}"

    loaded_models = data.get("loaded_models", [])
    assert isinstance(loaded_models, list) and len(loaded_models) > 0, "loaded_models must be a non-empty list"
    assert any("trocr" in model.lower() for model in loaded_models), f"TrOCR model missing from loaded_models: {loaded_models}"


@pytest.mark.azure
def test_azure_frontend_homepage() -> None:
    """Validate Azure frontend serves HTTP 200 HTML with mandatory security headers."""
    response = requests.get(FRONTEND_URL, timeout=HEALTH_TIMEOUT)
    assert response.status_code == 200, f"Expected HTTP 200, got {response.status_code}"

    # Verify HTML structure
    html = response.text.lower()
    assert "<!doctype html>" in html or "<html" in html, "Frontend did not return HTML content"

    # Verify security headers injected by Next.js configuration
    headers = {k.lower(): v for k, v in response.headers.items()}
    assert headers.get("x-frame-options") == "DENY", f"Missing or invalid X-Frame-Options: {headers.get('x-frame-options')}"
    assert headers.get("x-content-type-options") == "nosniff", f"Missing or invalid X-Content-Type-Options: {headers.get('x-content-type-options')}"


@pytest.mark.azure
def test_azure_single_page_prescription_transcription() -> None:
    """Upload sample_prescription.png to live Azure backend /v1/recognize and validate structured response."""
    sample_file = SAMPLES_DIR / "sample_prescription.png"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"

    url = RECOGNIZE_URL
    with open(sample_file, "rb") as f:
        response = requests.post(
            url,
            files={"file": ("sample_prescription.png", f, "image/png")},
            params={"beam_width": 1},
            timeout=INFERENCE_TIMEOUT,
        )

    assert response.status_code == 200, f"Expected HTTP 200, got {response.status_code}: {response.text}"
    data = response.json()

    # Document-level validation
    assert "document_id" in data and data["document_id"], "Missing or empty document_id"
    assert data.get("total_pages") == 1, f"Expected total_pages=1, got {data.get('total_pages')}"
    assert "pages" in data and len(data["pages"]) == 1, "Expected exactly 1 page in pages list"

    page = data["pages"][0]
    assert page["page_number"] == 1
    assert page["width"] > 0 and page["height"] > 0
    assert 0.0 <= page.get("mean_confidence", 0.0) <= 1.0

    # Line and Word validation
    lines = page.get("lines", [])
    assert len(lines) >= 4, f"Expected at least 4 extracted lines, got {len(lines)}"

    for idx, line in enumerate(lines):
        assert "text" in line and len(line["text"].strip()) > 0, f"Line {idx} text is empty"
        assert 0.0 <= line.get("confidence", 0.0) <= 1.0, f"Line {idx} confidence out of range"
        _validate_bbox(line["bbox"], f"Line {idx} bbox")

        words = line.get("words", [])
        assert len(words) > 0, f"Line {idx} must contain at least one word token"
        for w_idx, word in enumerate(words):
            assert "text" in word and len(word["text"]) > 0, f"Word {w_idx} in line {idx} has empty text"
            assert 0.0 <= word.get("confidence", 0.0) <= 1.0
            _validate_bbox(word["bbox"], f"Word {w_idx} in line {idx} bbox")

    # Medical prescription tokens verification
    full_text = page.get("full_text", "").lower()
    rx_keywords = ["amoxicillin", "capsule", "disp", "refill", "sig", "ibuprofen", "take", "tab", "dr"]
    matched_keywords = [kw for kw in rx_keywords if kw in full_text]
    assert len(matched_keywords) >= 3, f"Expected at least 3 prescription keywords in full_text, matched {matched_keywords}. Text: {full_text}"


@pytest.mark.azure
def test_azure_cursive_handwriting_transcription() -> None:
    """Upload sample_clean_cursive.png to live Azure backend and validate line extractions & bounding boxes."""
    sample_file = SAMPLES_DIR / "sample_clean_cursive.png"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"

    url = RECOGNIZE_URL
    with open(sample_file, "rb") as f:
        response = requests.post(
            url,
            files={"file": ("sample_clean_cursive.png", f, "image/png")},
            params={"beam_width": 1},
            timeout=INFERENCE_TIMEOUT,
        )

    assert response.status_code == 200, f"Expected HTTP 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data.get("total_pages") == 1
    page = data["pages"][0]
    lines = page.get("lines", [])
    assert len(lines) >= 3, f"Expected at least 3 lines for clean cursive sample, got {len(lines)}"

    for idx, line in enumerate(lines):
        assert len(line["text"].strip()) > 0
        _validate_bbox(line["bbox"], f"Cursive line {idx} bbox")

    full_text = page.get("full_text", "").lower()
    cursive_keywords = [
        "quick", "brown", "fox", "cursive", "handwriting", "practice",
        "stroke", "curve", "precision", "signed", "steady", "flow"
    ]
    matched = [kw for kw in cursive_keywords if kw in full_text]
    assert len(matched) >= 2, f"Expected at least 2 cursive keywords, matched: {matched}. Text: {full_text}"


@pytest.mark.azure
def test_azure_multipage_pdf_transcription() -> None:
    """Upload sample_multipage.pdf to live Azure backend and validate 3-page transcription & bounding boxes."""
    sample_file = SAMPLES_DIR / "sample_multipage.pdf"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"

    url = RECOGNIZE_URL
    with open(sample_file, "rb") as f:
        response = requests.post(
            url,
            files={"file": ("sample_multipage.pdf", f, "application/pdf")},
            params={"beam_width": 1},
            timeout=INFERENCE_TIMEOUT,
        )

    assert response.status_code == 200, f"Expected HTTP 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data.get("total_pages") == 3, f"Expected total_pages=3, got {data.get('total_pages')}"
    pages = data.get("pages", [])
    assert len(pages) == 3, f"Expected 3 pages in pages list, got {len(pages)}"

    for page_idx, page in enumerate(pages):
        assert page["page_number"] == page_idx + 1
        lines = page.get("lines", [])
        assert len(lines) >= 2, f"Page {page_idx + 1} expected at least 2 lines, got {len(lines)}"
        for l_idx, line in enumerate(lines):
            assert len(line["text"].strip()) > 0
            _validate_bbox(line["bbox"], f"Page {page_idx + 1} line {l_idx} bbox")


@pytest.mark.azure
def test_azure_zero_mock_fallback_enforcement() -> None:
    """Assert zero mock generator sentences or mock indicators appear in live backend responses."""
    sample_file = SAMPLES_DIR / "sample_clean_cursive.png"
    assert sample_file.exists()

    url = RECOGNIZE_URL
    with open(sample_file, "rb") as f:
        response = requests.post(
            url,
            files={"file": ("sample_clean_cursive.png", f, "image/png")},
            params={"beam_width": 1},
            timeout=INFERENCE_TIMEOUT,
        )

    assert response.status_code == 200
    data = response.json()

    full_text = "\n".join(p.get("full_text", "") for p in data.get("pages", []))
    for forbidden in FORBIDDEN_MOCK_STRINGS:
        assert forbidden.lower() not in full_text.lower(), f"Detected forbidden mock string '{forbidden}' in live response"

    # Assert no mock flags
    assert data.get("is_mock") is not True, "Response contains is_mock=True"


@pytest.mark.azure
def test_azure_frontend_api_proxy() -> None:
    """Upload sample to live Next.js frontend /api/recognize and validate proxy response."""
    sample_file = SAMPLES_DIR / "sample_clean_cursive.png"
    assert sample_file.exists()

    url = f"{FRONTEND_URL}/api/recognize"
    with open(sample_file, "rb") as f:
        response = requests.post(
            url,
            files={"file": ("sample_clean_cursive.png", f, "image/png")},
            timeout=INFERENCE_TIMEOUT,
        )

    assert response.status_code == 200, f"Expected HTTP 200 from frontend proxy, got {response.status_code}: {response.text}"

    # Assert proxy response originates from live backend container, not fallback mock engine
    assert (
        response.headers.get("x-recognition-provider") == "backend"
    ), f"Expected backend recognition provider, got: {response.headers.get('x-recognition-provider')}"

    data = response.json()
    assert data.get("is_mock") is not True, "Frontend proxy returned mock response (is_mock is True)"

    full_text = "\n".join(p.get("full_text", "") for p in data.get("pages", []))
    for forbidden in FORBIDDEN_MOCK_STRINGS:
        assert forbidden.lower() not in full_text.lower(), f"Detected forbidden mock string '{forbidden}' in proxy response"

    assert "document_id" in data or "id" in data, f"Response missing document ID: {data}"
    assert "pages" in data and len(data["pages"]) >= 1, f"Response missing pages: {data}"
    page = data["pages"][0]
    assert len(page.get("lines", [])) >= 1, f"Proxy response returned 0 lines: {page}"
    for idx, line in enumerate(page.get("lines", [])):
        assert "text" in line and len(line["text"].strip()) > 0
        _validate_bbox(line["bbox"], f"Proxy line {idx} bbox")

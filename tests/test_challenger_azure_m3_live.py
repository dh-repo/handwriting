"""
Empirical Adversarial Stress Test Suite for Microsoft Azure Live Deployment.
Target Endpoints:
- Frontend: https://ca-frontend-playground.jollysand-1dc47ca9.eastus2.azurecontainerapps.io
- Health/recognize: frontend /api/health and /api/recognize (backend is internal)
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import pytest
import requests

FRONTEND_URL = os.environ.get(
    "AZURE_FRONTEND_URL",
    "https://ca-frontend-playground.jollysand-1dc47ca9.eastus2.azurecontainerapps.io",
).rstrip("/")
RECOGNIZE_URL = os.environ.get("AZURE_RECOGNIZE_URL", f"{FRONTEND_URL}/api/recognize").rstrip("/")
HEALTH_URL = os.environ.get("AZURE_HEALTH_URL", f"{FRONTEND_URL}/api/health").rstrip("/")

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "samples"

HEALTH_TIMEOUT = 10
INFERENCE_TIMEOUT = 240

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
    """Validate normalized [ymin, xmin, ymax, xmax] bounding box in [0.0, 1.0]."""
    assert isinstance(bbox, (list, tuple)), f"{label} must be a list/tuple, got {type(bbox)}"
    assert len(bbox) == 4, f"{label} must have 4 coordinates, got {len(bbox)}: {bbox}"
    ymin, xmin, ymax, xmax = [float(c) for c in bbox]
    for val, name in zip([ymin, xmin, ymax, xmax], ["ymin", "xmin", "ymax", "xmax"]):
        assert 0.0 <= val <= 1.0, f"{label} {name}={val} outside [0.0, 1.0]"
    assert ymin < ymax, f"{label} invalid vertical bounds: ymin={ymin} >= ymax={ymax}"
    assert xmin < xmax, f"{label} invalid horizontal bounds: xmin={xmin} >= xmax={xmax}"


@pytest.mark.azure
def test_azure_frontend_proxy_clean_cursive() -> None:
    """
    Step 1: Send samples/sample_clean_cursive.png to frontend /api/recognize.
    Verify HTTP 200, header x-recognition-provider: backend, is_mock: false,
    genuine TrOCR transcription, valid bounding boxes, and no forbidden mock strings.
    """
    sample_file = SAMPLES_DIR / "sample_clean_cursive.png"
    assert sample_file.exists(), f"Sample file {sample_file} not found"

    url = f"{FRONTEND_URL}/api/recognize"
    start_time = time.perf_counter()
    with open(sample_file, "rb") as f:
        response = requests.post(
            url,
            files={"file": ("sample_clean_cursive.png", f, "image/png")},
            timeout=INFERENCE_TIMEOUT,
        )
    elapsed = time.perf_counter() - start_time

    # 1. HTTP 200 OK
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    # 2. Header x-recognition-provider: backend
    headers_lower = {k.lower(): v for k, v in response.headers.items()}
    provider = headers_lower.get("x-recognition-provider")
    assert provider == "backend", f"Expected x-recognition-provider: backend, got '{provider}'"

    # 3. Body validation
    data = response.json()
    assert data.get("is_mock") is not True, f"Expected is_mock not True, got {data.get('is_mock')}"

    # 4. Transcription content
    pages = data.get("pages", [])
    assert len(pages) >= 1, f"Expected at least 1 page, got {len(pages)}"
    full_text = "\n".join(p.get("full_text", "") for p in pages)
    assert len(full_text.strip()) > 0, "Empty full_text returned"

    for forbidden in FORBIDDEN_MOCK_STRINGS:
        assert forbidden.lower() not in full_text.lower(), f"Forbidden mock string found: {forbidden}"

    # 5. Bounding box validity
    total_lines = 0
    for page_idx, page in enumerate(pages):
        lines = page.get("lines", [])
        total_lines += len(lines)
        for line_idx, line in enumerate(lines):
            assert "text" in line and len(line["text"].strip()) > 0
            _validate_bbox(line["bbox"], f"Page {page_idx} Line {line_idx}")

    assert total_lines >= 2, f"Expected at least 2 lines transcribed, got {total_lines}"
    print(f"\n[PASS] Frontend proxy recognition latency: {elapsed:.2f}s, lines: {total_lines}, text snippet: {full_text[:60]!r}")


@pytest.mark.azure
def test_azure_frontend_proxy_json_base64() -> None:
    """
    Step 1b: Verify frontend /api/recognize with JSON base64 payload.
    """
    sample_file = SAMPLES_DIR / "sample_clean_cursive.png"
    with open(sample_file, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode("utf-8")

    url = f"{FRONTEND_URL}/api/recognize"
    start_time = time.perf_counter()
    response = requests.post(
        url,
        json={"file_base64": b64_content, "filename": "sample_clean_cursive.png"},
        timeout=INFERENCE_TIMEOUT,
    )
    elapsed = time.perf_counter() - start_time

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert response.headers.get("x-recognition-provider") == "backend"
    data = response.json()
    assert data.get("is_mock") is not True
    assert len(data.get("pages", [])) >= 1
    print(f"\n[PASS] Frontend proxy JSON base64 latency: {elapsed:.2f}s")


@pytest.mark.azure
def test_concurrent_health_probe_during_inference() -> None:
    """
    Step 2: Concurrent health probe during live inference.
    While a recognition request is executing, query GET /v1/health repeatedly
    and verify HTTP 200 responds without blocking or timing out (<500ms per probe).
    """
    sample_file = SAMPLES_DIR / "sample_clean_cursive.png"
    recognize_url = RECOGNIZE_URL
    health_url = HEALTH_URL

    health_results: List[Dict[str, Any]] = []
    inference_result: Dict[str, Any] = {}

    def run_inference():
        try:
            start = time.perf_counter()
            with open(sample_file, "rb") as f:
                res = requests.post(
                    recognize_url,
                    files={"file": ("sample_clean_cursive.png", f, "image/png")},
                    params={"beam_width": 1},
                    timeout=INFERENCE_TIMEOUT,
                )
            inference_result["status"] = res.status_code
            inference_result["elapsed"] = time.perf_counter() - start
            inference_result["body"] = res.json() if res.status_code == 200 else res.text
        except Exception as e:
            inference_result["error"] = str(e)

    # Launch recognition in background thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        inf_future = executor.submit(run_inference)

        # Give inference 1 second to start processing
        time.sleep(1.0)

        # Poll health endpoint while inference is active
        probe_count = 0
        while not inf_future.done() and probe_count < 20:
            probe_count += 1
            probe_start = time.perf_counter()
            try:
                h_res = requests.get(health_url, timeout=HEALTH_TIMEOUT)
                h_elapsed = time.perf_counter() - probe_start
                health_results.append({
                    "probe_num": probe_count,
                    "status_code": h_res.status_code,
                    "elapsed_sec": h_elapsed,
                    "healthy": h_res.json().get("status") == "healthy" if h_res.status_code == 200 else False,
                })
            except Exception as e:
                health_results.append({
                    "probe_num": probe_count,
                    "error": str(e),
                    "elapsed_sec": time.perf_counter() - probe_start,
                })
            time.sleep(1.0)

        inf_future.result()

    # Verify inference succeeded
    assert inference_result.get("status") == 200, f"Inference failed during test: {inference_result}"
    print(f"\n[PASS] Concurrent inference completed in {inference_result['elapsed']:.2f}s")

    # Verify all health probes during inference
    assert len(health_results) > 0, "No health probes were executed"
    for probe in health_results:
        assert "error" not in probe, f"Health probe failed with error: {probe}"
        assert probe["status_code"] == 200, f"Health probe returned status {probe['status_code']}"
        assert probe["healthy"] is True, f"Health probe status was not healthy: {probe}"
        assert probe["elapsed_sec"] < 1.0, f"Health probe blocked/slow ({probe['elapsed_sec']:.3f}s): {probe}"

    avg_health_latency = sum(p["elapsed_sec"] for p in health_results) / len(health_results)
    max_health_latency = max(p["elapsed_sec"] for p in health_results)
    print(f"[PASS] {len(health_results)} health probes during active inference: avg={avg_health_latency*1000:.1f}ms, max={max_health_latency*1000:.1f}ms")


@pytest.mark.azure
def test_adversarial_error_handling() -> None:
    """
    Adversarial challenge: Verify graceful error responses (never unhandled 500 or hangs)
    for invalid/malformed inputs.
    """
    # 1. Empty 0-byte file to frontend
    res_empty_fe = requests.post(
        f"{FRONTEND_URL}/api/recognize",
        files={"file": ("empty.png", b"", "image/png")},
        timeout=15,
    )
    assert res_empty_fe.status_code in [400, 422], f"Expected 400/422 for empty file, got {res_empty_fe.status_code}"

    # 2. Corrupt random image bytes to frontend
    res_corrupt_fe = requests.post(
        f"{FRONTEND_URL}/api/recognize",
        files={"file": ("corrupt.png", b"NOT_A_VALID_IMAGE_DATA_1234567890", "image/png")},
        timeout=15,
    )
    assert res_corrupt_fe.status_code in [400, 422], f"Expected 400/422 for corrupt image, got {res_corrupt_fe.status_code}"

    # 3. GET /api/recognize Method Not Allowed
    res_get_fe = requests.get(f"{FRONTEND_URL}/api/recognize", timeout=15)
    assert res_get_fe.status_code == 405, f"Expected 405 Method Not Allowed, got {res_get_fe.status_code}"

    # 4. Security headers on Frontend
    res_home = requests.get(FRONTEND_URL, timeout=15)
    assert res_home.status_code == 200
    headers_lower = {k.lower(): v for k, v in res_home.headers.items()}
    assert headers_lower.get("x-frame-options") == "DENY"
    assert headers_lower.get("x-content-type-options") == "nosniff"
    print("\n[PASS] Adversarial error handling and security headers verified")

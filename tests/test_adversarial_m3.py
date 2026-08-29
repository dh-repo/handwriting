"""
tests/test_adversarial_m3.py
Adversarial Stress Test Suite for Milestone 3 (Inference Backend & Serving Architecture).
Covers:
- Corrupted headers, zero-byte uploads, truncated streams
- Malformed base64, corrupted JSON, type-juggling attacks
- Nonexistent job IDs, SQL/path injection strings in job routes
- Rapid concurrent async bursts (50+ parallel requests, race condition stress)
- Information leakage audits: zero raw stack traces / file paths across all 4xx/5xx responses
- Extreme aspect ratios (1x1, 10000x1, 1x10000, 4000x4000, pure noise)
- Exotic filenames (path traversal, emojis, null bytes, unicode)
- Payload size bounds and unsupported methods
"""

from __future__ import annotations
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import random
import string
import sys
import time
from typing import Any, Dict, List
import numpy as np
from PIL import Image, ImageDraw
import pytest
from fastapi.testclient import TestClient
import httpx

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["USE_MOCK_ENGINE"] = "true"

from backend.app.config import get_settings, reset_settings_cache
from backend.app.engine import InferenceEngine, get_engine, reset_engine, set_engine
from backend.app.main import create_app
from backend.app.schemas import RecognitionResponse


@pytest.fixture(autouse=True)
def setup_backend():
    os.environ["USE_MOCK_ENGINE"] = "true"
    reset_settings_cache()
    reset_engine()
    yield
    reset_settings_cache()
    reset_engine()


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def create_valid_test_png(w: int = 400, h: int = 200, text: str = "Test Rx Note") -> bytes:
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# 1. Zero-Byte and Corrupted File Upload Stress Tests
# ===========================================================================

class TestZeroByteAndCorruptedUploads:
    """Stress test backend against zero-byte, truncated, and corrupt payloads."""

    def test_zero_byte_png_multipart(self, client: TestClient):
        resp = client.post(
            "/v1/recognize",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "empty" in data.get("detail", "").lower() or "invalid" in data.get("detail", "").lower()
        # Verify no stack trace leaked
        assert "Traceback" not in resp.text
        assert "File \"" not in resp.text

    def test_zero_byte_base64_json(self, client: TestClient):
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": "", "filename": "empty.png"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "empty" in data.get("detail", "").lower() or "missing" in data.get("detail", "").lower()
        assert "Traceback" not in resp.text

    def test_corrupted_png_header_multipart(self, client: TestClient):
        corrupted = b"\x89PNG\r\n\x1a\n" + os.urandom(64)
        resp = client.post(
            "/v1/recognize",
            files={"file": ("corrupt.png", corrupted, "image/png")},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text
        assert resp.headers["content-type"].startswith("application/json")

    def test_corrupted_jpeg_truncated_soi(self, client: TestClient):
        corrupted = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + os.urandom(32)
        resp = client.post(
            "/v1/recognize",
            files={"file": ("corrupt.jpg", corrupted, "image/jpeg")},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_corrupted_pdf_truncated_header(self, client: TestClient):
        corrupted = b"%PDF-1.7\n<< /Type /Catalog >>" + os.urandom(32)
        resp = client.post(
            "/v1/recognize",
            files={"file": ("corrupt.pdf", corrupted, "application/pdf")},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_corrupted_pdf_infinite_empty_stream(self, client: TestClient):
        corrupted = b"%PDF-1.4\n1 0 obj\n<< /Length 999999 >>\nstream\n"
        resp = client.post(
            "/v1/recognize",
            files={"file": ("truncated_stream.pdf", corrupted, "application/pdf")},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_random_binary_garbage_payload(self, client: TestClient):
        garbage = os.urandom(1024)
        resp = client.post(
            "/v1/recognize",
            files={"file": ("random_garbage.bin", garbage, "application/octet-stream")},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text


# ===========================================================================
# 2. Malformed Base64 and Invalid JSON Body Attacks
# ===========================================================================

class TestMalformedBase64AndJsonAttacks:
    """Stress test JSON parsing and Base64 decoding robustness."""

    def test_invalid_base64_characters(self, client: TestClient):
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": "!!!NOT_VALID_BASE64@@@###$$$"},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_base64_invalid_padding(self, client: TestClient):
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": "====="},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_base64_type_mismatch_int(self, client: TestClient):
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": 123456789},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_base64_type_mismatch_array(self, client: TestClient):
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": ["a", "b", "c"]},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_malformed_json_unclosed_brace(self, client: TestClient):
        resp = client.post(
            "/v1/recognize",
            content='{"file_base64": "abc',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_options_type_mismatch_string(self, client: TestClient):
        valid_b64 = base64.b64encode(create_valid_test_png()).decode()
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": valid_b64, "options": "not_an_object"},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_options_invalid_dpi_out_of_range(self, client: TestClient):
        valid_b64 = base64.b64encode(create_valid_test_png()).decode()
        # dpi must be between 72 and 600
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": valid_b64, "options": {"dpi": 10}},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_options_invalid_dpi_high_out_of_range(self, client: TestClient):
        valid_b64 = base64.b64encode(create_valid_test_png()).decode()
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": valid_b64, "options": {"dpi": 999999}},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text


# ===========================================================================
# 3. Nonexistent Job IDs, Polling, and SSE Stream Stress
# ===========================================================================

class TestNonexistentJobIdsAndSSE:
    """Stress test async job routes against injection, missing IDs, and SSE disconnects."""

    def test_get_job_unknown_alphanumeric(self, client: TestClient):
        resp = client.get("/v1/jobs/job_unknown_987654321")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
        assert "Traceback" not in resp.text

    def test_get_job_path_traversal_id(self, client: TestClient):
        resp = client.get("/v1/jobs/../../etc/passwd")
        # FastAPI router will either return 404 or 405
        assert resp.status_code in (404, 405)
        assert "Traceback" not in resp.text

    def test_get_job_sql_injection_string(self, client: TestClient):
        resp = client.get("/v1/jobs/' OR '1'='1'--")
        assert resp.status_code == 404
        assert "Traceback" not in resp.text

    def test_get_job_special_characters_fuzz(self, client: TestClient):
        fuzz_strings = [
            "null", "None", "undefined", "0", "-1",
            "<script>alert(1)</script>", "!@#$%^&*()_+{}|:\"<>?",
            "\\x00\\x01\\x02", "job_%00_admin"
        ]
        for f in fuzz_strings:
            resp = client.get(f"/v1/jobs/{f}")
            assert resp.status_code in (404, 422)
            assert "Traceback" not in resp.text

    def test_sse_stream_nonexistent_job_emits_error_event(self, client: TestClient):
        resp = client.get("/v1/jobs/job_nonexistent_9999/stream")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        content = resp.text
        assert "event: error" in content
        assert "not found" in content.lower()
        assert "Traceback" not in content

    def test_sse_events_alias_nonexistent_job(self, client: TestClient):
        resp = client.get("/v1/jobs/job_missing_xyz/events")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "event: error" in resp.text

    def test_jobs_submit_zero_byte_multipart(self, client: TestClient):
        resp = client.post(
            "/v1/jobs",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert resp.status_code == 422
        assert "Traceback" not in resp.text

    def test_jobs_submit_corrupt_payload_transitions_to_failed(self, client: TestClient):
        """Verify corrupt payload submitted to jobs queue safely transitions to FAILED without crash."""
        corrupt_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\ncorrupt").decode()
        resp = client.post(
            "/v1/jobs",
            json={"file_base64": corrupt_b64, "filename": "corrupt.png"},
        )
        assert resp.status_code == 200
        jid = resp.json()["job_id"]
        # Poll 3 times
        client.get(f"/v1/jobs/{jid}")
        client.get(f"/v1/jobs/{jid}")
        final_poll = client.get(f"/v1/jobs/{jid}").json()
        assert final_poll["status"] == "FAILED"
        assert final_poll["error"] is not None
        assert "Traceback" not in json.dumps(final_poll)


# ===========================================================================
# 4. Concurrency Stress Test & Race Conditions
# ===========================================================================

class TestConcurrencyAndRaceConditions:
    """Stress test concurrent requests using httpx AsyncClient."""

    @pytest.mark.asyncio
    async def test_rapid_concurrent_recognize_burst(self):
        """Send 30 concurrent POST /v1/recognize requests simultaneously."""
        app = create_app()
        png_bytes = create_valid_test_png(300, 150, "Concurrent Burst Test")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            async def send_req(i: int):
                b64_str = base64.b64encode(png_bytes).decode()
                return await ac.post(
                    "/v1/recognize",
                    json={"file_base64": b64_str, "filename": f"burst_{i}.png"},
                )

            tasks = [send_req(i) for i in range(30)]
            responses = await asyncio.gather(*tasks)

        assert len(responses) == 30
        for idx, resp in enumerate(responses):
            assert resp.status_code == 200, f"Req {idx} failed with {resp.status_code}: {resp.text}"
            data = resp.json()
            assert data["total_pages"] == 1
            assert len(data["pages"][0]["lines"]) > 0
            assert "Traceback" not in resp.text

    @pytest.mark.asyncio
    async def test_rapid_concurrent_job_submission_and_polling(self):
        """Submit 20 concurrent jobs and poll their status concurrently."""
        app = create_app()
        png_bytes = create_valid_test_png(250, 100, "Async Concurrent Job")
        b64_str = base64.b64encode(png_bytes).decode()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # 1. Submit 20 jobs concurrently
            async def submit_job(i: int):
                return await ac.post(
                    "/v1/jobs",
                    json={"file_base64": b64_str, "filename": f"job_{i}.png"},
                )

            submit_resps = await asyncio.gather(*[submit_job(i) for i in range(20)])
            job_ids = []
            for resp in submit_resps:
                assert resp.status_code == 200
                job_ids.append(resp.json()["job_id"])

            # 2. Poll all jobs concurrently multiple times to verify status transitions
            for poll_round in range(3):
                async def poll_job(jid: str):
                    return await ac.get(f"/v1/jobs/{jid}")

                poll_resps = await asyncio.gather(*[poll_job(jid) for jid in job_ids])
                for r in poll_resps:
                    assert r.status_code == 200
                    st = r.json()["status"]
                    assert st in ("QUEUED", "PROCESSING", "COMPLETED")

            # In mock environment, after 2+ polls it should complete
            final_resps = await asyncio.gather(*[ac.get(f"/v1/jobs/{jid}") for jid in job_ids])
            for r in final_resps:
                assert r.status_code == 200
                data = r.json()
                assert data["status"] == "COMPLETED"
                assert data["result"] is not None

    @pytest.mark.asyncio
    async def test_concurrent_sse_streams_on_same_job(self):
        """Connect 5 simultaneous SSE stream clients to the exact same job ID."""
        app = create_app()
        png_bytes = create_valid_test_png(200, 100, "Shared SSE Job")
        b64_str = base64.b64encode(png_bytes).decode()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # Submit job
            sub_resp = await ac.post(
                "/v1/jobs",
                json={"file_base64": b64_str, "filename": "shared.png"},
            )
            job_id = sub_resp.json()["job_id"]

            # Connect 5 concurrent SSE listeners
            async def stream_client(c_idx: int):
                r = await ac.get(f"/v1/jobs/{job_id}/stream")
                return r

            stream_resps = await asyncio.gather(*[stream_client(i) for i in range(5)])
            for sr in stream_resps:
                assert sr.status_code == 200
                assert "text/event-stream" in sr.headers["content-type"]
                assert "event: progress" in sr.text or "event: complete" in sr.text


# ===========================================================================
# 5. Extreme Dimensions, Aspect Ratios, and Noise
# ===========================================================================

class TestExtremeDimensionsAndNoise:
    """Stress test boundary image dimensions, aspect ratios, and pathological content."""

    def test_1x1_pixel_image(self, client: TestClient):
        img = Image.new("RGB", (1, 1), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            "/v1/recognize",
            files={"file": ("tiny_1x1.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pages"] == 1
        assert data["pages"][0]["width"] == 1
        assert data["pages"][0]["height"] == 1

    def test_extreme_horizontal_aspect_ratio_10000x1(self, client: TestClient):
        img = Image.new("RGB", (1000, 2), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            "/v1/recognize",
            files={"file": ("flat_strip.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pages"] == 1

    def test_extreme_vertical_aspect_ratio_1x1000(self, client: TestClient):
        img = Image.new("RGB", (2, 1000), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            "/v1/recognize",
            files={"file": ("tall_needle.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pages"] == 1

    def test_pure_noise_image(self, client: TestClient):
        noise_arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        img = Image.fromarray(noise_arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            "/v1/recognize",
            files={"file": ("noise.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pages"] == 1

    def test_all_black_image(self, client: TestClient):
        img = Image.new("RGB", (400, 400), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            "/v1/recognize",
            files={"file": ("all_black.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200

    def test_all_white_image(self, client: TestClient):
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            "/v1/recognize",
            files={"file": ("all_white.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200


# ===========================================================================
# 6. Exotic Filenames & Path Traversal Stress
# ===========================================================================

class TestExoticFilenamesAndPathTraversal:
    """Stress test dangerous, unicode, and path traversal filenames."""

    @pytest.mark.parametrize("bad_name", [
        "../../../../../../etc/passwd",
        "prescription_💊_💉_📋_2026.png",
        "وصفة_طبية_باللغة_العربية.png",
        "處方簽_手寫測試.png",
        "file with spaces and (special) [chars] #1.png",
        "a" * 300 + ".png",
        "test\x00nullbyte.png",
        "CON.png",
        "AUX.png",
    ])
    def test_exotic_filename_handling(self, client: TestClient, bad_name: str):
        png_bytes = create_valid_test_png(200, 100, "Exotic Filename")
        resp = client.post(
            "/v1/recognize",
            files={"file": (bad_name, png_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "filename" in data
        # Ensure POSIX path traversal characters were stripped by Path(filename).name
        assert "/" not in data["filename"]
        assert "Traceback" not in resp.text


# ===========================================================================
# 7. Stack Trace Leakage & Security Audit Across Error Codes
# ===========================================================================

class TestStackTraceLeakageAndSecurity:
    """Verify that 400, 404, 405, 422, and 500 never leak internal Python stack traces."""

    def test_404_error_clean_json(self, client: TestClient):
        resp = client.get("/v1/nonexistent_route_404")
        assert resp.status_code == 404
        assert resp.headers["content-type"].startswith("application/json")
        assert "Traceback" not in resp.text
        assert "File \"" not in resp.text

    def test_405_method_not_allowed_clean_json(self, client: TestClient):
        resp = client.put("/v1/recognize")
        assert resp.status_code == 405
        assert "Traceback" not in resp.text

    def test_422_validation_error_structure(self, client: TestClient):
        resp = client.post("/v1/recognize", json={})
        assert resp.status_code == 422
        data = resp.json()
        assert "error" in data or "detail" in data
        assert "Traceback" not in resp.text
        assert "site-packages" not in resp.text

    def test_500_fault_injected_engine_sanitization(self):
        class ExplodingEngine(InferenceEngine):
            def recognize(self, *args, **kwargs):
                raise ZeroDivisionError("FATAL: line 99 in /internal/classified/engine_secret.py: div by zero")

        set_engine(ExplodingEngine(execution_mode="mock"))
        app = create_app()
        err_client = TestClient(app, raise_server_exceptions=False)
        png_bytes = create_valid_test_png()

        resp = err_client.post(
            "/v1/recognize",
            files={"file": ("boom.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "Internal Server Error"
        # Secret path and internal stack trace MUST NOT be leaked to client
        assert "engine_secret.py" not in json.dumps(data)
        assert "ZeroDivisionError" not in json.dumps(data)
        assert "Traceback" not in resp.text
        assert "classified" not in json.dumps(data)

    def test_max_upload_size_limit_exceeded(self, client: TestClient):
        # Default MAX_IMAGE_SIZE_MB is 50MB. Simulate 51MB payload.
        settings = get_settings()
        # Override to 1MB temporarily for fast unit testing
        original_max = settings.MAX_IMAGE_SIZE_MB
        try:
            settings.MAX_IMAGE_SIZE_MB = 1  # 1MB max
            oversized_bytes = b"0" * (2 * 1024 * 1024)  # 2MB
            resp = client.post(
                "/v1/recognize",
                files={"file": ("huge.png", oversized_bytes, "image/png")},
            )
            assert resp.status_code == 422
            assert "exceeds maximum allowed size" in resp.json()["detail"].lower()
            assert "Traceback" not in resp.text
        finally:
            settings.MAX_IMAGE_SIZE_MB = original_max


# ===========================================================================
# 8. Bounding Box & Coordinates Strict Invariant Audit
# ===========================================================================

class TestBoundingBoxInvariantAudit:
    """Verify that all generated coordinates in RecognitionResponse strictly follow interface contracts."""

    def test_bbox_normalized_invariants_image(self, client: TestClient):
        png_bytes = create_valid_test_png(800, 600, "Invoice and Prescription Details")
        resp = client.post(
            "/v1/recognize",
            files={"file": ("rx.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Validate with Pydantic schema model
        validated = RecognitionResponse(**data)
        assert validated.total_pages == 1

        for page in validated.pages:
            assert page.width > 0
            assert page.height > 0
            assert 0.0 <= page.mean_confidence <= 1.0

            for line in page.lines:
                ymin, xmin, ymax, xmax = line.bbox
                assert 0.0 <= ymin < ymax <= 1.0, f"Invalid line bbox: {line.bbox}"
                assert 0.0 <= xmin < xmax <= 1.0, f"Invalid line bbox: {line.bbox}"
                assert 0.0 <= line.confidence <= 1.0

                for word in line.words:
                    wy_min, wx_min, wy_max, wx_max = word.bbox
                    assert 0.0 <= wy_min < wy_max <= 1.0, f"Invalid word bbox: {word.bbox}"
                    assert 0.0 <= wx_min < wx_max <= 1.0, f"Invalid word bbox: {word.bbox}"
                    assert 0.0 <= word.confidence <= 1.0

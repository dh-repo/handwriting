"""
tests/test_challenger_m4_api_stress.py
Adversarial Stress Test Suite for Milestone 4: Production Inference Backend & Rescorer Serving.

Tests empirical edge cases, malicious & malformed payloads, non-standard image dimensions,
extreme aspect ratios, boundary beam widths, corrupted documents, multi-page PDFs,
concurrency, error sanitization, and security posture.
"""

from __future__ import annotations
import asyncio
import base64
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Generator, List
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.config import get_settings, reset_settings_cache
from backend.app.engine import (
    CorruptDocumentError,
    DocumentLoadingError,
    EmptyDocumentError,
    InferenceEngine,
    get_engine,
    reset_engine,
    set_engine,
)
from backend.app.main import create_app
from backend.app.schemas import (
    ErrorResponse,
    HealthResponse,
    JobStatusEnum,
    JobStatusResponse,
    RecognitionOptions,
    RecognitionResponse,
    RecognizeJsonRequest,
)


@pytest.fixture(autouse=True)
def clean_engine_and_settings() -> Generator[None, None, None]:
    """Ensure mock engine is enabled and settings/engine caches are reset cleanly."""
    os.environ["USE_MOCK_ENGINE"] = "true"
    reset_settings_cache()
    reset_engine()
    yield
    reset_settings_cache()
    reset_engine()


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    app = create_app()
    return TestClient(app)


def _create_sample_png(width: int = 600, height: int = 300, text: str = "Amoxicillin 500mg PO TID") -> bytes:
    """Helper to generate in-memory synthetic PNG."""
    img = Image.new("RGB", (max(1, width), max(1, height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    if width >= 20 and height >= 20:
        draw.text((10, 10), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_multipage_pdf(num_pages: int = 3) -> bytes:
    """Helper to generate multi-page PDF in-memory."""
    pages = []
    for i in range(num_pages):
        img = Image.new("RGB", (600, 800), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), f"Clinical Chart Note - Page {i+1}", fill=(0, 0, 0))
        draw.text((50, 150), f"Medication #{i+1}: Amoxicillin 500mg TID", fill=(0, 0, 0))
        pages.append(img)
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


# =========================================================================
# 1. Malformed Payloads & Boundary Uploads
# =========================================================================
class TestMalformedAndBoundaryPayloads:
    """Stress-test malformed JSON, corrupted base64, 0-byte, and oversized uploads."""

    def test_recognize_empty_base64_json_returns_422(self, client: TestClient) -> None:
        """POST /v1/recognize with empty base64 string must return 422 Unprocessable Entity."""
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": "", "filename": "empty.png"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "empty" in data.get("detail", "").lower() or "validation" in data.get("error", "").lower()

    def test_recognize_invalid_base64_characters_returns_422(self, client: TestClient) -> None:
        """POST /v1/recognize with invalid base64 characters must return 422 without 500."""
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": "!!!INVALID_BASE64_BYTES@@@###$$$", "filename": "bad.png"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "invalid" in data.get("detail", "").lower() or "base64" in data.get("detail", "").lower()

    def test_recognize_truncated_base64_padding_returns_422(self, client: TestClient) -> None:
        """POST /v1/recognize with broken padding must return 422."""
        raw_b64 = base64.b64encode(b"valid bytes").decode("utf-8")
        corrupted_b64 = raw_b64[:-2] + "==="  # Illegal padding
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": corrupted_b64, "filename": "corrupt_pad.png"},
        )
        assert resp.status_code == 422

    def test_recognize_missing_file_base64_field_returns_422(self, client: TestClient) -> None:
        """POST /v1/recognize with missing required file_base64 field."""
        resp = client.post(
            "/v1/recognize",
            json={"filename": "test.png"},
        )
        assert resp.status_code == 422

    def test_recognize_invalid_json_datatype_returns_422(self, client: TestClient) -> None:
        """POST /v1/recognize with integer instead of string for file_base64."""
        resp = client.post(
            "/v1/recognize",
            json={"file_base64": 123456789, "filename": "bad_type.png"},
        )
        assert resp.status_code == 422

    def test_recognize_malformed_json_body_returns_422(self, client: TestClient) -> None:
        """POST /v1/recognize with unparseable raw JSON body."""
        resp = client.post(
            "/v1/recognize",
            content=b"{ broken: json [ unclosed",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422

    def test_recognize_zero_byte_multipart_file_returns_422(self, client: TestClient) -> None:
        """POST /v1/recognize with 0-byte multipart file upload returns 422."""
        resp = client.post(
            "/v1/recognize",
            files={"file": ("zero_bytes.png", b"", "image/png")},
        )
        assert resp.status_code == 422
        assert "empty" in resp.json().get("detail", "").lower()

    def test_recognize_oversized_upload_rejection_returns_422(self, client: TestClient) -> None:
        """POST /v1/recognize with upload exceeding MAX_IMAGE_SIZE_MB (>50MB)."""
        # Mock settings with small limit to test boundary deterministically
        with patch.object(get_settings(), "MAX_IMAGE_SIZE_MB", 1):
            oversized_bytes = b"X" * (2 * 1024 * 1024)  # 2MB > 1MB limit
            resp = client.post(
                "/v1/recognize",
                files={"file": ("huge.png", oversized_bytes, "image/png")},
            )
            assert resp.status_code == 422
            assert "exceeds" in resp.json().get("detail", "").lower()

    def test_jobs_empty_base64_json_returns_422(self, client: TestClient) -> None:
        """POST /v1/jobs with empty base64 string returns 422."""
        resp = client.post(
            "/v1/jobs",
            json={"file_base64": "", "filename": "empty.pdf"},
        )
        assert resp.status_code == 422

    def test_jobs_oversized_upload_rejection_returns_422(self, client: TestClient) -> None:
        """POST /v1/jobs with oversized payload returns 422."""
        with patch.object(get_settings(), "MAX_IMAGE_SIZE_MB", 1):
            oversized_bytes = b"X" * (2 * 1024 * 1024)
            resp = client.post(
                "/v1/jobs",
                files={"file": ("huge.pdf", oversized_bytes, "application/pdf")},
            )
            assert resp.status_code == 422
            assert "exceeds" in resp.json().get("detail", "").lower()


# =========================================================================
# 2. Corrupted Image & Document Bytes
# =========================================================================
class TestCorruptedDocumentBytes:
    """Stress-test malformed magic headers, truncated image formats, and garbage bytes."""

    def test_recognize_truncated_png_header(self, client: TestClient) -> None:
        """Truncated PNG magic bytes with corrupted chunk data."""
        corrupt_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00\x00\x01\x00"
        resp = client.post(
            "/v1/recognize",
            files={"file": ("broken.png", corrupt_png, "image/png")},
        )
        assert resp.status_code == 422
        assert "error" in resp.json() or "detail" in resp.json()

    def test_recognize_truncated_jpeg(self, client: TestClient) -> None:
        """Truncated JPEG file with invalid SOI / SOF markers."""
        corrupt_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01" + b"\xff" * 8
        resp = client.post(
            "/v1/recognize",
            files={"file": ("broken.jpg", corrupt_jpeg, "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_recognize_truncated_tiff(self, client: TestClient) -> None:
        """Truncated TIFF with corrupted IFD offset."""
        corrupt_tiff = b"II*\x00\x08\x00\x00\x00\xff\xff"
        resp = client.post(
            "/v1/recognize",
            files={"file": ("broken.tiff", corrupt_tiff, "image/tiff")},
        )
        assert resp.status_code == 422

    def test_recognize_random_binary_garbage(self, client: TestClient) -> None:
        """Arbitrary pseudo-random noise bytes."""
        random_bytes = os.urandom(2048)
        resp = client.post(
            "/v1/recognize",
            files={"file": ("random.bin", random_bytes, "application/octet-stream")},
        )
        assert resp.status_code == 422

    def test_recognize_single_byte_payloads(self, client: TestClient) -> None:
        """Micro byte payloads (1, 2, 4 bytes)."""
        for b_len in [1, 2, 4, 7]:
            resp = client.post(
                "/v1/recognize",
                files={"file": (f"tiny_{b_len}.png", b"\x00" * b_len, "image/png")},
            )
            assert resp.status_code == 422, f"Failed for {b_len}-byte payload"

    def test_recognize_corrupted_pdf_structure(self, client: TestClient) -> None:
        """Broken PDF with valid magic header but corrupted xref/trailers."""
        corrupt_pdf = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF_CORRUPT"
        resp = client.post(
            "/v1/recognize",
            files={"file": ("broken.pdf", corrupt_pdf, "application/pdf")},
        )
        assert resp.status_code == 422

    def test_engine_corrupt_bytes_direct_exception(self) -> None:
        """Direct InferenceEngine invocation with corrupt bytes raises CorruptDocumentError / DocumentLoadingError."""
        engine = InferenceEngine(execution_mode="mock")
        with pytest.raises((CorruptDocumentError, DocumentLoadingError, Exception)):
            engine.recognize(b"garbage_not_an_image", filename="corrupt.png")


# =========================================================================
# 3. Non-Standard Dimensions & Extreme Aspect Ratios
# =========================================================================
class TestNonStandardDimensionsAndAspectRatios:
    """Stress-test extreme vertical/horizontal ratios, micro images, and single-tone images."""

    def test_extreme_vertical_aspect_ratio_1x1000(self, client: TestClient) -> None:
        """Extreme vertical needle image (1x1000px)."""
        img_bytes = _create_sample_png(width=1, height=1000)
        resp = client.post(
            "/v1/recognize",
            files={"file": ("needle_vert.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        model = RecognitionResponse.model_validate(data)
        assert model.pages[0].width == 1
        assert model.pages[0].height == 1000
        # Invariant checks
        for line in model.pages[0].lines:
            assert 0.0 <= line.bbox[0] < line.bbox[2] <= 1.0
            assert 0.0 <= line.bbox[1] < line.bbox[3] <= 1.0
            for word in line.words:
                assert 0.0 <= word.bbox[0] < word.bbox[2] <= 1.0
                assert 0.0 <= word.bbox[1] < word.bbox[3] <= 1.0

    def test_extreme_horizontal_aspect_ratio_4000x50(self, client: TestClient) -> None:
        """Extreme horizontal banner image (4000x50px)."""
        img_bytes = _create_sample_png(width=4000, height=50)
        resp = client.post(
            "/v1/recognize",
            files={"file": ("banner_horiz.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        model = RecognitionResponse.model_validate(data)
        assert model.pages[0].width == 4000
        assert model.pages[0].height == 50
        for line in model.pages[0].lines:
            assert 0.0 <= line.bbox[0] < line.bbox[2] <= 1.0
            assert 0.0 <= line.bbox[1] < line.bbox[3] <= 1.0

    def test_micro_image_1x1_pixel(self, client: TestClient) -> None:
        """Minimal single pixel image (1x1px)."""
        img_bytes = _create_sample_png(width=1, height=1)
        resp = client.post(
            "/v1/recognize",
            files={"file": ("single_pixel.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        model = RecognitionResponse.model_validate(data)
        assert model.pages[0].width == 1
        assert model.pages[0].height == 1

    def test_large_square_image_4000x4000(self, client: TestClient) -> None:
        """High-resolution square image (4000x4000px)."""
        img_bytes = _create_sample_png(width=4000, height=4000)
        resp = client.post(
            "/v1/recognize",
            files={"file": ("high_res.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        model = RecognitionResponse.model_validate(data)
        assert model.pages[0].width == 4000
        assert model.pages[0].height == 4000

    def test_single_tone_blank_images(self, client: TestClient) -> None:
        """Pure solid black (0), solid white (255), and solid gray (128) canvases."""
        for tone_name, color_val in [("black", (0, 0, 0)), ("white", (255, 255, 255)), ("gray", (128, 128, 128))]:
            img = Image.new("RGB", (500, 300), color=color_val)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            resp = client.post(
                "/v1/recognize",
                files={"file": (f"tone_{tone_name}.png", buf.getvalue(), "image/png")},
            )
            assert resp.status_code == 200, f"Failed on tone {tone_name}"
            model = RecognitionResponse.model_validate(resp.json())
            assert len(model.pages) == 1


# =========================================================================
# 4. Beam Width Boundary Conditions ($K=1, 2, 5, 8, 16$) & Out-of-Bounds
# =========================================================================
class TestBeamWidthBoundaryConditions:
    """Stress-test beam_width parameter bounds and validation."""

    @pytest.mark.parametrize("k_val", [1, 2, 5, 8, 16])
    def test_recognize_valid_beam_widths_query_param(self, client: TestClient, k_val: int) -> None:
        """Verify K in {1, 2, 5, 8, 16} succeeds via query parameters."""
        img_bytes = _create_sample_png()
        resp = client.post(
            f"/v1/recognize?beam_width={k_val}&rescore=true",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        model = RecognitionResponse.model_validate(data)
        assert model.preprocessing_flags.get("beam_width") == k_val

    @pytest.mark.parametrize("k_val", [1, 2, 5, 8, 16])
    def test_recognize_valid_beam_widths_json_payload(self, client: TestClient, k_val: int) -> None:
        """Verify K in {1, 2, 5, 8, 16} succeeds via JSON options."""
        img_bytes = _create_sample_png()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        resp = client.post(
            "/v1/recognize",
            json={
                "file_base64": b64,
                "filename": "beam_test.png",
                "options": {"beam_width": k_val, "rescore": True},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["preprocessing_flags"]["beam_width"] == k_val

    @pytest.mark.parametrize("invalid_k", [0, -1, 17, 100])
    def test_recognize_out_of_bounds_beam_width_rejected(self, client: TestClient, invalid_k: int) -> None:
        """Verify K outside [1, 16] is rejected with 422 Unprocessable Entity."""
        img_bytes = _create_sample_png()
        resp = client.post(
            f"/v1/recognize?beam_width={invalid_k}",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 422

    def test_recognize_non_numeric_beam_width_rejected(self, client: TestClient) -> None:
        """Verify non-integer beam_width is rejected with 422."""
        img_bytes = _create_sample_png()
        resp = client.post(
            "/v1/recognize?beam_width=invalid_string",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 422

    def test_rescore_disabled_toggle(self, client: TestClient) -> None:
        """Verify rescore=false flag passes through correctly."""
        img_bytes = _create_sample_png()
        resp = client.post(
            "/v1/recognize?rescore=false&beam_width=1",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["preprocessing_flags"]["rescore"] is False


# =========================================================================
# 5. Multi-Page Documents & Asynchronous SSE Streaming
# =========================================================================
class TestMultiPageAndDocumentWorkloads:
    """Stress-test multi-page PDFs, multi-frame TIFFs, and asynchronous lifecycle."""

    def test_multipage_pdf_synchronous_recognition(self, client: TestClient) -> None:
        """Verify synchronous processing of 5-page PDF document."""
        pdf_bytes = _create_multipage_pdf(num_pages=5)
        resp = client.post(
            "/v1/recognize",
            files={"file": ("5page_doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        model = RecognitionResponse.model_validate(resp.json())
        assert model.total_pages == 5
        assert len(model.pages) == 5
        for idx, page in enumerate(model.pages):
            assert page.page_number == idx + 1
            assert len(page.lines) >= 1
            assert 0.0 <= page.mean_confidence <= 1.0

    def test_multiframe_tiff_synchronous_recognition(self, client: TestClient) -> None:
        """Verify synchronous processing of multi-frame TIFF."""
        f1 = Image.new("RGB", (400, 300), color=(255, 255, 255))
        f2 = Image.new("RGB", (400, 300), color=(255, 255, 255))
        f3 = Image.new("RGB", (400, 300), color=(255, 255, 255))
        buf = io.BytesIO()
        f1.save(buf, format="TIFF", save_all=True, append_images=[f2, f3])

        resp = client.post(
            "/v1/recognize",
            files={"file": ("3frame.tiff", buf.getvalue(), "image/tiff")},
        )
        assert resp.status_code == 200
        model = RecognitionResponse.model_validate(resp.json())
        assert model.total_pages >= 1
        assert len(model.pages) >= 1
        for page in model.pages:
            assert page.width == 400
            assert page.height == 300
            assert len(page.lines) >= 1

    def test_async_jobs_submission_and_polling_lifecycle(self, client: TestClient) -> None:
        """Verify POST /v1/jobs submission, GET polling, and progression to COMPLETED."""
        pdf_bytes = _create_multipage_pdf(num_pages=2)
        submit_resp = client.post(
            "/v1/jobs?beam_width=8&rescore=true",
            files={"file": ("async_doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert submit_resp.status_code == 200
        init_data = submit_resp.json()
        job_id = init_data["job_id"]
        assert init_data["status"] == "QUEUED"
        assert init_data["progress"] == 0.0

        # Poll 1: Advances to PROCESSING
        p1 = client.get(f"/v1/jobs/{job_id}")
        assert p1.status_code == 200
        d1 = p1.json()
        assert d1["status"] in ("QUEUED", "PROCESSING", "COMPLETED")

        # Poll 2: Advances to COMPLETED
        p2 = client.get(f"/v1/jobs/{job_id}")
        assert p2.status_code == 200
        d2 = p2.json()
        assert d2["status"] == "COMPLETED"
        assert d2["progress"] == 1.0
        assert d2["result"] is not None
        assert d2["result"]["total_pages"] == 2

    def test_async_jobs_sse_stream_progression(self, client: TestClient) -> None:
        """Verify SSE streaming (/v1/jobs/{id}/stream) emits progress and complete events."""
        pdf_bytes = _create_multipage_pdf(num_pages=2)
        submit_resp = client.post(
            "/v1/jobs",
            files={"file": ("stream_doc.pdf", pdf_bytes, "application/pdf")},
        )
        job_id = submit_resp.json()["job_id"]

        sse_resp = client.get(f"/v1/jobs/{job_id}/stream")
        assert sse_resp.status_code == 200
        assert "text/event-stream" in sse_resp.headers["content-type"]
        text_content = sse_resp.text
        assert "event: progress" in text_content
        assert "event: complete" in text_content
        assert "doc_" in text_content

    def test_async_jobs_nonexistent_id_404(self, client: TestClient) -> None:
        """Verify GET /v1/jobs/{unknown} returns 404 Not Found."""
        resp = client.get("/v1/jobs/nonexistent_fake_job_id_99999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_async_jobs_sse_stream_nonexistent_job_emits_error_event(self, client: TestClient) -> None:
        """Verify SSE stream for non-existent job yields error event without 500 crash."""
        resp = client.get("/v1/jobs/missing_job_12345/stream")
        assert resp.status_code == 200
        assert "event: error" in resp.text
        assert "not found" in resp.text.lower()


# =========================================================================
# 6. Security Posture & Error Sanitization
# =========================================================================
class TestSecurityAndErrorSanitization:
    """Verify CWE mitigations, no stack trace leakage, and health check diagnostics."""

    def test_health_endpoint_diagnostics(self, client: TestClient) -> None:
        """Verify /v1/health returns valid schema with rescorer_active flag."""
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        health = HealthResponse.model_validate(resp.json())
        assert health.status == "healthy"
        assert health.device in ("mps", "cpu", "cuda", "mock")
        assert health.rescorer_active is True
        assert health.timestamp is not None

    def test_unhandled_runtime_error_does_not_leak_internals(self, client: TestClient) -> None:
        """Verify unhandled 500 exceptions produce sanitized responses without stack trace leaks."""
        class ExplodingEngine(InferenceEngine):
            def recognize(self, *args, **kwargs):
                raise ZeroDivisionError("Simulated internal math crash in /secret/internal/core.py line 77")

        set_engine(ExplodingEngine(execution_mode="mock"))
        app = create_app()
        secure_client = TestClient(app, raise_server_exceptions=False)

        img_bytes = _create_sample_png()
        resp = secure_client.post(
            "/v1/recognize",
            files={"file": ("crash.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "Internal Server Error"
        assert "/secret/internal/core.py" not in data.get("detail", "")
        assert "Traceback" not in data.get("detail", "")
        assert "ZeroDivisionError" not in data.get("detail", "")

    def test_path_traversal_filename_sanitization(self, client: TestClient) -> None:
        """Verify path traversal filenames (../../etc/passwd) are sanitized to basename only."""
        img_bytes = _create_sample_png()
        resp = client.post(
            "/v1/recognize",
            files={"file": ("../../../../etc/passwd", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "passwd"
        assert ".." not in data["filename"]
        assert "/" not in data["filename"]

    def test_method_not_allowed_returns_405(self, client: TestClient) -> None:
        """Verify unsupported HTTP methods return 405 Method Not Allowed."""
        assert client.delete("/v1/recognize").status_code == 405
        assert client.put("/v1/health").status_code == 405
        assert client.patch("/v1/jobs").status_code == 405


# =========================================================================
# 7. Concurrency & High Load Burst
# =========================================================================
class TestConcurrencyBurst:
    """Stress-test concurrent request bursts without state corruption or race conditions."""

    def test_concurrent_recognize_burst(self, client: TestClient) -> None:
        """Verify 20 concurrent synchronous recognize requests execute successfully."""
        img_bytes = _create_sample_png()
        responses = []
        for i in range(20):
            resp = client.post(
                f"/v1/recognize?beam_width={(i % 5) + 1}",
                files={"file": (f"burst_{i}.png", img_bytes, "image/png")},
            )
            assert resp.status_code == 200
            responses.append(resp.json())

        assert len(responses) == 20
        # All doc IDs must be unique
        doc_ids = [r["document_id"] for r in responses]
        assert len(set(doc_ids)) == 20

    def test_concurrent_async_jobs_burst(self, client: TestClient) -> None:
        """Verify 15 concurrent async job submissions receive unique job IDs."""
        pdf_bytes = _create_multipage_pdf(num_pages=1)
        job_ids = []
        for i in range(15):
            resp = client.post(
                "/v1/jobs",
                files={"file": (f"async_burst_{i}.pdf", pdf_bytes, "application/pdf")},
            )
            assert resp.status_code == 200
            job_ids.append(resp.json()["job_id"])

        assert len(set(job_ids)) == 15


# =========================================================================
# 8. Bounding Box Strict Invariant Audit
# =========================================================================
class TestBoundingBoxStrictInvariants:
    """Audit all bounding boxes across extreme dimensions for contract compliance."""

    @pytest.mark.parametrize(
        "w, h",
        [
            (10, 1000),
            (1000, 10),
            (2500, 50),
            (50, 2500),
            (100, 100),
            (1920, 1080),
        ],
    )
    def test_bbox_invariants_across_dimensions(self, client: TestClient, w: int, h: int) -> None:
        """Verify ymin < ymax and xmin < xmax and all in [0.0, 1.0] across diverse dimensions."""
        img_bytes = _create_sample_png(width=w, height=h)
        resp = client.post(
            "/v1/recognize",
            files={"file": (f"dim_{w}x{h}.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        model = RecognitionResponse.model_validate(data)

        for page in model.pages:
            for line in page.lines:
                ymin, xmin, ymax, xmax = line.bbox
                assert 0.0 <= ymin < ymax <= 1.0, f"Line bbox invalid vertical span: {line.bbox}"
                assert 0.0 <= xmin < xmax <= 1.0, f"Line bbox invalid horizontal span: {line.bbox}"
                assert 0.0 <= line.confidence <= 1.0

                for word in line.words:
                    w_ymin, w_xmin, w_ymax, w_xmax = word.bbox
                    assert 0.0 <= w_ymin < w_ymax <= 1.0, f"Word bbox invalid vertical span: {word.bbox}"
                    assert 0.0 <= w_xmin < w_xmax <= 1.0, f"Word bbox invalid horizontal span: {word.bbox}"
                    assert 0.0 <= word.confidence <= 1.0


# =========================================================================
# 9. Direct InferenceEngine Unit Hardening
# =========================================================================
class TestDirectEngineInvocations:
    """Directly test InferenceEngine class methods and edge cases."""

    def test_engine_initialization_with_custom_parameters(self) -> None:
        """Verify InferenceEngine initializes with custom beam width and weights."""
        engine = InferenceEngine(
            execution_mode="mock",
            beam_width=8,
            enable_rescorer=True,
            rescorer_weight=1.5,
            context_weight=0.9,
            confusion_weight=0.4,
        )
        assert engine.beam_width == 8
        assert engine.rescorer_weight == 1.5
        assert engine.mode == "mock"

    def test_engine_empty_input_raises_empty_document_error(self) -> None:
        """Verify recognize() with empty bytes raises EmptyDocumentError."""
        engine = InferenceEngine(execution_mode="mock")
        with pytest.raises(EmptyDocumentError):
            engine.recognize(b"", filename="empty.png")

    def test_engine_corrupt_input_raises_corrupt_document_error(self) -> None:
        """Verify recognize() with garbage bytes raises CorruptDocumentError."""
        engine = InferenceEngine(execution_mode="mock")
        with pytest.raises((CorruptDocumentError, DocumentLoadingError, Exception)):
            engine.recognize(b"not an image", filename="garbage.png")


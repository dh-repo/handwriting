#!/usr/bin/env python3
"""
tests/e2e/test_empirical_m5_stress.py
Empirical Adversarial Stress Harness for Milestone 5 Verification.
Executes deep stress testing on:
1. Extreme aspect ratios (1x10000, 10000x1, 1x20000, 20000x1, 1x1, 2x5000)
2. Corrupted, truncated, and malicious PDF streams
3. 100-request concurrent bursts across /v1/recognize, /v1/jobs, and SSE streams
4. Information disclosure audits: zero unhandled exceptions, zero 500 errors, zero leaked stack traces.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
import numpy as np
from PIL import Image, ImageDraw
import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["USE_MOCK_ENGINE"] = "true"

from backend.app.config import get_settings, reset_settings_cache
from backend.app.engine import InferenceEngine, reset_engine
from backend.app.main import create_app
from backend.app.schemas import JobStatusEnum, RecognitionResponse
from pipeline.preprocessing.image_enhancement import (
    adaptive_binarize,
    deskew_image,
    enhance_contrast,
    flatten_illumination,
    normalize_image,
    pad_to_size,
)
from pipeline.preprocessing.line_segmenter import LineSegmenter
from pipeline.preprocessing.pdf_loader import (
    CorruptDocumentError,
    DocumentLoadingError,
    EmptyDocumentError,
    PDFLoader,
    UnsupportedFormatError,
    load_document,
    load_image,
    load_pdf,
)
from pipeline.preprocessing.pipeline import PreprocessedPage, PreprocessingPipeline
from tests.fixtures.mock_engine import MockInferenceEngine, create_mock_app


@pytest.fixture(autouse=True)
def setup_test_env():
    os.environ["USE_MOCK_ENGINE"] = "true"
    reset_settings_cache()
    reset_engine()
    yield
    reset_settings_cache()
    reset_engine()


@pytest.fixture
def app_client() -> TestClient:
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def create_sample_png_bytes(w: int = 300, h: int = 150, text: str = "Empirical Stress Test") -> bytes:
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((15, 20), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# 1. Extreme Aspect Ratios & Degenerate Geometries
# ===========================================================================
class TestExtremeAspectRatiosEmpirical:
    """Stress tests extreme aspect ratio arrays, needle/banner images, and single pixels."""

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 10000, 3),   # 1 x 10,000 extreme horizontal strip
            (10000, 1, 3),   # 10,000 x 1 extreme vertical needle
            (1, 20000, 3),   # 1 x 20,000 ultra-wide
            (20000, 1, 3),   # 20,000 x 1 ultra-tall
            (2, 5000, 3),    # 2 x 5,000 thin banner
            (5000, 2, 3),    # 5,000 x 2 thin column
            (1, 1, 3),       # 1 x 1 single pixel
            (2, 2, 3),       # 2 x 2 minimal block
        ],
    )
    def test_pipeline_extreme_aspect_ratios(self, shape):
        """Verify preprocessing pipeline does not crash with cv2/numpy dimension errors."""
        arr = np.full(shape, 255, dtype=np.uint8)
        arr[0, 0] = [0, 0, 0]

        pipeline = PreprocessingPipeline(
            deskew=True,
            enhance_contrast=True,
            binarization_method="sauvola",
            seam_carving=True,
        )

        # Must execute cleanly without unhandled exceptions
        page = pipeline.process_image(arr)
        assert isinstance(page, PreprocessedPage)
        assert page.original_image.shape == shape
        assert page.binarized_image.shape == shape[:2]
        assert isinstance(page.lines, list)
        assert not np.isnan(page.enhanced_image).any()

    @pytest.mark.parametrize("shape", [(1, 10000), (10000, 1), (1, 1)])
    def test_api_extreme_aspect_ratios_png_upload(self, app_client: TestClient, shape):
        """Upload extreme aspect ratio PNGs to FastAPI /v1/recognize endpoint."""
        h, w = shape
        img = Image.new("RGB", (w, h), color=(255, 255, 255))
        img.putpixel((0, 0), (0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        resp = app_client.post(
            "/v1/recognize",
            files={"file": (f"extreme_{w}x{h}.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"].startswith("doc_")
        assert len(data["pages"]) == 1
        assert "Traceback" not in resp.text
        assert "site-packages" not in resp.text


# ===========================================================================
# 2. Corrupted & Hostile PDF Streams
# ===========================================================================
class TestCorruptedPdfStreamsEmpirical:
    """Stress tests damaged, truncated, malformed, and non-PDF payloads."""

    MALFORMED_PDF_PAYLOADS = [
        ("empty.pdf", b"", "Empty 0-byte payload"),
        ("truncated_header.pdf", b"%PDF-1.7", "Truncated header without trailer"),
        ("garbage_bytes.pdf", os.urandom(2048), "Random binary noise"),
        ("fake_pdf_elf.pdf", b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00", "ELF binary disguised as PDF"),
        ("fake_pdf_macho.pdf", b"\xfe\xed\xfa\xce\x00\x00\x00\x00", "Mach-O binary disguised as PDF"),
        ("fake_pdf_pe.pdf", b"MZ\x90\x00\x03\x00\x00\x00", "PE binary disguised as PDF"),
        ("corrupted_xref.pdf", b"%PDF-1.4\nxref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 >>\nstartxref\n99999\n%%EOF", "Broken xref table"),
        ("malformed_stream.pdf", b"%PDF-1.4\n1 0 obj\n<< /Length 100 >>\nstream\nTRUNCATED_STREAM\nendobj\n%%EOF", "Truncated stream object"),
        ("html_as_pdf.pdf", b"<!DOCTYPE html><html><body><h1>Exploit</h1></body></html>", "HTML payload disguised as PDF"),
        ("shell_script.pdf", b"#!/bin/bash\necho 'attacking system'\nrm -rf /", "Shell script disguised as PDF"),
    ]

    @pytest.mark.parametrize("filename, payload, desc", MALFORMED_PDF_PAYLOADS)
    def test_pdf_loader_direct_rejection(self, filename, payload, desc):
        """Verify PDFLoader and load_document raise handled DocumentLoadingError."""
        with pytest.raises((DocumentLoadingError, EmptyDocumentError, CorruptDocumentError, UnsupportedFormatError)):
            load_document(payload)

    @pytest.mark.parametrize("filename, payload, desc", MALFORMED_PDF_PAYLOADS)
    def test_api_recognize_corrupted_pdf_rejection(self, app_client: TestClient, filename, payload, desc):
        """Verify /v1/recognize returns clean 422 with zero 500s or stack traces."""
        resp = app_client.post(
            "/v1/recognize",
            files={"file": (filename, payload, "application/pdf")},
        )
        assert resp.status_code == 422, f"Failed on {desc}: got {resp.status_code}"
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert "detail" in body or "error" in body

        # Strict security assertions: zero information leakage
        assert "Traceback" not in resp.text
        assert "site-packages" not in resp.text
        assert "line " not in resp.text or "File \"" not in resp.text
        assert "Exception" not in body.get("error", "")

    @pytest.mark.parametrize("filename, payload, desc", MALFORMED_PDF_PAYLOADS)
    def test_api_jobs_corrupted_pdf_rejection_or_failure_state(self, app_client: TestClient, filename, payload, desc):
        """Verify /v1/jobs handles corrupt uploads either by 422 or state=FAILED."""
        resp = app_client.post(
            "/v1/jobs",
            files={"file": (filename, payload, "application/pdf")},
        )
        # Should either reject immediately with 422 or accept and transition job to FAILED
        if resp.status_code == 422:
            assert "Traceback" not in resp.text
        elif resp.status_code == 200:
            job_id = resp.json()["job_id"]
            # Poll status twice to allow background store to advance to terminal state
            poll1 = app_client.get(f"/v1/jobs/{job_id}")
            assert poll1.status_code == 200
            poll2 = app_client.get(f"/v1/jobs/{job_id}")
            assert poll2.status_code == 200
            poll_data = poll2.json()
            assert poll_data["status"] in ("FAILED", "PROCESSING", "QUEUED")
            if poll_data["status"] == "FAILED":
                assert poll_data.get("error") is not None
                assert "Traceback" not in poll_data.get("error", "")


# ===========================================================================
# 3. 100-Request Concurrent Bursts & Race Conditions
# ===========================================================================
class TestConcurrentBurstsEmpirical:
    """Stress tests backend under 100-request high concurrency bursts."""

    @pytest.mark.asyncio
    async def test_100_concurrent_recognize_requests(self):
        """Fire 100 simultaneous POST /v1/recognize requests."""
        engine = MockInferenceEngine()
        app = create_mock_app(engine)
        png_bytes = create_sample_png_bytes(300, 150, "Concurrent Burst 100")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            t0 = time.time()
            tasks = [
                client.post(
                    "/v1/recognize",
                    files={"file": (f"req_{i:03d}.png", png_bytes, "image/png")},
                )
                for i in range(100)
            ]
            responses = await asyncio.gather(*tasks)
            elapsed = time.time() - t0

            # 1. Assert all 100 requests returned HTTP 200
            status_codes = [r.status_code for r in responses]
            assert all(code == 200 for code in status_codes), f"Status code breakdown: {set(status_codes)}"

            # 2. Assert all 100 document IDs are strictly unique
            doc_ids = [r.json()["document_id"] for r in responses]
            assert len(set(doc_ids)) == 100, f"Collisions detected: {100 - len(set(doc_ids))}"

            # 3. Assert zero leaked traces across all 100 responses
            for r in responses:
                assert "Traceback" not in r.text
                assert "site-packages" not in r.text

            # 4. Concurrency throughput assertion (< 5 seconds for 100 mock requests)
            assert elapsed < 5.0, f"100 concurrent requests took too long: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_100_concurrent_jobs_submission_and_polling(self):
        """Fire 100 simultaneous POST /v1/jobs requests and poll each job."""
        engine = MockInferenceEngine()
        app = create_mock_app(engine)
        png_bytes = create_sample_png_bytes(300, 150, "Job Burst 100")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Submit 100 jobs concurrently
            t0 = time.time()
            submit_tasks = [
                client.post(
                    "/v1/jobs",
                    files={"file": (f"job_{i:03d}.png", png_bytes, "image/png")},
                )
                for i in range(100)
            ]
            submit_resps = await asyncio.gather(*submit_tasks)
            submit_elapsed = time.time() - t0

            assert all(r.status_code == 200 for r in submit_resps)
            job_ids = [r.json()["job_id"] for r in submit_resps]
            assert len(set(job_ids)) == 100, "Duplicate job_ids returned during concurrent submission"

            # 2. Poll all 100 jobs concurrently
            poll_tasks = [client.get(f"/v1/jobs/{jid}") for jid in job_ids]
            poll_resps = await asyncio.gather(*poll_tasks)

            assert all(r.status_code == 200 for r in poll_resps)
            for r in poll_resps:
                data = r.json()
                assert data["job_id"] in job_ids
                assert data["status"] in ("QUEUED", "PROCESSING", "COMPLETED", "FAILED")
                assert "Traceback" not in r.text

    @pytest.mark.asyncio
    async def test_100_concurrent_sse_event_connections(self):
        """Fire 100 concurrent SSE subscription streams across jobs."""
        engine = MockInferenceEngine()
        app = create_mock_app(engine)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Connect to 100 distinct job SSE streams (including non-existent and valid IDs)
            sse_tasks = [
                client.get(f"/v1/jobs/job_concurrent_sse_{i:03d}/events")
                for i in range(100)
            ]
            sse_resps = await asyncio.gather(*sse_tasks)

            assert all(r.status_code == 200 for r in sse_resps)
            for r in sse_resps:
                assert r.headers["content-type"].startswith("text/event-stream")
                assert "event: " in r.text
                assert "Traceback" not in r.text

    @pytest.mark.asyncio
    async def test_mixed_adversarial_concurrent_burst_50_valid_50_hostile(self):
        """Fire 50 valid requests and 50 hostile/corrupt requests simultaneously."""
        engine = MockInferenceEngine()
        app = create_mock_app(engine)
        valid_bytes = create_sample_png_bytes(200, 100, "Valid note")
        hostile_payloads = [
            (f"hostile_{i}.bin", os.urandom(512), "application/octet-stream")
            for i in range(50)
        ]

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            valid_tasks = [
                client.post("/v1/recognize", files={"file": (f"valid_{i}.png", valid_bytes, "image/png")})
                for i in range(50)
            ]
            hostile_tasks = [
                client.post("/v1/recognize", files={"file": (name, payload, ctype)})
                for name, payload, ctype in hostile_payloads
            ]

            all_tasks = valid_tasks + hostile_tasks
            results = await asyncio.gather(*all_tasks)

            valid_results = results[:50]
            hostile_results = results[50:]

            # Valid requests should all succeed
            assert all(r.status_code == 200 for r in valid_results)

            # Hostile requests should all be rejected cleanly with 422 (never 500)
            assert all(r.status_code == 422 for r in hostile_results)

            # Audit zero stack traces leaked in either stream
            for r in results:
                assert "Traceback" not in r.text
                assert "site-packages" not in r.text
                assert r.status_code != 500


# ===========================================================================
# 4. Zero Unhandled Exception / Stack Trace Audit Across Error Routes
# ===========================================================================
class TestSecurityAndLeakageAuditEmpirical:
    """Verifies complete absence of sensitive internals across various hostile routes."""

    @pytest.mark.parametrize(
        "path, method, json_payload",
        [
            ("/v1/nonexistent_route", "get", None),
            ("/v1/jobs/../../../../etc/passwd", "get", None),
            ("/v1/jobs/' UNION SELECT * FROM users--", "get", None),
            ("/v1/recognize", "post", {"file_base64": "invalid_base64!@#$"}),
            ("/v1/recognize", "post", {"file_base64": "", "filename": "test.png"}),
            ("/v1/recognize", "delete", None),
        ],
    )
    def test_clean_error_responses(self, app_client: TestClient, path, method, json_payload):
        """Verify errors return well-structured JSON without stack trace leakage."""
        if method == "get":
            resp = app_client.get(path)
        elif method == "post":
            resp = app_client.post(path, json=json_payload)
        elif method == "delete":
            resp = app_client.delete(path)

        assert resp.status_code in (404, 405, 422)
        assert resp.status_code != 500
        assert "Traceback" not in resp.text
        assert "site-packages" not in resp.text
        assert "File \"" not in resp.text
        assert "Traceback (most recent call last)" not in resp.text

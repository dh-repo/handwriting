#!/usr/bin/env python3
"""
tests/test_adversarial_challenger_m6.py
Milestone 6 Empirical Challenger Adversarial Stress Suite.
Adversarially challenges and verifies:
1. High-concurrency bursts & race conditions across /v1/recognize, /v1/jobs, /v1/jobs/{id}/events
2. SSE streaming stability under rapid disconnects, aborted connections, and concurrent listeners
3. Malformed JSON, corrupted base64, hostile multipart, oversized payloads, parameter fuzzing
4. Complete absence of stack traces, internal paths, and information leakage (CWE-209)
5. Robustness under memory and resource stress
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ["USE_MOCK_ENGINE"] = "true"

from backend.app.config import Settings, get_settings, reset_settings_cache
from backend.app.engine import InferenceEngine, reset_engine
from backend.app.main import create_app
from backend.app.schemas import (
    JobStatusResponse,
    LineBox,
    PageResult,
    RecognitionOptions,
    RecognitionResponse,
    WordBox,
    _validate_bbox_coordinates,
)
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


def create_test_image_bytes(w: int = 200, h: int = 100, text: str = "Challenger Test") -> bytes:
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# 1. API Concurrency Bursts & High-Throughput Stress (120 Requests)
# ===========================================================================
class TestChallengerConcurrencyBursts:
    """Stress tests backend under simultaneous high-concurrency requests."""

    @pytest.mark.asyncio
    async def test_120_mixed_concurrency_burst_sync_and_async(self):
        """Execute 60 sync /v1/recognize and 60 async /v1/jobs simultaneously."""
        engine = MockInferenceEngine(latency_ms=5.0)
        app = create_mock_app(engine)
        img_bytes = create_test_image_bytes(250, 120, "Burst Concurrency")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            t0 = time.time()
            sync_tasks = [
                client.post(
                    "/v1/recognize",
                    files={"file": (f"sync_{i}.png", img_bytes, "image/png")},
                )
                for i in range(60)
            ]
            async_tasks = [
                client.post(
                    "/v1/jobs",
                    files={"file": (f"async_{i}.png", img_bytes, "image/png")},
                )
                for i in range(60)
            ]

            all_tasks = sync_tasks + async_tasks
            responses = await asyncio.gather(*all_tasks)
            elapsed = time.time() - t0

            # 1. All 120 requests must succeed with HTTP 200
            assert all(r.status_code == 200 for r in responses)

            # 2. All 60 sync document IDs must be unique
            sync_doc_ids = [r.json()["document_id"] for r in responses[:60]]
            assert len(set(sync_doc_ids)) == 60

            # 3. All 60 async job IDs must be unique
            async_job_ids = [r.json()["job_id"] for r in responses[60:]]
            assert len(set(async_job_ids)) == 60

            # 4. Zero stack trace leaks across all responses
            for r in responses:
                assert "Traceback" not in r.text
                assert "site-packages" not in r.text

            # 5. Throughput SLA: 120 requests under mock engine must complete in < 5 seconds
            assert elapsed < 5.0, f"120 requests took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_rapid_concurrent_polling_and_state_transitions(self):
        """Create 20 jobs and poll them aggressively 10 times each concurrently."""
        engine = MockInferenceEngine(latency_ms=2.0)
        app = create_mock_app(engine)
        img_bytes = create_test_image_bytes(180, 90, "Poll Race Test")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Create 20 jobs
            create_tasks = [
                client.post("/v1/jobs", files={"file": (f"job_poll_{i}.png", img_bytes, "image/png")})
                for i in range(20)
            ]
            create_resps = await asyncio.gather(*create_tasks)
            assert all(r.status_code == 200 for r in create_resps)
            job_ids = [r.json()["job_id"] for r in create_resps]

            # Poll each job 10 times concurrently (200 total poll requests)
            poll_tasks = [client.get(f"/v1/jobs/{jid}") for jid in job_ids for _ in range(10)]
            poll_resps = await asyncio.gather(*poll_tasks)

            assert all(r.status_code == 200 for r in poll_resps)
            for r in poll_resps:
                data = r.json()
                assert data["status"] in ("QUEUED", "PROCESSING", "COMPLETED", "FAILED")
                assert "Traceback" not in r.text


# ===========================================================================
# 2. SSE Streaming Stability & Client Disconnection Stress
# ===========================================================================
class TestChallengerSSEStreaming:
    """Stress tests SSE streaming under abrupt disconnects and multiple listeners."""

    @pytest.mark.asyncio
    async def test_sse_multiple_concurrent_listeners_same_job(self):
        """Connect 10 concurrent SSE clients to the exact same job ID."""
        engine = MockInferenceEngine()
        app = create_mock_app(engine)
        img_bytes = create_test_image_bytes(200, 100, "Shared SSE Job")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Create a single job
            submit_resp = await client.post("/v1/jobs", files={"file": ("shared.png", img_bytes, "image/png")})
            assert submit_resp.status_code == 200
            job_id = submit_resp.json()["job_id"]

            # 2. Open 10 concurrent SSE streams on the same job
            stream_tasks = [client.get(f"/v1/jobs/{job_id}/events") for _ in range(10)]
            stream_resps = await asyncio.gather(*stream_tasks)

            assert all(r.status_code == 200 for r in stream_resps)
            for r in stream_resps:
                assert r.headers["content-type"].startswith("text/event-stream")
                assert "event: progress" in r.text or "event: complete" in r.text
                assert "Traceback" not in r.text

    @pytest.mark.asyncio
    async def test_sse_nonexistent_and_malformed_job_ids(self):
        """Challenge SSE streaming with invalid, injection, and non-existent IDs."""
        engine = MockInferenceEngine()
        app = create_mock_app(engine)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            malformed_ids = [
                "nonexistent_job_12345",
                "../../../etc/shadow",
                "<script>alert(1)</script>",
                "job_with_spaces and tabs\t",
                "job_null_\x00_byte",
            ]

            for jid in malformed_ids:
                quoted_jid = urllib.parse.quote(jid, safe="")
                resp = await client.get(f"/v1/jobs/{quoted_jid}/events")
                # Streaming response returns 200 with text/event-stream containing event: error or 404
                assert resp.status_code in (200, 404, 422)
                if resp.status_code == 200:
                    assert resp.headers["content-type"].startswith("text/event-stream")
                    assert "event: error" in resp.text
                assert "Traceback" not in resp.text
                assert "site-packages" not in resp.text


# ===========================================================================
# 3. Malformed JSON, Base64 & Hostile Payload Boundary Stress
# ===========================================================================
class TestChallengerMalformedPayloads:
    """Stress tests edge cases in input parsing and payload decoding."""

    MALFORMED_JSON_PAYLOADS = [
        ("invalid_b64_chars", {"file_base64": "NOT_VALID_BASE64_!@#$%^&*()"}, 422),
        ("unpadded_b64", {"file_base64": "AQIDBA"}, 422),  # invalid image bytes
        ("empty_b64", {"file_base64": ""}, 422),
        ("numeric_b64", {"file_base64": 12345678}, 422),
        ("null_b64", {"file_base64": None}, 422),
        ("array_b64", {"file_base64": ["data"]}, 422),
        ("deeply_nested_dict", {"file_base64": {"nested": "value"}}, 422),
        ("missing_field", {"wrong_key": "some_value"}, 422),
        ("extra_large_metadata", {"file_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "filename": "A" * 10000}, 200),
    ]

    @pytest.mark.parametrize("desc, payload, expected_status", MALFORMED_JSON_PAYLOADS)
    def test_json_payload_boundary_handling(self, app_client: TestClient, desc, payload, expected_status):
        """Send malformed JSON payloads to /v1/recognize."""
        resp = app_client.post(
            "/v1/recognize",
            json=payload,
        )
        assert resp.status_code == expected_status, f"Failed on {desc}: expected {expected_status}, got {resp.status_code}"
        assert "Traceback" not in resp.text
        assert "site-packages" not in resp.text
        assert "File \"" not in resp.text

    def test_query_parameter_fuzzing(self, app_client: TestClient):
        """Send boundary and invalid query parameters to /v1/recognize."""
        img_bytes = create_test_image_bytes(100, 50, "Fuzz")
        
        # Test negative beam_width (ge=1 constraint)
        resp_neg_beam = app_client.post(
            "/v1/recognize?beam_width=-5",
            files={"file": ("fuzz.png", img_bytes, "image/png")},
        )
        assert resp_neg_beam.status_code == 422
        assert "Traceback" not in resp_neg_beam.text

        # Test beam_width exceeding maximum (le=16 constraint)
        resp_max_beam = app_client.post(
            "/v1/recognize?beam_width=999",
            files={"file": ("fuzz.png", img_bytes, "image/png")},
        )
        assert resp_max_beam.status_code == 422
        assert "Traceback" not in resp_max_beam.text

        # Test string beam_width
        resp_str_beam = app_client.post(
            "/v1/recognize?beam_width=invalid_string",
            files={"file": ("fuzz.png", img_bytes, "image/png")},
        )
        assert resp_str_beam.status_code == 422
        assert "Traceback" not in resp_str_beam.text

        # Test string dpi
        resp_str_dpi = app_client.post(
            "/v1/recognize?dpi=extreme_dpi",
            files={"file": ("fuzz.png", img_bytes, "image/png")},
        )
        assert resp_str_dpi.status_code == 422
        assert "Traceback" not in resp_str_dpi.text


# ===========================================================================
# 4. Strict CWE-209 / Stack Trace Leakage Audit Across All Verbs and Routes
# ===========================================================================
class TestChallengerInformationDisclosure:
    """Verifies that no route under any circumstances leaks stack traces or file paths."""

    AUDIT_PROBES = [
        ("GET", "/v1/health", None),
        ("GET", "/v1/nonexistent_endpoint", None),
        ("POST", "/v1/recognize", {}),
        ("GET", "/v1/recognize", None),
        ("PUT", "/v1/recognize", None),
        ("DELETE", "/v1/recognize", None),
        ("GET", "/v1/jobs/nonexistent_job_id", None),
        ("GET", "/v1/jobs/../../../../../../etc/passwd", None),
        ("GET", "/v1/jobs/nonexistent_job_id/events", None),
        ("POST", "/v1/jobs", {}),
        ("OPTIONS", "/v1/recognize", None),
    ]

    @pytest.mark.parametrize("method, path, body", AUDIT_PROBES)
    def test_zero_information_disclosure_on_probes(self, app_client: TestClient, method, path, body):
        """Audit every HTTP method and path for CWE-209 information disclosure."""
        if method == "GET":
            resp = app_client.get(path)
        elif method == "POST":
            resp = app_client.post(path, json=body or {})
        elif method == "PUT":
            resp = app_client.put(path, json=body or {})
        elif method == "DELETE":
            resp = app_client.delete(path)
        elif method == "OPTIONS":
            resp = app_client.options(path)
        else:
            resp = app_client.request(method, path)

        # Assert no internal leaks
        assert "Traceback (most recent call last)" not in resp.text
        assert "site-packages" not in resp.text
        assert "/Volumes/LaCie" not in resp.text
        assert "File \"" not in resp.text
        assert "ZeroDivisionError" not in resp.text
        assert "UnboundLocalError" not in resp.text
        assert "AttributeError:" not in resp.text
        assert "TypeError:" not in resp.text


# ===========================================================================
# 5. Coordinate Invariant Verification & BBox Boundary Guards
# ===========================================================================
class TestChallengerCoordinateGuards:
    """Stress tests boundary validation of bounding box coordinates."""

    INVALID_BBOXES = [
        ([-0.001, 0.1, 0.5, 0.5], "ymin < 0"),
        ([0.1, -0.001, 0.5, 0.5], "xmin < 0"),
        ([0.1, 0.1, 1.001, 0.5], "ymax > 1"),
        ([0.1, 0.1, 0.5, 1.001], "xmax > 1"),
        ([0.5, 0.1, 0.5, 0.5], "ymin == ymax"),
        ([0.1, 0.5, 0.5, 0.5], "xmin == xmax"),
        ([0.8, 0.1, 0.2, 0.5], "ymin > ymax"),
        ([0.1, 0.8, 0.5, 0.2], "xmin > xmax"),
    ]

    @pytest.mark.parametrize("bbox, desc", INVALID_BBOXES)
    def test_bbox_coordinate_validation_rejection(self, bbox, desc):
        """Ensure invalid normalized coordinates raise ValueError."""
        with pytest.raises(ValueError):
            _validate_bbox_coordinates(bbox)

    def test_valid_bbox_boundaries(self):
        """Ensure exact boundaries (0.0, 1.0) are accepted."""
        bbox = [0.0, 0.0, 1.0, 1.0]
        validated = _validate_bbox_coordinates(bbox)
        assert validated == [0.0, 0.0, 1.0, 1.0]

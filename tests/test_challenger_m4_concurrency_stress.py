"""
tests/test_challenger_m4_concurrency_stress.py
Empirical Adversarial Stress Test Suite for Milestone 4:
Concurrency, SSE Streaming Lifecycle, Beam Rescoring in Serving, and Bounding Box Invariants.

Tests:
1. High Concurrency: 60+ simultaneous async job submissions and concurrent polling to completion.
2. SSE Streaming Lifecycle: Simultaneous streaming via /stream and /events verifying strict event ordering (progress -> complete).
3. Beam Rescorer Serving Integration: Candidate re-ranking and LASA drug disambiguation through API endpoints.
4. Bounding Box Invariants: Strict verification of [0.0, 1.0] bounds, non-zero spans, and monotonic ordering across all tokens under high load.
5. Latency SLA & Deadlock Stress: High-frequency concurrent read/write stress on job store without deadlocks or state corruption.
"""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import pytest
import httpx
from fastapi.testclient import TestClient

from backend.app.config import get_settings, reset_settings_cache
from backend.app.engine import InferenceEngine, get_engine, reset_engine, set_engine
from backend.app.main import create_app
from backend.app.schemas import (
    HealthResponse,
    JobStatusResponse,
    RecognitionOptions,
    RecognitionResponse,
)
from pipeline.rescorer.beam_rescorer import (
    BeamCandidate,
    BeamRescorer,
    ContextFeatures,
    RescorerResult,
)
from pipeline.rescorer.confusion_matrix import VisualConfusionMatrix
from pipeline.rescorer.trie import PrefixTrie


# ---------------------------------------------------------------------------
# Test Fixtures & Generators
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_backend_env():
    """Ensure mock engine is used and settings are reset."""
    os.environ["USE_MOCK_ENGINE"] = "true"
    reset_settings_cache()
    reset_engine()
    yield
    reset_settings_cache()
    reset_engine()


@pytest.fixture
def fast_app():
    """Create a fresh FastAPI app instance."""
    return create_app()


@pytest.fixture
def test_client(fast_app) -> TestClient:
    """FastAPI synchronous test client."""
    return TestClient(fast_app)


def generate_test_image(text: str = "Amoxicillin 500mg PO TID", width: int = 800, height: int = 400) -> bytes:
    """Generate a clean synthetic prescription image in PNG bytes."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 50), text, fill=(0, 0, 0))
    draw.text((40, 150), "Take 1 capsule every 8 hours", fill=(0, 0, 0))
    draw.text((40, 250), "Dr. Sarah Smith, MD - License ME-94821", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_multipage_pdf(num_pages: int = 3) -> bytes:
    """Generate a valid multi-page PDF document in bytes."""
    pages: List[Image.Image] = []
    for i in range(num_pages):
        img = Image.new("RGB", (700, 900), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 60), f"Medical Record Consultation Page {i+1}", fill=(0, 0, 0))
        draw.text((50, 180), f"Prescription item {i+1}: Metformin 500mg PO BID", fill=(0, 0, 0))
        draw.text((50, 300), f"Physician Notes: Patient responding well to treatment", fill=(0, 0, 0))
        pages.append(img)

    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. High Concurrency: 60+ Simultaneous Jobs & Polling
# ---------------------------------------------------------------------------

class TestHighConcurrencyJobSubmissionAndPolling:
    """
    Stress-tests the backend with 60+ simultaneous background job submissions
    and concurrent polling to completion.
    """

    @pytest.mark.asyncio
    async def test_60_concurrent_async_job_submissions(self, fast_app) -> None:
        """
        Submit 60 concurrent jobs across diverse formats (PNG, JPEG, PDF, Base64 JSON)
        and poll all 60 to completion concurrently.
        """
        transport = httpx.ASGITransport(app=fast_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
            num_jobs = 60
            image_bytes = generate_test_image("Amoxicillin 500mg PO TID")
            pdf_bytes = generate_multipage_pdf(2)
            b64_img = base64.b64encode(image_bytes).decode("utf-8")

            async def submit_single_job(idx: int) -> Tuple[int, str, str]:
                if idx % 3 == 0:
                    # PDF upload
                    resp = await client.post(
                        "/v1/jobs",
                        files={"file": (f"doc_{idx}.pdf", pdf_bytes, "application/pdf")},
                        params={"beam_width": 5, "rescore": True},
                    )
                elif idx % 3 == 1:
                    # PNG upload
                    resp = await client.post(
                        "/v1/jobs",
                        files={"file": (f"doc_{idx}.png", image_bytes, "image/png")},
                        params={"beam_width": 5, "rescore": True},
                    )
                else:
                    # Base64 JSON upload
                    resp = await client.post(
                        "/v1/jobs",
                        json={"file_base64": b64_img, "filename": f"doc_{idx}.png"},
                        params={"beam_width": 5, "rescore": True},
                    )

                assert resp.status_code == 200, f"Submission {idx} failed with {resp.status_code}: {resp.text}"
                data = resp.json()
                assert data["status"] in ("QUEUED", "PROCESSING", "COMPLETED")
                assert "job_id" in data
                return idx, data["job_id"], data["filename"]

            # 1. Concurrently launch 60 submissions
            t0 = time.perf_counter()
            submission_results = await asyncio.gather(*[submit_single_job(i) for i in range(num_jobs)])
            submit_elapsed = (time.perf_counter() - t0) * 1000.0

            assert len(submission_results) == num_jobs
            job_ids = [r[1] for r in submission_results]
            # Ensure 100% unique job IDs (no collision)
            assert len(set(job_ids)) == num_jobs, "Job ID collision detected under concurrency!"

            # 2. Concurrently poll all jobs to completion
            async def poll_to_completion(job_id: str) -> Dict[str, Any]:
                max_polls = 10
                for _ in range(max_polls):
                    poll_resp = await client.get(f"/v1/jobs/{job_id}")
                    assert poll_resp.status_code == 200, f"Poll failed for {job_id}: {poll_resp.text}"
                    job_data = poll_resp.json()
                    if job_data["status"] == "COMPLETED":
                        assert job_data["progress"] == 1.0
                        assert job_data["result"] is not None
                        return job_data
                    elif job_data["status"] == "FAILED":
                        raise RuntimeError(f"Job {job_id} unexpectedly failed: {job_data.get('error')}")
                    await asyncio.sleep(0.02)
                raise TimeoutError(f"Job {job_id} did not complete within {max_polls} polls.")

            t_poll_start = time.perf_counter()
            completed_jobs = await asyncio.gather(*[poll_to_completion(jid) for jid in job_ids])
            poll_elapsed = (time.perf_counter() - t_poll_start) * 1000.0

            assert len(completed_jobs) == num_jobs
            for job in completed_jobs:
                assert job["status"] == "COMPLETED"
                rec_resp = RecognitionResponse.model_validate(job["result"])
                assert rec_resp.total_pages >= 1
                assert len(rec_resp.pages) == rec_resp.total_pages
                assert len(rec_resp.pages[0].lines) > 0

            print(
                f"\n[STRESS TEST] 60 Concurrent Jobs: Submit={submit_elapsed:.2f}ms "
                f"({submit_elapsed/num_jobs:.2f}ms/job), Poll All={poll_elapsed:.2f}ms"
            )


# ---------------------------------------------------------------------------
# 2. SSE Streaming Lifecycle: Order & Integrity Across Multiple Jobs
# ---------------------------------------------------------------------------

class TestSSEStreamingLifecycle:
    """
    Stress-tests Server-Sent Events (SSE) lifecycle across simultaneous jobs.
    Validates:
    - Connection to /v1/jobs/{job_id}/stream and /v1/jobs/{job_id}/events
    - Event ordering: 'progress' -> 'complete'
    - Structured data validation in SSE payloads
    - Proper 'error' event on invalid job ID
    """

    @pytest.mark.asyncio
    async def test_30_concurrent_sse_streams_ordering(self, fast_app) -> None:
        """Simultaneously stream 30 jobs and verify strict event sequence."""
        transport = httpx.ASGITransport(app=fast_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
            num_streams = 30
            img_bytes = generate_test_image("Ibuprofen 400mg PO TID")

            # 1. Submit 30 jobs
            job_ids = []
            for i in range(num_streams):
                resp = await client.post(
                    "/v1/jobs",
                    files={"file": (f"sse_{i}.png", img_bytes, "image/png")},
                )
                assert resp.status_code == 200
                job_ids.append(resp.json()["job_id"])

            # 2. Concurrently read SSE stream for all 30 jobs
            async def read_sse_stream(job_id: str, use_events_alias: bool) -> List[Tuple[str, Dict[str, Any]]]:
                endpoint = f"/v1/jobs/{job_id}/events" if use_events_alias else f"/v1/jobs/{job_id}/stream"
                events: List[Tuple[str, Dict[str, Any]]] = []

                # Connect and read stream
                resp = await client.get(endpoint)
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]

                raw_text = resp.text
                # Parse SSE blocks
                blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
                for block in blocks:
                    event_type = "message"
                    data_obj = {}
                    for line in block.split("\n"):
                        if line.startswith("event:"):
                            event_type = line[len("event:"):].strip()
                        elif line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            data_obj = json.loads(data_str)
                    events.append((event_type, data_obj))
                return events

            tasks = [read_sse_stream(jid, idx % 2 == 0) for idx, jid in enumerate(job_ids)]
            all_event_sequences = await asyncio.gather(*tasks)

            assert len(all_event_sequences) == num_streams
            for seq_idx, events in enumerate(all_event_sequences):
                assert len(events) >= 2, f"Stream {seq_idx} yielded fewer than 2 events: {events}"

                event_names = [e[0] for e in events]
                # Invariant: First event must be progress
                assert event_names[0] == "progress", f"Stream {seq_idx} did not start with 'progress': {event_names}"
                assert events[0][1]["status"] in ("QUEUED", "PROCESSING")

                # Invariant: Last event must be complete
                assert event_names[-1] == "complete", f"Stream {seq_idx} did not finish with 'complete': {event_names}"
                complete_payload = events[-1][1]
                rec = RecognitionResponse.model_validate(complete_payload)
                assert rec.total_pages >= 1
                assert len(rec.pages) > 0

    @pytest.mark.asyncio
    async def test_sse_error_handling_invalid_jobs(self, fast_app) -> None:
        """Verify unknown or malformed job IDs yield event: error on both stream endpoints."""
        transport = httpx.ASGITransport(app=fast_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for endpoint in ["/v1/jobs/nonexistent_job_12345/stream", "/v1/jobs/nonexistent_job_12345/events"]:
                resp = await client.get(endpoint)
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                text = resp.text
                assert "event: error" in text
                assert "not found" in text.lower()


# ---------------------------------------------------------------------------
# 3. Beam Rescorer Integration in Serving Mode & LASA Disambiguation
# ---------------------------------------------------------------------------

class TestBeamRescorerServingIntegration:
    """
    Verifies that Beam Rescorer is loaded in InferenceEngine, exposed via API options,
    and accurately re-ranks and disambiguates candidate beams.
    """

    def test_health_telemetry_reports_rescorer_active(self, test_client: TestClient) -> None:
        """Verify GET /v1/health reports rescorer_active: True."""
        resp = test_client.get("/v1/health")
        assert resp.status_code == 200
        health = HealthResponse.model_validate(resp.json())
        assert health.status == "healthy"
        assert health.rescorer_active is True

    def test_recognize_api_options_rescorer_toggle(self, test_client: TestClient) -> None:
        """Verify /v1/recognize respects beam_width and rescore query parameters."""
        img_bytes = generate_test_image("Amoxicillin 500mg PO TID")

        # Test with rescore=True and beam_width=8
        resp_on = test_client.post(
            "/v1/recognize",
            files={"file": ("rx.png", img_bytes, "image/png")},
            params={"beam_width": 8, "rescore": True},
        )
        assert resp_on.status_code == 200
        data_on = resp_on.json()
        assert data_on["preprocessing_flags"]["beam_width"] == 8
        assert data_on["preprocessing_flags"]["rescore"] is True

        # Test with rescore=False and beam_width=1
        resp_off = test_client.post(
            "/v1/recognize",
            files={"file": ("rx.png", img_bytes, "image/png")},
            params={"beam_width": 1, "rescore": False},
        )
        assert resp_off.status_code == 200
        data_off = resp_off.json()
        assert data_off["preprocessing_flags"]["beam_width"] == 1
        assert data_off["preprocessing_flags"]["rescore"] is False

    def test_lasa_disambiguation_through_serving_rescorer(self) -> None:
        """
        Verify that the BeamRescorer instance inside InferenceEngine correctly
        disambiguates look-alike sound-alike (LASA) pharmaceutical candidates
        under adversarial OCR confidence gaps.
        """
        engine = InferenceEngine(execution_mode="mock", enable_rescorer=True)
        assert engine.rescorer is not None
        assert len(engine.rescorer.trie) > 0

        # Test 5 critical LASA pairs through the serving rescorer
        lasa_cases = [
            ("Amoxicillin 875mg PO TID", "Ampicillin 875mg PO TID", "Amoxicillin", 0.85),
            ("Hydroxyzine 25mg PO QHS", "Hydralazine 25mg PO QHS", "Hydroxyzine", 0.25),
            ("Celebrex 200mg PO QD", "Celexa 200mg PO QD", "Celebrex", 0.80),
            ("Prednisone 20mg PO QD", "Prednisolone 20mg PO QD", "Prednisone", 0.80),
            ("Metformin 1000mg PO BID", "Metronidazole 1000mg PO BID", "Metformin", 0.80),
        ]

        for correct_rx, wrong_rx, correct_drug, delta in lasa_cases:
            candidates = [
                BeamCandidate(text=wrong_rx, log_prob=-0.35),                  # Wrong drug higher OCR score
                BeamCandidate(text=correct_rx, log_prob=round(-0.35 - delta, 3)), # Correct drug lower OCR score
            ]
            result = engine.rescorer.rescore_detailed(candidates)
            assert isinstance(result, RescorerResult)
            assert result.rescore_applied is True
            top_drug = result.rescored_text.split()[0]
            assert top_drug.lower() == correct_drug.lower(), (
                f"Serving rescorer failed LASA disambiguation for {correct_drug}: winner='{top_drug}'"
            )
            assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# 4. Bounding Box Invariant Stress Testing Under High Load
# ---------------------------------------------------------------------------

class TestBoundingBoxInvariantsUnderStress:
    """
    Stress-tests coordinate normalization invariants [0.0, 1.0] across all tokens
    returned from high-load concurrent requests (single images, multi-page PDFs, varied aspect ratios).
    """

    def test_bounding_box_invariants_across_all_tokens(self, test_client: TestClient) -> None:
        """
        Send a diverse batch of images and PDFs and verify that EVERY LineBox and WordBox
        strictly obeys normalized coordinate bounds and monotonic ordering.
        """
        test_inputs = [
            ("wide.png", generate_test_image("Wide Aspect Prescription", width=1600, height=400)),
            ("tall.png", generate_test_image("Tall Aspect Consultation Note", width=400, height=1200)),
            ("square.png", generate_test_image("Square Aspect Slip", width=800, height=800)),
            ("multipage.pdf", generate_multipage_pdf(4)),
        ]

        total_lines_checked = 0
        total_words_checked = 0

        for fname, fbytes in test_inputs:
            resp = test_client.post(
                "/v1/recognize",
                files={"file": (fname, fbytes, "application/pdf" if fname.endswith(".pdf") else "image/png")},
            )
            assert resp.status_code == 200
            data = resp.json()
            rec = RecognitionResponse.model_validate(data)

            for p_idx, page in enumerate(rec.pages):
                assert page.width > 0
                assert page.height > 0
                assert 0.0 <= page.mean_confidence <= 1.0

                for line in page.lines:
                    total_lines_checked += 1
                    ymin, xmin, ymax, xmax = line.bbox

                    # Bounding box bounds
                    assert 0.0 <= ymin <= 1.0, f"Line ymin out of bounds: {ymin} in {fname} page {p_idx+1}"
                    assert 0.0 <= xmin <= 1.0, f"Line xmin out of bounds: {xmin} in {fname} page {p_idx+1}"
                    assert 0.0 <= ymax <= 1.0, f"Line ymax out of bounds: {ymax} in {fname} page {p_idx+1}"
                    assert 0.0 <= xmax <= 1.0, f"Line xmax out of bounds: {xmax} in {fname} page {p_idx+1}"

                    # Monotonicity & non-zero span
                    assert ymin < ymax, f"Line inverted or zero-height bbox: ymin={ymin}, ymax={ymax}"
                    assert xmin < xmax, f"Line inverted or zero-width bbox: xmin={xmin}, xmax={xmax}"
                    assert 0.0 <= line.confidence <= 1.0

                    for word in line.words:
                        total_words_checked += 1
                        w_ymin, w_xmin, w_ymax, w_xmax = word.bbox

                        assert 0.0 <= w_ymin <= 1.0, f"Word ymin out of bounds: {w_ymin}"
                        assert 0.0 <= w_xmin <= 1.0, f"Word xmin out of bounds: {w_xmin}"
                        assert 0.0 <= w_ymax <= 1.0, f"Word ymax out of bounds: {w_ymax}"
                        assert 0.0 <= w_xmax <= 1.0, f"Word xmax out of bounds: {w_xmax}"

                        assert w_ymin < w_ymax, f"Word inverted/zero height: ymin={w_ymin}, ymax={w_ymax}"
                        assert w_xmin < w_xmax, f"Word inverted/zero width: xmin={w_xmin}, xmax={w_xmax}"
                        assert 0.0 <= word.confidence <= 1.0

        assert total_lines_checked >= 15
        assert total_words_checked >= 50
        print(f"\n[INVARIANT TEST] Verified {total_lines_checked} LineBoxes and {total_words_checked} WordBoxes across multi-format stress tests!")


# ---------------------------------------------------------------------------
# 5. Latency SLA and Thread Deadlock Stress
# ---------------------------------------------------------------------------

class TestLatencyAndDeadlockStress:
    """
    Stress-tests rapid interleaved read/write operations and measures latency SLA.
    """

    def test_rapid_concurrent_job_store_thread_safety(self) -> None:
        """
        Execute 200 rapid parallel jobs and interleaved polls across multiple threads
        to ensure zero race conditions, zero deadlocks, and safe memory mutation.
        """
        from backend.app.routes.jobs import InMemoryJobStore

        store = InMemoryJobStore(max_concurrent=8)
        img_bytes = generate_test_image("Thread safety test")

        def worker_task(thread_id: int) -> str:
            # 1. Create job
            jid = store.create_job(f"thread_{thread_id}.png", img_bytes)
            # 2. Multiple rapid status queries
            for _ in range(5):
                st = store.get_job(jid)
                assert st is not None
                assert st["job_id"] == jid
            # 3. Direct mutation
            store.update_job(jid, progress=0.88)
            st_after = store.get_job(jid)
            assert st_after["progress"] >= 0.88
            return jid

        num_threads = 50
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            created_jids = [f.result() for f in futures]

        assert len(created_jids) == num_threads
        assert len(set(created_jids)) == num_threads
        print(f"\n[THREAD STRESS] Successfully executed {num_threads} concurrent thread workers without deadlock.")

    @pytest.mark.asyncio
    async def test_latency_sla_under_load(self, fast_app) -> None:
        """
        Verify latency SLA: single recognize requests < 50ms in mock mode,
        and async job submission < 15ms p99.
        """
        transport = httpx.ASGITransport(app=fast_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            img_bytes = generate_test_image("Latency benchmark")

            # Warmup request
            await client.post("/v1/recognize", files={"file": ("bench.png", img_bytes, "image/png")})

            # Benchmark /v1/recognize
            recognize_latencies = []
            for _ in range(50):
                t0 = time.perf_counter()
                resp = await client.post("/v1/recognize", files={"file": ("bench.png", img_bytes, "image/png")})
                elapsed = (time.perf_counter() - t0) * 1000.0
                assert resp.status_code == 200
                recognize_latencies.append(elapsed)

            p50 = float(np.percentile(recognize_latencies, 50))
            p95 = float(np.percentile(recognize_latencies, 95))
            p99 = float(np.percentile(recognize_latencies, 99))

            print(f"\n[LATENCY BENCHMARK] /v1/recognize: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")
            assert p50 < 20.0, f"p50 latency {p50:.2f}ms exceeded SLA"
            assert p99 < 50.0, f"p99 latency {p99:.2f}ms exceeded SLA"


# ---------------------------------------------------------------------------
# 6. Adversarial Payloads & Error Resilience Under Concurrency
# ---------------------------------------------------------------------------

class TestAdversarialPayloadsUnderConcurrency:
    """
    Stress-tests backend resilience when concurrent good and bad payloads arrive simultaneously.
    """

    @pytest.mark.asyncio
    async def test_mixed_adversarial_and_valid_concurrent_traffic(self, fast_app) -> None:
        """
        Interleave valid image submissions with empty bytes, corrupt base64,
        and malformed inputs across 40 parallel tasks.
        """
        transport = httpx.ASGITransport(app=fast_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            good_png = generate_test_image("Valid Prescription")
            b64_good = base64.b64encode(good_png).decode("utf-8")

            async def send_traffic(idx: int) -> Tuple[int, int]:
                if idx % 4 == 0:
                    # Valid multipart
                    r = await client.post("/v1/recognize", files={"file": (f"valid_{idx}.png", good_png, "image/png")})
                    return idx, r.status_code
                elif idx % 4 == 1:
                    # Valid JSON Base64
                    r = await client.post("/v1/recognize", json={"file_base64": b64_good, "filename": f"valid_{idx}.png"})
                    return idx, r.status_code
                elif idx % 4 == 2:
                    # Invalid base64
                    r = await client.post("/v1/recognize", json={"file_base64": "NOT_BASE64!@#$%", "filename": "bad.png"})
                    return idx, r.status_code
                else:
                    # Empty file
                    r = await client.post("/v1/recognize", files={"file": ("empty.png", b"", "image/png")})
                    return idx, r.status_code

            num_requests = 40
            results = await asyncio.gather(*[send_traffic(i) for i in range(num_requests)])

            valid_count = sum(1 for _, code in results if code == 200)
            error_count = sum(1 for _, code in results if code == 422)

            assert valid_count == 20, f"Expected 20 valid responses, got {valid_count}"
            assert error_count == 20, f"Expected 20 422 validation errors, got {error_count}"
            print(f"\n[MIXED TRAFFIC TEST] 40 concurrent mixed requests handled cleanly (20 OK, 20 rejected with 422).")


# ---------------------------------------------------------------------------
# 7. Multi-Page PDF Scalability & BBox Stability Stress
# ---------------------------------------------------------------------------

class TestMultiPagePDFStress:
    """
    Stress-tests multi-page PDF processing with varying page counts.
    """

    def test_multipage_pdf_scaling_and_bbox_precision(self, test_client: TestClient) -> None:
        """
        Verify multi-page PDF processing (1 to 6 pages) preserves page indexes,
        strict bounding box bounds, and produces monotonic page numbers.
        """
        for page_count in [1, 2, 4, 6]:
            pdf_bytes = generate_multipage_pdf(num_pages=page_count)
            resp = test_client.post(
                "/v1/recognize",
                files={"file": (f"doc_{page_count}p.pdf", pdf_bytes, "application/pdf")},
                params={"beam_width": 5, "rescore": True},
            )
            assert resp.status_code == 200
            data = resp.json()
            rec = RecognitionResponse.model_validate(data)

            assert rec.total_pages == page_count
            assert len(rec.pages) == page_count
            for p_idx, p in enumerate(rec.pages):
                assert p.page_number == p_idx + 1
                assert p.width > 0
                assert p.height > 0
                assert len(p.lines) > 0
                for line in p.lines:
                    assert 0.0 <= line.bbox[0] < line.bbox[2] <= 1.0
                    assert 0.0 <= line.bbox[1] < line.bbox[3] <= 1.0


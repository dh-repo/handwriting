"""
backend/tests/test_jobs.py
Integration tests for async background job submission, status polling, and SSE streaming.
"""

import base64
import time
from fastapi.testclient import TestClient

from backend.app.schemas import JobStatusResponse, RecognitionResponse


def test_submit_job_returns_queued(client: TestClient, sample_image_bytes: bytes) -> None:
    """Verify POST /v1/jobs creates a job with status QUEUED."""
    resp = client.post(
        "/v1/jobs",
        files={"file": ("doc.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "QUEUED"
    assert "job_id" in data
    assert data["job_id"].startswith("job_")


def test_job_polling_progression_to_completion(client: TestClient, sample_image_bytes: bytes) -> None:
    """Verify polling /v1/jobs/{id} progresses from QUEUED/PROCESSING to COMPLETED."""
    post_resp = client.post(
        "/v1/jobs",
        files={"file": ("doc.png", sample_image_bytes, "image/png")},
    )
    job_id = post_resp.json()["job_id"]

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        s3 = client.get(f"/v1/jobs/{job_id}").json()
        if s3["status"] in ("COMPLETED", "FAILED"): break
        time.sleep(0.01)
    assert s3["status"] == "COMPLETED"
    assert s3["progress"] == 1.0

    rec_res = RecognitionResponse.model_validate(s3["result"])
    assert rec_res.total_pages == 1


def test_job_sse_streaming(client: TestClient, sample_image_bytes: bytes) -> None:
    """Verify GET /v1/jobs/{id}/stream streams progress and complete events."""
    post_resp = client.post(
        "/v1/jobs",
        files={"file": ("doc.png", sample_image_bytes, "image/png")},
    )
    job_id = post_resp.json()["job_id"]

    stream_resp = client.get(f"/v1/jobs/{job_id}/stream")
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    text = stream_resp.text
    assert "event: complete" in text


def test_job_sse_events_alias(client: TestClient, sample_image_bytes: bytes) -> None:
    """Verify GET /v1/jobs/{id}/events alias route works."""
    post_resp = client.post(
        "/v1/jobs",
        files={"file": ("doc.png", sample_image_bytes, "image/png")},
    )
    job_id = post_resp.json()["job_id"]

    stream_resp = client.get(f"/v1/jobs/{job_id}/events")
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    assert "event: complete" in stream_resp.text


def test_sse_stream_nonexistent_job(client: TestClient) -> None:
    """Verify SSE streaming for unknown job yields event: error."""
    resp = client.get("/v1/jobs/job_missing_123/stream")
    assert resp.status_code == 200
    assert "event: error" in resp.text


def test_multiple_concurrent_jobs_isolation(client: TestClient, sample_image_bytes: bytes, sample_jpeg_bytes: bytes) -> None:
    """Verify multiple jobs maintain isolated states and unique IDs."""
    j1 = client.post("/v1/jobs", files={"file": ("f1.png", sample_image_bytes, "image/png")}).json()["job_id"]
    j2 = client.post("/v1/jobs", files={"file": ("f2.jpg", sample_jpeg_bytes, "image/jpeg")}).json()["job_id"]

    assert j1 != j2
    st1 = client.get(f"/v1/jobs/{j1}").json()
    st2 = client.get(f"/v1/jobs/{j2}").json()

    assert st1["job_id"] == j1
    assert st2["job_id"] == j2


def test_submit_job_base64_json(client: TestClient, sample_image_bytes: bytes) -> None:
    """Verify POST /v1/jobs with application/json base64 payload."""
    b64_str = base64.b64encode(sample_image_bytes).decode("utf-8")
    payload = {
        "file_base64": b64_str,
        "filename": "json_job.png",
    }
    resp = client.post("/v1/jobs", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "QUEUED"
    assert data["filename"] == "json_job.png"

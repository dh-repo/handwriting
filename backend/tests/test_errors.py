"""
backend/tests/test_errors.py
Error handling, payload validation, and sanitization boundary tests.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.app.engine import InferenceEngine, set_engine
from backend.app.main import create_app



def test_recognize_missing_file(client: TestClient) -> None:
    """Verify POST /v1/recognize without file returns 422."""
    resp = client.post("/v1/recognize")
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data or "detail" in data


def test_recognize_empty_file_returns_422(client: TestClient) -> None:
    """Verify POST /v1/recognize with 0-byte file returns 422."""
    resp = client.post(
        "/v1/recognize",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower() or "invalid" in resp.json()["detail"].lower()


def test_recognize_corrupted_image_returns_422(client: TestClient, corrupted_image_bytes: bytes) -> None:
    """Verify POST /v1/recognize with corrupted image returns 422."""
    resp = client.post(
        "/v1/recognize",
        files={"file": ("corrupted.png", corrupted_image_bytes, "image/png")},
    )
    assert resp.status_code == 422


def test_recognize_corrupted_pdf_returns_422(client: TestClient, corrupted_pdf_bytes: bytes) -> None:
    """Verify POST /v1/recognize with corrupted PDF returns 422."""
    resp = client.post(
        "/v1/recognize",
        files={"file": ("corrupted.pdf", corrupted_pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 422


def test_jobs_empty_file_returns_422(client: TestClient) -> None:
    """Verify POST /v1/jobs with empty payload returns 422."""
    resp = client.post(
        "/v1/jobs",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 422


def test_jobs_nonexistent_id_returns_404(client: TestClient) -> None:
    """Verify GET /v1/jobs/{unknown} returns 404."""
    resp = client.get("/v1/jobs/nonexistent_uuid_99999")
    assert resp.status_code == 404
    data = resp.json()
    assert "not found" in data["detail"].lower()


def test_unhandled_exception_sanitization(sample_image_bytes: bytes) -> None:
    """Verify unhandled internal errors return sanitized 500 JSON without stack trace."""
    class FaultyEngine(InferenceEngine):
        def recognize(self, *args, **kwargs):
            raise RuntimeError("Database connection timed out at line 42 in /secret/path/db.py")

    set_engine(FaultyEngine(execution_mode="mock"))
    app = create_app()
    err_client = TestClient(app, raise_server_exceptions=False)

    resp = err_client.post(
        "/v1/recognize",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 503
    data = resp.json()
    assert "error" in data
    # Ensure raw secret path / stack trace is not exposed
    assert "/secret/path/db.py" not in data.get("detail", "")


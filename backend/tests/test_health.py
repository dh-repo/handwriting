"""
backend/tests/test_health.py
Integration tests for /v1/health endpoint and OpenAPI spec generation.
"""

from fastapi.testclient import TestClient


def test_live_endpoint_200(client: TestClient) -> None:
    """Verify GET /v1/live is cheap and does not depend on model payload fields."""
    resp = client.get("/v1/live")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "live"
    assert "timestamp" in data
    assert "loaded_models" not in data


def test_health_endpoint_200(client: TestClient) -> None:
    """Verify GET /v1/health returns 200 OK with expected schema."""
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "healthy"
    assert "device" in data
    assert "version" in data
    assert "timestamp" in data
    assert "memory_usage_mb" in data
    assert "loaded_models" in data
    assert isinstance(data["loaded_models"], list)
    assert "rescorer_active" in data
    assert data["rescorer_active"] is False


def test_openapi_spec_paths(client: TestClient) -> None:
    """Verify FastAPI generates complete OpenAPI specification including /v1 routes."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()

    assert "paths" in spec
    assert "/v1/live" in spec["paths"]
    assert "/v1/health" in spec["paths"]
    assert "/v1/recognize" in spec["paths"]
    assert "/v1/jobs" in spec["paths"]
    assert "/v1/jobs/{job_id}" in spec["paths"]
    assert "/v1/jobs/{job_id}/stream" in spec["paths"]

"""
backend/tests/test_feedback_route.py
Comprehensive unit, integration, and concurrency tests for /v1/feedback endpoint.
"""

from __future__ import annotations
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, get_settings, reset_settings_cache
from backend.app.engine import get_engine, reset_engine, set_engine
from backend.app.main import create_app
from backend.app.schemas import (
    ConfusionUpdateRecord,
    FeedbackCorrectionPayload,
    FeedbackResponse,
    FeedbackStatsResponse,
)


@pytest.fixture
def feedback_env(tmp_path: Path) -> Generator[Dict[str, Path], None, None]:
    """Provide isolated paths for feedback manifest and crops directory."""
    fb_dir = tmp_path / "feedback_test"
    manifest_path = fb_dir / "manifest.jsonl"
    crops_dir = fb_dir / "crops"

    fb_dir.mkdir(parents=True, exist_ok=True)

    test_settings = Settings(
        FEEDBACK_DIR=str(fb_dir),
        FEEDBACK_MANIFEST_PATH=str(manifest_path),
        FEEDBACK_CROPS_DIR=str(crops_dir),
        CONFUSION_LEARNING_RATE=0.10,
        USE_MOCK_ENGINE=True,
    )

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings

    yield {
        "client": TestClient(app),
        "fb_dir": fb_dir,
        "manifest_path": manifest_path,
        "crops_dir": crops_dir,
        "settings": test_settings,
    }

    app.dependency_overrides.clear()
    reset_settings_cache()
    reset_engine()


@pytest.fixture
def valid_png_b64() -> str:
    """Generate a small valid PNG in raw base64."""
    img = Image.new("RGB", (120, 40), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def valid_jpeg_data_url() -> str:
    """Generate a valid JPEG formatted as a Data URL."""
    img = Image.new("RGB", (100, 30), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{raw_b64}"


# ---------------------------------------------------------------------------
# 1. Configuration & Schemas Unit Tests
# ---------------------------------------------------------------------------

def test_feedback_config_defaults() -> None:
    """Verify default feedback configuration values."""
    reset_settings_cache()
    s = get_settings()
    assert s.FEEDBACK_DIR == "data/feedback"
    assert s.FEEDBACK_MANIFEST_PATH == "data/feedback/manifest.jsonl"
    assert s.FEEDBACK_CROPS_DIR == "data/feedback/crops"
    assert s.CONFUSION_LEARNING_RATE == 0.10


def test_feedback_config_env_overrides() -> None:
    """Verify feedback settings accept environment variable overrides."""
    reset_settings_cache()
    with patch.dict(os.environ, {
        "FEEDBACK_DIR": "/tmp/custom_feedback",
        "FEEDBACK_MANIFEST_PATH": "/tmp/custom_feedback/custom_manifest.jsonl",
        "FEEDBACK_CROPS_DIR": "/tmp/custom_feedback/custom_crops",
        "CONFUSION_LEARNING_RATE": "0.25",
    }):
        reset_settings_cache()
        s = get_settings()
        assert s.FEEDBACK_DIR == "/tmp/custom_feedback"
        assert s.FEEDBACK_MANIFEST_PATH == "/tmp/custom_feedback/custom_manifest.jsonl"
        assert s.FEEDBACK_CROPS_DIR == "/tmp/custom_feedback/custom_crops"
        assert s.CONFUSION_LEARNING_RATE == 0.25


def test_feedback_schemas_validation() -> None:
    """Verify Pydantic validation on FeedbackCorrectionPayload."""
    # Valid payload
    valid_payload = FeedbackCorrectionPayload(
        document_id="doc_001",
        line_id="line_001",
        page_number=2,
        original_prediction="cydindamycin",
        operator_correction="clindamycin",
        confidence=0.85,
        bbox=[0.1, 0.2, 0.3, 0.8],
        word_id="w_1",
    )
    assert valid_payload.document_id == "doc_001"
    assert valid_payload.page_number == 2
    assert valid_payload.sync_confusion_matrix is True

    # Empty document_id rejected
    with pytest.raises(ValueError, match="document_id"):
        FeedbackCorrectionPayload(
            document_id="   ",
            line_id="line_001",
            original_prediction="a",
            operator_correction="b",
        )

    # Empty line_id rejected
    with pytest.raises(ValueError, match="line_id"):
        FeedbackCorrectionPayload(
            document_id="doc_001",
            line_id="",
            original_prediction="a",
            operator_correction="b",
        )

    # Invalid confidence rejected (< 0 or > 1)
    with pytest.raises(ValueError, match="confidence"):
        FeedbackCorrectionPayload(
            document_id="doc_001",
            line_id="line_001",
            original_prediction="a",
            operator_correction="b",
            confidence=1.5,
        )

    with pytest.raises(ValueError, match="confidence"):
        FeedbackCorrectionPayload(
            document_id="doc_001",
            line_id="line_001",
            original_prediction="a",
            operator_correction="b",
            confidence=-0.1,
        )

    # Invalid bbox length
    with pytest.raises(ValueError, match="exactly 4 coordinates"):
        FeedbackCorrectionPayload(
            document_id="doc_001",
            line_id="line_001",
            original_prediction="a",
            operator_correction="b",
            bbox=[0.1, 0.2, 0.3],
        )

    # Inverted vertical span
    with pytest.raises(ValueError, match="invalid vertical span"):
        FeedbackCorrectionPayload(
            document_id="doc_001",
            line_id="line_001",
            original_prediction="a",
            operator_correction="b",
            bbox=[0.5, 0.1, 0.2, 0.9],
        )


def test_confusion_update_record_schema() -> None:
    """Verify ConfusionUpdateRecord data contract."""
    record = ConfusionUpdateRecord(
        operation="substitution",
        source="yd",
        target="nd",
        previous_cost=1.20,
        updated_cost=0.45,
    )
    assert record.operation == "substitution"
    assert record.source == "yd"
    assert record.target == "nd"
    assert record.previous_cost == 1.20
    assert record.updated_cost == 0.45


# ---------------------------------------------------------------------------
# 2. Integration Tests: POST /v1/feedback
# ---------------------------------------------------------------------------

def test_submit_feedback_happy_path_with_crop(
    feedback_env: Dict[str, Any],
    valid_png_b64: str,
) -> None:
    """Submit correction with raw base64 line crop; verify persistence to manifest and disk."""
    client: TestClient = feedback_env["client"]
    manifest_path: Path = feedback_env["manifest_path"]
    crops_dir: Path = feedback_env["crops_dir"]

    payload = {
        "document_id": "doc_rx101",
        "line_id": "p1_l3",
        "page_number": 1,
        "word_id": "p1_l3_w2",
        "original_prediction": "prednlsone 10mg",
        "operator_correction": "prednisone 10mg",
        "confidence": 0.72,
        "bbox": [0.15, 0.10, 0.22, 0.75],
        "line_crop_base64": valid_png_b64,
        "sync_confusion_matrix": True,
    }

    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert "feedback_id" in data
    assert re.match(r"^fb_\d{8}_\d{6}_[0-9a-f]{8}$", data["feedback_id"])
    assert data["document_id"] == "doc_rx101"
    assert data["line_id"] == "p1_l3"
    assert data["status"] == "persisted"
    assert data["manifest_path"] == str(manifest_path)
    assert data["crop_path"] is not None
    assert "timestamp" in data

    # Verify manifest file exists and has 1 record
    assert manifest_path.exists()
    lines = manifest_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["feedback_id"] == data["feedback_id"]
    assert record["document_id"] == "doc_rx101"
    assert record["line_id"] == "p1_l3"
    assert record["page_number"] == 1
    assert record["word_id"] == "p1_l3_w2"
    assert record["original_prediction"] == "prednlsone 10mg"
    assert record["operator_correction"] == "prednisone 10mg"
    assert record["confidence"] == 0.72
    assert record["bbox"] == [0.15, 0.10, 0.22, 0.75]
    assert record["image_crop_path"] == data["crop_path"]

    # Verify crop image exists on disk and is a valid PNG
    crop_file = Path(data["crop_path"])
    assert crop_file.exists()
    assert crop_file.parent == crops_dir
    with Image.open(crop_file) as img:
        assert img.format == "PNG"
        assert img.size == (120, 40)


def test_submit_feedback_data_url_format(
    feedback_env: Dict[str, Any],
    valid_jpeg_data_url: str,
) -> None:
    """Submit correction with Data URL (data:image/jpeg;base64,...), saved as PNG crop."""
    client: TestClient = feedback_env["client"]
    manifest_path: Path = feedback_env["manifest_path"]

    payload = {
        "document_id": "doc_data_url",
        "line_id": "l_1",
        "original_prediction": "amox",
        "operator_correction": "amoxicillin",
        "line_crop_base64": valid_jpeg_data_url,
    }

    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["crop_path"] is not None
    crop_file = Path(data["crop_path"])
    assert crop_file.exists()
    with Image.open(crop_file) as img:
        assert img.format == "PNG"
        assert img.size == (100, 30)

    # Manifest record has correct image_crop_path
    record = json.loads(manifest_path.read_text().strip())
    assert record["image_crop_path"] == data["crop_path"]


def test_submit_feedback_without_crop(feedback_env: Dict[str, Any]) -> None:
    """Submit correction without crop image; crop_path is None in response and manifest."""
    client: TestClient = feedback_env["client"]
    manifest_path: Path = feedback_env["manifest_path"]

    payload = {
        "document_id": "doc_no_crop",
        "line_id": "line_99",
        "original_prediction": "lipitor 20mg",
        "operator_correction": "Lipitor 20mg",
        "confidence": 0.95,
        "line_crop_base64": None,
    }

    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["crop_path"] is None
    record = json.loads(manifest_path.read_text().strip())
    assert record["image_crop_path"] is None


# ---------------------------------------------------------------------------
# 3. Validation & Error Handling Tests (422)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_field,bad_payload",
    [
        ("missing document_id", {"line_id": "l1", "original_prediction": "a", "operator_correction": "b"}),
        ("empty document_id", {"document_id": "  ", "line_id": "l1", "original_prediction": "a", "operator_correction": "b"}),
        ("missing line_id", {"document_id": "d1", "original_prediction": "a", "operator_correction": "b"}),
        ("empty line_id", {"document_id": "d1", "line_id": "", "original_prediction": "a", "operator_correction": "b"}),
        ("missing original_prediction", {"document_id": "d1", "line_id": "l1", "operator_correction": "b"}),
        ("missing operator_correction", {"document_id": "d1", "line_id": "l1", "original_prediction": "a"}),
        ("confidence < 0", {"document_id": "d1", "line_id": "l1", "original_prediction": "a", "operator_correction": "b", "confidence": -0.5}),
        ("confidence > 1", {"document_id": "d1", "line_id": "l1", "original_prediction": "a", "operator_correction": "b", "confidence": 1.05}),
        ("page_number < 1", {"document_id": "d1", "line_id": "l1", "original_prediction": "a", "operator_correction": "b", "page_number": 0}),
        ("bbox invalid length", {"document_id": "d1", "line_id": "l1", "original_prediction": "a", "operator_correction": "b", "bbox": [0.1, 0.2]}),
        ("bbox inverted y", {"document_id": "d1", "line_id": "l1", "original_prediction": "a", "operator_correction": "b", "bbox": [0.8, 0.1, 0.2, 0.9]}),
        ("bbox inverted x", {"document_id": "d1", "line_id": "l1", "original_prediction": "a", "operator_correction": "b", "bbox": [0.1, 0.9, 0.8, 0.2]}),
        ("bbox out of bounds", {"document_id": "d1", "line_id": "l1", "original_prediction": "a", "operator_correction": "b", "bbox": [0.1, 0.2, 1.5, 0.8]}),
        ("invalid base64", {"document_id": "d1", "line_id": "l1", "original_prediction": "a", "operator_correction": "b", "line_crop_base64": "!!!not_b64"}),
        ("empty crop base64", {"document_id": "d1", "line_id": "l1", "original_prediction": "a", "operator_correction": "b", "line_crop_base64": "   "}),
        ("corrupted image base64", {"document_id": "d1", "line_id": "l1", "original_prediction": "a", "operator_correction": "b", "line_crop_base64": base64.b64encode(b"not an image file").decode("utf-8")}),
    ],
)
def test_submit_feedback_validation_errors(
    feedback_env: Dict[str, Any],
    bad_field: str,
    bad_payload: Dict[str, Any],
) -> None:
    """Verify invalid payloads are rejected with HTTP 422 Unprocessable Entity."""
    client: TestClient = feedback_env["client"]
    resp = client.post("/v1/feedback", json=bad_payload)
    assert resp.status_code == 422, f"Failed on {bad_field}: got {resp.status_code}, response: {resp.text}"


# ---------------------------------------------------------------------------
# 4. Dynamic Confusion Matrix Adaptation Hook Tests
# ---------------------------------------------------------------------------

def test_dynamic_confusion_hook_active(feedback_env: Dict[str, Any]) -> None:
    """When engine rescorer confusion_matrix has adapt_from_correction, updates are returned."""
    client: TestClient = feedback_env["client"]
    manifest_path: Path = feedback_env["manifest_path"]

    mock_cm = MagicMock()
    mock_cm.adapt_from_correction.return_value = [
        {
            "operation": "substitution",
            "source": "l",
            "target": "i",
            "previous_cost": 0.30,
            "updated_cost": 0.20,
        },
        {
            "operation": "contraction",
            "source": "rn",
            "target": "m",
            "previous_cost": 0.25,
            "updated_cost": 0.15,
        },
    ]

    mock_rescorer = MagicMock()
    mock_rescorer.confusion_matrix = mock_cm

    mock_engine = MagicMock()
    mock_engine.rescorer = mock_rescorer

    with patch("backend.app.routes.feedback.get_engine", return_value=mock_engine):
        payload = {
            "document_id": "doc_dyn",
            "line_id": "l_1",
            "original_prediction": "prednlsone",
            "operator_correction": "prednisone",
            "sync_confusion_matrix": True,
        }
        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["confusion_pairs_updated"]) == 2
        assert data["confusion_pairs_updated"][0]["source"] == "l"
        assert data["confusion_pairs_updated"][0]["target"] == "i"
        assert data["confusion_pairs_updated"][0]["updated_cost"] == 0.20
        assert data["confusion_pairs_updated"][1]["source"] == "rn"

        mock_cm.adapt_from_correction.assert_called_once_with(
            "prednlsone",
            "prednisone",
            learning_rate=0.10,
        )

        # Manifest includes alignment_operations
        record = json.loads(manifest_path.read_text().strip())
        assert len(record["alignment_operations"]) == 2


def test_dynamic_confusion_hook_disabled(feedback_env: Dict[str, Any]) -> None:
    """When sync_confusion_matrix=False, confusion matrix is not adapted."""
    client: TestClient = feedback_env["client"]
    mock_cm = MagicMock()
    mock_rescorer = MagicMock()
    mock_rescorer.confusion_matrix = mock_cm
    mock_engine = MagicMock()
    mock_engine.rescorer = mock_rescorer

    with patch("backend.app.routes.feedback.get_engine", return_value=mock_engine):
        payload = {
            "document_id": "doc_no_sync",
            "line_id": "l_1",
            "original_prediction": "metfomin",
            "operator_correction": "metformin",
            "sync_confusion_matrix": False,
        }
        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["confusion_pairs_updated"] == []
        mock_cm.adapt_from_correction.assert_not_called()


def test_dynamic_confusion_hook_missing_method_fallback(feedback_env: Dict[str, Any]) -> None:
    """When engine rescorer has no adapt_from_correction (M2 pending), gracefully returns empty list."""
    client: TestClient = feedback_env["client"]
    mock_engine = MagicMock()
    mock_engine.rescorer = MagicMock()
    # No confusion_matrix attribute or confusion_matrix has no adapt_from_correction
    mock_engine.rescorer.confusion_matrix = object()

    with patch("backend.app.routes.feedback.get_engine", return_value=mock_engine):
        payload = {
            "document_id": "doc_fallback",
            "line_id": "l_1",
            "original_prediction": "aspirn",
            "operator_correction": "aspirin",
            "sync_confusion_matrix": True,
        }
        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confusion_pairs_updated"] == []


# ---------------------------------------------------------------------------
# 5. Stats Endpoint: GET /v1/feedback/stats
# ---------------------------------------------------------------------------

def test_feedback_stats_endpoint(
    feedback_env: Dict[str, Any],
    valid_png_b64: str,
) -> None:
    """Verify /v1/feedback/stats accurately reports record count and crop count."""
    client: TestClient = feedback_env["client"]

    # Initial stats on clean directory
    s0 = client.get("/v1/feedback/stats")
    assert s0.status_code == 200
    d0 = s0.json()
    assert d0["total_records"] == 0
    assert d0["total_crops"] == 0

    # Submit 1 with crop
    client.post("/v1/feedback", json={
        "document_id": "d1",
        "line_id": "l1",
        "original_prediction": "a",
        "operator_correction": "b",
        "line_crop_base64": valid_png_b64,
    })

    # Submit 1 without crop
    client.post("/v1/feedback", json={
        "document_id": "d2",
        "line_id": "l2",
        "original_prediction": "x",
        "operator_correction": "y",
    })

    s1 = client.get("/v1/feedback/stats")
    assert s1.status_code == 200
    d1 = s1.json()
    assert d1["total_records"] == 2
    assert d1["total_crops"] == 1


# ---------------------------------------------------------------------------
# 6. Concurrency Stress Test: Process-Safe & Thread-Safe Locking
# ---------------------------------------------------------------------------

def test_feedback_concurrency_stress(
    feedback_env: Dict[str, Any],
    valid_png_b64: str,
) -> None:
    """
    Simulate multi-threaded concurrent Darkroom feedback submissions.
    Verify that manifest.jsonl records all N lines without corruption, data interleaving, or loss.
    """
    client: TestClient = feedback_env["client"]
    manifest_path: Path = feedback_env["manifest_path"]
    crops_dir: Path = feedback_env["crops_dir"]
    num_requests = 25

    def _submit(idx: int) -> Dict[str, Any]:
        payload = {
            "document_id": f"doc_concurrent_{idx}",
            "line_id": f"line_{idx}",
            "page_number": 1,
            "original_prediction": f"pred_{idx}",
            "operator_correction": f"corr_{idx}",
            "confidence": 0.5 + (idx % 50) / 100.0,
            "line_crop_base64": valid_png_b64,
        }
        res = client.post("/v1/feedback", json=payload)
        return {"status_code": res.status_code, "data": res.json()}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_submit, i) for i in range(num_requests)]
        results = [f.result() for f in futures]

    # Verify all requests succeeded
    for res in results:
        assert res["status_code"] == 200
        assert res["data"]["status"] == "persisted"

    # Verify manifest file has exactly num_requests valid JSON lines
    assert manifest_path.exists()
    raw_lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    assert len(raw_lines) == num_requests, f"Expected {num_requests} lines, got {len(raw_lines)}"

    seen_ids = set()
    for line in raw_lines:
        parsed = json.loads(line)
        fid = parsed["feedback_id"]
        assert fid not in seen_ids, f"Duplicate feedback_id found: {fid}"
        seen_ids.add(fid)
        assert Path(parsed["image_crop_path"]).exists()

    # Verify crop directory contains exactly num_requests files
    crop_files = list(crops_dir.glob("*.png"))
    assert len(crop_files) == num_requests


# ---------------------------------------------------------------------------
# 7. Frontend Darkroom Compatibility & DoS / Resource Limits Unit Tests
# ---------------------------------------------------------------------------

def test_submit_feedback_darkroom_frontend_compatibility(feedback_env: Dict[str, Any]) -> None:
    """
    Darkroom frontend payload compatibility:
    Verifies that requests using 'original_text' and 'corrected_text' return HTTP 200,
    echo 'document_id' and 'line_id' in response, and store correctly in manifest.
    """
    client: TestClient = feedback_env["client"]
    manifest_path: Path = feedback_env["manifest_path"]

    payload = {
        "document_id": "doc_darkroom_001",
        "line_id": "p1_l1",
        "original_text": "prednlsone 10mg",
        "corrected_text": "prednisone 10mg",
        "confidence": 0.88,
    }

    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["status"] == "persisted"
    assert data["document_id"] == "doc_darkroom_001"
    assert data["line_id"] == "p1_l1"

    # Verify manifest record has normalized original_prediction and operator_correction
    record = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert record["document_id"] == "doc_darkroom_001"
    assert record["line_id"] == "p1_l1"
    assert record["original_prediction"] == "prednlsone 10mg"
    assert record["operator_correction"] == "prednisone 10mg"


def test_submit_feedback_string_length_exceeds_500_rejected(feedback_env: Dict[str, Any]) -> None:
    """Strings exceeding max_length=500 in prediction or correction must return HTTP 422."""
    client: TestClient = feedback_env["client"]

    # 1. original_prediction > 500
    resp_pred = client.post("/v1/feedback", json={
        "document_id": "doc_toolong",
        "line_id": "l_1",
        "original_prediction": "a" * 501,
        "operator_correction": "valid",
    })
    assert resp_pred.status_code == 422

    # 2. operator_correction > 500
    resp_corr = client.post("/v1/feedback", json={
        "document_id": "doc_toolong",
        "line_id": "l_1",
        "original_prediction": "valid",
        "operator_correction": "b" * 501,
    })
    assert resp_corr.status_code == 422

    # 3. Frontend alias original_text > 500
    resp_alias_pred = client.post("/v1/feedback", json={
        "document_id": "doc_toolong",
        "line_id": "l_1",
        "original_text": "c" * 501,
        "corrected_text": "valid",
    })
    assert resp_alias_pred.status_code == 422

    # 4. Frontend alias corrected_text > 500
    resp_alias_corr = client.post("/v1/feedback", json={
        "document_id": "doc_toolong",
        "line_id": "l_1",
        "original_text": "valid",
        "corrected_text": "d" * 501,
    })
    assert resp_alias_corr.status_code == 422


def test_submit_feedback_crop_exceeds_5mb_rejected(feedback_env: Dict[str, Any]) -> None:
    """Decoded crop bytes exceeding 5MB limit must be rejected with HTTP 422."""
    client: TestClient = feedback_env["client"]
    oversized_bytes = b"0" * (5 * 1024 * 1024 + 100)
    oversized_b64 = base64.b64encode(oversized_bytes).decode("utf-8")

    payload = {
        "document_id": "doc_oversized",
        "line_id": "line_oversized",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "line_crop_base64": oversized_b64,
    }
    resp = client.post("/v1/feedback", json=payload)
    assert resp.status_code == 422
    assert "Line crop exceeds 5MB limit" in resp.json()["detail"]


def test_submit_feedback_crop_dimensions_exceed_bounds_rejected(feedback_env: Dict[str, Any]) -> None:
    """Line crops exceeding 4096 width or 2048 height must return HTTP 422."""
    client: TestClient = feedback_env["client"]

    # Exceed width: 4097 x 50
    img_wide = Image.new("L", (4097, 50), color=255)
    buf_wide = io.BytesIO()
    img_wide.save(buf_wide, format="PNG")
    wide_b64 = base64.b64encode(buf_wide.getvalue()).decode("utf-8")

    payload_wide = {
        "document_id": "doc_wide",
        "line_id": "line_wide",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "line_crop_base64": wide_b64,
    }
    resp_wide = client.post("/v1/feedback", json=payload_wide)
    assert resp_wide.status_code == 422
    assert "Line crop dimensions exceed maximum bounds" in resp_wide.json()["detail"]

    # Exceed height: 50 x 2049
    img_tall = Image.new("L", (50, 2049), color=255)
    buf_tall = io.BytesIO()
    img_tall.save(buf_tall, format="PNG")
    tall_b64 = base64.b64encode(buf_tall.getvalue()).decode("utf-8")

    payload_tall = {
        "document_id": "doc_tall",
        "line_id": "line_tall",
        "original_prediction": "pred",
        "operator_correction": "corr",
        "line_crop_base64": tall_b64,
    }
    resp_tall = client.post("/v1/feedback", json=payload_tall)
    assert resp_tall.status_code == 422
    assert "Line crop dimensions exceed maximum bounds" in resp_tall.json()["detail"]


def test_dynamic_confusion_hook_skipped_for_long_text(feedback_env: Dict[str, Any]) -> None:
    """When text length > 300, dynamic confusion update is skipped to protect event loop."""
    client: TestClient = feedback_env["client"]
    mock_cm = MagicMock()
    mock_rescorer = MagicMock()
    mock_rescorer.confusion_matrix = mock_cm
    mock_engine = MagicMock()
    mock_engine.rescorer = mock_rescorer

    with patch("backend.app.routes.feedback.get_engine", return_value=mock_engine):
        payload = {
            "document_id": "doc_long_text",
            "line_id": "line_long_text",
            "original_prediction": "a" * 350,
            "operator_correction": "b" * 350,
            "sync_confusion_matrix": True,
        }
        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confusion_pairs_updated"] == []
        mock_cm.adapt_from_correction.assert_not_called()


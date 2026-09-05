"""
tests/unit/test_azure_storage_feedback.py
Unit tests for Azure Blob Storage feedback sink and local fallback durability.
"""

from __future__ import annotations
import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import get_settings
from backend.app.main import app
from backend.app.routes.feedback import AzureBlobStorageSink, _save_line_crop, _append_to_manifest


@pytest.fixture
def sample_crop_b64() -> str:
    img = Image.new("RGB", (100, 30), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_azure_sink_unconfigured() -> None:
    sink = AzureBlobStorageSink()
    assert not sink.is_configured
    assert sink.get_client() is None
    assert sink.upload_crop("fb_1", b"fake_bytes") is None
    assert sink.append_manifest_record({"id": "fb_1"}) is None


def test_azure_sink_upload_crop_success() -> None:
    mock_blob_client = MagicMock()
    mock_blob_client.url = "https://myaccount.blob.core.windows.net/feedback-crops/crops/fb_123.png"

    mock_container_client = MagicMock()
    mock_container_client.get_blob_client.return_value = mock_blob_client

    mock_service_client = MagicMock()
    mock_service_client.get_container_client.return_value = mock_container_client

    with patch("backend.app.routes.feedback.BlobServiceClient") as mock_bsc:
        mock_bsc.from_connection_string.return_value = mock_service_client

        sink = AzureBlobStorageSink(
            connection_string="DefaultEndpointsProtocol=https;AccountName=myaccount;AccountKey=fake;EndpointSuffix=core.windows.net",
            container_crops="feedback-crops",
        )
        assert sink.is_configured

        url = sink.upload_crop("fb_123", b"test_png_bytes")
        assert url == "https://myaccount.blob.core.windows.net/feedback-crops/crops/fb_123.png"
        mock_container_client.get_blob_client.assert_called_once_with("crops/fb_123.png")
        mock_blob_client.upload_blob.assert_called_once_with(
            b"test_png_bytes", overwrite=True, content_type="image/png"
        )


def test_azure_sink_upload_crop_error_fallback(sample_crop_b64: str, tmp_path: Path) -> None:
    mock_blob_client = MagicMock()
    mock_blob_client.upload_blob.side_effect = ConnectionError("Azure network timeout")

    mock_container_client = MagicMock()
    mock_container_client.get_blob_client.return_value = mock_blob_client

    mock_service_client = MagicMock()
    mock_service_client.get_container_client.return_value = mock_container_client

    with patch("backend.app.routes.feedback.BlobServiceClient") as mock_bsc:
        mock_bsc.from_connection_string.return_value = mock_service_client

        sink = AzureBlobStorageSink(
            connection_string="fake_conn_str",
            container_crops="feedback-crops",
        )

        # Call _save_line_crop with failing Azure sink
        crops_dir = tmp_path / "crops"
        crop_path = _save_line_crop(
            crop_b64=sample_crop_b64,
            feedback_id="fb_fallback_1",
            crops_dir=crops_dir,
            azure_sink=sink,
            storage_mode="auto",
        )

        # Must fall back gracefully to local disk path
        assert crop_path.endswith("fb_fallback_1.png")
        assert Path(crop_path).exists()


def test_azure_sink_append_manifest_success() -> None:
    mock_blob_client = MagicMock()
    mock_blob_client.url = "https://myaccount.blob.core.windows.net/feedback-manifests/manifest.jsonl"
    mock_blob_client.exists.return_value = True

    mock_container_client = MagicMock()
    mock_container_client.get_blob_client.return_value = mock_blob_client

    mock_service_client = MagicMock()
    mock_service_client.get_container_client.return_value = mock_container_client

    with patch("backend.app.routes.feedback.BlobServiceClient") as mock_bsc:
        mock_bsc.from_connection_string.return_value = mock_service_client

        sink = AzureBlobStorageSink(
            connection_string="fake_conn_str",
            container_manifests="feedback-manifests",
        )

        url = sink.append_manifest_record({"feedback_id": "fb_manifest_1"})
        assert url == "https://myaccount.blob.core.windows.net/feedback-manifests/manifest.jsonl"


def test_feedback_api_with_azure_blob_sink(sample_crop_b64: str, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    crops_dir = tmp_path / "crops"

    mock_crop_blob = MagicMock()
    mock_crop_blob.url = "https://myaccount.blob.core.windows.net/feedback-crops/crops/fb_test.png"

    mock_manifest_blob = MagicMock()
    mock_manifest_blob.url = "https://myaccount.blob.core.windows.net/feedback-manifests/manifest.jsonl"
    mock_manifest_blob.exists.return_value = True

    mock_container = MagicMock()
    def get_blob_side_effect(name: str):
        if "crops/" in name:
            return mock_crop_blob
        return mock_manifest_blob

    mock_container.get_blob_client.side_effect = get_blob_side_effect

    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch("backend.app.routes.feedback.BlobServiceClient") as mock_bsc:
        mock_bsc.from_connection_string.return_value = mock_service

        settings = get_settings()
        orig_conn = settings.AZURE_STORAGE_CONNECTION_STRING
        orig_crops = settings.FEEDBACK_CROPS_DIR
        orig_manifest = settings.FEEDBACK_MANIFEST_PATH

        try:
            settings.AZURE_STORAGE_CONNECTION_STRING = "fake_conn_str"
            settings.FEEDBACK_CROPS_DIR = str(crops_dir)
            settings.FEEDBACK_MANIFEST_PATH = str(manifest_path)

            client = TestClient(app)
            payload = {
                "document_id": "doc_cloud_1",
                "line_id": "p1_l1",
                "original_prediction": "amoxicilin",
                "operator_correction": "amoxicillin",
                "line_crop_base64": sample_crop_b64,
                "confidence": 0.95,
            }

            resp = client.post("/v1/feedback", json=payload)
            assert resp.status_code == 200
            data = resp.json()

            assert data["status"] == "persisted"
            # Crop path should be Azure blob URL
            assert Path(data["crop_path"]).is_file()
            # Manifest path should be Azure blob URL
            assert data["manifest_path"] == str(manifest_path)
            # And local disk file was also written as local durability mirror
            assert manifest_path.exists()
        finally:
            settings.AZURE_STORAGE_CONNECTION_STRING = orig_conn
            settings.FEEDBACK_CROPS_DIR = orig_crops
            settings.FEEDBACK_MANIFEST_PATH = orig_manifest

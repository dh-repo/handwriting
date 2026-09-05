"""
backend/app/routes/feedback.py
FastAPI route for Darkroom operator feedback ingestion, atomic manifest storage,
line crop persistence, and dynamic confusion matrix recalibration.
"""

from __future__ import annotations
import asyncio
import hashlib
import uuid
import base64
from datetime import datetime, timezone
import fcntl
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from PIL import Image

try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    AZURE_STORAGE_AVAILABLE = True
except ImportError:
    AZURE_STORAGE_AVAILABLE = False

from backend.app.config import Settings, get_settings
from backend.app.engine import get_engine
from backend.app.schemas import (
    ConfusionUpdateRecord,
    FeedbackCorrectionPayload,
    FeedbackResponse,
    FeedbackStatsResponse,
)

logger = logging.getLogger("handwriting_backend.feedback")

router = APIRouter()


class AzureBlobStorageSink:
    """
    Azure Blob Storage sink for operator feedback manifests and line crops.
    Supports connection string and Managed Identity (DefaultAzureCredential).
    Falls back gracefully to local disk when offline or unconfigured.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        container_crops: str = "feedback-crops",
        container_manifests: str = "feedback-manifests",
    ):
        self.connection_string = connection_string
        self.account_name = account_name
        self.container_crops = container_crops
        self.container_manifests = container_manifests
        self._client: Optional[Any] = None
        self._initialized = False

    @property
    def is_configured(self) -> bool:
        return bool(self.connection_string or self.account_name)

    def get_client(self) -> Optional[Any]:
        if not self._initialized:
            self._client = self._init_client()
            self._initialized = True
        return self._client

    def _init_client(self) -> Optional[Any]:
        if not AZURE_STORAGE_AVAILABLE:
            logger.debug("azure-storage-blob library not installed; using local disk sink.")
            return None
        try:
            if self.connection_string:
                return BlobServiceClient.from_connection_string(self.connection_string)
            elif self.account_name:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                credential = DefaultAzureCredential()
                return BlobServiceClient(account_url=account_url, credential=credential)
        except Exception as exc:
            logger.warning(f"Failed to initialize Azure BlobServiceClient: {exc}")
            return None
        return None

    def upload_crop(self, feedback_id: str, image_bytes: bytes) -> Optional[str]:
        """Upload line crop bytes to Azure Blob Storage. Returns blob URL or None."""
        client = self.get_client()
        if not client:
            return None
        try:
            container = client.get_container_client(self.container_crops)
            try:
                container.create_container()
            except Exception:
                pass  # Container already exists
            blob_name = f"crops/{feedback_id}.png"
            blob_client = container.get_blob_client(blob_name)
            blob_client.upload_blob(image_bytes, overwrite=True, content_type="image/png")
            return str(blob_client.url)
        except Exception as exc:
            logger.warning(f"Azure Blob crop upload failed ({exc}); falling back to local disk.")
            return None

    def append_manifest_record(self, record: dict[str, Any]) -> Optional[str]:
        """Append feedback JSON record to Azure Blob Storage. Returns blob URI or None."""
        client = self.get_client()
        if not client:
            return None
        try:
            container = client.get_container_client(self.container_manifests)
            try:
                container.create_container()
            except Exception:
                pass  # Container already exists
            blob_name = "manifest.jsonl"
            blob_client = container.get_blob_client(blob_name)
            record_bytes = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")

            # Try append_block first (if Append Blob exists or can be created)
            try:
                if not blob_client.exists():
                    blob_client.create_append_blob()
                blob_client.append_block(record_bytes)
                return str(blob_client.url)
            except Exception:
                # Fallback to block blob download-append-upload
                existing = b""
                if blob_client.exists():
                    existing = blob_client.download_blob().readall()
                blob_client.upload_blob(existing + record_bytes, overwrite=True)
                return str(blob_client.url)
        except Exception as exc:
            logger.warning(f"Azure Blob manifest append failed ({exc}); falling back to local disk.")
            return None


def _get_azure_sink(settings: Settings) -> AzureBlobStorageSink:
    return AzureBlobStorageSink(
        connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
        account_name=settings.AZURE_STORAGE_ACCOUNT_NAME,
        container_crops=settings.AZURE_STORAGE_CONTAINER_CROPS,
        container_manifests=settings.AZURE_STORAGE_CONTAINER_MANIFESTS,
    )


def _save_line_crop(
    crop_b64: str,
    feedback_id: str,
    crops_dir: Path,
    azure_sink: Optional[AzureBlobStorageSink] = None,
    storage_mode: str = "auto",
) -> str:
    """
    Decode a base64 line crop image (raw or Data URL) and persist it as a PNG file.
    If Azure Blob Storage is configured and enabled, streams crop bytes to Azure Blob container.
    Always creates a local copy for local caching/durability.
    Raises HTTPException(422) if decoding, size/dimension limits, or image parsing fails.
    """
    data_str = crop_b64.strip()
    if not data_str:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="line_crop_base64 string is empty.",
        )

    # Handle Data URL scheme (e.g. data:image/png;base64,<data>)
    if "," in data_str and data_str.startswith("data:"):
        _, data_str = data_str.split(",", 1)
        data_str = data_str.strip()

    try:
        img_bytes = base64.b64decode(data_str, validate=True)
    except Exception as exc:
        logger.warning(f"Failed to decode base64 line crop: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid base64 encoding for line crop: {exc}",
        ) from exc

    if not img_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Decoded line crop contains 0 bytes.",
        )

    if len(img_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Line crop exceeds 5MB limit.",
        )

    try:
        image = Image.open(io.BytesIO(img_bytes))
        image.load()
    except Exception as exc:
        logger.warning(f"Corrupted image in line crop payload: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Decoded data is not a valid image: {exc}",
        ) from exc

    if image.width > 4096 or image.height > 2048:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Line crop dimensions exceed maximum bounds.",
        )

    from pipeline.evaluation.holdout import is_holdout_image
    if is_holdout_image(image):
        raise HTTPException(status_code=422, detail="Frozen evaluation images cannot enter the feedback collection")
    crops_dir.mkdir(parents=True, exist_ok=True)
    local_crop_path = crops_dir / f"{feedback_id}.png"

    try:
        with local_crop_path.open("wb") as crop_file:
            image.save(crop_file, format="PNG")
            crop_file.flush()
            os.fsync(crop_file.fileno())
    except Exception as exc:
        logger.error(f"Failed to save line crop to {local_crop_path}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist decoded line crop image.",
        ) from exc

    # Attempt upload to Azure Blob Storage if configured
    if azure_sink and azure_sink.is_configured and storage_mode != "local":
        azure_url = azure_sink.upload_crop(feedback_id, img_bytes)
        if azure_url:
            return azure_url

    return str(local_crop_path)


def _append_to_manifest(
    manifest_path: Path,
    record: dict[str, Any],
    azure_sink: Optional[AzureBlobStorageSink] = None,
    storage_mode: str = "auto",
) -> Optional[str]:
    """
    Thread-safe and process-safe atomic append of a feedback record to JSONL manifest
    using POSIX advisory file locking (fcntl.flock).
    If Azure Blob Storage is configured and enabled, also uploads/appends to Azure Blob container.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record_line = json.dumps(record, ensure_ascii=False) + "\n"

    with open(manifest_path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(record_line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    azure_manifest_url = None
    if azure_sink and azure_sink.is_configured and storage_mode != "local":
        azure_manifest_url = azure_sink.append_manifest_record(record)

    return azure_manifest_url


def _trigger_dynamic_confusion_update(
    original_prediction: str,
    operator_correction: str,
    learning_rate: float,
) -> List[ConfusionUpdateRecord]:
    """
    Hook into live InferenceEngine rescorer confusion matrix if available.
    Returns list of ConfusionUpdateRecord updates.
    """
    if len(original_prediction) > 300 or len(operator_correction) > 300:
        logger.info("Skipping dynamic confusion update: text length exceeds 300 characters.")
        return []

    updates: List[ConfusionUpdateRecord] = []
    try:
        engine = get_engine()
        rescorer = getattr(engine, "rescorer", None)
        cm = getattr(rescorer, "confusion_matrix", None)
        adapt_fn = getattr(cm, "adapt_from_correction", None)
        if callable(adapt_fn):
            raw_updates = adapt_fn(
                original_prediction,
                operator_correction,
                learning_rate=learning_rate,
            )
            if isinstance(raw_updates, list):
                for item in raw_updates:
                    if isinstance(item, ConfusionUpdateRecord):
                        updates.append(item)
                    elif isinstance(item, dict):
                        updates.append(ConfusionUpdateRecord(**item))
                    elif hasattr(item, "operation"):
                        updates.append(
                            ConfusionUpdateRecord(
                                operation=getattr(item, "operation"),
                                source=getattr(item, "source", getattr(item, "source_chars", "")),
                                target=getattr(item, "target", getattr(item, "target_chars", "")),
                                previous_cost=float(
                                    getattr(item, "previous_cost", getattr(item, "old_cost", 0.0))
                                ),
                                updated_cost=float(
                                    getattr(item, "updated_cost", getattr(item, "new_cost", 0.0))
                                ),
                            )
                        )
    except Exception as exc:
        logger.warning(f"Dynamic confusion matrix update hook encountered exception: {exc}")

    return updates


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackCorrectionPayload, settings: Settings = Depends(get_settings)) -> FeedbackResponse:
    if payload.is_demo:
        raise HTTPException(status_code=422, detail="Demo results cannot be used for learning")
    return await asyncio.to_thread(_persist_correction, payload, settings)


def _persist_correction(payload: FeedbackCorrectionPayload, settings: Settings) -> FeedbackResponse:
    manifest = Path(settings.FEEDBACK_MANIFEST_PATH)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    feedback_id = payload.submission_id or f"fb_{uuid.uuid4().hex}"
    fingerprint = hashlib.sha256(json.dumps(payload.model_dump(exclude={"timestamp"}), sort_keys=True).encode()).hexdigest()
    with open(str(manifest) + ".lock", "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if manifest.exists():
            for raw in manifest.read_text().splitlines():
                record = json.loads(raw)
                if record.get("feedback_id") == feedback_id:
                    if record.get("fingerprint") != fingerprint:
                        raise HTTPException(status_code=409, detail="Submission ID already belongs to another correction")
                    return FeedbackResponse(**record["response"])
        crop_path = None
        if payload.line_crop_base64:
            crop_path = _save_line_crop(payload.line_crop_base64, feedback_id, Path(settings.FEEDBACK_CROPS_DIR), azure_sink=None, storage_mode="local")
        response = FeedbackResponse(feedback_id=feedback_id, document_id=payload.document_id, line_id=payload.line_id,
            manifest_path=str(manifest), crop_path=crop_path, timestamp=datetime.now(timezone.utc).isoformat())
        record = {**payload.model_dump(), "feedback_id": feedback_id, "fingerprint": fingerprint,
                  "image_crop_path": crop_path, "response": response.model_dump()}
        record.pop("line_crop_base64", None)
        _append_to_manifest(manifest, record, azure_sink=None, storage_mode="local")
        return response


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(settings: Settings = Depends(get_settings)) -> FeedbackStatsResponse:
    """Return summary statistics for stored feedback manifest entries and line crops."""
    manifest_path = Path(settings.FEEDBACK_MANIFEST_PATH)
    crops_dir = Path(settings.FEEDBACK_CROPS_DIR)

    total_records = 0
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    if line.strip():
                        total_records += 1
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    total_crops = 0
    if crops_dir.exists():
        total_crops = len([p for p in crops_dir.glob("*.png") if p.is_file()])

    return FeedbackStatsResponse(
        total_records=total_records,
        total_crops=total_crops,
        manifest_path=str(manifest_path),
        crops_dir=str(crops_dir),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

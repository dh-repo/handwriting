"""
backend/app/routes/feedback.py
FastAPI route for Darkroom operator feedback ingestion, atomic manifest storage,
line crop persistence, and dynamic confusion matrix recalibration.
"""

from __future__ import annotations
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


def _save_line_crop(crop_b64: str, feedback_id: str, crops_dir: Path) -> str:
    """
    Decode a base64 line crop image (raw or Data URL) and persist it as a PNG file.
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

    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crops_dir / f"{feedback_id}.png"

    try:
        image.save(crop_path, format="PNG")
    except Exception as exc:
        logger.error(f"Failed to save line crop to {crop_path}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist decoded line crop image.",
        ) from exc

    return str(crop_path)


def _append_to_manifest(manifest_path: Path, record: dict[str, Any]) -> None:
    """
    Thread-safe and process-safe atomic append of a feedback record to JSONL manifest
    using POSIX advisory file locking (fcntl.flock).
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


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_200_OK)
async def submit_feedback(
    payload: FeedbackCorrectionPayload,
    settings: Settings = Depends(get_settings),
) -> FeedbackResponse:
    """
    Ingest human operator corrections from the Darkroom UI.

    - Validates correction payload and bounding boxes.
    - Decodes and persists line crop image (PNG) if provided.
    - Atomically appends feedback metadata to append-only JSONL manifest.
    - Dynamically adapts live confusion matrix substitution/ligature costs.
    """
    ts_now = datetime.now(timezone.utc)
    ts_str = ts_now.strftime("%Y%m%d_%H%M%S")
    feedback_id = f"fb_{ts_str}_{uuid.uuid4().hex[:8]}"
    timestamp = payload.timestamp or ts_now.isoformat()

    # 1. Line crop storage
    crop_path: Optional[str] = None
    if payload.line_crop_base64:
        crops_dir = Path(settings.FEEDBACK_CROPS_DIR)
        crop_path = _save_line_crop(payload.line_crop_base64, feedback_id, crops_dir)

    # 2. Dynamic confusion matrix adaptation hook
    confusion_pairs_updated: List[ConfusionUpdateRecord] = []
    if payload.sync_confusion_matrix:
        confusion_pairs_updated = _trigger_dynamic_confusion_update(
            original_prediction=payload.original_prediction,
            operator_correction=payload.operator_correction,
            learning_rate=settings.CONFUSION_LEARNING_RATE,
        )

    # 3. Build manifest record
    manifest_path = Path(settings.FEEDBACK_MANIFEST_PATH)
    record = {
        "feedback_id": feedback_id,
        "document_id": payload.document_id,
        "page_number": payload.page_number,
        "line_id": payload.line_id,
        "word_id": payload.word_id,
        "original_prediction": payload.original_prediction,
        "operator_correction": payload.operator_correction,
        "confidence": payload.confidence,
        "bbox": payload.bbox,
        "image_crop_path": crop_path,
        "timestamp": timestamp,
        "alignment_operations": [u.model_dump() for u in confusion_pairs_updated],
    }

    # 4. Atomic append to JSONL manifest
    _append_to_manifest(manifest_path, record)

    return FeedbackResponse(
        feedback_id=feedback_id,
        document_id=payload.document_id,
        line_id=payload.line_id,
        status="persisted",
        manifest_path=str(manifest_path),
        crop_path=crop_path,
        confusion_pairs_updated=confusion_pairs_updated,
        timestamp=timestamp,
    )


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

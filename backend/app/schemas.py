"""
backend/app/schemas.py
Pydantic v2 Data Contracts matching PROJECT.md § Interface Contracts.
"""

from __future__ import annotations
import math
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


def _validate_bbox_coordinates(v: List[float], label: str = "Bounding box") -> List[float]:
    """Validate 4-element normalized bounding box [ymin, xmin, ymax, xmax]."""
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise ValueError(f"{label} must contain exactly 4 coordinates: [ymin, xmin, ymax, xmax]")
    ymin, xmin, ymax, xmax = [float(x) for x in v]
    for val, name in zip([ymin, xmin, ymax, xmax], ["ymin", "xmin", "ymax", "xmax"]):
        if math.isnan(val) or math.isinf(val) or not (0.0 <= val <= 1.0):
            raise ValueError(f"{label} coordinate {name}={val} is outside normalized range [0.0, 1.0]")
    if ymin >= ymax:
        raise ValueError(f"{label} invalid vertical span: ymin ({ymin}) >= ymax ({ymax})")
    if xmin >= xmax:
        raise ValueError(f"{label} invalid horizontal span: xmin ({xmin}) >= xmax ({xmax})")
    return [ymin, xmin, ymax, xmax]


class WordBox(BaseModel):
    """Word-level bounding box and recognition confidence."""

    word_id: str = Field(..., description="Unique word identifier, e.g. 'p1_l1_w1'")
    text: str = Field(..., description="Transcribed token text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Recognition confidence score in [0.0, 1.0]")
    bbox: List[float] = Field(..., description="Normalized bounding box [ymin, xmin, ymax, xmax] in [0.0, 1.0]")
    is_proper_noun: bool = Field(default=False, description="Whether token is recognized as a proper noun, name, or initial")

    model_config = ConfigDict(extra="ignore")

    @field_validator("confidence")
    @classmethod
    def check_confidence(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v) or not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be a valid float between 0.0 and 1.0")
        return v

    @field_validator("bbox")
    @classmethod
    def check_bbox(cls, v: List[float]) -> List[float]:
        return _validate_bbox_coordinates(v, "Word bounding box")


class LineBox(BaseModel):
    """Line-level bounding box, recognition confidence, and constituent words."""

    line_id: str = Field(..., description="Unique line identifier, e.g. 'p1_l1'")
    text: str = Field(..., description="Transcribed line text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Line recognition confidence in [0.0, 1.0]")
    bbox: List[float] = Field(..., description="Normalized bounding box [ymin, xmin, ymax, xmax] in [0.0, 1.0]")
    words: List[WordBox] = Field(default_factory=list, description="Word token crops within this line")

    model_config = ConfigDict(extra="ignore")

    @field_validator("confidence")
    @classmethod
    def check_confidence(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v) or not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be a valid float between 0.0 and 1.0")
        return v

    @field_validator("bbox")
    @classmethod
    def check_bbox(cls, v: List[float]) -> List[float]:
        return _validate_bbox_coordinates(v, "Line bounding box")


class PageResult(BaseModel):
    """Page-level document transcription and layout segmentation."""

    page_number: int = Field(..., ge=1, description="1-indexed page sequence number")
    width: int = Field(..., ge=1, description="Page width in pixels")
    height: int = Field(..., ge=1, description="Page height in pixels")
    full_text: str = Field(..., description="Newline-joined page text")
    mean_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean confidence score across all lines")
    lines: List[LineBox] = Field(default_factory=list, description="Segmented lines on this page")

    model_config = ConfigDict(extra="ignore")


class RecognitionResponse(BaseModel):
    """Hierarchical document recognition response."""

    document_id: str = Field(..., description="Unique document processing identifier, e.g. 'doc_12345678'")
    filename: str = Field(..., description="Original uploaded filename")
    total_pages: int = Field(..., ge=1, description="Total number of pages processed")
    pages: List[PageResult] = Field(..., description="Transcribed page records")
    processing_time_ms: float = Field(..., ge=0.0, description="Total processing latency in milliseconds")
    preprocessing_flags: Optional[Dict[str, Any]] = Field(default=None, description="Preprocessing flags applied")
    engine_used: Optional[str] = Field(default="trocr", description="Engine used ('turbo-vlm' | 'trocr')")

    model_config = ConfigDict(extra="ignore")


class JobStatusEnum(str, Enum):
    """Asynchronous job lifecycle states."""

    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobSubmissionResponse(BaseModel):
    """Immediate response returned upon async job submission."""

    job_id: str = Field(..., description="Job identifier, e.g. 'job_1234567890ab'")
    status: JobStatusEnum = Field(default=JobStatusEnum.QUEUED, description="Initial job status")
    filename: str = Field(..., description="Target file name")
    created_at: str = Field(..., description="ISO-8601 UTC creation timestamp")

    model_config = ConfigDict(extra="ignore")


class JobStatusResponse(BaseModel):
    """Detailed asynchronous job status and result model."""

    job_id: str = Field(..., description="Job identifier")
    filename: str = Field(..., description="Target file name")
    status: JobStatusEnum = Field(..., description="Current job lifecycle status")
    progress: float = Field(..., ge=0.0, le=1.0, description="Job progress percentage in [0.0, 1.0]")
    result: Optional[RecognitionResponse] = Field(default=None, description="Completed transcription result")
    error: Optional[str] = Field(default=None, description="Error message if job failed")
    created_at: str = Field(..., description="ISO-8601 UTC creation timestamp")
    updated_at: str = Field(..., description="ISO-8601 UTC last status update timestamp")

    model_config = ConfigDict(extra="ignore")


class LiveResponse(BaseModel):
    """Cheap liveness signal that does not load the inference engine."""

    status: str = Field(default="live", description="Process liveness state")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")

    model_config = ConfigDict(extra="ignore")


class HealthResponse(BaseModel):
    """Service health and runtime information."""

    status: str = Field(default="healthy", description="Service health state ('healthy' | 'degraded')")
    device: str = Field(..., description="Execution device ('mps' | 'cpu' | 'cuda' | 'mock')")
    version: str = Field(default="1.0.0", description="API version")
    memory_usage_mb: Optional[float] = Field(default=None, description="Process memory usage in MB")
    loaded_models: Optional[List[str]] = Field(default=None, description="List of initialized model names")
    mps_available: Optional[bool] = Field(default=None, description="Whether Apple Silicon MPS is available")
    execution_mode: Optional[str] = Field(default=None, description="Configured execution mode")
    rescorer_active: bool = Field(default=True, description="Whether RxNorm Beam Rescorer is active")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")

    model_config = ConfigDict(extra="ignore")


class RecognitionOptions(BaseModel):
    """Optional preprocessing, segmentation, and beam rescoring parameters."""

    deskew: bool = Field(default=True, description="Enable automated Hough deskewing")
    enhance_contrast: bool = Field(default=True, description="Enable CLAHE and illumination flattening")
    binarization_method: str = Field(default="sauvola", description="Binarization algorithm ('sauvola' | 'otsu' | 'none')")
    extract_words: bool = Field(default=True, description="Enable word segmentation within line crops")
    dpi: int = Field(default=300, ge=72, le=600, description="DPI resolution for PDF rasterization")
    beam_width: int = Field(default=1, ge=1, le=16, description="Beam search candidate width")
    rescore: bool = Field(default=False, description="Enable RxNorm beam rescoring")
    adaptive: bool = Field(default=True, description="Enable adaptive fast 2-pass decoding")
    turbo: bool = Field(default=True, description="Enable Turbo VLM mode for sub-3s high-accuracy recognition")

    model_config = ConfigDict(extra="ignore")


class RecognizeJsonRequest(BaseModel):
    """JSON payload for base64 encoded document recognition."""

    file_base64: str = Field(..., description="Base64-encoded file bytes (image or PDF)")
    filename: Optional[str] = Field(default="upload.png", description="Original file name with extension")
    options: Optional[RecognitionOptions] = Field(default=None, description="Optional preprocessing configuration")

    model_config = ConfigDict(extra="ignore")


class ErrorResponse(BaseModel):
    """Sanitized structured error response."""

    error: str = Field(..., description="Error category or exception type name")
    detail: str = Field(..., description="Sanitized, human-readable error description")
    code: Optional[str] = Field(default=None, description="Machine-readable error code")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")

    model_config = ConfigDict(extra="ignore")


class ConfusionUpdateRecord(BaseModel):
    """Record of an online visual confusion cost adjustment."""

    operation: str = Field(..., description="DP alignment operation ('substitution', 'contraction', 'expansion', 'substitution_2_2')")
    source: str = Field(..., description="Source character(s) or ligature")
    target: str = Field(..., description="Target character(s) or ligature")
    previous_cost: float = Field(..., description="Visual confusion cost before adaptation")
    updated_cost: float = Field(..., description="Visual confusion cost after adaptation")

    model_config = ConfigDict(extra="ignore")


class FeedbackCorrectionPayload(BaseModel):
    """Payload for Darkroom operator corrections and feedback ingestion."""

    document_id: str = Field(..., min_length=1, description="Document identifier, e.g. 'doc_12345678'")
    line_id: str = Field(..., min_length=1, description="Line identifier, e.g. 'p1_l2'")
    page_number: int = Field(default=1, ge=1, description="1-indexed page sequence number")
    original_prediction: str = Field(
        ...,
        validation_alias=AliasChoices("original_prediction", "original_text"),
        max_length=500,
        description="Original model prediction before operator edit",
    )
    operator_correction: str = Field(
        ...,
        validation_alias=AliasChoices("operator_correction", "corrected_text"),
        max_length=500,
        description="Verified operator correction text",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Line/word confidence score in [0.0, 1.0]")
    bbox: Optional[List[float]] = Field(default=None, description="Normalized bounding box [ymin, xmin, ymax, xmax]")
    word_id: Optional[str] = Field(default=None, description="Optional word identifier if word-level edit")
    line_crop_base64: Optional[str] = Field(default=None, description="Base64 encoded PNG/JPEG line crop image")
    sync_confusion_matrix: bool = Field(default=True, description="Whether to dynamically update live confusion matrix")
    timestamp: Optional[str] = Field(default=None, description="Optional ISO-8601 UTC timestamp")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("document_id", "line_id")
    @classmethod
    def check_non_empty(cls, v: str, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must not be empty or whitespace only")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def check_confidence(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v) or not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be a valid float between 0.0 and 1.0")
        return v

    @field_validator("bbox")
    @classmethod
    def check_bbox(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            return _validate_bbox_coordinates(v, "Feedback bbox")
        return v


class FeedbackResponse(BaseModel):
    """Response returned upon successful feedback correction ingestion."""

    feedback_id: str = Field(..., description="Unique generated feedback ID, e.g. 'fb_20260903_143000_a1b2c3'")
    document_id: Optional[str] = Field(default=None, description="Document identifier echoed from request")
    line_id: Optional[str] = Field(default=None, description="Line identifier echoed from request")
    status: str = Field(default="persisted", description="Ingestion status ('persisted')")
    manifest_path: str = Field(..., description="Absolute or relative path to append-only manifest JSONL")
    crop_path: Optional[str] = Field(default=None, description="Path to saved crop image file, or None if omitted")
    confusion_pairs_updated: List[ConfusionUpdateRecord] = Field(
        default_factory=list,
        description="Updated confusion pairs from dynamic alignment",
    )
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")

    model_config = ConfigDict(extra="ignore")


class FeedbackStatsResponse(BaseModel):
    """Feedback repository statistics."""

    total_records: int = Field(default=0, ge=0, description="Total feedback records in manifest")
    total_crops: int = Field(default=0, ge=0, description="Total line crops stored on disk")
    manifest_path: str = Field(..., description="Path to feedback manifest file")
    crops_dir: str = Field(..., description="Path to feedback crops directory")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")

    model_config = ConfigDict(extra="ignore")

#!/usr/bin/env python3
"""
Deterministic Mock Inference Engine & Test Harness for Handwriting Recognition E2E Testing.
Provides:
- Manifest-aware deterministic recognition responses
- Multi-page PDF extraction and streaming simulation
- Async Job Management & Server-Sent Events (SSE) streaming
- Fault injection (timeouts, memory overflow, bad payloads)
- Mock Preprocessing routines (Deskew, CLAHE, Sauvola binarization, HPP segmentation)
- Mock Training & Evaluation loops (loss curves, checkpointing, CER/WER metrics)
- Lightweight FastAPI app factory for end-to-end HTTP testing
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Interface Contract Schemas (Matching PROJECT.md & backend/app/schemas.py)
# ---------------------------------------------------------------------------

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
    word_id: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(description="Normalized [ymin, xmin, ymax, xmax] in [0, 1]")

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
    line_id: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(description="Normalized [ymin, xmin, ymax, xmax] in [0, 1]")
    words: List[WordBox] = Field(default_factory=list)

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
    page_number: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    full_text: str
    mean_confidence: float = Field(ge=0.0, le=1.0)
    lines: List[LineBox] = Field(default_factory=list)


class RecognitionResponse(BaseModel):
    document_id: str
    filename: str
    total_pages: int = Field(ge=1)
    pages: List[PageResult]
    processing_time_ms: float = Field(ge=0.0)


class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: str = Field(description="QUEUED | PROCESSING | COMPLETED | FAILED")
    progress: float = Field(ge=0.0, le=1.0)
    result: Optional[RecognitionResponse] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class LineCrop(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    line_index: int
    image: Optional[Any] = None  # np.ndarray or PIL.Image
    bbox: List[float] = Field(description="[ymin, xmin, ymax, xmax] normalized")


class PreprocessedPage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    page_index: int
    original_image: Optional[Any] = None
    enhanced_image: Optional[Any] = None
    binarized_image: Optional[Any] = None
    lines: List[LineCrop] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Levenshtein Edit Distance & Evaluation Helpers
# ---------------------------------------------------------------------------

def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein distance between two sequences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_cer(hypothesis: str, reference: str) -> float:
    """Compute Character Error Rate (CER = distance / len(ref))."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    dist = compute_levenshtein_distance(hypothesis, reference)
    return float(dist) / len(reference)


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate (WER = word_distance / len(ref_words))."""
    h_words = hypothesis.strip().split()
    r_words = reference.strip().split()
    if not r_words:
        return 0.0 if not h_words else 1.0
    dist = compute_levenshtein_distance(h_words, r_words)
    return float(dist) / len(r_words)


# ---------------------------------------------------------------------------
# Mock Inference Engine & Async Job Manager
# ---------------------------------------------------------------------------

class MockInferenceEngine:
    """Self-contained deterministic recognition engine and job manager."""

    def __init__(
        self,
        manifest_path: Optional[Path | str] = None,
        latency_ms: float = 0.0,
        failure_rate: float = 0.0,
    ) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path else Path("tests/fixtures/manifest.json")
        self.manifest: Dict[str, Any] = {}
        self.load_manifest()

        # Fault injection controls
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.force_error: Optional[str] = None
        self.memory_limit_mb: Optional[float] = None

        # In-memory Async Job Store
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def load_manifest(self) -> None:
        """Load manifest.json if present."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception:
                self.manifest = {"fixtures": {}}
        else:
            self.manifest = {"fixtures": {}}

    def configure(
        self,
        latency_ms: Optional[float] = None,
        failure_rate: Optional[float] = None,
        force_error: Optional[str] = None,
        memory_limit_mb: Optional[float] = None,
    ) -> None:
        """Configure simulation and fault injection parameters."""
        if latency_ms is not None:
            self.latency_ms = latency_ms
        if failure_rate is not None:
            self.failure_rate = failure_rate
        if force_error is not None:
            self.force_error = force_error
        if memory_limit_mb is not None:
            self.memory_limit_mb = memory_limit_mb

    def _check_fault_injection(self) -> None:
        """Apply latency and simulated faults."""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        if self.force_error:
            err = self.force_error
            self.force_error = None
            raise RuntimeError(f"Simulated fault: {err}")
        if self.failure_rate > 0 and np.random.random() < self.failure_rate:
            raise RuntimeError("Simulated transient engine failure")

    def _lookup_manifest_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Find matching entry in manifest by filename."""
        fixtures = self.manifest.get("fixtures", {})
        # Exact match
        if filename in fixtures:
            return fixtures[filename]
        # Match base filename
        base = Path(filename).name
        if base in fixtures:
            return fixtures[base]
        # Check relative keys
        for k, v in fixtures.items():
            if Path(k).name == base:
                return v
        return None

    def recognize_image(
        self, file_bytes: bytes, filename: str = "sample.png"
    ) -> RecognitionResponse:
        """Recognize single image file bytes deterministically."""
        t0 = time.time()
        self._check_fault_injection()

        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("InvalidImageError: Input image is empty (0 bytes).")

        # Check for plain text disguised as image
        if file_bytes.startswith(b"This is plain text") or (not file_bytes.startswith(b"\x89PNG") and not file_bytes.startswith(b"\xff\xd8") and not file_bytes.startswith(b"II*\x00") and not file_bytes.startswith(b"MM\x00*") and not file_bytes.startswith(b"BM") and not file_bytes.startswith(b"RIFF")):
            if not filename.lower().endswith(".pdf"):
                raise ValueError("InvalidImageSignatureError: File has invalid image magic header.")

        # Check manifest match
        entry = self._lookup_manifest_by_filename(filename)

        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            width, height = pil_img.size
        except Exception as e:
            raise ValueError(f"UnidentifiedImageError: Cannot decode image bytes: {e}")

        # If manifest contains ground truth line details
        if entry and "lines" in entry:
            lines_data = [
                LineBox(
                    line_id=l["line_id"],
                    text=l["text"],
                    confidence=l.get("confidence", 0.98),
                    bbox=l["bbox"],
                    words=[
                        WordBox(
                            word_id=w["word_id"],
                            text=w["text"],
                            confidence=w.get("confidence", 0.98),
                            bbox=w["bbox"],
                        )
                        for w in l.get("words", [])
                    ],
                )
                for l in entry["lines"]
            ]
            full_text = entry.get("ground_truth_text", "\n".join(l.text for l in lines_data))
            mean_conf = float(np.mean([l.confidence for l in lines_data])) if lines_data else 1.0

            elapsed_ms = (time.time() - t0) * 1000.0 + self.latency_ms
            return RecognitionResponse(
                document_id=f"doc_{uuid.uuid4().hex[:8]}",
                filename=Path(filename).name,
                total_pages=1,
                pages=[
                    PageResult(
                        page_number=1,
                        width=width,
                        height=height,
                        full_text=full_text,
                        mean_confidence=round(mean_conf, 3),
                        lines=lines_data,
                    )
                ],
                processing_time_ms=round(elapsed_ms, 2),
            )

        # If manifest has text but no detailed lines
        if entry and "ground_truth_text" in entry:
            gt_text = entry["ground_truth_text"]
            lines_raw = [l for l in gt_text.split("\n") if l.strip()]
            lines_data = []
            num_lines = max(1, len(lines_raw))
            for idx, lt in enumerate(lines_raw):
                ymin = round(0.1 + (idx / (num_lines + 1)) * 0.75, 4)
                ymax = round(ymin + (0.6 / (num_lines + 1)), 4)
                words_raw = lt.split(" ")
                words_data = []
                num_w = max(1, len(words_raw))
                for w_idx, wt in enumerate(words_raw):
                    w_xmin = round(0.1 + (w_idx / (num_w + 1)) * 0.78, 4)
                    w_xmax = round(w_xmin + (0.7 / (num_w + 1)), 4)
                    words_data.append(
                        WordBox(
                            word_id=f"p1_l{idx+1}_w{w_idx+1}",
                            text=wt,
                            confidence=0.96,
                            bbox=[ymin, w_xmin, ymax, min(0.96, w_xmax)],
                        )
                    )
                lines_data.append(
                    LineBox(
                        line_id=f"p1_l{idx+1}",
                        text=lt,
                        confidence=0.96,
                        bbox=[ymin, 0.08, ymax, 0.94],
                        words=words_data,
                    )
                )

            elapsed_ms = (time.time() - t0) * 1000.0 + self.latency_ms
            return RecognitionResponse(
                document_id=f"doc_{uuid.uuid4().hex[:8]}",
                filename=Path(filename).name,
                total_pages=1,
                pages=[
                    PageResult(
                        page_number=1,
                        width=width,
                        height=height,
                        full_text=gt_text,
                        mean_confidence=0.96,
                        lines=lines_data,
                    )
                ],
                processing_time_ms=round(elapsed_ms, 2),
            )

        # Unknown image: dynamically generate plausible OCR results based on aspect ratio
        num_lines = max(1, min(10, height // 120))
        lines_data = []
        full_text_lines = []
        for i in range(num_lines):
            line_text = f"Recognized handwritten line {i+1}"
            full_text_lines.append(line_text)
            ymin = round(0.08 + (i / (num_lines + 1)) * 0.80, 4)
            ymax = round(ymin + (0.65 / (num_lines + 1)), 4)
            words_list = line_text.split()
            n_w = len(words_list)
            words_data = []
            for w_i, word in enumerate(words_list):
                w_xmin = round(0.08 + (w_i / (n_w + 1)) * 0.80, 4)
                w_xmax = round(w_xmin + (0.70 / (n_w + 1)), 4)
                words_data.append(
                    WordBox(
                        word_id=f"p1_l{i+1}_w{w_i+1}",
                        text=word,
                        confidence=0.95,
                        bbox=[ymin, w_xmin, ymax, min(0.96, w_xmax)],
                    )
                )
            lines_data.append(
                LineBox(
                    line_id=f"p1_l{i+1}",
                    text=line_text,
                    confidence=0.95,
                    bbox=[ymin, 0.08, ymax, 0.94],
                    words=words_data,
                )
            )

        elapsed_ms = (time.time() - t0) * 1000.0 + self.latency_ms
        return RecognitionResponse(
            document_id=f"doc_{uuid.uuid4().hex[:8]}",
            filename=Path(filename).name,
            total_pages=1,
            pages=[
                PageResult(
                    page_number=1,
                    width=width,
                    height=height,
                    full_text="\n".join(full_text_lines),
                    mean_confidence=0.95,
                    lines=lines_data,
                )
            ],
            processing_time_ms=round(elapsed_ms, 2),
        )

    def recognize_pdf(
        self, file_bytes: bytes, filename: str = "document.pdf"
    ) -> RecognitionResponse:
        """Recognize multi-page PDF document deterministically."""
        t0 = time.time()
        self._check_fault_injection()

        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("InvalidPDFError: Input PDF is empty (0 bytes).")

        if not file_bytes.startswith(b"%PDF"):
            raise ValueError("InvalidPDFSignatureError: File does not start with %PDF header.")

        # Check corrupted PDF fixture
        if b"BROKEN_BYTES" in file_bytes or b"GARBAGE" in file_bytes:
            raise ValueError("CorruptedPDFError: PDF structure or xref table is damaged.")

        entry = self._lookup_manifest_by_filename(filename)

        # Check if pypdfium2 is installed for real rasterization count
        page_count = 1
        page_sizes: List[Tuple[int, int]] = []
        try:
            import pypdfium2
            pdf = pypdfium2.PdfDocument(file_bytes)
            page_count = len(pdf)
            for p in pdf:
                page_sizes.append((int(p.get_width() * 2), int(p.get_height() * 2)))
        except Exception:
            # Fallback estimation
            if entry and "pages" in entry:
                page_count = entry["pages"]
            else:
                page_count = max(1, file_bytes.count(b"/Page\n") + file_bytes.count(b"/Page "))

        pages_result: List[PageResult] = []

        if entry and "page_details" in entry:
            for p_info in entry["page_details"]:
                p_num = p_info["page_number"]
                p_w = p_info.get("width", 1200)
                p_h = p_info.get("height", 1600)
                p_text = p_info.get("text", "")
                p_lines_raw = [l for l in p_text.split("\n") if l.strip()]
                num_lines = max(1, len(p_lines_raw))

                lines_data = []
                for l_idx, lt in enumerate(p_lines_raw):
                    ymin = round(0.08 + (l_idx / (num_lines + 1)) * 0.78, 4)
                    ymax = round(ymin + (0.65 / (num_lines + 1)), 4)
                    words_raw = lt.split()
                    num_words = max(1, len(words_raw))
                    words_data = []
                    for w_i, w in enumerate(words_raw):
                        w_xmin = round(0.08 + (w_i / (num_words + 1)) * 0.80, 4)
                        w_xmax = round(w_xmin + (0.70 / (num_words + 1)), 4)
                        words_data.append(
                            WordBox(
                                word_id=f"p{p_num}_l{l_idx+1}_w{w_i+1}",
                                text=w,
                                confidence=0.97,
                                bbox=[ymin, w_xmin, ymax, min(0.96, w_xmax)],
                            )
                        )
                    lines_data.append(
                        LineBox(
                            line_id=f"p{p_num}_l{l_idx+1}",
                            text=lt,
                            confidence=0.97,
                            bbox=[ymin, 0.08, ymax, 0.94],
                            words=words_data,
                        )
                    )

                pages_result.append(
                    PageResult(
                        page_number=p_num,
                        width=p_w,
                        height=p_h,
                        full_text=p_text,
                        mean_confidence=0.97,
                        lines=lines_data,
                    )
                )
        else:
            for p_idx in range(page_count):
                p_num = p_idx + 1
                p_w, p_h = page_sizes[p_idx] if p_idx < len(page_sizes) else (1200, 1600)
                p_text = f"Consultation Document Page {p_num}\nPatient record details line {p_num}"
                lines_data = [
                    LineBox(
                        line_id=f"p{p_num}_l1",
                        text=f"Consultation Document Page {p_num}",
                        confidence=0.96,
                        bbox=[0.1, 0.1, 0.2, 0.8],
                        words=[
                            WordBox(
                                word_id=f"p{p_num}_l1_w{w_i+1}",
                                text=w,
                                confidence=0.96,
                                bbox=[0.1, round(0.1 + w_i * 0.15, 3), 0.2, round(0.22 + w_i * 0.15, 3)],
                            )
                            for w_i, w in enumerate(f"Consultation Document Page {p_num}".split())
                        ],
                    )
                ]
                pages_result.append(
                    PageResult(
                        page_number=p_num,
                        width=p_w,
                        height=p_h,
                        full_text=p_text,
                        mean_confidence=0.96,
                        lines=lines_data,
                    )
                )

        elapsed_ms = (time.time() - t0) * 1000.0 + self.latency_ms
        return RecognitionResponse(
            document_id=f"doc_{uuid.uuid4().hex[:8]}",
            filename=Path(filename).name,
            total_pages=len(pages_result),
            pages=pages_result,
            processing_time_ms=round(elapsed_ms, 2),
        )

    def recognize(
        self, file_bytes: bytes, filename: str = "document.png"
    ) -> RecognitionResponse:
        """Route to image or PDF recognizer based on extension."""
        if filename.lower().endswith(".pdf") or file_bytes.startswith(b"%PDF"):
            return self.recognize_pdf(file_bytes, filename)
        return self.recognize_image(file_bytes, filename)

    # -----------------------------------------------------------------------
    # Asynchronous Job Management
    # -----------------------------------------------------------------------

    def submit_job(self, file_bytes: bytes, filename: str) -> str:
        """Submit background recognition job and return job_id."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        self._jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "file_bytes": file_bytes,
            "status": "QUEUED",
            "progress": 0.0,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "step": 0,
        }
        return job_id

    def get_job_status(self, job_id: str) -> JobStatusResponse:
        """Get current job status and auto-advance state for testing."""
        if job_id not in self._jobs:
            raise KeyError(f"Job {job_id} not found.")

        job = self._jobs[job_id]
        now = datetime.now(timezone.utc).isoformat()

        # Step simulation
        if job["status"] == "QUEUED":
            job["status"] = "PROCESSING"
            job["progress"] = 0.33
            job["step"] = 1
            job["updated_at"] = now
        elif job["status"] == "PROCESSING" and job["step"] == 1:
            job["status"] = "PROCESSING"
            job["progress"] = 0.66
            job["step"] = 2
            job["updated_at"] = now
        elif job["status"] == "PROCESSING" and job["step"] >= 2:
            try:
                result = self.recognize(job["file_bytes"], job["filename"])
                job["result"] = result
                job["status"] = "COMPLETED"
                job["progress"] = 1.0
                job["step"] = 3
                job["updated_at"] = now
            except Exception as e:
                job["status"] = "FAILED"
                job["error"] = str(e)
                job["updated_at"] = now

        return JobStatusResponse(
            job_id=job["job_id"],
            filename=job["filename"],
            status=job["status"],
            progress=job["progress"],
            result=job.get("result"),
            error=job.get("error"),
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )

    async def stream_job_events(self, job_id: str) -> AsyncIterator[str]:
        """Stream SSE formatted events for a job."""
        if job_id not in self._jobs:
            yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        job = self._jobs[job_id]

        # 1. Queued event
        yield f"event: progress\ndata: {json.dumps({'job_id': job_id, 'status': 'QUEUED', 'progress': 0.0})}\n\n"
        await asyncio.sleep(0.01)

        # 2. Processing event
        yield f"event: progress\ndata: {json.dumps({'job_id': job_id, 'status': 'PROCESSING', 'progress': 0.5})}\n\n"
        await asyncio.sleep(0.01)

        # 3. Complete or Error event
        try:
            result = self.recognize(job["file_bytes"], job["filename"])
            job["result"] = result
            job["status"] = "COMPLETED"
            job["progress"] = 1.0
            yield f"event: complete\ndata: {result.model_dump_json()}\n\n"
        except Exception as e:
            job["status"] = "FAILED"
            job["error"] = str(e)
            yield f"event: error\ndata: {json.dumps({'job_id': job_id, 'error': str(e)})}\n\n"

    # -----------------------------------------------------------------------
    # Mock Preprocessing Routines (Deskew, CLAHE, Sauvola, Segmentation)
    # -----------------------------------------------------------------------

    def mock_deskew(
        self, image: Union[np.ndarray, Image.Image], angle_hint: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """Simulate document deskewing via Hough transform."""
        if isinstance(image, Image.Image):
            arr = np.array(image)
        else:
            arr = image.copy()

        detected_angle = angle_hint if angle_hint is not None else 12.5
        pil_img = Image.fromarray(arr)
        deskewed = pil_img.rotate(-detected_angle, expand=True, fillcolor=(255, 255, 255))
        return np.array(deskewed), detected_angle

    def mock_clahe(
        self, image: Union[np.ndarray, Image.Image], clip_limit: float = 2.0
    ) -> np.ndarray:
        """Simulate Contrast Limited Adaptive Histogram Equalization."""
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = Image.fromarray(image).convert("RGB")

        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(1.4)
        return np.array(enhanced)

    def mock_binarize(
        self, image: Union[np.ndarray, Image.Image], method: str = "sauvola"
    ) -> np.ndarray:
        """Simulate adaptive Sauvola thresholding."""
        if isinstance(image, Image.Image):
            gray = image.convert("L")
        else:
            gray = Image.fromarray(image).convert("L")

        arr = np.array(gray)
        thresh = np.mean(arr) * 0.85
        binary = (arr > thresh).astype(np.uint8) * 255
        return binary

    def mock_segment_lines(
        self, image: Union[np.ndarray, Image.Image]
    ) -> List[LineCrop]:
        """Simulate HPP & seam-carving line segmentation."""
        if isinstance(image, Image.Image):
            arr = np.array(image)
        else:
            arr = image
        h, w = arr.shape[:2]

        num_lines = max(2, min(6, h // 150))
        lines: List[LineCrop] = []

        for i in range(num_lines):
            ymin = round(i / num_lines, 4)
            ymax = round((i + 1) / num_lines, 4)
            y0 = int(ymin * h)
            y1 = int(ymax * h)
            patch = arr[y0:y1, :]
            lines.append(
                LineCrop(
                    line_index=i,
                    image=patch,
                    bbox=[ymin, 0.05, ymax, 0.95],
                )
            )
        return lines

    # -----------------------------------------------------------------------
    # Mock Training Loop & Evaluation Harness
    # -----------------------------------------------------------------------

    def mock_train_loop(
        self, dataset_path: str, epochs: int = 3, out_dir: str = "checkpoints"
    ) -> Dict[str, Any]:
        """Simulate TrOCR fine-tuning loop on Apple Silicon MPS."""
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        csv_path = out_p / "losses.csv"
        plot_path = out_p / "loss_curve.png"
        best_ckpt = out_p / "best_model.pt"

        loss_history = []
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_cer,val_wer\n")
            for epoch in range(1, epochs + 1):
                t_loss = round(1.85 / epoch + 0.12 * np.random.random(), 4)
                v_cer = round(0.18 / epoch + 0.02 * np.random.random(), 4)
                v_wer = round(0.35 / epoch + 0.04 * np.random.random(), 4)
                f.write(f"{epoch},{t_loss},{v_cer},{v_wer}\n")
                loss_history.append({"epoch": epoch, "loss": t_loss, "cer": v_cer, "wer": v_wer})

                # Write checkpoint
                ckpt_path = out_p / f"checkpoint_epoch_{epoch}.pt"
                with open(ckpt_path, "wb") as cf:
                    cf.write(b"MOCK_PYTORCH_WEIGHTS_EPOCH_" + str(epoch).encode())

        # Best model weight
        with open(best_ckpt, "wb") as bf:
            bf.write(b"MOCK_PYTORCH_BEST_MODEL_WEIGHTS")

        # Generate loss curve plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot([x["epoch"] for x in loss_history], [x["loss"] for x in loss_history], "b-o", label="Train Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss", color="b")
            ax2 = ax1.twinx()
            ax2.plot([x["epoch"] for x in loss_history], [x["cer"] for x in loss_history], "r--s", label="Val CER")
            ax2.set_ylabel("CER", color="r")
            plt.title("Mock TrOCR Training Loss & CER")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except Exception:
            Image.new("RGB", (600, 400), "white").save(plot_path)

        return {
            "epochs": epochs,
            "final_loss": loss_history[-1]["loss"],
            "final_cer": loss_history[-1]["cer"],
            "loss_csv": str(csv_path),
            "loss_plot": str(plot_path),
            "best_checkpoint": str(best_ckpt),
        }

    def mock_evaluate(
        self, predictions: List[str], ground_truth: List[str]
    ) -> Dict[str, Any]:
        """Compute CER, WER, and throughput metrics across prediction sets."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and Ground Truth lists must have equal length.")

        cer_scores = [compute_cer(p, g) for p, g in zip(predictions, ground_truth)]
        wer_scores = [compute_wer(p, g) for p, g in zip(predictions, ground_truth)]

        latencies = [15.0 + 5.0 * np.random.random() for _ in predictions]

        return {
            "sample_count": len(predictions),
            "mean_cer": round(float(np.mean(cer_scores)), 4) if cer_scores else 0.0,
            "mean_wer": round(float(np.mean(wer_scores)), 4) if wer_scores else 0.0,
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
            "p90_latency_ms": round(float(np.percentile(latencies, 90)), 2) if latencies else 0.0,
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0,
            "throughput_samples_per_sec": round(len(predictions) / (sum(latencies) / 1000.0), 2) if latencies else 0.0,
        }


# ---------------------------------------------------------------------------
# FastAPI Test Application Factory
# ---------------------------------------------------------------------------

def create_mock_app(engine: Optional[MockInferenceEngine] = None) -> FastAPI:
    """Create a FastAPI application routing to the MockInferenceEngine."""
    app = FastAPI(title="Handwriting Recognition Mock API", version="1.0.0")
    eng = engine or MockInferenceEngine()

    @app.get("/v1/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "device": "mock",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/v1/recognize")
    async def recognize(request: Request) -> Any:
        form = await request.form()
        upload = form.get("file")
        if not upload or not hasattr(upload, "read"):
            raise HTTPException(status_code=422, detail="Missing file in multipart upload")
        file_bytes = await upload.read()
        filename = getattr(upload, "filename", "uploaded_file.png") or "uploaded_file.png"
        try:
            result = eng.recognize(file_bytes, filename)
            return result.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/jobs")
    async def create_job(request: Request) -> Dict[str, Any]:
        form = await request.form()
        upload = form.get("file")
        if not upload or not hasattr(upload, "read"):
            raise HTTPException(status_code=422, detail="Missing file in multipart upload")
        file_bytes = await upload.read()
        if not file_bytes:
            raise HTTPException(status_code=422, detail="Empty file payload")
        filename = getattr(upload, "filename", "job_file.pdf") or "job_file.pdf"
        job_id = eng.submit_job(file_bytes, filename)
        return {"job_id": job_id, "status": "QUEUED"}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> Any:
        try:
            status = eng.get_job_status(job_id)
            return status.model_dump()
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    @app.get("/v1/jobs/{job_id}/stream")
    @app.get("/v1/jobs/{job_id}/events")
    async def stream_job(job_id: str) -> Any:
        return StreamingResponse(
            eng.stream_job_events(job_id),
            media_type="text/event-stream",
        )

    return app


if __name__ == "__main__":
    engine = MockInferenceEngine()
    print(f"Mock Engine initialized with {len(engine.manifest.get('fixtures', {}))} fixtures.")
    sample = engine.recognize_image(
        Path("tests/fixtures/sample_clean_handwriting.png").read_bytes(),
        "sample_clean_handwriting.png",
    )
    print(f"Recognized sample: {sample.document_id} with {len(sample.pages[0].lines)} lines.")

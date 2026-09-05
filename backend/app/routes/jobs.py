"""
backend/app/routes/jobs.py
Asynchronous multi-page background job submission, status polling, and SSE streaming.
"""

from __future__ import annotations
import asyncio
import base64
from datetime import datetime, timezone
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from starlette.responses import StreamingResponse

from backend.app.config import Settings, get_settings
from backend.app.engine import get_engine
from backend.app.schemas import (
    JobStatusEnum,
    JobStatusResponse,
    JobSubmissionResponse,
    RecognitionOptions,
    RecognitionResponse,
    RecognizeJsonRequest,
)

router = APIRouter()
logger = logging.getLogger("handwriting_backend.jobs")


class InMemoryJobStore:
    """Thread-safe and asyncio-safe in-memory job state store."""

    def __init__(self, max_concurrent: int = 4, retention_seconds: int = 3600):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._retention = retention_seconds
        self._poll_counts: Dict[str, int] = {}

    def create_job(
        self,
        filename: str,
        file_bytes: bytes,
        options: Optional[RecognitionOptions] = None,
    ) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        self._jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "file_bytes": file_bytes,
            "options": options,
            "status": "QUEUED",
            "progress": 0.0,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        self._poll_counts[job_id] = 0
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        self.expire_jobs()
        job = self._jobs.get(job_id)
        if job is None:
            return None

        return job

    def expire_jobs(self) -> None:
        now = datetime.now(timezone.utc)
        for key, job in list(self._jobs.items()):
            if job["status"] in ("COMPLETED", "FAILED") and (now - datetime.fromisoformat(job["updated_at"])).total_seconds() >= self._retention:
                del self._jobs[key]
                self._poll_counts.pop(key, None)

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        result: Optional[RecognitionResponse] = None,
        error: Optional[str] = None,
    ) -> None:
        if job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        now = datetime.now(timezone.utc).isoformat()
        if status is not None:
            job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        job["updated_at"] = now
        if status in ("COMPLETED", "FAILED"):
            job["file_bytes"] = b""

    async def execute_background_job(self, job_id: str) -> None:
        """Worker task processing document recognition under concurrency throttle."""
        async with self._semaphore:
            job = self._jobs.get(job_id)
            if not job or job["status"] != "QUEUED":
                return

            self.update_job(job_id, status="PROCESSING", progress=0.2)
            try:
                engine = get_engine()
                result = await asyncio.to_thread(
                    engine.recognize,
                    job["file_bytes"],
                    job["filename"],
                    job.get("options"),
                )
                self.update_job(job_id, status="COMPLETED", progress=1.0, result=result)
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
                self.update_job(job_id, status="FAILED", progress=0.0, error=str(e))


job_store = InMemoryJobStore(get_settings().MAX_CONCURRENT_JOBS, get_settings().JOB_RETENTION_SECONDS)


@router.post("/jobs", response_model=JobStatusResponse)
async def submit_job(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    deskew: bool = Query(default=True),
    enhance_contrast: bool = Query(default=True),
    binarization_method: str = Query(default="sauvola"),
    extract_words: bool = Query(default=True),
    dpi: int = Query(default=300),
    beam_width: int = Query(default=5, ge=1, le=16),
    rescore: bool = Query(default=False),
    processing_mode: str = Query(default="local", pattern="^(local|cloud)$"),
    turbo: Optional[bool] = Query(default=None),
    settings: Settings = Depends(get_settings),
) -> JobStatusResponse:
    """Submit document for asynchronous background handwriting recognition."""
    content_type = request.headers.get("content-type", "")
    file_bytes: Optional[bytes] = None
    filename: str = "document.pdf"
    options = RecognitionOptions(
        deskew=deskew,
        enhance_contrast=enhance_contrast,
        binarization_method=binarization_method,
        extract_words=extract_words,
        dpi=dpi,
        beam_width=beam_width,
        rescore=rescore,
        processing_mode=processing_mode,
        turbo=turbo,
    )

    if "application/json" in content_type:
        try:
            body = await request.json()
            req = RecognizeJsonRequest(**body)
            try:
                file_bytes = base64.b64decode(req.file_base64)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid base64 payload: {e}",
                )
            filename = req.filename or "upload.pdf"
            if req.options:
                options = req.options
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse JSON body: {e}",
            )
    elif file is not None:
        filename = file.filename or "upload.pdf"
        file_bytes = await file.read()
    else:
        try:
            form = await request.form()
            upload_item = form.get("file")
            if upload_item is not None and hasattr(upload_item, "read"):
                filename = getattr(upload_item, "filename", "upload.pdf") or "upload.pdf"
                file_bytes = await upload_item.read()
            elif "file_base64" in form:
                file_bytes = base64.b64decode(str(form["file_base64"]))
                filename = str(form.get("filename", "upload.pdf"))
        except Exception:
            pass

    if file_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing file. Please upload a file or supply file_base64.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty file payload (0 bytes).",
        )

    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File exceeds maximum allowed size ({settings.MAX_IMAGE_SIZE_MB}MB).",
        )

    job_id = job_store.create_job(filename, file_bytes, options)
    asyncio.create_task(job_store.execute_background_job(job_id))

    job = job_store._jobs[job_id]
    return JobStatusResponse(**job)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Retrieve current processing status and result of background job."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    return JobStatusResponse(**job)


async def _generate_sse_stream(job_id: str) -> AsyncIterator[str]:
    """Generate Server-Sent Events for job progress and completion."""
    last_state = None
    while True:
        job = job_store.get_job(job_id)
        if not job:
            yield f"event: error\ndata: {json.dumps({'error': 'Job expired or interrupted; submit again'})}\n\n"
            return
        if job["status"] == "COMPLETED":
            result = job["result"]
            data = result.model_dump() if hasattr(result, "model_dump") else result
            yield f"event: complete\ndata: {json.dumps(data)}\n\n"
            return
        if job["status"] == "FAILED":
            yield f"event: error\ndata: {json.dumps({'error': job['error']})}\n\n"
            return
        state = (job["status"], job["progress"])
        if state != last_state:
            yield f"event: progress\ndata: {json.dumps({'job_id': job_id, 'status': state[0], 'progress': state[1]})}\n\n"
            last_state = state
        await asyncio.sleep(0.25)


@router.get("/jobs/{job_id}/stream")
@router.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    """Stream real-time SSE progress and completion events for a job."""
    return StreamingResponse(
        _generate_sse_stream(job_id),
        media_type="text/event-stream",
    )

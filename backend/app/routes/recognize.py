"""
backend/app/routes/recognize.py
Synchronous handwriting recognition endpoint for image and PDF documents.
"""

from __future__ import annotations
import base64
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from PIL import UnidentifiedImageError

from backend.app.config import Settings, get_settings
from backend.app.engine import (
    CorruptDocumentError,
    DocumentLoadingError,
    EmptyDocumentError,
    InferenceEngine,
    PasswordProtectedPDFError,
    UnsupportedFormatError,
    get_engine,
)
from backend.app.schemas import (
    RecognitionOptions,
    RecognitionResponse,
    RecognizeJsonRequest,
)

router = APIRouter()
logger = logging.getLogger("handwriting_backend.recognize")


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_document(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    deskew: bool = Query(default=True),
    enhance_contrast: bool = Query(default=True),
    binarization_method: str = Query(default="sauvola"),
    extract_words: bool = Query(default=True),
    dpi: int = Query(default=300),
    beam_width: int = Query(default=5, ge=1, le=16),
    rescore: bool = Query(default=True),
    settings: Settings = Depends(get_settings),
) -> RecognitionResponse:
    """
    Synchronously transcribe a single/multi-page image or PDF.
    Accepts both multipart/form-data (`file` field) and JSON (`file_base64` field).
    """
    engine = get_engine()
    content_type = request.headers.get("content-type", "")

    file_bytes: Optional[bytes] = None
    filename: str = "document.png"
    options = RecognitionOptions(
        deskew=deskew,
        enhance_contrast=enhance_contrast,
        binarization_method=binarization_method,
        extract_words=extract_words,
        dpi=dpi,
        beam_width=beam_width,
        rescore=rescore,
    )

    # 1. Check if request is JSON body with base64 payload
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
            filename = req.filename or "upload.png"
            if req.options:
                options = req.options
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse JSON body: {e}",
            )

    # 2. Check if multipart/form-data
    elif file is not None:
        filename = file.filename or "upload.png"
        file_bytes = await file.read()

    else:
        # Check raw form fields if UploadFile injection was bypassed
        try:
            form = await request.form()
            upload_item = form.get("file")
            if upload_item is not None and hasattr(upload_item, "read"):
                filename = getattr(upload_item, "filename", "upload.png") or "upload.png"
                file_bytes = await upload_item.read()
            elif "file_base64" in form:
                file_bytes = base64.b64decode(str(form["file_base64"]))
                filename = str(form.get("filename", "upload.png"))
        except Exception:
            pass

    # 3. Validate presence of file payload
    if file_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing file. Please provide a file upload or file_base64 JSON payload.",
        )

    # 4. Validate non-empty payload
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="InvalidImageError: Input image is empty (0 bytes).",
        )

    # 5. Validate file size
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File exceeds maximum allowed size ({settings.MAX_IMAGE_SIZE_MB}MB).",
        )

    # 6. Execute inference
    try:
        response = engine.recognize(file_bytes, filename=filename, options=options)
        return response
    except (EmptyDocumentError, ValueError, UnidentifiedImageError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except (CorruptDocumentError, PasswordProtectedPDFError, UnsupportedFormatError, DocumentLoadingError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


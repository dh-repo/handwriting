"""
backend/app/main.py
FastAPI ASGI application entrypoint with lifespan, CORS, and sanitized error handlers.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Any, AsyncIterator
import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import get_settings
from backend.app.engine import get_engine
from backend.app.routes import feedback, health, jobs, recognize

logger = logging.getLogger("handwriting_backend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """ASGI lifespan event context manager for startup initialization and graceful shutdown."""
    settings = get_settings()
    device = settings.resolve_device()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Resolved execution device: {device} (USE_MOCK_ENGINE={settings.USE_MOCK_ENGINE})")

    # Warm up engine singleton on startup
    try:
        engine = get_engine()
        logger.info(f"Inference engine initialized (mode={engine.mode}, device={engine.device})")
    except Exception as e:
        logger.warning(f"Engine initialization warning during startup: {e}")

    yield

    # Clean shutdown
    logger.info("Shutting down backend service...")
    if device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Scalable Inference Service for Handwritten Document & Prescription OCR",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "Validation Error",
                "detail": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Error",
                "detail": exc.detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred during document processing.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # Include Routers
    app.include_router(health.router, prefix="/v1", tags=["Health"])
    app.include_router(recognize.router, prefix="/v1", tags=["Recognition"])
    app.include_router(jobs.router, prefix="/v1", tags=["Async Jobs"])
    app.include_router(feedback.router, prefix="/v1", tags=["Feedback"])

    return app


app = create_app()

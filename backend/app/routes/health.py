"""
backend/app/routes/health.py
Health check and system diagnostics endpoint.
"""

from datetime import datetime, timezone
import os
import resource
import sys
from typing import List, Optional
import torch
from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.engine import get_engine
from backend.app.schemas import HealthResponse, LiveResponse

router = APIRouter()


def _get_memory_usage_mb() -> float:
    """Return process RSS memory usage in megabytes."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            # On macOS, ru_maxrss is in bytes
            return round(usage / (1024.0 * 1024.0), 2)
        else:
            # On Linux, ru_maxrss is in kilobytes
            return round(usage / 1024.0, 2)
    except Exception:
        return 0.0


@router.get("/live", response_model=LiveResponse)
async def liveness() -> LiveResponse:
    """Return process liveness without touching the inference engine."""
    return LiveResponse(timestamp=datetime.now(timezone.utc).isoformat())


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check system health, hardware acceleration device, memory usage, and loaded models.
    """
    settings = get_settings()
    engine = get_engine()

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = engine.mode if engine.mode == "mock" else str(engine.device)

    loaded_models: List[str] = []
    if engine.model is not None:
        loaded_models.append(engine.model_name)

    rescorer_active = bool(
        getattr(engine, "enable_rescorer", True)
        and (getattr(engine, "rescorer", None) is not None or engine.mode == "mock")
    )

    return HealthResponse(
        status="healthy",
        device=device,
        version=settings.APP_VERSION,
        memory_usage_mb=_get_memory_usage_mb(),
        loaded_models=loaded_models,
        mps_available=mps_available,
        execution_mode=settings.resolve_device(),
        rescorer_active=rescorer_active,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

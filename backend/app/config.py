"""
backend/app/config.py
Configuration management using Pydantic v2 BaseSettings.
"""

from __future__ import annotations
from functools import lru_cache
from typing import List, Literal, Optional
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application runtime configuration with environment variable support."""

    APP_NAME: str = Field(default="Handwriting Recognition Inference API", description="API Name")
    APP_VERSION: str = Field(default="1.0.0", description="API Version")
    DEBUG: bool = Field(default=False, description="Debug mode flag")

    # Server settings
    HOST: str = Field(default="0.0.0.0", description="ASGI host")
    PORT: int = Field(default=8000, description="ASGI port")
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], description="CORS allowed origins")

    # Model & Engine settings
    DEVICE: str = Field(default="auto", description="Execution device: auto, mps, cuda, cpu, mock")
    EXECUTION_MODE: Optional[str] = Field(default=None, description="Optional execution mode alias")
    MODEL_NAME_OR_PATH: str = Field(
        default="microsoft/trocr-large-handwritten",
        description="HuggingFace model identifier or local checkpoint path",
    )
    STAGE2_CHECKPOINT_PATH: Optional[str] = Field(
        default="checkpoints/stage2_doctor_specialization/best_model",
        description="Stage 2 Doctor fine-tuned model path",
    )
    STAGE1_CHECKPOINT_PATH: Optional[str] = Field(
        default="checkpoints/stage1_general_adaptation/best_model",
        description="Stage 1 General adaptation model path",
    )
    FALLBACK_CHECKPOINT_PATH: Optional[str] = Field(
        default=None,
        description="Root best model checkpoint",
    )
    USE_FP16: bool = Field(default=False, description="Enable float16 mixed precision on MPS/CUDA")
    USE_MOCK_ENGINE: bool = Field(
        default=False,
        description="Force deterministic mock engine for testing/preview",
    )
    DEFAULT_DPI: int = Field(default=300, description="PDF rasterization DPI")
    EXTRACT_WORDS: bool = Field(default=True, description="Enable word bounding box segmentation")
    MAX_IMAGE_SIZE_MB: int = Field(default=50, description="Max allowed upload size in MB")
    MAX_PDF_PAGES: int = Field(default=50, description="Max allowed PDF pages")

    # Beam Search & Rescorer settings
    ENABLE_RESCORER: bool = Field(default=True, description="Enable RxNorm beam rescoring")
    BEAM_WIDTH: int = Field(default=1, ge=1, le=16, description="Beam search width (K candidates)")
    NUM_RETURN_SEQUENCES: int = Field(default=1, ge=1, le=16, description="Candidate beam count")
    VOCAB_DIR: str = Field(
        default="data/reference_handwriting/vocabularies",
        description="Path to pharmaceutical and clinical vocabularies directory",
    )
    LEXICON_PATH: Optional[str] = Field(
        default="data/reference_handwriting/vocabularies",
        description="Lexicon directory or medication JSON path",
    )
    CONFUSION_MATRIX_PATH: Optional[str] = Field(
        default=None,
        description="Optional path to custom confusion matrix JSON",
    )
    RESCORER_WEIGHT: float = Field(default=1.0, ge=0.0, description="Primary lexicon score weight lambda_1")
    LAMBDA_LEXICON: float = Field(default=1.0, ge=0.0, description="Lexicon prior weight lambda_1")
    CONTEXT_WEIGHT: float = Field(default=0.8, ge=0.0, description="Clinical context weight lambda_2")
    LAMBDA_CONTEXT: float = Field(default=0.8, ge=0.0, description="Clinical context weight lambda_2")
    CONFUSION_WEIGHT: float = Field(default=0.5, ge=0.0, description="Visual confusion penalty weight lambda_3")
    LAMBDA_CONFUSION: float = Field(default=0.5, ge=0.0, description="Visual confusion penalty weight lambda_3")
    MAX_SAFE_MG: float = Field(default=4000.0, description="Upper bound for realistic single dose")

    # Apple Silicon MPS & Batching settings
    LINE_BATCH_SIZE: int = Field(default=16, ge=1, le=64, description="Line crop micro-batch size")
    MPS_HIGH_WATERMARK_RATIO: float = Field(default=0.85, description="MPS memory high watermark ratio")
    MPS_EMPTY_CACHE_INTERVAL: int = Field(default=10, description="Steps between empty_cache calls")

    # Job Queue & Concurrency settings
    JOB_RETENTION_SECONDS: int = Field(default=3600, description="Async job TTL in seconds")
    MAX_CONCURRENT_JOBS: int = Field(default=4, description="Max concurrent background jobs")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def resolve_device(self) -> str:
        """
        Resolve the target execution device.
        Returns one of: 'mps', 'cuda', 'cpu', 'mock'.
        """
        if self.USE_MOCK_ENGINE:
            return "mock"

        mode = (self.EXECUTION_MODE or self.DEVICE).lower()

        if mode == "mock":
            return "mock"
        if mode == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if mode == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if mode == "cpu":
            return "cpu"

        # Auto detection
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def resolve_model_path(self) -> str:
        """
        Resolve model path in prioritized order:
        1. Explicitly configured path if exists on disk
        2. Stage 2 doctor specialization checkpoint
        3. Fallback root checkpoint (checkpoints/best_model.pt or checkpoints/best_model_hf)
        4. Stage 1 general adaptation checkpoint
        5. Configured MODEL_NAME_OR_PATH or HuggingFace ID
        """
        from pathlib import Path

        # If explicit path is configured and exists on disk
        if self.MODEL_NAME_OR_PATH and Path(self.MODEL_NAME_OR_PATH).exists():
            return self.MODEL_NAME_OR_PATH

        candidates = [
            self.STAGE2_CHECKPOINT_PATH,
            self.STAGE1_CHECKPOINT_PATH,
            self.FALLBACK_CHECKPOINT_PATH,
        ]
        for cand in candidates:
            if cand:
                p = Path(cand)
                if p.exists():
                    return str(p)
                if Path(f"{cand}_hf").exists():
                    return f"{cand}_hf"
                if Path(f"{cand}.pt").exists():
                    return f"{cand}.pt"

        return self.MODEL_NAME_OR_PATH or "microsoft/trocr-large-handwritten"



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the LRU cache for settings (useful for tests modifying env vars)."""
    get_settings.cache_clear()

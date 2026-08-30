"""
backend/tests/test_config.py
Tests for configuration management, environment variable loading, and device resolution.
"""

import os
from unittest.mock import patch
import pytest
import torch

from backend.app.config import Settings, get_settings, reset_settings_cache


def test_default_settings() -> None:
    """Verify default Settings values."""
    reset_settings_cache()
    settings = get_settings()
    assert settings.APP_NAME == "Handwriting Recognition Inference API"
    assert settings.PORT == 8000
    assert settings.HOST == "0.0.0.0"
    assert settings.MODEL_NAME_OR_PATH == "microsoft/trocr-large-handwritten"
    assert settings.USE_FP16 is False
    assert settings.DEFAULT_DPI == 300
    assert settings.MAX_IMAGE_SIZE_MB >= 25
    assert settings.ENABLE_RESCORER is True
    assert settings.BEAM_WIDTH == 5
    assert settings.NUM_RETURN_SEQUENCES == 5
    assert settings.VOCAB_DIR == "data/reference_handwriting/vocabularies"
    assert settings.RESCORER_WEIGHT == 1.0
    assert settings.LAMBDA_LEXICON == 1.0
    assert settings.CONTEXT_WEIGHT == 0.8
    assert settings.LAMBDA_CONTEXT == 0.8
    assert settings.CONFUSION_WEIGHT == 0.5
    assert settings.LAMBDA_CONFUSION == 0.5
    assert settings.MAX_SAFE_MG == 4000.0
    assert settings.LINE_BATCH_SIZE == 8
    assert settings.MPS_HIGH_WATERMARK_RATIO == 0.85
    assert settings.MPS_EMPTY_CACHE_INTERVAL == 10


def test_environment_variable_overrides() -> None:
    """Verify settings pick up environment variable overrides."""
    reset_settings_cache()
    with patch.dict(os.environ, {
        "PORT": "9090",
        "DEVICE": "cpu",
        "USE_MOCK_ENGINE": "true",
        "DEFAULT_DPI": "150",
        "MODEL_NAME_OR_PATH": "custom/trocr-large",
        "ENABLE_RESCORER": "false",
        "BEAM_WIDTH": "8",
        "RESCORER_WEIGHT": "2.5",
        "LINE_BATCH_SIZE": "16",
    }):
        reset_settings_cache()
        s = get_settings()
        assert s.PORT == 9090
        assert s.DEVICE == "cpu"
        assert s.USE_MOCK_ENGINE is True
        assert s.DEFAULT_DPI == 150
        assert s.MODEL_NAME_OR_PATH == "custom/trocr-large"
        assert s.ENABLE_RESCORER is False
        assert s.BEAM_WIDTH == 8
        assert s.RESCORER_WEIGHT == 2.5
        assert s.LINE_BATCH_SIZE == 16
        assert s.resolve_device() == "mock"


def test_device_resolution_mock() -> None:
    """Verify mock device resolution when requested."""
    s = Settings(USE_MOCK_ENGINE=True)
    assert s.resolve_device() == "mock"

    s2 = Settings(DEVICE="mock", USE_MOCK_ENGINE=False)
    assert s2.resolve_device() == "mock"


def test_device_resolution_cpu() -> None:
    """Verify CPU resolution."""
    s = Settings(DEVICE="cpu", USE_MOCK_ENGINE=False)
    assert s.resolve_device() == "cpu"


def test_device_resolution_mps() -> None:
    """Verify MPS device resolution on Apple Silicon."""
    s = Settings(DEVICE="mps", USE_MOCK_ENGINE=False)
    resolved = s.resolve_device()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        assert resolved == "mps"
    else:
        assert resolved == "cpu"


def test_device_resolution_auto() -> None:
    """Verify auto device resolution picks available hardware."""
    s = Settings(DEVICE="auto", USE_MOCK_ENGINE=False)
    resolved = s.resolve_device()
    assert resolved in ("mps", "cuda", "cpu")


def test_resolve_model_path(tmp_path: pytest.TempPathFactory) -> None:
    """Verify resolve_model_path priority order."""
    # 1. Non-existent paths fall back to MODEL_NAME_OR_PATH
    s = Settings(
        MODEL_NAME_OR_PATH="microsoft/trocr-large-handwritten",
        STAGE2_CHECKPOINT_PATH="non_existent/stage2",
        STAGE1_CHECKPOINT_PATH="non_existent/stage1",
        FALLBACK_CHECKPOINT_PATH="non_existent/fallback",
    )
    assert s.resolve_model_path() == "microsoft/trocr-large-handwritten"

    # 2. Stage 2 checkpoint priority
    stage2_dir = tmp_path / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    (stage2_dir / "config.json").write_text("{}")

    s2 = Settings(
        MODEL_NAME_OR_PATH="microsoft/trocr-large-handwritten",
        STAGE2_CHECKPOINT_PATH=str(stage2_dir),
        STAGE1_CHECKPOINT_PATH="non_existent/stage1",
        FALLBACK_CHECKPOINT_PATH="non_existent/fallback",
    )
    assert s2.resolve_model_path() == str(stage2_dir)

    # 3. Explicit existing MODEL_NAME_OR_PATH priority
    custom_model_dir = tmp_path / "custom_model"
    custom_model_dir.mkdir(parents=True, exist_ok=True)
    s3 = Settings(
        MODEL_NAME_OR_PATH=str(custom_model_dir),
        STAGE2_CHECKPOINT_PATH=str(stage2_dir),
    )
    assert s3.resolve_model_path() == str(custom_model_dir)


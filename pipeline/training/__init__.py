"""
pipeline/training/__init__.py
Clean exports for model fine-tuning subsystem on Apple Silicon.
"""

from pipeline.training.config import (
    CurriculumConfig,
    CurriculumStageConfig,
    TrainingConfig,
    get_default_num_workers,
)
from pipeline.training.curriculum import (
    CurriculumExecutionResult,
    MultiStageCurriculumTrainer,
)
from pipeline.training.dataset import (
    DummyImageProcessor,
    DummyProcessor,
    DummyTokenizer,
    MMapOCRDataset,
    OCRDataCollator,
    OCRDataset,
    create_dummy_processor,
    load_trocr_processor,
)
from pipeline.training.loss_logger import LossLogger
from pipeline.training.prefetcher import AsyncDevicePrefetcher
from pipeline.training.profiler import StepTiming, TrainingStepProfiler
from pipeline.training.train import (
    TrOCRTrainer,
    configure_gradient_checkpointing,
    create_tiny_mock_model,
    freeze_encoder_layers,
    get_autocast_context,
    get_optimal_device,
    load_trocr_model,
)

__all__ = [
    "TrainingConfig",
    "CurriculumStageConfig",
    "CurriculumConfig",
    "get_default_num_workers",
    "MultiStageCurriculumTrainer",
    "CurriculumExecutionResult",
    "OCRDataset",
    "MMapOCRDataset",
    "OCRDataCollator",
    "AsyncDevicePrefetcher",
    "StepTiming",
    "TrainingStepProfiler",
    "TrOCRTrainer",
    "LossLogger",
    "load_trocr_processor",
    "create_dummy_processor",
    "create_tiny_mock_model",
    "get_optimal_device",
    "get_autocast_context",
    "load_trocr_model",
    "configure_gradient_checkpointing",
    "freeze_encoder_layers",
    "DummyTokenizer",
    "DummyImageProcessor",
    "DummyProcessor",
]

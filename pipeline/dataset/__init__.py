"""
pipeline/dataset module initialization.
Exposes synthetic handwriting generators, background generators, IAM/prescription dataset loaders,
and PyTorch dataset/dataloader adapters.
"""

from pipeline.dataset.synthetic_generator import (
    HandwritingFontManager,
    BackgroundGenerator,
    SyntheticHandwritingGenerator,
)
from pipeline.dataset.dataset_loader import (
    HandwritingSample,
    PrescriptionItem,
    MedicalPrescriptionSample,
    IAMDatasetParser,
    MedicalPrescriptionDatasetLoader,
    HandwritingPyTorchDataset,
    StreamingSyntheticDataset,
    handwriting_collate_fn,
    is_synthetic_training_record,
)

__all__ = [
    # Synthetic Generator
    "HandwritingFontManager",
    "BackgroundGenerator",
    "SyntheticHandwritingGenerator",
    # Dataset Models & Parsers
    "HandwritingSample",
    "PrescriptionItem",
    "MedicalPrescriptionSample",
    "IAMDatasetParser",
    "MedicalPrescriptionDatasetLoader",
    # PyTorch Adapters
    "HandwritingPyTorchDataset",
    "StreamingSyntheticDataset",
    "handwriting_collate_fn",
    "is_synthetic_training_record",
]

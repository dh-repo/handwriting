"""
pipeline/preprocessing module initialization.
Exposes document loaders, image enhancers, line and word segmenters, and the unified pipeline.
"""

from pipeline.preprocessing.pdf_loader import (
    PDFLoader,
    load_pdf,
    load_image,
    load_document,
    detect_format,
    DocumentLoadingError,
    CorruptDocumentError,
    PasswordProtectedPDFError,
    EmptyDocumentError,
    UnsupportedFormatError,
)
from pipeline.preprocessing.image_enhancement import (
    ImageEnhancer,
    deskew_image,
    flatten_illumination,
    enhance_contrast,
    binarize_sauvola,
    binarize_otsu,
    adaptive_binarize,
    normalize_image,
    pad_to_size,
    to_rgb,
    to_grayscale,
    normalize_tensor,
)
from pipeline.preprocessing.line_segmenter import (
    LineSegmenter,
    LineCrop,
    WordCrop,
)
from pipeline.preprocessing.pipeline import (
    PreprocessingPipeline,
    PreprocessedPage,
)

__all__ = [
    # PDF and Image Loading
    "PDFLoader",
    "load_pdf",
    "load_image",
    "load_document",
    "detect_format",
    "DocumentLoadingError",
    "CorruptDocumentError",
    "PasswordProtectedPDFError",
    "EmptyDocumentError",
    "UnsupportedFormatError",
    # Image Enhancement & Binarization
    "ImageEnhancer",
    "deskew_image",
    "flatten_illumination",
    "enhance_contrast",
    "binarize_sauvola",
    "binarize_otsu",
    "adaptive_binarize",
    "normalize_image",
    "pad_to_size",
    "to_rgb",
    "to_grayscale",
    "normalize_tensor",
    # Line & Word Segmentation
    "LineSegmenter",
    "LineCrop",
    "WordCrop",
    # Unified Pipeline
    "PreprocessingPipeline",
    "PreprocessedPage",
]

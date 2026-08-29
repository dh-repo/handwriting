"""
Unit and integration tests for pipeline/preprocessing/pipeline.py.
"""

import numpy as np
import pytest

from pipeline.preprocessing.pipeline import PreprocessingPipeline, PreprocessedPage
from pipeline.preprocessing.line_segmenter import LineCrop


def test_pipeline_process_single_image(synthetic_multiline_image: np.ndarray):
    """Test full pipeline processing on a single image array."""
    pipeline = PreprocessingPipeline(
        dpi=300,
        deskew=True,
        enhance_contrast=True,
        binarization_method="sauvola",
        seam_carving=True
    )

    page = pipeline.process_image(synthetic_multiline_image, page_index=0)

    assert isinstance(page, PreprocessedPage)
    assert page.page_index == 0
    assert page.original_image.shape == synthetic_multiline_image.shape
    assert page.enhanced_image.shape == synthetic_multiline_image.shape
    assert page.binarized_image.shape == synthetic_multiline_image.shape[:2]
    assert len(page.lines) >= 1
    assert all(isinstance(l, LineCrop) for l in page.lines)


def test_pipeline_process_document_pdf(temp_pdf_multipage: str):
    """Test full pipeline processing on a multi-page PDF document."""
    pipeline = PreprocessingPipeline(dpi=150)
    pages = pipeline.process_document(temp_pdf_multipage)

    assert len(pages) == 3
    for idx, page in enumerate(pages):
        assert isinstance(page, PreprocessedPage)
        assert page.page_index == idx
        assert page.original_image.ndim == 3
        assert page.enhanced_image.ndim == 3
        assert page.binarized_image.ndim == 2


def test_pipeline_pydantic_model_serialization(synthetic_multiline_image: np.ndarray):
    """Verify PreprocessedPage and LineCrop Pydantic models instantiate and validate."""
    pipeline = PreprocessingPipeline()
    page = pipeline.process_image(synthetic_multiline_image, page_index=1)

    # Check Pydantic fields
    assert page.page_index == 1
    assert hasattr(page, "original_image")
    assert hasattr(page, "enhanced_image")
    assert hasattr(page, "binarized_image")
    assert hasattr(page, "lines")

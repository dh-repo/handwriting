"""
backend/tests/test_engine.py
Unit tests for InferenceEngine, preprocessing integration, and recognition execution.
"""

import pytest

from backend.app.engine import (
    CorruptDocumentError,
    EmptyDocumentError,
    InferenceEngine,
    get_engine,
    reset_engine,
    set_engine,
)
from backend.app.schemas import RecognitionOptions, RecognitionResponse


def test_engine_mock_mode_initialization() -> None:
    """Verify engine initializes cleanly in mock mode."""
    engine = InferenceEngine(execution_mode="mock")
    assert engine.mode == "mock"
    assert engine.model is None


def test_engine_recognize_image(sample_image_bytes: bytes) -> None:
    """Verify recognition on synthetic image returns valid RecognitionResponse."""
    engine = InferenceEngine(execution_mode="mock")
    res = engine.recognize(sample_image_bytes, filename="test.png")

    assert isinstance(res, RecognitionResponse)
    assert res.total_pages == 1
    assert len(res.pages) == 1
    assert res.pages[0].page_number == 1
    assert res.pages[0].width > 0
    assert res.pages[0].height > 0
    assert len(res.pages[0].lines) >= 1
    assert 0.0 <= res.pages[0].mean_confidence <= 1.0

    for line in res.pages[0].lines:
        assert 0.0 <= line.confidence <= 1.0
        ymin, xmin, ymax, xmax = line.bbox
        assert 0.0 <= ymin < ymax <= 1.0
        assert 0.0 <= xmin < xmax <= 1.0
        for w in line.words:
            assert 0.0 <= w.confidence <= 1.0
            w_ymin, w_xmin, w_ymax, w_xmax = w.bbox
            assert 0.0 <= w_ymin < w_ymax <= 1.0
            assert 0.0 <= w_xmin < w_xmax <= 1.0


def test_engine_recognize_multipage_pdf(sample_pdf_bytes: bytes) -> None:
    """Verify recognition on multi-page PDF extracts structured pages."""
    engine = InferenceEngine(execution_mode="mock")
    res = engine.recognize(sample_pdf_bytes, filename="consultation.pdf")

    assert isinstance(res, RecognitionResponse)
    assert res.total_pages >= 2
    assert len(res.pages) == res.total_pages
    for idx, page in enumerate(res.pages):
        assert page.page_number == idx + 1
        assert len(page.lines) >= 1
        assert page.full_text != ""


def test_engine_recognize_formats(
    sample_jpeg_bytes: bytes,
    sample_tiff_bytes: bytes,
    sample_bmp_bytes: bytes,
    sample_webp_bytes: bytes,
) -> None:
    """Verify recognition across multiple image formats."""
    engine = InferenceEngine(execution_mode="mock")

    # JPEG
    res_jpeg = engine.recognize(sample_jpeg_bytes, filename="sample.jpg")
    assert res_jpeg.total_pages == 1
    assert len(res_jpeg.pages[0].lines) >= 1

    # TIFF
    res_tiff = engine.recognize(sample_tiff_bytes, filename="sample.tiff")
    assert res_tiff.total_pages >= 1

    # BMP
    res_bmp = engine.recognize(sample_bmp_bytes, filename="sample.bmp")
    assert res_bmp.total_pages == 1

    # WebP
    res_webp = engine.recognize(sample_webp_bytes, filename="sample.webp")
    assert res_webp.total_pages == 1


def test_engine_empty_input_rejection() -> None:
    """Verify engine raises EmptyDocumentError on empty input."""
    engine = InferenceEngine(execution_mode="mock")
    with pytest.raises(EmptyDocumentError):
        engine.recognize(b"", filename="empty.png")


def test_engine_corrupt_input_rejection(corrupted_image_bytes: bytes) -> None:
    """Verify engine raises error on corrupted image."""
    engine = InferenceEngine(execution_mode="mock")
    with pytest.raises((CorruptDocumentError, Exception)):
        engine.recognize(corrupted_image_bytes, filename="corrupt.png")


def test_engine_singleton_lifecycle() -> None:
    """Verify get_engine, set_engine, and reset_engine lifecycle."""
    reset_engine()
    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2

    custom_e = InferenceEngine(execution_mode="mock")
    set_engine(custom_e)
    assert get_engine() is custom_e

    reset_engine()
    e3 = get_engine()
    assert e3 is not custom_e


def test_engine_rescorer_initialization() -> None:
    """Verify BeamRescorer is initialized with RxNorm lexicon and default parameters."""
    engine = InferenceEngine(execution_mode="mock")
    assert engine.enable_rescorer is True
    assert engine.beam_width == 5
    assert engine.rescorer is not None
    assert hasattr(engine.rescorer, "trie")
    assert len(engine.rescorer.trie) > 0


def test_engine_rescorer_disabled() -> None:
    """Verify rescorer is not initialized when disabled."""
    engine = InferenceEngine(execution_mode="mock", enable_rescorer=False)
    assert engine.enable_rescorer is False
    assert engine.rescorer is None


def test_engine_rescorer_parameter_customization() -> None:
    """Verify custom rescorer weights and parameters are correctly passed."""
    engine = InferenceEngine(
        execution_mode="mock",
        beam_width=8,
        rescorer_weight=2.0,
        context_weight=1.2,
        confusion_weight=0.4,
        max_safe_mg=2500.0,
        line_batch_size=16,
    )
    assert engine.beam_width == 8
    assert engine.rescorer_weight == 2.0
    assert engine.context_weight == 1.2
    assert engine.confusion_weight == 0.4
    assert engine.max_safe_mg == 2500.0
    assert engine.line_batch_size == 16
    assert engine.rescorer is not None
    assert engine.rescorer.lambda_lexicon == 2.0
    assert engine.rescorer.lambda_context == 1.2
    assert engine.rescorer.lambda_confusion == 0.4
    assert engine.rescorer.max_safe_mg == 2500.0


def test_engine_polymorphic_loading_fallback() -> None:
    """Verify nonexistent model path falls back cleanly to mock mode without raising."""
    engine = InferenceEngine(
        execution_mode="cpu",
        model_name_or_path="nonexistent/checkpoint/path/that/does/not/exist",
    )
    assert engine.mode == "mock"
    assert engine.model is None


def test_engine_recognize_with_rescorer_options(sample_image_bytes: bytes) -> None:
    """Verify recognition executes with explicit rescore options."""
    engine = InferenceEngine(execution_mode="mock")
    opts_rescored = RecognitionOptions(beam_width=5, rescore=True)
    res1 = engine.recognize(sample_image_bytes, filename="presc.png", options=opts_rescored)
    assert isinstance(res1, RecognitionResponse)
    assert res1.total_pages == 1
    assert len(res1.pages[0].lines) >= 1

    opts_no_rescore = RecognitionOptions(beam_width=1, rescore=False)
    res2 = engine.recognize(sample_image_bytes, filename="presc.png", options=opts_no_rescore)
    assert isinstance(res2, RecognitionResponse)
    assert res2.total_pages == 1


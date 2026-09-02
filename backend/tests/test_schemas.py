"""
backend/tests/test_schemas.py
Unit tests for Pydantic v2 data contracts and bounding box validation.
"""

import json
import pytest
from pydantic import ValidationError

from backend.app.schemas import (
    ErrorResponse,
    HealthResponse,
    LiveResponse,
    JobStatusEnum,
    JobStatusResponse,
    JobSubmissionResponse,
    LineBox,
    PageResult,
    RecognitionOptions,
    RecognitionResponse,
    RecognizeJsonRequest,
    WordBox,
)


def test_wordbox_valid() -> None:
    """Verify WordBox instantiation with valid coordinates."""
    wb = WordBox(
        word_id="p1_l1_w1",
        text="Amoxicillin",
        confidence=0.965,
        bbox=[0.10, 0.15, 0.18, 0.45],
    )
    assert wb.word_id == "p1_l1_w1"
    assert wb.text == "Amoxicillin"
    assert wb.confidence == 0.965
    assert wb.bbox == [0.10, 0.15, 0.18, 0.45]


def test_wordbox_invalid_confidence() -> None:
    """Verify WordBox rejects confidence outside [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        WordBox(word_id="w1", text="test", confidence=1.2, bbox=[0.1, 0.1, 0.2, 0.3])
    with pytest.raises(ValidationError):
        WordBox(word_id="w1", text="test", confidence=-0.05, bbox=[0.1, 0.1, 0.2, 0.3])


def test_wordbox_invalid_bbox_length() -> None:
    """Verify WordBox rejects bounding boxes not having 4 coordinates."""
    with pytest.raises(ValidationError):
        WordBox(word_id="w1", text="test", confidence=0.9, bbox=[0.1, 0.2, 0.3])
    with pytest.raises(ValidationError):
        WordBox(word_id="w1", text="test", confidence=0.9, bbox=[0.1, 0.2, 0.3, 0.4, 0.5])


def test_wordbox_inverted_bbox() -> None:
    """Verify WordBox rejects inverted vertical or horizontal coordinates."""
    # ymin >= ymax
    with pytest.raises(ValidationError):
        WordBox(word_id="w1", text="test", confidence=0.9, bbox=[0.5, 0.1, 0.2, 0.3])
    # xmin >= xmax
    with pytest.raises(ValidationError):
        WordBox(word_id="w1", text="test", confidence=0.9, bbox=[0.1, 0.8, 0.2, 0.3])


def test_linebox_valid_with_words() -> None:
    """Verify LineBox containing multiple WordBoxes."""
    w1 = WordBox(word_id="p1_l1_w1", text="Take", confidence=0.95, bbox=[0.10, 0.15, 0.18, 0.25])
    w2 = WordBox(word_id="p1_l1_w2", text="1", confidence=0.98, bbox=[0.10, 0.26, 0.18, 0.30])
    w3 = WordBox(word_id="p1_l1_w3", text="tablet", confidence=0.94, bbox=[0.10, 0.31, 0.18, 0.50])

    lb = LineBox(
        line_id="p1_l1",
        text="Take 1 tablet",
        confidence=0.957,
        bbox=[0.10, 0.15, 0.18, 0.50],
        words=[w1, w2, w3],
    )
    assert lb.line_id == "p1_l1"
    assert len(lb.words) == 3


def test_pageresult_hierarchical_structure() -> None:
    """Verify PageResult containing lines and calculating metadata."""
    lb = LineBox(
        line_id="p1_l1",
        text="Amoxicillin 500mg",
        confidence=0.94,
        bbox=[0.1, 0.1, 0.2, 0.6],
        words=[
            WordBox(word_id="p1_l1_w1", text="Amoxicillin", confidence=0.95, bbox=[0.1, 0.1, 0.2, 0.4]),
            WordBox(word_id="p1_l1_w2", text="500mg", confidence=0.93, bbox=[0.1, 0.42, 0.2, 0.6]),
        ],
    )
    page = PageResult(
        page_number=1,
        width=1200,
        height=800,
        full_text="Amoxicillin 500mg",
        mean_confidence=0.94,
        lines=[lb],
    )
    assert page.page_number == 1
    assert page.width == 1200
    assert len(page.lines) == 1


def test_recognition_response_json_roundtrip() -> None:
    """Verify RecognitionResponse serialization and deserialization."""
    resp = RecognitionResponse(
        document_id="doc_test123",
        filename="prescription.png",
        total_pages=1,
        pages=[
            PageResult(
                page_number=1,
                width=1000,
                height=1500,
                full_text="Line 1\nLine 2",
                mean_confidence=0.92,
                lines=[
                    LineBox(line_id="p1_l1", text="Line 1", confidence=0.92, bbox=[0.1, 0.1, 0.2, 0.5]),
                    LineBox(line_id="p1_l2", text="Line 2", confidence=0.92, bbox=[0.25, 0.1, 0.35, 0.5]),
                ],
            )
        ],
        processing_time_ms=123.4,
    )

    dumped_json = resp.model_dump_json()
    loaded_resp = RecognitionResponse.model_validate_json(dumped_json)
    assert loaded_resp.document_id == "doc_test123"
    assert loaded_resp.total_pages == 1
    assert len(loaded_resp.pages[0].lines) == 2


def test_live_response_serialization() -> None:
    """Verify LiveResponse stays a process-only signal."""
    live = LiveResponse(timestamp="2026-08-29T20:00:00Z")
    assert live.status == "live"
    dumped = live.model_dump()
    assert "loaded_models" not in dumped


def test_health_response_serialization() -> None:
    """Verify HealthResponse structure and serialization."""
    hr = HealthResponse(
        status="healthy",
        device="mps",
        version="1.0.0",
        memory_usage_mb=145.2,
        loaded_models=["microsoft/trocr-base-handwritten"],
        mps_available=True,
        execution_mode="mps",
        rescorer_active=True,
        timestamp="2026-08-26T20:00:00Z",
    )
    assert hr.status == "healthy"
    assert hr.device == "mps"
    assert hr.mps_available is True
    assert hr.rescorer_active is True


def test_recognition_options_defaults() -> None:
    """Verify RecognitionOptions defaults for beam search and rescoring."""
    ro = RecognitionOptions()
    assert ro.deskew is True
    assert ro.enhance_contrast is True
    assert ro.binarization_method == "sauvola"
    assert ro.extract_words is True
    assert ro.dpi == 300
    assert ro.beam_width == 10
    assert ro.rescore is False


def test_job_schemas() -> None:
    """Verify JobSubmissionResponse and JobStatusResponse models."""
    sub = JobSubmissionResponse(
        job_id="job_abc123",
        status=JobStatusEnum.QUEUED,
        filename="doc.pdf",
        created_at="2026-08-26T20:00:00Z",
    )
    assert sub.status == JobStatusEnum.QUEUED

    st = JobStatusResponse(
        job_id="job_abc123",
        filename="doc.pdf",
        status=JobStatusEnum.PROCESSING,
        progress=0.45,
        created_at="2026-08-26T20:00:00Z",
        updated_at="2026-08-26T20:00:02Z",
    )
    assert st.progress == 0.45
    assert st.status == JobStatusEnum.PROCESSING


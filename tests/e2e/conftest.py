from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.generator import FixtureGenerator
from tests.fixtures.mock_engine import (
    LineBox,
    LineCrop,
    MockInferenceEngine,
    PageResult,
    PreprocessedPage,
    RecognitionResponse,
    WordBox,
    compute_cer,
    compute_levenshtein_distance,
    compute_wer,
    create_mock_app,
)


# ---------------------------------------------------------------------------
# Marker Registration & Session Start Auto-Generation
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for the 4-tier E2E testing framework."""
    config.addinivalue_line("markers", "tier1: Tier 1 Feature isolation tests covering F1-F20")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary, corruption, and edge-case tests")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-feature pairwise integration tests")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-world end-to-end workload scenarios")
    config.addinivalue_line("markers", "tier5: Tier 5 Adversarial hardening and edge-case stress tests")
    config.addinivalue_line("markers", "gpu: Tests requiring Apple Silicon MPS hardware acceleration")
    config.addinivalue_line("markers", "slow: Long-running benchmark or training tests")
    config.addinivalue_line("markers", "mock_only: Tests verified strictly with MockInferenceEngine")


def pytest_sessionstart(session: pytest.Session) -> None:
    """Auto-generate test fixtures if manifest or assets are missing on session start."""
    fixture_dir = Path("tests/fixtures").resolve()
    manifest_file = fixture_dir / "manifest.json"
    key_asset = fixture_dir / "sample_clean_handwriting.png"

    if not manifest_file.exists() or not key_asset.exists():
        generator = FixtureGenerator(out_dir=fixture_dir, seed=42)
        generator.generate_all()


# ---------------------------------------------------------------------------
# Custom Assertion Helpers
# ---------------------------------------------------------------------------

def assert_valid_bbox(
    bbox: List[float] | Tuple[float, ...],
    name: str = "Bounding box",
) -> None:
    """
    Validate that bbox is [ymin, xmin, ymax, xmax] with coordinates in [0.0, 1.0],
    satisfying ymin < ymax and xmin < xmax.
    """
    assert isinstance(bbox, (list, tuple)), f"{name} must be a list/tuple, got {type(bbox)}"
    assert len(bbox) == 4, f"{name} must contain exactly 4 coordinates, got {len(bbox)}: {bbox}"

    ymin, xmin, ymax, xmax = bbox
    for val, coord_name in zip(bbox, ["ymin", "xmin", "ymax", "xmax"]):
        assert 0.0 <= val <= 1.0, f"{name} {coord_name}={val} is outside [0.0, 1.0]"

    assert ymin < ymax, f"{name} invalid vertical span: ymin ({ymin}) >= ymax ({ymax})"
    assert xmin < xmax, f"{name} invalid horizontal span: xmin ({xmin}) >= xmax ({xmax})"


def assert_bbox_iou(
    box1: List[float] | Tuple[float, ...],
    box2: List[float] | Tuple[float, ...],
    min_iou: float = 0.5,
) -> float:
    """Compute 2D Intersection-over-Union (IoU) and assert it meets min_iou threshold."""
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    inter_ymin = max(y1_min, y2_min)
    inter_xmin = max(x1_min, x2_min)
    inter_ymax = min(y1_max, y2_max)
    inter_xmax = min(x1_max, x2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (y1_max - y1_min) * (x1_max - x1_min)
    area2 = (y2_max - y2_min) * (x2_max - x2_min)
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    assert iou >= min_iou, f"Bounding box IoU {iou:.4f} is below minimum threshold {min_iou:.4f}"
    return iou


def assert_cer_below(
    hypothesis: str,
    reference: str,
    max_cer: float = 0.15,
) -> float:
    """Compute Character Error Rate (CER) and assert it is below max_cer."""
    cer = compute_cer(hypothesis, reference)
    assert cer <= max_cer, f"Observed CER {cer:.4f} exceeds maximum threshold {max_cer:.4f}.\nHypothesis: {hypothesis!r}\nReference: {reference!r}"
    return cer


def assert_wer_below(
    hypothesis: str,
    reference: str,
    max_wer: float = 0.25,
) -> float:
    """Compute Word Error Rate (WER) and assert it is below max_wer."""
    wer = compute_wer(hypothesis, reference)
    assert wer <= max_wer, f"Observed WER {wer:.4f} exceeds maximum threshold {max_wer:.4f}.\nHypothesis: {hypothesis!r}\nReference: {reference!r}"
    return wer


def assert_valid_recognition_response(
    response: Dict[str, Any] | RecognitionResponse,
) -> None:
    """Validate that recognition response adheres strictly to PROJECT.md interface contract."""
    data = response.model_dump() if isinstance(response, RecognitionResponse) else response

    assert "document_id" in data and isinstance(data["document_id"], str) and data["document_id"], "Missing or empty document_id"
    assert "filename" in data and isinstance(data["filename"], str), "Missing filename"
    assert "total_pages" in data and isinstance(data["total_pages"], int) and data["total_pages"] >= 1, "total_pages must be >= 1"
    assert "pages" in data and isinstance(data["pages"], list) and len(data["pages"]) >= 1, "pages list must contain at least 1 page"
    assert "processing_time_ms" in data and data["processing_time_ms"] >= 0.0, "processing_time_ms must be >= 0.0"

    for page in data["pages"]:
        assert page["page_number"] >= 1, f"Invalid page_number {page['page_number']}"
        assert page["width"] > 0 and page["height"] > 0, f"Invalid page dimensions: {page['width']}x{page['height']}"
        assert isinstance(page["full_text"], str), "full_text must be a string"
        assert 0.0 <= page["mean_confidence"] <= 1.0, f"mean_confidence {page['mean_confidence']} outside [0.0, 1.0]"

        for line in page.get("lines", []):
            assert "line_id" in line and isinstance(line["line_id"], str), "Missing line_id"
            assert "text" in line and isinstance(line["text"], str), "Missing line text"
            assert 0.0 <= line["confidence"] <= 1.0, f"line confidence {line['confidence']} outside [0.0, 1.0]"
            assert_valid_bbox(line["bbox"], f"Line {line['line_id']} bbox")

            for word in line.get("words", []):
                assert "word_id" in word and isinstance(word["word_id"], str), "Missing word_id"
                assert "text" in word and isinstance(word["text"], str), "Missing word text"
                assert 0.0 <= word["confidence"] <= 1.0, f"word confidence {word['confidence']} outside [0.0, 1.0]"
                assert_valid_bbox(word["bbox"], f"Word {word['word_id']} bbox")


class AssertionHelpers:
    """Namespace container for custom assertions and metrics."""
    assert_valid_bbox = staticmethod(assert_valid_bbox)
    assert_bbox_iou = staticmethod(assert_bbox_iou)
    assert_cer_below = staticmethod(assert_cer_below)
    assert_wer_below = staticmethod(assert_wer_below)
    assert_valid_recognition_response = staticmethod(assert_valid_recognition_response)
    compute_cer = staticmethod(compute_cer)
    compute_wer = staticmethod(compute_wer)
    compute_levenshtein_distance = staticmethod(compute_levenshtein_distance)


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Return absolute path to test fixtures directory."""
    path = Path("tests/fixtures").resolve()
    if not (path / "manifest.json").exists():
        FixtureGenerator(out_dir=path).generate_all()
    return path


@pytest.fixture(scope="session")
def corrupted_dir(fixture_dir: Path) -> Path:
    """Return path to corrupted fixtures directory."""
    return fixture_dir / "corrupted"


@pytest.fixture(scope="session")
def manifest(fixture_dir: Path) -> Dict[str, Any]:
    """Load and return cached manifest.json dictionary."""
    manifest_file = fixture_dir / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_engine(manifest: Dict[str, Any]) -> MockInferenceEngine:
    """Provide fresh MockInferenceEngine instance."""
    engine = MockInferenceEngine()
    engine.manifest = manifest
    return engine


@pytest.fixture
def api_client(mock_engine: MockInferenceEngine) -> TestClient:
    """Provide FastAPI TestClient wired to MockInferenceEngine."""
    app = create_mock_app(mock_engine)
    return TestClient(app)


@pytest.fixture
def tmp_workspace() -> Generator[Path, None, None]:
    """Provide an isolated temporary workspace directory with automatic cleanup."""
    temp_dir = Path(tempfile.mkdtemp(prefix="e2e_hw_test_"))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def clean_image_path(fixture_dir: Path) -> Path:
    return fixture_dir / "sample_clean_handwriting.png"


@pytest.fixture(scope="session")
def messy_image_path(fixture_dir: Path) -> Path:
    return fixture_dir / "sample_messy_prescription.png"


@pytest.fixture(scope="session")
def skewed_image_path(fixture_dir: Path) -> Path:
    return fixture_dir / "sample_skewed_document.png"


@pytest.fixture(scope="session")
def low_contrast_image_path(fixture_dir: Path) -> Path:
    return fixture_dir / "sample_low_contrast_faded.png"


@pytest.fixture(scope="session")
def multipage_pdf_path(fixture_dir: Path) -> Path:
    return fixture_dir / "sample_multipage_consultation.pdf"


@pytest.fixture(scope="session")
def corrupted_fixtures(corrupted_dir: Path) -> Dict[str, Path]:
    """Map of all corrupted fixture paths."""
    return {
        "zero_byte": corrupted_dir / "sample_zero_byte.png",
        "truncated_header": corrupted_dir / "sample_truncated_header.jpg",
        "corrupted_pdf": corrupted_dir / "sample_corrupted_pdf.pdf",
        "extreme_1x1": corrupted_dir / "sample_extreme_1x1.png",
        "extreme_aspect": corrupted_dir / "sample_extreme_aspect.png",
        "disguised_text": corrupted_dir / "sample_disguised_text.png",
        "cmyk_image": corrupted_dir / "sample_cmyk.jpg",
    }


@pytest.fixture(scope="session")
def helpers() -> AssertionHelpers:
    """Fixture providing access to all custom assertion helpers."""
    return AssertionHelpers()

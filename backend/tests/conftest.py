"""
backend/tests/conftest.py
Shared pytest fixtures for backend inference service tests.
"""

from __future__ import annotations
import io
import os
from pathlib import Path
import sys
from typing import Generator
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.config import get_settings, reset_settings_cache
from backend.app.engine import InferenceEngine, get_engine, reset_engine, set_engine
from backend.app.main import create_app


os.environ["USE_MOCK_ENGINE"] = "true"


@pytest.fixture(autouse=True)
def reset_backend_state() -> Generator[None, None, None]:
    """Reset settings and engine caches before each test."""
    os.environ["USE_MOCK_ENGINE"] = "true"
    reset_settings_cache()
    reset_engine()
    yield
    reset_settings_cache()
    reset_engine()



@pytest.fixture
def client() -> TestClient:
    """Provide FastAPI TestClient wired to real application instance."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a clean synthetic handwriting image (PNG bytes)."""
    img = Image.new("RGB", (800, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Amoxicillin 500mg capsules", fill=(0, 0, 0))
    draw.text((50, 150), "Take 1 capsule every 8 hours", fill=(0, 0, 0))
    draw.text((50, 250), "Dr. Sarah Smith, MD", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generate a synthetic JPEG image."""
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 40), "Ibuprofen 400mg PO TID", fill=(20, 20, 20))
    draw.text((40, 140), "Dispense #30 refills 0", fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_tiff_bytes() -> bytes:
    """Generate a multi-frame TIFF image."""
    f1 = Image.new("RGB", (500, 300), color=(255, 255, 255))
    d1 = ImageDraw.Draw(f1)
    d1.text((30, 30), "Page 1: Prescription Note", fill=(0, 0, 0))

    f2 = Image.new("RGB", (500, 300), color=(255, 255, 255))
    d2 = ImageDraw.Draw(f2)
    d2.text((30, 30), "Page 2: Dosage Schedule", fill=(0, 0, 0))

    buf = io.BytesIO()
    f1.save(buf, format="TIFF", save_all=True, append_images=[f2])
    return buf.getvalue()


@pytest.fixture
def sample_bmp_bytes() -> bytes:
    """Generate a BMP image."""
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "Metformin 500mg", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


@pytest.fixture
def sample_webp_bytes() -> bytes:
    """Generate a WebP image."""
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "Lisinopril 10mg daily", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a minimal valid 2-page PDF document."""
    # If pypdfium2 / reportlab or PIL can generate PDF:
    try:
        import pypdfium2 as pdfium
        # Use existing fixture if present
        pdf_path = Path("tests/fixtures/sample_multipage_consultation.pdf")
        if pdf_path.exists():
            return pdf_path.read_bytes()
    except Exception:
        pass

    # Generate multi-page PDF via PIL
    f1 = Image.new("RGB", (600, 800), color=(255, 255, 255))
    d1 = ImageDraw.Draw(f1)
    d1.text((50, 100), "Consultation Note Page 1", fill=(0, 0, 0))
    d1.text((50, 200), "Patient presents with persistent cough", fill=(0, 0, 0))

    f2 = Image.new("RGB", (600, 800), color=(255, 255, 255))
    d2 = ImageDraw.Draw(f2)
    d2.text((50, 100), "Consultation Note Page 2", fill=(0, 0, 0))
    d2.text((50, 200), "Prescribed Albuterol inhaler PRN", fill=(0, 0, 0))

    buf = io.BytesIO()
    f1.save(buf, format="PDF", save_all=True, append_images=[f2])
    return buf.getvalue()


@pytest.fixture
def corrupted_image_bytes() -> bytes:
    """Return truncated/corrupt image bytes."""
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\xff" * 16


@pytest.fixture
def corrupted_pdf_bytes() -> bytes:
    """Return truncated/corrupt PDF bytes."""
    return b"%PDF-1.7\nCorrupted structure xref table broken \x00\xff"

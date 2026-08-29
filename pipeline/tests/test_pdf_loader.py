"""
Unit and integration tests for pipeline/preprocessing/pdf_loader.py.
"""

import io
import os
import tempfile
import numpy as np
from PIL import Image
import pytest

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


def test_detect_format():
    """Test magic bytes format detection."""
    assert detect_format(b"%PDF-1.7") == "pdf"
    assert detect_format(b"\x89PNG\r\n\x1a\n\x00\x00") == "png"
    assert detect_format(b"\xff\xd8\xff\xe0\x00\x10JFIF") == "jpeg"
    assert detect_format(b"II*\x00\x08\x00\x00\x00") == "tiff"
    assert detect_format(b"MM\x00*\x00\x00\x00\x08") == "tiff"
    assert detect_format(b"BM\x36\x00\x00\x00") == "bmp"
    assert detect_format(b"RIFF\x00\x00\x00\x00WEBP") == "webp"
    assert detect_format(b"random text data") == "unknown"

    with pytest.raises(EmptyDocumentError):
        detect_format(b"")


def test_load_image_numpy_inputs():
    """Test loading from 2D, 3D, and 4D NumPy arrays."""
    # 2D grayscale
    gray = np.full((100, 150), 128, dtype=np.uint8)
    pages = load_image(gray)
    assert len(pages) == 1
    assert pages[0].shape == (100, 150, 3)
    assert pages[0].dtype == np.uint8
    assert np.all(pages[0] == 128)

    # 3D RGB
    rgb = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
    pages = load_image(rgb)
    assert len(pages) == 1
    assert pages[0].shape == (80, 120, 3)
    assert np.array_equal(pages[0], rgb)

    # 4D RGBA transparent (composite over white)
    rgba = np.zeros((50, 50, 4), dtype=np.uint8)  # fully transparent
    pages = load_image(rgba)
    assert len(pages) == 1
    assert pages[0].shape == (50, 50, 3)
    assert np.all(pages[0] == 255)  # white canvas

    with pytest.raises(EmptyDocumentError):
        load_image(np.zeros((0, 0, 3), dtype=np.uint8))


def test_load_image_pil_input():
    """Test loading directly from a PIL Image instance."""
    pil_img = Image.new("RGB", (200, 100), color=(10, 20, 30))
    pages = load_image(pil_img)
    assert len(pages) == 1
    assert pages[0].shape == (100, 200, 3)
    assert pages[0][0, 0, 0] == 10
    assert pages[0][0, 0, 1] == 20
    assert pages[0][0, 0, 2] == 30


def test_load_image_bytes_and_file():
    """Test loading PNG and JPEG from bytes and file paths."""
    pil_img = Image.new("RGB", (160, 90), color=(200, 100, 50))

    # In-memory PNG bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    pages = load_image(png_bytes)
    assert len(pages) == 1
    assert pages[0].shape == (90, 160, 3)

    # Temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pil_img.save(tmp_path, format="JPEG")
        pages_file = load_image(tmp_path)
        assert len(pages_file) == 1
        assert pages_file[0].shape == (90, 160, 3)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_load_multipage_tiff():
    """Test multi-frame TIFF file loading."""
    f1 = Image.new("RGB", (100, 80), color=(255, 0, 0))
    f2 = Image.new("RGB", (100, 80), color=(0, 255, 0))
    f3 = Image.new("RGB", (100, 80), color=(0, 0, 255))

    buf = io.BytesIO()
    f1.save(buf, format="TIFF", save_all=True, append_images=[f2, f3])
    tiff_bytes = buf.getvalue()

    pages = load_image(tiff_bytes)
    assert len(pages) == 3
    assert pages[0][0, 0, 0] == 255  # Red
    assert pages[1][0, 0, 1] == 255  # Green
    assert pages[2][0, 0, 2] == 255  # Blue


def test_load_pdf_and_dpi_scaling(temp_pdf_multipage: str):
    """Test PDF loading and scale multiplier."""
    # Default 300 DPI
    pages = load_pdf(temp_pdf_multipage, dpi=150)
    assert len(pages) == 3
    assert all(p.ndim == 3 and p.shape[2] == 3 for p in pages)
    assert all(p.dtype == np.uint8 for p in pages)

    # Scale 1.0 vs 2.0
    pages_s1 = load_pdf(temp_pdf_multipage, scale=1.0)
    pages_s2 = load_pdf(temp_pdf_multipage, scale=2.0)
    assert pages_s2[0].shape[0] == pytest.approx(pages_s1[0].shape[0] * 2, abs=2)
    assert pages_s2[0].shape[1] == pytest.approx(pages_s1[0].shape[1] * 2, abs=2)


def test_load_document_auto_dispatch(temp_pdf_multipage: str):
    """Test universal load_document dispatcher."""
    # PDF path
    pages_pdf = load_document(temp_pdf_multipage, scale=1.0)
    assert len(pages_pdf) == 3

    # Image array
    img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    pages_img = load_document(img_arr)
    assert len(pages_img) == 1
    assert pages_img[0].shape == (100, 100, 3)


def test_pdf_loader_class_methods(temp_pdf_multipage: str):
    """Test PDFLoader class wrapper."""
    loader = PDFLoader(default_dpi=150)
    pages = loader.load_pages(temp_pdf_multipage)
    assert len(pages) == 3

    single = loader.load_single_image(temp_pdf_multipage)
    assert single.shape == pages[0].shape
    assert np.array_equal(single, pages[0])


def test_error_handling():
    """Test robust error handling and custom exceptions."""
    # Missing file
    with pytest.raises(FileNotFoundError):
        load_pdf("non_existent_file_xyz123.pdf")

    with pytest.raises(FileNotFoundError):
        load_image("non_existent_image_xyz123.png")

    # Empty file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        empty_path = tmp.name
    try:
        with pytest.raises(EmptyDocumentError):
            load_pdf(empty_path)
        with pytest.raises(EmptyDocumentError):
            load_image(empty_path)
    finally:
        if os.path.exists(empty_path):
            os.remove(empty_path)

    # Empty bytes
    with pytest.raises(EmptyDocumentError):
        load_pdf(b"")
    with pytest.raises(EmptyDocumentError):
        load_image(b"")

    # Corrupt data
    with pytest.raises(CorruptDocumentError):
        load_pdf(b"%PDF-corrupted-and-truncated-garbage")
    with pytest.raises(CorruptDocumentError):
        load_image(b"\x89PNG\r\n\x1a\ncorrupted-png-body")

    # Invalid source types
    with pytest.raises(UnsupportedFormatError):
        load_pdf(12345)  # type: ignore
    with pytest.raises(UnsupportedFormatError):
        load_image(object())  # type: ignore

    # Invalid scale / DPI
    with pytest.raises(ValueError):
        load_pdf(b"%PDF-1.4", dpi=-10)

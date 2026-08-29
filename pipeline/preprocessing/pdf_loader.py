"""
pipeline/preprocessing/pdf_loader.py
Pure-Python multi-format document and PDF rasterizer using pypdfium2 and Pillow.
Supports single and multi-page PDFs, arbitrary DPI/scale, multi-frame TIFFs,
safe alpha compositing over white canvas, and auto-format detection via magic bytes.
"""

import io
from pathlib import Path
from typing import BinaryIO, List, Optional, Union
import numpy as np
from PIL import Image, ImageSequence
import pypdfium2 as pdfium


# ---------------------------------------------------------------------------
# Custom Exception Hierarchy
# ---------------------------------------------------------------------------

class DocumentLoadingError(Exception):
    """Base exception for all document loading and rasterization failures."""
    pass


class CorruptDocumentError(DocumentLoadingError):
    """Raised when the document file is corrupted, truncated, or invalid."""
    pass


class PasswordProtectedPDFError(DocumentLoadingError):
    """Raised when a PDF is password protected or encrypted and cannot be opened."""
    pass


class EmptyDocumentError(DocumentLoadingError):
    """Raised when the document contains 0 pages, 0 frames, or 0 bytes."""
    pass


class UnsupportedFormatError(DocumentLoadingError):
    """Raised when the file format or source type is unrecognized or unsupported."""
    pass


# ---------------------------------------------------------------------------
# Magic Byte Auto-Detection
# ---------------------------------------------------------------------------

PDF_MAGIC = b"%PDF-"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
TIFF_LE_MAGIC = b"II*\x00"
TIFF_BE_MAGIC = b"MM\x00*"
BMP_MAGIC = b"BM"
WEBP_MAGIC = b"RIFF"


def detect_format(data: bytes) -> str:
    """
    Inspect leading magic bytes to determine document/image format.

    Args:
        data: Initial byte sequence (at least 12 bytes recommended).

    Returns:
        One of 'pdf', 'png', 'jpeg', 'tiff', 'bmp', 'webp', or 'unknown'.

    Raises:
        EmptyDocumentError: If data is empty.
    """
    if len(data) == 0:
        raise EmptyDocumentError("Input byte stream is empty (0 bytes).")
    if data.startswith(PDF_MAGIC):
        return "pdf"
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    if data.startswith(TIFF_LE_MAGIC) or data.startswith(TIFF_BE_MAGIC):
        return "tiff"
    if data.startswith(BMP_MAGIC):
        return "bmp"
    if data.startswith(WEBP_MAGIC) and len(data) >= 12 and data[8:12] == b"WEBP":
        return "webp"
    return "unknown"


# ---------------------------------------------------------------------------
# PIL to RGB Array Normalization with Alpha Compositing
# ---------------------------------------------------------------------------

def _normalize_pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    """
    Safely convert any PIL Image mode to a 3-channel RGB uint8 numpy array,
    compositing any alpha transparency over a solid clean white background.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba_img = img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(white_bg, rgba_img).convert("RGB")
        return np.array(composited, dtype=np.uint8)
    elif img.mode == "RGB":
        return np.array(img, dtype=np.uint8)
    elif img.mode in ("L", "1"):
        rgb = img.convert("RGB")
        return np.array(rgb, dtype=np.uint8)
    elif img.mode == "CMYK":
        rgb = img.convert("RGB")
        return np.array(rgb, dtype=np.uint8)
    else:
        rgb = img.convert("RGB")
        return np.array(rgb, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Core PDF and Image Loaders
# ---------------------------------------------------------------------------

def load_pdf(
    source: Union[str, Path, bytes, BinaryIO],
    dpi: int = 300,
    scale: Optional[float] = None,
    password: Optional[str] = None
) -> List[np.ndarray]:
    """
    Rasterize a PDF document into a list of RGB uint8 numpy arrays (one per page).

    Args:
        source: File path, raw bytes, or readable binary stream.
        dpi: Target rasterization resolution in dots per inch (default: 300).
        scale: Direct scale factor multiplier (overrides dpi if provided).
        password: Optional password string for encrypted PDFs.

    Returns:
        List of uint8 numpy arrays with shape (H, W, 3) in RGB channel order.

    Raises:
        FileNotFoundError: If a file path is provided that does not exist.
        EmptyDocumentError: If the source contains 0 bytes or 0 pages.
        PasswordProtectedPDFError: If the PDF is encrypted and cannot be opened.
        CorruptDocumentError: If the PDF data is malformed or corrupted.
    """
    effective_scale = scale if scale is not None else (float(dpi) / 72.0)
    if effective_scale <= 0:
        raise ValueError(f"Invalid DPI/scale: dpi={dpi}, scale={scale}")

    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {path}")
            if path.stat().st_size == 0:
                raise EmptyDocumentError(f"PDF file is empty (0 bytes): {path}")
            doc = pdfium.PdfDocument(str(path), password=password)
        elif isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise EmptyDocumentError("PDF byte stream is empty (0 bytes).")
            doc = pdfium.PdfDocument(bytes(source), password=password)
        elif hasattr(source, "read"):
            data = source.read()
            if len(data) == 0:
                raise EmptyDocumentError("PDF stream is empty (0 bytes).")
            doc = pdfium.PdfDocument(data, password=password)
        else:
            raise UnsupportedFormatError(f"Unsupported source type for PDF loader: {type(source)}")
    except pdfium.PdfiumError as e:
        err_msg = str(e).lower()
        if "password" in err_msg or "encrypted" in err_msg or "auth" in err_msg:
            raise PasswordProtectedPDFError(f"PDF is password protected: {e}") from e
        raise CorruptDocumentError(f"Failed to parse PDF document: {e}") from e
    except (FileNotFoundError, EmptyDocumentError, PasswordProtectedPDFError, UnsupportedFormatError, CorruptDocumentError):
        raise
    except Exception as e:
        raise CorruptDocumentError(f"Unexpected error opening PDF: {e}") from e

    pages: List[np.ndarray] = []
    try:
        num_pages = len(doc)
        if num_pages == 0:
            raise EmptyDocumentError("PDF contains 0 pages.")
        for page_idx in range(num_pages):
            page = doc[page_idx]
            bitmap = page.render(scale=effective_scale)
            pil_img = bitmap.to_pil()
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img, dtype=np.uint8)
            pages.append(arr)
    except Exception as e:
        if isinstance(e, DocumentLoadingError):
            raise
        raise CorruptDocumentError(f"Error rendering PDF page: {e}") from e
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return pages


def load_image(
    source: Union[str, Path, bytes, BinaryIO, Image.Image, np.ndarray]
) -> List[np.ndarray]:
    """
    Load an image or multi-frame image (e.g. multi-page TIFF) into a list of RGB numpy arrays.

    Supports: PNG, JPEG, TIFF (multi-page), BMP, WebP, GIF, PIL Image, and numpy arrays.

    Args:
        source: Image path, bytes, file buffer, PIL Image instance, or numpy array.

    Returns:
        List of uint8 numpy arrays with shape (H, W, 3) in RGB channel order.

    Raises:
        FileNotFoundError: If a file path is provided that does not exist.
        EmptyDocumentError: If the source contains 0 bytes or 0 decodable frames.
        CorruptDocumentError: If the image is truncated or invalid.
    """
    # 1. Direct NumPy Array Input
    if isinstance(source, np.ndarray):
        if source.size == 0:
            raise EmptyDocumentError("Input numpy array has 0 elements.")
        if source.ndim == 2:  # Grayscale (H, W)
            return [np.stack([source] * 3, axis=-1).astype(np.uint8)]
        elif source.ndim == 3:
            if source.shape[2] == 1:
                return [np.repeat(source, 3, axis=-1).astype(np.uint8)]
            elif source.shape[2] == 3:
                return [source.astype(np.uint8)]
            elif source.shape[2] == 4:
                # RGBA array alpha composite over white background
                alpha = source[:, :, 3:4].astype(np.float32) / 255.0
                rgb = source[:, :, :3].astype(np.float32)
                white = np.ones_like(rgb) * 255.0
                out = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
                return [out]
        raise ValueError(f"Invalid numpy array image shape: {source.shape}")

    # 2. Direct PIL Image Input
    if isinstance(source, Image.Image):
        frames: List[np.ndarray] = []
        for frame in ImageSequence.Iterator(source):
            frames.append(_normalize_pil_to_rgb_array(frame))
        if len(frames) == 0:
            raise EmptyDocumentError("PIL Image contained no decodable frames.")
        return frames

    # 3. Path, Bytes, or File-like Object
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            if path.stat().st_size == 0:
                raise EmptyDocumentError(f"Image file is empty (0 bytes): {path}")
            pil_img = Image.open(str(path))
        elif isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise EmptyDocumentError("Image byte stream is empty (0 bytes).")
            pil_img = Image.open(io.BytesIO(source))
        elif hasattr(source, "read"):
            data = source.read()
            if len(data) == 0:
                raise EmptyDocumentError("Image stream is empty (0 bytes).")
            pil_img = Image.open(io.BytesIO(data))
        else:
            raise UnsupportedFormatError(f"Unsupported source type for image loader: {type(source)}")
    except (FileNotFoundError, EmptyDocumentError, UnsupportedFormatError):
        raise
    except Exception as e:
        raise CorruptDocumentError(f"Failed to decode image: {e}") from e

    frames = []
    try:
        for frame in ImageSequence.Iterator(pil_img):
            frames.append(_normalize_pil_to_rgb_array(frame))
    except Exception as e:
        raise CorruptDocumentError(f"Error reading image frames: {e}") from e
    finally:
        try:
            pil_img.close()
        except Exception:
            pass

    if len(frames) == 0:
        raise EmptyDocumentError("Image file contained no decodable frames.")
    return frames


def load_document(
    source: Union[str, Path, bytes, BinaryIO, Image.Image, np.ndarray],
    dpi: int = 300,
    scale: Optional[float] = None,
    password: Optional[str] = None
) -> List[np.ndarray]:
    """
    Universal document and image loader with auto-detection for PDFs and raster images.

    Args:
        source: File path, bytes, file stream, PIL Image, or numpy array.
        dpi: DPI for PDF rasterization (default: 300).
        scale: Scale factor for PDF rasterization (overrides dpi).
        password: Password for encrypted PDFs.

    Returns:
        List of RGB uint8 numpy arrays (one per page/frame).
    """
    if isinstance(source, (np.ndarray, Image.Image)):
        return load_image(source)

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.stat().st_size == 0:
            raise EmptyDocumentError(f"File is empty (0 bytes): {p}")
        with open(p, "rb") as f:
            header = f.read(16)
        if header.startswith(PDF_MAGIC) or p.suffix.lower() == ".pdf":
            return load_pdf(p, dpi=dpi, scale=scale, password=password)
        else:
            return load_image(p)

    elif isinstance(source, (bytes, bytearray)):
        if len(source) == 0:
            raise EmptyDocumentError("Input bytes are empty (0 bytes).")
        if source.startswith(PDF_MAGIC):
            return load_pdf(source, dpi=dpi, scale=scale, password=password)
        else:
            return load_image(source)

    elif hasattr(source, "read"):
        data = source.read()
        if len(data) == 0:
            raise EmptyDocumentError("Input stream is empty (0 bytes).")
        if data.startswith(PDF_MAGIC):
            return load_pdf(data, dpi=dpi, scale=scale, password=password)
        else:
            return load_image(data)

    raise UnsupportedFormatError(f"Cannot load document from source of type {type(source)}")


# ---------------------------------------------------------------------------
# High-Level PDFLoader Class Wrapper
# ---------------------------------------------------------------------------

class PDFLoader:
    """
    Object-oriented document loader supporting multi-format image ingestion
    and high-resolution PDF rendering via pypdfium2.
    """

    def __init__(self, default_dpi: int = 300):
        self.default_dpi = default_dpi

    def load_pages(
        self,
        source: Union[str, Path, bytes, BinaryIO, Image.Image, np.ndarray],
        dpi: Optional[int] = None,
        scale: Optional[float] = None,
        password: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Load all pages/frames from a document or image into a list of RGB numpy arrays.
        """
        effective_dpi = dpi if dpi is not None else self.default_dpi
        return load_document(source, dpi=effective_dpi, scale=scale, password=password)

    def load_single_image(
        self,
        source: Union[str, Path, bytes, BinaryIO, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Load the first page or image frame as a single RGB numpy array (H, W, 3).
        """
        pages = self.load_pages(source)
        if not pages:
            raise EmptyDocumentError("Document produced 0 pages.")
        return pages[0]

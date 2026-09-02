"""
backend/tests/test_recognize.py
Integration tests for POST /v1/recognize endpoint with multipart and JSON payloads.
"""

import base64
import io
from types import MethodType

from fastapi.testclient import TestClient
from PIL import Image
import torch

from backend.app.engine import InferenceEngine
from backend.app.routes.recognize import _decode_line_crop
from backend.app.schemas import RecognitionResponse


def test_recognize_multipart_png(client: TestClient, sample_image_bytes: bytes) -> None:
    """Verify POST /v1/recognize with multipart PNG upload returns valid response."""
    resp = client.post(
        "/v1/recognize",
        files={"file": ("prescription.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    model = RecognitionResponse.model_validate(data)

    assert model.total_pages == 1
    assert len(model.pages) == 1
    assert model.pages[0].page_number == 1
    assert len(model.pages[0].lines) >= 1
    assert model.processing_time_ms >= 0.0


def test_recognize_multipart_multipage_pdf(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Verify POST /v1/recognize with multi-page PDF returns all pages."""
    resp = client.post(
        "/v1/recognize",
        files={"file": ("consultation.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    model = RecognitionResponse.model_validate(data)

    assert model.total_pages >= 2
    assert len(model.pages) == model.total_pages
    for i, p in enumerate(model.pages):
        assert p.page_number == i + 1
        assert len(p.lines) >= 1


def test_recognize_json_base64(client: TestClient, sample_image_bytes: bytes) -> None:
    """Verify POST /v1/recognize with application/json base64 payload."""
    b64_str = base64.b64encode(sample_image_bytes).decode("utf-8")
    payload = {
        "file_base64": b64_str,
        "filename": "json_upload.png",
        "options": {
            "deskew": True,
            "enhance_contrast": True,
            "binarization_method": "sauvola",
            "extract_words": True,
            "dpi": 300,
        },
    }
    resp = client.post(
        "/v1/recognize",
        json=payload,
    )
    assert resp.status_code == 200
    data = resp.json()
    model = RecognitionResponse.model_validate(data)
    assert model.filename == "json_upload.png"
    assert len(model.pages) == 1


def test_recognize_with_query_options(client: TestClient, sample_image_bytes: bytes) -> None:
    """Verify POST /v1/recognize passing preprocessing options as query parameters."""
    resp = client.post(
        "/v1/recognize?deskew=false&binarization_method=otsu&dpi=150",
        files={"file": ("custom.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["preprocessing_flags"]["deskew"] is False
    assert data["preprocessing_flags"]["binarization_method"] == "otsu"
    assert data["preprocessing_flags"]["dpi"] == 150


def test_recognize_all_formats(
    client: TestClient,
    sample_jpeg_bytes: bytes,
    sample_tiff_bytes: bytes,
    sample_bmp_bytes: bytes,
    sample_webp_bytes: bytes,
) -> None:
    """Verify POST /v1/recognize supports JPEG, TIFF, BMP, and WebP."""
    # JPEG
    r1 = client.post("/v1/recognize", files={"file": ("doc.jpg", sample_jpeg_bytes, "image/jpeg")})
    assert r1.status_code == 200

    # TIFF
    r2 = client.post("/v1/recognize", files={"file": ("doc.tiff", sample_tiff_bytes, "image/tiff")})
    assert r2.status_code == 200

    # BMP
    r3 = client.post("/v1/recognize", files={"file": ("doc.bmp", sample_bmp_bytes, "image/bmp")})
    assert r3.status_code == 200

    # WebP
    r4 = client.post("/v1/recognize", files={"file": ("doc.webp", sample_webp_bytes, "image/webp")})
    assert r4.status_code == 200


def test_recognize_line_returns_text_ms_model_id(client: TestClient, sample_image_bytes: bytes) -> None:
    resp = client.post(
        "/v1/recognize-line",
        files={"file": ("line.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "text" in body
    assert "ms" in body
    assert body["model_id"] == "microsoft/trocr-large-handwritten"


def test_decode_line_crop_uses_page_generate() -> None:
    seen: dict[str, object] = {}

    class FakeProc:
        def __call__(self, images, return_tensors="pt", padding=False):
            class Out:
                pixel_values = torch.zeros(1, 3, 16, 16)

            return Out()

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["hello line"]

    class FakeModel:
        def generate(self, *args, **kwargs):
            seen.update(kwargs)

            class Out:
                sequences = torch.tensor([[0, 1, 2]])

            return Out()

    engine = InferenceEngine.__new__(InferenceEngine)
    engine.processor = FakeProc()
    engine.model = FakeModel()
    engine.device = torch.device("cpu")
    engine.beam_width = 4
    engine.use_fp16 = False
    engine.recognize_single_crop = MethodType(InferenceEngine.recognize_single_crop, engine)

    text = _decode_line_crop(engine, Image.new("RGB", (64, 24), "white"))
    assert seen.get("num_beams") == 4
    assert seen.get("max_new_tokens") == 128
    assert seen.get("num_return_sequences") == 4
    assert text == "hello line"


def test_recognize_line_rejects_tiny_crop(client: TestClient) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buf, format="PNG")
    resp = client.post(
        "/v1/recognize-line",
        files={"file": ("tiny.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 400

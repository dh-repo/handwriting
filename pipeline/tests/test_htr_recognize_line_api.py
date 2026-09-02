import io

from fastapi.testclient import TestClient
from PIL import Image
import torch

from pipeline.training.serve import create_app


def test_recognize_line_rejects_tiny_and_huge() -> None:
    app = create_app(model=None, processor=None, model_id="test-model", preload=False)
    client = TestClient(app)
    tiny = Image.new("RGB", (8, 8), color="white")
    buf = io.BytesIO()
    tiny.save(buf, format="PNG")
    resp = client.post("/v1/recognize-line", files={"file": ("tiny.png", buf.getvalue(), "image/png")})
    assert resp.status_code == 400

    huge = b"x" * (10 * 1024 * 1024 + 8)
    resp = client.post("/v1/recognize-line", files={"file": ("big.png", huge, "image/png")})
    assert resp.status_code == 400


def test_recognize_line_returns_text_ms_model_id() -> None:
    class FakeProc:
        def __call__(self, images, return_tensors="pt"):
            class Out:
                pixel_values = torch.zeros(1, 3, 16, 16)

            return Out()

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["hello line"]

    class FakeModel:
        def generate(self, *args, **kwargs):
            return torch.tensor([[0, 1, 2]])

    app = create_app(model=FakeModel(), processor=FakeProc(), model_id="fake-base", preload=False)
    client = TestClient(app)
    img = Image.new("RGB", (64, 24), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = client.post("/v1/recognize-line", files={"file": ("line.png", buf.getvalue(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "hello line"
    assert body["model_id"] == "fake-base"
    assert "ms" in body


def test_no_page_endpoint() -> None:
    app = create_app(model=None, processor=None, model_id="test", preload=False)
    client = TestClient(app)
    assert client.post("/v1/recognize-page").status_code == 404

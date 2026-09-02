from PIL import Image
import torch

from pipeline.training.dataset import create_dummy_processor
from pipeline.training.infer import recognize_line, recognize_lines
from pipeline.training.model_contract import apply_trocr_generation_config
from pipeline.training.train import create_tiny_mock_model


def test_recognize_line_always_sends_max_new_tokens() -> None:
    seen: dict[str, object] = {}
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)
    apply_trocr_generation_config(model, processor, max_length=16, num_beams=1)
    original = model.generate

    def _generate(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    model.generate = _generate
    model.eval()
    recognize_line(
        model,
        processor,
        Image.new("RGB", (64, 32), color="white"),
        max_length=16,
        device=torch.device("cpu"),
    )
    assert seen.get("max_new_tokens") == 16


def test_recognize_line_returns_non_empty_string() -> None:
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)
    apply_trocr_generation_config(model, processor, max_length=16, num_beams=1)
    model.eval()
    text = recognize_line(
        model,
        processor,
        Image.new("RGB", (64, 32), color="white"),
        max_length=16,
        device=torch.device("cpu"),
    )
    assert isinstance(text, str)
    assert len(text) > 0


def test_recognize_line_rejects_tiny_crop() -> None:
    processor = create_dummy_processor(vocab_size=32, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=32, image_size=64)
    try:
        recognize_line(model, processor, Image.new("RGB", (8, 8), color="white"))
    except ValueError as exc:
        assert "16" in str(exc)
    else:
        raise AssertionError("tiny crop must be rejected")


def test_recognize_lines_batches_and_returns_one_string_each() -> None:
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)
    apply_trocr_generation_config(model, processor, max_length=16, num_beams=1)
    model.eval()
    images = [Image.new("RGB", (64, 32), color="white") for _ in range(3)]
    texts = recognize_lines(model, processor, images, max_length=16, device=torch.device("cpu"), batch_size=2)
    assert len(texts) == 3
    assert all(isinstance(text, str) and text for text in texts)

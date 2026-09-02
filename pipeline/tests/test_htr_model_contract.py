from pipeline.training.model_contract import apply_trocr_generation_config, resolve_htr_device
from pipeline.training.train import create_tiny_mock_model
from pipeline.training.dataset import create_dummy_processor


def test_apply_trocr_generation_config_sets_section_5_2_fields() -> None:
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)
    apply_trocr_generation_config(model, processor, max_length=128, num_beams=1)
    tok = processor.tokenizer
    assert model.config.decoder_start_token_id == tok.cls_token_id
    assert model.config.pad_token_id == tok.pad_token_id
    assert model.config.eos_token_id == tok.sep_token_id
    assert model.config.vocab_size == model.config.decoder.vocab_size
    assert model.generation_config.max_length == 128
    assert model.generation_config.early_stopping is False
    assert model.generation_config.num_beams == 1


def test_resolve_htr_device_returns_torch_device() -> None:
    device = resolve_htr_device()
    assert device.type in {"mps", "cuda", "cpu"}


def test_resolve_htr_device_honors_htr_device_env(monkeypatch) -> None:
    monkeypatch.setenv("HTR_DEVICE", "cpu")
    assert resolve_htr_device().type == "cpu"

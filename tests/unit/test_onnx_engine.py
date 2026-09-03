"""
tests/unit/test_onnx_engine.py
Unit tests for the High-Performance ONNX Runtime Serving Engine on CPU.
"""

from pathlib import Path
from PIL import Image
import pytest
import torch

from backend.app.onnx_engine import (
    is_onnx_available,
    load_onnx_htr_model,
    OnnxTrOCRModel,
    ONNXRUNTIME_AVAILABLE,
)
from backend.app.engine import InferenceEngine

ONNX_EXPORT_DIR = Path("export/trocr_base_iam_onnx")


@pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not installed")
@pytest.mark.skipif(not ONNX_EXPORT_DIR.exists(), reason="ONNX model export dir not found")
class TestOnnxEngine:

    def test_is_onnx_available(self) -> None:
        assert is_onnx_available(ONNX_EXPORT_DIR) is True
        assert is_onnx_available("non_existent_dir_12345") is False

    def test_load_onnx_model_and_processor(self) -> None:
        model, processor = load_onnx_htr_model(ONNX_EXPORT_DIR, num_threads=2)
        assert isinstance(model, OnnxTrOCRModel)
        assert model.config.decoder_start_token_id == 0
        assert model.config.eos_token_id == 2
        assert model.num_threads == 2

    def test_onnx_greedy_generation(self) -> None:
        model, processor = load_onnx_htr_model(ONNX_EXPORT_DIR, num_threads=2)
        img = Image.open("tests/fixtures/sample_clean_handwriting.png").convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values

        output = model.generate(
            pixel_values,
            num_beams=1,
            max_new_tokens=20,
            return_dict_in_generate=True,
            output_scores=True,
        )

        assert hasattr(output, "sequences")
        assert hasattr(output, "sequences_scores")
        assert output.sequences.shape[0] == 1
        decoded = processor.decode(output.sequences[0], skip_special_tokens=True).strip()
        assert len(decoded) > 0

    def test_onnx_beam_search_generation(self) -> None:
        model, processor = load_onnx_htr_model(ONNX_EXPORT_DIR, num_threads=2)
        img = Image.open("tests/fixtures/sample_clean_handwriting.png").convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values

        output = model.generate(
            pixel_values,
            num_beams=3,
            num_return_sequences=3,
            max_new_tokens=20,
            return_dict_in_generate=True,
            output_scores=True,
        )

        assert output.sequences.shape[0] == 3
        assert len(output.sequences_scores) == 3
        # Scores should be sorted descending
        scores = [float(s) for s in output.sequences_scores]
        assert scores[0] >= scores[1] >= scores[2]

    def test_inference_engine_onnx_integration(self) -> None:
        engine = InferenceEngine(mode="onnx")
        assert isinstance(engine.model, OnnxTrOCRModel)
        assert engine.device.type == "cpu"

        img = Image.open("tests/fixtures/sample_clean_handwriting.png")
        text = engine.recognize_single_crop(img)
        assert isinstance(text, str)
        assert len(text.strip()) > 0

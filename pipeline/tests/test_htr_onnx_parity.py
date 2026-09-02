from PIL import Image
import pytest

from pipeline.training.export_onnx import assert_greedy_parity


def test_assert_greedy_parity_detects_drift() -> None:
    class Proc:
        def __call__(self, images, return_tensors="pt"):
            class Out:
                pixel_values = 1

            return Out()

        def batch_decode(self, ids, skip_special_tokens=True):
            return [str(ids)]

    class Left:
        def generate(self, *args, **kwargs):
            return "hello"

    class Right:
        def generate(self, *args, **kwargs):
            return "hallo"

    with pytest.raises(AssertionError, match="ONNX greedy drift"):
        assert_greedy_parity(Left(), Right(), Proc(), [Image.new("RGB", (32, 16))])


def test_assert_greedy_parity_passes_when_equal() -> None:
    class Proc:
        def __call__(self, images, return_tensors="pt"):
            class Out:
                pixel_values = 1

            return Out()

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["same"]

    class M:
        def generate(self, *args, **kwargs):
            return "same"

    assert_greedy_parity(M(), M(), Proc(), [Image.new("RGB", (32, 16))])

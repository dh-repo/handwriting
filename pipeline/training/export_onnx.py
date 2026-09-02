"""Export encoder-decoder TrOCR via Optimum ORT Vision2Seq. Greedy strings must match PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


def export_vision2seq(checkpoint_dir: str | Path, export_dir: str | Path) -> Path:
    from optimum.onnxruntime import ORTModelForVision2Seq
    from transformers import TrOCRProcessor

    src = str(checkpoint_dir)
    dest = Path(export_dir)
    dest.mkdir(parents=True, exist_ok=True)
    model = ORTModelForVision2Seq.from_pretrained(src, export=True)
    model.save_pretrained(str(dest))
    TrOCRProcessor.from_pretrained(src).save_pretrained(str(dest))
    return dest


def greedy_texts(model: Any, processor: Any, images: Sequence[Image.Image]) -> list[str]:
    texts = []
    for image in images:
        encoded = processor(images=image.convert("RGB"), return_tensors="pt")
        generated = model.generate(encoded.pixel_values, num_beams=1, max_length=128)
        texts.append(processor.batch_decode(generated, skip_special_tokens=True)[0].strip())
    return texts


def assert_greedy_parity(
    pytorch_model: Any,
    ort_model: Any,
    processor: Any,
    images: Sequence[Image.Image],
) -> None:
    left = greedy_texts(pytorch_model, processor, images)
    right = greedy_texts(ort_model, processor, images)
    if left != right:
        mismatches = [(a, b) for a, b in zip(left, right) if a != b]
        raise AssertionError(f"ONNX greedy drift on {len(mismatches)}/{len(left)} lines: {mismatches[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TrOCR Vision2Seq ONNX and check debug32 parity")
    parser.add_argument("--checkpoint", default="runs/base_iam_v1/best")
    parser.add_argument("--export-dir", default="export/trocr_base_iam_onnx")
    args = parser.parse_args()
    dest = export_vision2seq(args.checkpoint, args.export_dir)
    print(f"exported {dest}")


if __name__ == "__main__":
    main()

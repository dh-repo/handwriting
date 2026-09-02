"""Single line-crop CLI. A page photo is out of contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import numpy as np
import torch

from pipeline.preprocessing.image_enhancement import suppress_ruling_lines

from pipeline.training.model_contract import (
    apply_trocr_generation_config,
    load_htr_model,
    load_htr_processor,
    resolve_htr_device,
)


def _prepare_line_image(
    image: Image.Image | str | Path,
    *,
    suppress_ruling: bool = True,
) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    rgb = image.convert("RGB")
    if min(rgb.size) < 16:
        raise ValueError("line crop is smaller than 16 px in one dimension")
    if not suppress_ruling:
        return rgb
    cleaned = suppress_ruling_lines(np.array(rgb))
    return Image.fromarray(cleaned)


def recognize_line(
    model: Any,
    processor: Any,
    image: Image.Image | str | Path,
    max_length: int = 128,
    num_beams: int = 1,
    device: str | torch.device | None = None,
    suppress_ruling: bool = True,
) -> str:
    rgb = _prepare_line_image(image, suppress_ruling=suppress_ruling)
    encoded = processor(images=rgb, return_tensors="pt")
    pixel_values = encoded.pixel_values
    target = device
    if target is None and hasattr(model, "parameters"):
        try:
            target = next(model.parameters()).device
        except StopIteration:
            target = None
    if target is not None and hasattr(model, "to"):
        pixel_values = pixel_values.to(target)
        model = model.to(target)
    generate_kwargs: dict[str, Any] = {
        "num_beams": num_beams,
        "max_new_tokens": max_length,
    }
    with torch.inference_mode():
        generated = model.generate(pixel_values, **generate_kwargs)
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    tok = getattr(processor, "tokenizer", processor)
    return tok.batch_decode(generated, skip_special_tokens=True)[0].strip()


def recognize_lines(
    model: Any,
    processor: Any,
    images: Sequence[Image.Image | str | Path],
    max_length: int = 128,
    num_beams: int = 1,
    device: str | torch.device | None = None,
    batch_size: int = 8,
) -> list[str]:
    """Batched line recognition. Same contract as recognize_line, one string per crop."""
    opened: list[Image.Image] = [_prepare_line_image(image) for image in images]

    target = device
    if target is None and hasattr(model, "parameters"):
        try:
            target = next(model.parameters()).device
        except StopIteration:
            target = None

    texts: list[str] = []
    step = max(1, int(batch_size))
    for start in range(0, len(opened), step):
        chunk = opened[start : start + step]
        encoded = processor(images=chunk, return_tensors="pt")
        pixel_values = encoded.pixel_values
        if target is not None and hasattr(model, "to"):
            pixel_values = pixel_values.to(target)
            model = model.to(target)
        generate_kwargs: dict[str, Any] = {
            "num_beams": num_beams,
            "max_new_tokens": max_length,
        }
        with torch.inference_mode():
            generated = model.generate(pixel_values, **generate_kwargs)
        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(generated, skip_special_tokens=True)
        else:
            tok = getattr(processor, "tokenizer", processor)
            decoded = tok.batch_decode(generated, skip_special_tokens=True)
        texts.extend(text.strip() for text in decoded)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize one handwritten line crop")
    parser.add_argument("image", type=str)
    parser.add_argument("--model", type=str, default="microsoft/trocr-base-handwritten")
    parser.add_argument("--beams", type=int, default=1)
    args = parser.parse_args()

    processor = load_htr_processor(args.model)
    model = load_htr_model(args.model)
    apply_trocr_generation_config(model, processor, max_length=128, num_beams=args.beams)
    device = resolve_htr_device()
    model.to(device)
    text = recognize_line(model, processor, args.image, num_beams=args.beams, device=device)
    print(text)


if __name__ == "__main__":
    main()

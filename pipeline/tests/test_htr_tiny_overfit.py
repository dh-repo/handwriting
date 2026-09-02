from pathlib import Path

from PIL import Image
import torch

from pipeline.training.data import LineCropCollator, LineCropDataset
from pipeline.training.dataset import create_dummy_processor
from pipeline.training.model_contract import apply_trocr_generation_config
from pipeline.training.train import create_tiny_mock_model


def test_tiny_teacher_force_overfit_two_lines(tmp_path: Path) -> None:
    split = tmp_path / "debug2"
    images = split / "images"
    images.mkdir(parents=True)
    rows = []
    for i, text in enumerate(["AB", "CD"]):
        name = f"train_{i:05d}.png"
        Image.new("RGB", (64, 64), color=(200, 180, 160)).save(images / name)
        rows.append(f"{name}\t{text}")
    (split / "labels.tsv").write_text("\n".join(rows), encoding="utf-8")

    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)
    apply_trocr_generation_config(model, processor, max_length=16, num_beams=1)
    ds = LineCropDataset(split, processor=processor, is_train=True, augment=False, max_target_length=16)
    collator = LineCropCollator(processor, max_target_length=16)
    batch = collator([ds[0], ds[1]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    first = None
    last = None
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        out = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        out.loss.backward()
        optimizer.step()
        last = float(out.loss.detach())
        if first is None:
            first = last
    assert last < first
    assert last < 2.0

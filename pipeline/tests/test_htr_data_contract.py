from pathlib import Path

import pytest
from PIL import Image
import torch

from pipeline.training.data import (
    LineCropCollator,
    LineCropDataset,
    parse_labels_tsv,
    parse_private_row,
    split_rows_by_writer,
)
from pipeline.training.dataset import create_dummy_processor
from pipeline.training.model_contract import load_htr_processor


def _write_split(root: Path, texts: list[str]) -> Path:
    images = root / "images"
    images.mkdir(parents=True)
    rows = []
    for i, text in enumerate(texts):
        name = f"train_{i:05d}.png"
        Image.new("RGB", (120, 32), color=(210, 200, 190)).save(images / name)
        rows.append(f"{name}\t{text}")
    (root / "labels.tsv").write_text("\n".join(rows), encoding="utf-8")
    return root


def test_parse_labels_tsv_is_tab_delimited(tmp_path: Path) -> None:
    split = _write_split(tmp_path, ["Hello", "World"])
    rows = parse_labels_tsv(split)
    assert rows[0]["text"] == "Hello"
    assert rows[0]["image_path"].name == "train_00000.png"


def test_dataset_returns_processor_pixel_values_and_rgb(tmp_path: Path) -> None:
    processor = create_dummy_processor(size=(64, 64), vocab_size=50)
    ds = LineCropDataset(_write_split(tmp_path, ["Hello"]), processor=processor, is_train=False)
    item = ds[0]
    assert item["pixel_values"].shape[0] == 3
    assert item["text"] == "Hello"
    assert "input_ids" in item


def test_collator_masks_pad_with_minus_100() -> None:
    processor = create_dummy_processor(vocab_size=50)
    collator = LineCropCollator(processor, max_target_length=16)
    batch = collator(
        [
            {
                "pixel_values": torch.zeros(3, 32, 32),
                "text": "Hi",
                "input_ids": torch.tensor([0, 10, 2]),
            },
            {
                "pixel_values": torch.zeros(3, 32, 32),
                "text": "Hello there",
                "input_ids": torch.tensor([0, 11, 12, 13, 2]),
            },
        ]
    )
    assert (batch["labels"] == -100).any()
    assert batch["labels"].shape[0] == 2


def test_augmentation_flag_only_on_train(tmp_path: Path) -> None:
    processor = create_dummy_processor(size=(64, 64), vocab_size=50)
    split = _write_split(tmp_path, ["Hello"])
    train_ds = LineCropDataset(split, processor=processor, is_train=True, augment=True)
    eval_ds = LineCropDataset(split, processor=processor, is_train=False, augment=True)
    assert train_ds.augment is True
    assert eval_ds.augment is False


def test_private_row_and_writer_split() -> None:
    row = parse_private_row(
        "a.png\tTake two\tsrc1\twriter_a\t0,0,10,10\t1"
    )
    assert row["source_id"] == "src1"
    assert row["writer_hash"] == "writer_a"
    assert row["human_corrected"] is True
    groups = split_rows_by_writer(
        [
            {"writer_hash": "a", "text": "1"},
            {"writer_hash": "a", "text": "2"},
            {"writer_hash": "b", "text": "3"},
        ]
    )
    assert [r["text"] for r in groups["a"]] == ["1", "2"]
    assert [r["text"] for r in groups["b"]] == ["3"]


@pytest.mark.skipif(not Path("runs/debug32/best").is_dir(), reason="debug32 checkpoint not on disk")
def test_live_processor_letterboxes_to_384() -> None:
    processor = load_htr_processor("runs/debug32/best")
    encoded = processor(images=Image.new("RGB", (800, 40), color="white"), return_tensors="pt")
    assert tuple(encoded.pixel_values.shape[-2:]) == (384, 384)

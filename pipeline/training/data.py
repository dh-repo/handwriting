"""
Line-crop dataset for TrOCR. Images stay on disk as clean PNGs.
Resize/normalize is TrOCRProcessor only — no custom 384 interpolate.
Augmentation runs on PIL before processor(), train split only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset


def parse_labels_tsv(split_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(split_dir)
    rows: list[dict[str, Any]] = []
    for line in (root / "labels.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, text = line.split("\t", 1)
        rows.append({"image_path": root / "images" / name, "text": text, "filename": name})
    return rows


def parse_private_row(line: str) -> dict[str, Any]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 6:
        raise ValueError(
            "private TSV row must be filename, text, source_id, writer_hash, crop_box, human_corrected"
        )
    filename, text, source_id, writer_hash, crop_box, corrected = parts[:6]
    flag = str(corrected).strip().lower()
    return {
        "filename": filename,
        "text": text,
        "source_id": source_id,
        "writer_hash": writer_hash,
        "crop_box": crop_box,
        "human_corrected": flag in {"1", "true", "yes"},
    }


def split_rows_by_writer(rows: Iterable[dict[str, Any]], field: str = "writer_hash") -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        groups.setdefault(key, []).append(row)
    return groups


def _maybe_augment(image: Image.Image) -> Image.Image:
    try:
        import albumentations as A
        import numpy as np
    except ImportError:
        return image
    arr = np.array(image)
    transform = A.Compose(
        [
            A.Affine(rotate=(-3, 3), shear=(-4, 4), scale=(0.96, 1.04), p=0.5),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.RandomBrightnessContrast(p=0.4),
        ]
    )
    return Image.fromarray(transform(image=arr)["image"])


class LineCropDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        processor: Any,
        max_target_length: int = 128,
        is_train: bool = False,
        augment: bool = False,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.processor = processor
        self.max_target_length = max_target_length
        self.is_train = is_train
        self.augment = bool(augment and is_train)
        self.transform = transform
        self.rows = parse_labels_tsv(self.split_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.augment:
            image = self.transform(image) if self.transform is not None else _maybe_augment(image)
        encoded = self.processor(images=image, return_tensors="pt")
        pixel_values = encoded.pixel_values.squeeze(0)
        tok = getattr(self.processor, "tokenizer", self.processor)
        tokens = tok(
            row["text"],
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)
        return {
            "pixel_values": pixel_values,
            "text": row["text"],
            "input_ids": input_ids,
            "image_path": str(row["image_path"]),
        }


class LineCropCollator:
    def __init__(self, processor: Any, max_target_length: int = 128) -> None:
        self.processor = processor
        self.max_target_length = max_target_length
        tok = getattr(processor, "tokenizer", processor)
        self.pad_token_id = getattr(tok, "pad_token_id", 1) or 1

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
        ids = []
        for item in batch:
            raw = item["input_ids"]
            if not isinstance(raw, torch.Tensor):
                raw = torch.tensor(raw, dtype=torch.long)
            ids.append(raw[: self.max_target_length])
        padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        if padded.shape[1] > self.max_target_length:
            padded = padded[:, : self.max_target_length]
        labels = padded.clone()
        labels[padded == self.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

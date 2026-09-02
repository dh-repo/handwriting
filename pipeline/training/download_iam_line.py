"""
Download Teklia/IAM-line into the line-crop TSV contract.

Contract: data/iam_line/{split}/images/{split}_{i:05d}.png
          data/iam_line/{split}/labels.tsv   filename<TAB>transcript
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

logger = logging.getLogger(__name__)

IAM_LINE_COUNTS = {"train": 6482, "validation": 976, "test": 2915}
DATASET_ID = "Teklia/IAM-line"


def write_labels_tsv(path: Path, lines: Sequence[str]) -> None:
    """Write filename<TAB>transcript rows with a trailing newline so wc -l matches count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def data_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("HTR_DATA_ROOT")
    if env:
        return Path(env)
    return Path("data")


def iam_line_root(explicit: str | Path | None = None) -> Path:
    return data_root(explicit) / "iam_line"


def _as_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def write_split_from_examples(
    examples: Iterable[Mapping[str, Any]],
    split_dir: Path,
    split_name: str,
) -> list[tuple[str, str]]:
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    widths: list[int] = []
    for i, ex in enumerate(examples):
        text = str(ex.get("text") or "").replace("\t", " ").strip()
        if not text:
            raise ValueError(f"empty transcript at {split_name} index {i}")
        name = f"{split_name}_{i:05d}.png"
        rgb = _as_rgb(ex["image"])
        widths.append(rgb.size[0])
        rgb.save(images_dir / name)
        rows.append((name, text))
    write_labels_tsv(split_dir / "labels.tsv", [f"{name}\t{text}" for name, text in rows])
    if widths:
        logger.info("%s rows=%s max_width=%s", split_name, len(rows), max(widths))
    return rows


def write_debug32_subset(train_dir: Path, dest: Path, n: int = 32) -> int:
    labels_path = train_dir / "labels.tsv"
    lines = [ln for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    take = lines[:n]
    dest_images = dest / "images"
    dest_images.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for line in take:
        name, _text = line.split("\t", 1)
        src = train_dir / "images" / name
        shutil.copy2(src, dest_images / name)
        written.append(line)
    write_labels_tsv(dest / "labels.tsv", written)
    return len(written)


def download_iam_line(
    dest_root: str | Path | None = None,
    splits: Sequence[str] | None = None,
    dataset_id: str = DATASET_ID,
) -> Path:
    from datasets import load_dataset

    root = iam_line_root(dest_root)
    ds = load_dataset(dataset_id)
    wanted = list(splits) if splits is not None else list(IAM_LINE_COUNTS)
    for split in wanted:
        expected = IAM_LINE_COUNTS.get(split)
        rows = write_split_from_examples(ds[split], root / split, split_name=split)
        if expected is not None and len(rows) != expected:
            raise ValueError(f"{split} count {len(rows)} != Teklia pack {expected}")
    write_debug32_subset(root / "train", data_root(dest_root) / "debug32", n=32)
    (data_root(dest_root) / "private" / "train").mkdir(parents=True, exist_ok=True)
    (data_root(dest_root) / "private" / "validation").mkdir(parents=True, exist_ok=True)
    return root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Download Teklia/IAM-line as PNG + labels.tsv")
    parser.add_argument("--dest", type=str, default=None, help="Override data root (or set HTR_DATA_ROOT)")
    args = parser.parse_args()
    root = download_iam_line(args.dest)
    print(f"Wrote IAM-line pack under {root}")


if __name__ == "__main__":
    main()

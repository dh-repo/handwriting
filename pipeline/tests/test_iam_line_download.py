from pathlib import Path

import pytest
from PIL import Image

from pipeline.training.download_iam_line import (
    IAM_LINE_COUNTS,
    write_debug32_subset,
    write_labels_tsv,
    write_split_from_examples,
)


def _example(text: str, size=(180, 48), color=(200, 180, 160)) -> dict:
    return {"image": Image.new("RGB", size, color=color), "text": text}


def test_write_split_uses_tab_tsv_and_png_names(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    rows = write_split_from_examples(
        [_example("Hello World"), _example("Second\tline")],
        split_dir,
        split_name="train",
    )
    assert rows == [("train_00000.png", "Hello World"), ("train_00001.png", "Second line")]
    labels = (split_dir / "labels.tsv").read_text(encoding="utf-8")
    assert labels.endswith("\n")
    assert labels.count("\n") == 2
    assert "\t" in labels
    assert "," not in labels.splitlines()[0].split("\t")[0]
    assert (split_dir / "images" / "train_00000.png").is_file()
    img = Image.open(split_dir / "images" / "train_00000.png")
    assert img.mode == "RGB"


def test_write_split_rejects_empty_labels(tmp_path: Path) -> None:
    try:
        write_split_from_examples([_example("   ")], tmp_path / "train", split_name="train")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("empty transcript must be rejected")


def test_write_labels_tsv_trailing_newline(tmp_path: Path) -> None:
    dest = tmp_path / "labels.tsv"
    write_labels_tsv(dest, ["a.png\thello"])
    assert dest.read_text(encoding="utf-8") == "a.png\thello\n"


@pytest.mark.skipif(
    not Path("data/iam_line/train/labels.tsv").is_file(),
    reason="Teklia pack not on disk",
)
def test_live_iam_line_pack_counts() -> None:
    from pipeline.training.data import parse_labels_tsv

    for split, expected in IAM_LINE_COUNTS.items():
        rows = parse_labels_tsv(Path("data/iam_line") / split)
        assert len(rows) == expected
        assert all(row["image_path"].is_file() for row in rows)
    debug_lines = [
        line for line in Path("data/debug32/labels.tsv").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(debug_lines) == 32


def test_iam_line_counts_match_teklia_pack() -> None:
    assert IAM_LINE_COUNTS == {"train": 6482, "validation": 976, "test": 2915}
    assert sum(IAM_LINE_COUNTS.values()) == 10373


def test_debug32_copies_first_32_train_rows(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    write_split_from_examples(
        [_example(f"line {i}") for i in range(40)],
        train_dir,
        split_name="train",
    )
    dest = tmp_path / "debug32"
    n = write_debug32_subset(train_dir, dest, n=32)
    assert n == 32
    lines = (dest / "labels.tsv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32
    assert (dest / "images" / "train_00000.png").is_file()

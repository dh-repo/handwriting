"""Unit tests for public handwriting dataset curation helpers."""

from pathlib import Path

from PIL import Image

from pipeline.dataset.download_and_curate_multisource import (
    EXCLUDED_SYNTHETIC_REPOS,
    HF_SOURCES,
    MIN_HEIGHT,
    MIN_WIDTH,
    build_record,
    canonicalize_split_name,
    enabled_sources,
    extract_transcription,
    extract_writer_id,
    pad_to_min_size,
    read_png_size,
    split_from_writer,
    to_pil_image,
)


def test_enabled_sources_exclude_synthetic_repos():
    keys = {src.repo_id for src in enabled_sources()}
    assert keys.isdisjoint(EXCLUDED_SYNTHETIC_REPOS)
    assert "Teklia/IAM-line" in keys
    assert "priyank-m/IAM_words_text_recognition" in keys
    assert "cyttic/eng-iam-text" not in keys


def test_enabled_sources_filter_by_key():
    selected = enabled_sources(["iam_line", "rimes_2011_line"])
    assert [src.source_key for src in selected] == ["iam_line", "rimes_2011_line"]


def test_extract_transcription_normalizes_iam_pipes_and_json():
    assert extract_transcription({"text": "A|MOVE|to|stop"}, "text") == "A MOVE to stop"
    assert extract_transcription({"text": "   "}, "text") == ""
    assert extract_transcription({"text": "---"}, "text") == ""
    dumped = extract_transcription({"ground_truth": {"DATE": "8-3-89"}}, "ground_truth")
    assert "8-3-89" in dumped


def test_writer_ids_are_source_namespaced_and_splits_are_stable():
    writer = extract_writer_id({"writer": "writer59"}, "iam_words_writer_labeled", "writer", "fallback")
    assert writer == "iam_words_writer_labeled::writer59"
    assert split_from_writer(writer, seed=42) == split_from_writer(writer, seed=42)
    fallback = extract_writer_id({}, "iam_line", None, "iam_line_train_00000001")
    assert fallback.startswith("iam_line::")


def test_canonical_splits_and_record_schema():
    assert canonicalize_split_name("validation") == "val"
    record = build_record(
        sample_id="iam_line_train_00000000",
        relative_image_path="images/iam_line/iam_line_train_00000000.png",
        transcription="put down a resolution on the subject",
        writer_id="iam_line::iam_line_train_00000000",
        source_key="iam_line",
        repo_id="Teklia/IAM-line",
        category="general_cursive_line",
        split="train",
        width=2467,
        height=128,
        language="en",
        homepage="https://huggingface.co/datasets/Teklia/IAM-line",
    )
    required = {
        "id", "sample_id", "image_path", "relative_image_path",
        "transcription", "text", "writer_id", "dataset_source",
        "source", "category", "is_lasa", "therapeutic_class",
        "width", "height", "augmentation_params",
    }
    assert required.issubset(record.keys())
    assert record["augmentation_params"]["downloaded_original"] is True
    assert record["source"] == "Teklia/IAM-line"


def test_pad_and_pil_conversion(tmp_path: Path):
    tiny = Image.new("RGB", (10, 8), (0, 0, 0))
    padded = pad_to_min_size(tiny)
    assert padded.size[0] >= MIN_WIDTH
    assert padded.size[1] >= MIN_HEIGHT

    dest = tmp_path / "tiny.png"
    tiny.save(dest)
    loaded = to_pil_image({"path": str(dest)})
    assert loaded is not None
    assert loaded.mode == "RGB"


def test_read_png_size_from_header(tmp_path: Path):
    dest = tmp_path / "header.png"
    Image.new("RGB", (64, 24), (12, 34, 56)).save(dest)
    assert read_png_size(dest) == (64, 24)


def test_catalog_has_no_synthetic_generator_sources():
    assert all(src.repo_id not in EXCLUDED_SYNTHETIC_REPOS for src in HF_SOURCES)
    assert len(HF_SOURCES) >= 8

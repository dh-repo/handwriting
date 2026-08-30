"""
Download and curate real public handwriting corpora into data/reference_handwriting/.

Only online reference datasets are ingested. Synthetic / font-rendered generators are
not used for the training corpus.

Sources (ungated Hugging Face image+transcription sets, Latin-script HTR):
- Teklia/IAM-line
- priyank-m/IAM_words_text_recognition
- alpayariyak/IAM_Sentences
- Teklia/RIMES-2011-line
- fhswf/german_handwriting
- Adarsh203/Handwriting_dataset_new
- jhc90/IMGUR5K_handwriting_cropped_data
- ift/handwriting_forms
- Teklia/Belfort-line
- Teklia/POPP-line
- Teklia/Esposalles-line
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multisource_curator")

MIN_WIDTH = 32
MIN_HEIGHT = 16
DEFAULT_HF_CACHE = PROJECT_ROOT / "data" / ".hf_cache"

# Explicitly excluded: font-rendered or synthetic corpora (not scanned handwriting).
EXCLUDED_SYNTHETIC_REPOS = frozenset(
    {
        "cyttic/eng-iam-text",
        "chinmays18/medical-prescription-dataset",
        "nehaMe123/indian-doctor-handwriting-synthetic",
    }
)


@dataclass(frozen=True)
class HuggingFaceHandwritingSource:
    repo_id: str
    source_key: str
    category: str
    image_field: str = "image"
    text_field: str = "text"
    writer_field: Optional[str] = None
    official_splits: Tuple[str, ...] = ("train", "validation", "test")
    language: str = "en"
    homepage: str = ""
    license_note: str = "see Hugging Face dataset card"
    streaming: bool = False
    enabled: bool = True


# Real, ungated, image+transcription handwriting datasets. No synthetic generators.
HF_SOURCES: Tuple[HuggingFaceHandwritingSource, ...] = (
    HuggingFaceHandwritingSource(
        repo_id="Teklia/IAM-line",
        source_key="iam_line",
        category="general_cursive_line",
        homepage="https://huggingface.co/datasets/Teklia/IAM-line",
        license_note="IAM handwriting database redistribution; see Teklia card / FKI terms",
    ),
    HuggingFaceHandwritingSource(
        repo_id="priyank-m/IAM_words_text_recognition",
        source_key="iam_words",
        category="general_cursive_word",
        homepage="https://huggingface.co/datasets/priyank-m/IAM_words_text_recognition",
        license_note="IAM word images; see dataset card / FKI terms",
        official_splits=("train", "val", "test"),
    ),
    HuggingFaceHandwritingSource(
        repo_id="alpayariyak/IAM_Sentences",
        source_key="iam_sentences",
        category="general_cursive_sentence",
        official_splits=("train",),
        homepage="https://huggingface.co/datasets/alpayariyak/IAM_Sentences",
        license_note="IAM sentence images; see dataset card / FKI terms",
    ),
    HuggingFaceHandwritingSource(
        repo_id="Teklia/RIMES-2011-line",
        source_key="rimes_2011_line",
        category="general_cursive_line",
        language="fr",
        homepage="https://huggingface.co/datasets/Teklia/RIMES-2011-line",
        license_note="RIMES 2011; see Teklia dataset card",
    ),
    HuggingFaceHandwritingSource(
        repo_id="fhswf/german_handwriting",
        source_key="german_handwriting",
        category="general_cursive_line",
        official_splits=("train",),
        language="de",
        homepage="https://huggingface.co/datasets/fhswf/german_handwriting",
        license_note="see fhswf/german_handwriting dataset card",
    ),
    HuggingFaceHandwritingSource(
        repo_id="Adarsh203/Handwriting_dataset_new",
        source_key="iam_words_writer_labeled",
        category="general_cursive_word",
        text_field="label",
        writer_field="writer",
        official_splits=("train",),
        homepage="https://huggingface.co/datasets/Adarsh203/Handwriting_dataset_new",
        license_note="IAM-derived word crops with writer ids; see dataset card",
    ),
    HuggingFaceHandwritingSource(
        repo_id="jhc90/IMGUR5K_handwriting_cropped_data",
        source_key="imgur5k_words",
        category="in_the_wild_word",
        image_field="img_style",
        text_field="content_style",
        homepage="https://huggingface.co/datasets/jhc90/IMGUR5K_handwriting_cropped_data",
        license_note="IMGUR5K CC-BY-NC 4.0 (Facebook AI TextStyleBrush)",
        official_splits=("train", "val", "test"),
        streaming=True,
    ),
    HuggingFaceHandwritingSource(
        repo_id="ift/handwriting_forms",
        source_key="handwriting_forms",
        category="handwritten_form",
        text_field="ground_truth",
        homepage="https://huggingface.co/datasets/ift/handwriting_forms",
        license_note="see ift/handwriting_forms dataset card",
    ),
    HuggingFaceHandwritingSource(
        repo_id="Teklia/Belfort-line",
        source_key="belfort_line",
        category="historical_line",
        language="fr",
        homepage="https://huggingface.co/datasets/Teklia/Belfort-line",
        license_note="see Teklia/Belfort-line dataset card",
    ),
    HuggingFaceHandwritingSource(
        repo_id="Teklia/POPP-line",
        source_key="popp_line",
        category="historical_line",
        language="fr",
        homepage="https://huggingface.co/datasets/Teklia/POPP-line",
        license_note="see Teklia/POPP-line dataset card",
    ),
    HuggingFaceHandwritingSource(
        repo_id="Teklia/Esposalles-line",
        source_key="esposalles_line",
        category="historical_line",
        language="ca",
        homepage="https://huggingface.co/datasets/Teklia/Esposalles-line",
        license_note="see Teklia/Esposalles-line dataset card",
    ),
)


def enabled_sources(source_keys: Optional[Sequence[str]] = None) -> List[HuggingFaceHandwritingSource]:
    selected = list(HF_SOURCES)
    if source_keys:
        wanted = {key.strip() for key in source_keys if key.strip()}
        selected = [src for src in selected if src.source_key in wanted]
        missing = wanted - {src.source_key for src in selected}
        if missing:
            raise ValueError(f"Unknown source keys: {sorted(missing)}")
    return [src for src in selected if src.enabled and src.repo_id not in EXCLUDED_SYNTHETIC_REPOS]


def extract_transcription(row: Dict[str, Any], text_field: str) -> str:
    value = row.get(text_field)
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).replace("|", " ").strip()
    if text in {"", ".", "-", "--", "---", "nan", "None", "null"}:
        return ""
    return " ".join(text.split())


def extract_writer_id(row: Dict[str, Any], source_key: str, writer_field: Optional[str], fallback: str) -> str:
    if writer_field:
        raw = row.get(writer_field)
        if raw is not None:
            writer = str(raw).strip()
            if writer:
                return f"{source_key}::{writer}"
    return f"{source_key}::{fallback}"


def split_from_writer(writer_id: str, seed: int = 42) -> str:
    digest = hashlib.md5(f"{seed}:{writer_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def canonicalize_split_name(split: str) -> str:
    if split in {"validation", "val", "valid"}:
        return "val"
    if split == "test":
        return "test"
    return "train"


def to_pil_image(image_value: Any) -> Optional[Image.Image]:
    if image_value is None:
        return None
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict):
        raw_bytes = image_value.get("bytes")
        if raw_bytes:
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        path = image_value.get("path")
        if path and Path(path).exists():
            return Image.open(path).convert("RGB")
    return None


def pad_to_min_size(image: Image.Image, min_width: int = MIN_WIDTH, min_height: int = MIN_HEIGHT) -> Image.Image:
    width, height = image.size
    pad_w = max(0, min_width - width)
    pad_h = max(0, min_height - height)
    if pad_w == 0 and pad_h == 0:
        return image
    padded = Image.new("RGB", (width + pad_w, height + pad_h), (255, 255, 255))
    padded.paste(image, (pad_w // 2, pad_h // 2))
    return padded


def build_record(
    *,
    sample_id: str,
    relative_image_path: str,
    transcription: str,
    writer_id: str,
    source_key: str,
    repo_id: str,
    category: str,
    split: str,
    width: int,
    height: int,
    language: str,
    homepage: str,
) -> Dict[str, Any]:
    image_path = str(Path("data/reference_handwriting") / relative_image_path)
    return {
        "id": sample_id,
        "sample_id": sample_id,
        "image_path": image_path,
        "relative_image_path": relative_image_path,
        "transcription": transcription,
        "text": transcription,
        "writer_id": writer_id,
        "dataset_source": source_key,
        "source": repo_id,
        "category": category,
        "split": split,
        "is_lasa": False,
        "therapeutic_class": "general",
        "width": width,
        "height": height,
        "language": language,
        "homepage": homepage,
        "augmentation_params": {"downloaded_original": True},
        "metadata": {
            "origin": "huggingface",
            "repo_id": repo_id,
        },
    }


def _configure_hf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))


def _iter_split_rows(repo_id: str, split: str, streaming: bool) -> Iterator[Dict[str, Any]]:
    dataset = load_dataset(repo_id, split=split, streaming=streaming)
    try:
        for row in dataset:
            yield dict(row)
    finally:
        del dataset


def read_png_size(path: Path) -> Tuple[int, int]:
    """Read width/height from a PNG header without decoding pixels."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height
    with Image.open(path) as existing:
        return existing.size


def _save_png(image: Image.Image, dest: Path) -> Tuple[int, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rgb = pad_to_min_size(image.convert("RGB"))
    rgb.save(dest, format="PNG", optimize=False)
    return rgb.size


def _remove_synthetic_images(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    removed = 0
    for path in images_dir.glob("*_syn.png"):
        path.unlink(missing_ok=True)
        removed += 1
    leftover_dirs = [p for p in images_dir.iterdir() if p.is_file()]
    for path in leftover_dirs:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.name.startswith("sample_"):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def ingest_huggingface_source(
    source: HuggingFaceHandwritingSource,
    output_dir: Path,
    seed: int,
    max_per_source: Optional[int],
    existing_only: bool = False,
) -> List[Dict[str, Any]]:
    if source.repo_id in EXCLUDED_SYNTHETIC_REPOS:
        logger.warning("Skipping excluded synthetic repo %s", source.repo_id)
        return []

    images_root = output_dir / "images" / source.source_key
    images_root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    taken = 0
    has_official_holdout = any(
        canonicalize_split_name(name) in {"val", "test"} for name in source.official_splits
    )

    for raw_split in source.official_splits:
        canonical_split = canonicalize_split_name(raw_split)
        logger.info("Downloading %s split=%s", source.repo_id, raw_split)
        try:
            rows = _iter_split_rows(source.repo_id, raw_split, streaming=source.streaming)
        except Exception as exc:
            logger.warning("Skipping %s split %s: %s", source.repo_id, raw_split, exc)
            continue

        for index, row in enumerate(tqdm(rows, desc=f"{source.source_key}/{raw_split}", unit="img")):
            if max_per_source is not None and taken >= max_per_source:
                break

            sample_id = f"{source.source_key}_{canonical_split}_{index:08d}"
            relative_image_path = f"images/{source.source_key}/{sample_id}.png"
            dest = output_dir / relative_image_path
            already_on_disk = dest.exists() and dest.stat().st_size > 100

            transcription = extract_transcription(row, source.text_field)
            if not transcription:
                continue

            writer_id = extract_writer_id(row, source.source_key, source.writer_field, fallback=sample_id)
            split = canonical_split if has_official_holdout else split_from_writer(writer_id, seed=seed)

            try:
                if already_on_disk:
                    width, height = read_png_size(dest)
                elif existing_only:
                    continue
                else:
                    image = to_pil_image(row.get(source.image_field))
                    if image is None:
                        continue
                    width, height = _save_png(image, dest)
            except Exception as exc:
                logger.debug("Failed to save %s: %s", sample_id, exc)
                continue

            records.append(
                build_record(
                    sample_id=sample_id,
                    relative_image_path=relative_image_path,
                    transcription=transcription,
                    writer_id=writer_id,
                    source_key=source.source_key,
                    repo_id=source.repo_id,
                    category=source.category,
                    split=split,
                    width=width,
                    height=height,
                    language=source.language,
                    homepage=source.homepage,
                )
            )
            taken += 1
        if max_per_source is not None and taken >= max_per_source:
            break

    logger.info("Ingested %s samples from %s", len(records), source.repo_id)
    return records


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifests(output_dir: Path, records: List[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    test_records = [r for r in records if r["split"] == "test"]

    _write_jsonl(output_dir / "train_manifest.jsonl", train_records)
    _write_jsonl(output_dir / "val_manifest.jsonl", val_records)
    _write_jsonl(output_dir / "test_manifest.jsonl", test_records)
    _write_jsonl(output_dir / "full_manifest.jsonl", records)

    train_writers = {r["writer_id"] for r in train_records}
    val_writers = {r["writer_id"] for r in val_records}
    test_writers = {r["writer_id"] for r in test_records}

    sources_count: Dict[str, int] = {}
    categories_count: Dict[str, int] = {}
    language_count: Dict[str, int] = {}
    for record in records:
        sources_count[record["dataset_source"]] = sources_count.get(record["dataset_source"], 0) + 1
        categories_count[record["category"]] = categories_count.get(record["category"], 0) + 1
        language_count[record.get("language", "unknown")] = language_count.get(record.get("language", "unknown"), 0) + 1

    summary = {
        "dataset_name": "Public Reference Handwriting Corpora",
        "version": "3.0.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "origin": "downloaded_online_reference_datasets_only",
        "synthetic_samples": 0,
        "total_samples": len(records),
        "splits": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "split_ratios": {
            "train": round(len(train_records) / max(len(records), 1), 4),
            "val": round(len(val_records) / max(len(records), 1), 4),
            "test": round(len(test_records) / max(len(records), 1), 4),
        },
        # unique_writers from writer_id sets is not a census: fallback IDs are source_key::sample_id.
        "unique_writers": {
            "total": len(train_writers | val_writers | test_writers),
            "train": len(train_writers),
            "val": len(val_writers),
            "test": len(test_writers),
            "writer_overlap_train_val": len(train_writers & val_writers),
            "writer_overlap_train_test": len(train_writers & test_writers),
            "writer_overlap_val_test": len(val_writers & test_writers),
        },
        "sources": sources_count,
        "categories": categories_count,
        "languages": language_count,
        "huggingface_repos": [
            {
                "repo_id": src.repo_id,
                "source_key": src.source_key,
                "homepage": src.homepage,
                "license_note": src.license_note,
            }
            for src in HF_SOURCES
            if src.source_key in sources_count
        ],
        "excluded_synthetic_repos": sorted(EXCLUDED_SYNTHETIC_REPOS),
        "download_time_seconds": round(elapsed, 2),
        "throughput_samples_per_sec": round(len(records) / max(elapsed, 0.001), 2),
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def curate_multisource_dataset(
    output_dir: str = "data/reference_handwriting",
    vocab_dir: str = "data/reference_handwriting/vocabularies",
    total_target: int = 50500,
    max_workers: int = 20,
    seed: int = 42,
    source_keys: Optional[Sequence[str]] = None,
    max_per_source: Optional[int] = None,
    replace_synthetic: bool = True,
    cache_dir: Optional[str] = None,
    existing_only: bool = False,
) -> Dict[str, Any]:
    """
    Download public handwriting datasets and write 80/10/10 (or official) manifests.

    `vocab_dir`, `total_target`, and `max_workers` are accepted for CLI compatibility
    with the previous curator. Vocabularies are left untouched. Samples are not
    synthetically generated.
    """
    del vocab_dir, max_workers
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "images").mkdir(parents=True, exist_ok=True)

    _configure_hf_cache(Path(cache_dir) if cache_dir else DEFAULT_HF_CACHE)

    if replace_synthetic:
        removed = _remove_synthetic_images(out_path / "images")
        logger.info("Removed %s locally generated synthetic images", removed)

    sources = enabled_sources(source_keys)
    logger.info("Curating %s Hugging Face handwriting sources (no synthetic generation)", len(sources))
    started = time.time()
    all_records: List[Dict[str, Any]] = []
    failures: List[str] = []

    for source in sources:
        try:
            records = ingest_huggingface_source(
                source,
                output_dir=out_path,
                seed=seed,
                max_per_source=max_per_source,
                existing_only=existing_only,
            )
            all_records.extend(records)
        except Exception as exc:
            logger.exception("Failed to ingest %s", source.repo_id)
            failures.append(f"{source.repo_id}: {exc}")

    if not all_records:
        raise RuntimeError(
            "No handwriting samples were downloaded. Check network access and Hugging Face availability. "
            + (f"Failures: {failures}" if failures else "")
        )

    if total_target > 0 and len(all_records) > total_target:
        logger.info("Downloaded %s samples; keeping all (target was a minimum of %s)", len(all_records), total_target)

    elapsed = time.time() - started
    summary = write_manifests(out_path, all_records, elapsed)
    summary["failures"] = failures
    with open(out_path / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info(
        "Curation complete: %s samples (train=%s val=%s test=%s) from %s sources",
        summary["total_samples"],
        summary["splits"]["train"],
        summary["splits"]["val"],
        summary["splits"]["test"],
        len(summary["sources"]),
    )
    if failures:
        logger.warning("Partial failures: %s", failures)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download public handwriting reference datasets")
    parser.add_argument("--output-dir", default="data/reference_handwriting")
    parser.add_argument("--vocab-dir", default="data/reference_handwriting/vocabularies")
    parser.add_argument("--total-target", type=int, default=50500, help="Minimum desired count; extra samples are kept")
    parser.add_argument("--workers", type=int, default=20, help="Unused; kept for CLI compatibility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sources", default="", help="Comma-separated source_key filter")
    parser.add_argument("--max-per-source", type=int, default=None)
    parser.add_argument("--keep-synthetic", action="store_true", help="Do not delete previously generated *_syn.png images")
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Rebuild manifests from images already on disk; do not download missing files",
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_HF_CACHE))
    args = parser.parse_args()

    keys = [part.strip() for part in args.sources.split(",") if part.strip()] or None
    curate_multisource_dataset(
        output_dir=args.output_dir,
        vocab_dir=args.vocab_dir,
        total_target=args.total_target,
        max_workers=args.workers,
        seed=args.seed,
        source_keys=keys,
        max_per_source=args.max_per_source,
        replace_synthetic=not args.keep_synthetic,
        cache_dir=args.cache_dir,
        existing_only=args.existing_only,
    )

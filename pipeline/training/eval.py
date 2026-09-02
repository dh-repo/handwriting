"""Frozen test eval: JSON report + worst-N TSV. Headline CER is case-sensitive."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import transformers

from pipeline.training.data import parse_labels_tsv
from pipeline.training.infer import recognize_lines
from pipeline.training.metrics import cased_cer, htr_metric_bundle
from pipeline.training.model_contract import (
    apply_trocr_generation_config,
    load_htr_model,
    load_htr_processor,
    resolve_htr_device,
)


REPORT_FIELDS = (
    "split_name",
    "checkpoint",
    "checkpoint_hash",
    "cer",
    "wer",
    "uncased_cer",
    "exact_match",
    "num_beams",
    "date",
    "torch_version",
    "transformers_version",
)


def checkpoint_hash_for(path: str | Path) -> str:
    p = Path(path)
    if p.is_file():
        data = p.read_bytes()
    elif p.is_dir():
        pieces = []
        for child in sorted(p.rglob("*")):
            if child.is_file():
                pieces.append(child.name.encode())
                pieces.append(str(child.stat().st_size).encode())
        data = b"".join(pieces) or b"empty"
    else:
        data = str(path).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def write_test_report(
    dest: str | Path,
    *,
    split_name: str,
    checkpoint: str,
    checkpoint_hash: str,
    cer: float,
    wer: float,
    uncased_cer: float,
    exact_match: float,
    beams: int,
    torch_version: str,
    transformers_version: str,
    date: str | None = None,
) -> Path:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_name": split_name,
        "checkpoint": checkpoint,
        "checkpoint_hash": checkpoint_hash,
        "cer": cer,
        "wer": wer,
        "uncased_cer": uncased_cer,
        "exact_match": exact_match,
        "num_beams": beams,
        "date": date or datetime.now(timezone.utc).date().isoformat(),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "claim": "Teklia-pack CER, not paper IAM 3.42%",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_worst_tsv(
    dest: str | Path,
    rows: Sequence[Mapping[str, Any]],
    n: int = 100,
) -> Path:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: float(r.get("cer", 0.0)), reverse=True)[:n]
    lines = ["image_path\treference\thypothesis\tcer"]
    for row in ranked:
        lines.append(
            f"{row.get('image_path', '')}\t{row.get('reference', '')}\t{row.get('hypothesis', '')}\t{row.get('cer', '')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def score_pairs(
    references: Sequence[str],
    hypotheses: Sequence[str],
    image_paths: Iterable[str] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    bundle = htr_metric_bundle(references, hypotheses)
    paths = list(image_paths) if image_paths is not None else [""] * len(references)
    rows = []
    for ref, hyp, img in zip(references, hypotheses, paths):
        rows.append(
            {
                "image_path": img,
                "reference": ref,
                "hypothesis": hyp,
                "cer": cased_cer([ref], [hyp]),
            }
        )
    return bundle, rows


def evaluate_checkpoint(
    *,
    split_dir: str | Path,
    checkpoint: str,
    output_dir: str | Path,
    beams: int = 4,
    split_name: str = "Teklia/IAM-line test",
    batch_size: int = 8,
    max_lines: int | None = None,
) -> dict[str, float]:
    device = resolve_htr_device()
    processor = load_htr_processor(checkpoint)
    model = load_htr_model(checkpoint)
    apply_trocr_generation_config(model, processor, max_length=128, num_beams=beams)
    model.to(device)
    model.eval()
    print(f"eval device={device} checkpoint={checkpoint} split={split_dir}", flush=True)

    rows = parse_labels_tsv(split_dir)
    if max_lines is not None:
        rows = rows[: max(0, int(max_lines))]
    refs, hyps, paths = [], [], []
    total = len(rows)
    step = max(1, int(batch_size))
    for start in range(0, total, step):
        chunk = rows[start : start + step]
        batch_hyps = recognize_lines(
            model,
            processor,
            [row["image_path"] for row in chunk],
            num_beams=beams,
            device=device,
            batch_size=len(chunk),
        )
        refs.extend(row["text"] for row in chunk)
        hyps.extend(batch_hyps)
        paths.extend(str(row["image_path"]) for row in chunk)
        done = min(start + len(chunk), total)
        if done % 50 < step or done == total:
            print(f"eval {done}/{total}", flush=True)
    bundle, scored = score_pairs(refs, hyps, paths)
    out = Path(output_dir)
    write_test_report(
        out / "test_report.json",
        split_name=split_name,
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_hash_for(checkpoint),
        cer=bundle["cer"],
        wer=bundle["wer"],
        uncased_cer=bundle["uncased_cer"],
        exact_match=bundle["exact_match"],
        beams=beams,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )
    write_worst_tsv(out / "worst100.tsv", scored, n=100)
    print(bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen test eval on a labels.tsv split")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/base_iam_v1")
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--split-name", default="Teklia/IAM-line test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-lines", type=int, default=None)
    args = parser.parse_args()
    evaluate_checkpoint(
        split_dir=args.split_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        beams=args.beams,
        split_name=args.split_name,
        batch_size=args.batch_size,
        max_lines=args.max_lines,
    )


if __name__ == "__main__":
    main()


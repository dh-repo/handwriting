"""
pipeline/evaluation/evaluate.py
Standalone CLI & programmatic evaluation engine for TrOCR / VisionEncoderDecoder checkpoints.
Evaluates model checkpoints on held-out test splits or manifests,
computing CER, WER, error breakdowns, throughput, and latency percentiles.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
import time

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from pipeline.dataset.dataset_loader import (
    HandwritingSample,
    IAMDatasetParser,
    MedicalPrescriptionDatasetLoader,
    MedicalPrescriptionSample,
)
from pipeline.evaluation.metrics import (
    ErrorBreakdown,
    LatencyMetrics,
    LatencyTracker,
    MetricResult,
    NormalizationConfig,
    ThroughputMetrics,
    compute_cer,
    compute_metrics,
    compute_wer,
    normalize_text,
)
from pipeline.training.config import TrainingConfig
from pipeline.training.dataset import OCRDataCollator, OCRDataset, create_dummy_processor, load_trocr_processor
from pipeline.training.train import create_tiny_mock_model, get_optimal_device

logger = logging.getLogger("pipeline.evaluation.evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class EvaluationReport:
    """Structured report containing metrics, latency, throughput, and sample predictions."""
    metrics: MetricResult = field(default_factory=MetricResult)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    sample_predictions: List[Dict[str, Any]] = field(default_factory=list)
    device: str = "cpu"
    checkpoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        return {
            "checkpoint": self.checkpoint,
            "device": self.device,
            "mean_cer": self.metrics.normalized_cer,
            "mean_wer": self.metrics.normalized_wer,
            "raw_cer": self.metrics.cer,
            "raw_wer": self.metrics.wer,
            "sample_count": self.metrics.sample_count,
            "char_breakdown": self.metrics.char_breakdown.to_dict(),
            "word_breakdown": self.metrics.word_breakdown.to_dict(),
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
            "p50_latency_ms": self.latency.p50_ms,
            "throughput_samples_per_sec": self.throughput.samples_per_second,
            "sample_predictions": self.sample_predictions,
        }

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serialize report to JSON string or file."""
        s = json.dumps(self.to_dict(), indent=indent)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s, encoding="utf-8")
        return s

    def summary_table(self) -> str:
        """Format human-readable ASCII summary table."""
        m = self.metrics
        lat = self.latency
        tp = self.throughput
        cb = m.char_breakdown
        wb = m.word_breakdown

        lines = [
            "=" * 80,
            "                    Handwriting Recognition Evaluation Report",
            "=" * 80,
            f"Model Checkpoint : {self.checkpoint or 'In-Memory Model'}",
            f"Target Device    : {self.device}",
            f"Evaluated Split  : test (N = {m.sample_count} samples)",
            "-" * 80,
            f"{'METRIC':<24} {'RAW':<22} {'NORMALIZED':<22}",
            f"{'Character Error Rate':<24} {m.cer:<22.4f} {m.normalized_cer:<22.4f}",
            f"{'Word Error Rate':<24} {m.wer:<22.4f} {m.normalized_wer:<22.4f}",
            "-" * 80,
            f"{'ERROR BREAKDOWN':<24} {'CHARACTERS':<22} {'WORDS':<22}",
            f"{'Substitutions':<24} {cb.substitutions:<22} {wb.substitutions:<22}",
            f"{'Deletions':<24} {cb.deletions:<22} {wb.deletions:<22}",
            f"{'Insertions':<24} {cb.insertions:<22} {wb.insertions:<22}",
            f"{'Correct Hits':<24} {cb.hits:<22} {wb.hits:<22}",
            f"{'Total Reference Units':<24} {cb.total_reference_units:<22} {wb.total_reference_units:<22}",
            "-" * 80,
            "THROUGHPUT & LATENCY PERFORMANCE",
            f"Throughput               {tp.samples_per_second:.1f} samples/sec ({tp.characters_per_second:.1f} chars/sec)",
            f"Total Time Elapsed       {tp.total_time_seconds:.2f} seconds",
            f"Latency (Median p50)     {lat.p50_ms:.1f} ms / sample",
            f"Latency (p90)            {lat.p90_ms:.1f} ms / sample",
            f"Latency (p95)            {lat.p95_ms:.1f} ms / sample",
            f"Latency (p99)            {lat.p99_ms:.1f} ms / sample",
            f"Latency (Mean ± Std)     {lat.mean_ms:.1f} ms ± {lat.std_ms:.1f} ms",
            "=" * 80,
        ]
        return "\n".join(lines)


def load_model_and_processor(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[VisionEncoderDecoderModel, Any]:
    """
    Load VisionEncoderDecoderModel and TrOCRProcessor / AutoProcessor.
    Supports directory format (save_pretrained) and standalone .pt state dicts.
    """
    dev = device or get_optimal_device("auto")
    ckpt = Path(checkpoint_path)

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")

    if ckpt.is_dir():
        # Check for config.json or pytorch_model.bin / model.safetensors
        try:
            model = VisionEncoderDecoderModel.from_pretrained(str(ckpt))
        except Exception as e:
            logger.warning(f"VisionEncoderDecoderModel.from_pretrained failed ({e}). Creating tiny config.")
            model = create_tiny_mock_model()

        try:
            processor = load_trocr_processor(str(ckpt))
        except Exception:
            processor = create_dummy_processor()

    elif ckpt.is_file() and ckpt.suffix in (".pt", ".pth", ".bin"):
        # Load state dict
        state = torch.load(str(ckpt), map_location="cpu")
        model_state = state.get("model_state_dict", state) if isinstance(state, dict) else state

        # Determine architecture from config if saved
        if isinstance(state, dict) and "config" in state and isinstance(state["config"], dict):
            m_name = state["config"].get("model_name_or_path", "microsoft/trocr-small-handwritten")
        else:
            m_name = "microsoft/trocr-small-handwritten"

        try:
            model = VisionEncoderDecoderModel.from_pretrained(m_name)
            model.load_state_dict(model_state)
            processor = load_trocr_processor(m_name)
        except Exception as e:
            logger.warning(f"Could not load base model for state dict ({e}). Initializing mock model.")
            model = create_tiny_mock_model()
            processor = create_dummy_processor()
            try:
                model.load_state_dict(model_state, strict=False)
            except Exception:
                pass

    for attr in ["max_length", "early_stopping", "no_repeat_ngram_size", "length_penalty", "num_beams"]:
        if hasattr(model.config, attr):
            try:
                delattr(model.config, attr)
            except Exception:
                pass

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.decoder_start_token_id = 0
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = 1
        model.generation_config.max_new_tokens = 64
        model.generation_config.no_repeat_ngram_size = 3

    model.to(dev)
    model.eval()
    return model, processor


def evaluate_model(
    model: VisionEncoderDecoderModel,
    processor: Any,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    normalization_config: Optional[NormalizationConfig] = None,
    max_new_tokens: int = 64,
    num_beams: int = 1,
    rescorer_fn: Optional[Callable[[List[str], List[float]], Tuple[str, float]]] = None,
    checkpoint_path: str = "",
) -> EvaluationReport:
    """
    Evaluate a loaded VisionEncoderDecoderModel over a PyTorch DataLoader.
    Computes exact CER, WER, error breakdown, and latency/throughput telemetry.
    Supports multi-beam generation and optional beam rescoring hook.
    """
    dev = device or next(model.parameters()).device
    norm_cfg = normalization_config or NormalizationConfig()
    tracker = LatencyTracker()

    references: List[str] = []
    hypotheses: List[str] = []
    sample_records: List[Dict[str, Any]] = []

    is_mps = dev.type == "mps"
    if is_mps and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    model.eval()
    tracker.start()

    with torch.inference_mode():
        for batch in dataloader:
            if "pixel_values" not in batch:
                continue

            pixel_values = batch["pixel_values"].to(dev)
            batch_texts = batch.get("texts", [])
            sample_ids = batch.get("sample_ids", [f"s_{i}" for i in range(len(pixel_values))])
            batch_size = pixel_values.shape[0]

            t0 = time.perf_counter()
            if num_beams > 1 and rescorer_fn is not None:
                # Multi-beam generation with rescoring hook
                try:
                    gen_out = model.generate(
                        pixel_values=pixel_values,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
                except Exception:
                    sequences = model.generate(
                        pixel_values=pixel_values,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
            else:
                sequences = model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )

            if is_mps and hasattr(torch, "mps"):
                torch.mps.synchronize()
            t1 = time.perf_counter()

            batch_duration = max(1e-6, t1 - t0)

            # Decode sequences
            if hasattr(processor, "batch_decode"):
                all_decoded = processor.batch_decode(sequences, skip_special_tokens=True)
            elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "batch_decode"):
                all_decoded = processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            else:
                all_decoded = [f"pred_{i}" for i in range(len(sequences))]

            # Process candidates per item in batch
            pred_texts: List[str] = []
            if num_beams > 1 and rescorer_fn is not None and len(all_decoded) == batch_size * num_beams:
                for b_idx in range(batch_size):
                    cands = all_decoded[b_idx * num_beams : (b_idx + 1) * num_beams]
                    dummy_scores = [0.0] * len(cands)
                    try:
                        best_text, _ = rescorer_fn(cands, dummy_scores)
                        pred_texts.append(best_text)
                    except Exception:
                        pred_texts.append(cands[0] if cands else "")
            elif len(all_decoded) == batch_size:
                pred_texts = all_decoded
            else:
                # If num_return_sequences produced extra candidates without rescorer, take top candidate
                pred_texts = [all_decoded[i] for i in range(0, len(all_decoded), max(1, len(all_decoded) // batch_size))]

            batch_char_count = sum(len(t) for t in pred_texts)
            batch_word_count = sum(len(t.split()) for t in pred_texts)
            tracker.record_batch(batch_size, batch_char_count, batch_word_count, batch_duration)

            for s_id, ref, hyp in zip(sample_ids, batch_texts, pred_texts):
                ref_clean = str(ref).strip()
                hyp_clean = str(hyp).strip()
                references.append(ref_clean)
                hypotheses.append(hyp_clean)

                s_cer = compute_cer(ref_clean, hyp_clean, norm_cfg)
                s_wer = compute_wer(ref_clean, hyp_clean, norm_cfg)
                sample_records.append({
                    "sample_id": s_id,
                    "reference": ref_clean,
                    "hypothesis": hyp_clean,
                    "cer": round(s_cer, 4),
                    "wer": round(s_wer, 4),
                })

    tracker.stop()

    metrics = compute_metrics(references, hypotheses, norm_cfg)
    lat_metrics, tp_metrics = tracker.compute_metrics()

    return EvaluationReport(
        metrics=metrics,
        latency=lat_metrics,
        throughput=tp_metrics,
        sample_predictions=sample_records,
        device=str(dev),
        checkpoint=checkpoint_path,
    )


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path],
    dataset: Optional[Dataset] = None,
    manifest_path: Optional[str] = None,
    iam_lines_path: Optional[str] = None,
    iam_root_dir: Optional[str] = None,
    split: str = "test",
    device: str = "auto",
    batch_size: int = 16,
    num_workers: int = 0,
    normalization_config: Optional[NormalizationConfig] = None,
    max_samples: Optional[int] = None,
) -> EvaluationReport:
    """
    High-level programmatic interface to evaluate a model checkpoint on a target dataset or manifest.
    """
    dev = get_optimal_device(device)
    model, processor = load_model_and_processor(checkpoint_path, dev)

    if dataset is not None:
        eval_ds = dataset
    elif iam_lines_path and os.path.exists(iam_lines_path):
        eval_ds = OCRDataset.from_iam(iam_lines_path, iam_root_dir, split=split, processor=processor, is_training=False)
    elif manifest_path and os.path.exists(manifest_path):
        eval_ds = OCRDataset.from_manifest(manifest_path, processor=processor, is_training=False)
    else:
        raise FileNotFoundError(
            "Evaluation requires --manifest or --iam-lines pointing at real handwriting data. "
            "Synthetic evaluation fallbacks are disabled."
        )

    if max_samples is not None and len(eval_ds) > max_samples:
        eval_ds.samples = eval_ds.samples[:max_samples]

    collator = OCRDataCollator(processor=processor)
    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    return evaluate_model(
        model=model,
        processor=processor,
        dataloader=loader,
        device=dev,
        normalization_config=normalization_config,
        checkpoint_path=str(checkpoint_path),
    )


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Automated CER/WER Evaluation and Throughput Profiler")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir or .pt state dict")
    parser.add_argument("--manifest", type=str, default=None, help="Path to JSON/CSV prescription manifest")
    parser.add_argument("--iam-lines", type=str, default=None, help="Path to IAM ascii/lines.txt")
    parser.add_argument("--iam-root", type=str, default=None, help="Path to IAM root directory")
    parser.add_argument("--synthetic-count", type=int, default=None, help="Deprecated alias for --max-samples")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (test, val, train)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, mps, cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit total samples evaluated")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max generated tokens per line")
    parser.add_argument("--output-json", type=str, default=None, help="Path to export JSON evaluation report")
    parser.add_argument("--save-predictions", type=str, default=None, help="Path to save predictions JSON")
    parser.add_argument("--case-sensitive", action="store_true", help="Preserve case during normalization")
    parser.add_argument("--strip-punctuation", action="store_true", help="Strip punctuation during normalization")
    parser.add_argument("--quiet", action="store_true", help="Suppress console summary table")
    args = parser.parse_args()

    norm_cfg = NormalizationConfig(
        lowercase=not args.case_sensitive,
        strip_punctuation=args.strip_punctuation,
    )

    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        iam_lines_path=args.iam_lines,
        iam_root_dir=args.iam_root,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalization_config=norm_cfg,
        max_samples=args.max_samples or args.synthetic_count,
    )

    if not args.quiet:
        print(report.summary_table())

    if args.output_json:
        report.to_json(args.output_json)
        logger.info(f"Saved evaluation report to {args.output_json}")

    if args.save_predictions:
        p = Path(args.save_predictions)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report.sample_predictions, indent=2), encoding="utf-8")
        logger.info(f"Saved predictions to {args.save_predictions}")


if __name__ == "__main__":
    main()

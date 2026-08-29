#!/usr/bin/env python3
"""
pipeline/training/benchmark_throughput.py
Standalone CLI benchmark tool comparing baseline vs accelerated pipeline throughput.
Evaluates end-to-end samples/sec throughput, latency breakdowns, memory stability,
and dataloader I/O wait elimination.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is in sys.path when executed directly
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pipeline.training.config import TrainingConfig, get_default_num_workers
from pipeline.training.dataset import OCRDataCollator, OCRDataset, create_dummy_processor
from pipeline.training.prefetcher import AsyncDevicePrefetcher
from pipeline.training.profiler import StepTiming, TrainingStepProfiler
from pipeline.training.train import (
    create_tiny_mock_model,
    get_autocast_context,
    get_optimal_device,
    load_trocr_model,
)

logger = logging.getLogger("pipeline.training.benchmark_throughput")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SyntheticBenchmarkDataset(Dataset):
    """Fast, in-memory synthetic dataset for training throughput benchmarking."""

    def __init__(
        self,
        num_samples: int = 256,
        image_shape: Tuple[int, int, int] = (3, 64, 64),
        seq_length: int = 16,
    ) -> None:
        self.num_samples = max(1, num_samples)
        self.image_shape = image_shape
        self.seq_length = seq_length
        # Pre-allocate random tensor data
        self.images = torch.randn(self.num_samples, *image_shape)
        self.labels = torch.randint(1, 100, (self.num_samples, seq_length), dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "pixel_values": self.images[idx],
            "labels": self.labels[idx],
        }


def run_single_benchmark_mode(
    mode_name: str,
    device: torch.device,
    batch_size: int = 8,
    num_steps: int = 50,
    num_samples: int = 256,
    mixed_precision: str = "fp16",
    attn_implementation: str = "sdpa",
    num_workers: int = 4,
    use_prefetcher: bool = True,
    empty_cache_steps: int = 100,
    warmup_steps: int = 5,
    image_size: int = 64,
    custom_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Execute training loop under specified acceleration configuration and measure throughput.
    """
    dataset = SyntheticBenchmarkDataset(
        num_samples=max(num_samples, (num_steps + warmup_steps) * batch_size),
        image_shape=(3, image_size, image_size),
    )

    # Configure DataLoader
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(dataset, **loader_kwargs)

    # Initialize Model
    if custom_model is not None:
        model = custom_model.to(device)
    else:
        try:
            model = create_tiny_mock_model(
                vocab_size=256,
                image_size=image_size,
                attn_implementation=attn_implementation,
            ).to(device)
        except Exception:
            class MiniTrOCR(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv = nn.Conv2d(3, 16, kernel_size=4, stride=4)
                    self.head = nn.Linear(16 * (image_size // 4) * (image_size // 4), 256)
                    self.loss_fn = nn.CrossEntropyLoss()

                def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Any:
                    feat = self.conv(pixel_values)
                    logits = self.head(feat.flatten(1))
                    loss = None
                    if labels is not None:
                        loss = self.loss_fn(logits, labels[:, 0] % 256)
                    return type("Output", (), {"loss": loss, "logits": logits})()

            model = MiniTrOCR().to(device)

    model.train()

    # Configure SDPA attention flag if present
    if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = attn_implementation

    # Configure Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler_enabled = (mixed_precision.lower() in ("fp16", "bf16")) and device.type in ("mps", "cuda")
    scaler = (
        torch.amp.GradScaler(device.type, enabled=scaler_enabled)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and device.type in ("mps", "cuda")
        else None
    )

    autocast_ctx = get_autocast_context(device, mixed_precision=mixed_precision)
    profiler = TrainingStepProfiler(warmup_steps=warmup_steps, device=device, enabled=True)

    if use_prefetcher and num_workers > 0:
        data_stream = AsyncDevicePrefetcher(
            loader,
            device=device,
            mixed_precision=mixed_precision,
            queue_size=3,
        )
    else:
        data_stream = loader

    total_executed_steps = 0
    t_data_start = time.perf_counter()

    try:
        for batch in data_stream:
            if total_executed_steps >= (num_steps + warmup_steps):
                break

            t_data = time.perf_counter() - t_data_start
            profiler.record_data_wait(t_data)

            pixel_values = batch["pixel_values"]
            labels = batch["labels"]

            t_trans_start = time.perf_counter()
            if not use_prefetcher or pixel_values.device != device:
                pixel_values = pixel_values.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            t_transfer = time.perf_counter() - t_trans_start

            optimizer.zero_grad(set_to_none=True)

            t_fwd_start = time.perf_counter()
            with autocast_ctx:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            t_fwd = time.perf_counter() - t_fwd_start

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                t_data_start = time.perf_counter()
                continue

            t_bwd_start = time.perf_counter()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            t_bwd = time.perf_counter() - t_bwd_start

            t_opt_start = time.perf_counter()
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            t_opt = time.perf_counter() - t_opt_start

            total_executed_steps += 1
            profiler.record_step(
                t_data=t_data,
                t_transfer=t_transfer,
                t_fwd=t_fwd,
                t_bwd=t_bwd,
                t_opt=t_opt,
                num_samples=len(pixel_values),
            )

            if empty_cache_steps > 0 and total_executed_steps % empty_cache_steps == 0:
                if device.type == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                elif device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            t_data_start = time.perf_counter()
    finally:
        if use_prefetcher and hasattr(data_stream, "close"):
            data_stream.close()

    summary = profiler.get_summary()
    summary["mode"] = mode_name
    summary["device"] = str(device)
    summary["attn_implementation"] = attn_implementation
    summary["mixed_precision"] = mixed_precision
    summary["num_workers"] = num_workers
    summary["batch_size"] = batch_size
    return summary


def run_benchmark(
    mode: str = "both",
    batch_size: int = 8,
    num_steps: int = 50,
    num_samples: Optional[int] = None,
    num_workers: Optional[int] = None,
    mixed_precision: str = "fp16",
    attn_implementation: str = "sdpa",
    device: str = "auto",
    output_json: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = None,
    warmup_steps: int = 5,
    image_size: int = 64,
    custom_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Run baseline vs accelerated comparison benchmark and compute comparative telemetry.
    """
    dev = get_optimal_device(device)
    samples_count = num_samples or ((num_steps + warmup_steps) * batch_size * 2)
    workers = num_workers if num_workers is not None else get_default_num_workers()

    results: Dict[str, Any] = {
        "device": str(dev),
        "batch_size": batch_size,
        "num_steps": num_steps,
    }

    baseline_summary: Optional[Dict[str, Any]] = None
    accelerated_summary: Optional[Dict[str, Any]] = None

    if mode in ("baseline", "both"):
        logger.info("Running BASELINE benchmark (Eager attention, 0 workers, FP32, synchronous cache empty)...")
        baseline_summary = run_single_benchmark_mode(
            mode_name="baseline",
            device=dev,
            batch_size=batch_size,
            num_steps=num_steps,
            num_samples=samples_count,
            mixed_precision="none",
            attn_implementation="eager",
            num_workers=0,
            use_prefetcher=False,
            empty_cache_steps=1,
            warmup_steps=warmup_steps,
            image_size=image_size,
            custom_model=custom_model,
        )
        results["baseline"] = baseline_summary
        results["baseline_samples_per_sec"] = baseline_summary["samples_per_sec"]

    if mode in ("accelerated", "both"):
        logger.info(f"Running ACCELERATED benchmark (SDPA attention, {workers} workers, {mixed_precision}, AsyncPrefetcher)...")
        accelerated_summary = run_single_benchmark_mode(
            mode_name="accelerated",
            device=dev,
            batch_size=batch_size,
            num_steps=num_steps,
            num_samples=samples_count,
            mixed_precision=mixed_precision,
            attn_implementation=attn_implementation,
            num_workers=workers,
            use_prefetcher=True,
            empty_cache_steps=100,
            warmup_steps=warmup_steps,
            image_size=image_size,
            custom_model=custom_model,
        )
        results["accelerated"] = accelerated_summary
        results["accelerated_samples_per_sec"] = accelerated_summary["samples_per_sec"]
        results["p50_latency_ms"] = accelerated_summary["step_latency_ms"]["p50"]

    # Comparative speedup calculations
    if baseline_summary and accelerated_summary:
        base_sps = baseline_summary["samples_per_sec"]
        acc_sps = accelerated_summary["samples_per_sec"]
        speedup = (acc_sps / base_sps) if base_sps > 1e-6 else 1.0
        results["speedup_ratio"] = round(speedup, 2)

        base_lat = baseline_summary["step_latency_ms"]["mean"]
        acc_lat = accelerated_summary["step_latency_ms"]["mean"]
        latency_reduction = ((base_lat - acc_lat) / base_lat * 100.0) if base_lat > 1e-6 else 0.0
        results["latency_reduction_pct"] = round(latency_reduction, 2)

        base_data_wait = baseline_summary["mean_data_wait_sec"] * 1000.0
        acc_data_wait = accelerated_summary["mean_data_wait_sec"] * 1000.0
        wait_elimination = ((base_data_wait - acc_data_wait) / base_data_wait * 100.0) if base_data_wait > 1e-6 else 100.0
        results["data_wait_elimination_pct"] = round(wait_elimination, 2)

    # Format output telemetry table
    table_str = format_benchmark_table(results)
    results["formatted_table"] = table_str
    print("\n" + table_str + "\n")

    if output_json:
        p_json = Path(output_json)
        p_json.parent.mkdir(parents=True, exist_ok=True)
        p_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(f"Exported benchmark JSON results to: {output_json}")

    if output_csv and "accelerated" in results:
        p_csv = Path(output_csv)
        p_csv.parent.mkdir(parents=True, exist_ok=True)
        lines = ["metric,baseline,accelerated,speedup\n"]
        if baseline_summary and accelerated_summary:
            lines.append(f"samples_per_sec,{baseline_summary['samples_per_sec']:.2f},{accelerated_summary['samples_per_sec']:.2f},{results.get('speedup_ratio', 1.0):.2f}\n")
            lines.append(f"mean_step_latency_ms,{baseline_summary['step_latency_ms']['mean']:.2f},{accelerated_summary['step_latency_ms']['mean']:.2f},{results.get('latency_reduction_pct', 0.0):.2f}%\n")
            lines.append(f"p50_latency_ms,{baseline_summary['step_latency_ms']['p50']:.2f},{accelerated_summary['step_latency_ms']['p50']:.2f},-\n")
            lines.append(f"data_wait_ms,{baseline_summary['mean_data_wait_sec']*1000.0:.2f},{accelerated_summary['mean_data_wait_sec']*1000.0:.2f},{results.get('data_wait_elimination_pct', 0.0):.2f}%\n")
        p_csv.write_text("".join(lines), encoding="utf-8")
        logger.info(f"Exported benchmark CSV results to: {output_csv}")

    return results


def format_benchmark_table(results: Dict[str, Any]) -> str:
    """Format comparative benchmark telemetry table."""
    base = results.get("baseline")
    acc = results.get("accelerated")
    device = results.get("device", "unknown")
    batch_size = results.get("batch_size", 8)

    lines = [
        "=" * 88,
        "                    TrOCR TRAINING ACCELERATION THROUGHPUT BENCHMARK",
        "=" * 88,
        f" Device: {device} | Micro-Batch Size: {batch_size}",
        "-" * 88,
        f" {'Metric':<30} {'Baseline':>16} {'Accelerated':>16} {'Speedup / Delta':>18}",
        "-" * 88,
    ]

    if base and acc:
        base_sps = base["samples_per_sec"]
        acc_sps = acc["samples_per_sec"]
        speedup = results.get("speedup_ratio", 1.0)
        lines.append(f" {'Throughput (samples/sec)':<30} {base_sps:>16.2f} {acc_sps:>16.2f} {f'{speedup:.2f}x':>18}")

        base_lat = base["step_latency_ms"]["mean"]
        acc_lat = acc["step_latency_ms"]["mean"]
        lat_pct = results.get("latency_reduction_pct", 0.0)
        lat_delta_str = f"-{lat_pct:.1f}%" if lat_pct >= 0 else f"+{abs(lat_pct):.1f}%"
        lines.append(f" {'Mean Step Latency (ms)':<30} {base_lat:>13.2f} ms {acc_lat:>13.2f} ms {lat_delta_str:>18}")

        lines.append(f" {'p50 Step Latency (ms)':<30} {base['step_latency_ms']['p50']:>13.2f} ms {acc['step_latency_ms']['p50']:>13.2f} ms {'-':>18}")
        lines.append(f" {'p90 Step Latency (ms)':<30} {base['step_latency_ms']['p90']:>13.2f} ms {acc['step_latency_ms']['p90']:>13.2f} ms {'-':>18}")
        lines.append(f" {'p99 Step Latency (ms)':<30} {base['step_latency_ms']['p99']:>13.2f} ms {acc['step_latency_ms']['p99']:>13.2f} ms {'-':>18}")

        b_wait = base["mean_data_wait_sec"] * 1000.0
        a_wait = acc["mean_data_wait_sec"] * 1000.0
        wait_elim = results.get("data_wait_elimination_pct", 0.0)
        wait_delta_str = f"-{wait_elim:.1f}%" if wait_elim >= 0 else f"+{abs(wait_elim):.1f}%"
        lines.append(f" {'Data Wait Latency (ms)':<30} {b_wait:>13.2f} ms {a_wait:>13.2f} ms {wait_delta_str:>18}")

        lines.append("-" * 88)
        lines.append(f" {'Attention Kernel':<30} {'eager':>16} {'sdpa':>16} {'Kernel Fusion':>18}")
        lines.append(f" {'Mixed Precision':<30} {'none (FP32)':>16} {'fp16 (AMP)':>16} {'Tensor Core':>18}")
        lines.append(f" {'Device Prefetcher':<30} {'Disabled':>16} {'Async Prefetch':>16} {'Zero IO Wait':>18}")
        lines.append(f" {'Worker Multiprocessing':<30} {'0 (Sync)':>16} {str(acc.get('num_workers', 4)):>16} {'Multi-Process':>18}")
    elif acc:
        acc_sps = acc["samples_per_sec"]
        lines.append(f" {'Throughput (samples/sec)':<30} {'N/A':>16} {acc_sps:>16.2f} {'-':>18}")
        lines.append(f" {'Mean Step Latency (ms)':<30} {'N/A':>16} {acc['step_latency_ms']['mean']:>13.2f} ms {'-':>18}")
        lines.append(f" {'p50 Step Latency (ms)':<30} {'N/A':>16} {acc['step_latency_ms']['p50']:>13.2f} ms {'-':>18}")
    elif base:
        base_sps = base["samples_per_sec"]
        lines.append(f" {'Throughput (samples/sec)':<30} {base_sps:>16.2f} {'N/A':>16} {'-':>18}")
        lines.append(f" {'Mean Step Latency (ms)':<30} {base['step_latency_ms']['mean']:>13.2f} ms {'N/A':>16} {'-':>18}")

    lines.append("=" * 88)
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for standalone throughput benchmarking."""
    parser = argparse.ArgumentParser(description="TrOCR Training Acceleration Throughput Benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "accelerated", "both"],
        default="both",
        help="Benchmark execution mode (default: both)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of benchmark steps (default: 50)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples (optional)")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers (optional)")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="fp16",
        choices=["none", "fp16", "bf16", "fp32"],
        help="Mixed precision mode (default: fp16)",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="Attention kernel implementation (default: sdpa)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'mps', 'cuda', 'cpu')")
    parser.add_argument("--baseline", action="store_true", help="Shortcut to run baseline mode")
    parser.add_argument("--output-json", type=str, default=None, help="Path to export results JSON")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to export results CSV")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warmup steps (default: 5)")
    parser.add_argument("--image-size", type=int, default=64, help="Benchmark synthetic image dimension (default: 64)")

    args = parser.parse_args()

    mode = "baseline" if args.baseline else args.mode

    run_benchmark(
        mode=mode,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        attn_implementation=args.attn_implementation,
        device=args.device,
        output_json=args.output_json,
        output_csv=args.output_csv,
        warmup_steps=args.warmup_steps,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()

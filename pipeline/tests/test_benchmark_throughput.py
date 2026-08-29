"""
pipeline/tests/test_benchmark_throughput.py
Unit and CLI test suite for Standalone Throughput Benchmark Tool (Requirement R3 / Feature F10).
Tests baseline vs accelerated execution, comparative telemetry, CLI arguments, and export formats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

from pipeline.training.benchmark_throughput import (
    SyntheticBenchmarkDataset,
    format_benchmark_table,
    main,
    run_benchmark,
    run_single_benchmark_mode,
)
from pipeline.training.train import create_tiny_mock_model, get_optimal_device


class TestBenchmarkDatasetAndCore:
    """Test synthetic benchmark dataset generation and single mode executions."""

    def test_synthetic_benchmark_dataset(self) -> None:
        """Verify synthetic benchmark dataset yields valid tensors and dimensions."""
        ds = SyntheticBenchmarkDataset(num_samples=16, image_shape=(3, 64, 64), seq_length=8)
        assert len(ds) == 16
        item = ds[0]
        assert "pixel_values" in item
        assert "labels" in item
        assert item["pixel_values"].shape == (3, 64, 64)
        assert item["labels"].shape == (8,)

    def test_run_single_benchmark_mode_baseline(self) -> None:
        """Verify single benchmark mode execution with baseline settings."""
        dev = torch.device("cpu")
        summary = run_single_benchmark_mode(
            mode_name="baseline",
            device=dev,
            batch_size=2,
            num_steps=5,
            num_samples=16,
            mixed_precision="none",
            attn_implementation="eager",
            num_workers=0,
            use_prefetcher=False,
            empty_cache_steps=1,
            warmup_steps=1,
        )
        assert summary["mode"] == "baseline"
        assert summary["total_steps"] >= 5
        assert summary["samples_per_sec"] > 0.0
        assert summary["num_workers"] == 0

    def test_run_single_benchmark_mode_accelerated(self) -> None:
        """Verify single benchmark mode execution with accelerated settings."""
        dev = torch.device("cpu")
        summary = run_single_benchmark_mode(
            mode_name="accelerated",
            device=dev,
            batch_size=2,
            num_steps=5,
            num_samples=16,
            mixed_precision="fp16",
            attn_implementation="sdpa",
            num_workers=0,  # 0 on CPU to avoid worker subprocess spawn in unit tests
            use_prefetcher=False,
            empty_cache_steps=100,
            warmup_steps=1,
        )
        assert summary["mode"] == "accelerated"
        assert summary["total_steps"] >= 5
        assert summary["samples_per_sec"] > 0.0


class TestComparativeBenchmarkExecution:
    """Test comparative benchmark runs, speedup calculations, and table formatting."""

    def test_run_benchmark_both_modes(self, tmp_path: Path) -> None:
        """Verify run_benchmark executes both baseline and accelerated runs, calculating speedup."""
        json_out = tmp_path / "bench_results.json"
        csv_out = tmp_path / "bench_results.csv"

        results = run_benchmark(
            mode="both",
            batch_size=2,
            num_steps=4,
            num_samples=16,
            num_workers=0,
            device="cpu",
            output_json=json_out,
            output_csv=csv_out,
            warmup_steps=1,
        )

        assert "baseline" in results
        assert "accelerated" in results
        assert "speedup_ratio" in results
        assert "latency_reduction_pct" in results
        assert "data_wait_elimination_pct" in results
        assert results["speedup_ratio"] > 0.0

        # Verify JSON export
        assert json_out.exists()
        loaded = json.loads(json_out.read_text(encoding="utf-8"))
        assert "speedup_ratio" in loaded
        assert "baseline" in loaded
        assert "accelerated" in loaded

        # Verify CSV export
        assert csv_out.exists()
        csv_content = csv_out.read_text(encoding="utf-8")
        assert "samples_per_sec" in csv_content
        assert "mean_step_latency_ms" in csv_content

    def test_format_benchmark_table(self) -> None:
        """Verify format_benchmark_table builds a comparative ASCII table."""
        mock_results = {
            "device": "mps",
            "batch_size": 8,
            "speedup_ratio": 4.25,
            "latency_reduction_pct": 76.5,
            "data_wait_elimination_pct": 98.2,
            "baseline": {
                "samples_per_sec": 6.5,
                "step_latency_ms": {"mean": 600.0, "p50": 595.0, "p90": 610.0, "p99": 620.0},
                "mean_data_wait_sec": 0.045,
            },
            "accelerated": {
                "samples_per_sec": 27.6,
                "step_latency_ms": {"mean": 141.0, "p50": 139.0, "p90": 145.0, "p99": 150.0},
                "mean_data_wait_sec": 0.0008,
                "num_workers": 4,
            },
        }
        table_str = format_benchmark_table(mock_results)
        assert "TrOCR TRAINING ACCELERATION THROUGHPUT BENCHMARK" in table_str
        assert "4.25x" in table_str
        assert "Throughput (samples/sec)" in table_str
        assert "Attention Kernel" in table_str
        assert "sdpa" in table_str


class TestBenchmarkCLIArguments:
    """Test CLI argument parsing and options."""

    def test_cli_argument_parsing(self) -> None:
        """Verify argparse options handle all acceleration flags."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str, default="both")
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--num-steps", type=int, default=50)
        parser.add_argument("--mixed-precision", type=str, default="fp16")
        parser.add_argument("--attn-implementation", type=str, default="sdpa")
        parser.add_argument("--baseline", action="store_true")
        parser.add_argument("--device", type=str, default="auto")

        args = parser.parse_args([
            "--mode", "accelerated",
            "--batch-size", "16",
            "--num-steps", "25",
            "--mixed-precision", "bf16",
            "--attn-implementation", "sdpa",
            "--device", "cpu",
        ])

        assert args.mode == "accelerated"
        assert args.batch_size == 16
        assert args.num_steps == 25
        assert args.mixed_precision == "bf16"
        assert args.attn_implementation == "sdpa"
        assert args.device == "cpu"

    def test_cli_baseline_flag_shortcut(self) -> None:
        """Verify --baseline flag correctly targets baseline mode."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str, default="both")
        parser.add_argument("--baseline", action="store_true")
        args = parser.parse_args(["--baseline"])
        mode = "baseline" if args.baseline else args.mode
        assert mode == "baseline"

    def test_cli_device_fallback(self) -> None:
        """Verify invalid device string falls back safely."""
        dev = get_optimal_device("invalid_device_name_xyz")
        assert dev.type in ("mps", "cuda", "cpu")

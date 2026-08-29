"""
tests/unit/test_benchmark_ablation.py
Unit tests for Ablation Benchmark engine, PNDA calculation, report formatting, and CLI routing.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import pytest

from pipeline.evaluation.benchmark_ablation import (
    AblationBenchmarkReport,
    calculate_pnda,
    run_ablation_benchmark,
)


class TestPNDACalculation:
    """Test Pharmaceutical Name Disambiguation Accuracy (PNDA) calculation."""

    def test_pnda_exact_matches(self) -> None:
        refs = ["Amoxicillin 500mg PO TID", "Hydroxyzine 25mg QHS"]
        hyps = ["Amoxicillin 500mg PO TID", "Hydroxyzine 25mg QHS"]
        acc = calculate_pnda(refs, hyps)
        assert acc == 1.0

    def test_pnda_partial_mismatches(self) -> None:
        refs = ["Amoxicillin 500mg PO TID", "Hydroxyzine 25mg QHS"]
        hyps = ["Amoxicillin 500mg PO TID", "Hydralazine 25mg QHS"]
        acc = calculate_pnda(refs, hyps)
        assert acc == 0.5

    def test_pnda_empty_inputs(self) -> None:
        assert calculate_pnda([], []) == 0.0


class TestAblationBenchmarkEngine:
    """Test 4-stage ablation benchmark execution and output formatting."""

    def test_run_ablation_benchmark_from_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "test_manifest.jsonl"
        with open(manifest, "w", encoding="utf-8") as handle:
            for i in range(10):
                handle.write(json.dumps({
                    "sample_id": f"s_{i}",
                    "text": "Amoxicillin 500mg PO TID" if i % 2 == 0 else "Hydroxyzine 25mg QHS",
                    "category": "prescription_item" if i % 2 == 0 else "clinical_note",
                    "is_lasa": i % 2 == 1,
                    "split": "test",
                    "dataset_source": "iam_line",
                }) + "\n")
        report = run_ablation_benchmark(
            manifest_path=manifest,
            split="test",
            beam_width=5,
            max_samples=10,
        )
        assert isinstance(report, AblationBenchmarkReport)
        assert len(report.stages) == 4
        assert "stage_1_baseline" in report.stages
        assert "stage_2_lexicon_only" in report.stages
        assert "stage_3_lexicon_confusion" in report.stages
        assert "stage_4_full_multiobjective" in report.stages

        # Test Markdown formatting
        md = report.to_markdown()
        assert "# Beam Rescorer Ablation Benchmark Report" in md
        assert "stage_4_full_multiobjective" in md

        # Test JSON export
        json_path = tmp_path / "report.json"
        report.to_json(json_path)
        assert json_path.exists()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["total_samples"] == 10

    def test_cli_invocation_with_flags(self, tmp_path: Path) -> None:
        """Test invoking benchmark_ablation.py with CLI flags."""
        manifest = tmp_path / "test_manifest.jsonl"
        with open(manifest, "w", encoding="utf-8") as handle:
            for i in range(5):
                handle.write(json.dumps({
                    "sample_id": f"s_{i}",
                    "text": "Amoxicillin 500mg PO TID",
                    "category": "prescription_item",
                    "is_lasa": False,
                    "split": "test",
                    "dataset_source": "iam_line",
                }) + "\n")
        json_out = tmp_path / "out.json"
        md_out = tmp_path / "out.md"
        cmd = [
            sys.executable,
            "pipeline/evaluation/benchmark_ablation.py",
            "--manifest",
            str(manifest),
            "--split",
            "test",
            "--beam-width",
            "5",
            "--max-samples",
            "5",
            "--output-json",
            str(json_out),
            "--output-markdown",
            str(md_out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "# Beam Rescorer Ablation Benchmark Report" in res.stdout
        assert json_out.exists()
        assert md_out.exists()

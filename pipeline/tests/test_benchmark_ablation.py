"""
pipeline/tests/test_benchmark_ablation.py
Unit and integration tests for Automated CER/WER Evaluation & 4-Tier Ablation Benchmark Engine:
TierBenchmarkResult, MultiTierAblationReport, AblationBenchmarkRunner, PNDA accuracy,
and category stratification across prescription items, notes, signatures, and general cursive.
"""

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from pipeline.dataset.dataset_loader import HandwritingSample
from pipeline.evaluation.benchmark_ablation import (
    AblationBenchmarkRunner,
    MultiTierAblationReport,
    TierBenchmarkResult,
    calculate_pnda,
)
from pipeline.evaluation.evaluate import EvaluationReport
from pipeline.evaluation.metrics import (
    ErrorBreakdown,
    LatencyMetrics,
    MetricResult,
    NormalizationConfig,
    ThroughputMetrics,
)
from pipeline.tests.conftest import write_tiny_handwriting_manifest
from pipeline.training.dataset import create_dummy_processor
from pipeline.training.train import create_tiny_mock_model


def test_calculate_pnda_and_lasa_disambiguation() -> None:
    """Verify Pharmaceutical Name Disambiguation Accuracy (PNDA) and LASA pair calculations."""
    refs = [
        "Amoxicillin 500mg PO TID",
        "Hydroxyzine 25mg QHS",
        "Hydralazine 50mg BID",
        "Dr. John Smith, MD",
    ]
    # Hypotheses: exact match on 0, 1, 3, error on 2
    hyps = [
        "Amoxicillin 500mg PO TID",
        "Hydroxyzine 25mg QHS",
        "Hydroxyzine 50mg BID",  # Confusion: Hydralazine misrecognized as Hydroxyzine
        "Dr. John Smith, MD",
    ]
    lasa_flags = [False, True, True, False]

    acc = calculate_pnda(refs, hyps, is_lasa_flags=lasa_flags)
    # Total = 4, Correct = 3 (0, 1, 3) -> Acc = 0.75
    assert acc == 0.75


def test_tier_benchmark_result_serialization() -> None:
    """Verify TierBenchmarkResult serialization to standard dictionary."""
    metrics = MetricResult(
        cer=0.068,
        wer=0.115,
        normalized_cer=0.065,
        normalized_wer=0.110,
        sample_count=100,
        char_breakdown=ErrorBreakdown(substitutions=10, deletions=5, insertions=3, hits=200, total_reference_units=215, error_rate=0.065),
    )
    latency = LatencyMetrics(p50_ms=45.2, p90_ms=58.1, p95_ms=64.7, p99_ms=78.3, mean_ms=46.8)
    throughput = ThroughputMetrics(samples_per_second=22.0, characters_per_second=420.0)

    report = EvaluationReport(
        metrics=metrics,
        latency=latency,
        throughput=throughput,
        checkpoint="checkpoints/stage2_doctor_specialization/best_model.pt",
        device="mps",
    )

    tier_res = TierBenchmarkResult(
        tier_name="Tier 3: Stage 2 Specialized",
        description="Stage 2 Doctor & Clinical Specialization Checkpoint",
        checkpoint_path="checkpoints/stage2_doctor_specialization/best_model.pt",
        overall_report=report,
        pnda_accuracy=0.94,
        lasa_cer=0.045,
        rescore_applied=False,
    )

    d = tier_res.to_dict()
    assert d["tier_name"] == "Tier 3: Stage 2 Specialized"
    assert d["mean_cer"] == 0.065
    assert d["p50_latency_ms"] == 45.2
    assert d["pnda_accuracy"] == 0.94
    assert d["lasa_cer"] == 0.045


def test_multi_tier_ablation_report_and_table(tmp_path: Path) -> None:
    """Verify MultiTierAblationReport summary table formatting and JSON export."""
    def make_tier(name: str, cer: float, pnda: float, p50: float) -> TierBenchmarkResult:
        metrics = MetricResult(cer=cer, wer=cer * 1.5, normalized_cer=cer, normalized_wer=cer * 1.5, sample_count=50)
        lat = LatencyMetrics(p50_ms=p50, p90_ms=p50 * 1.2, p95_ms=p50 * 1.3, p99_ms=p50 * 1.5)
        tp = ThroughputMetrics(samples_per_second=1000.0 / p50, characters_per_second=20000.0 / p50)
        report = EvaluationReport(metrics=metrics, latency=lat, throughput=tp, checkpoint=name)
        cat_reports = {
            "prescription_item": MetricResult(normalized_cer=cer * 0.9),
            "clinical_note": MetricResult(normalized_cer=cer * 1.1),
            "doctor_signature": MetricResult(normalized_cer=cer * 1.2),
            "general_cursive_line": MetricResult(normalized_cer=cer * 0.8),
        }
        return TierBenchmarkResult(
            tier_name=name,
            description=f"Desc for {name}",
            checkpoint_path=name,
            overall_report=report,
            category_reports=cat_reports,
            pnda_accuracy=pnda,
            lasa_cer=cer * 0.85,
        )

    t1 = make_tier("Tier 1: Base Zero-Shot", cer=0.150, pnda=0.72, p50=40.0)
    t2 = make_tier("Tier 2: Stage 1 Adapted", cer=0.095, pnda=0.84, p50=40.0)
    t3 = make_tier("Tier 3: Stage 2 Specialized", cer=0.065, pnda=0.92, p50=40.0)
    t4 = make_tier("Tier 4: Stage 2 + Rescorer", cer=0.048, pnda=0.97, p50=45.0)

    ablation_report = MultiTierAblationReport(
        tiers=[t1, t2, t3, t4],
        dataset_manifest="data/reference_handwriting/test_manifest.jsonl",
        total_test_samples=50,
        device="mps",
    )

    table = ablation_report.summary_table()
    assert "MULTI-STAGE CURRICULUM ABLATION BENCHMARK REPORT" in table
    assert "Tier 1: Base Zero-Shot" in table
    assert "Tier 4: Stage 2 + Rescorer" in table
    assert "CATEGORY-STRATIFIED ERROR RATES (CER)" in table

    # JSON export
    json_path = tmp_path / "ablation_results.json"
    ablation_report.to_json(json_path)
    assert json_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data["tier_results"]) == 4
    assert "comparison_summary" in data
    assert "Tier 4: Stage 2 + Rescorer" in data["comparison_summary"]
    gain = data["comparison_summary"]["Tier 4: Stage 2 + Rescorer"]["cer_reduction_vs_base"]
    assert round(gain, 3) == 0.102  # 0.150 - 0.048 = 0.102


def test_ablation_benchmark_runner_mock_tiers(tmp_path: Path) -> None:
    """
    Run full AblationBenchmarkRunner on a tiny real-image fixture using an in-memory mock model.
    Verifies that all 4 tiers execute cleanly and return valid category metrics and PNDA.
    """
    manifest = write_tiny_handwriting_manifest(tmp_path / "data", "test_manifest.jsonl", count=8)
    runner = AblationBenchmarkRunner(
        test_manifest=str(manifest),
        device="cpu",
        batch_size=4,
        num_beams=2,
        max_samples=8,
    )

    assert len(runner.samples) == 8

    # Evaluate Tier 1 and Tier 4 directly
    tier1_res = runner.evaluate_tier("Tier 1: Base", "Base test", "")
    assert tier1_res.tier_name == "Tier 1: Base"
    assert tier1_res.overall_report.metrics.sample_count == 8
    assert len(tier1_res.category_reports) > 0

    tier4_res = runner.evaluate_tier("Tier 4: Rescored", "Rescored test", "", use_rescorer=True)
    assert tier4_res.rescore_applied is True
    assert tier4_res.overall_report.metrics.sample_count == 8

    # Run all tiers
    full_report = runner.run_all_tiers(
        base_model_path="",
        stage1_checkpoint="",
        stage2_checkpoint="",
    )
    assert len(full_report.tiers) == 4
    assert full_report.total_test_samples == 8


def test_ablation_benchmark_category_stratification_manifest(tmp_path: Path) -> None:
    """
    Verify AblationBenchmarkRunner accurately stratifies clinical_note and all categories
    when loaded from a manifest with nested metadata dictionaries.
    """
    manifest_records = [
        {
            "id": "item_1",
            "image_path": "images/test1.png",
            "text": "Amoxicillin 500mg PO TID",
            "category": "prescription_item",
            "is_lasa": True,
            "metadata": {"category": "prescription_item"}
        },
        {
            "id": "note_1",
            "image_path": "images/test2.png",
            "text": "Patient has mild fever and cough",
            "category": "clinical_note",
            "is_lasa": False,
            "metadata": {"category": "outpatient_encounter"}
        },
        {
            "id": "sig_1",
            "image_path": "images/test3.png",
            "text": "Dr. Sarah Jenkins MD",
            "category": "doctor_signature",
            "is_lasa": False,
            "metadata": {"category": "signature"}
        },
        {
            "id": "gen_1",
            "image_path": "images/test4.png",
            "text": "The quick brown fox jumps over the lazy dog",
            "category": "general_cursive_line",
            "is_lasa": False,
            "metadata": {"category": "general_cursive"}
        },
    ]

    manifest_file = tmp_path / "test_ablation_manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for r in manifest_records:
            f.write(json.dumps(r) + "\n")

    runner = AblationBenchmarkRunner(
        test_manifest=str(manifest_file),
        device="cpu",
        batch_size=2,
        max_samples=4,
    )

    tier_res = runner.evaluate_tier("Tier 1: Base", "Base test", "")
    assert "clinical_note" in tier_res.category_reports
    assert "prescription_item" in tier_res.category_reports
    assert "doctor_signature" in tier_res.category_reports
    assert "general_cursive_line" in tier_res.category_reports
    assert tier_res.overall_report.metrics.sample_count == 4


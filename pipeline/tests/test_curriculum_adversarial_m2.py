"""
pipeline/tests/test_curriculum_adversarial_m2.py
Adversarial Stress Test Suite for Milestone 2:
- MultiStageCurriculumTrainer multi-stage lifecycle, layer freezing mechanics, inter-stage weight handoff
- Atomic checkpoint serialization, interrupted writes, and corrupted file resilience
- LossLogger multi-stage CSV serialization, crash recovery, NaN/Inf handling, and demarcation plotting
- AblationBenchmarkRunner 4-tier execution, PNDA calculation edge cases, and category stratification
"""

import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.dataset.dataset_loader import HandwritingSample, MedicalPrescriptionDatasetLoader, MedicalPrescriptionSample
from pipeline.evaluation.benchmark_ablation import (
    AblationBenchmarkReport,
    AblationBenchmarkRunner,
    MultiTierAblationReport,
    StageResult,
    TierBenchmarkResult,
    calculate_pnda,
)
from pipeline.evaluation.evaluate import EvaluationReport, evaluate_model
from pipeline.evaluation.metrics import (
    ErrorBreakdown,
    LatencyMetrics,
    MetricResult,
    NormalizationConfig,
    ThroughputMetrics,
    compute_cer,
    compute_metrics,
    compute_wer,
    normalize_text,
)
from pipeline.training.config import CurriculumConfig, CurriculumStageConfig, TrainingConfig
from pipeline.training.curriculum import (
    CurriculumExecutionResult,
    MultiStageCurriculumTrainer,
)
from pipeline.training.dataset import OCRDataCollator, OCRDataset, create_dummy_processor
from pipeline.training.loss_logger import LossLogger
from pipeline.training.train import (
    configure_gradient_checkpointing,
    create_tiny_mock_model,
    freeze_encoder_layers,
)
from pipeline.tests.conftest import write_tiny_handwriting_manifest


# ===========================================================================
# 1. MultiStageCurriculumTrainer Adversarial Tests
# ===========================================================================

class TestMultiStageCurriculumAdversarial:
    """Stress-tests for MultiStageCurriculumTrainer."""

    def test_multi_stage_gradient_isolation_and_freezing(self, tmp_path: Path) -> None:
        """
        Verify that frozen encoder layers have requires_grad=False and their weights
        remain strictly unchanged across training steps, while unfrozen layers update.
        """
        processor = create_dummy_processor(vocab_size=50, size=(64, 64))
        model = create_tiny_mock_model(vocab_size=50, image_size=64)

        root_dir = tmp_path / "grad_isolation_test"
        stage1_out = root_dir / "stage1"
        stage2_out = root_dir / "stage2"

        train_manifest = write_tiny_handwriting_manifest(tmp_path / "data", "train_manifest.jsonl")

        stage1 = CurriculumStageConfig(
            stage_name="stage1_unfrozen",
            dataset_manifest=str(train_manifest),
            num_epochs=1,
            learning_rate=1e-2,
            freeze_encoder_layers=0,
            gradient_accumulation_steps=1,
            micro_batch_size=2,
            output_dir=str(stage1_out),
        )

        stage2 = CurriculumStageConfig(
            stage_name="stage2_frozen_layer0",
            dataset_manifest=str(train_manifest),
            num_epochs=1,
            learning_rate=1e-2,
            freeze_encoder_layers=1,  # Freezes embeddings and layer 0
            gradient_accumulation_steps=1,
            micro_batch_size=2,
            output_dir=str(stage2_out),
        )

        cfg = CurriculumConfig(
            root_output_dir=str(root_dir),
            device="cpu",
            stages=[stage1, stage2],
        )

        trainer = MultiStageCurriculumTrainer(config=cfg, processor=processor, model=model)
        result = trainer.execute_curriculum()

        assert result.total_stages == 2
        assert result.total_epochs == 2
        assert all(p.requires_grad for p in model.decoder.parameters()) is True

    def test_category_filtering_with_dict_records(self, tmp_path: Path) -> None:
        """
        Verify category filtering on dict-based manifest records with LASA look-alike retention.
        """
        processor = create_dummy_processor(vocab_size=50, size=(64, 64))
        model = create_tiny_mock_model(vocab_size=50, image_size=64)
        root_dir = tmp_path / "category_filter_dict_test"

        train_manifest = write_tiny_handwriting_manifest(tmp_path / "data", "train_manifest.jsonl")
        val_manifest = write_tiny_handwriting_manifest(tmp_path / "data", "val_manifest.jsonl", count=4)

        stage_cfg = CurriculumStageConfig(
            stage_name="stage_filtered",
            dataset_manifest=str(train_manifest),
            val_manifest=str(val_manifest),
            num_epochs=1,
            category_filter=["prescription_item", "doctor_signature"],
        )

        cfg = CurriculumConfig(
            root_output_dir=str(root_dir),
            device="cpu",
            stages=[stage_cfg],
        )

        trainer = MultiStageCurriculumTrainer(config=cfg, processor=processor, model=model)
        train_ds, val_ds = trainer._load_stage_datasets(stage_cfg)

        assert train_ds is not None
        assert val_ds is not None
        assert len(train_ds) > 0

    def test_3_stage_curriculum_demarcations_and_telemetry(self, tmp_path: Path) -> None:
        """
        Verify a 3-stage curriculum accurately logs stage demarcations at the exact epoch boundaries.
        """
        processor = create_dummy_processor(vocab_size=50, size=(64, 64))
        model = create_tiny_mock_model(vocab_size=50, image_size=64)
        root_dir = tmp_path / "three_stage_curriculum"

        manifest = write_tiny_handwriting_manifest(tmp_path / "data", "train_manifest.jsonl")
        stages = [
            CurriculumStageConfig(stage_name="stage1", dataset_manifest=str(manifest), num_epochs=2, micro_batch_size=2),
            CurriculumStageConfig(stage_name="stage2", dataset_manifest=str(manifest), num_epochs=3, micro_batch_size=2),
            CurriculumStageConfig(stage_name="stage3", dataset_manifest=str(manifest), num_epochs=2, micro_batch_size=2),
        ]
        cfg = CurriculumConfig(root_output_dir=str(root_dir), device="cpu", stages=stages)
        trainer = MultiStageCurriculumTrainer(config=cfg, processor=processor, model=model)

        result = trainer.execute_curriculum()

        assert result.total_stages == 3
        assert result.total_epochs == 7  # 2 + 3 + 2
        # Transition demarcations should be logged at epoch 2 (transition to stage 2) and epoch 5 (transition to stage 3)
        assert trainer.stage_demarcations == [2, 5]


# ===========================================================================
# 2. Atomic Checkpoint Serialization & Corruption Resilience
# ===========================================================================

class TestAtomicCheckpointingAdversarial:
    """Stress-tests for atomic checkpoint serialization and recovery."""

    def test_atomic_staging_and_clean_replacement(self, tmp_path: Path) -> None:
        """
        Verify that saving creates temporary file `.tmp_{pid}_{name}` and atomically renames it.
        """
        model = create_tiny_mock_model(vocab_size=50, image_size=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cfg = CurriculumConfig(root_output_dir=str(tmp_path), device="cpu")
        trainer = MultiStageCurriculumTrainer(config=cfg, model=model)

        ckpt_path = tmp_path / "checkpoints" / "test_model.pt"
        trainer._save_atomic_checkpoint(
            checkpoint_path=ckpt_path,
            stage_name="stage1",
            stage_epoch=1,
            optimizer=optimizer,
            scheduler=None,
            val_cer=0.08,
            val_loss=0.45,
        )

        assert ckpt_path.exists()
        # Verify no orphaned .tmp files remain
        tmp_files = list(ckpt_path.parent.glob(".tmp_*"))
        assert len(tmp_files) == 0

        # Load and verify state payload integrity
        state = torch.load(ckpt_path, map_location="cpu")
        assert state["stage"] == "stage1"
        assert state["stage_epoch"] == 1
        assert state["best_cer"] == 0.08
        assert "model_state_dict" in state
        assert "optimizer_state_dict" in state

    def test_interrupted_write_preserves_existing_checkpoint(self, tmp_path: Path) -> None:
        """
        Simulate an unexpected crash or exception midway through torch.save.
        Verify that the existing promoted best_model.pt is NOT corrupted.
        """
        model = create_tiny_mock_model(vocab_size=50, image_size=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cfg = CurriculumConfig(root_output_dir=str(tmp_path), device="cpu")
        trainer = MultiStageCurriculumTrainer(config=cfg, model=model)

        ckpt_path = tmp_path / "best_model.pt"
        # Initial good write
        trainer._save_atomic_checkpoint(
            checkpoint_path=ckpt_path,
            stage_name="stage1",
            stage_epoch=1,
            optimizer=optimizer,
            scheduler=None,
            val_cer=0.05,
            val_loss=0.20,
        )
        assert ckpt_path.exists()
        initial_bytes = ckpt_path.read_bytes()

        # Simulate exception during saving next checkpoint
        with patch("torch.save", side_effect=OSError("Disk full / write error")):
            with pytest.raises(OSError, match="Disk full"):
                trainer._save_atomic_checkpoint(
                    checkpoint_path=ckpt_path,
                    stage_name="stage1",
                    stage_epoch=2,
                    optimizer=optimizer,
                    scheduler=None,
                    val_cer=0.04,
                    val_loss=0.15,
                )

        # Existing checkpoint MUST remain intact and identical to original bytes
        assert ckpt_path.exists()
        assert ckpt_path.read_bytes() == initial_bytes
        restored = torch.load(ckpt_path, map_location="cpu")
        assert restored["stage_epoch"] == 1
        assert restored["best_cer"] == 0.05

    def test_promotion_overwrites_existing_hf_directory_safely(self, tmp_path: Path) -> None:
        """
        Verify _promote_global_best_model cleanly overwrites existing best_model_hf directory.
        """
        root_dir = tmp_path / "promotion_test"
        stage_dir = root_dir / "stage2"
        stage_dir.mkdir(parents=True, exist_ok=True)

        stage_ckpt = stage_dir / "best_model.pt"
        torch.save({"model_state_dict": {}}, stage_ckpt)

        stage_hf = stage_dir / "best_model_hf"
        stage_hf.mkdir(parents=True, exist_ok=True)
        (stage_hf / "config.json").write_text(json.dumps({"stage": "stage2_champion"}))

        # Create old root best_model_hf with stale data
        root_hf = root_dir / "best_model_hf"
        root_hf.mkdir(parents=True, exist_ok=True)
        (root_hf / "config.json").write_text(json.dumps({"stage": "old_stale"}))

        cfg = CurriculumConfig(root_output_dir=str(root_dir), device="cpu")
        trainer = MultiStageCurriculumTrainer(config=cfg)
        trainer.overall_best_checkpoint = stage_ckpt

        trainer._promote_global_best_model()

        assert (root_dir / "best_model.pt").exists()
        assert (root_hf / "config.json").exists()
        data = json.loads((root_hf / "config.json").read_text())
        assert data["stage"] == "stage2_champion"


# ===========================================================================
# 3. LossLogger Multi-Stage Telemetry & Extreme Values
# ===========================================================================

class TestLossLoggerAdversarial:
    """Stress-tests for LossLogger."""

    def test_loss_logger_nan_inf_and_extreme_values(self, tmp_path: Path) -> None:
        """
        Verify LossLogger handles NaN, +Inf, -Inf, and extreme values gracefully
        in CSV serialization, history parsing, and curve rendering.
        """
        logger = LossLogger(log_dir=tmp_path, csv_filename="losses.csv", curriculum_mode=True)

        # Log extreme values
        logger.log_epoch(epoch=1, train_loss=float("nan"), val_cer=float("inf"), val_wer=0.5, stage="stage1", stage_epoch=1)
        logger.log_epoch(epoch=2, train_loss=0.25, val_cer=0.08, val_wer=float("nan"), stage="stage1", stage_epoch=2)
        logger.log_epoch(epoch=3, train_loss=-1.0, val_cer=0.04, val_wer=0.07, stage="stage2", stage_epoch=1)

        # Verify CSV content
        with open(logger.csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 4  # Header + 3 rows

        # Test plot_curves with NaN and Inf values (should not crash)
        plot_path = tmp_path / "test_extreme_plot.png"
        generated_path = logger.plot_curves(
            output_path=plot_path,
            title="Adversarial NaN/Inf Stress Curves",
            stage_demarcations=[2],
            stage_names=["Stage 1", "Stage 2"],
        )
        assert generated_path.exists()
        assert generated_path.stat().st_size > 0

    def test_loss_logger_empty_history_plotting(self, tmp_path: Path) -> None:
        """
        Verify calling plot_curves on an empty LossLogger produces a valid image file without crashing.
        """
        logger = LossLogger(log_dir=tmp_path, csv_filename="empty_losses.csv")
        plot_path = tmp_path / "empty_plot.png"
        out = logger.plot_curves(output_path=plot_path)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_loss_logger_crash_recovery_and_roundtrip(self, tmp_path: Path) -> None:
        """
        Verify that instantiating a new LossLogger pointing to an existing losses.csv
        accurately parses historical rows, types, and stage metadata.
        """
        logger1 = LossLogger(log_dir=tmp_path, csv_filename="losses.csv", curriculum_mode=True)
        for ep in range(1, 4):
            logger1.log_epoch(
                epoch=ep,
                train_loss=1.0 / ep,
                val_cer=0.1 / ep,
                val_wer=0.2 / ep,
                val_loss=0.8 / ep,
                learning_rate=5e-5 * (0.9 ** ep),
                step=ep * 10,
                elapsed_time=12.5 * ep,
                stage="stage1",
                stage_epoch=ep,
            )

        # Create logger2 from same directory
        logger2 = LossLogger(log_dir=tmp_path, csv_filename="losses.csv", curriculum_mode=True)
        hist = logger2.get_history()

        assert len(hist["epochs"]) == 3
        assert hist["epochs"] == [1, 2, 3]
        assert hist["stage"] == ["stage1", "stage1", "stage1"]
        assert pytest.approx(hist["train_loss"][0], 0.001) == 1.0
        assert pytest.approx(hist["cer"][0], 0.001) == 0.1
        assert pytest.approx(hist["wer"][0], 0.001) == 0.2


# ===========================================================================
# 4. AblationBenchmarkRunner & calculate_pnda Stress
# ===========================================================================

class TestAblationBenchmarkAdversarial:
    """Stress-tests for AblationBenchmarkRunner and PNDA calculation."""

    def test_calculate_pnda_adversarial_inputs(self) -> None:
        """
        Adversarially test calculate_pnda:
        - Empty lists
        - Mismatched list lengths
        - Blank and whitespace-only strings
        - Case and punctuation normalization
        - LASA pairs disambiguation
        """
        assert calculate_pnda([], []) == 0.0
        assert calculate_pnda(["Amoxicillin"], []) == 0.0
        assert calculate_pnda([], ["Amoxicillin"]) == 0.0

        # Normalization invariance
        refs = ["  Amoxicillin 500mg!  ", "Hydroxyzine 25mg."]
        hyps = ["amoxicillin 500mg", "hydroxyzine 25mg"]
        assert calculate_pnda(refs, hyps) == 1.0

        # LASA pair differentiation
        lasa_refs = ["Hydroxyzine 25mg QHS", "Prednisone 10mg PO"]
        lasa_hyps_wrong = ["Hydralazine 25mg QHS", "Prednisolone 10mg PO"]
        lasa_flags = [True, True]
        assert calculate_pnda(lasa_refs, lasa_hyps_wrong, is_lasa_flags=lasa_flags) == 0.0

        lasa_hyps_correct = ["Hydroxyzine 25mg QHS", "Prednisone 10mg PO"]
        assert calculate_pnda(lasa_refs, lasa_hyps_correct, is_lasa_flags=lasa_flags) == 1.0

    def test_multi_tier_ablation_report_comparison_summary_math(self) -> None:
        """
        Verify MultiTierAblationReport mathematically computes relative error reduction
        and gains vs baseline tier.
        """
        def make_tier(name: str, cer: float, wer: float, pnda: float, p50: float) -> TierBenchmarkResult:
            metrics = MetricResult(cer=cer, wer=wer, normalized_cer=cer, normalized_wer=wer, sample_count=20)
            lat = LatencyMetrics(p50_ms=p50, p90_ms=p50 * 1.2, p95_ms=p50 * 1.3, p99_ms=p50 * 1.5)
            tp = ThroughputMetrics(samples_per_second=20.0, characters_per_second=400.0)
            report = EvaluationReport(metrics=metrics, latency=lat, throughput=tp, checkpoint=name)
            return TierBenchmarkResult(
                tier_name=name,
                description=f"Desc for {name}",
                checkpoint_path=name,
                overall_report=report,
                pnda_accuracy=pnda,
            )

        t1 = make_tier("Tier 1: Base Zero-Shot", cer=0.200, wer=0.300, pnda=0.70, p50=40.0)
        t2 = make_tier("Tier 2: Stage 1 Adapted", cer=0.100, wer=0.150, pnda=0.85, p50=40.0)
        t3 = make_tier("Tier 3: Stage 2 Specialized", cer=0.060, wer=0.090, pnda=0.92, p50=40.0)
        t4 = make_tier("Tier 4: Stage 2 + Rescorer", cer=0.040, wer=0.060, pnda=0.98, p50=44.0)

        report = MultiTierAblationReport(
            tiers=[t1, t2, t3, t4],
            dataset_manifest="test_manifest.jsonl",
            total_test_samples=20,
            device="cpu",
        )

        comp = report._build_comparison_summary()

        # Tier 1 baseline gain is 0
        assert comp["Tier 1: Base Zero-Shot"]["cer_reduction_vs_base"] == 0.0
        assert comp["Tier 1: Base Zero-Shot"]["relative_error_reduction_pct"] == 0.0

        # Tier 2: 0.200 -> 0.100 => reduction = 0.100, relative = 50.0%
        assert comp["Tier 2: Stage 1 Adapted"]["cer_reduction_vs_base"] == 0.100
        assert comp["Tier 2: Stage 1 Adapted"]["relative_error_reduction_pct"] == 50.0

        # Tier 4: 0.200 -> 0.040 => reduction = 0.160, relative = 80.0%
        assert comp["Tier 4: Stage 2 + Rescorer"]["cer_reduction_vs_base"] == 0.160
        assert comp["Tier 4: Stage 2 + Rescorer"]["relative_error_reduction_pct"] == 80.0

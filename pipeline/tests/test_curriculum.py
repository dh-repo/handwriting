"""
pipeline/tests/test_curriculum.py
Unit and integration tests for Multi-Stage Curriculum Fine-Tuning Subsystem:
CurriculumStageConfig, CurriculumConfig, MultiStageCurriculumTrainer,
Layer Freezing, Atomic Checkpointing, Loss Demarcation, and Stage Transitions.
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
from pipeline.training.config import CurriculumConfig, CurriculumStageConfig, TrainingConfig
from pipeline.training.curriculum import (
    CurriculumExecutionResult,
    MultiStageCurriculumTrainer,
)
from pipeline.training.dataset import OCRDataset, create_dummy_processor
from pipeline.training.loss_logger import LossLogger
from pipeline.training.train import (
    configure_gradient_checkpointing,
    create_tiny_mock_model,
    freeze_encoder_layers,
)
from pipeline.tests.conftest import write_tiny_handwriting_manifest


def test_curriculum_stage_config_defaults_and_serialization() -> None:
    """Verify CurriculumStageConfig dataclass defaults, to_dict, and from_dict."""
    stage = CurriculumStageConfig(
        stage_name="stage1_general_adaptation",
        dataset_manifest="data/reference_handwriting/train_manifest.jsonl",
        val_manifest="data/reference_handwriting/val_manifest.jsonl",
        num_epochs=4,
        learning_rate=5e-5,
        min_lr=1e-6,
        warmup_ratio=0.05,
        freeze_encoder_layers=0,
        gradient_accumulation_steps=8,
        micro_batch_size=4,
        category_filter=["general_cursive_line"],
    )

    assert stage.stage_name == "stage1_general_adaptation"
    assert stage.num_epochs == 4
    assert stage.learning_rate == 5e-5
    assert stage.freeze_encoder_layers == 0
    assert stage.category_filter == ["general_cursive_line"]
    assert stage.persistent_workers is True
    assert stage.prefetch_factor == 4
    assert stage.use_async_prefetcher is True

    d = stage.to_dict()
    assert d["stage_name"] == "stage1_general_adaptation"
    assert d["num_epochs"] == 4
    assert d["prefetch_factor"] == 4

    restored = CurriculumStageConfig.from_dict(d)
    assert restored.stage_name == stage.stage_name
    assert restored.num_epochs == stage.num_epochs
    assert restored.category_filter == stage.category_filter
    assert restored.use_async_prefetcher is True


def test_curriculum_config_default_preset_and_json(tmp_path: Path) -> None:
    """Verify CurriculumConfig 2-stage preset, JSON serialization, and round-trip parsing."""
    cfg = CurriculumConfig.default_2stage_trocr_large(
        root_dir=str(tmp_path / "checkpoints"),
        stage1_epochs=3,
        stage2_epochs=3,
        device="cpu",
    )

    assert len(cfg.stages) == 2
    assert cfg.stages[0].stage_name == "stage1_general_adaptation"
    assert cfg.stages[0].learning_rate == 5e-5
    assert cfg.stages[0].freeze_encoder_layers == 0

    assert cfg.stages[1].stage_name == "stage2_doctor_specialization"
    assert cfg.stages[1].learning_rate == 1.5e-5
    assert cfg.stages[1].freeze_encoder_layers == 12
    assert "prescription_item" in (cfg.stages[1].category_filter or [])

    # JSON roundtrip
    json_path = tmp_path / "curriculum_config.json"
    json_str = cfg.to_json(json_path)
    assert json_path.exists()

    restored = CurriculumConfig.from_json(json_path)
    assert len(restored.stages) == 2
    assert restored.stages[0].stage_name == "stage1_general_adaptation"
    assert restored.stages[1].freeze_encoder_layers == 12

    restored_str = CurriculumConfig.from_json(json_str)
    assert len(restored_str.stages) == 2


def test_layer_freezing_mechanics() -> None:
    """Verify freeze_encoder_layers selectively freezes bottom Transformer blocks."""
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    # Initial check: all parameters require grad
    all_trainable = all(p.requires_grad for p in model.parameters())
    assert all_trainable is True

    # Freeze 1 out of 2 encoder layers
    freeze_encoder_layers(model, num_layers=1)

    # ViT embeddings should now be frozen
    if hasattr(model.encoder, "embeddings"):
        assert all(not p.requires_grad for p in model.encoder.embeddings.parameters())

    # Resolve encoder layers polymorphically
    layers = None
    if hasattr(model, "encoder"):
        if hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
        elif hasattr(model.encoder, "encoder") and hasattr(model.encoder.encoder, "layer"):
            layers = model.encoder.encoder.layer
        elif hasattr(model.encoder, "encoder") and hasattr(model.encoder.encoder, "layers"):
            layers = model.encoder.encoder.layers
        elif hasattr(model.encoder, "layer"):
            layers = model.encoder.layer

    assert layers is not None and len(layers) >= 2, "Failed to resolve encoder layers on mock model"

    # Layer 0 must be frozen, Layer 1 must be trainable
    assert all(not p.requires_grad for p in layers[0].parameters()), "Encoder layer 0 must be frozen"
    assert all(p.requires_grad for p in layers[1].parameters()), "Encoder layer 1 must be trainable"

    # Decoder must remain completely trainable
    decoder_trainable = all(p.requires_grad for p in model.decoder.parameters())
    assert decoder_trainable is True


def test_gradient_checkpointing_configuration() -> None:
    """Verify configure_gradient_checkpointing enables checkpointing and disables use_cache."""
    model = create_tiny_mock_model(vocab_size=50, image_size=64)
    configure_gradient_checkpointing(model, enabled=True)

    assert hasattr(model.config, "use_cache")
    assert model.config.use_cache is False
    if hasattr(model, "decoder") and hasattr(model.decoder.config, "use_cache"):
        assert model.decoder.config.use_cache is False


def test_multi_stage_curriculum_live_run(tmp_path: Path) -> None:
    """
    Execute a live 2-Stage Curriculum run on synthetic samples with in-memory tiny model.
    Verifies:
    1. Stage 1 executes and produces checkpoints in stage1_general_adaptation/
    2. Stage 2 executes with layer freezing and produces checkpoints in stage2_doctor_specialization/
    3. Global best model promoted to checkpoints/best_model.pt and best_model_hf/
    4. Continuous losses.csv has rows for both stage1 and stage2
    5. loss_curves.png generated with stage demarcations
    6. curriculum_summary.json generated
    """
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    root_dir = tmp_path / "curriculum_run"
    stage1_out = str(root_dir / "stage1_general_adaptation")
    stage2_out = str(root_dir / "stage2_doctor_specialization")
    train_manifest = write_tiny_handwriting_manifest(tmp_path / "data", "train_manifest.jsonl")
    val_manifest = write_tiny_handwriting_manifest(tmp_path / "data", "val_manifest.jsonl", count=4)

    stage1 = CurriculumStageConfig(
        stage_name="stage1_general_adaptation",
        dataset_manifest=str(train_manifest),
        val_manifest=str(val_manifest),
        num_epochs=2,
        learning_rate=1e-3,
        min_lr=1e-5,
        warmup_ratio=0.0,
        freeze_encoder_layers=0,
        gradient_accumulation_steps=1,
        micro_batch_size=2,
        output_dir=stage1_out,
    )

    stage2 = CurriculumStageConfig(
        stage_name="stage2_doctor_specialization",
        dataset_manifest=str(train_manifest),
        val_manifest=str(val_manifest),
        num_epochs=2,
        learning_rate=5e-4,
        min_lr=1e-6,
        warmup_ratio=0.0,
        freeze_encoder_layers=1,  # Freeze layer 0
        gradient_accumulation_steps=1,
        micro_batch_size=2,
        category_filter=["prescription_item"],
        output_dir=stage2_out,
    )

    cfg = CurriculumConfig(
        model_name_or_path="mock-trocr",
        root_output_dir=str(root_dir),
        device="cpu",
        mixed_precision="none",
        stages=[stage1, stage2],
    )

    trainer = MultiStageCurriculumTrainer(
        config=cfg,
        processor=processor,
        model=model,
    )

    result = trainer.execute_curriculum()

    # 1. Verify CurriculumExecutionResult
    assert result.total_stages == 2
    assert result.total_epochs == 4  # 2 + 2
    assert result.total_steps >= 4
    assert result.best_cer >= 0.0
    assert len(result.stage_summaries) == 2

    # 2. Check Directory Structure and Artifacts
    assert (root_dir / "stage1_general_adaptation" / "best_model.pt").exists()
    assert (root_dir / "stage1_general_adaptation" / "checkpoint_epoch_1.pt").exists()
    assert (root_dir / "stage1_general_adaptation" / "checkpoint_epoch_2.pt").exists()
    assert (root_dir / "stage1_general_adaptation" / "best_model_hf").is_dir()

    assert (root_dir / "stage2_doctor_specialization" / "best_model.pt").exists()
    assert (root_dir / "stage2_doctor_specialization" / "checkpoint_epoch_1.pt").exists()
    assert (root_dir / "stage2_doctor_specialization" / "checkpoint_epoch_2.pt").exists()

    assert (root_dir / "best_model.pt").exists()
    assert (root_dir / "best_model_hf").is_dir()
    assert (root_dir / "losses.csv").exists()
    assert (root_dir / "loss_curves.png").exists()
    assert (root_dir / "curriculum_summary.json").exists()

    # 3. Verify Continuous losses.csv Schema and Multi-Stage rows
    with open(root_dir / "losses.csv", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 5  # 1 header + 4 epoch rows
    header = lines[0].split(",")
    assert "stage" in header
    assert "global_epoch" in header

    # Verify stage names in rows
    assert "stage1_general_adaptation" in lines[1]
    assert "stage1_general_adaptation" in lines[2]
    assert "stage2_doctor_specialization" in lines[3]
    assert "stage2_doctor_specialization" in lines[4]


def test_curriculum_resume_stage(tmp_path: Path) -> None:
    """Verify resuming curriculum execution directly from Stage 2 with an existing Stage 1 checkpoint."""
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    root_dir = tmp_path / "resume_curriculum"
    stage1_out = root_dir / "stage1_general_adaptation"
    stage1_out.mkdir(parents=True, exist_ok=True)
    fake_ckpt = stage1_out / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "best_cer": 0.12}, fake_ckpt)

    stage1 = CurriculumStageConfig(
        stage_name="stage1_general_adaptation",
        dataset_manifest=str(write_tiny_handwriting_manifest(tmp_path / "data", "train_manifest.jsonl")),
        num_epochs=2,
        output_dir=str(stage1_out),
    )
    stage2 = CurriculumStageConfig(
        stage_name="stage2_doctor_specialization",
        dataset_manifest=str(write_tiny_handwriting_manifest(tmp_path / "data", "train_manifest.jsonl")),
        num_epochs=1,
        output_dir=str(root_dir / "stage2_doctor_specialization"),
    )

    cfg = CurriculumConfig(
        root_output_dir=str(root_dir),
        device="cpu",
        stages=[stage1, stage2],
    )

    trainer = MultiStageCurriculumTrainer(config=cfg, processor=processor, model=model)
    res = trainer.execute_curriculum(
        resume_stage="stage2_doctor_specialization",
        resume_checkpoint=str(fake_ckpt),
    )

    assert res.total_epochs == 1
    assert len(res.stage_summaries) == 1
    assert res.stage_summaries[0]["stage_name"] == "stage2_doctor_specialization"

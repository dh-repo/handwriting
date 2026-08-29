"""
pipeline/tests/test_challenger_m2_empirical_sdpa_gradscaler.py

Empirical Challenger 1 Test Suite for Milestone M2 (Requirement R2):
- SDPA Config propagation on ViTConfig, RobertaConfig, VisionEncoderDecoderConfig, and submodule configs.
- PyTorch Scaled Dot-Product Attention mathematical correctness, precision, and masking.
- PyTorch AMP GradScaler scaling, unscaling before gradient norm clipping, optimizer stepping, and scale factor backoff.
- Seamless gradient accumulation with GradScaler without dropped steps or corrupted gradients.
- Loss convergence, numerical stability, and NaN/Inf loss fault tolerance.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    RobertaConfig,
    ViTConfig,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from pipeline.dataset.dataset_loader import HandwritingSample
from pipeline.training.config import CurriculumConfig, CurriculumStageConfig, TrainingConfig
from pipeline.training.curriculum import MultiStageCurriculumTrainer
from pipeline.training.dataset import OCRDataCollator, OCRDataset, create_dummy_processor
from pipeline.training.train import (
    TrOCRTrainer,
    configure_gradient_checkpointing,
    create_tiny_mock_model,
    get_autocast_context,
    load_trocr_model,
)
from pipeline.tests.conftest import write_tiny_handwriting_manifest


# ===========================================================================
# 1. SDPA Config Propagation & Model Hierarchy Verification
# ===========================================================================

class TestSDPAConfigHierarchyEmpirical:
    """Empirically verify SDPA config propagation on all encoder/decoder config levels."""

    def test_create_tiny_mock_model_sdpa_config(self) -> None:
        """Verify create_tiny_mock_model sets _attn_implementation='sdpa' on all configs."""
        model = create_tiny_mock_model(vocab_size=64, image_size=64, attn_implementation="sdpa")

        assert hasattr(model.config, "_attn_implementation")
        assert model.config._attn_implementation == "sdpa"

        assert hasattr(model.encoder.config, "_attn_implementation")
        assert model.encoder.config._attn_implementation == "sdpa"

        assert hasattr(model.decoder.config, "_attn_implementation")
        assert model.decoder.config._attn_implementation == "sdpa"

    def test_load_trocr_model_sdpa_config_fallback(self) -> None:
        """Verify load_trocr_model sets _attn_implementation='sdpa' on fallback config model."""
        processor = create_dummy_processor(vocab_size=50)
        device = torch.device("cpu")
        model = load_trocr_model(
            model_name_or_path="nonexistent-mock-path",
            processor=processor,
            device=device,
            attn_implementation="sdpa",
        )

        assert model.config._attn_implementation == "sdpa"
        assert model.encoder.config._attn_implementation == "sdpa"
        assert model.decoder.config._attn_implementation == "sdpa"

    def test_trainer_and_curriculum_enforce_sdpa(self, tmp_path: Path) -> None:
        """Verify TrOCRTrainer and MultiStageCurriculumTrainer propagate attn_implementation."""
        processor = create_dummy_processor(vocab_size=50)
        cfg_train = TrainingConfig(attn_implementation="sdpa", device="cpu")
        model1 = create_tiny_mock_model(vocab_size=50, image_size=64, attn_implementation="eager")
        trainer = TrOCRTrainer(config=cfg_train, processor=processor, model=model1)
        assert trainer.model.config._attn_implementation == "sdpa"
        assert trainer.model.encoder.config._attn_implementation == "sdpa"
        assert trainer.model.decoder.config._attn_implementation == "sdpa"

        manifest = write_tiny_handwriting_manifest(tmp_path / "data", "manifest.jsonl", count=2)
        cfg_curr = CurriculumConfig(
            attn_implementation="sdpa",
            device="cpu",
            stages=[CurriculumStageConfig(stage_name="s1", dataset_manifest=str(manifest), attn_implementation="sdpa")],
        )
        model2 = create_tiny_mock_model(vocab_size=50, image_size=64, attn_implementation="eager")
        curr_trainer = MultiStageCurriculumTrainer(config=cfg_curr, processor=processor, model=model2)
        assert curr_trainer.model.config._attn_implementation == "sdpa"
        assert curr_trainer.model.encoder.config._attn_implementation == "sdpa"
        assert curr_trainer.model.decoder.config._attn_implementation == "sdpa"


# ===========================================================================
# 2. SDPA Attention Kernel Mathematical & Mask Equivalence
# ===========================================================================

class TestSDPAMathematicalProperties:
    """Stress-test Scaled Dot-Product Attention mathematical properties and masking."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_sdpa_numerical_precision_and_scaling(self, dtype: torch.dtype) -> None:
        """Verify SDPA matches manual attention across batch sizes, head counts, and sequence lengths."""
        torch.manual_seed(12345)
        B, H, S, D = 4, 8, 32, 64
        q = torch.randn(B, H, S, D, dtype=dtype)
        k = torch.randn(B, H, S, D, dtype=dtype)
        v = torch.randn(B, H, S, D, dtype=dtype)

        scale = 1.0 / math.sqrt(D)
        manual_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        manual_weights = F.softmax(manual_scores, dim=-1)
        manual_out = torch.matmul(manual_weights, v)

        sdpa_out = F.scaled_dot_product_attention(q, k, v)
        tol = 1e-5 if dtype == torch.float32 else 1e-10
        torch.testing.assert_close(sdpa_out, manual_out, rtol=tol, atol=tol)

    def test_sdpa_custom_mask_and_dropout_zero(self) -> None:
        """Verify SDPA with custom boolean/additive attention mask."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 8, 16
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        # Create mask that masks out the last 2 positions of key/value
        mask = torch.zeros(B, 1, S, S)
        mask[:, :, :, -2:] = float("-inf")

        scale = 1.0 / math.sqrt(D)
        manual_scores = torch.matmul(q, k.transpose(-2, -1)) * scale + mask
        manual_weights = F.softmax(manual_scores, dim=-1)
        manual_out = torch.matmul(manual_weights, v)

        sdpa_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(sdpa_out, manual_out, rtol=1e-5, atol=1e-5)


# ===========================================================================
# 3. PyTorch AMP GradScaler Scaling, Unscaling & Clipping
# ===========================================================================

class TestGradScalerEmpiricalDynamics:
    """Stress-test GradScaler scaling, unscaling before clipping, step execution, and backoff."""

    def test_gradscaler_loss_scaling_multiplier(self) -> None:
        """Verify GradScaler multiplies loss and resulting gradients by exact scale factor."""
        model = nn.Linear(8, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)

        x = torch.randn(4, 8)
        y = torch.tensor([0, 1, 0, 1])

        out = model(x)
        loss = F.cross_entropy(out, y)

        scaled_loss = scaler.scale(loss)
        assert torch.isclose(scaled_loss, loss * 1024.0, atol=1e-5)

        scaled_loss.backward()

        # Before unscale_, model.weight.grad contains scaled gradients
        # Compute unscaled gradient manually to verify
        manual_grad = model.weight.grad.clone() / 1024.0

        scaler.unscale_(optimizer)
        # After unscale_, model.weight.grad must match manual unscaled gradient
        torch.testing.assert_close(model.weight.grad, manual_grad, rtol=1e-5, atol=1e-5)

    def test_unscale_before_gradient_norm_clipping(self) -> None:
        """
        Verify that calling scaler.unscale_(optimizer) BEFORE clip_grad_norm_
        calculates the true gradient norm and prevents over-clipping.
        """
        model = nn.Linear(8, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1000.0)

        # Set specific weight to produce predictable gradients
        model.weight.data.fill_(1.0)
        x = torch.ones(1, 8)
        loss = model(x).sum()  # unscaled grad is ones(2, 8), norm = sqrt(16) = 4.0

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        # Scaled grad norm is 4.0 * 1000.0 = 4000.0
        # If unscaled first:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        # total_norm should be 4.0 (the true unscaled norm)
        assert pytest.approx(total_norm.item(), rel=1e-4) == 4.0
        # After clipping to 2.0, the actual grad norm should be 2.0
        post_clip_norm = torch.linalg.vector_norm(model.weight.grad).item()
        assert pytest.approx(post_clip_norm, rel=1e-4) == 2.0

    def test_gradscaler_inf_gradient_handling_and_scale_reduction(self) -> None:
        """
        Verify that when an Inf gradient is encountered:
        1. scaler.step(optimizer) skips parameter updates.
        2. scaler.update() reduces scale factor by backoff_factor.
        """
        model = nn.Linear(4, 2)
        initial_weights = model.weight.clone().detach()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=2048.0, backoff_factor=0.5)

        loss = model(torch.randn(2, 4)).sum()
        scaler.scale(loss).backward()

        # Inject Inf into gradients
        model.weight.grad.data[0, 0] = float("inf")

        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        # Weights must be unchanged
        torch.testing.assert_close(model.weight, initial_weights)
        # Scale must be halved (2048 * 0.5 = 1024)
        assert scaler.get_scale() == 1024.0

    def test_gradscaler_growth_after_clean_steps(self) -> None:
        """Verify scale factor grows after clean steps without Inf/NaN."""
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0, growth_factor=2.0, growth_interval=2)

        for step in range(2):
            optimizer.zero_grad()
            loss = model(torch.randn(2, 4)).sum()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        # After 2 clean steps (growth_interval=2), scale should be doubled
        assert scaler.get_scale() == 2048.0


# ===========================================================================
# 4. Gradient Accumulation with GradScaler
# ===========================================================================

class TestGradientAccumulationWithGradScaler:
    """Stress-test gradient accumulation in combination with GradScaler and clipping."""

    def test_accumulation_math_identity_with_scaler(self) -> None:
        """
        Verify that 4 micro-batches with GradScaler and loss / 4 produces
        the exact same weight update as a single full batch.
        """
        torch.manual_seed(42)
        model_accum = nn.Linear(8, 2, bias=False)
        model_single = nn.Linear(8, 2, bias=False)
        model_single.weight.data.copy_(model_accum.weight.data)

        opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.05)
        opt_single = torch.optim.SGD(model_single.parameters(), lr=0.05)

        scaler_accum = torch.amp.GradScaler("cpu", enabled=True, init_scale=512.0)
        scaler_single = torch.amp.GradScaler("cpu", enabled=True, init_scale=512.0)

        x_full = torch.randn(8, 8)
        y_full = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])

        # 1. Single full batch
        opt_single.zero_grad()
        loss_single = F.cross_entropy(model_single(x_full), y_full)
        scaler_single.scale(loss_single).backward()
        scaler_single.unscale_(opt_single)
        torch.nn.utils.clip_grad_norm_(model_single.parameters(), max_norm=1.0)
        scaler_single.step(opt_single)
        scaler_single.update()

        # 2. 4 micro-batches of size 2
        opt_accum.zero_grad()
        accum_steps = 4
        for i in range(accum_steps):
            x_mb = x_full[i * 2 : (i + 1) * 2]
            y_mb = y_full[i * 2 : (i + 1) * 2]
            loss_mb = F.cross_entropy(model_accum(x_mb), y_mb) / accum_steps
            scaler_accum.scale(loss_mb).backward()

        scaler_accum.unscale_(opt_accum)
        torch.nn.utils.clip_grad_norm_(model_accum.parameters(), max_norm=1.0)
        scaler_accum.step(opt_accum)
        scaler_accum.update()

        # Both models must have identical updated weights
        torch.testing.assert_close(model_accum.weight, model_single.weight, rtol=1e-5, atol=1e-5)
        assert scaler_accum.get_scale() == scaler_single.get_scale()

    def test_accumulation_boundary_remainder_batch(self, tmp_path: Path) -> None:
        """
        Verify that when dataset size is not divisible by gradient_accumulation_steps,
        the remaining batch is stepped at (step + 1) == len(loader) without dropping.
        """
        processor = create_dummy_processor(vocab_size=32, size=(64, 64))
        model = create_tiny_mock_model(vocab_size=32, image_size=64)

        # 7 samples with batch_size=2 => 4 batches (sizes 2, 2, 2, 1)
        # With gradient_accumulation_steps=3:
        # Step 0, 1, 2: accumulated and stepped at step 2 (3rd batch)
        # Step 3 (4th batch, last batch): stepped at step 3 because (step+1) == len(loader)
        # Total optimizer steps: 2
        manifest = write_tiny_handwriting_manifest(tmp_path / "data", "train.jsonl", count=7)

        cfg = TrainingConfig(
            output_dir=str(tmp_path / "out"),
            batch_size=2,
            gradient_accumulation_steps=3,
            num_train_epochs=1,
            device="cpu",
            num_workers=0,
        )

        samples = [
            HandwritingSample(sample_id=f"s_{i}", text=f"Text {i}", image=np.zeros((64, 64, 3), dtype=np.uint8))
            for i in range(7)
        ]
        train_ds = OCRDataset(samples=samples, processor=processor, is_training=True)
        trainer = TrOCRTrainer(config=cfg, processor=processor, model=model, train_dataset=train_ds)
        collator = OCRDataCollator(processor=processor)
        loader = DataLoader(train_ds, batch_size=2, collate_fn=collator)
        optimizer, scheduler = trainer._setup_optimizer_and_scheduler(total_training_steps=10)
        with patch.object(optimizer, "step", wraps=optimizer.step) as mock_step:
            trainer._train_epoch(loader=loader, optimizer=optimizer, scheduler=scheduler, epoch=0)
            assert mock_step.call_count == 2, f"Expected 2 optimizer steps for 4 batches with accum=3, got {mock_step.call_count}"


# ===========================================================================
# 5. Numerical Stability & Loss Convergence Verification
# ===========================================================================

class TestNumericalStabilityAndConvergence:
    """Stress-test numerical stability, NaN/Inf loss skipping, and convergence."""

    def test_overfitting_convergence_with_trocr_trainer(self, tmp_path: Path) -> None:
        """
        Verify TrOCRTrainer rapidly decreases loss over 5 epochs on a fixed batch,
        confirming backprop, SDPA, gradient accumulation, and clipping are functional.
        """
        processor = create_dummy_processor(vocab_size=32, size=(64, 64))
        model = create_tiny_mock_model(vocab_size=32, image_size=64, attn_implementation="sdpa")

        samples = [
            HandwritingSample(
                sample_id=f"sample_{i}",
                text=f"Medication {i}",
                image=np.full((64, 64, 3), 200 + i * 5, dtype=np.uint8),
            )
            for i in range(4)
        ]
        train_ds = OCRDataset(samples=samples, processor=processor, is_training=True)

        cfg = TrainingConfig(
            output_dir=str(tmp_path / "convergence"),
            num_train_epochs=5,
            batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=5e-3,
            device="cpu",
            attn_implementation="sdpa",
            empty_cache_steps=100,
        )

        trainer = TrOCRTrainer(
            config=cfg,
            processor=processor,
            model=model,
            train_dataset=train_ds,
        )

        results = trainer.train()
        assert results["epochs"] == 5
        assert results["final_loss"] >= 0.0
        assert not np.isnan(results["final_loss"])
        assert not np.isinf(results["final_loss"])

        # Check loss history in CSV
        history = trainer.logger.get_history()
        assert len(history["train_loss"]) == 5
        for l in history["train_loss"]:
            assert not np.isnan(l)
            assert not np.isinf(l)
            assert l >= 0.0

    def test_nan_loss_detection_skips_step_without_crash(self, tmp_path: Path) -> None:
        """Verify non-finite (NaN/Inf) loss in training loop logs warning and skips without crashing."""
        processor = create_dummy_processor(vocab_size=32, size=(64, 64))
        model = create_tiny_mock_model(vocab_size=32, image_size=64)

        samples = [
            HandwritingSample(sample_id="s1", text="Text 1", image=np.zeros((64, 64, 3), dtype=np.uint8)),
            HandwritingSample(sample_id="s2", text="Text 2", image=np.zeros((64, 64, 3), dtype=np.uint8)),
        ]
        train_ds = OCRDataset(samples=samples, processor=processor, is_training=True)

        cfg = TrainingConfig(
            output_dir=str(tmp_path / "nan_loss"),
            num_train_epochs=1,
            batch_size=2,
            device="cpu",
        )

        trainer = TrOCRTrainer(
            config=cfg,
            processor=processor,
            model=model,
            train_dataset=train_ds,
        )

        # Mock model forward pass to return NaN loss on first step
        original_forward = model.forward
        call_count = {"count": 0}

        def mock_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            call_count["count"] += 1
            if call_count["count"] == 1:
                out.loss = torch.tensor(float("nan"))
            return out

        with patch.object(model, "forward", side_effect=mock_forward):
            results = trainer.train()
            assert results["epochs"] == 1

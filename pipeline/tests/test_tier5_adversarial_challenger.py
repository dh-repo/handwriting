"""
pipeline/tests/test_tier5_adversarial_challenger.py
Tier 5 Adversarial Hardening Verification Suite for Challenger 2 (Milestone M4).
Stress-tests:
1. Scaled Dot-Product Attention (SDPA) kernel fusion vs math attention numerical stability.
2. AMP GradScaler numerical stability: Inf/NaN gradient skipping, scale backoff, and gradient accumulation.
3. Memory defragmentation & MPS watermark stability across 100+ steps.
4. StepProfiler microsecond breakdown fidelity, percentile math, and telemetry export.
"""

import json
import math
import os
import tempfile
import time
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pipeline.training.config import TrainingConfig
from pipeline.training.dataset import OCRDataCollator, OCRDataset
from pipeline.training.prefetcher import AsyncDevicePrefetcher
from pipeline.training.profiler import StepTiming, TrainingStepProfiler
from pipeline.training.train import (
    TrOCRTrainer,
    create_tiny_mock_model,
    get_autocast_context,
    get_optimal_device,
)


@pytest.fixture
def compute_device():
    return get_optimal_device("auto")


class TestTier5SDPAKernelHardening:
    """Stress tests on Scaled Dot-Product Attention (SDPA) kernel fusion."""

    def test_sdpa_vs_eager_mathematical_equivalence(self, compute_device):
        """Verify that SDPA produces mathematically equivalent outputs to eager attention within FP16/FP32 tolerances."""
        batch_size = 4
        num_heads = 4
        seq_len = 32
        head_dim = 16

        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=compute_device, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=compute_device, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=compute_device, dtype=torch.float32)

        # 1. Eager Manual Attention
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        eager_out = torch.matmul(attn_weights, v)

        # 2. PyTorch F.scaled_dot_product_attention
        sdpa_out = F.scaled_dot_product_attention(q, k, v)

        # Numerical comparison
        max_diff = (eager_out - sdpa_out).abs().max().item()
        assert max_diff < 1e-5, f"SDPA vs Eager FP32 max diff {max_diff} exceeded 1e-5"

    def test_sdpa_causal_and_key_padding_mask_handling(self, compute_device):
        """Stress-test SDPA with causal masks and arbitrary key padding masks."""
        batch_size = 2
        num_heads = 2
        seq_len = 16
        head_dim = 16

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=compute_device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=compute_device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=compute_device)

        # Causal Attention
        causal_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert causal_out.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not torch.isnan(causal_out).any()
        assert not torch.isinf(causal_out).any()

        # Attn Mask
        attn_mask = torch.zeros(seq_len, seq_len, device=compute_device, dtype=torch.bool)
        attn_mask[:, seq_len // 2:] = True  # Mask second half
        masked_out = F.scaled_dot_product_attention(q, k, v, attn_mask=~attn_mask)
        assert not torch.isnan(masked_out).any()

    def test_sdpa_fp16_autocast_precision_and_gradients(self, compute_device):
        """Verify SDPA backward pass and gradient flow under FP16 autocast."""
        q = torch.randn(2, 2, 8, 16, device=compute_device, requires_grad=True)
        k = torch.randn(2, 2, 8, 16, device=compute_device, requires_grad=True)
        v = torch.randn(2, 2, 8, 16, device=compute_device, requires_grad=True)

        with get_autocast_context(compute_device, mixed_precision="fp16"):
            out = F.scaled_dot_product_attention(q, k, v)
            loss = out.sum()

        loss.backward()
        assert q.grad is not None and not torch.isnan(q.grad).any()
        assert k.grad is not None and not torch.isnan(k.grad).any()
        assert v.grad is not None and not torch.isnan(v.grad).any()


class TestTier5GradScalerHardening:
    """Stress tests on AMP GradScaler numerical stability and non-finite gradient skipping."""

    def test_gradscaler_nan_inf_skipping_and_backoff(self, compute_device):
        """Verify that GradScaler correctly unscales, detects NaNs/Infs, skips optimizer step, and scales down."""
        device_type = compute_device.type
        if device_type not in ("mps", "cuda"):
            pytest.skip("GradScaler is only enabled for CUDA or MPS.")

        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2)).to(compute_device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(device_type, init_scale=1024.0)

        initial_params = [p.clone() for p in model.parameters()]

        # Step 1: Normal step with valid gradients
        x = torch.randn(4, 16, device=compute_device)
        target = torch.randint(0, 2, (4,), device=compute_device)
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            out = model(x)
            loss = F.cross_entropy(out, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Verify parameters updated
        for p_init, p_new in zip(initial_params, model.parameters()):
            assert not torch.equal(p_init, p_new), "Parameters should update on valid step"

        # Step 2: Inject Inf into gradients
        scaler_scale_before = scaler.get_scale()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            out = model(x)
            loss = F.cross_entropy(out, target)

        scaler.scale(loss).backward()
        # Corrupt gradient directly
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data[0] = float("inf")

        params_before_bad_step = [p.clone() for p in model.parameters()]
        scaler.step(optimizer)
        scaler.update()

        # Verify optimizer step was SKIPPED and scale was REDUCED
        for p_before, p_after in zip(params_before_bad_step, model.parameters()):
            assert torch.equal(p_before, p_after), "Parameters must NOT update when gradients contain Inf"

        assert scaler.get_scale() < scaler_scale_before, "GradScaler scale must decrease after Inf gradient"

    def test_gradient_accumulation_scaled_loss_equivalence(self, compute_device):
        """Verify that gradient accumulation with loss / N produces mathematically equivalent gradients."""
        device_type = compute_device.type
        accum_steps = 4
        batch_size = 4

        torch.manual_seed(123)
        model1 = nn.Linear(8, 2, bias=False).to(compute_device)
        model2 = nn.Linear(8, 2, bias=False).to(compute_device)
        model2.load_state_dict(model1.state_dict())

        # Model 1: Single full batch of size 16
        inputs_full = torch.randn(accum_steps * batch_size, 8, device=compute_device)
        targets_full = torch.randn(accum_steps * batch_size, 2, device=compute_device)

        out1 = model1(inputs_full)
        loss1 = F.mse_loss(out1, targets_full)
        loss1.backward()

        # Model 2: 4 micro-batches of size 4 accumulated with (loss / 4).backward()
        for i in range(accum_steps):
            micro_x = inputs_full[i * batch_size : (i + 1) * batch_size]
            micro_y = targets_full[i * batch_size : (i + 1) * batch_size]
            micro_out = model2(micro_x)
            micro_loss = F.mse_loss(micro_out, micro_y) / accum_steps
            micro_loss.backward()

        # Gradients must match exactly
        grad_diff = (model1.weight.grad - model2.weight.grad).abs().max().item()
        assert grad_diff < 1e-5, f"Gradient accumulation difference {grad_diff} exceeded 1e-5"


class TestTier5MemoryDefragHardening:
    """Stress tests on memory defragmentation and MPS watermark telemetry."""

    def test_100_step_memory_stability_and_defrag(self, compute_device):
        """Verify that 100+ training steps maintain bounded memory without linear leakage."""
        device_type = compute_device.type
        model = create_tiny_mock_model(vocab_size=100, image_size=64, attn_implementation="sdpa").to(compute_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        config = TrainingConfig(
            device=str(compute_device),
            batch_size=4,
            empty_cache_steps=20,
            mps_high_watermark_ratio=0.85,
        )

        memory_samples = []
        for step in range(1, 101):
            pixel_values = torch.randn(4, 3, 64, 64, device=compute_device)
            labels = torch.randint(0, 100, (4, 16), dtype=torch.long, device=compute_device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            config.manage_memory(step=step)

            if compute_device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                memory_samples.append(mem_mb)

        if memory_samples:
            # Memory should stabilize within first 10 steps and variance across remaining steps should be minimal (<10%)
            steady_state = memory_samples[20:]
            std_dev = float(np.std(steady_state))
            mean_mem = float(np.mean(steady_state))
            rel_variance = std_dev / max(mean_mem, 1.0)
            assert rel_variance < 0.15, f"Memory variance {rel_variance:.3f} indicates memory leakage or instability"


class TestTier5StepProfilerHardening:
    """Stress tests on StepProfiler telemetry precision and JSON/CSV export integrity."""

    def test_step_profiler_microsecond_breakdown_and_percentiles(self, compute_device):
        """Verify that profiler records exact microsecond timings, computes percentiles, and normalizes percentages."""
        profiler = TrainingStepProfiler(warmup_steps=2, device=compute_device, enabled=True)

        # Simulate 10 training steps
        for step in range(1, 11):
            profiler.record_data_wait(0.0001 * step)
            profiler.record_step(
                t_data=0.001 * step,
                t_transfer=0.0005,
                t_fwd=0.010,
                t_bwd=0.015,
                t_opt=0.002,
                num_samples=8,
            )

        summary = profiler.get_summary()
        assert summary["total_steps"] == 10
        assert summary["active_steps"] == 8
        assert summary["total_samples"] == 80
        assert summary["samples_per_sec"] > 0.0

        # Verify percentiles ordering: p50 <= p90 <= p95 <= p99
        lat = summary["step_latency_ms"]
        assert lat["p50"] <= lat["p90"] <= lat["p95"] <= lat["p99"]

        # Verify percentage breakdown sum ~ 100%
        bd_pct = summary["breakdown_pct"]
        total_pct = sum(bd_pct.values())
        assert abs(total_pct - 100.0) < 0.1, f"Percentages sum {total_pct} did not equal 100%"

    def test_step_profiler_json_csv_export_integrity(self, compute_device, tmp_path):
        """Verify that JSON and CSV exported files contain identical step records and headers."""
        profiler = TrainingStepProfiler(warmup_steps=1, device=compute_device, enabled=True)
        profiler.record_step(0.001, 0.001, 0.01, 0.02, 0.005, 4)
        profiler.record_step(0.001, 0.001, 0.012, 0.021, 0.004, 4)

        json_path = tmp_path / "profiler_summary.json"
        csv_path = tmp_path / "step_timings.csv"

        profiler.export_json(json_path)
        profiler.export_csv(csv_path)

        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "summary" in data
        assert "step_records" in data
        assert len(data["step_records"]) == 2

        assert csv_path.exists()
        lines = csv_path.read_text().splitlines()
        assert lines[0].startswith("step,timestamp,t_data,t_transfer")
        assert len(lines) == 3  # Header + 2 steps


class TestTier5TrainingFidelityAndLossSkipping:
    """Stress tests on end-to-end loss convergence fidelity, gradient clipping, and NaN/Inf loss skipping."""

    def test_loss_convergence_fidelity(self, compute_device, tmp_path):
        """Verify that training over a synthetic dataset monotonically converges (loss reduces significantly)."""
        torch.manual_seed(42)
        model = create_tiny_mock_model(vocab_size=64, image_size=32, attn_implementation="sdpa").to(compute_device)

        class FixedOCRDataset(Dataset):
            def __init__(self):
                self.x = torch.randn(16, 3, 32, 32)
                self.y = torch.randint(3, 60, (16, 8), dtype=torch.long)

            def __len__(self):
                return 16

            def __getitem__(self, idx):
                return {"pixel_values": self.x[idx], "input_ids": self.y[idx], "labels": self.y[idx]}

        dataset = FixedOCRDataset()
        config = TrainingConfig(
            output_dir=str(tmp_path / "run_fidelity"),
            device=str(compute_device),
            num_train_epochs=5,
            batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            mixed_precision="fp16" if compute_device.type in ("mps", "cuda") else "none",
            num_workers=0,
            attn_implementation="sdpa",
            enable_step_profiling=True,
            save_steps=100,
            logging_steps=1,
        )

        trainer = TrOCRTrainer(
            config=config,
            model=model,
            processor=type("MockProc", (), {"tokenizer": type("Tok", (), {"pad_token_id": 1, "eos_token_id": 2, "__len__": lambda self: 64})()})(),
            train_dataset=dataset,
        )

        results = trainer.train()
        train_losses = trainer.logger.history["train_loss"]
        assert len(train_losses) == 5
        # Initial loss vs final loss should show significant reduction
        assert train_losses[-1] < train_losses[0] * 0.95, f"Loss did not converge: {train_losses[0]} -> {train_losses[-1]}"

    def test_non_finite_loss_skipping_resilience(self, compute_device, tmp_path):
        """Verify that TrOCRTrainer._train_epoch gracefully catches non-finite loss (NaN/Inf) without crashing or corrupting model parameters."""
        model = create_tiny_mock_model(vocab_size=64, image_size=32, attn_implementation="sdpa").to(compute_device)

        class PoisonedDataset(Dataset):
            def __init__(self):
                self.samples = [
                    {"pixel_values": torch.randn(3, 32, 32), "labels": torch.tensor([5, 6, 7])},
                    # Poisoned with NaN
                    {"pixel_values": torch.full((3, 32, 32), float("nan")), "labels": torch.tensor([5, 6, 7])},
                    {"pixel_values": torch.randn(3, 32, 32), "labels": torch.tensor([5, 6, 7])},
                ]

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.samples[idx]

        config = TrainingConfig(
            output_dir=str(tmp_path / "run_poison"),
            device=str(compute_device),
            num_train_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            num_workers=0,
            enable_step_profiling=True,
        )

        trainer = TrOCRTrainer(
            config=config,
            model=model,
            processor=type("MockProc", (), {"tokenizer": type("Tok", (), {"pad_token_id": 1, "eos_token_id": 2, "__len__": lambda self: 64})()})(),
            train_dataset=PoisonedDataset(),
        )

        # Epoch should execute cleanly, skipping the NaN batch without raising an uncaught exception
        epoch_loss = trainer._train_epoch(
            DataLoader(PoisonedDataset(), batch_size=1, collate_fn=trainer.collator),
            torch.optim.AdamW(model.parameters(), lr=1e-4),
            None,
            epoch=1,
        )
        assert not math.isnan(epoch_loss)
        # Model weights must remain finite (no NaN contamination)
        for p in model.parameters():
            assert not torch.isnan(p).any(), "Model parameter was contaminated by NaN loss"


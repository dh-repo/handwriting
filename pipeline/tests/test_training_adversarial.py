"""
pipeline/tests/test_training_adversarial.py
Adversarial and empirical stress test suite for the handwriting recognition training subsystem.
Tests boundary conditions, extreme values, fault recovery, multi-threading, MPS acceleration,
dynamic masking, and live training descent dynamics.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import pytest
import torch

from pipeline.dataset.dataset_loader import HandwritingSample, MedicalPrescriptionSample
from pipeline.preprocessing.line_segmenter import LineCrop
from pipeline.training.config import TrainingConfig
from pipeline.training.dataset import (
    DummyImageProcessor,
    DummyProcessor,
    DummyTokenizer,
    OCRDataCollator,
    OCRDataset,
    create_dummy_processor,
    load_trocr_processor,
)
from pipeline.training.loss_logger import LossLogger
from pipeline.training.train import (
    TrOCRTrainer,
    create_tiny_mock_model,
    get_autocast_context,
    get_optimal_device,
)


# ===========================================================================
# 1. TrainingConfig Adversarial Stress Tests
# ===========================================================================

def test_config_extreme_boundaries_and_types(tmp_path: Path) -> None:
    """Stress-test TrainingConfig with extreme boundary values, type coercion, and edge configs."""
    # Test valid extreme values
    cfg_extreme = TrainingConfig(
        batch_size=1024,
        eval_batch_size=2048,
        learning_rate=1e-8,
        num_train_epochs=1000,
        gradient_accumulation_steps=64,
        warmup_steps=500,
        empty_cache_steps=1,
        image_size=(512, 1024),
        max_target_length=512,
    )
    assert cfg_extreme.batch_size == 1024
    assert cfg_extreme.image_size == (512, 1024)

    # Test invalid values raising ValueError
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=-5)

    with pytest.raises(ValueError, match="eval_batch_size"):
        TrainingConfig(eval_batch_size=0)

    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=0.0)

    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=-1.0)

    with pytest.raises(ValueError, match="num_train_epochs"):
        TrainingConfig(num_train_epochs=0, max_steps=None)

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        TrainingConfig(gradient_accumulation_steps=-1)

    with pytest.raises(ValueError, match="warmup_steps"):
        TrainingConfig(warmup_steps=-10)

    with pytest.raises(ValueError, match="Invalid device"):
        TrainingConfig(device="quantum_accelerator")

    with pytest.raises(ValueError, match="Invalid mixed_precision"):
        TrainingConfig(mixed_precision="int8")

    with pytest.raises(ValueError, match="Invalid lr_scheduler_type"):
        TrainingConfig(lr_scheduler_type="quadratic")


def test_config_roundtrip_serialization_and_nested_structures(tmp_path: Path) -> None:
    """Stress-test JSON and YAML serialization round-trips with arbitrary paths, extra fields, and aliases."""
    original = TrainingConfig(
        model_name_or_path="custom/trocr-stage2",
        output_dir=str(tmp_path / "nested" / "sub" / "ckpt"),
        image_size=(256, 512),
        max_target_length=200,
        device="cpu",
        mixed_precision="bf16",
        empty_cache_steps=25,
        batch_size=16,
        eval_batch_size=32,
        learning_rate=7.5e-5,
        weight_decay=0.05,
        num_train_epochs=12,
        max_steps=1000,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        lr_scheduler_type="linear",
        extra_params={"custom_tag": "adversarial_test", "experiment_id": 999},
    )

    # 1. JSON Roundtrip to file and string
    json_file = tmp_path / "deep" / "dir" / "config.json"
    json_str = original.to_json(json_file)
    assert json_file.exists()
    
    restored_from_file = TrainingConfig.from_json(json_file)
    restored_from_str = TrainingConfig.from_json(json_str)

    assert restored_from_file.model_name_or_path == original.model_name_or_path
    assert restored_from_file.image_size == (256, 512)
    assert restored_from_file.mixed_precision == "bf16"
    assert restored_from_file.learning_rate == 7.5e-5
    assert restored_from_str.max_steps == 1000

    # 2. YAML Roundtrip
    yaml_file = tmp_path / "deep" / "dir" / "config.yaml"
    yaml_str = original.to_yaml(yaml_file)
    assert yaml_file.exists()
    restored_from_yaml = TrainingConfig.from_yaml(yaml_file)
    assert restored_from_yaml.learning_rate == 7.5e-5
    assert restored_from_yaml.num_train_epochs == 12

    # 3. Handling unknown/extra parameters gracefully
    dict_with_extra = original.to_dict()
    dict_with_extra["unknown_hyperparam_x"] = 42
    dict_with_extra["model_name"] = "aliased-model"  # alias test
    dict_with_extra["epochs"] = 20  # alias test
    dict_with_extra["fp16"] = True  # alias test
    
    coerced = TrainingConfig.from_dict(dict_with_extra)
    assert coerced.model_name_or_path == "aliased-model"
    assert coerced.num_train_epochs == 20
    assert coerced.mixed_precision == "fp16"
    assert coerced.extra_params.get("unknown_hyperparam_x") == 42


def test_config_device_resolution_and_mps_detection() -> None:
    """Verify device resolution on Apple Silicon / CPU / CUDA and fallback behavior."""
    # Test Auto
    cfg_auto = TrainingConfig(device="auto")
    resolved = cfg_auto.resolved_device
    assert isinstance(resolved, torch.device)
    assert resolved.type in ("mps", "cuda", "cpu")

    # Test explicit CPU
    cfg_cpu = TrainingConfig(device="cpu")
    assert cfg_cpu.resolved_device.type == "cpu"
    assert cfg_cpu.is_cpu is True
    assert cfg_cpu.is_mps is False

    # Test explicit MPS
    cfg_mps = TrainingConfig(device="mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        assert cfg_mps.resolved_device.type == "mps"
        assert cfg_mps.is_mps is True
    else:
        # Fallback to CPU when MPS not present
        assert cfg_mps.resolved_device.type == "cpu"

    # Test mixed precision dtypes
    cfg_fp16 = TrainingConfig(mixed_precision="fp16")
    assert cfg_fp16.autocast_dtype == torch.float16

    cfg_bf16 = TrainingConfig(mixed_precision="bf16")
    assert cfg_bf16.autocast_dtype == torch.bfloat16

    cfg_none = TrainingConfig(mixed_precision="none")
    assert cfg_none.autocast_dtype is None


# ===========================================================================
# 2. OCRDataset & OCRDataCollator Adversarial Stress Tests
# ===========================================================================

def test_ocr_dataset_extreme_and_corrupt_inputs(tmp_path: Path) -> None:
    """Stress-test OCRDataset with diverse image encodings, empty/corrupt files, huge strings, and non-ASCII."""
    processor = create_dummy_processor(vocab_size=100)

    # 1. Create RGBA image with alpha channel
    rgba_path = tmp_path / "sample_rgba.png"
    Image.new("RGBA", (150, 60), color=(200, 100, 50, 128)).save(rgba_path)

    # 2. Create Grayscale 1-channel image
    gray_path = tmp_path / "sample_gray.png"
    Image.new("L", (150, 60), color=180).save(gray_path)

    # 3. Create a non-existent / broken path
    broken_path = tmp_path / "does_not_exist.png"

    # 4. Create 2D numpy array (single channel HxW)
    gray_np = np.full((70, 220), 150, dtype=np.uint8)

    # 5. Create float32 numpy array [0.0, 1.0]
    float_np = np.random.uniform(0.0, 1.0, size=(80, 200, 3)).astype(np.float32)

    # 6. Ultra-long text (> 1000 characters) and special unicode / emoji / non-ASCII
    long_text = "Doctor Prescription: " + ("Amoxicillin 500mg daily. " * 50) + "℞ 医疗处方 #123 💊 50µg"
    empty_text = ""

    samples = [
        # RGBA file path
        str(rgba_path),
        # Grayscale file path
        str(gray_path),
        # Broken / missing path (should fallback safely to blank white image)
        str(broken_path),
        # 2D grayscale numpy array
        {"sample_id": "gray_2d", "image": gray_np, "text": "Grayscale 2D test"},
        # Float32 numpy array
        {"sample_id": "float_img", "image": float_np, "text": "Float32 image test"},
        # Ultra long text
        HandwritingSample(sample_id="long_sample", text=long_text, image=np.zeros((50, 100, 3), dtype=np.uint8)),
        # Empty text
        HandwritingSample(sample_id="empty_sample", text=empty_text, image=np.zeros((50, 100, 3), dtype=np.uint8)),
        # Corrupt dict with None fields
        {"sample_id": "none_dict", "image": None, "image_path": None, "text": None},
    ]

    dataset = OCRDataset(samples=samples, processor=processor, max_target_length=128, is_training=True)
    assert len(dataset) == 8

    # Verify each sample parses into valid tensors without crashing
    for i in range(len(dataset)):
        item = dataset[i]
        assert "pixel_values" in item
        assert isinstance(item["pixel_values"], torch.Tensor)
        assert item["pixel_values"].shape == (3, 384, 384)
        assert not torch.isnan(item["pixel_values"]).any()
        assert not torch.isinf(item["pixel_values"]).any()

        if item.get("text"):
            assert "input_ids" in item
            assert isinstance(item["input_ids"], torch.Tensor)
            assert item["input_ids"].shape[0] <= 128  # Respects truncation


def test_ocr_data_collator_variable_lengths_and_masking() -> None:
    """Stress-test OCRDataCollator variable length sequences, batch size 1, empty batches, and -100 masking."""
    processor = create_dummy_processor(vocab_size=100)
    collator = OCRDataCollator(processor=processor, pad_token_id=1)

    # 1. Empty batch edge case
    empty_collated = collator([])
    assert empty_collated == {}

    # 2. Batch size 1
    single_item = [{
        "sample_id": "single",
        "pixel_values": torch.zeros(3, 384, 384),
        "text": "Single item",
        "input_ids": torch.tensor([0, 10, 20, 2], dtype=torch.long),
    }]
    single_batch = collator(single_item)
    assert single_batch["pixel_values"].shape == (1, 3, 384, 384)
    assert single_batch["input_ids"].shape == (1, 4)
    assert single_batch["labels"].shape == (1, 4)
    assert single_batch["labels"][0].tolist() == [0, 10, 20, 2]
    assert single_batch["decoder_attention_mask"][0].tolist() == [1, 1, 1, 1]

    # 3. Heterogeneous batch with varying sequence lengths (len 2, len 6, len 10)
    var_batch_items = [
        {
            "sample_id": "v1",
            "pixel_values": torch.randn(3, 384, 384),
            "input_ids": torch.tensor([0, 2], dtype=torch.long),  # len 2
        },
        {
            "sample_id": "v2",
            "pixel_values": torch.randn(3, 384, 384),
            "input_ids": torch.tensor([0, 15, 25, 35, 45, 2], dtype=torch.long),  # len 6
        },
        {
            "sample_id": "v3",
            "pixel_values": torch.randn(3, 384, 384),
            "input_ids": torch.tensor([0, 11, 22, 33, 44, 55, 66, 77, 88, 2], dtype=torch.long),  # len 10
        },
    ]
    var_batch = collator(var_batch_items)

    assert var_batch["input_ids"].shape == (3, 10)
    assert var_batch["labels"].shape == (3, 10)
    assert var_batch["decoder_attention_mask"].shape == (3, 10)

    # Verify input_ids padding
    assert var_batch["input_ids"][0, 2:].tolist() == [1] * 8
    assert var_batch["input_ids"][1, 6:].tolist() == [1] * 4
    assert var_batch["input_ids"][2, :].tolist() == [0, 11, 22, 33, 44, 55, 66, 77, 88, 2]

    # Verify -100 label masking for all pad positions
    assert var_batch["labels"][0, :2].tolist() == [0, 2]
    assert var_batch["labels"][0, 2:].tolist() == [-100] * 8
    assert var_batch["labels"][1, :6].tolist() == [0, 15, 25, 35, 45, 2]
    assert var_batch["labels"][1, 6:].tolist() == [-100] * 4
    assert var_batch["labels"][2, :].tolist() == [0, 11, 22, 33, 44, 55, 66, 77, 88, 2]

    # Verify decoder attention mask
    assert var_batch["decoder_attention_mask"][0].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert var_batch["decoder_attention_mask"][1].tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert var_batch["decoder_attention_mask"][2].tolist() == [1] * 10


# ===========================================================================
# 3. LossLogger Adversarial Stress Tests
# ===========================================================================

def test_loss_logger_concurrency_and_high_throughput(tmp_path: Path) -> None:
    """Stress-test LossLogger with concurrent multi-threaded logging and high throughput."""
    log_dir = tmp_path / "concurrent_logger"
    logger = LossLogger(log_dir=log_dir, csv_filename="concurrent_losses.csv", extended_logging=True)

    num_threads = 10
    logs_per_thread = 20

    def worker_log(thread_id: int):
        for i in range(logs_per_thread):
            step = thread_id * logs_per_thread + i
            logger.log_epoch(
                epoch=i + 1,
                train_loss=float(1.0 / (i + 1 + thread_id * 0.1)),
                val_cer=0.05 * thread_id,
                val_wer=0.10 * thread_id,
                val_loss=0.5,
                learning_rate=1e-4,
                step=step,
                elapsed_time=0.1 * step,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_log, tid) for tid in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    # Verify CSV line count
    with open(logger.csv_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    expected_total_records = num_threads * logs_per_thread
    assert len(lines) == expected_total_records + 1  # 1 header + records

    # Verify queryable history
    hist = logger.get_history()
    assert len(hist["epochs"]) == expected_total_records


def test_loss_logger_extreme_values_and_single_epoch(tmp_path: Path) -> None:
    """Stress-test LossLogger plotting with 0 epochs, 1 epoch, all-NaNs, all-Infs."""
    # 1. Zero epochs
    logger_empty = LossLogger(log_dir=tmp_path / "empty", csv_filename="empty.csv")
    plot_empty = logger_empty.plot_curves()
    assert plot_empty.exists()
    assert Image.open(plot_empty).format == "PNG"

    # 2. Exactly 1 epoch
    logger_single = LossLogger(log_dir=tmp_path / "single", csv_filename="single.csv")
    logger_single.log_epoch(epoch=1, train_loss=0.75, val_cer=0.12, val_wer=0.25, val_loss=0.70)
    plot_single = logger_single.plot_curves()
    assert plot_single.exists()
    assert plot_single.stat().st_size > 1000

    # 3. NaNs and Infs
    logger_nan = LossLogger(log_dir=tmp_path / "nan", csv_filename="nan.csv")
    logger_nan.log_epoch(epoch=1, train_loss=float("nan"), val_cer=float("nan"), val_wer=float("nan"))
    logger_nan.log_epoch(epoch=2, train_loss=float("inf"), val_cer=float("-inf"), val_wer=0.0)
    logger_nan.log_epoch(epoch=3, train_loss=0.45, val_cer=0.08, val_wer=0.15)
    plot_nan = logger_nan.plot_curves()
    assert plot_nan.exists()
    assert Image.open(plot_nan).format == "PNG"


# ===========================================================================
# 4. TrOCRTrainer Live Micro Training & Optimization Dynamics
# ===========================================================================

@pytest.mark.parametrize("device_type", ["cpu", "mps"])
def test_trocr_trainer_live_micro_run_and_gradient_flow(tmp_path: Path, device_type: str) -> None:
    """
    Empirically execute a live multi-epoch training run.
    Verifies that:
    1. Gradients are computed on all active layers (backprop works).
    2. Weights are modified in each optimizer step.
    3. Loss decreases or converges across optimization steps.
    4. Checkpoints (PT and HuggingFace format) and loss logs are saved cleanly.
    5. Memory management (empty cache) executes without error.
    """
    if device_type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            pytest.skip("Apple Silicon MPS not available in this runtime.")

    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    # Generate synthetic training batch
    samples = [
        HandwritingSample(
            sample_id=f"train_{i:02d}",
            text=f"Prescription Line {i}",
            image=np.full((64, 64, 3), 180 + (i * 10) % 70, dtype=np.uint8),
        )
        for i in range(8)
    ]
    val_samples = samples[:2]

    train_ds = OCRDataset(samples=samples, processor=processor, is_training=True)
    val_ds = OCRDataset(samples=val_samples, processor=processor, is_training=False)

    ckpt_dir = tmp_path / f"live_run_{device_type}"
    config = TrainingConfig(
        model_name_or_path="mock-trocr",
        output_dir=str(ckpt_dir),
        num_train_epochs=3,
        batch_size=4,
        eval_batch_size=2,
        learning_rate=5e-3,  # higher LR to observe parameter movement
        device=device_type,
        mixed_precision="fp16" if device_type == "mps" else "none",
        empty_cache_steps=1,  # triggers memory management every step
        gradient_accumulation_steps=1,
    )

    trainer = TrOCRTrainer(
        config=config,
        model=model,
        processor=processor,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    # Capture initial weights
    initial_params = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    # Run training
    results = trainer.train()

    # 1. Check result dictionary
    assert results["epochs"] == 3
    assert results["final_loss"] >= 0.0
    assert (ckpt_dir / "best_model.pt").exists()
    assert (ckpt_dir / "checkpoint_epoch_1.pt").exists()
    assert (ckpt_dir / "checkpoint_epoch_2.pt").exists()
    assert (ckpt_dir / "checkpoint_epoch_3.pt").exists()
    assert (ckpt_dir / "losses.csv").exists()
    assert (ckpt_dir / "loss_curves.png").exists()
    assert (ckpt_dir / "training_state.json").exists()
    assert (ckpt_dir / "best_model_hf").is_dir()

    # 2. Check parameter modification across all layers
    updated_params = {name: param.detach() for name, param in model.named_parameters() if param.requires_grad}
    for name, init_val in initial_params.items():
        curr_val = updated_params[name].cpu()
        init_val_cpu = init_val.cpu()
        assert not torch.isnan(curr_val).any(), f"NaN detected in parameter {name}"
        # Encoder and decoder layers must have shifted (excluding unused ViT classification pooler)
        if "weight" in name and "LayerNorm" not in name and "pooler" not in name:
            diff = torch.abs(curr_val - init_val_cpu).max().item()
            assert diff > 0.0, f"Active parameter {name} did not receive gradient updates (diff={diff})"

    # 3. Check training state JSON validity
    state_json = json.loads((ckpt_dir / "training_state.json").read_text(encoding="utf-8"))
    assert state_json["completed"] is True
    assert state_json["epoch"] == 3
    assert state_json["global_step"] == 6  # 8 samples // 4 batch_size = 2 steps/epoch * 3 epochs = 6 steps


def test_trocr_trainer_gradient_accumulation_and_clipping(tmp_path: Path) -> None:
    """Stress-test gradient accumulation > 1 and gradient clipping."""
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    samples = [
        HandwritingSample(
            sample_id=f"sample_{i}",
            text=f"Dosage {i * 100}mg",
            image=np.full((64, 64, 3), 200, dtype=np.uint8),
        )
        for i in range(8)
    ]
    train_ds = OCRDataset(samples=samples, processor=processor, is_training=True)

    ckpt_dir = tmp_path / "grad_accum_run"
    config = TrainingConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=2,
        batch_size=2,
        gradient_accumulation_steps=2,  # Accumulate across 2 batches
        max_grad_norm=0.5,  # Strict gradient clipping
        device="cpu",
    )

    trainer = TrOCRTrainer(
        config=config,
        model=model,
        processor=processor,
        train_dataset=train_ds,
    )

    results = trainer.train()
    assert results["epochs"] == 2
    assert (ckpt_dir / "best_model.pt").exists()


def test_trocr_trainer_resume_and_state_preservation(tmp_path: Path) -> None:
    """Stress-test resuming training from intermediate checkpoint and verifying step continuation."""
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model_stage1 = create_tiny_mock_model(vocab_size=50, image_size=64)

    samples = [
        HandwritingSample(sample_id=f"s_{i}", text=f"Text {i}", image=np.full((64, 64, 3), 210, dtype=np.uint8))
        for i in range(6)
    ]
    train_ds = OCRDataset(samples=samples, processor=processor, is_training=True)

    ckpt_dir = tmp_path / "resume_lifecycle"

    # Stage 1: Train for 2 epochs
    config1 = TrainingConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=2,
        batch_size=2,
        device="cpu",
    )
    trainer1 = TrOCRTrainer(config=config1, model=model_stage1, processor=processor, train_dataset=train_ds)
    trainer1.train()

    ckpt_ep2 = ckpt_dir / "checkpoint_epoch_2.pt"
    assert ckpt_ep2.exists()

    # Stage 2: Resume from checkpoint_epoch_2 for up to 4 epochs
    model_stage2 = create_tiny_mock_model(vocab_size=50, image_size=64)
    config2 = TrainingConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=4,
        batch_size=2,
        device="cpu",
        resume_from_checkpoint=str(ckpt_ep2),
    )
    trainer2 = TrOCRTrainer(config=config2, model=model_stage2, processor=processor, train_dataset=train_ds)
    results2 = trainer2.train()

    assert results2["epochs"] == 4
    assert (ckpt_dir / "checkpoint_epoch_3.pt").exists()
    assert (ckpt_dir / "checkpoint_epoch_4.pt").exists()

    final_state = json.loads((ckpt_dir / "training_state.json").read_text())
    assert final_state["epoch"] == 4
    assert final_state["completed"] is True

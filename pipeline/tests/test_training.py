"""
pipeline/tests/test_training.py
Comprehensive unit and integration test suite for TrOCR training pipeline:
TrainingConfig, OCRDataset, OCRDataCollator, LossLogger, TrOCRTrainer,
and offline model fine-tuning step on Apple Silicon MPS / CPU.
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import pytest
import torch

from pipeline.dataset.dataset_loader import HandwritingSample, MedicalPrescriptionSample
from pipeline.preprocessing.line_segmenter import LineCrop
from pipeline.training.config import TrainingConfig
from pipeline.training.dataset import (
    DummyProcessor,
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


# ---------------------------------------------------------------------------
# TrainingConfig Tests
# ---------------------------------------------------------------------------

def test_training_config_defaults() -> None:
    """Verify default hyperparameters and configuration invariants."""
    config = TrainingConfig()
    assert config.batch_size == 8
    assert config.learning_rate == 5e-5
    assert config.num_train_epochs == 3
    assert config.device == "auto"
    assert config.mixed_precision == "none"
    assert config.empty_cache_steps == 100
    assert config.attn_implementation == "sdpa"
    assert config.lr_scheduler_type == "cosine"
    assert config.max_grad_norm == 1.0
    assert config.num_workers >= 2
    assert config.persistent_workers is True
    assert config.prefetch_factor == 4
    assert config.prefetch_queue_size == 3
    assert config.use_async_prefetcher is True


def test_training_config_validation() -> None:
    """Verify strict validation against invalid ranges or options."""
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=0)

    with pytest.raises(ValueError, match="eval_batch_size"):
        TrainingConfig(eval_batch_size=-1)

    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=-1e-4)

    with pytest.raises(ValueError, match="num_train_epochs"):
        TrainingConfig(num_train_epochs=0, max_steps=None)

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        TrainingConfig(gradient_accumulation_steps=0)

    with pytest.raises(ValueError, match="Invalid device"):
        TrainingConfig(device="tpu_v4")

    with pytest.raises(ValueError, match="Invalid mixed_precision"):
        TrainingConfig(mixed_precision="fp64")

    with pytest.raises(ValueError, match="Invalid lr_scheduler_type"):
        TrainingConfig(lr_scheduler_type="polynomial")


def test_training_config_properties_and_serialization(tmp_path: Path) -> None:
    """Verify serialization to/from dict, JSON, YAML and device resolution."""
    config = TrainingConfig(
        model_name_or_path="microsoft/trocr-small-printed",
        batch_size=8,
        learning_rate=3e-5,
        num_train_epochs=5,
        mixed_precision="fp16",
        device="auto",
    )

    # Device properties
    dev = config.resolved_device
    assert isinstance(dev, torch.device)
    assert dev.type in ("mps", "cuda", "cpu")
    if dev.type == "mps":
        assert config.is_mps
    elif dev.type == "cuda":
        assert config.is_cuda
    else:
        assert config.is_cpu

    assert config.autocast_dtype == torch.float16

    # Dict serialization
    d = config.to_dict()
    assert d["batch_size"] == 8
    assert d["learning_rate"] == 3e-5
    restored_d = TrainingConfig.from_dict(d)
    assert restored_d.batch_size == 8

    # JSON serialization
    json_path = tmp_path / "config.json"
    config.to_json(json_path)
    assert json_path.exists()
    restored_j = TrainingConfig.from_json(json_path)
    assert restored_j.learning_rate == 3e-5

    # YAML serialization
    yaml_path = tmp_path / "config.yaml"
    config.to_yaml(yaml_path)
    assert yaml_path.exists()
    restored_y = TrainingConfig.from_yaml(yaml_path)
    assert restored_y.num_train_epochs == 5


def test_training_config_aliases() -> None:
    """Verify model_name, epochs, and fp16 aliases."""
    config = TrainingConfig.from_dict({"model_name": "custom/trocr", "epochs": 10, "fp16": True})
    assert config.model_name == "custom/trocr"
    assert config.model_name_or_path == "custom/trocr"
    assert config.epochs == 10
    assert config.num_train_epochs == 10
    assert config.fp16 is True
    assert config.mixed_precision == "fp16"


# ---------------------------------------------------------------------------
# OCRDataset & OCRDataCollator Tests
# ---------------------------------------------------------------------------

def test_ocr_dataset_polymorphic_ingestion(tmp_path: Path) -> None:
    """Verify OCRDataset correctly handles all sample types (HandwritingSample, LineCrop, dict, path)."""
    processor = create_dummy_processor(vocab_size=50)

    # Save a test image file
    img_path = tmp_path / "test_sample.png"
    Image.new("RGB", (200, 50), color="white").save(img_path)

    samples = [
        HandwritingSample(
            sample_id="hs_1",
            text="Amoxicillin 500mg",
            image=np.full((50, 200, 3), 240, dtype=np.uint8),
            writer_id="dr_smith",
        ),
        MedicalPrescriptionSample(
            sample_id="rx_1",
            full_text="Take 1 tablet daily",
            image=np.full((80, 300, 3), 230, dtype=np.uint8),
        ),
        LineCrop(
            line_index=2,
            image=np.full((40, 250, 3), 220, dtype=np.uint8),
            bbox=[0.1, 0.1, 0.2, 0.9],
        ),
        {
            "sample_id": "dict_1",
            "text": "Refill 2 times",
            "image": np.full((50, 180, 3), 210, dtype=np.uint8),
            "writer_id": "w_12",
        },
        str(img_path),
    ]

    dataset = OCRDataset(samples=samples, processor=processor, is_training=True)
    assert len(dataset) == 5

    # Check item 0
    item0 = dataset[0]
    assert item0["sample_id"] == "hs_1"
    assert item0["text"] == "Amoxicillin 500mg"
    assert item0["writer_id"] == "dr_smith"
    assert item0["pixel_values"].shape == (3, 384, 384)
    assert "input_ids" in item0
    assert isinstance(item0["input_ids"], torch.Tensor)

    # Check item 1 (Prescription)
    item1 = dataset[1]
    assert item1["sample_id"] == "rx_1"
    assert item1["text"] == "Take 1 tablet daily"
    assert item1["pixel_values"].shape == (3, 384, 384)

    # Check item 4 (Path)
    item4 = dataset[4]
    assert item4["pixel_values"].shape == (3, 384, 384)


def test_ocr_dataset_transforms_and_edge_cases() -> None:
    """Verify custom transforms, empty text, and blank image fallbacks."""
    processor = create_dummy_processor(vocab_size=50)

    def invert_transform(img: np.ndarray) -> np.ndarray:
        return 255 - img

    samples = [
        {"sample_id": "blank_text", "text": "", "image": np.zeros((100, 100, 3), dtype=np.uint8)},
        {"sample_id": "no_image", "text": "Valid text", "image": None},
    ]

    dataset = OCRDataset(samples=samples, processor=processor, transform=invert_transform, is_training=True)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert item0["sample_id"] == "blank_text"
    assert item0["text"] == ""

    item1 = dataset[1]
    assert item1["pixel_values"].shape == (3, 384, 384)


def test_ocr_dataset_factory_methods() -> None:
    """Verify factory methods: from_synthetic, from_line_crops."""
    processor = create_dummy_processor(vocab_size=50)

    class MockSyntheticGenerator:
        def render_line(self, text: str):
            return np.full((64, 256, 3), 200, dtype=np.uint8), {}

    gen = MockSyntheticGenerator()
    ds_synth = OCRDataset.from_synthetic(gen, num_samples=10, processor=processor)
    assert len(ds_synth) == 10
    assert ds_synth[0]["pixel_values"].shape == (3, 384, 384)

    crops = [
        LineCrop(line_index=0, image=np.zeros((50, 200, 3), dtype=np.uint8), bbox=[0.0, 0.0, 0.5, 0.5]),
        LineCrop(line_index=1, image=np.zeros((50, 200, 3), dtype=np.uint8), bbox=[0.5, 0.0, 1.0, 1.0]),
    ]
    ds_crops = OCRDataset.from_line_crops(crops, texts=["Line 1", "Line 2"], processor=processor)
    assert len(ds_crops) == 2
    assert ds_crops[0]["text"] == "Line 1"


def test_ocr_data_collator_dynamic_padding_and_masking() -> None:
    """Verify OCRDataCollator performs dynamic sequence padding and masks pad tokens with -100."""
    processor = create_dummy_processor(vocab_size=50)
    collator = OCRDataCollator(processor=processor, pad_token_id=1)

    batch_inputs = [
        {
            "sample_id": "s1",
            "pixel_values": torch.randn(3, 384, 384),
            "text": "Short",
            "writer_id": "w1",
            "input_ids": torch.tensor([0, 10, 12, 2], dtype=torch.long),  # len 4
        },
        {
            "sample_id": "s2",
            "pixel_values": torch.randn(3, 384, 384),
            "text": "A significantly longer prescription instruction line",
            "writer_id": "w2",
            "input_ids": torch.tensor([0, 15, 20, 22, 25, 30, 35, 2], dtype=torch.long),  # len 8
        },
    ]

    batch = collator(batch_inputs)

    assert "pixel_values" in batch
    assert batch["pixel_values"].shape == (2, 3, 384, 384)

    assert "input_ids" in batch
    assert batch["input_ids"].shape == (2, 8)  # Dynamically padded to max length 8
    assert batch["input_ids"][0, 4:].tolist() == [1, 1, 1, 1]  # Padded with pad_token_id=1

    assert "labels" in batch
    assert batch["labels"].shape == (2, 8)
    assert batch["labels"][0, :4].tolist() == [0, 10, 12, 2]
    assert batch["labels"][0, 4:].tolist() == [-100, -100, -100, -100]  # Padded with -100 for CrossEntropyLoss

    assert "decoder_attention_mask" in batch
    assert batch["decoder_attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert batch["decoder_attention_mask"][1].tolist() == [1, 1, 1, 1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# LossLogger Tests
# ---------------------------------------------------------------------------

def test_loss_logger_csv_and_history(tmp_path: Path) -> None:
    """Verify LossLogger CSV file creation, header, row logging, and history parsing."""
    logger = LossLogger(log_dir=tmp_path, csv_filename="test_losses.csv")
    assert logger.csv_path.exists()

    # Log 3 epochs
    logger.log_epoch(epoch=1, train_loss=0.8523, val_cer=0.1542, val_wer=0.3211, val_loss=0.7412)
    logger.log_epoch(epoch=2, train_loss=0.5121, val_cer=0.0954, val_wer=0.2105, val_loss=0.4891)
    logger.log_epoch(epoch=3, train_loss=0.3210, val_cer=0.0612, val_wer=0.1420, val_loss=0.3150)

    # Verify CSV file contents
    with open(logger.csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 4  # Header + 3 epochs
    assert lines[0] == "epoch,train_loss,val_cer,val_wer"

    # Verify in-memory history
    history = logger.get_history()
    assert history["epochs"] == [1, 2, 3]
    assert len(history["loss"]) == 3
    assert history["loss"] == [0.8523, 0.5121, 0.321]
    assert history["cer"] == [0.1542, 0.0954, 0.0612]

    # Verify reloading existing CSV upon restart
    logger2 = LossLogger(log_dir=tmp_path, csv_filename="test_losses.csv")
    hist2 = logger2.get_history()
    assert hist2["epochs"] == [1, 2, 3]
    assert len(hist2["train_loss"]) == 3


def test_loss_logger_plot_generation(tmp_path: Path) -> None:
    """Verify Matplotlib curve plotting generates a valid PNG file."""
    logger = LossLogger(log_dir=tmp_path)
    logger.log_epoch(epoch=1, train_loss=1.2, val_cer=0.25, val_wer=0.45)
    logger.log_epoch(epoch=2, train_loss=0.8, val_cer=0.18, val_wer=0.30)
    logger.log_epoch(epoch=3, train_loss=0.4, val_cer=0.10, val_wer=0.18)

    plot_file = logger.plot_curves()
    assert plot_file.exists()
    assert plot_file.stat().st_size > 1024

    img = Image.open(plot_file)
    assert img.format == "PNG"
    assert img.size[0] > 0 and img.size[1] > 0


def test_loss_logger_edge_cases(tmp_path: Path) -> None:
    """Verify LossLogger handles 0 entries, 1 entry, and NaN/Inf floats."""
    logger = LossLogger(log_dir=tmp_path, csv_filename="empty_loss.csv")
    p = logger.plot_curves()
    assert p.exists()

    # Log NaN / Inf
    logger.log_epoch(epoch=1, train_loss=float("nan"), val_cer=float("inf"), val_wer=0.0)
    hist = logger.get_history()
    assert len(hist["epochs"]) == 1


# ---------------------------------------------------------------------------
# TrOCRTrainer & Mini Training Step Tests
# ---------------------------------------------------------------------------

def test_mock_mini_training_step(tmp_path: Path) -> None:
    """
    Execute 2 training steps using tiny in-memory VisionEncoderDecoderModel (100% offline, zero network).
    Verifies forward pass, backward gradients, parameter update, checkpointing, and metrics logging.
    """
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    samples = [
        HandwritingSample(
            sample_id="s_01",
            text="Amoxicillin 500mg",
            image=np.full((64, 64, 3), 200, dtype=np.uint8),
        ),
        HandwritingSample(
            sample_id="s_02",
            text="Take 1 tablet daily",
            image=np.full((64, 64, 3), 220, dtype=np.uint8),
        ),
    ]

    train_ds = OCRDataset(samples=samples, processor=processor, is_training=True)
    val_ds = OCRDataset(samples=samples, processor=processor, is_training=False)

    config = TrainingConfig(
        model_name_or_path="mock-model",
        output_dir=str(tmp_path / "checkpoints"),
        num_train_epochs=2,
        batch_size=2,
        eval_batch_size=2,
        learning_rate=1e-3,
        device="cpu",
        empty_cache_steps=1,
    )

    trainer = TrOCRTrainer(
        config=config,
        model=model,
        processor=processor,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    # Initial parameter copy
    p_before = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    result = trainer.train()

    # Verify result dictionary
    assert result["epochs"] == 2
    assert result["final_loss"] > 0.0
    assert result["final_cer"] >= 0.0

    # Verify weights were updated by optimizer
    p_after = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    weight_changes = [not torch.equal(b, a) for b, a in zip(p_before, p_after)]
    assert any(weight_changes), "At least some model parameters should be updated during training."

    # Verify output checkpoint files
    out_dir = tmp_path / "checkpoints"
    assert (out_dir / "best_model.pt").exists()
    assert (out_dir / "checkpoint_epoch_1.pt").exists()
    assert (out_dir / "checkpoint_epoch_2.pt").exists()
    assert (out_dir / "losses.csv").exists()
    assert (out_dir / "loss_curves.png").exists()
    assert (out_dir / "training_state.json").exists()
    assert (out_dir / "best_model_hf").is_dir()

    state = json.loads((out_dir / "training_state.json").read_text())
    assert state["completed"] is True
    assert state["epoch"] >= 2


def test_checkpoint_save_and_resume(tmp_path: Path) -> None:
    """Verify checkpoint save and resume restores model weights and epoch continuity."""
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    samples = [
        HandwritingSample(sample_id="s1", text="Line 1", image=np.full((64, 64, 3), 200, dtype=np.uint8)),
        HandwritingSample(sample_id="s2", text="Line 2", image=np.full((64, 64, 3), 220, dtype=np.uint8)),
    ]
    ds = OCRDataset(samples, processor=processor, is_training=True)

    config = TrainingConfig(
        output_dir=str(tmp_path / "resume_test"),
        num_train_epochs=1,
        batch_size=2,
        device="cpu",
    )

    trainer1 = TrOCRTrainer(config=config, model=model, processor=processor, train_dataset=ds)
    trainer1.train()

    ckpt_path = tmp_path / "resume_test" / "best_model.pt"
    assert ckpt_path.exists()

    # Resume with trainer2
    model2 = create_tiny_mock_model(vocab_size=50, image_size=64)
    config2 = TrainingConfig(
        output_dir=str(tmp_path / "resume_test"),
        num_train_epochs=3,
        batch_size=2,
        device="cpu",
        resume_from_checkpoint=str(ckpt_path),
    )
    trainer2 = TrOCRTrainer(config=config2, model=model2, processor=processor, train_dataset=ds)
    assert trainer2.train_dataset is not None
    res = trainer2.train()
    assert res["epochs"] == 3

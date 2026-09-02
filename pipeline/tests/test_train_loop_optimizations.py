from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pipeline.dataset.dataset_loader import HandwritingSample
from pipeline.training.config import TrainingConfig
from pipeline.training.dataset import OCRDataCollator, OCRDataset, create_dummy_processor
from pipeline.training.train import TrOCRTrainer, create_tiny_mock_model


def test_fast_line_tuple_path_resizes_and_normalizes(tmp_path: Path) -> None:
    img_path = tmp_path / "line.png"
    Image.new("RGB", (240, 48), color=(200, 180, 160)).save(img_path)
    ds = OCRDataset(
        samples=[(str(img_path), "the quick brown", "s1", "w1")],
        processor=create_dummy_processor(size=(64, 64)),
        is_training=True,
        target_size=(64, 64),
    )
    item = ds[0]
    assert item["text"] == "the quick brown"
    assert item["sample_id"] == "s1"
    assert item["writer_id"] == "w1"
    assert item["pixel_values"].shape == (3, 64, 64)
    assert item["pixel_values"].dtype == torch.float32
    assert item["pixel_values"].min() >= -1.0
    assert item["pixel_values"].max() <= 1.0
    assert "input_ids" not in item


def test_collator_batch_tokenizes_text_only_items() -> None:
    processor = create_dummy_processor(vocab_size=50)
    collator = OCRDataCollator(processor=processor, pad_token_id=1)
    batch = collator(
        [
            {
                "sample_id": "a",
                "pixel_values": torch.zeros(3, 32, 32),
                "text": "Hi",
                "writer_id": "w1",
            },
            {
                "sample_id": "b",
                "pixel_values": torch.zeros(3, 32, 32),
                "text": "Hello there",
                "writer_id": "w2",
            },
        ]
    )
    assert batch["labels"].shape[0] == 2
    assert batch["input_ids"].shape[0] == 2
    assert int((batch["labels"] == -100).sum()) > 0


def test_evaluate_loss_only_skips_generate(tmp_path: Path) -> None:
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    def boom(*_args, **_kwargs):
        raise AssertionError("generate should not run during loss-only eval")

    model.generate = boom
    samples = [
        HandwritingSample(
            sample_id="s_01",
            text="hello",
            image=np.full((64, 64, 3), 200, dtype=np.uint8),
        )
    ]
    val_ds = OCRDataset(samples=samples, processor=processor, is_training=False)
    trainer = TrOCRTrainer(
        config=TrainingConfig(
            output_dir=str(tmp_path / "ckpt"),
            device="cpu",
            eval_batch_size=1,
            generate_on_eval=False,
            enable_step_profiling=False,
        ),
        model=model,
        processor=processor,
        val_dataset=val_ds,
    )
    cer, wer, loss = trainer.evaluate(generate=False)
    assert cer == float("inf")
    assert wer == float("inf")
    assert loss >= 0.0


def test_should_generate_on_eval_honors_schedule(tmp_path: Path) -> None:
    trainer = TrOCRTrainer(
        config=TrainingConfig(
            output_dir=str(tmp_path / "ckpt"),
            device="cpu",
            num_train_epochs=4,
            generate_on_eval=True,
            cer_eval_every_epochs=2,
            enable_step_profiling=False,
        ),
        model=create_tiny_mock_model(vocab_size=32, image_size=32),
    )
    assert trainer._should_generate_on_eval(2) is True
    assert trainer._should_generate_on_eval(3) is False
    assert trainer._should_generate_on_eval(4) is True

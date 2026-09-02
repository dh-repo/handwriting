"""Seq2SeqTrainer entry for line-level TrOCR. This is the design-doc training path."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from pipeline.training.data import LineCropCollator, LineCropDataset
from pipeline.training.eval import evaluate_checkpoint
from pipeline.training.metrics import htr_metric_bundle
from pipeline.training.model_contract import (
    apply_trocr_generation_config,
    load_htr_model,
    load_htr_processor,
    resolve_htr_device,
)
from pipeline.training.ship_gate import decide_ship

logger = logging.getLogger(__name__)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def resolve_device() -> torch.device:
    return resolve_htr_device()


def _bf16_ok(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    if device.type == "mps":
        return True
    return False


def build_compute_metrics(processor: Any):
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        bundle = htr_metric_bundle([r.strip() for r in refs], [h.strip() for h in hyps])
        return {"cer": bundle["cer"], "wer": bundle["wer"], "exact_match": bundle["exact_match"]}

    return compute_metrics


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("device_map") == "auto":
        raise ValueError("device_map=auto is not supported on MPS")
    if config.get("mix_private"):
        raise ValueError("private mix is disabled until the IAM loop is green")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = resolve_device()
    model_name = config.get("model_name_or_path", "microsoft/trocr-base-handwritten")
    processor = load_htr_processor(model_name)
    model = load_htr_model(model_name)
    apply_trocr_generation_config(
        model,
        processor,
        max_length=int(config.get("generation_max_length", 128)),
        num_beams=int(config.get("generation_num_beams", 1)),
    )
    model.to(device)

    train_dir = Path(config["train_dir"])
    val_dir = Path(config["val_dir"])
    train_ds = LineCropDataset(
        train_dir,
        processor=processor,
        max_target_length=int(config.get("max_target_length", 128)),
        is_train=True,
        augment=bool(config.get("augment", False)),
    )
    val_ds = LineCropDataset(
        val_dir,
        processor=processor,
        max_target_length=int(config.get("max_target_length", 128)),
        is_train=False,
        augment=False,
    )
    collator = LineCropCollator(processor, max_target_length=int(config.get("max_target_length", 128)))

    use_bf16 = bool(config.get("bf16", True)) and _bf16_ok(device)
    use_fp16 = bool(config.get("fp16", False)) and not use_bf16 and device.type == "cuda"
    max_steps = int(config["max_steps"]) if config.get("max_steps") is not None else -1
    batch = int(config.get("per_device_train_batch_size", 8))
    accum = int(config.get("gradient_accumulation_steps", 2))
    epochs = float(config.get("num_train_epochs", 5))
    steps_from_epochs = max(1, int((len(train_ds) / max(1, batch * accum)) * epochs))
    planned_steps = max_steps if max_steps > 0 else steps_from_epochs
    if config.get("warmup_steps") is not None:
        warmup_steps = int(config["warmup_steps"])
    else:
        warmup_steps = max(0, int(float(config.get("warmup_ratio", 0.05)) * planned_steps))
    args = Seq2SeqTrainingArguments(
        output_dir=config.get("output_dir", "runs/base_iam_v1"),
        predict_with_generate=True,
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=int(config.get("eval_steps", 200)),
        save_steps=int(config.get("save_steps", 200)),
        logging_steps=int(config.get("logging_steps", 20)),
        save_total_limit=int(config.get("save_total_limit", 3)),
        load_best_model_at_end=True,
        metric_for_best_model=config.get("metric_for_best_model", "cer"),
        greater_is_better=False,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=accum,
        learning_rate=float(config.get("learning_rate", 5e-5)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        max_steps=max_steps,
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=int(config.get("dataloader_num_workers", 2)),
        dataloader_pin_memory=device.type == "cuda",
        report_to=config.get("report_to", ["tensorboard"]),
        generation_max_length=int(config.get("generation_max_length", 128)),
        generation_num_beams=int(config.get("generation_num_beams", 1)),
        torch_empty_cache_steps=int(config.get("torch_empty_cache_steps", 50)),
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=build_compute_metrics(processor),
    )
    result = trainer.train()
    best_dir = Path(args.output_dir) / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    summary = {"metrics": result.metrics, "best_dir": str(best_dir)}
    test_dir = config.get("test_dir")
    if test_dir:
        summary["test"] = evaluate_checkpoint(
            split_dir=test_dir,
            checkpoint=str(best_dir),
            output_dir=args.output_dir,
            beams=int(config.get("test_num_beams", 4)),
            split_name=str(config.get("test_split_name", "Teklia/IAM-line test")),
            batch_size=int(config.get("test_batch_size", 8)),
        )
        summary["ship"] = decide_ship(summary["test"], output_dir=args.output_dir)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on line crops (Seq2SeqTrainer)")
    parser.add_argument("--config", required=True, help="Path to debug32.yaml / base_iam.yaml")
    args = parser.parse_args()
    summary = run_training(load_yaml_config(args.config))
    print(summary)


if __name__ == "__main__":
    main()

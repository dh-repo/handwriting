"""
pipeline/training/lora_train.py
Parameter-Efficient Fine-Tuning (LoRA via PEFT) for TrOCR on Apple Silicon MPS.

Trains low-rank adaptation matrices on attention projections while keeping
pretrained visual encoder and language decoder weights frozen. Prevents validation
collapse and produces shippable checkpoints that improve line-HTR cased CER.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from transformers import VisionEncoderDecoderConfig
from peft import LoraConfig, get_peft_model

# PEFT save_pretrained inspects model.config.__class__.from_pretrained(model_id).vocab_size.
# For VisionEncoderDecoderConfig, vocab_size resides on decoder.vocab_size.
if not hasattr(VisionEncoderDecoderConfig, "vocab_size"):
    VisionEncoderDecoderConfig.vocab_size = property(
        lambda self: getattr(self.decoder, "vocab_size", 50265),
        lambda self, val: None,
    )
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dictionary: {path}")
    return data


def build_compute_metrics(processor: Any):
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = label_ids.copy()
        tok = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tok, "pad_token_id", 1) or 1
        label_ids[label_ids == -100] = pad_id
        hyps = processor.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        bundle = htr_metric_bundle([r.strip() for r in refs], [h.strip() for h in hyps])
        return {
            "cer": round(bundle["cer"], 5),
            "wer": round(bundle["wer"], 5),
            "exact_match": round(bundle["exact_match"], 5),
        }

    return compute_metrics


def train_lora(config: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = resolve_htr_device()
    logger.info("Starting LoRA training on device: %s", device)

    model_name = config.get("model_name_or_path", "microsoft/trocr-base-handwritten")
    output_dir = Path(config.get("output_dir", "runs/lora_trocr_base_iam"))
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = load_htr_processor(model_name)
    base_model = load_htr_model(model_name)
    apply_trocr_generation_config(
        base_model,
        processor,
        max_length=int(config.get("generation_max_length", 128)),
        num_beams=int(config.get("generation_num_beams", 1)),
    )

    lora_r = int(config.get("lora_r", 16))
    lora_alpha = int(config.get("lora_alpha", 32))
    lora_dropout = float(config.get("lora_dropout", 0.05))
    target_modules = config.get("target_modules", ["q_proj", "v_proj"])

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    train_dir = Path(config["train_dir"])
    val_dir = Path(config["val_dir"])
    max_target_len = int(config.get("max_target_length", 128))

    train_ds = LineCropDataset(
        train_dir,
        processor=processor,
        max_target_length=max_target_len,
        is_train=True,
        augment=bool(config.get("augment", False)),
    )
    val_ds = LineCropDataset(
        val_dir,
        processor=processor,
        max_target_length=max_target_len,
        is_train=False,
        augment=False,
    )
    collator = LineCropCollator(processor, max_target_length=max_target_len)

    batch_size = int(config.get("per_device_train_batch_size", 8))
    eval_batch_size = int(config.get("per_device_eval_batch_size", 8))
    grad_accum = int(config.get("gradient_accumulation_steps", 2))
    epochs = int(config.get("num_train_epochs", 3))
    lr = float(config.get("learning_rate", 5e-4))
    warmup_steps = int(config.get("warmup_steps", 50))

    max_steps = int(config.get("max_steps", -1))
    eval_strat = "no" if 0 < max_steps < 20 else "epoch"
    save_strat = "no" if 0 < max_steps < 20 else "epoch"
    load_best = bool(eval_strat != "no")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        max_steps=max_steps,
        eval_strategy=eval_strat,
        save_strategy=save_strat,
        save_total_limit=2,
        logging_steps=10 if max_steps > 0 else 50,
        load_best_model_at_end=load_best,
        metric_for_best_model="cer" if load_best else None,
        greater_is_better=False,
        predict_with_generate=True,
        dataloader_num_workers=int(config.get("dataloader_num_workers", 0)),
        fp16=False,
        bf16=bool(config.get("bf16", False) and device.type == "mps"),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if eval_strat != "no" else None,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor) if eval_strat != "no" else None,
    )

    logger.info("Executing LoRA fine-tuning...")
    train_result = trainer.train()

    # Save adapter weights directly
    try:
        from peft import get_peft_model_state_dict
        adapter_state = get_peft_model_state_dict(model)
        torch.save(adapter_state, output_dir / "adapter_weights.pt")
        logger.info("Saved adapter weights to %s", output_dir / "adapter_weights.pt")
    except Exception as exc:
        logger.warning("Could not save adapter state dict: %s", exc)

    # Merge LoRA adapter weights into base model for high-throughput standalone inference
    logger.info("Merging LoRA weights into standalone checkpoint...")
    merged_model = model.merge_and_unload()
    merged_dir = output_dir / "best_model_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))

    # Also export to checkpoints/ directory if requested
    export_dir = config.get("export_dir")
    if export_dir:
        exp_path = Path(export_dir)
        exp_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(exp_path))
        processor.save_pretrained(str(exp_path))
        logger.info("Exported merged checkpoint to %s", exp_path)

    # Evaluate on frozen Teklia test set if test_dir is provided
    test_metrics: Dict[str, Any] = {}
    test_dir = config.get("test_dir")
    if test_dir and Path(test_dir).exists():
        logger.info("Evaluating merged checkpoint on frozen test split: %s", test_dir)
        test_beams = int(config.get("test_num_beams", 4))
        test_metrics = evaluate_checkpoint(
            split_dir=test_dir,
            checkpoint=str(merged_dir),
            output_dir=output_dir / "teklia_test_report",
            beams=test_beams,
            split_name=str(config.get("test_split_name", "Teklia/IAM-line test")),
            batch_size=eval_batch_size,
        )
        logger.info("Teklia Test Results (beams=%d): %s", test_beams, test_metrics)

        # Decide shipping promotion
        report_path = output_dir / "teklia_test_report" / "test_report.json"
        if report_path.is_file():
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            decision = decide_ship(rep, output_dir=output_dir)
            logger.info("Ship decision: %s", decision)

    return {
        "train_result": train_result,
        "merged_dir": str(merged_dir),
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for TrOCR on Apple Silicon")
    parser.add_argument("--config", required=True, help="Path to YAML training configuration")
    args = parser.parse_args()

    config = load_config(args.config)
    train_lora(config)


if __name__ == "__main__":
    main()

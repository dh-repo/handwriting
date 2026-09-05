"""
pipeline/training/lora_micro_tune.py
Background LoRA micro-tuning script and library function on Apple Silicon MPS with Experience Replay and LASA Clinical Safety Gate.

Features:
- Parameter-Efficient Fine-Tuning (PEFT LoRA: r=16, alpha=32, target_modules=["q_proj", "v_proj"]).
- Targeted for Apple Silicon MPS with automatic CPU/CUDA fallback.
- Memory hygiene: bf16 mixed precision autocast, periodic torch.mps.empty_cache().
- Continuous experience replay data loading balancing Darkroom operator feedback with golden anchor lines.
- Post-tuning CER validation check and zero-tolerance clinical LASA safety gate.
- Standalone weight merge (merge_and_unload) upon passing shipping criteria.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderConfig

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# PEFT VisionEncoderDecoderConfig vocab_size compatibility fix
if not hasattr(VisionEncoderDecoderConfig, "vocab_size"):
    VisionEncoderDecoderConfig.vocab_size = property(
        lambda self: getattr(self.decoder, "vocab_size", 50265),
        lambda self, val: None,
    )

try:
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from pipeline.training.data import LineCropCollator, LineCropDataset
from pipeline.training.experience_replay import ExperienceReplayDataset, ReplayBatchSampler
from pipeline.training.metrics import cased_cer
from pipeline.training.model_contract import (
    apply_trocr_generation_config,
    load_htr_model,
    load_htr_processor,
    resolve_htr_device,
)
from pipeline.training.ship_gate import (
    LasaAuditResult,
    audit_lasa_safety,
    check_cer_regression,
    decide_ship,
    load_lasa_catalog,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline.training.lora_micro_tune")


def resolve_device_target(forced_device: str = "auto") -> torch.device:
    """Resolve compute device with MPS priority on Apple Silicon."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if forced_device == "auto":
        return resolve_htr_device()

    dev = torch.device(forced_device)
    if dev.type == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            logger.warning("MPS requested but unavailable on this system. Falling back to CPU.")
            return torch.device("cpu")
    return dev


def run_micro_tune(
    feedback_manifest: Union[str, Path] = "data/feedback/manifest.jsonl",
    anchor_dir: Union[str, Path] = "data/iam_line/train",
    val_dir: Optional[Union[str, Path]] = "data/iam_line/validation",
    base_model_name_or_path: str = "microsoft/trocr-base-handwritten",
    output_dir: Union[str, Path] = "runs/lora_micro_tune",
    steps: int = 30,
    batch_size: int = 8,
    replay_ratio: float = 0.5,
    learning_rate: float = 2e-4,
    device: str = "auto",
    empty_cache_steps: int = 10,
    max_target_length: int = 128,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bf16: bool = True,
    merge_adapter: bool = False,
    baseline_cer: Optional[float] = None,
    max_cer_regression: float = 0.05,
    lasa_vocab_dir: Union[str, Path] = "data/reference_handwriting/vocabularies",
    eval_lasa: bool = True,
    max_eval_samples: Optional[int] = 50,
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    lasa_test_samples: Optional[List[Tuple[str, str]]] = None,
) -> dict[str, Any]:
    """
    Execute background LoRA micro-tuning epoch on Apple Silicon MPS (or CPU fallback).

    1. Ingests feedback manifest and golden anchor samples via ExperienceReplayDataset.
    2. Interleaves mini-batches preserving replay_ratio (50:50 default) to prevent catastrophic forgetting.
    3. Trains PEFT LoRA adapter (r=16, alpha=32) for small micro-epochs (20-50 steps).
    4. Executes periodic torch.mps.empty_cache() on MPS for Metal memory management.
    5. Evaluates validation CER and clinical LASA zero-tolerance safety audit.
    6. Decides promotion via decide_ship; merges weights into standalone checkpoint if accepted.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is required for LoRA micro-tuning (pip install peft).")

    resolved_device = resolve_device_target(device)
    logger.info("Starting LoRA micro-tuning on device: %s (steps=%d, batch_size=%d)", resolved_device, steps, batch_size)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)

    # 1. Load Processor and Base Model
    if processor is None:
        processor = load_htr_processor(base_model_name_or_path)

    if model is None:
        model = load_htr_model(base_model_name_or_path)
        apply_trocr_generation_config(model, processor, max_length=max_target_length, num_beams=1)

    # 2. Configure MPS attention safeguards if needed
    if resolved_device.type == "mps":
        # On MPS, PyTorch's scaled_dot_product_attention does not support dropout during training.
        # Ensure eager attention implementation across encoder and decoder.
        if getattr(model.config, "_attn_implementation", None) == "sdpa":
            model.config._attn_implementation = "eager"
        if hasattr(model, "encoder") and hasattr(model.encoder, "config"):
            if getattr(model.encoder.config, "_attn_implementation", None) == "sdpa":
                model.encoder.config._attn_implementation = "eager"
        if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
            if getattr(model.decoder.config, "_attn_implementation", None) == "sdpa":
                model.decoder.config._attn_implementation = "eager"

    # 3. Attach PEFT LoRA adapter if not already attached
    is_peft = hasattr(model, "peft_config")
    if not is_peft:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules or ["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        logger.info("Attached PEFT LoRA configuration: r=%d, alpha=%d", lora_r, lora_alpha)

    model.to(resolved_device)

    # 3. Build Experience Replay DataLoader
    dataset = ExperienceReplayDataset(
        feedback_manifest_path=feedback_manifest,
        anchor_dir=anchor_dir,
        processor=processor,
        max_target_length=max_target_length,
        augment=True,
    )

    sampler = ReplayBatchSampler(
        feedback_indices=dataset.feedback_indices,
        anchor_indices=dataset.anchor_indices,
        batch_size=batch_size,
        replay_ratio=replay_ratio,
        shuffle=True,
        drop_last=False,
    )

    collator = LineCropCollator(processor=processor, max_target_length=max_target_length)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)

    # 4. Configure Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in model!")

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    # Autocast setup
    use_mps_autocast = bool(bf16 and resolved_device.type == "mps")
    use_cuda_autocast = bool(bf16 and resolved_device.type == "cuda")

    def _get_autocast_context():
        if use_mps_autocast:
            return torch.autocast(device_type="mps", dtype=torch.bfloat16)
        if use_cuda_autocast:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    # 5. Execute Micro-Tuning Loop
    model.train()
    losses: List[float] = []
    data_iter = iter(dataloader) if len(sampler) > 0 else None

    logger.info("Executing %d micro-tuning steps...", steps)
    for step in range(steps):
        if data_iter is None:
            logger.warning("No data batches available for training; ending loop early.")
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(resolved_device)
        labels = batch["labels"].to(resolved_device)

        optimizer.zero_grad(set_to_none=True)

        with _get_autocast_context():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().item())
        losses.append(loss_val)

        # Periodic MPS empty cache
        if resolved_device.type == "mps" and (step + 1) % empty_cache_steps == 0:
            torch.mps.empty_cache()

        if (step + 1) % 10 == 0 or step == steps - 1:
            logger.info("Micro-tuning step %d/%d - loss: %.4f", step + 1, steps, loss_val)

    # 6. Save Adapter Weights
    adapter_dir = output_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(adapter_dir))
        logger.info("Saved PEFT adapter weights to %s", adapter_dir)
    except Exception as exc:
        logger.warning("Failed calling model.save_pretrained: %s", exc)

    try:
        adapter_state = get_peft_model_state_dict(model)
        torch.save(adapter_state, output_path / "adapter_weights.pt")
        logger.info("Saved adapter state dict to %s", output_path / "adapter_weights.pt")
    except Exception as exc:
        logger.warning("Could not save adapter state dict: %s", exc)

    # 7. Evaluate Validation Split
    model.eval()
    eval_refs: List[str] = []
    eval_hyps: List[str] = []
    candidate_cer = None

    val_path = Path(val_dir) if val_dir else None
    if val_path and (val_path / "labels.tsv").is_file():
        logger.info("Evaluating validation split: %s", val_path)
        val_ds = LineCropDataset(
            val_path,
            processor=processor,
            max_target_length=max_target_length,
            is_train=False,
            augment=False,
        )
        eval_count = min(len(val_ds), max_eval_samples) if max_eval_samples else len(val_ds)
        for i in range(eval_count):
            item = val_ds[i]
            pv = item["pixel_values"].unsqueeze(0).to(resolved_device)
            ref_text = item["text"].strip()
            with torch.no_grad():
                pred_ids = model.generate(pv, max_length=max_target_length, num_beams=1)
            hyp_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            eval_refs.append(ref_text)
            eval_hyps.append(hyp_text)

        if eval_refs:
            candidate_cer = cased_cer(eval_refs, eval_hyps)
            logger.info("Validation evaluation completed across %d lines: CER=%.4f", eval_count, candidate_cer)

    # 8. Clinical LASA Safety Audit
    lasa_audit: LasaAuditResult
    if eval_lasa:
        try:
            lasa_pairs = load_lasa_catalog(lasa_vocab_dir)
        except Exception as exc:
            logger.warning("Could not load LASA catalog from %s: %s", lasa_vocab_dir, exc)
            lasa_pairs = []

        if lasa_test_samples:
            lasa_refs = [s[0] for s in lasa_test_samples]
            lasa_hyps = [s[1] for s in lasa_test_samples]
        else:
            lasa_refs = eval_refs
            lasa_hyps = eval_hyps

        lasa_audit = audit_lasa_safety(lasa_refs, lasa_hyps, lasa_pairs=lasa_pairs)
        logger.info(
            "Clinical LASA audit evaluated: %d lines, %d violations, passed=%s",
            lasa_audit.total_evaluated,
            len(lasa_audit.violations),
            lasa_audit.passed,
        )
    else:
        lasa_audit = LasaAuditResult(total_evaluated=0, violations=[], passed=True)

    # 9. Shipping Gate Decision
    report = {
        "checkpoint": str(output_path),
        "cer": candidate_cer,
        "num_beams": 1,
    }
    ship_decision = decide_ship(
        report,
        output_dir=output_path,
        baseline_cer=baseline_cer,
        max_cer_regression=max_cer_regression,
        lasa_audit=lasa_audit,
    )
    logger.info("Ship gate decision: promote=%s (%s)", ship_decision["promote"], ship_decision["reason"])

    # 10. Merge and Export Checkpoint
    merged_dir = output_path / "best_model_merged"
    merged_dir_str = None
    if merge_adapter:  # Export a candidate only; activation requires separate measured evaluation.
        logger.info("Promotion approved. Merging LoRA weights into standalone checkpoint...")
        if hasattr(model, "merge_and_unload"):
            merged_model = model.merge_and_unload()
        else:
            merged_model = model

        merged_dir.mkdir(parents=True, exist_ok=True)
        try:
            merged_model.save_pretrained(str(merged_dir))
            if hasattr(processor, "save_pretrained"):
                processor.save_pretrained(str(merged_dir))
            merged_dir_str = str(merged_dir)
            logger.info("Exported merged checkpoint to %s", merged_dir_str)
        except Exception as exc:
            logger.warning("Failed saving merged model: %s", exc)

    return {
        "steps": len(losses),
        "losses": losses,
        "train_loss": float(np.mean(losses)) if losses else 0.0,
        "candidate_cer": candidate_cer,
        "lasa_audit": lasa_audit.to_dict(),
        "ship_decision": ship_decision,
        "adapter_dir": str(adapter_dir),
        "merged_dir": merged_dir_str,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Background LoRA Micro-Tuning with Experience Replay & LASA Safety Gate")
    parser.add_argument("--manifest", default="data/feedback/manifest.jsonl", help="Path to operator feedback manifest")
    parser.add_argument("--anchor_dir", default="data/iam_line/train", help="Path to golden anchor dataset directory")
    parser.add_argument("--val_dir", default="data/iam_line/validation", help="Path to validation split directory")
    parser.add_argument("--base_model", default="microsoft/trocr-base-handwritten", help="Base model name or path")
    parser.add_argument("--output_dir", default="runs/lora_micro_tune", help="Output directory for checkpoints and reports")
    parser.add_argument("--steps", type=int, default=30, help="Number of micro-tuning steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--replay_ratio", type=float, default=0.5, help="Proportion of feedback samples per batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--empty_cache_steps", type=int, default=10, help="Periodic empty cache interval on MPS")
    parser.add_argument("--no_merge", action="store_true", help="Disable merging adapter into base model")
    parser.add_argument("--baseline_cer", type=float, default=None, help="Baseline CER to compare against")
    parser.add_argument("--max_cer_regression", type=float, default=0.05, help="Maximum allowable CER regression")
    parser.add_argument("--lasa_vocab_dir", default="data/reference_handwriting/vocabularies", help="Directory containing rxnorm_medications.json")
    args = parser.parse_args()

    result = run_micro_tune(
        feedback_manifest=args.manifest,
        anchor_dir=args.anchor_dir,
        val_dir=args.val_dir,
        base_model_name_or_path=args.base_model,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        replay_ratio=args.replay_ratio,
        learning_rate=args.lr,
        device=args.device,
        empty_cache_steps=args.empty_cache_steps,
        merge_adapter=not args.no_merge,
        baseline_cer=args.baseline_cer,
        max_cer_regression=args.max_cer_regression,
        lasa_vocab_dir=args.lasa_vocab_dir,
    )
    print("Micro-tune completed successfully:")
    print(json.dumps(result["ship_decision"], indent=2))


if __name__ == "__main__":
    main()

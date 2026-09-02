"""
pipeline/training/curriculum.py
Multi-Stage Curriculum Training Orchestrator for TrOCR-Large (558M) on Apple Silicon MPS.
Coordinates Stage 1 (General Cursive Adaptation) and Stage 2 (Doctor & Clinical Specialization),
handling layer freezing, optimizer re-initialization, continuous telemetry, and atomic checkpointing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from pipeline.dataset.dataset_loader import (
    HandwritingSample,
    MedicalPrescriptionDatasetLoader,
    MedicalPrescriptionSample,
)
from pipeline.training.config import CurriculumConfig, CurriculumStageConfig, TrainingConfig, get_default_num_workers
from pipeline.training.dataset import OCRDataCollator, OCRDataset, create_dummy_processor, load_trocr_processor
from pipeline.training.loss_logger import LossLogger
from pipeline.training.prefetcher import AsyncDevicePrefetcher
from pipeline.training.profiler import StepTiming, TrainingStepProfiler
from pipeline.training.train import (
    TrOCRTrainer,
    configure_gradient_checkpointing,
    create_tiny_mock_model,
    freeze_encoder_layers,
    get_autocast_context,
    get_optimal_device,
    load_trocr_model,
)

logger = logging.getLogger("pipeline.training.curriculum")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class CurriculumExecutionResult:
    """Summary metrics and artifact paths returned upon curriculum completion."""
    total_stages: int
    total_epochs: int
    total_steps: int
    best_cer: float
    best_wer: float
    best_loss: float
    best_checkpoint_path: str
    best_hf_dir: str
    losses_csv_path: str
    loss_curves_path: str
    stage_summaries: List[Dict[str, Any]]
    total_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s, encoding="utf-8")
        return s


class MultiStageCurriculumTrainer:
    """
    Coordinator for multi-stage curriculum training of TrOCR models on Apple Silicon MPS.
    """

    def __init__(
        self,
        config: CurriculumConfig,
        processor: Optional[Any] = None,
        loss_logger: Optional[LossLogger] = None,
        model: Optional[VisionEncoderDecoderModel] = None,
    ) -> None:
        self.config = config
        self.root_output_dir = Path(config.root_output_dir)
        self.root_output_dir.mkdir(parents=True, exist_ok=True)

        self.device = get_optimal_device(config.device)
        logger.info(f"MultiStageCurriculumTrainer initialized on device: {self.device}")

        self.processor = processor or load_trocr_processor(config.model_name_or_path)
        attn_impl = getattr(config, "attn_implementation", "sdpa")
        self.model: Optional[VisionEncoderDecoderModel] = model.to(self.device) if model is not None else None
        if self.model is not None:
            if hasattr(self.model.config, "_attn_implementation"):
                self.model.config._attn_implementation = attn_impl
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "config") and hasattr(self.model.encoder.config, "_attn_implementation"):
                self.model.encoder.config._attn_implementation = attn_impl
            if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "config") and hasattr(self.model.decoder.config, "_attn_implementation"):
                self.model.decoder.config._attn_implementation = attn_impl

        device_type = self.device.type
        scaler_enabled = (self.config.mixed_precision.lower() in ("fp16", "bf16")) and device_type in ("mps", "cuda")
        self.scaler = (
            torch.amp.GradScaler(device_type, enabled=scaler_enabled)
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and device_type in ("mps", "cuda")
            else None
        )

        # Shared continuous loss logger across all stages
        self.loss_logger = loss_logger or LossLogger(
            log_dir=self.root_output_dir,
            csv_filename="losses.csv",
            plot_filename="loss_curves.png",
            curriculum_mode=True,
        )

        self.global_step = 0
        self.global_epoch = 0
        self.stage_demarcations: List[int] = []
        self.stage_names: List[str] = [s.stage_name for s in config.stages]
        self.overall_best_cer = float("inf")
        self.overall_best_wer = float("inf")
        self.overall_best_loss = float("inf")
        self.overall_best_checkpoint: Optional[Path] = None

        if getattr(self.config, "enable_step_profiling", True):
            self.profiler: Optional[TrainingStepProfiler] = TrainingStepProfiler(
                warmup_steps=2,
                device=self.device,
                enabled=True,
            )
        else:
            self.profiler = None

    def _enable_gradient_checkpointing(self) -> None:
        """Enable activation recomputation on model, encoder, and decoder."""
        if self.model is not None and self.config.gradient_checkpointing:
            configure_gradient_checkpointing(self.model, enabled=True)
            logger.info("Gradient checkpointing successfully enabled across ViT encoder and RoBERTa decoder.")

    def _apply_layer_freezing(self, num_encoder_layers: int) -> None:
        """Freeze patch embeddings and bottom N Transformer layers of the ViT encoder."""
        if self.model is not None and num_encoder_layers > 0:
            freeze_encoder_layers(self.model, num_layers=num_encoder_layers)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            pct = (trainable / total * 100.0) if total > 0 else 0.0
            logger.info(f"Trainable parameters after freezing: {trainable:,} / {total:,} ({pct:.1f}%)")

    def _load_stage_datasets(
        self, stage_cfg: CurriculumStageConfig
    ) -> Tuple[OCRDataset, Optional[OCRDataset]]:
        """Load and filter OCR datasets for a specific curriculum stage."""
        loader = MedicalPrescriptionDatasetLoader()

        # Training dataset
        if stage_cfg.dataset_manifest and os.path.exists(stage_cfg.dataset_manifest):
            train_samples = loader.load_manifest(stage_cfg.dataset_manifest)
            if not train_samples:
                raise ValueError(
                    f"No real handwriting samples found in '{stage_cfg.dataset_manifest}'. "
                    "Synthetic corpus records are excluded."
                )
            if stage_cfg.category_filter:
                cat_set = set(stage_cfg.category_filter)

                def _sample_matches(s: Any) -> bool:
                    cat = ""
                    is_lasa = False
                    if isinstance(s, dict):
                        cat = s.get("category", "")
                        is_lasa = bool(s.get("is_lasa", False))
                        if not cat and isinstance(s.get("metadata"), dict):
                            cat = s["metadata"].get("category", "")
                        if not is_lasa and isinstance(s.get("metadata"), dict):
                            is_lasa = bool(s["metadata"].get("is_lasa", False))
                    else:
                        cat = getattr(s, "category", "")
                        is_lasa = bool(getattr(s, "is_lasa", False))
                        if not cat and hasattr(s, "metadata") and isinstance(s.metadata, dict):
                            cat = s.metadata.get("category", "")
                        if not is_lasa and hasattr(s, "metadata") and isinstance(s.metadata, dict):
                            is_lasa = bool(s.metadata.get("is_lasa", False))
                    return (cat in cat_set) or is_lasa

                filtered = [s for s in train_samples if _sample_matches(s)]
                if filtered:
                    train_samples = filtered
                    logger.info(f"Filtered stage '{stage_cfg.stage_name}' training dataset to {len(train_samples)} samples matching categories: {stage_cfg.category_filter}")
            train_ds = OCRDataset(samples=train_samples, processor=self.processor, is_training=True)
        else:
            raise FileNotFoundError(
                f"Curriculum stage '{stage_cfg.stage_name}' requires a real dataset manifest "
                f"at '{stage_cfg.dataset_manifest}'. Synthetic training fallbacks are disabled."
            )

        # Validation dataset
        val_ds = None
        if stage_cfg.val_manifest and os.path.exists(stage_cfg.val_manifest):
            val_samples = loader.load_manifest(stage_cfg.val_manifest)
            if stage_cfg.category_filter:
                cat_set = set(stage_cfg.category_filter)
                v_filtered = [s for s in val_samples if _sample_matches(s)]
                if v_filtered:
                    val_samples = v_filtered
            val_ds = OCRDataset(samples=val_samples, processor=self.processor, is_training=False)

        return train_ds, val_ds

    def execute_curriculum(
        self,
        resume_stage: Optional[str] = None,
        resume_checkpoint: Optional[str] = None,
    ) -> CurriculumExecutionResult:
        """
        Execute full curriculum across all configured stages sequentially.
        """
        t_curriculum_start = time.time()
        stage_summaries: List[Dict[str, Any]] = []

        # Load initial model if not already provided
        attn_impl = getattr(self.config, "attn_implementation", "sdpa")
        if self.model is None:
            self.model = load_trocr_model(
                self.config.model_name_or_path,
                self.processor,
                self.device,
                gradient_checkpointing=self.config.gradient_checkpointing,
                attn_implementation=attn_impl,
            )
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = attn_impl
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "config") and hasattr(self.model.encoder.config, "_attn_implementation"):
            self.model.encoder.config._attn_implementation = attn_impl
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "config") and hasattr(self.model.decoder.config, "_attn_implementation"):
            self.model.decoder.config._attn_implementation = attn_impl
        self._enable_gradient_checkpointing()

        start_stage_idx = 0
        if resume_stage:
            for idx, stage in enumerate(self.config.stages):
                if stage.stage_name == resume_stage:
                    start_stage_idx = idx
                    break

        for stage_idx in range(start_stage_idx, len(self.config.stages)):
            stage_cfg = self.config.stages[stage_idx]
            logger.info("=" * 80)
            logger.info(f"STARTING CURRICULUM STAGE {stage_idx + 1}/{len(self.config.stages)}: {stage_cfg.stage_name}")
            logger.info("=" * 80)

            # Record stage transition demarcation
            if stage_idx > 0:
                self.stage_demarcations.append(self.global_epoch)

            # Apply layer freezing if configured
            if stage_cfg.freeze_encoder_layers > 0:
                self._apply_layer_freezing(stage_cfg.freeze_encoder_layers)

            # Execute single stage
            stage_summary = self._train_stage(
                stage_cfg=stage_cfg,
                stage_index=stage_idx,
                resume_checkpoint=resume_checkpoint if stage_idx == start_stage_idx else None,
            )
            stage_summaries.append(stage_summary)

            # Checkpoint promotion between stages: Load best stage model for subsequent stage
            best_stage_ckpt = Path(stage_cfg.output_dir or (self.root_output_dir / stage_cfg.stage_name)) / "best_model.pt"
            if best_stage_ckpt.exists():
                logger.info(f"Loading best weights from {best_stage_ckpt} for subsequent curriculum stage.")
                state = torch.load(best_stage_ckpt, map_location=self.device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"])
                elif isinstance(state, dict):
                    self.model.load_state_dict(state)

        # Final Promotion to Root Checkpoint Directory
        self._promote_global_best_model()

        # Render Final Stage-Demarcated Training Curves
        self.loss_logger.plot_curves(
            output_path=self.root_output_dir / "loss_curves.png",
            title="TrOCR-Large Multi-Stage Curriculum Loss & CER Telemetry",
            stage_demarcations=self.stage_demarcations,
            stage_names=self.stage_names,
        )

        total_time = time.time() - t_curriculum_start
        result = CurriculumExecutionResult(
            total_stages=len(self.config.stages),
            total_epochs=self.global_epoch,
            total_steps=self.global_step,
            best_cer=round(self.overall_best_cer, 4) if self.overall_best_cer != float("inf") else 0.0,
            best_wer=round(self.overall_best_wer, 4) if self.overall_best_wer != float("inf") else 0.0,
            best_loss=round(self.overall_best_loss, 4) if self.overall_best_loss != float("inf") else 0.0,
            best_checkpoint_path=str(self.root_output_dir / "best_model.pt"),
            best_hf_dir=str(self.root_output_dir / "best_model_hf"),
            losses_csv_path=str(self.loss_logger.csv_path),
            loss_curves_path=str(self.root_output_dir / "loss_curves.png"),
            stage_summaries=stage_summaries,
            total_time_seconds=round(total_time, 2),
        )

        if self.profiler is not None:
            logger.info("\n" + self.profiler.format_summary())
            self.profiler.export_json(self.root_output_dir / "step_profiler_summary.json")
            self.profiler.export_csv(self.root_output_dir / "step_timings.csv")

        result.to_json(self.root_output_dir / "curriculum_summary.json")
        logger.info(f"Curriculum Training Finished in {total_time:.2f}s. Best Overall CER: {result.best_cer:.4f}")
        return result

    def _train_stage(
        self,
        stage_cfg: CurriculumStageConfig,
        stage_index: int,
        resume_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute an individual curriculum stage."""
        stage_out = Path(stage_cfg.output_dir or (self.root_output_dir / stage_cfg.stage_name))
        stage_out.mkdir(parents=True, exist_ok=True)

        train_ds, val_ds = self._load_stage_datasets(stage_cfg)
        collator = OCRDataCollator(processor=self.processor)

        # Determine worker multiprocessing settings
        num_workers = stage_cfg.num_workers
        if num_workers is None:
            num_workers = getattr(self.config, "num_workers", None)
        if num_workers is None:
            if len(train_ds) < 32:
                num_workers = 0
            else:
                num_workers = get_default_num_workers()

        persistent_workers = bool(getattr(stage_cfg, "persistent_workers", True) and (num_workers > 0))
        prefetch_factor = getattr(stage_cfg, "prefetch_factor", 4) if num_workers > 0 else None
        pin_memory = stage_cfg.pin_memory if stage_cfg.pin_memory is not None else (self.device.type == "cuda")

        train_loader_kwargs: Dict[str, Any] = {
            "batch_size": stage_cfg.micro_batch_size,
            "shuffle": True,
            "collate_fn": collator,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            train_loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                train_loader_kwargs["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(train_ds, **train_loader_kwargs)

        steps_per_epoch = max(1, len(train_loader) // max(1, stage_cfg.gradient_accumulation_steps))
        total_stage_steps = steps_per_epoch * stage_cfg.num_epochs

        # Optimizer: only optimize parameters with requires_grad=True
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias", "layer_norm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=stage_cfg.learning_rate, eps=1e-8)

        warmup_steps = max(0, int(total_stage_steps * stage_cfg.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_stage_steps)
        )

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            logger.info(f"Resuming stage from checkpoint: {resume_checkpoint}")
            state = torch.load(resume_checkpoint, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
                if "optimizer_state_dict" in state and state["optimizer_state_dict"]:
                    try:
                        optimizer.load_state_dict(state["optimizer_state_dict"])
                    except Exception:
                        pass
                if "scheduler_state_dict" in state and state["scheduler_state_dict"]:
                    try:
                        scheduler.load_state_dict(state["scheduler_state_dict"])
                    except Exception:
                        pass

        best_stage_cer = float("inf")
        best_stage_loss = float("inf")
        t_stage_start = time.time()
        autocast_ctx = get_autocast_context(self.device, mixed_precision=self.config.mixed_precision)
        use_prefetcher = getattr(stage_cfg, "use_async_prefetcher", True)
        prefetch_q_size = getattr(stage_cfg, "prefetch_queue_size", 3)

        for stage_epoch in range(1, stage_cfg.num_epochs + 1):
            self.global_epoch += 1
            self.model.train()
            epoch_loss_t = None
            accum_loss_t = None
            n_opt = 0
            optimizer.zero_grad(set_to_none=True)

            if use_prefetcher:
                data_iter = AsyncDevicePrefetcher(
                    train_loader,
                    device=self.device,
                    mixed_precision=self.config.mixed_precision,
                    queue_size=prefetch_q_size,
                )
            else:
                data_iter = train_loader

            t_data_start = time.perf_counter()
            try:
                for step, batch in enumerate(data_iter):
                    t_data = time.perf_counter() - t_data_start
                    if self.profiler is not None:
                        self.profiler.record_data_wait(t_data)

                    if "pixel_values" not in batch or "labels" not in batch:
                        t_data_start = time.perf_counter()
                        continue

                    pixel_values = batch["pixel_values"]
                    labels = batch["labels"]

                    t_trans_start = time.perf_counter()
                    if not use_prefetcher:
                        pixel_values = pixel_values.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                    t_transfer = time.perf_counter() - t_trans_start

                    t_fwd_start = time.perf_counter()
                    with autocast_ctx:
                        outputs = self.model(pixel_values=pixel_values, labels=labels)
                        raw_loss = outputs.loss
                        loss = raw_loss / max(1, stage_cfg.gradient_accumulation_steps)
                    t_fwd = time.perf_counter() - t_fwd_start

                    t_bwd_start = time.perf_counter()
                    if self.scaler is not None and self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    t_bwd = time.perf_counter() - t_bwd_start
                    accum_loss_t = raw_loss.detach() if accum_loss_t is None else accum_loss_t + raw_loss.detach()

                    t_opt_start = time.perf_counter()
                    is_opt_step = ((step + 1) % stage_cfg.gradient_accumulation_steps == 0) or ((step + 1) == len(train_loader))
                    if is_opt_step:
                        if self.scaler is not None and self.scaler.is_enabled():
                            self.scaler.unscale_(optimizer)
                            if trainable_params:
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            if trainable_params:
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                        if accum_loss_t is not None:
                            step_loss = accum_loss_t / max(1, (step % stage_cfg.gradient_accumulation_steps + 1))
                            epoch_loss_t = step_loss if epoch_loss_t is None else epoch_loss_t + step_loss
                            n_opt += 1
                        accum_loss_t = None

                        if self.config.empty_cache_steps > 0 and self.global_step % self.config.empty_cache_steps == 0:
                            if self.device.type == "mps" and hasattr(torch, "mps"):
                                torch.mps.empty_cache()

                    t_opt = time.perf_counter() - t_opt_start if is_opt_step else 0.0

                    num_samples = len(pixel_values) if hasattr(pixel_values, "__len__") else stage_cfg.micro_batch_size
                    if self.profiler is not None:
                        self.profiler.record_step(
                            t_data=t_data,
                            t_transfer=t_transfer,
                            t_fwd=t_fwd,
                            t_bwd=t_bwd,
                            t_opt=t_opt,
                            num_samples=num_samples,
                        )

                    t_data_start = time.perf_counter()
            finally:
                if use_prefetcher and hasattr(data_iter, "close"):
                    data_iter.close()

            epoch_train_loss = float((epoch_loss_t / max(1, n_opt)).item()) if epoch_loss_t is not None else 0.0

            # Evaluation
            val_cer, val_wer, val_loss = self._evaluate_stage(val_ds, stage_cfg.eval_batch_size, collator)

            # Log to continuous LossLogger
            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else stage_cfg.learning_rate
            elapsed = time.time() - t_stage_start
            self.loss_logger.log_epoch(
                epoch=self.global_epoch,
                train_loss=round(epoch_train_loss, 4),
                val_cer=round(val_cer, 4),
                val_wer=round(val_wer, 4),
                val_loss=round(val_loss, 4),
                learning_rate=current_lr,
                step=self.global_step,
                elapsed_time=round(elapsed, 2),
                stage=stage_cfg.stage_name,
                stage_epoch=stage_epoch,
            )

            # Atomic save periodic checkpoint
            ckpt_path = stage_out / f"checkpoint_epoch_{stage_epoch}.pt"
            self._save_atomic_checkpoint(
                ckpt_path, stage_cfg.stage_name, stage_epoch, optimizer, scheduler, val_cer, val_loss
            )

            # Check best for this stage
            is_best_stage = val_cer < best_stage_cer or (val_cer == best_stage_cer and epoch_train_loss < best_stage_loss)
            if is_best_stage or stage_epoch == 1:
                best_stage_cer = val_cer
                best_stage_loss = val_loss
                best_pt = stage_out / "best_model.pt"
                self._save_atomic_checkpoint(
                    best_pt, stage_cfg.stage_name, stage_epoch, optimizer, scheduler, val_cer, val_loss, is_best=True
                )
                self._save_hf_directory(stage_out / "best_model_hf")

            # Check overall best across curriculum
            if val_cer < self.overall_best_cer or (val_cer == self.overall_best_cer and epoch_train_loss < self.overall_best_loss):
                self.overall_best_cer = val_cer
                self.overall_best_wer = val_wer
                self.overall_best_loss = val_loss
                self.overall_best_checkpoint = stage_out / "best_model.pt"

            logger.info(
                f"[{stage_cfg.stage_name}] Epoch {stage_epoch}/{stage_cfg.num_epochs} "
                f"(Global Epoch {self.global_epoch}) - Train Loss: {epoch_train_loss:.4f} | Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f}"
            )

        if self.profiler is not None:
            self.profiler.export_json(stage_out / "step_profiler_summary.json")
            self.profiler.export_csv(stage_out / "step_timings.csv")

        return {
            "stage_name": stage_cfg.stage_name,
            "epochs": stage_cfg.num_epochs,
            "best_cer": round(best_stage_cer, 4),
            "best_loss": round(best_stage_loss, 4),
            "output_dir": str(stage_out),
            "best_checkpoint": str(stage_out / "best_model.pt"),
            "best_hf_dir": str(stage_out / "best_model_hf"),
        }

    def _evaluate_stage(
        self,
        val_dataset: Optional[OCRDataset],
        eval_batch_size: int,
        collator: OCRDataCollator,
    ) -> Tuple[float, float, float]:
        """Evaluate current model on validation dataset."""
        if val_dataset is None or len(val_dataset) == 0:
            return 0.0, 0.0, 0.0

        self.model.eval()
        loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collator)
        total_val_loss = 0.0
        predictions: List[str] = []
        references: List[str] = []
        autocast_ctx = get_autocast_context(self.device, mixed_precision=self.config.mixed_precision)

        with torch.inference_mode():
            for batch in loader:
                if "pixel_values" not in batch:
                    continue
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch.get("labels")

                with autocast_ctx:
                    if labels is not None:
                        outputs = self.model(pixel_values=pixel_values, labels=labels.to(self.device, non_blocking=True))
                        total_val_loss += float(outputs.loss.detach().item())
                    generated_ids = self.model.generate(pixel_values, max_new_tokens=64)

                if hasattr(self.processor, "batch_decode"):
                    pred_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                elif hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
                    pred_texts = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                else:
                    pred_texts = [f"pred_{i}" for i in range(len(generated_ids))]

                ref_texts = batch.get("texts", [])
                predictions.extend([p.strip() for p in pred_texts])
                references.extend([r.strip() for r in ref_texts])

        import jiwer
        mean_cer = float(jiwer.cer(references, predictions)) if references else 0.0
        mean_wer = float(jiwer.wer(references, predictions)) if references else 0.0
        mean_loss = total_val_loss / max(1, len(loader))
        return mean_cer, mean_wer, mean_loss

    def _save_atomic_checkpoint(
        self,
        checkpoint_path: Path,
        stage_name: str,
        stage_epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        val_cer: float,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """Atomically persist PyTorch checkpoint state dict."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.parent / f".tmp_{os.getpid()}_{checkpoint_path.name}"

        payload = {
            "stage": stage_name,
            "stage_epoch": stage_epoch,
            "global_epoch": self.global_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_cer": val_cer,
            "best_loss": val_loss,
            "config": asdict(self.config),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        torch.save(payload, tmp_path)
        os.replace(tmp_path, checkpoint_path)

    def _save_hf_directory(self, hf_dir: Path) -> None:
        """Export Hugging Face format artifacts."""
        hf_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.model.save_pretrained(hf_dir)
            if hasattr(self.processor, "save_pretrained"):
                self.processor.save_pretrained(hf_dir)
        except Exception as e:
            logger.warning(f"Hugging Face save_pretrained failed ({e}). Writing config fallback.")
            (hf_dir / "config.json").write_text(json.dumps({"model_type": "vision-encoder-decoder"}))

    def _promote_global_best_model(self) -> None:
        """Copy overall best model to checkpoints/best_model.pt and best_model_hf/."""
        root_best_pt = self.root_output_dir / "best_model.pt"
        root_best_hf = self.root_output_dir / "best_model_hf"

        if self.overall_best_checkpoint and self.overall_best_checkpoint.exists():
            shutil.copy2(self.overall_best_checkpoint, root_best_pt)
            stage_hf_dir = self.overall_best_checkpoint.parent / "best_model_hf"
            if stage_hf_dir.exists():
                if root_best_hf.exists():
                    shutil.rmtree(root_best_hf)
                shutil.copytree(stage_hf_dir, root_best_hf)
            logger.info(f"Promoted overall best checkpoint from {self.overall_best_checkpoint} to {root_best_pt}")
        elif not root_best_pt.exists() and self.model is not None:
            # Fallback atomic save
            torch.save({"model_state_dict": self.model.state_dict()}, root_best_pt)
            self._save_hf_directory(root_best_hf)


def main() -> None:
    """CLI entry point for Multi-Stage Curriculum Training."""
    parser = argparse.ArgumentParser(description="TrOCR-Large Multi-Stage Curriculum Fine-Tuning on Apple Silicon MPS")
    parser.add_argument("--config", type=str, default=None, help="Path to CurriculumConfig JSON file")
    parser.add_argument("--model-name", type=str, default="microsoft/trocr-large-handwritten")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--stage1-manifest", type=str, default="data/reference_handwriting/train_manifest.jsonl")
    parser.add_argument("--stage2-manifest", type=str, default="data/reference_handwriting/train_manifest.jsonl")
    parser.add_argument("--val-manifest", type=str, default="data/reference_handwriting/val_manifest.jsonl")
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--stage1-lr", type=float, default=5e-5)
    parser.add_argument("--stage2-lr", type=float, default=1.5e-5)
    parser.add_argument("--freeze-encoder-layers", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mixed-precision", type=str, default="fp16")
    parser.add_argument("--resume-stage", type=str, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        curriculum_cfg = CurriculumConfig.from_json(args.config)
    else:
        stage1 = CurriculumStageConfig(
            stage_name="stage1_general_adaptation",
            dataset_manifest=args.stage1_manifest,
            val_manifest=args.val_manifest,
            num_epochs=args.stage1_epochs,
            learning_rate=args.stage1_lr,
            min_lr=1e-6,
            warmup_ratio=0.05,
            freeze_encoder_layers=0,
            gradient_accumulation_steps=8,
            micro_batch_size=4,
            output_dir=os.path.join(args.output_dir, "stage1_general_adaptation"),
        )
        stage2 = CurriculumStageConfig(
            stage_name="stage2_doctor_specialization",
            dataset_manifest=args.stage2_manifest,
            val_manifest=args.val_manifest,
            num_epochs=args.stage2_epochs,
            learning_rate=args.stage2_lr,
            min_lr=5e-7,
            warmup_ratio=0.03,
            freeze_encoder_layers=args.freeze_encoder_layers,
            gradient_accumulation_steps=8,
            micro_batch_size=4,
            category_filter=["prescription_item", "clinical_note", "doctor_signature"],
            output_dir=os.path.join(args.output_dir, "stage2_doctor_specialization"),
        )
        curriculum_cfg = CurriculumConfig(
            model_name_or_path=args.model_name,
            root_output_dir=args.output_dir,
            device=args.device,
            mixed_precision=args.mixed_precision,
            stages=[stage1, stage2],
        )

    trainer = MultiStageCurriculumTrainer(config=curriculum_cfg)
    result = trainer.execute_curriculum(
        resume_stage=args.resume_stage,
        resume_checkpoint=args.resume_checkpoint,
    )
    print("Curriculum Execution Complete:", result.to_dict())


if __name__ == "__main__":
    main()

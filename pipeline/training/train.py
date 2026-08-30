"""
pipeline/training/train.py
Standalone, reproducible fine-tuning engine for TrOCR and Vision-Encoder-Decoder models.
Optimized for Apple Silicon MPS GPU acceleration, mixed precision autocast, gradient accumulation,
dynamic padding collation, and robust checkpoint management.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    TrOCRProcessor,
    ViTConfig,
    ViTImageProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from pipeline.dataset.dataset_loader import (
    HandwritingSample,
    IAMDatasetParser,
    MedicalPrescriptionDatasetLoader,
    MedicalPrescriptionSample,
)
from pipeline.training.config import TrainingConfig
from pipeline.training.dataset import OCRDataCollator, OCRDataset, create_dummy_processor, load_trocr_processor
from pipeline.training.loss_logger import LossLogger
from pipeline.training.prefetcher import AsyncDevicePrefetcher
from pipeline.training.profiler import StepTiming, TrainingStepProfiler

logger = logging.getLogger("pipeline.training.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def get_optimal_device(requested_device: Optional[str] = None) -> torch.device:
    """
    Resolve target PyTorch compute device.
    Prefers Apple Silicon MPS if available, falls back safely to CUDA or CPU.
    """
    if requested_device and requested_device.lower() != "auto":
        req = requested_device.lower()
        if req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            return torch.device("mps")
        elif req == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif req == "cpu":
            return torch.device("cpu")
        else:
            logger.warning(f"Requested device '{requested_device}' not available or invalid. Falling back to auto.")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_autocast_context(device: torch.device, mixed_precision: str = "none"):
    """
    Return device-appropriate autocast mixed-precision context manager.
    Supports FP16/BF16 on MPS/CUDA and BFloat16 on CPU.
    """
    mp = mixed_precision.lower() if isinstance(mixed_precision, str) else ("fp16" if mixed_precision else "none")
    if mp in ("none", "fp32", "false"):
        return nullcontext()

    if device.type == "mps":
        try:
            dtype = torch.bfloat16 if mp == "bf16" else torch.float16
            return torch.autocast(device_type="mps", dtype=dtype, enabled=True)
        except Exception:
            return nullcontext()
    elif device.type == "cuda":
        dtype = torch.bfloat16 if mp == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
    elif device.type == "cpu":
        if mp == "bf16":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def create_tiny_mock_model(
    vocab_size: int = 100,
    image_size: int = 64,
    attn_implementation: str = "sdpa",
) -> VisionEncoderDecoderModel:
    """Instantiate a lightweight, 2-layer in-memory model for 100% offline testing."""
    enc_cfg = ViTConfig(
        image_size=image_size,
        patch_size=16,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        _attn_implementation=attn_implementation,
    )
    dec_cfg = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        is_decoder=True,
        add_cross_attention=True,
        _attn_implementation=attn_implementation,
    )
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc_cfg, dec_cfg)
    config.decoder_start_token_id = 0
    config.pad_token_id = 1
    config.eos_token_id = 2
    config._attn_implementation = attn_implementation
    model = VisionEncoderDecoderModel(config=config)
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = attn_implementation
    if hasattr(model, "encoder") and hasattr(model.encoder, "config") and hasattr(model.encoder.config, "_attn_implementation"):
        model.encoder.config._attn_implementation = attn_implementation
    if hasattr(model, "decoder") and hasattr(model.decoder, "config") and hasattr(model.decoder.config, "_attn_implementation"):
        model.decoder.config._attn_implementation = attn_implementation
    return model


def configure_gradient_checkpointing(model: VisionEncoderDecoderModel, enabled: bool = True) -> None:
    """Enable or disable gradient checkpointing and disable KV-cache for training."""
    if not enabled or model is None:
        return

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception as e:
            logger.debug(f"model.gradient_checkpointing_enable failed: {e}")

    if hasattr(model, "encoder") and hasattr(model.encoder, "gradient_checkpointing_enable"):
        try:
            model.encoder.gradient_checkpointing_enable()
        except Exception as e:
            logger.debug(f"encoder.gradient_checkpointing_enable failed: {e}")

    if hasattr(model, "decoder") and hasattr(model.decoder, "gradient_checkpointing_enable"):
        try:
            model.decoder.gradient_checkpointing_enable()
        except Exception as e:
            logger.debug(f"decoder.gradient_checkpointing_enable failed: {e}")

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
        model.decoder.config.use_cache = False


def freeze_encoder_layers(model: VisionEncoderDecoderModel, num_layers: int = 0) -> None:
    """Freeze patch embeddings and bottom N Transformer layers of the ViT encoder."""
    if model is None or num_layers <= 0:
        return

    # Freeze ViT patch embeddings if available
    if hasattr(model, "encoder") and hasattr(model.encoder, "embeddings"):
        for param in model.encoder.embeddings.parameters():
            param.requires_grad = False

    # Freeze bottom N encoder layers polymorphically across transformers versions
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

    if layers is not None:
        num_to_freeze = min(num_layers, len(layers))
        for i in range(num_to_freeze):
            for param in layers[i].parameters():
                param.requires_grad = False
        logger.info(f"Frozen bottom {num_to_freeze}/{len(layers)} ViT encoder layers.")


def load_trocr_model(
    model_name_or_path: str,
    processor: Any,
    device: torch.device,
    gradient_checkpointing: bool = True,
    freeze_encoder_layers_count: int = 0,
    attn_implementation: str = "sdpa",
) -> VisionEncoderDecoderModel:
    """
    Instantiate VisionEncoderDecoderModel from pretrained weights or configure config-based instance.
    """
    try:
        try:
            model = VisionEncoderDecoderModel.from_pretrained(
                model_name_or_path,
                attn_implementation=attn_implementation,
            )
        except Exception:
            model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    except Exception as e:
        logger.warning(f"Could not load pretrained weights from '{model_name_or_path}' ({e}). Initializing config model.")
        vocab_size = len(getattr(processor, "tokenizer", processor)) if hasattr(processor, "tokenizer") or hasattr(processor, "__len__") else 1000
        enc_cfg = ViTConfig(
            image_size=384,
            patch_size=16,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            _attn_implementation=attn_implementation,
        )
        dec_cfg = RobertaConfig(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            is_decoder=True,
            add_cross_attention=True,
            _attn_implementation=attn_implementation,
        )
        v_cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc_cfg, dec_cfg)
        v_cfg._attn_implementation = attn_implementation
        model = VisionEncoderDecoderModel(config=v_cfg)

    # Explicitly configure SDPA attention on model and submodules
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = attn_implementation
    if hasattr(model, "encoder") and hasattr(model.encoder, "config") and hasattr(model.encoder.config, "_attn_implementation"):
        model.encoder.config._attn_implementation = attn_implementation
    if hasattr(model, "decoder") and hasattr(model.decoder, "config") and hasattr(model.decoder.config, "_attn_implementation"):
        model.decoder.config._attn_implementation = attn_implementation

    # Configure special token IDs
    tok = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tok, "pad_token_id", 1) or 1
    bos_id = 0  # <s> for RoBERTa decoder
    eos_id = getattr(tok, "eos_token_id", 2) or getattr(tok, "sep_token_id", 2) or 2

    model.config.decoder_start_token_id = bos_id
    model.config.pad_token_id = pad_id
    model.config.eos_token_id = eos_id
    if hasattr(model.config, "decoder") and hasattr(model.config.decoder, "vocab_size"):
        model.config.vocab_size = model.config.decoder.vocab_size

    # Clean up deprecated generation attributes on model.config for transformers 5.x compatibility
    for attr in ["max_length", "early_stopping", "no_repeat_ngram_size", "length_penalty", "num_beams"]:
        if hasattr(model.config, attr):
            try:
                delattr(model.config, attr)
            except Exception:
                pass

    # Configure generation_config
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.decoder_start_token_id = bos_id
        model.generation_config.pad_token_id = pad_id
        model.generation_config.eos_token_id = eos_id
        model.generation_config.max_new_tokens = 64
        model.generation_config.no_repeat_ngram_size = 3

    if gradient_checkpointing:
        configure_gradient_checkpointing(model, enabled=True)

    if freeze_encoder_layers_count > 0:
        freeze_encoder_layers(model, num_layers=freeze_encoder_layers_count)

    model.to(device)
    return model


class TrOCRTrainer:
    """
    Core Fine-Tuning Engine for TrOCR on Apple Silicon MPS with unified memory management.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model: Optional[VisionEncoderDecoderModel] = None,
        processor: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        loss_logger: Optional[LossLogger] = None,
    ) -> None:
        self.config = config if config is not None else TrainingConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = self.config.resolved_device
        logger.info(f"Initialized TrOCRTrainer on compute device: {self.device}")

        if processor is not None:
            self.processor = processor
        else:
            self.processor = load_trocr_processor(self.config.model_name_or_path)

        attn_impl = getattr(self.config, "attn_implementation", "sdpa")
        if model is not None:
            self.model = model.to(self.device)
            if hasattr(self.model.config, "_attn_implementation"):
                self.model.config._attn_implementation = attn_impl
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "config") and hasattr(self.model.encoder.config, "_attn_implementation"):
                self.model.encoder.config._attn_implementation = attn_impl
            if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "config") and hasattr(self.model.decoder.config, "_attn_implementation"):
                self.model.decoder.config._attn_implementation = attn_impl
            if self.config.gradient_checkpointing:
                configure_gradient_checkpointing(self.model, enabled=True)
            if self.config.freeze_encoder_layers > 0:
                freeze_encoder_layers(self.model, num_layers=self.config.freeze_encoder_layers)
        else:
            self.model = load_trocr_model(
                self.config.model_name_or_path,
                self.processor,
                self.device,
                gradient_checkpointing=self.config.gradient_checkpointing,
                freeze_encoder_layers_count=self.config.freeze_encoder_layers,
                attn_implementation=attn_impl,
            )

        if self.config.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                logger.info("Enabled torch.compile on VisionEncoderDecoderModel.")
            except Exception as e:
                logger.debug(f"torch.compile skipped ({e}).")

        device_type = self.device.type
        scaler_enabled = (self.config.mixed_precision.lower() in ("fp16", "bf16")) and device_type == "cuda"
        self.scaler = (
            torch.amp.GradScaler(device_type, enabled=scaler_enabled)
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and device_type == "cuda"
            else None
        )

        self.trainable_params: List[torch.nn.Parameter] = [p for p in self.model.parameters() if p.requires_grad]

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.logger = loss_logger if loss_logger is not None else LossLogger(
            log_dir=self.output_dir,
            csv_filename="losses.csv",
            plot_filename="loss_curves.png",
        )

        self.collator = OCRDataCollator(processor=self.processor)

        self.global_step = 0
        self.start_epoch = 1
        self.best_cer = float("inf")
        self.best_loss = float("inf")
        self.checkpoint_history: List[Path] = []

        if getattr(self.config, "enable_step_profiling", True):
            self.profiler: Optional[TrainingStepProfiler] = TrainingStepProfiler(
                warmup_steps=2,
                device=self.device,
                enabled=True,
            )
        else:
            self.profiler = None

    def _setup_optimizer_and_scheduler(self, total_training_steps: int) -> Tuple[torch.optim.Optimizer, Any]:
        """Configure AdamW optimizer with weight decay exclusion and warmup scheduler."""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias", "layer_norm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

        warmup_steps = (
            self.config.warmup_steps
            if self.config.warmup_steps is not None
            else int(total_training_steps * self.config.warmup_ratio)
        )
        warmup_steps = max(0, warmup_steps)

        if self.config.lr_scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_training_steps)
            )
        elif self.config.lr_scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_training_steps)
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        return optimizer, scheduler

    def train(self, resume_from_checkpoint: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Execute full training loop with periodic validation, metrics logging, and checkpointing.
        """
        if self.train_dataset is None:
            raise ValueError("train_dataset must be provided to execute training.")

        num_workers = self.config.num_workers
        if len(self.train_dataset) < 32 and num_workers > 0:
            num_workers = 0

        train_loader_kwargs: Dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "shuffle": True,
            "collate_fn": self.collator,
            "num_workers": num_workers,
            "pin_memory": (self.device.type == "cuda" or self.config.pin_memory),
            "drop_last": self.config.dataloader_drop_last,
        }
        if num_workers > 0:
            train_loader_kwargs["persistent_workers"] = self.config.persistent_workers
            if self.config.prefetch_factor is not None:
                train_loader_kwargs["prefetch_factor"] = self.config.prefetch_factor

        train_loader = DataLoader(self.train_dataset, **train_loader_kwargs)

        steps_per_epoch = max(1, len(train_loader) // self.config.gradient_accumulation_steps)
        total_steps = self.config.max_steps or (steps_per_epoch * self.config.num_train_epochs)

        optimizer, scheduler = self._setup_optimizer_and_scheduler(total_steps)

        ckpt_to_resume = resume_from_checkpoint or self.config.resume_from_checkpoint
        if ckpt_to_resume:
            self.load_checkpoint(ckpt_to_resume, optimizer, scheduler)

        logger.info(f"Starting training: {self.config.num_train_epochs} epochs, {total_steps} total optimizer steps.")
        t_start = time.time()

        for epoch in range(self.start_epoch, self.config.num_train_epochs + 1):
            epoch_loss = self._train_epoch(train_loader, optimizer, scheduler, epoch)

            # Validation Round
            val_cer, val_wer, val_loss = 0.0, 0.0, 0.0
            if self.val_dataset is not None and len(self.val_dataset) > 0:
                val_cer, val_wer, val_loss = self.evaluate()

            # Record metrics
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=round(epoch_loss, 4),
                val_cer=round(val_cer, 4),
                val_wer=round(val_wer, 4),
                val_loss=round(val_loss, 4),
                learning_rate=current_lr,
                step=self.global_step,
                elapsed_time=round(elapsed, 2),
            )

            # Generate Updated Curves
            self.logger.plot_curves()

            # Save Periodic Checkpoint
            periodic_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
            self.save_checkpoint(periodic_path, epoch, optimizer, scheduler)
            self._manage_checkpoint_history(periodic_path)

            # Save Best Model Checkpoint
            is_best = False
            if self.config.metric_for_best_model == "val_loss":
                if val_loss < self.best_loss or (val_loss == self.best_loss and epoch_loss < self.best_loss):
                    is_best = True
                    self.best_loss = val_loss
            else:
                if val_cer < self.best_cer or (val_cer == self.best_cer and epoch_loss < self.best_loss):
                    is_best = True
                    self.best_cer = val_cer
                    self.best_loss = epoch_loss

            if is_best or epoch == 1:
                if is_best:
                    self.best_cer = val_cer
                    self.best_loss = epoch_loss
                best_pt_path = self.output_dir / "best_model.pt"
                self.save_checkpoint(best_pt_path, epoch, optimizer, scheduler, is_best=True)
                self._save_hf_model(self.output_dir / "best_model_hf")

            # Clean memory cache
            self.config.manage_memory(step=self.global_step)
            if self.device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()

            logger.info(
                f"Epoch {epoch}/{self.config.num_train_epochs} Finished - "
                f"Train Loss: {epoch_loss:.4f} | Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f}"
            )

            # Check for external stop signal
            stop_flag = self.output_dir / "stop_after_epoch.flag"
            if stop_flag.exists() or (self.output_dir / "stop_training.flag").exists():
                logger.info(f"Stop flag detected ({stop_flag.name}) — cleanly stopping training after Epoch {epoch}.")
                try:
                    if stop_flag.exists():
                        stop_flag.unlink()
                except Exception:
                    pass
                break

        # Write final training state
        self._write_state_json(completed=True)

        profiler_summary = None
        profiler_json = None
        profiler_csv = None
        if self.profiler is not None:
            logger.info("\n" + self.profiler.format_summary())
            profiler_json = str(self.output_dir / "step_profiler_summary.json")
            profiler_csv = str(self.output_dir / "step_timings.csv")
            self.profiler.export_json(profiler_json)
            self.profiler.export_csv(profiler_csv)
            profiler_summary = self.profiler.get_summary()

        return {
            "epochs": self.config.num_train_epochs,
            "final_loss": self.logger.history["train_loss"][-1] if self.logger.history["train_loss"] else 0.0,
            "final_cer": self.best_cer if self.best_cer != float("inf") else 0.0,
            "loss_csv": str(self.logger.csv_path),
            "loss_plot": str(self.logger.plot_path),
            "best_checkpoint": str(self.output_dir / "best_model.pt"),
            "best_hf_dir": str(self.output_dir / "best_model_hf"),
            "profiler_summary": profiler_summary,
            "profiler_json": profiler_json,
            "profiler_csv": profiler_csv,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
    ) -> float:
        """Run single training epoch with gradient accumulation and clipping."""
        self.model.train()
        total_loss = 0.0
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = get_autocast_context(self.device, mixed_precision=self.config.mixed_precision)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        use_prefetcher = getattr(self.config, "use_async_prefetcher", True)
        prefetch_q_size = getattr(self.config, "prefetch_queue_size", 3)

        if use_prefetcher:
            data_iter = AsyncDevicePrefetcher(
                loader,
                device=self.device,
                mixed_precision=self.config.mixed_precision,
                queue_size=prefetch_q_size,
            )
        else:
            data_iter = loader

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
                if not use_prefetcher or pixel_values.device != self.device:
                    pixel_values = pixel_values.to(self.device, non_blocking=self.config.non_blocking)
                    labels = labels.to(self.device, non_blocking=self.config.non_blocking)
                t_transfer = time.perf_counter() - t_trans_start

                t_fwd_start = time.perf_counter()
                with autocast_ctx:
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    raw_loss = outputs.loss
                    loss = raw_loss / self.config.gradient_accumulation_steps
                t_fwd = time.perf_counter() - t_fwd_start

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Non-finite loss detected at epoch {epoch} step {step}. Skipping gradient update.")
                    optimizer.zero_grad(set_to_none=True)
                    t_data_start = time.perf_counter()
                    continue

                t_bwd_start = time.perf_counter()
                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                t_bwd = time.perf_counter() - t_bwd_start
                accum_loss += raw_loss.item()

                t_opt_start = time.perf_counter()
                is_opt_step = ((step + 1) % self.config.gradient_accumulation_steps == 0) or ((step + 1) == len(loader))
                if is_opt_step:
                    if self.scaler is not None and self.scaler.is_enabled():
                        self.scaler.unscale_(optimizer)
                        if trainable_params:
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.config.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        if trainable_params:
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.config.max_grad_norm)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    recent_loss = accum_loss / max(1, (step % self.config.gradient_accumulation_steps + 1))
                    total_loss += recent_loss
                    accum_loss = 0.0

                    if self.global_step % self.config.logging_steps == 0:
                        lr = optimizer.param_groups[0]["lr"]
                        mps_mem = ""
                        if self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                            mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                            mps_mem = f" | Metal GPU Memory: {mem_mb:.1f} MB"
                        logger.info(
                            f"Epoch {epoch} [Step {self.global_step}] Loss: {recent_loss:.4f} | LR: {lr:.2e}{mps_mem}"
                        )
                        if hasattr(self.logger, "log_step"):
                            self.logger.log_step(
                                step=self.global_step,
                                loss=recent_loss,
                                lr=lr,
                                epoch=epoch + (step / max(1, len(loader))),
                            )

                    # Inter-step unified memory cleanup
                    if self.config.empty_cache_steps > 0 and self.global_step % self.config.empty_cache_steps == 0:
                        self.config.manage_memory(step=self.global_step)

                    if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                        break

                t_opt = time.perf_counter() - t_opt_start if is_opt_step else 0.0

                num_samples = len(pixel_values) if hasattr(pixel_values, "__len__") else self.config.batch_size
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

        num_optimizer_steps = max(1, len(loader) // self.config.gradient_accumulation_steps)
        return total_loss / num_optimizer_steps

    def evaluate(self) -> Tuple[float, float, float]:
        """
        Evaluate current model on validation dataset computing CER, WER, and loss.
        """
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return 0.0, 0.0, 0.0

        self.model.eval()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        total_val_loss = 0.0
        predictions: List[str] = []
        references: List[str] = []
        autocast_ctx = get_autocast_context(self.device, mixed_precision=self.config.mixed_precision)

        with torch.inference_mode():
            for batch in val_loader:
                if "pixel_values" not in batch:
                    continue

                pixel_values = batch["pixel_values"].to(self.device, non_blocking=self.config.non_blocking)
                labels = batch.get("labels")

                with autocast_ctx:
                    if labels is not None:
                        outputs = self.model(pixel_values=pixel_values, labels=labels.to(self.device, non_blocking=self.config.non_blocking))
                        total_val_loss += outputs.loss.item()

                    # Autoregressive generation
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_new_tokens=64,
                        num_beams=self.config.num_beams,
                    )

                if hasattr(self.processor, "batch_decode"):
                    pred_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                elif hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
                    pred_texts = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                else:
                    pred_texts = [f"pred_{i}" for i in range(len(generated_ids))]

                ref_texts = batch.get("texts", [])
                predictions.extend([p.strip() for p in pred_texts])
                references.extend([r.strip() for r in ref_texts])

        # Compute CER and WER
        if predictions and references:
            try:
                import jiwer
                mean_cer = float(jiwer.cer(references, predictions))
                mean_wer = float(jiwer.wer(references, predictions))
            except Exception:
                # Fallback simple error calculation
                cer_scores = []
                for p, r in zip(predictions, references):
                    if not r:
                        cer_scores.append(0.0 if not p else 1.0)
                    else:
                        cer_scores.append(abs(len(p) - len(r)) / max(1, len(r)))
                mean_cer = float(np.mean(cer_scores)) if cer_scores else 0.0
                mean_wer = mean_cer
        else:
            mean_cer, mean_wer = 0.0, 0.0

        mean_loss = total_val_loss / max(1, len(val_loader))
        return mean_cer, mean_wer, mean_loss

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        is_best: bool = False,
    ) -> None:
        """Serialize PyTorch weights and training state dictionary."""
        p = Path(checkpoint_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.current_epoch = epoch

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_cer": self.best_cer,
            "best_loss": self.best_loss,
            "config": asdict(self.config) if is_dataclass(self.config) else dict(self.config),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        torch.save(state, p)
        self._write_state_json(epoch=epoch, last_ckpt=str(p))

    def _save_hf_model(self, hf_dir: Path) -> None:
        """Export model and processor in Hugging Face pretrained format."""
        hf_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.model.save_pretrained(hf_dir)
            if hasattr(self.processor, "save_pretrained"):
                self.processor.save_pretrained(hf_dir)
        except Exception as e:
            logger.warning(f"save_pretrained failed ({e}), writing standard config.")
            (hf_dir / "config.json").write_text(json.dumps({"model_type": "vision-encoder-decoder"}))

    def _manage_checkpoint_history(self, new_checkpoint: Path) -> None:
        """Prune old periodic checkpoints if save_total_limit is reached."""
        self.checkpoint_history.append(new_checkpoint)
        limit = self.config.save_total_limit
        if limit is not None and limit > 0 and len(self.checkpoint_history) > limit:
            to_remove = self.checkpoint_history[:-limit]
            self.checkpoint_history = self.checkpoint_history[-limit:]
            for old_ckpt in to_remove:
                if old_ckpt.exists() and "best_model" not in old_ckpt.name:
                    try:
                        old_ckpt.unlink()
                    except OSError:
                        pass

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        """Restore model weights, optimizer, and training step state from checkpoint."""
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        logger.info(f"Loading checkpoint from {p}")
        state = torch.load(p, map_location=self.device)

        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            if optimizer and "optimizer_state_dict" in state and state["optimizer_state_dict"]:
                try:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                except Exception as e:
                    logger.warning(f"Could not restore optimizer state ({e}).")
            if scheduler and "scheduler_state_dict" in state and state["scheduler_state_dict"]:
                try:
                    scheduler.load_state_dict(state["scheduler_state_dict"])
                except Exception as e:
                    logger.warning(f"Could not restore scheduler state ({e}).")
            self.start_epoch = state.get("epoch", 0) + 1
            self.current_epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
            self.best_cer = state.get("best_cer", float("inf"))
            self.best_loss = state.get("best_loss", float("inf"))
        elif isinstance(state, dict):
            # Raw state dict
            self.model.load_state_dict(state)

    def _write_state_json(self, epoch: Optional[int] = None, last_ckpt: Optional[str] = None, completed: bool = False) -> None:
        """Persist high-level training state metadata."""
        current_ep = epoch if epoch is not None else getattr(self, "current_epoch", self.start_epoch)
        state_data = {
            "epoch": current_ep,
            "global_step": self.global_step,
            "best_cer": self.best_cer if self.best_cer != float("inf") else 0.0,
            "best_loss": self.best_loss if self.best_loss != float("inf") else 0.0,
            "best_checkpoint": str(self.output_dir / "best_model.pt"),
            "last_checkpoint": last_ckpt or str(self.output_dir / "best_model.pt"),
            "completed": completed,
        }
        with open(self.output_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="TrOCR Fine-Tuning Engine on Apple Silicon MPS")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML TrainingConfig file")
    parser.add_argument("--model-name", type=str, default="microsoft/trocr-small-handwritten")
    parser.add_argument("--output-dir", type=str, default="checkpoints/trocr-handwritten")
    parser.add_argument("--iam-lines", type=str, default=None, help="Path to IAM ascii/lines.txt")
    parser.add_argument("--iam-root", type=str, default=None, help="Path to IAM root directory")
    parser.add_argument("--manifest", type=str, default="data/reference_handwriting/train_manifest.jsonl", help="Path to training manifest JSON/JSONL/CSV")
    parser.add_argument("--val-manifest", type=str, default="data/reference_handwriting/val_manifest.jsonl", help="Path to validation manifest JSON/JSONL/CSV")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mixed-precision", type=str, default="none")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        config = TrainingConfig.from_json(args.config)
    else:
        config = TrainingConfig(
            model_name_or_path=args.model_name,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            mixed_precision=args.mixed_precision,
            resume_from_checkpoint=args.resume,
        )

    # Initialize datasets
    processor = load_trocr_processor(config.model_name_or_path)
    train_ds = None
    val_ds = None

    if args.iam_lines and os.path.exists(args.iam_lines):
        train_ds = OCRDataset.from_iam(args.iam_lines, args.iam_root, split="train", processor=processor)
        val_ds = OCRDataset.from_iam(args.iam_lines, args.iam_root, split="val", processor=processor)
    elif args.manifest and os.path.exists(args.manifest):
        train_ds = OCRDataset.from_manifest(args.manifest, processor=processor, is_training=True)
        if args.val_manifest and os.path.exists(args.val_manifest):
            val_ds = OCRDataset.from_manifest(args.val_manifest, processor=processor, is_training=False)
        else:
            val_ds = None
    else:
        raise FileNotFoundError(
            "No real handwriting dataset specified. Pass --manifest pointing at "
            "data/reference_handwriting/train_manifest.jsonl after downloading public corpora, "
            "or --iam-lines. Synthetic training data is disabled."
        )

    trainer = TrOCRTrainer(
        config=config,
        processor=processor,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    results = trainer.train()
    print("Training Complete:", results)


if __name__ == "__main__":
    main()

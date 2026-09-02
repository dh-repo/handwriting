"""Section 5.2 TrOCR config after load. Wrong decoder_start_token_id makes overfit fail."""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoImageProcessor, RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel


def resolve_htr_device() -> torch.device:
    """MPS first (with fallback), then CUDA, then CPU. Never device_map=auto."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    forced = os.environ.get("HTR_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_htr_processor(model_name_or_path: str) -> Any:
    """Load the matching TrOCRProcessor. Never fall back to DummyProcessor."""
    try:
        return TrOCRProcessor.from_pretrained(model_name_or_path)
    except Exception:
        image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
        return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def load_htr_model(model_name_or_path: str) -> Any:
    return VisionEncoderDecoderModel.from_pretrained(model_name_or_path)


def apply_trocr_generation_config(
    model: Any,
    processor: Any,
    max_length: int = 128,
    num_beams: int = 1,
) -> Any:
    tok = getattr(processor, "tokenizer", processor)
    cls_id = getattr(tok, "cls_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)
    sep_id = getattr(tok, "sep_token_id", None) or getattr(tok, "eos_token_id", None)
    if cls_id is None or pad_id is None or sep_id is None:
        raise ValueError("processor tokenizer is missing cls/pad/sep token ids")

    model.config.decoder_start_token_id = cls_id
    model.config.pad_token_id = pad_id
    model.config.eos_token_id = sep_id
    if hasattr(model.config, "decoder") and hasattr(model.config.decoder, "vocab_size"):
        model.config.vocab_size = model.config.decoder.vocab_size
    # transformers 5+ rejects generation knobs on model.config; keep them on generation_config.

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = cls_id
        model.generation_config.pad_token_id = pad_id
        model.generation_config.eos_token_id = sep_id
        model.generation_config.max_length = max_length
        if hasattr(model.generation_config, "max_new_tokens"):
            model.generation_config.max_new_tokens = None
        model.generation_config.num_beams = num_beams
        model.generation_config.early_stopping = bool(num_beams and num_beams > 1)
    return model

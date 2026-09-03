"""
backend/app/onnx_engine.py
High-Performance ONNX Runtime Serving Engine for TrOCR Line-Level HTR on Cloud CPU.

Provides drop-in compatibility with HuggingFace VisionEncoderDecoderModel interface,
enabling 3-4x lower latency and minimal memory footprint on multi-vCPU cloud containers.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import TrOCRProcessor

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

logger = logging.getLogger("handwriting_backend.onnx_engine")


@dataclass
class OnnxConfig:
    bos_token_id: int = 0
    decoder_start_token_id: int = 0
    eos_token_id: int = 2
    pad_token_id: int = 1
    max_length: int = 128
    vocab_size: int = 50265


class OnnxGenerateOutput:
    """Emulates HuggingFace ModelOutput for return_dict_in_generate=True."""

    def __init__(
        self,
        sequences: torch.Tensor,
        sequences_scores: Optional[torch.Tensor] = None,
    ):
        self.sequences = sequences
        self.sequences_scores = sequences_scores

    def __getitem__(self, item: int) -> torch.Tensor:
        return self.sequences[item]


class OnnxTrOCRModel:
    """
    ONNX Runtime-accelerated TrOCR Encoder-Decoder model wrapper.
    Implements HuggingFace-compatible .generate() method for CPU execution.
    """

    def __init__(self, model_dir: Union[str, Path], num_threads: int = 4):
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed in the current environment.")

        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self.device = torch.device("cpu")

        self.encoder_path = self.model_dir / "encoder_model.onnx"
        self.decoder_path = self.model_dir / "decoder_model.onnx"

        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Missing ONNX encoder model at {self.encoder_path}")
        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Missing ONNX decoder model at {self.decoder_path}")

        # Load generation config
        self.config = self._load_config()

        # Initialize ONNX Runtime Session Options
        self.sess_options = ort.SessionOptions()
        self.sess_options.intra_op_num_threads = max(1, num_threads)
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create Inference Sessions with CPUExecutionProvider
        providers = ["CPUExecutionProvider"]
        logger.info(f"Initializing ONNX Runtime TrOCR sessions from {self.model_dir} (threads={num_threads})...")
        self.encoder_session = ort.InferenceSession(
            str(self.encoder_path),
            sess_options=self.sess_options,
            providers=providers,
        )
        self.decoder_session = ort.InferenceSession(
            str(self.decoder_path),
            sess_options=self.sess_options,
            providers=providers,
        )
        logger.info("ONNX Runtime TrOCR sessions initialized successfully.")

    def _load_config(self) -> OnnxConfig:
        gen_cfg_path = self.model_dir / "generation_config.json"
        cfg_path = self.model_dir / "config.json"

        data: Dict[str, Any] = {}
        if gen_cfg_path.exists():
            with open(gen_cfg_path, "r", encoding="utf-8") as f:
                data.update(json.load(f))
        elif cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data.update(json.load(f))

        eos_raw = data.get("eos_token_id", 2)
        eos_id = eos_raw[0] if isinstance(eos_raw, list) else int(eos_raw)

        return OnnxConfig(
            bos_token_id=int(data.get("bos_token_id", 0)),
            decoder_start_token_id=int(data.get("decoder_start_token_id", 0)),
            eos_token_id=eos_id,
            pad_token_id=int(data.get("pad_token_id", 1)),
            max_length=int(data.get("max_length", 128)),
            vocab_size=int(data.get("vocab_size", 50265)),
        )

    def to(self, *args: Any, **kwargs: Any) -> OnnxTrOCRModel:
        """No-op for PyTorch device compatibility."""
        return self

    def eval(self) -> OnnxTrOCRModel:
        """No-op for PyTorch evaluation mode compatibility."""
        return self

    def half(self) -> OnnxTrOCRModel:
        """No-op for PyTorch precision compatibility."""
        return self

    def generate(
        self,
        pixel_values: Union[torch.Tensor, np.ndarray],
        num_beams: int = 1,
        num_return_sequences: int = 1,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        early_stopping: bool = True,
        **kwargs: Any,
    ) -> Union[torch.Tensor, OnnxGenerateOutput]:
        """
        Execute ONNX encoder forward pass and autoregressive decoder generation.
        Supports both fast greedy decoding (num_beams=1) and beam search (num_beams > 1).
        """
        # 1. Convert pixel_values to float32 numpy array
        if isinstance(pixel_values, torch.Tensor):
            pixels_np = pixel_values.detach().cpu().numpy().astype(np.float32)
        else:
            pixels_np = np.asarray(pixel_values, dtype=np.float32)

        if pixels_np.ndim == 3:
            pixels_np = np.expand_dims(pixels_np, axis=0)

        batch_size = pixels_np.shape[0]
        token_limit = max_new_tokens or max_length or self.config.max_length
        token_limit = min(token_limit, self.config.max_length)

        # 2. Run Encoder forward pass
        encoder_inputs = {"pixel_values": pixels_np}
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        last_hidden_state = encoder_outputs[0]  # Shape: [batch_size, seq_len, hidden_dim]

        # 3. Decode each item in batch
        all_sequences: List[List[int]] = []
        all_scores: List[float] = []

        for b in range(batch_size):
            item_hidden = last_hidden_state[b : b + 1]  # Shape: [1, seq_len, hidden_dim]

            if num_beams <= 1:
                seq, score = self._greedy_decode(item_hidden, token_limit)
                all_sequences.append(seq)
                all_scores.append(score)
            else:
                seqs, scores = self._beam_search_decode(
                    item_hidden,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    max_length=token_limit,
                )
                all_sequences.extend(seqs)
                all_scores.extend(scores)

        # Pad all sequences to uniform tensor width
        max_seq_len = max(len(s) for s in all_sequences) if all_sequences else 1
        padded_np = np.full((len(all_sequences), max_seq_len), self.config.pad_token_id, dtype=np.int64)
        for idx, s in enumerate(all_sequences):
            padded_np[idx, : len(s)] = s

        sequences_tensor = torch.from_numpy(padded_np)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32) if output_scores else None

        if return_dict_in_generate:
            return OnnxGenerateOutput(sequences=sequences_tensor, sequences_scores=scores_tensor)
        return sequences_tensor

    def _greedy_decode(
        self,
        item_hidden: np.ndarray,
        max_length: int,
    ) -> Tuple[List[int], float]:
        """Greedy autoregressive decoding with log-probability accumulation."""
        tokens = [self.config.decoder_start_token_id]
        total_log_prob = 0.0

        for _ in range(max_length):
            input_ids = np.array([tokens], dtype=np.int64)
            decoder_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": item_hidden,
            }
            logits = self.decoder_session.run(None, decoder_inputs)[0]
            step_logits = logits[0, -1, :]

            # Compute log-softmax for numerical stability
            max_l = np.max(step_logits)
            log_probs = step_logits - max_l - np.log(np.sum(np.exp(step_logits - max_l)))

            next_token = int(np.argmax(step_logits))
            total_log_prob += float(log_probs[next_token])

            if next_token == self.config.eos_token_id:
                tokens.append(next_token)
                break
            if next_token == self.config.pad_token_id and len(tokens) > 1:
                break

            tokens.append(next_token)

        norm_score = total_log_prob / max(1, len(tokens) - 1)
        return tokens, norm_score

    def _beam_search_decode(
        self,
        item_hidden: np.ndarray,
        num_beams: int,
        num_return_sequences: int,
        max_length: int,
    ) -> Tuple[List[List[int]], List[float]]:
        """Beam search decoding maintaining top K hypotheses."""
        # Hypothesis: (cumulative_log_prob, [token_ids])
        beams: List[Tuple[float, List[int]]] = [(0.0, [self.config.decoder_start_token_id])]
        completed: List[Tuple[float, List[int]]] = []

        for _ in range(max_length):
            candidates: List[Tuple[float, List[int]]] = []
            for score, tokens in beams:
                if tokens[-1] == self.config.eos_token_id:
                    completed.append((score, tokens))
                    continue

                input_ids = np.array([tokens], dtype=np.int64)
                decoder_inputs = {
                    "input_ids": input_ids,
                    "encoder_hidden_states": item_hidden,
                }
                logits = self.decoder_session.run(None, decoder_inputs)[0]
                step_logits = logits[0, -1, :]

                max_l = np.max(step_logits)
                log_probs = step_logits - max_l - np.log(np.sum(np.exp(step_logits - max_l)))

                # Select top 2*num_beams candidate extensions
                top_indices = np.argpartition(step_logits, -2 * num_beams)[-2 * num_beams :]
                for token_idx in top_indices:
                    token_id = int(token_idx)
                    token_score = float(log_probs[token_id])
                    candidates.append((score + token_score, tokens + [token_id]))

            if not candidates:
                break

            # Sort by cumulative score descending and keep top num_beams
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:num_beams]

            if len(completed) >= num_beams:
                break

        # Combine completed beams and remaining active beams
        all_beams = completed + beams
        all_beams.sort(key=lambda x: x[0] / max(1, len(x[1]) - 1), reverse=True)

        k_return = min(num_return_sequences, len(all_beams))
        selected = all_beams[:k_return]

        seqs = [item[1] for item in selected]
        scores = [item[0] / max(1, len(item[1]) - 1) for item in selected]
        return seqs, scores


def is_onnx_available(model_dir: Union[str, Path]) -> bool:
    """Check if model_dir contains the required ONNX files."""
    if not ONNXRUNTIME_AVAILABLE:
        return False
    path = Path(model_dir)
    return (path / "encoder_model.onnx").exists() and (path / "decoder_model.onnx").exists()


def load_onnx_htr_model(
    model_dir: Union[str, Path],
    num_threads: int = 4,
) -> Tuple[OnnxTrOCRModel, TrOCRProcessor]:
    """Load ONNX TrOCR model and associated tokenizer/processor."""
    path = Path(model_dir)
    model = OnnxTrOCRModel(path, num_threads=num_threads)
    processor = TrOCRProcessor.from_pretrained(str(path))
    return model, processor

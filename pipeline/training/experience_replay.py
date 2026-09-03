"""
pipeline/training/experience_replay.py
Experience Replay Dataset and Sampler for TrOCR Continuous LoRA Adaptation.

Combines incoming operator feedback lines from Darkroom UI manifest (manifest.jsonl)
with golden anchor lines (labels.tsv / images/) using an interleaved batch sampler
to guarantee an exact replay ratio (default 50% feedback : 50% anchor) and prevent
catastrophic forgetting.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
import random
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


def _maybe_augment(image: Image.Image) -> Image.Image:
    """Apply slight geometric and brightness distortions during training."""
    try:
        import albumentations as A
        import numpy as np
    except ImportError:
        return image

    arr = np.array(image)
    transform = A.Compose(
        [
            A.Affine(rotate=(-3, 3), shear=(-3, 3), scale=(0.97, 1.03), p=0.4),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.RandomBrightnessContrast(p=0.3),
        ]
    )
    return Image.fromarray(transform(image=arr)["image"])


class ExperienceReplayDataset(Dataset):
    """
    Dataset combining operator feedback corrections with golden anchor samples.

    Indices [0, len(feedback) - 1] correspond to feedback samples.
    Indices [len(feedback), len(feedback) + len(anchor) - 1] correspond to anchor samples.
    """

    def __init__(
        self,
        feedback_manifest_path: Optional[Union[str, Path]] = None,
        anchor_dir: Optional[Union[str, Path]] = None,
        processor: Optional[Any] = None,
        max_target_length: int = 128,
        augment: bool = False,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        max_feedback_samples: Optional[int] = None,
        max_anchor_samples: Optional[int] = None,
    ) -> None:
        self.feedback_manifest_path = Path(feedback_manifest_path) if feedback_manifest_path else None
        self.anchor_dir = Path(anchor_dir) if anchor_dir else None
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        self.transform = transform

        # Parse feedback samples from JSONL manifest
        self.feedback_samples: List[dict[str, Any]] = []
        if self.feedback_manifest_path and self.feedback_manifest_path.exists():
            self.feedback_samples = self._parse_feedback_manifest(
                self.feedback_manifest_path, max_samples=max_feedback_samples
            )

        # Parse golden anchor samples from labels.tsv
        self.anchor_samples: List[dict[str, Any]] = []
        if self.anchor_dir and self.anchor_dir.exists():
            self.anchor_samples = self._parse_anchor_dir(
                self.anchor_dir, max_samples=max_anchor_samples
            )

        self._num_feedback = len(self.feedback_samples)
        self._num_anchor = len(self.anchor_samples)

        logger.info(
            "Initialized ExperienceReplayDataset: %d feedback samples, %d anchor samples (total=%d)",
            self._num_feedback,
            self._num_anchor,
            len(self),
        )

    def _resolve_image_file(self, path_str: str, manifest_dir: Optional[Path] = None) -> Optional[Path]:
        """Resolve image path across absolute and relative candidate locations."""
        if not path_str:
            return None
        p = Path(path_str)
        if p.is_file():
            return p
        if manifest_dir:
            cand = manifest_dir / p
            if cand.is_file():
                return cand
            cand_crops = manifest_dir / "crops" / p.name
            if cand_crops.is_file():
                return cand_crops
        # Cwd candidates
        cwd_cand = Path.cwd() / p
        if cwd_cand.is_file():
            return cwd_cand
        data_cand = Path("data/feedback") / p
        if data_cand.is_file():
            return data_cand
        data_crops = Path("data/feedback/crops") / p.name
        if data_crops.is_file():
            return data_crops
        return None

    def _parse_feedback_manifest(
        self, manifest_path: Path, max_samples: Optional[int] = None
    ) -> List[dict[str, Any]]:
        samples: List[dict[str, Any]] = []
        manifest_dir = manifest_path.parent
        with open(manifest_path, "r", encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception as err:
                    logger.warning("Skipping malformed manifest line %d: %s", line_idx, err)
                    continue

                crop_path_str = (
                    data.get("line_crop")
                    or data.get("crop_path")
                    or data.get("image_path")
                    or data.get("relative_image_path")
                    or ""
                )
                text = (
                    data.get("operator_correction")
                    or data.get("text")
                    or data.get("transcription")
                    or data.get("ground_truth")
                    or data.get("label")
                    or ""
                )
                base64_crop = data.get("line_crop_base64")
                sample_id = data.get("feedback_id") or data.get("sample_id") or f"fb_{line_idx:05d}"

                resolved_path = self._resolve_image_file(crop_path_str, manifest_dir=manifest_dir)

                samples.append(
                    {
                        "image_path": str(resolved_path) if resolved_path else crop_path_str,
                        "text": str(text),
                        "sample_id": str(sample_id),
                        "line_crop_base64": base64_crop,
                        "source": "feedback",
                    }
                )
                if max_samples is not None and len(samples) >= max_samples:
                    break
        return samples

    def _parse_anchor_dir(self, anchor_dir: Path, max_samples: Optional[int] = None) -> List[dict[str, Any]]:
        samples: List[dict[str, Any]] = []
        labels_path = anchor_dir / "labels.tsv" if anchor_dir.is_dir() else anchor_dir
        if not labels_path.is_file():
            logger.warning("Anchor labels file not found: %s", labels_path)
            return samples

        base_dir = labels_path.parent
        images_dir = base_dir / "images"

        with open(labels_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                filename = parts[0]
                text = parts[1] if len(parts) > 1 else ""

                img_path = images_dir / filename if images_dir.exists() else base_dir / filename
                sample_id = Path(filename).stem or f"anchor_{line_idx:05d}"

                samples.append(
                    {
                        "image_path": str(img_path),
                        "text": str(text),
                        "sample_id": str(sample_id),
                        "source": "anchor",
                    }
                )
                if max_samples is not None and len(samples) >= max_samples:
                    break
        return samples

    @property
    def feedback_indices(self) -> List[int]:
        """Indices of all feedback samples in this dataset."""
        return list(range(self._num_feedback))

    @property
    def anchor_indices(self) -> List[int]:
        """Indices of all golden anchor samples in this dataset."""
        return list(range(self._num_feedback, self._num_feedback + self._num_anchor))

    def get_feedback_indices(self) -> List[int]:
        return self.feedback_indices

    def get_anchor_indices(self) -> List[int]:
        return self.anchor_indices

    def __len__(self) -> int:
        return self._num_feedback + self._num_anchor

    def _load_pil_image(self, record: dict[str, Any]) -> Image.Image:
        """Load image from disk file path, base64 payload, or fallback white canvas."""
        img_path = record.get("image_path")
        if img_path and Path(img_path).is_file():
            try:
                return Image.open(img_path).convert("RGB")
            except Exception as exc:
                logger.warning("Failed opening image file %s: %exc", img_path, exc)

        # Check base64 fallback
        b64 = record.get("line_crop_base64")
        if b64:
            try:
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                data = base64.b64decode(b64)
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as exc:
                logger.warning("Failed decoding base64 image: %s", exc)

        # Fallback blank image
        return Image.new("RGB", (384, 64), color=(255, 255, 255))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        if idx < self._num_feedback:
            record = self.feedback_samples[idx]
        else:
            record = self.anchor_samples[idx - self._num_feedback]

        image = self._load_pil_image(record)
        if self.augment:
            image = self.transform(image) if self.transform is not None else _maybe_augment(image)

        text = record.get("text", "")

        result: dict[str, Any] = {
            "sample_id": record.get("sample_id", f"sample_{idx:05d}"),
            "text": text,
            "image_path": record.get("image_path", ""),
            "source": record.get("source", "unknown"),
        }

        if self.processor is not None:
            # Process image to pixel_values
            encoded_img = self.processor(images=image, return_tensors="pt")
            pv = encoded_img.pixel_values
            if pv.ndim == 4 and pv.shape[0] == 1:
                pv = pv.squeeze(0)
            result["pixel_values"] = pv

            # Tokenize text to input_ids
            tok = getattr(self.processor, "tokenizer", self.processor)
            encoded_tok = tok(
                text,
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
            input_ids = encoded_tok.input_ids
            if input_ids.ndim == 2 and input_ids.shape[0] == 1:
                input_ids = input_ids.squeeze(0)
            result["input_ids"] = input_ids

        return result


class ReplayBatchSampler(Sampler[List[int]]):
    """
    Mini-batch sampler combining feedback and golden anchor indices with a fixed ratio.

    Guarantees every mini-batch contains `n_feedback` samples from feedback buffer
    and `n_anchor` samples from anchor pool. If feedback samples are fewer than anchor
    samples (the standard flywheel scenario), feedback samples are drawn round-robin
    with replacement.
    """

    def __init__(
        self,
        feedback_indices: Sequence[int],
        anchor_indices: Sequence[int],
        batch_size: int = 8,
        replay_ratio: float = 0.5,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
        max_batches: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not (0.0 <= replay_ratio <= 1.0):
            raise ValueError(f"replay_ratio must be in [0.0, 1.0], got {replay_ratio}")

        self.feedback_indices = list(feedback_indices)
        self.anchor_indices = list(anchor_indices)
        self.batch_size = batch_size
        self.replay_ratio = replay_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.max_batches = max_batches

        # Determine number of items per batch from each pool
        has_feedback = len(self.feedback_indices) > 0
        has_anchor = len(self.anchor_indices) > 0

        if has_feedback and has_anchor:
            self.n_feedback = int(round(self.batch_size * self.replay_ratio))
            # Ensure at least 1 from each if ratio is strictly between 0 and 1
            if self.n_feedback == 0 and self.replay_ratio > 0.0:
                self.n_feedback = 1
            if self.n_feedback == self.batch_size and self.replay_ratio < 1.0:
                self.n_feedback = self.batch_size - 1
            self.n_anchor = self.batch_size - self.n_feedback
        elif has_feedback:
            self.n_feedback = self.batch_size
            self.n_anchor = 0
        elif has_anchor:
            self.n_feedback = 0
            self.n_anchor = self.batch_size
        else:
            self.n_feedback = 0
            self.n_anchor = 0

    def __len__(self) -> int:
        if not self.feedback_indices and not self.anchor_indices:
            return 0

        # Batches determined by the primary non-empty pool(s)
        batches_candidates: List[int] = []
        if self.n_feedback > 0 and self.feedback_indices:
            q, r = divmod(len(self.feedback_indices), self.n_feedback)
            batches_candidates.append(q if (self.drop_last or r == 0) else q + 1)
        if self.n_anchor > 0 and self.anchor_indices:
            q, r = divmod(len(self.anchor_indices), self.n_anchor)
            batches_candidates.append(q if (self.drop_last or r == 0) else q + 1)

        if not batches_candidates:
            return 0

        # Total batches is the maximum over active pools to allow the larger pool to be covered
        total = max(batches_candidates)
        if self.max_batches is not None:
            total = min(total, self.max_batches)
        return max(1, total) if not self.drop_last else total

    def __iter__(self) -> Iterator[List[int]]:
        total_batches = len(self)
        if total_batches == 0:
            return

        rng = random.Random(self.seed)

        # Helper infinite circular generator
        def _make_cycler(pool: List[int], randomize: bool) -> Iterator[int]:
            items = list(pool)
            while True:
                if randomize:
                    rng.shuffle(items)
                for item in items:
                    yield item

        fb_gen = _make_cycler(self.feedback_indices, self.shuffle) if self.feedback_indices else None
        an_gen = _make_cycler(self.anchor_indices, self.shuffle) if self.anchor_indices else None

        for _ in range(total_batches):
            batch: List[int] = []

            # Sample feedback items
            if self.n_feedback > 0 and fb_gen is not None:
                for _ in range(self.n_feedback):
                    batch.append(next(fb_gen))

            # Sample anchor items
            if self.n_anchor > 0 and an_gen is not None:
                for _ in range(self.n_anchor):
                    batch.append(next(an_gen))

            # If both were empty or couldn't fill batch_size
            if not batch:
                continue

            # Shuffle items within the batch to avoid position bias
            if self.shuffle:
                rng.shuffle(batch)

            yield batch

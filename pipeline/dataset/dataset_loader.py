"""
pipeline/dataset/dataset_loader.py
Dataset loaders, parsers, and PyTorch Dataset/DataLoader adapters for IAM handwriting,
medical prescription manifests, and infinite procedural synthetic streams.
Guarantees writer-independent train/val/test partitioning.
"""

import csv
from dataclasses import dataclass, field
import io
import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, IterableDataset

from pipeline.dataset.synthetic_generator import SyntheticHandwritingGenerator
from pipeline.preprocessing.image_enhancement import normalize_image, to_rgb

logger = logging.getLogger(__name__)

SYNTHETIC_DATASET_SOURCES = frozenset({
    "synthetic_medical_cursive",
    "synthetic_medical",
})


def is_synthetic_training_record(rec: Dict[str, Any]) -> bool:
    """True when a manifest row is locally generated font-rendered handwriting."""
    image_path = str(rec.get("image_path") or rec.get("relative_image_path") or rec.get("image") or "")
    if image_path.endswith("_syn.png"):
        return True
    source = str(rec.get("dataset_source") or "")
    return source in SYNTHETIC_DATASET_SOURCES


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class HandwritingSample:
    """
    Standard representation of a handwritten text sample.
    """
    sample_id: str
    image_path: Optional[str] = None
    image: Optional[np.ndarray] = None  # RGB (H, W, 3) uint8
    text: str = ""
    writer_id: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # [x, y, w, h]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrescriptionItem:
    """
    Individual medication item in a prescription manifest.
    """
    medication: str
    dosage: str
    sig: str
    dispense: str
    refills: str
    bbox: Optional[List[float]] = None


@dataclass
class MedicalPrescriptionSample:
    """
    Structured document-level medical prescription sample.
    """
    sample_id: str
    image_path: Optional[str] = None
    image: Optional[np.ndarray] = None
    clinic_name: str = ""
    doctor_name: str = ""
    patient_name: str = ""
    date: str = ""
    items: List[PrescriptionItem] = field(default_factory=list)
    full_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IAM Handwriting Dataset Parser
# ---------------------------------------------------------------------------

class IAMDatasetParser:
    """
    Parser for IAM Handwriting Database (lines.txt, forms.txt, and image paths).
    Includes automatic writer-independent dataset partitioner.
    """

    def __init__(self, iam_root_dir: Optional[str] = None):
        self.root_dir = Path(iam_root_dir) if iam_root_dir else None

    def parse_lines_txt(
        self,
        lines_txt_source: Union[str, Path, io.StringIO, io.TextIOBase]
    ) -> List[HandwritingSample]:
        """
        Parse IAM ascii/lines.txt into a list of HandwritingSample objects.

        IAM lines.txt format:
        a01-000u-00 ok 154 408 768 27 51 A|MOVE|to|stop|Mr.|Gaitskell|from
        """
        samples: List[HandwritingSample] = []

        if isinstance(lines_txt_source, str) and "\n" in lines_txt_source:
            lines = lines_txt_source.splitlines()
        elif isinstance(lines_txt_source, (str, Path)):
            path = Path(lines_txt_source)
            if not path.exists():
                raise FileNotFoundError(f"IAM lines.txt not found: {path}")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        elif hasattr(lines_txt_source, "readlines"):
            lines = lines_txt_source.readlines()
        elif isinstance(lines_txt_source, (list, tuple)):
            lines = list(lines_txt_source)
        else:
            raise ValueError(f"Unsupported lines_txt_source type: {type(lines_txt_source)}")

        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            parts = line_str.split(" ")
            if len(parts) < 8:
                continue

            line_id = parts[0]  # e.g., a01-000u-00
            seg_status = parts[1]  # 'ok' or 'err'
            threshold = int(parts[2]) if parts[2].lstrip("-").isdigit() else 0
            num_components = int(parts[3]) if parts[3].lstrip("-").isdigit() else 0
            x = int(parts[4]) if parts[4].lstrip("-").isdigit() else 0
            y = int(parts[5]) if parts[5].lstrip("-").isdigit() else 0
            w = int(parts[6]) if parts[6].lstrip("-").isdigit() else 0
            h = int(parts[7]) if parts[7].lstrip("-").isdigit() else 0

            # IAM replaces spaces with '|'
            if len(parts) >= 9:
                transcription = " ".join(parts[8:]).replace("|", " ")
            else:
                transcription = parts[7].replace("|", " ") if not parts[7].lstrip("-").isdigit() else ""

            # Derive writer ID from line_id (e.g. form 'a01-000u' maps to writer '000u')
            id_tokens = line_id.split("-")
            form_id = f"{id_tokens[0]}-{id_tokens[1]}" if len(id_tokens) >= 2 else line_id
            writer_id = id_tokens[1] if len(id_tokens) >= 2 else "unknown"

            img_path = None
            if self.root_dir:
                # IAM standard directory layout: lines/a01/a01-000u/a01-000u-00.png
                cand_path = self.root_dir / "lines" / id_tokens[0] / form_id / f"{line_id}.png"
                if cand_path.exists():
                    img_path = str(cand_path)

            samples.append(
                HandwritingSample(
                    sample_id=line_id,
                    image_path=img_path,
                    text=transcription,
                    writer_id=writer_id,
                    bbox=(x, y, w, h),
                    metadata={
                        "form_id": form_id,
                        "seg_status": seg_status,
                        "threshold": threshold,
                        "num_components": num_components,
                    }
                )
            )

        return samples

    def parse_forms_txt(
        self,
        forms_txt_source: Union[str, Path, io.StringIO, io.TextIOBase]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parse IAM ascii/forms.txt to map form IDs to writer IDs and sentence counts.
        """
        forms_map: Dict[str, Dict[str, Any]] = {}

        if isinstance(forms_txt_source, str) and "\n" in forms_txt_source:
            lines = forms_txt_source.splitlines()
        elif isinstance(forms_txt_source, (str, Path)):
            path = Path(forms_txt_source)
            if not path.exists():
                raise FileNotFoundError(f"IAM forms.txt not found: {path}")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        elif hasattr(forms_txt_source, "readlines"):
            lines = forms_txt_source.readlines()
        elif isinstance(forms_txt_source, (list, tuple)):
            lines = list(forms_txt_source)
        else:
            raise ValueError(f"Unsupported forms_txt_source type: {type(forms_txt_source)}")

        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            parts = line_str.split(" ")
            if len(parts) >= 2:
                form_id = parts[0]
                writer_id = parts[1]
                forms_map[form_id] = {
                    "form_id": form_id,
                    "writer_id": writer_id,
                    "raw_tokens": parts[2:] if len(parts) > 2 else []
                }

        return forms_map

    def create_writer_independent_splits(
        self,
        samples: List[HandwritingSample],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[HandwritingSample], List[HandwritingSample], List[HandwritingSample]]:
        """
        Partition dataset into train, val, and test splits guaranteeing ZERO writer overlap.
        (W_train ∩ W_val = ∅, W_train ∩ W_test = ∅, W_val ∩ W_test = ∅).
        """
        # Group samples by writer ID
        writer_to_samples: Dict[str, List[HandwritingSample]] = {}
        for s in samples:
            w_id = s.writer_id if s.writer_id else f"w_{s.sample_id}"
            writer_to_samples.setdefault(w_id, []).append(s)

        unique_writers = list(writer_to_samples.keys())
        rng = random.Random(seed)
        rng.shuffle(unique_writers)

        num_writers = len(unique_writers)
        n_train = max(1, int(round(num_writers * train_ratio)))
        n_val = max(1, int(round(num_writers * val_ratio))) if num_writers > 2 else 0

        train_writers: Set[str] = set(unique_writers[:n_train])
        val_writers: Set[str] = set(unique_writers[n_train:n_train + n_val])
        test_writers: Set[str] = set(unique_writers[n_train + n_val:])

        # If test set is empty due to small writer count, assign remainder
        if not test_writers and len(unique_writers) > 1:
            if len(train_writers) > 1:
                test_w = train_writers.pop()
                test_writers.add(test_w)

        train_samples: List[HandwritingSample] = []
        val_samples: List[HandwritingSample] = []
        test_samples: List[HandwritingSample] = []

        for w_id, w_samples in writer_to_samples.items():
            if w_id in train_writers:
                train_samples.extend(w_samples)
            elif w_id in val_writers:
                val_samples.extend(w_samples)
            else:
                test_samples.extend(w_samples)

        return train_samples, val_samples, test_samples


# ---------------------------------------------------------------------------
# Medical Prescription Dataset Loader
# ---------------------------------------------------------------------------

class MedicalPrescriptionDatasetLoader:
    """
    Loads and standardizes medical prescription manifests in JSON, JSONL, or CSV formats.
    """

    def __init__(self, data_root_dir: Optional[str] = None):
        self.root_dir = Path(data_root_dir) if data_root_dir else None

    def load_manifest(
        self,
        manifest_path: Union[str, Path]
    ) -> List[MedicalPrescriptionSample]:
        """
        Load a manifest file (.json, .jsonl, or .csv) into a list of MedicalPrescriptionSample objects.
        """
        p = Path(manifest_path)
        if not p.exists():
            raise FileNotFoundError(f"Prescription manifest not found: {p}")

        suffix = p.suffix.lower()
        if suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                raw_records = data
            elif isinstance(data, dict) and "samples" in data:
                raw_records = data["samples"]
            else:
                raw_records = [data]

        elif suffix == ".jsonl":
            raw_records = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        raw_records.append(json.loads(line.strip()))

        elif suffix == ".csv":
            raw_records = []
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_records.append(row)
        else:
            raise ValueError(f"Unsupported manifest file extension: {suffix}")

        samples: List[MedicalPrescriptionSample] = []
        skipped_synthetic = 0
        for idx, rec in enumerate(raw_records):
            if is_synthetic_training_record(rec):
                skipped_synthetic += 1
                continue
            sid = rec.get("sample_id", rec.get("id", f"rx_{idx:05d}"))
            img_p = rec.get("image_path", rec.get("image", None))
            if img_p and self.root_dir and not os.path.isabs(img_p):
                img_p = str(self.root_dir / img_p)

            items: List[PrescriptionItem] = []
            raw_items = rec.get("items", rec.get("medications", []))
            if isinstance(raw_items, list):
                for itm in raw_items:
                    if isinstance(itm, dict):
                        items.append(
                            PrescriptionItem(
                                medication=itm.get("medication", itm.get("name", "")),
                                dosage=itm.get("dosage", itm.get("strength", "")),
                                sig=itm.get("sig", itm.get("instructions", "")),
                                dispense=itm.get("dispense", itm.get("disp", "")),
                                refills=str(itm.get("refills", "0")),
                                bbox=itm.get("bbox", None)
                            )
                        )
                    elif isinstance(itm, (list, tuple)) and len(itm) >= 3:
                        items.append(
                            PrescriptionItem(
                                medication=itm[0],
                                dosage=itm[1],
                                sig=itm[2],
                                dispense=itm[3] if len(itm) > 3 else "#30",
                                refills=str(itm[4]) if len(itm) > 4 else "0"
                            )
                        )

            full_t = rec.get("full_text", rec.get("text", ""))
            if not full_t and items:
                lines = []
                for itm in items:
                    lines.append(f"{itm.medication} {itm.dosage}\nSig: {itm.sig}\nDisp: {itm.dispense} Refills: {itm.refills}")
                full_t = "\n".join(lines)

            # Preserve all top-level manifest attributes in metadata dict (top-level manifest takes precedence)
            meta = dict(rec.get("metadata", {}))
            if "category" in meta and "category" in rec and meta["category"] != rec["category"]:
                meta["template_category"] = meta["category"]
            for k in ("category", "is_lasa", "therapeutic_class", "writer_id", "source", "dataset_source", "split", "text", "transcription"):
                if k in rec:
                    meta[k] = rec[k]

            samples.append(
                MedicalPrescriptionSample(
                    sample_id=sid,
                    image_path=img_p,
                    clinic_name=rec.get("clinic_name", rec.get("clinic", "")),
                    doctor_name=rec.get("doctor_name", rec.get("doctor", "")),
                    patient_name=rec.get("patient_name", rec.get("patient", "")),
                    date=rec.get("date", ""),
                    items=items,
                    full_text=full_t,
                    metadata=meta
                )
            )

        if skipped_synthetic:
            logger.info("Skipped %s synthetic training records from %s", skipped_synthetic, p)
        return samples


# ---------------------------------------------------------------------------
# PyTorch Dataset Adapters
# ---------------------------------------------------------------------------

class HandwritingPyTorchDataset(Dataset):
    """
    Standard PyTorch Map-style Dataset wrapping a list of HandwritingSample objects.
    Loads images on-demand, resizes with aspect-ratio preservation, and normalizes to (3, H, W) tensors.
    """

    def __init__(
        self,
        samples: List[HandwritingSample],
        target_size: Optional[Tuple[int, int]] = (384, 384),
        transform: Optional[Any] = None
    ):
        self.samples = samples
        self.target_size = target_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # 1. Load Image
        if item.image is not None:
            img = to_rgb(item.image)
        elif item.image_path and os.path.exists(item.image_path):
            pil_img = Image.open(item.image_path).convert("RGB")
            img = np.array(pil_img, dtype=np.uint8)
        else:
            # Generate blank white fallback if image missing
            h = self.target_size[0] if self.target_size else 64
            w = self.target_size[1] if self.target_size else 384
            img = np.full((h, w, 3), 255, dtype=np.uint8)

        # 2. Resize & Normalize
        if self.target_size:
            img = normalize_image(img, target_size=self.target_size, keep_aspect_ratio=True)

        if self.transform is not None:
            img = self.transform(img)
            tensor_img = img if isinstance(img, torch.Tensor) else torch.tensor(img)
        else:
            # (H, W, 3) -> (3, H, W) float32 in [0.0, 1.0]
            float_img = img.astype(np.float32) / 255.0
            tensor_img = torch.tensor(np.transpose(float_img, (2, 0, 1)), dtype=torch.float32)

        return {
            "sample_id": item.sample_id,
            "pixel_values": tensor_img,
            "text": item.text,
            "writer_id": item.writer_id or "unknown"
        }


class StreamingSyntheticDataset(IterableDataset):
    """
    Infinite procedural streaming handwriting generator for synthetic pre-training.
    """

    DEFAULT_VOCAB = [
        "The quick brown fox jumps over the lazy dog",
        "Patient exhibits severe tremor in upper extremities",
        "Prescription filled for Amoxicillin 500mg capsules",
        "Take one tablet by mouth twice daily with meals",
        "Refills remaining: zero. Dispense as written.",
        "Clinical diagnosis indicates acute bronchitis",
        "Follow up in two weeks if symptoms persist",
        "Doctor ordered complete metabolic blood panel",
        "Handwritten signature verified by pharmacy staff",
        "Allergies noted: Penicillin and Sulfa drugs",
    ]

    def __init__(
        self,
        generator: Optional[SyntheticHandwritingGenerator] = None,
        vocab: Optional[List[str]] = None,
        target_size: Tuple[int, int] = (64, 384),
        slant_range: Tuple[float, float] = (-15.0, 20.0),
        tremor_range: Tuple[float, float] = (0.5, 2.0)
    ):
        self.generator = generator if generator is not None else SyntheticHandwritingGenerator()
        self.vocab = vocab if vocab is not None else self.DEFAULT_VOCAB
        self.target_size = target_size
        self.slant_range = slant_range
        self.tremor_range = tremor_range

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        idx = 0
        while True:
            text = random.choice(self.vocab)
            slant = random.uniform(*self.slant_range)
            tremor = random.uniform(*self.tremor_range)
            ink = random.choice(["blue", "black", "fountain", "pencil"])

            img, _ = self.generator.render_line(
                text=text,
                font_size=random.randint(28, 36),
                ink_color=ink,
                slant_deg=slant,
                tremor_sigma=tremor,
                canvas_width=self.target_size[1],
                canvas_height=self.target_size[0]
            )

            # Resize & normalize
            norm_img = normalize_image(img, target_size=self.target_size, keep_aspect_ratio=True)
            float_img = norm_img.astype(np.float32) / 255.0
            tensor_img = torch.tensor(np.transpose(float_img, (2, 0, 1)), dtype=torch.float32)

            yield {
                "sample_id": f"syn_{idx:08d}",
                "pixel_values": tensor_img,
                "text": text,
                "writer_id": f"syn_w_{random.randint(0, 50)}"
            }
            idx += 1


def handwriting_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collator function for DataLoader.
    Dynamically pads variable-width/height line tensors to maximum dimensions in batch with white background (1.0).
    """
    if not batch:
        return {}

    sample_ids = [item["sample_id"] for item in batch]
    texts = [item["text"] for item in batch]
    writer_ids = [item.get("writer_id", "unknown") for item in batch]

    tensors = [item["pixel_values"] for item in batch]
    # tensors: shape (3, H, W)
    max_c = max(t.shape[0] for t in tensors)
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)

    batch_size = len(tensors)
    # Fill with 1.0 (white background in normalized float [0, 1])
    padded_tensors = torch.ones((batch_size, max_c, max_h, max_w), dtype=tensors[0].dtype)

    for i, t in enumerate(tensors):
        c, h, w = t.shape
        padded_tensors[i, :c, :h, :w] = t

    return {
        "sample_ids": sample_ids,
        "pixel_values": padded_tensors,
        "texts": texts,
        "writer_ids": writer_ids
    }

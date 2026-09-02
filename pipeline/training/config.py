"""
pipeline/training/config.py
Hyperparameter and environment configuration for Apple Silicon MPS training.
Provides typed validation, hardware device resolution, mixed precision, and unified memory management.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


def get_default_num_workers() -> int:
    """Dynamically compute optimal DataLoader worker count based on CPU hardware topology."""
    return min(8, max(2, (os.cpu_count() or 4) // 2))


@dataclass
class TrainingConfig:
    """
    Comprehensive configuration for TrOCR / Vision-Encoder-Decoder fine-tuning on Apple Silicon MPS.
    """
    # Model & Architecture
    model_name_or_path: str = "microsoft/trocr-small-handwritten"
    processor_name_or_path: Optional[str] = None
    output_dir: str = "checkpoints/trocr-handwritten"
    image_size: Tuple[int, int] = (384, 384)
    max_target_length: int = 128
    attn_implementation: str = "sdpa"

    # Hardware & Device
    device: str = "auto"  # "auto", "mps", "cuda", "cpu"
    mixed_precision: str = "none"  # "none", "fp16", "bf16", "fp32"
    empty_cache_steps: int = 100  # Steps between torch.mps.empty_cache()
    mps_high_watermark_ratio: float = 0.85
    gradient_checkpointing: bool = True  # Enable activation recomputation for ViT + RoBERTa
    freeze_encoder_layers: int = 0  # Number of bottom ViT encoder layers to freeze (Stage 2)

    # Optimization Hyperparameters
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Training Schedule
    num_train_epochs: int = 3
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 4
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant", "cosine_with_restarts"
    min_lr: float = 1e-7

    # DataLoader Settings
    num_workers: int = field(default_factory=get_default_num_workers)
    pin_memory: bool = False  # False on MPS unified memory architecture
    persistent_workers: bool = True
    prefetch_factor: Optional[int] = 4
    prefetch_queue_size: int = 3
    use_async_prefetcher: bool = True
    non_blocking: bool = True
    compile_model: bool = False
    seed: int = 42
    dataloader_drop_last: bool = False

    # Checkpoint & Logging Intervals
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: Optional[int] = 3
    save_best_model: bool = True
    metric_for_best_model: str = "val_cer"  # "val_cer" or "val_loss"
    greater_is_better: bool = False
    num_beams: int = 1
    resume_from_checkpoint: Optional[str] = None
    enable_step_profiling: bool = True

    # Line-level packs only for v3 L1. Word crops (imgur5k, iam_words) stay out of this mix.
    categories: Optional[List[str]] = None
    max_eval_samples: Optional[int] = 256
    generate_on_eval: bool = True
    cer_eval_every_epochs: int = 1

    # Custom / extra attributes
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate hyperparameter invariants."""
        # Handle backward compatibility with model_name keyword if passed via extra_params
        if "model_name" in self.extra_params and self.model_name_or_path == "microsoft/trocr-small-handwritten":
            self.model_name_or_path = str(self.extra_params.pop("model_name"))
        if "epochs" in self.extra_params:
            self.num_train_epochs = int(self.extra_params.pop("epochs"))
        if "fp16" in self.extra_params:
            fp16_val = self.extra_params.pop("fp16")
            if isinstance(fp16_val, bool) and fp16_val:
                self.mixed_precision = "fp16"

        self.validate()

    def validate(self) -> None:
        """Enforce valid ranges and recognized option values."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.eval_batch_size < 1:
            raise ValueError(f"eval_batch_size must be >= 1, got {self.eval_batch_size}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0.0, got {self.learning_rate}")
        if self.num_train_epochs < 1 and (self.max_steps is None or self.max_steps <= 0):
            raise ValueError(f"num_train_epochs must be >= 1 or max_steps > 0, got epochs={self.num_train_epochs}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}")
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")

        valid_devices = {"auto", "mps", "cuda", "cpu"}
        if self.device.lower() not in valid_devices:
            raise ValueError(f"Invalid device '{self.device}'. Expected one of {valid_devices}")

        valid_precision = {"none", "fp16", "bf16", "fp32"}
        if self.mixed_precision.lower() not in valid_precision:
            raise ValueError(f"Invalid mixed_precision '{self.mixed_precision}'. Expected one of {valid_precision}")

        valid_schedulers = {"cosine", "linear", "constant", "cosine_with_restarts"}
        if self.lr_scheduler_type.lower() not in valid_schedulers:
            raise ValueError(f"Invalid lr_scheduler_type '{self.lr_scheduler_type}'. Expected one of {valid_schedulers}")

    @property
    def model_name(self) -> str:
        """Alias for model_name_or_path."""
        return self.model_name_or_path

    @model_name.setter
    def model_name(self, val: str) -> None:
        self.model_name_or_path = val

    @property
    def epochs(self) -> int:
        """Alias for num_train_epochs."""
        return self.num_train_epochs

    @epochs.setter
    def epochs(self, val: int) -> None:
        self.num_train_epochs = val

    @property
    def fp16(self) -> bool:
        """Boolean indicator if mixed_precision is fp16."""
        return self.mixed_precision.lower() == "fp16"

    @fp16.setter
    def fp16(self, val: bool) -> None:
        if val:
            self.mixed_precision = "fp16"
        elif self.mixed_precision == "fp16":
            self.mixed_precision = "none"

    @property
    def resolved_device(self) -> torch.device:
        """Resolve string target to verified PyTorch compute device."""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if self.mps_high_watermark_ratio > 0.0:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(self.mps_high_watermark_ratio)
            os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(round(self.mps_high_watermark_ratio * 0.8, 2))
        else:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"

        req = self.device.lower()
        if req == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
            return torch.device("cpu")
        elif req == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
            logger.warning("MPS device requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        elif req == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA device requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cpu")

    @property
    def is_mps(self) -> bool:
        return self.resolved_device.type == "mps"

    @property
    def is_cuda(self) -> bool:
        return self.resolved_device.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        return self.resolved_device.type == "cpu"

    @property
    def autocast_dtype(self) -> Optional[torch.dtype]:
        """Resolve mixed precision torch dtype for autocast context."""
        mp = self.mixed_precision.lower()
        if mp == "fp16":
            return torch.float16
        elif mp == "bf16":
            return torch.bfloat16
        return None

    def manage_memory(self, step: int) -> None:
        """Periodically empty MPS/CUDA cache to manage unified memory footprint."""
        if self.empty_cache_steps <= 0 or step % self.empty_cache_steps != 0:
            return

        if self.is_mps and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif self.is_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def stage1_preset(
        cls,
        output_dir: str = "checkpoints/stage1_general_adaptation",
        model_name_or_path: str = "microsoft/trocr-large-handwritten",
        **kwargs: Any,
    ) -> TrainingConfig:
        """Stage 1: General cursive adaptation preset."""
        return cls(
            model_name_or_path=model_name_or_path,
            output_dir=output_dir,
            learning_rate=5e-5,
            min_lr=1e-6,
            warmup_ratio=0.05,
            num_train_epochs=5,
            batch_size=4,
            gradient_accumulation_steps=8,
            mixed_precision="fp16",
            gradient_checkpointing=True,
            freeze_encoder_layers=0,
            **kwargs,
        )

    @classmethod
    def stage2_preset(
        cls,
        stage1_checkpoint: str = "checkpoints/stage1_general_adaptation/best_model.pt",
        output_dir: str = "checkpoints/stage2_doctor_specialization",
        **kwargs: Any,
    ) -> TrainingConfig:
        """Stage 2: Doctor and clinical specialization preset."""
        return cls(
            model_name_or_path=stage1_checkpoint,
            output_dir=output_dir,
            learning_rate=1.5e-5,
            min_lr=5e-7,
            warmup_ratio=0.03,
            num_train_epochs=5,
            batch_size=4,
            gradient_accumulation_steps=8,
            mixed_precision="fp16",
            gradient_checkpointing=True,
            freeze_encoder_layers=12,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary with primitive types."""
        d = asdict(self)
        d.pop("extra_params", None)
        # Ensure tuples are converted to lists for clean YAML/JSON serialization
        if isinstance(d.get("image_size"), tuple):
            d["image_size"] = list(d["image_size"])
        return d

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """Serialize configuration to JSON string or file."""
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s, encoding="utf-8")
        return s

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """Serialize configuration to YAML string or file (if pyyaml available, else JSON)."""
        try:
            import yaml
            s = yaml.safe_dump(self.to_dict(), default_flow_style=False)
        except ImportError:
            s = self.to_json()

        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s, encoding="utf-8")
        return s

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrainingConfig:
        """Create TrainingConfig from dictionary with field filtering and alias handling."""
        d_copy = dict(d)
        known_fields = set(cls.__dataclass_fields__.keys())

        # Translate aliases (aliases take precedence when explicitly supplied)
        if "model_name" in d_copy:
            d_copy["model_name_or_path"] = d_copy.pop("model_name")
        if "epochs" in d_copy:
            d_copy["num_train_epochs"] = d_copy.pop("epochs")
        if "fp16" in d_copy:
            fp16_val = d_copy.pop("fp16")
            if isinstance(fp16_val, bool) and fp16_val:
                d_copy["mixed_precision"] = "fp16"

        if "image_size" in d_copy and isinstance(d_copy["image_size"], (list, tuple)):
            d_copy["image_size"] = tuple(d_copy["image_size"])

        filtered = {k: v for k, v in d_copy.items() if k in known_fields}
        extra = {k: v for k, v in d_copy.items() if k not in known_fields}
        return cls(**filtered, extra_params=extra)

    @classmethod
    def from_json(cls, path_or_str: Union[str, Path]) -> TrainingConfig:
        """Load TrainingConfig from JSON string or file path safely."""
        data = None
        if isinstance(path_or_str, Path):
            try:
                if path_or_str.is_file():
                    with open(path_or_str, "r", encoding="utf-8") as f:
                        data = json.load(f)
            except (OSError, ValueError):
                pass
            if data is None:
                data = json.loads(str(path_or_str))
        elif isinstance(path_or_str, str):
            is_file = False
            stripped = path_or_str.strip()
            # Fast heuristic: file paths do not contain newlines, do not start with '{', and are within PATH_MAX
            if len(path_or_str) < 4096 and "\n" not in path_or_str and not stripped.startswith("{"):
                try:
                    p = Path(path_or_str)
                    if p.is_file():
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        is_file = True
                except (OSError, ValueError):
                    is_file = False
            if not is_file:
                data = json.loads(path_or_str)
        else:
            raise TypeError(f"Expected str or Path, got {type(path_or_str).__name__}")

        if not isinstance(data, dict):
            raise ValueError(f"Invalid configuration format: expected dictionary, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path_or_str: Union[str, Path]) -> TrainingConfig:
        """Load TrainingConfig from YAML string or file path safely."""
        content = None
        if isinstance(path_or_str, Path):
            try:
                if path_or_str.is_file():
                    content = path_or_str.read_text(encoding="utf-8")
            except (OSError, ValueError):
                pass
            if content is None:
                content = str(path_or_str)
        elif isinstance(path_or_str, str):
            is_file = False
            stripped = path_or_str.strip()
            # Fast heuristic: file paths do not contain newlines, do not start with '{', and are within PATH_MAX
            if len(path_or_str) < 4096 and "\n" not in path_or_str and not stripped.startswith("{"):
                try:
                    p = Path(path_or_str)
                    if p.is_file():
                        content = p.read_text(encoding="utf-8")
                        is_file = True
                except (OSError, ValueError):
                    is_file = False
            if not is_file:
                content = path_or_str
        else:
            raise TypeError(f"Expected str or Path, got {type(path_or_str).__name__}")

        try:
            import yaml
            data = yaml.safe_load(content)
        except ImportError:
            data = json.loads(content)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid configuration format: expected dictionary, got {type(data).__name__}")
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Curriculum Stage & Multi-Stage Configuration Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStageConfig:
    """Configuration for an individual curriculum training stage."""
    stage_name: str
    dataset_manifest: str
    val_manifest: Optional[str] = None
    num_epochs: int = 5
    learning_rate: float = 5e-5
    min_lr: float = 1e-6
    warmup_ratio: float = 0.05
    freeze_encoder_layers: int = 0
    gradient_accumulation_steps: int = 4
    micro_batch_size: int = 8
    eval_batch_size: int = 8
    attn_implementation: str = "sdpa"
    category_filter: Optional[List[str]] = None
    output_dir: str = ""
    save_total_limit: int = 3
    metric_for_best_model: str = "val_cer"
    num_workers: Optional[int] = None
    persistent_workers: bool = True
    prefetch_factor: Optional[int] = 4
    pin_memory: Optional[bool] = None
    use_async_prefetcher: bool = True
    prefetch_queue_size: int = 3
    enable_step_profiling: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CurriculumStageConfig:
        known = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class CurriculumConfig:
    """Global configuration coordinating multi-stage curriculum training."""
    model_name_or_path: str = "microsoft/trocr-large-handwritten"
    root_output_dir: str = "checkpoints"
    device: str = "auto"
    mixed_precision: str = "fp16"
    attn_implementation: str = "sdpa"
    seed: int = 42
    empty_cache_steps: int = 100
    gradient_checkpointing: bool = True
    num_workers: Optional[int] = None
    persistent_workers: bool = True
    prefetch_factor: Optional[int] = 4
    use_async_prefetcher: bool = True
    prefetch_queue_size: int = 3
    enable_step_profiling: bool = True
    stages: List[CurriculumStageConfig] = field(default_factory=list)

    @classmethod
    def default_2stage_trocr_large(
        cls,
        root_dir: str = "checkpoints",
        stage1_manifest: str = "data/reference_handwriting/train_manifest.jsonl",
        val_manifest: Optional[str] = "data/reference_handwriting/val_manifest.jsonl",
        stage1_epochs: int = 5,
        stage2_epochs: int = 5,
        device: str = "auto",
        mixed_precision: str = "fp16",
        attn_implementation: str = "sdpa",
    ) -> CurriculumConfig:
        """Standard 2-stage curriculum fine-tuning preset for TrOCR-Large 558M."""
        stage1 = CurriculumStageConfig(
            stage_name="stage1_general_adaptation",
            dataset_manifest=stage1_manifest,
            val_manifest=val_manifest,
            num_epochs=stage1_epochs,
            learning_rate=5e-5,
            min_lr=1e-6,
            warmup_ratio=0.05,
            freeze_encoder_layers=0,
            gradient_accumulation_steps=8,
            micro_batch_size=4,
            attn_implementation=attn_implementation,
            output_dir=os.path.join(root_dir, "stage1_general_adaptation"),
        )
        stage2 = CurriculumStageConfig(
            stage_name="stage2_doctor_specialization",
            dataset_manifest=stage1_manifest,
            val_manifest=val_manifest,
            num_epochs=stage2_epochs,
            learning_rate=1.5e-5,
            min_lr=5e-7,
            warmup_ratio=0.03,
            freeze_encoder_layers=12,
            gradient_accumulation_steps=8,
            micro_batch_size=4,
            attn_implementation=attn_implementation,
            category_filter=["prescription_item", "clinical_note", "doctor_signature"],
            output_dir=os.path.join(root_dir, "stage2_doctor_specialization"),
        )
        return cls(
            model_name_or_path="microsoft/trocr-large-handwritten",
            root_output_dir=root_dir,
            device=device,
            mixed_precision=mixed_precision,
            attn_implementation=attn_implementation,
            stages=[stage1, stage2],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "root_output_dir": self.root_output_dir,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "attn_implementation": self.attn_implementation,
            "seed": self.seed,
            "empty_cache_steps": self.empty_cache_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "use_async_prefetcher": self.use_async_prefetcher,
            "prefetch_queue_size": self.prefetch_queue_size,
            "enable_step_profiling": self.enable_step_profiling,
            "stages": [s.to_dict() for s in self.stages],
        }

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s, encoding="utf-8")
        return s

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CurriculumConfig:
        stages_data = d.get("stages", [])
        stages = [CurriculumStageConfig.from_dict(s) if isinstance(s, dict) else s for s in stages_data]
        return cls(
            model_name_or_path=d.get("model_name_or_path", "microsoft/trocr-large-handwritten"),
            root_output_dir=d.get("root_output_dir", "checkpoints"),
            device=d.get("device", "auto"),
            mixed_precision=d.get("mixed_precision", "fp16"),
            attn_implementation=d.get("attn_implementation", "sdpa"),
            seed=d.get("seed", 42),
            empty_cache_steps=d.get("empty_cache_steps", 100),
            gradient_checkpointing=d.get("gradient_checkpointing", True),
            num_workers=d.get("num_workers", None),
            persistent_workers=d.get("persistent_workers", True),
            prefetch_factor=d.get("prefetch_factor", 4),
            use_async_prefetcher=d.get("use_async_prefetcher", True),
            prefetch_queue_size=d.get("prefetch_queue_size", 3),
            enable_step_profiling=d.get("enable_step_profiling", True),
            stages=stages,
        )

    @classmethod
    def from_json(cls, path_or_str: Union[str, Path]) -> CurriculumConfig:
        if isinstance(path_or_str, Path) or (isinstance(path_or_str, str) and os.path.exists(path_or_str)):
            data = json.loads(Path(path_or_str).read_text(encoding="utf-8"))
        else:
            data = json.loads(str(path_or_str))
        return cls.from_dict(data)


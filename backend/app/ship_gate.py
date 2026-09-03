"""Refuse Azure/backend checkpoints that failed the Teklia-pack accuracy bar or violated clinical safety."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

# Re-export pipeline flywheel safety gate components for backend access
from pipeline.training.ship_gate import (
    LasaAuditResult,
    audit_lasa_safety,
    audit_lasa_safety_hardened,
    check_cer_regression,
    decide_ship,
    load_lasa_catalog,
)

FORBIDDEN_SUBSTRINGS = (
    "runs/base_iam_v1",
    "trocr-base-stage1",
)


def assert_shippable_checkpoint(model_id: str) -> str:
    """Raise if this id is the failed stage1 scientific run (52.5% Teklia CER)."""
    normalized = str(model_id).replace("\\", "/")
    for needle in FORBIDDEN_SUBSTRINGS:
        if needle in normalized:
            raise ValueError(
                f"refusing to load {model_id}: stage1 Teklia test CER was 0.525; "
                "use microsoft/trocr-large-handwritten or a Teklia-proven checkpoint"
            )
    path = Path(model_id)
    if path.exists() and "base_iam_v1" in str(path.resolve()):
        raise ValueError(f"refusing to load {model_id}: base_iam_v1 is not a ship candidate")
    return model_id


__all__ = [
    "FORBIDDEN_SUBSTRINGS",
    "assert_shippable_checkpoint",
    "load_lasa_catalog",
    "audit_lasa_safety",
    "audit_lasa_safety_hardened",
    "check_cer_regression",
    "decide_ship",
    "LasaAuditResult",
]

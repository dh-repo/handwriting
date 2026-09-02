"""Refuse checkpoints that failed the Teklia-pack accuracy bar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Teklia-measured line-HTR floor. Serve and Azure env pin this checkpoint at beams=4.
PROVEN_SHIP_MODEL = "microsoft/trocr-large-handwritten"
PROVEN_TEKLIA_CER_BEAMS1 = 0.048051586195916096
PROVEN_TEKLIA_CER_BEAMS4 = 0.03460573976037894
PROVEN_BASE_BEAMS4 = 0.04519364725550293
BASELINE_BEAMS4_REPORT = Path("runs/baseline_handwritten_teklia_test_beams4/test_report.json")

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
                f"refusing to ship {model_id}: stage1 Teklia test CER was 0.525, "
                f"worse than {PROVEN_SHIP_MODEL} at {PROVEN_TEKLIA_CER_BEAMS1:.4f}"
            )
    path = Path(model_id)
    if path.exists() and "base_iam_v1" in str(path.resolve()):
        raise ValueError(f"refusing to ship {model_id}: base_iam_v1 is not a ship candidate")
    return model_id


def decide_ship(
    report: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    baseline_cer: float | None = None,
    baseline_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Promote only if Teklia CER is strictly below a beams-matched baseline."""
    checkpoint = assert_shippable_checkpoint(str(report.get("checkpoint", "")))
    cer = float(report["cer"])
    beams = int(report.get("num_beams", 1))
    beams4_path = Path(baseline_report_path) if baseline_report_path is not None else BASELINE_BEAMS4_REPORT
    if baseline_cer is None and beams >= 4 and not beams4_path.is_file():
        decision = {
            "promote": False,
            "reason": "missing Teklia beams=4 handwritten-Base baseline",
            "candidate_cer": cer,
            "candidate_beams": beams,
            "checkpoint": checkpoint,
        }
    else:
        if baseline_cer is not None:
            floor = float(baseline_cer)
        elif beams >= 4 and beams4_path.is_file():
            floor = float(json.loads(beams4_path.read_text(encoding="utf-8"))["cer"])
        elif beams >= 4:
            floor = PROVEN_TEKLIA_CER_BEAMS4
        else:
            floor = PROVEN_TEKLIA_CER_BEAMS1
        promote = cer < floor
        decision = {
            "promote": promote,
            "reason": "strictly below baseline" if promote else "not below baseline",
            "candidate_cer": cer,
            "candidate_beams": beams,
            "baseline_cer": floor,
            "checkpoint": checkpoint,
        }
    if output_dir is not None:
        path = Path(output_dir) / "ship_decision.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
        decision["path"] = str(path)
    return decision

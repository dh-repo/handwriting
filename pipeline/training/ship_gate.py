"""Refuse checkpoints that failed the Teklia-pack accuracy bar or violated clinical safety."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, List, Optional, Sequence, Tuple

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


@dataclass
class LasaAuditResult:
    """Clinical LASA safety audit outcome."""

    total_evaluated: int
    violations: List[dict[str, Any]]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_lasa_catalog(
    vocab_dir: str | Path = "data/reference_handwriting/vocabularies",
) -> List[Tuple[str, str]]:
    """
    Load the 20 bidirectional LASA (Look-Alike / Sound-Alike) drug pairs from rxnorm_medications.json.

    Returns a list of unique canonical (drug_a, drug_b) pairs sorted alphabetically.
    """
    vocab_path = Path(vocab_dir) / "rxnorm_medications.json"
    if not vocab_path.exists():
        repo_root = Path(__file__).resolve().parents[2]
        alt = repo_root / vocab_dir / "rxnorm_medications.json"
        if alt.exists():
            vocab_path = alt

    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing LASA catalog at {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_pairs: set[Tuple[str, str]] = set()
    for med in data.get("medications", []):
        if med.get("is_lasa"):
            name = med["generic_name"].strip()
            for cp in med.get("confusion_pairs", []):
                target = cp["target_drug"].strip()
                pair = tuple(sorted([name, target]))
                unique_pairs.add(pair)

    return sorted(list(unique_pairs))


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Dynamic programming Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        c1 = s1[i - 1]
        for j in range(1, n + 1):
            c2 = s2[j - 1]
            if c1 == c2:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def audit_lasa_safety(
    references: Sequence[str],
    hypotheses: Sequence[str],
    lasa_pairs: Optional[List[Tuple[str, str]]] = None,
    *,
    near_miss_distance: int = 0,
    detect_multi_drug: bool = False,
) -> LasaAuditResult:
    """
    Scan model hypotheses against references for dangerous LASA medication substitutions.

    Zero tolerance: passed=True ONLY if len(violations) == 0.

    Parameters:
    - references: Ground truth prescription lines.
    - hypotheses: Model transcribed lines.
    - lasa_pairs: Optional explicit catalog of confusable pairs. Defaults to canonical 20 pairs.
    - near_miss_distance: Maximum Levenshtein edit distance to flag near-miss misspellings
      of the confusable counterpart (default 0 = exact word boundary match).
    - detect_multi_drug: When True, checks multi-drug context notes where both confusable
      drugs appear in reference for count imbalances / substitutions.
    """
    if lasa_pairs is None:
        lasa_pairs = load_lasa_catalog()

    compiled_pairs = []
    for drug_a, drug_b in lasa_pairs:
        pat_a = re.compile(r"\b" + re.escape(drug_a) + r"\b", re.IGNORECASE)
        pat_b = re.compile(r"\b" + re.escape(drug_b) + r"\b", re.IGNORECASE)
        compiled_pairs.append((drug_a, drug_b, pat_a, pat_b))

    violations: List[dict[str, Any]] = []
    total = len(references)

    for idx, (ref, hyp) in enumerate(zip(references, hypotheses)):
        ref_str = str(ref)
        hyp_str = str(hyp)
        hyp_tokens = re.findall(r"[A-Za-z0-9]+", hyp_str) if near_miss_distance > 0 else []

        for drug_a, drug_b, pat_a, pat_b in compiled_pairs:
            has_a_ref = bool(pat_a.search(ref_str))
            has_b_ref = bool(pat_b.search(ref_str))
            has_a_hyp = bool(pat_a.search(hyp_str))
            has_b_hyp = bool(pat_b.search(hyp_str))

            # Drug A prescribed, but hypothesis predicts confusable counterpart Drug B
            if has_a_ref and not has_b_ref and has_b_hyp:
                violations.append(
                    {
                        "index": idx,
                        "reference": ref_str,
                        "hypothesis": hyp_str,
                        "prescribed_drug": drug_a,
                        "confused_drug": drug_b,
                    }
                )

            # Drug B prescribed, but hypothesis predicts confusable counterpart Drug A
            elif has_b_ref and not has_a_ref and has_a_hyp:
                violations.append(
                    {
                        "index": idx,
                        "reference": ref_str,
                        "hypothesis": hyp_str,
                        "prescribed_drug": drug_b,
                        "confused_drug": drug_a,
                    }
                )

            # Multi-drug context: both drugs mentioned in reference, but one was replaced
            elif detect_multi_drug and has_a_ref and has_b_ref:
                count_a_ref = len(pat_a.findall(ref_str))
                count_b_ref = len(pat_b.findall(ref_str))
                count_a_hyp = len(pat_a.findall(hyp_str))
                count_b_hyp = len(pat_b.findall(hyp_str))

                if count_a_hyp < count_a_ref and count_b_hyp > count_b_ref:
                    violations.append(
                        {
                            "index": idx,
                            "reference": ref_str,
                            "hypothesis": hyp_str,
                            "prescribed_drug": drug_a,
                            "confused_drug": drug_b,
                            "context": "multi_drug_substitution",
                        }
                    )
                elif count_b_hyp < count_b_ref and count_a_hyp > count_a_ref:
                    violations.append(
                        {
                            "index": idx,
                            "reference": ref_str,
                            "hypothesis": hyp_str,
                            "prescribed_drug": drug_b,
                            "confused_drug": drug_a,
                            "context": "multi_drug_substitution",
                        }
                    )

            # Near-miss check: catch slight misspellings of the confusable partner
            if near_miss_distance > 0:
                if has_a_ref and not has_b_ref and not has_b_hyp:
                    for tok in hyp_tokens:
                        tok_lower = tok.lower()
                        d_b = _levenshtein_distance(tok_lower, drug_b.lower())
                        d_a = _levenshtein_distance(tok_lower, drug_a.lower())
                        if 1 <= d_b <= near_miss_distance and d_b < d_a:
                            violations.append(
                                {
                                    "index": idx,
                                    "reference": ref_str,
                                    "hypothesis": hyp_str,
                                    "prescribed_drug": drug_a,
                                    "confused_drug": drug_b,
                                    "near_miss_token": tok,
                                }
                            )
                            break
                elif has_b_ref and not has_a_ref and not has_a_hyp:
                    for tok in hyp_tokens:
                        tok_lower = tok.lower()
                        d_a = _levenshtein_distance(tok_lower, drug_a.lower())
                        d_b = _levenshtein_distance(tok_lower, drug_b.lower())
                        if 1 <= d_a <= near_miss_distance and d_a < d_b:
                            violations.append(
                                {
                                    "index": idx,
                                    "reference": ref_str,
                                    "hypothesis": hyp_str,
                                    "prescribed_drug": drug_b,
                                    "confused_drug": drug_a,
                                    "near_miss_token": tok,
                                }
                            )
                            break

    passed = len(violations) == 0
    return LasaAuditResult(total_evaluated=total, violations=violations, passed=passed)


def audit_lasa_safety_hardened(
    references: Sequence[str],
    hypotheses: Sequence[str],
    lasa_pairs: Optional[List[Tuple[str, str]]] = None,
    *,
    near_miss_distance: int = 2,
    detect_multi_drug: bool = True,
) -> LasaAuditResult:
    """
    Hardened clinical LASA safety gate audit enabling both near-miss confusable
    detection (Levenshtein distance <= 2) and multi-drug prescription context substitution detection.
    """
    return audit_lasa_safety(
        references=references,
        hypotheses=hypotheses,
        lasa_pairs=lasa_pairs,
        near_miss_distance=near_miss_distance,
        detect_multi_drug=detect_multi_drug,
    )


def check_cer_regression(
    candidate_cer: float,
    baseline_cer: float,
    max_cer_regression: float = 0.05,
) -> bool:
    """
    Verify candidate CER does not exceed baseline CER by more than max_cer_regression tolerance.

    candidate_cer <= baseline_cer * (1.0 + max_cer_regression)
    """
    threshold = float(baseline_cer) * (1.0 + float(max_cer_regression)) + 1e-9
    return float(candidate_cer) <= threshold


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
    max_cer_regression: float | None = None,
    lasa_audit: LasaAuditResult | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Promote candidate only if:
    1. Checkpoint is shippable (assert_shippable_checkpoint).
    2. Candidate CER satisfies baseline threshold (or regression tolerance).
    3. Clinical LASA safety audit passes with zero dangerous drug substitutions.
    """
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
        if output_dir is not None:
            path = Path(output_dir) / "ship_decision.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
            decision["path"] = str(path)
        return decision

    if baseline_cer is not None:
        floor = float(baseline_cer)
    elif beams >= 4 and beams4_path.is_file():
        floor = float(json.loads(beams4_path.read_text(encoding="utf-8"))["cer"])
    elif beams >= 4:
        floor = PROVEN_TEKLIA_CER_BEAMS4
    else:
        floor = PROVEN_TEKLIA_CER_BEAMS1

    # Check CER criteria
    if max_cer_regression is not None:
        cer_passed = check_cer_regression(cer, floor, max_cer_regression=max_cer_regression)
    else:
        cer_passed = cer < floor

    # Check LASA criteria
    lasa_passed = True
    violations: List[dict[str, Any]] = []
    if lasa_audit is not None:
        if isinstance(lasa_audit, LasaAuditResult):
            lasa_passed = bool(lasa_audit.passed)
            violations = list(lasa_audit.violations)
        elif isinstance(lasa_audit, dict):
            lasa_passed = bool(lasa_audit.get("passed", len(lasa_audit.get("violations", [])) == 0))
            violations = list(lasa_audit.get("violations", []))

    promote = cer_passed and lasa_passed

    if not cer_passed:
        reason = (
            f"CER {cer:.4f} regressed beyond baseline threshold {floor:.4f}"
            if max_cer_regression is not None
            else "not below baseline"
        )
    elif not lasa_passed:
        reason = f"Clinical LASA safety gate failed: {len(violations)} dangerous drug substitution(s) detected"
    else:
        reason = "strictly below baseline" if (max_cer_regression is None and lasa_audit is None) else "passed CER and LASA safety audit"

    decision = {
        "promote": promote,
        "reason": reason,
        "candidate_cer": cer,
        "candidate_beams": beams,
        "baseline_cer": floor,
        "cer_passed": cer_passed,
        "lasa_passed": lasa_passed,
        "lasa_violations": violations,
        "checkpoint": checkpoint,
    }

    if output_dir is not None:
        path = Path(output_dir) / "ship_decision.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
        decision["path"] = str(path)

    return decision

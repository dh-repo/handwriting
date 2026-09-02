"""Cased CER / WER / exact-match for line-level HTR. Headline CER is never lowercased."""

from __future__ import annotations

from typing import Iterable, Sequence


def _as_list(values: Sequence[str] | Iterable[str]) -> list[str]:
    return [str(v) for v in values]


def cased_cer(references: Sequence[str], predictions: Sequence[str]) -> float:
    refs, hyps = _as_list(references), _as_list(predictions)
    try:
        import jiwer
        return float(jiwer.cer(refs, hyps))
    except Exception:
        return _fallback_rate(refs, hyps, unit="char")


def uncased_cer(references: Sequence[str], predictions: Sequence[str]) -> float:
    refs = [r.lower() for r in _as_list(references)]
    hyps = [p.lower() for p in _as_list(predictions)]
    return cased_cer(refs, hyps)


def cased_wer(references: Sequence[str], predictions: Sequence[str]) -> float:
    refs, hyps = _as_list(references), _as_list(predictions)
    try:
        import jiwer
        return float(jiwer.wer(refs, hyps))
    except Exception:
        return _fallback_rate(refs, hyps, unit="word")


def exact_match(references: Sequence[str], predictions: Sequence[str]) -> float:
    refs, hyps = _as_list(references), _as_list(predictions)
    if not refs:
        return 0.0
    hits = sum(1 for r, p in zip(refs, hyps) if r == p)
    return float(hits / len(refs))


def htr_metric_bundle(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    return {
        "cer": cased_cer(references, predictions),
        "wer": cased_wer(references, predictions),
        "exact_match": exact_match(references, predictions),
        "uncased_cer": uncased_cer(references, predictions),
    }


def _fallback_rate(refs: list[str], hyps: list[str], unit: str) -> float:
    scores: list[float] = []
    for r, p in zip(refs, hyps):
        a = list(r) if unit == "char" else r.split()
        b = list(p) if unit == "char" else p.split()
        if not a:
            scores.append(0.0 if not b else 1.0)
            continue
        scores.append(abs(len(a) - len(b)) / max(1, len(a)))
    return float(sum(scores) / max(1, len(scores)))

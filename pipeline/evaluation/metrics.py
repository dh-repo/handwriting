"""
pipeline/evaluation/metrics.py
Standard evaluation metrics for handwriting recognition:
Character Error Rate (CER), Word Error Rate (WER), text normalizers,
granular edit distance breakdowns (substitutions/deletions/insertions/hits),
and throughput / latency profiling percentiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import re
import string
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import unicodedata

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

@dataclass
class NormalizationConfig:
    """Configuration for OCR transcription text normalization."""
    lowercase: bool = True
    strip_punctuation: bool = False
    normalize_whitespace: bool = True
    remove_accents: bool = False
    strip_surrounding_whitespace: bool = True
    custom_replacements: Optional[List[Tuple[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_text(text: str, config: Optional[NormalizationConfig] = None) -> str:
    """
    Apply configurable text normalizations for consistent CER/WER computation.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    if config is None:
        config = NormalizationConfig()

    s = text

    # 1. Lowercase (initial lowercasing ensures patterns match regardless of input case when lowercase is enabled)
    if config.lowercase:
        s = s.lower()

    # 2. Custom Regex Replacements
    if config.custom_replacements:
        for pattern, repl in config.custom_replacements:
            flags = re.IGNORECASE if (config.lowercase and isinstance(pattern, str)) else 0
            s = re.sub(pattern, repl, s, flags=flags)
        if config.lowercase:
            s = s.lower()

    # 3. Remove Accents / Diacritics
    if config.remove_accents:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")

    # 4. Strip Punctuation
    if config.strip_punctuation:
        # Retain whitespace, remove punctuation
        trans = str.maketrans("", "", string.punctuation)
        s = s.translate(trans)

    # 5. Normalize Whitespace (tabs, newlines, multiple spaces -> single space)
    if config.normalize_whitespace:
        s = re.sub(r"\s+", " ", s)

    # 6. Strip Surrounding Whitespace
    if config.strip_surrounding_whitespace:
        s = s.strip()

    return s


# ---------------------------------------------------------------------------
# Error Breakdowns & Metric Results
# ---------------------------------------------------------------------------

@dataclass
class ErrorBreakdown:
    """Granular edit operation counts for Character or Word level errors."""
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    total_reference_units: int = 0
    error_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricResult:
    """Comprehensive evaluation metrics summary."""
    cer: float = 0.0
    wer: float = 0.0
    normalized_cer: float = 0.0
    normalized_wer: float = 0.0
    char_breakdown: ErrorBreakdown = field(default_factory=ErrorBreakdown)
    word_breakdown: ErrorBreakdown = field(default_factory=ErrorBreakdown)
    sample_count: int = 0
    normalization_applied: NormalizationConfig = field(default_factory=NormalizationConfig)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Edit Distance / Alignment Helpers
# ---------------------------------------------------------------------------

def _normalize_input_sequences(
    reference: Union[str, Sequence[str]],
    hypothesis: Union[str, Sequence[str]],
) -> Tuple[List[str], List[str]]:
    """
    Convert string or sequence inputs to paired lists of strings,
    validating that sequence lengths match and raising ValueError on mismatch.
    """
    if isinstance(reference, str) and isinstance(hypothesis, str):
        return [reference], [hypothesis]

    if isinstance(reference, str):
        refs = [reference]
    elif isinstance(reference, (list, tuple)):
        refs = list(reference)
    elif hasattr(reference, "__iter__") and not isinstance(reference, (bytes, bytearray)):
        refs = [str(r) for r in reference]
    else:
        refs = [str(reference)]

    if isinstance(hypothesis, str):
        hyps = [hypothesis]
    elif isinstance(hypothesis, (list, tuple)):
        hyps = list(hypothesis)
    elif hasattr(hypothesis, "__iter__") and not isinstance(hypothesis, (bytes, bytearray)):
        hyps = [str(h) for h in hypothesis]
    else:
        hyps = [str(hypothesis)]

    if len(refs) != len(hyps):
        raise ValueError(
            f"Mismatched number of references ({len(refs)}) and hypotheses ({len(hyps)}). "
            f"Both collections must have identical length."
        )

    return refs, hyps


def _levenshtein_ops(ref: Sequence[Any], hyp: Sequence[Any]) -> Tuple[int, int, int, int]:
    """
    Compute substitutions (S), deletions (D), insertions (I), and hits (H)
    between reference and hypothesis sequences using dynamic programming.
    """
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return 0, 0, 0, 0
    if n == 0:
        return 0, 0, m, 0
    if m == 0:
        return 0, n, 0, 0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrace to count exact operation types
    i, j = n, m
    s, d, ins, h = 0, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            h += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            if i > 0 and j > 0:
                s += 1
                i -= 1
                j -= 1
            elif i > 0:
                d += 1
                i -= 1
            else:
                ins += 1
                j -= 1

    return s, d, ins, h


# ---------------------------------------------------------------------------
# Core Metric Computation Functions
# ---------------------------------------------------------------------------

def compute_cer(
    reference: Union[str, Sequence[str]],
    hypothesis: Union[str, Sequence[str]],
    normalization_config: Optional[NormalizationConfig] = None,
    **kwargs: Any,
) -> float:
    """
    Compute Character Error Rate (CER = (S + D + I) / max(1, N_ref)).
    Accepts single strings or sequences of strings. Raises ValueError on length mismatch.
    """
    refs, hyps = _normalize_input_sequences(reference, hypothesis)

    if not refs:
        return 0.0

    if normalization_config is not None:
        refs = [normalize_text(r, normalization_config) for r in refs]
        hyps = [normalize_text(h, normalization_config) for h in hyps]

    total_ref_chars = sum(len(r) for r in refs)
    total_hyp_chars = sum(len(h) for h in hyps)
    if total_ref_chars == 0:
        return 0.0 if total_hyp_chars == 0 else 1.0

    try:
        import jiwer
        out = jiwer.process_characters(refs, hyps)
        return float(out.cer)
    except Exception:
        pass

    # Dynamic programming fallback
    total_s, total_d, total_i = 0, 0, 0
    for r, h in zip(refs, hyps):
        s, d, ins, _ = _levenshtein_ops(list(r), list(h))
        total_s += s
        total_d += d
        total_i += ins

    return float(total_s + total_d + total_i) / float(total_ref_chars)


def compute_wer(
    reference: Union[str, Sequence[str]],
    hypothesis: Union[str, Sequence[str]],
    normalization_config: Optional[NormalizationConfig] = None,
    **kwargs: Any,
) -> float:
    """
    Compute Word Error Rate (WER = (S + D + I) / max(1, N_words_ref)).
    Accepts single strings or sequences of strings. Raises ValueError on length mismatch.
    """
    refs, hyps = _normalize_input_sequences(reference, hypothesis)

    if not refs:
        return 0.0

    if normalization_config is not None:
        refs = [normalize_text(r, normalization_config) for r in refs]
        hyps = [normalize_text(h, normalization_config) for h in hyps]

    total_ref_words = sum(len(r.strip().split()) for r in refs)
    total_hyp_words = sum(len(h.strip().split()) for h in hyps)
    if total_ref_words == 0:
        return 0.0 if total_hyp_words == 0 else 1.0

    try:
        import jiwer
        out = jiwer.process_words(refs, hyps)
        return float(out.wer)
    except Exception:
        pass

    # Dynamic programming fallback
    total_s, total_d, total_i = 0, 0, 0
    for r, h in zip(refs, hyps):
        r_words = r.strip().split()
        h_words = h.strip().split()
        s, d, ins, _ = _levenshtein_ops(r_words, h_words)
        total_s += s
        total_d += d
        total_i += ins

    return float(total_s + total_d + total_i) / float(total_ref_words)


def compute_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalization_config: Optional[NormalizationConfig] = None,
) -> MetricResult:
    """
    Compute comprehensive metrics: raw CER/WER, normalized CER/WER, and full error breakdown.
    Raises ValueError on length mismatch between references and hypotheses.
    """
    raw_refs, raw_hyps = _normalize_input_sequences(references, hypotheses)
    count = len(raw_refs)

    if count == 0:
        return MetricResult()

    norm_cfg = normalization_config if normalization_config is not None else NormalizationConfig()
    norm_refs = [normalize_text(r, norm_cfg) for r in raw_refs]
    norm_hyps = [normalize_text(h, norm_cfg) for h in raw_hyps]

    # Compute raw CER & WER
    raw_cer = compute_cer(raw_refs, raw_hyps)
    raw_wer = compute_wer(raw_refs, raw_hyps)

    # Compute normalized CER & WER + Error Breakdowns
    norm_cer = compute_cer(norm_refs, norm_hyps)
    norm_wer = compute_wer(norm_refs, norm_hyps)

    # Character Breakdown
    char_s, char_d, char_i, char_h, total_chars = 0, 0, 0, 0, 0
    for r, h in zip(norm_refs, norm_hyps):
        s, d, ins, h_count = _levenshtein_ops(list(r), list(h))
        char_s += s
        char_d += d
        char_i += ins
        char_h += h_count
        total_chars += len(r)

    char_err = float(char_s + char_d + char_i) / max(1, total_chars) if total_chars > 0 else (0.0 if char_i == 0 else 1.0)
    char_breakdown = ErrorBreakdown(
        substitutions=char_s,
        deletions=char_d,
        insertions=char_i,
        hits=char_h,
        total_reference_units=total_chars,
        error_rate=round(char_err, 4),
    )

    # Word Breakdown
    word_s, word_d, word_i, word_h, total_words = 0, 0, 0, 0, 0
    for r, h in zip(norm_refs, norm_hyps):
        r_w = r.strip().split()
        h_w = h.strip().split()
        s, d, ins, h_count = _levenshtein_ops(r_w, h_w)
        word_s += s
        word_d += d
        word_i += ins
        word_h += h_count
        total_words += len(r_w)

    word_err = float(word_s + word_d + word_i) / max(1, total_words) if total_words > 0 else (0.0 if word_i == 0 else 1.0)
    word_breakdown = ErrorBreakdown(
        substitutions=word_s,
        deletions=word_d,
        insertions=word_i,
        hits=word_h,
        total_reference_units=total_words,
        error_rate=round(word_err, 4),
    )

    return MetricResult(
        cer=round(raw_cer, 4),
        wer=round(raw_wer, 4),
        normalized_cer=round(norm_cer, 4),
        normalized_wer=round(norm_wer, 4),
        char_breakdown=char_breakdown,
        word_breakdown=word_breakdown,
        sample_count=count,
        normalization_applied=norm_cfg,
    )


# ---------------------------------------------------------------------------
# Latency & Throughput Profiler
# ---------------------------------------------------------------------------

@dataclass
class LatencyMetrics:
    """Statistical distribution of inference latencies in milliseconds."""
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThroughputMetrics:
    """Throughput rates for processed samples, characters, and words."""
    total_samples: int = 0
    total_characters: int = 0
    total_words: int = 0
    total_time_seconds: float = 0.0
    samples_per_second: float = 0.0
    characters_per_second: float = 0.0
    words_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LatencyTracker:
    """
    Captures per-sample and per-batch durations to compute latency percentiles and throughput rates.
    """

    def __init__(self) -> None:
        self.sample_latencies_ms: List[float] = []
        self.batch_latencies_ms: List[float] = []
        self.total_chars: int = 0
        self.total_words: int = 0
        self.total_samples: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_duration_sec: float = 0.0

    def start(self) -> None:
        """Start the global timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop the global timer."""
        if self.start_time is not None:
            self.end_time = time.perf_counter()
            self.total_duration_sec = self.end_time - self.start_time

    def record_sample(self, latency_ms: float, char_count: int = 0, word_count: int = 0) -> None:
        """Record a single sample latency in milliseconds."""
        self.sample_latencies_ms.append(latency_ms)
        self.total_samples += 1
        self.total_chars += char_count
        self.total_words += word_count

    def record_batch(
        self,
        batch_size: int,
        char_count: int,
        word_count: int,
        batch_duration_sec: float,
    ) -> None:
        """Record batch duration and apportion sample-level latencies."""
        batch_ms = batch_duration_sec * 1000.0
        self.batch_latencies_ms.append(batch_ms)
        sample_ms = batch_ms / max(1, batch_size)
        self.sample_latencies_ms.extend([sample_ms] * batch_size)
        self.total_samples += batch_size
        self.total_chars += char_count
        self.total_words += word_count

    def compute_metrics(self) -> Tuple[LatencyMetrics, ThroughputMetrics]:
        """Compute latency percentiles and throughput rates."""
        if self.total_duration_sec == 0.0 and self.start_time is not None:
            self.stop()

        duration = max(1e-6, self.total_duration_sec)
        if not self.sample_latencies_ms and self.batch_latencies_ms:
            self.sample_latencies_ms = list(self.batch_latencies_ms)

        if self.sample_latencies_ms:
            arr = np.array(self.sample_latencies_ms, dtype=np.float64)
            p50 = float(np.percentile(arr, 50))
            p90 = float(np.percentile(arr, 90))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
        else:
            p50 = p90 = p95 = p99 = mean = std = min_val = max_val = 0.0

        latency = LatencyMetrics(
            p50_ms=round(p50, 2),
            p90_ms=round(p90, 2),
            p95_ms=round(p95, 2),
            p99_ms=round(p99, 2),
            mean_ms=round(mean, 2),
            std_ms=round(std, 2),
            min_ms=round(min_val, 2),
            max_ms=round(max_val, 2),
        )

        samples_per_sec = float(self.total_samples) / duration
        chars_per_sec = float(self.total_chars) / duration
        words_per_sec = float(self.total_words) / duration

        throughput = ThroughputMetrics(
            total_samples=self.total_samples,
            total_characters=self.total_chars,
            total_words=self.total_words,
            total_time_seconds=round(duration, 4),
            samples_per_second=round(samples_per_sec, 2),
            characters_per_second=round(chars_per_sec, 2),
            words_per_second=round(words_per_sec, 2),
        )

        return latency, throughput


# Alias for backward compatibility
ThroughputProfiler = LatencyTracker

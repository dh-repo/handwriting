"""
pipeline/tests/test_evaluation.py
Comprehensive test suite for evaluation subsystem:
CER / WER mathematical correctness, text normalizers, error breakdowns,
latency tracker percentiles, throughput rates, in-memory model evaluation,
report JSON export, CLI parser, and edge case resilience.
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pipeline.dataset.dataset_loader import HandwritingSample
from pipeline.evaluation.evaluate import (
    EvaluationReport,
    evaluate_checkpoint,
    evaluate_model,
    load_model_and_processor,
)
from pipeline.evaluation.metrics import (
    ErrorBreakdown,
    LatencyMetrics,
    LatencyTracker,
    MetricResult,
    NormalizationConfig,
    ThroughputMetrics,
    compute_cer,
    compute_metrics,
    compute_wer,
    normalize_text,
)
from pipeline.training.dataset import OCRDataCollator, OCRDataset, create_dummy_processor
from pipeline.training.train import create_tiny_mock_model


# ---------------------------------------------------------------------------
# CER & WER Mathematical Correctness
# ---------------------------------------------------------------------------

def test_cer_wer_exact_math() -> None:
    """Verify exact CER/WER values against analytical ground truths."""
    # 1. Identity match -> 0.0 error
    assert compute_cer("Amoxicillin 500mg", "Amoxicillin 500mg") == 0.0
    assert compute_wer("Amoxicillin 500mg", "Amoxicillin 500mg") == 0.0

    # 2. Single character deletion (1 out of 11 characters -> 1/11)
    ref_word = "Amoxicillin"  # len 11
    hyp_word = "Amoxicilin"   # len 10
    cer_del = compute_cer(ref_word, hyp_word)
    assert abs(cer_del - (1.0 / 11.0)) < 1e-4

    # 3. Single character substitution (1 out of 5 characters -> 0.2)
    assert abs(compute_cer("apple", "apply") - 0.2) < 1e-4

    # 4. Single character insertion (1 out of 5 characters -> 0.2)
    assert abs(compute_cer("apple", "apples") - 0.2) < 1e-4

    # 5. Single word substitution (1 out of 4 words -> 0.25)
    ref_sig = "Take one pill daily"
    hyp_sig = "Take two pill daily"
    assert compute_wer(ref_sig, hyp_sig) == 0.25

    # 6. Complete mismatch -> 1.0 error
    assert compute_cer("AAAA", "BBBB") == 1.0
    assert compute_wer("one two three", "four five six") == 1.0


def test_text_normalization_variations() -> None:
    """Verify text normalizer operations (lowercase, punctuation, whitespace, accents, regex)."""
    cfg_default = NormalizationConfig(lowercase=True, strip_punctuation=False, normalize_whitespace=True)
    assert normalize_text("  Amoxicillin   500MG  \n", cfg_default) == "amoxicillin 500mg"

    cfg_no_punct = NormalizationConfig(lowercase=True, strip_punctuation=True, normalize_whitespace=True)
    assert normalize_text("Dr. Smith, M.D. - Prescription #123!", cfg_no_punct) == "dr smith md prescription 123"

    cfg_accents = NormalizationConfig(lowercase=True, remove_accents=True)
    assert normalize_text("Café Résumé Théâtre", cfg_accents) == "cafe resume theatre"

    cfg_custom = NormalizationConfig(
        lowercase=True,
        custom_replacements=[(r"\bmg\b", "milligrams"), (r"\btab\b", "tablet")],
    )
    assert normalize_text("Take 500 mg tab daily", cfg_custom) == "take 500 milligrams tablet daily"


def test_text_normalization_case_folding_and_custom_regex() -> None:
    """Verify case-folding regex matching and lowercase preservation under uppercase inputs."""
    # Uppercase inputs matching lowercase regex patterns
    cfg_rx = NormalizationConfig(
        lowercase=True,
        custom_replacements=[(r"\bmg\b", "milligrams"), (r"\bpo\b", "by mouth")],
    )
    assert normalize_text("Take 500 MG PO daily", cfg_rx) == "take 500 milligrams by mouth daily"

    # Lowercase inputs matching uppercase regex patterns
    cfg_rx_upper = NormalizationConfig(
        lowercase=True,
        custom_replacements=[(r"\bMG\b", "milligrams"), (r"\bPO\b", "by mouth")],
    )
    assert normalize_text("take 500 mg po daily", cfg_rx_upper) == "take 500 milligrams by mouth daily"

    # Replacement introducing uppercase characters is safely lowercased when lowercase=True
    cfg_rx_caps = NormalizationConfig(
        lowercase=True,
        custom_replacements=[(r"\bdr\b", "DOCTOR")],
    )
    assert normalize_text("Dr Smith", cfg_rx_caps) == "doctor smith"

    # When lowercase=False, case sensitivity is strictly preserved
    cfg_case_sensitive = NormalizationConfig(
        lowercase=False,
        custom_replacements=[(r"\bmg\b", "milligrams")],
    )
    assert normalize_text("Take 500 MG daily", cfg_case_sensitive) == "Take 500 MG daily"
    assert normalize_text("Take 500 mg daily", cfg_case_sensitive) == "Take 500 milligrams daily"


def test_error_breakdown_counts() -> None:
    """Verify granular error breakdown returns exact counts for S, D, I, H."""
    refs = ["the quick brown fox"]
    hyps = ["the fast brown dog jumps"]  # "fast" (S=1), "dog" (S=1), "jumps" (I=1), "the" (H=1), "brown" (H=1)

    res = compute_metrics(refs, hyps)
    wb = res.word_breakdown

    assert wb.hits == 2
    assert wb.substitutions == 2
    assert wb.insertions == 1
    assert wb.deletions == 0
    assert wb.total_reference_units == 4
    assert wb.error_rate == (2 + 1) / 4.0  # (S+D+I)/N_ref = 3/4 = 0.75


def test_metrics_edge_cases_and_guards() -> None:
    """Verify metric computation guards against empty strings, empty lists, and non-ASCII."""
    # Empty reference and empty hypothesis -> 0.0
    assert compute_cer("", "") == 0.0
    assert compute_wer("", "") == 0.0

    # Empty reference with non-empty hypothesis -> 1.0
    assert compute_cer("", "hallucinated text") == 1.0
    assert compute_wer("", "hallucinated text") == 1.0

    # Non-empty reference with empty hypothesis -> 1.0
    assert compute_cer("expected text", "") == 1.0
    assert compute_wer("expected text", "") == 1.0

    # Empty lists
    assert compute_cer([], []) == 0.0
    assert compute_wer([], []) == 0.0
    empty_res = compute_metrics([], [])
    assert empty_res.sample_count == 0
    assert empty_res.cer == 0.0

    # Unicode strings
    assert compute_cer("Αμοξικιλλίνη", "Αμοξικιλλίνη") == 0.0


def test_metric_sequence_length_mismatch_validation() -> None:
    """Verify compute_cer, compute_wer, and compute_metrics raise ValueError on mismatched list lengths."""
    refs = ["hello world", "second sentence"]
    hyps = ["hello world"]

    with pytest.raises(ValueError, match="Mismatched number of references"):
        compute_cer(refs, hyps)

    with pytest.raises(ValueError, match="Mismatched number of references"):
        compute_wer(refs, hyps)

    with pytest.raises(ValueError, match="Mismatched number of references"):
        compute_metrics(refs, hyps)

    with pytest.raises(ValueError, match="Mismatched number of references"):
        compute_cer([], ["orphan hypothesis"])

    with pytest.raises(ValueError, match="Mismatched number of references"):
        compute_wer(["orphan reference"], [])

    with pytest.raises(ValueError, match="Mismatched number of references"):
        compute_metrics([], ["a", "b"])


# ---------------------------------------------------------------------------
# LatencyTracker & Throughput Profiler Tests
# ---------------------------------------------------------------------------

def test_latency_tracker_percentiles() -> None:
    """Verify statistical accuracy of latency percentiles (p50, p90, p95, p99, mean, std)."""
    tracker = LatencyTracker()
    tracker.start()

    # Feed synthetic latencies from 1.0 ms to 100.0 ms
    for i in range(1, 101):
        tracker.record_sample(latency_ms=float(i), char_count=10, word_count=2)

    tracker.total_duration_sec = 2.0  # 2.0 seconds total

    lat, tp = tracker.compute_metrics()

    assert abs(lat.p50_ms - 50.5) <= 1.0
    assert abs(lat.p90_ms - 90.1) <= 1.0
    assert abs(lat.p95_ms - 95.05) <= 1.0
    assert abs(lat.p99_ms - 99.01) <= 1.0
    assert abs(lat.mean_ms - 50.5) <= 0.1
    assert lat.min_ms == 1.0
    assert lat.max_ms == 100.0


def test_throughput_profiler_rates() -> None:
    """Verify throughput rate arithmetic (samples/sec, chars/sec, words/sec)."""
    tracker = LatencyTracker()
    tracker.start()

    tracker.record_batch(batch_size=10, char_count=200, word_count=50, batch_duration_sec=0.5)
    tracker.record_batch(batch_size=10, char_count=300, word_count=70, batch_duration_sec=0.5)
    tracker.total_duration_sec = 1.0

    lat, tp = tracker.compute_metrics()

    assert tp.total_samples == 20
    assert tp.total_characters == 500
    assert tp.total_words == 120
    assert tp.samples_per_second == 20.0
    assert tp.characters_per_second == 500.0
    assert tp.words_per_second == 120.0


# ---------------------------------------------------------------------------
# In-Memory Model Evaluation & Report Generation
# ---------------------------------------------------------------------------

def test_evaluate_model_in_memory() -> None:
    """Verify evaluate_model executes over an in-memory VisionEncoderDecoderModel and returns an EvaluationReport."""
    processor = create_dummy_processor(vocab_size=50, size=(64, 64))
    model = create_tiny_mock_model(vocab_size=50, image_size=64)

    samples = [
        HandwritingSample(sample_id="s1", text="Amoxicillin 500mg", image=np.full((64, 64, 3), 200, dtype=np.uint8)),
        HandwritingSample(sample_id="s2", text="Take 1 tablet daily", image=np.full((64, 64, 3), 220, dtype=np.uint8)),
    ]

    dataset = OCRDataset(samples=samples, processor=processor, is_training=False)
    collator = OCRDataCollator(processor=processor)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

    report = evaluate_model(
        model=model,
        processor=processor,
        dataloader=loader,
        device=torch.device("cpu"),
        normalization_config=NormalizationConfig(lowercase=True),
        max_new_tokens=10,
    )

    assert isinstance(report, EvaluationReport)
    assert report.metrics.sample_count == 2
    assert len(report.sample_predictions) == 2
    assert report.throughput.total_samples == 2
    assert report.latency.mean_ms >= 0.0

    table = report.summary_table()
    assert "Handwriting Recognition Evaluation Report" in table
    assert "Character Error Rate" in table


def test_evaluate_json_report_export(tmp_path: Path) -> None:
    """Verify evaluation report serializes to structured JSON and can be deserialized."""
    report = EvaluationReport(
        metrics=MetricResult(
            cer=0.045,
            wer=0.120,
            normalized_cer=0.038,
            normalized_wer=0.095,
            char_breakdown=ErrorBreakdown(substitutions=10, deletions=2, insertions=1, hits=250, total_reference_units=262, error_rate=0.0496),
            word_breakdown=ErrorBreakdown(substitutions=5, deletions=1, insertions=0, hits=50, total_reference_units=56, error_rate=0.1071),
            sample_count=20,
        ),
        latency=LatencyMetrics(p50_ms=18.5, p90_ms=25.2, p95_ms=30.0, p99_ms=42.1, mean_ms=19.8, std_ms=4.5),
        throughput=ThroughputMetrics(total_samples=20, total_characters=1200, total_words=250, total_time_seconds=0.4, samples_per_second=50.0),
        checkpoint="checkpoints/best_model",
        device="cpu",
    )

    json_file = tmp_path / "eval_report.json"
    report.to_json(json_file)
    assert json_file.exists()

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["checkpoint"] == "checkpoints/best_model"
    assert data["sample_count"] == 20
    assert data["mean_cer"] == 0.038
    assert data["p50_latency_ms"] == 18.5
    assert data["throughput_samples_per_sec"] == 50.0
    assert "char_breakdown" in data
    assert "word_breakdown" in data


def test_evaluate_error_handling() -> None:
    """Verify evaluation loader raises FileNotFoundError on missing checkpoint."""
    with pytest.raises(FileNotFoundError):
        load_model_and_processor("non_existent_path_to_checkpoint")

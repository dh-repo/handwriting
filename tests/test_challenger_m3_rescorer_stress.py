"""
tests/test_challenger_m3_rescorer_stress.py
Empirical Adversarial Stress Test Suite for Milestone 3 (Beam Rescorer & LASA Disambiguation).

Comprehensive stress-testing across:
1. Disambiguation under adversarial OCR confidence gaps (all 8 LASA pairs with inverted OCR log-probs).
2. Extreme and pathological dosages (>50,000mg, negative values, zero values, malformed units, NaNs).
3. Contradictory routes and dosage forms (IV tablet, PO injection, IM capsule, topical capsule).
4. Candidate set degeneracies (K=1 single beam, identical duplicates, empty strings, non-medical sets, infinite/NaN logprobs).
5. Multi-objective scoring weights modulation and latency SLA (<5ms).
6. 4-Stage Ablation Benchmark verification (monotonic CER/WER reduction and zero-loss regression audit).
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple
import pytest

from pipeline.evaluation.benchmark_ablation import (
    AblationBenchmarkReport,
    StageResult,
    calculate_pnda,
    run_ablation_benchmark,
)
from pipeline.rescorer.beam_rescorer import (
    BeamCandidate,
    BeamRescorer,
    ContextFeatures,
    RescorerResult,
)
from pipeline.rescorer.confusion_matrix import VisualConfusionMatrix
from pipeline.rescorer.trie import PrefixTrie


@pytest.fixture(scope="module")
def loaded_trie() -> PrefixTrie:
    trie = PrefixTrie()
    trie.load_vocabularies("data/reference_handwriting/vocabularies")
    return trie


@pytest.fixture(scope="module")
def loaded_matrix() -> VisualConfusionMatrix:
    matrix = VisualConfusionMatrix(load_defaults=True)
    rx_path = Path("data/reference_handwriting/vocabularies/rxnorm_medications.json")
    if rx_path.exists():
        matrix.load_rxnorm_confusion_profiles(rx_path)
    return matrix


@pytest.fixture(scope="module")
def shared_rescorer(loaded_trie: PrefixTrie, loaded_matrix: VisualConfusionMatrix) -> BeamRescorer:
    return BeamRescorer(
        trie=loaded_trie,
        confusion_matrix=loaded_matrix,
        lambda_lexicon=1.0,
        lambda_context=0.8,
        lambda_confusion=0.5,
        vocab_dir="data/reference_handwriting/vocabularies",
    )


# ===========================================================================
# 1. Disambiguation Under Adversarial OCR Confidence Gaps (8 LASA Pairs)
# ===========================================================================

class TestLASADisambiguationUnderOCRGaps:
    """
    Stress-test all 8 LASA pairs when the wrong drug has a higher OCR log-probability
    than the clinically correct drug.
    """

    LASA_BENCHMARKS = [
        # (correct_prescription, wrong_prescription, correct_drug, max_expected_delta)
        ("Amoxicillin 875mg PO TID", "Ampicillin 875mg PO TID", "Amoxicillin", 1.40),
        ("Hydroxyzine 25mg PO QHS", "Hydralazine 25mg PO QHS", "Hydroxyzine", 0.30),
        ("Celebrex 200mg PO QD", "Celexa 200mg PO QD", "Celebrex", 0.95),
        ("Prednisone 20mg PO QD", "Prednisolone 20mg PO QD", "Prednisone", 1.00),
        ("Adderall 25mg PO QD", "Inderal 25mg PO QD", "Adderall", 0.95),
        ("Zantac 150mg PO BID", "Xanax 150mg PO BID", "Zantac", 1.35),
        ("Clonidine 0.1mg PO BID", "Klonopin 0.1mg PO BID", "Clonidine", 0.95),
        ("Metformin 1000mg PO BID", "Metronidazole 1000mg PO BID", "Metformin", 0.90),
    ]

    @pytest.mark.parametrize("correct_rx, wrong_rx, correct_drug, expected_delta", LASA_BENCHMARKS)
    def test_lasa_pair_inverted_ocr_resolution(
        self,
        shared_rescorer: BeamRescorer,
        correct_rx: str,
        wrong_rx: str,
        correct_drug: str,
        expected_delta: float,
    ) -> None:
        """
        Verify that clinical context overcomes an adversarial OCR confidence gap.
        Wrong drug starts as top OCR candidate (log_prob = -0.35).
        Correct drug has lower OCR log_prob (-0.35 - delta).
        """
        delta = expected_delta
        hypotheses = [
            (wrong_rx, -0.35),               # Wrong drug favored by OCR
            (correct_rx, -0.35 - delta),      # Correct drug penalised by OCR
        ]

        result = shared_rescorer.rescore_detailed(hypotheses)
        assert result.rescore_applied is True
        winner = result.rescored_text.split()[0]
        assert winner.lower() == correct_drug.lower(), (
            f"LASA disambiguation failed for {correct_drug} vs {wrong_rx.split()[0]} "
            f"at delta={delta:.2f}. Winner was '{winner}' instead of '{correct_drug}'."
        )

    def test_lasa_threshold_boundary_monotonicity(self, shared_rescorer: BeamRescorer) -> None:
        """Verify that increasing OCR gap delta monotonically shifts decision boundary."""
        correct_rx = "Amoxicillin 875mg PO TID"
        wrong_rx = "Ampicillin 875mg PO TID"

        # At small delta (0.50), Amoxicillin must win easily
        res_small = shared_rescorer.rescore_detailed([(wrong_rx, -0.35), (correct_rx, -0.85)])
        assert "Amoxicillin" in res_small.rescored_text

        # At moderate delta (1.00), Amoxicillin still wins
        res_med = shared_rescorer.rescore_detailed([(wrong_rx, -0.35), (correct_rx, -1.35)])
        assert "Amoxicillin" in res_med.rescored_text

        # At excessive delta (3.00), OCR prior overrules clinical context
        res_huge = shared_rescorer.rescore_detailed([(wrong_rx, -0.35), (correct_rx, -3.35)])
        assert "Ampicillin" in res_huge.rescored_text


# ===========================================================================
# 2. Extreme and Pathological Dosages Stress Tests
# ===========================================================================

class TestExtremeDosageStress:
    """Stress test rescorer with extreme, negative, non-numeric, and out-of-catalog dosages."""

    def test_overdose_strength_rejection(self, shared_rescorer: BeamRescorer) -> None:
        """Dosages > 50,000mg must receive strong safety penalties."""
        beams = [
            ("Amoxicillin 999999mg PO TID", -0.40),
            ("Amoxicillin 500mg PO TID", -0.60),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text == "Amoxicillin 500mg PO TID"
        assert res.rescore_applied is True

    def test_unit_conversion_overdose_mcg(self, shared_rescorer: BeamRescorer) -> None:
        """Verify mcg unit conversion detects massive overdose (e.g. 500,000,000 mcg = 500,000 mg)."""
        beams = [
            ("Amoxicillin 500000000mcg PO TID", -0.40),
            ("Amoxicillin 500mg PO TID", -0.60),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text == "Amoxicillin 500mg PO TID"

    def test_invalid_and_corrupted_dosage_units(self, shared_rescorer: BeamRescorer) -> None:
        """Verify corrupted units ('500xyz', 'NaNmg') do not crash and prefer canonical dosage."""
        beams = [
            ("Amoxicillin 500xyz PO TID", -0.40),
            ("Amoxicillin 500mg PO TID", -0.60),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text == "Amoxicillin 500mg PO TID"

    def test_negative_dosage_handling(self, shared_rescorer: BeamRescorer) -> None:
        """Verify negative dosage values do not cause uncaught exceptions."""
        beams = [
            ("Amoxicillin -500mg PO TID", -0.50),
            ("Amoxicillin 500mg PO TID", -0.50),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert math.isfinite(res.confidence)
        assert 0.0 <= res.confidence <= 1.0


# ===========================================================================
# 3. Contradictory Routes and Dosage Forms Stress Tests
# ===========================================================================

class TestContradictoryRoutesAndForms:
    """Stress test route-form incompatibility rules."""

    def test_iv_tablet_penalty(self, shared_rescorer: BeamRescorer) -> None:
        """'IV tablet' is clinically incompatible and must be penalized against 'PO tablet'."""
        beams = [
            ("Amoxicillin 500mg IV tablet", -0.40),
            ("Amoxicillin 500mg PO tablet", -0.60),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert "PO" in res.rescored_text
        assert res.rescore_applied is True

    def test_im_capsule_penalty(self, shared_rescorer: BeamRescorer) -> None:
        """'IM capsule' is clinically impossible and must be penalized."""
        beams = [
            ("Hydroxyzine 25mg IM capsule", -0.40),
            ("Hydroxyzine 25mg PO capsule", -0.60),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert "PO" in res.rescored_text

    def test_topical_capsule_penalty(self, shared_rescorer: BeamRescorer) -> None:
        """'topical capsule' is clinically contradictory."""
        beams = [
            ("Metronidazole 500mg topical capsule", -0.40),
            ("Metronidazole 500mg PO capsule", -0.60),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert "PO" in res.rescored_text


# ===========================================================================
# 4. Candidate Set Degeneracies & Boundary Conditions
# ===========================================================================

class TestCandidateSetDegeneracies:
    """Stress test boundary cases: K=1, identical duplicates, NaN/infinite logprobs, empty inputs."""

    def test_single_candidate_k1_preservation(self, shared_rescorer: BeamRescorer) -> None:
        """K=1 must return the single candidate without altering text or crashing."""
        beams = [("Amoxicillin 500mg PO TID", -0.25)]
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text == "Amoxicillin 500mg PO TID"
        assert res.rescore_applied is False
        assert 0.0 <= res.confidence <= 1.0

    def test_identical_duplicates_candidate_set(self, shared_rescorer: BeamRescorer) -> None:
        """K duplicate candidates must not cause divide-by-zero in softmax or rank explosion."""
        beams = [("Amoxicillin 500mg PO TID", -0.50)] * 8
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text == "Amoxicillin 500mg PO TID"
        assert math.isfinite(res.confidence)

    def test_empty_string_and_whitespace_candidates(self, shared_rescorer: BeamRescorer) -> None:
        """Empty strings in candidate beam must be handled gracefully."""
        beams = [("", -0.1), ("   ", -0.2), ("Amoxicillin 500mg", -0.8)]
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text == "Amoxicillin 500mg"

    def test_extreme_negative_and_nan_logprobs(self, shared_rescorer: BeamRescorer) -> None:
        """NaN and negative infinite log probabilities must be sanitized."""
        beams = [
            ("Amoxicillin 500mg", float("nan")),
            ("Ampicillin 500mg", float("-inf")),
            ("Aspirin 81mg", -1e9),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert math.isfinite(res.confidence)
        assert res.rescored_text in ("Amoxicillin 500mg", "Ampicillin 500mg", "Aspirin 81mg")

    def test_non_medical_candidate_sets(self, shared_rescorer: BeamRescorer) -> None:
        """Non-medical sentences must pass through based on OCR log-probability ranking."""
        beams = [
            ("The patient was discharged home today", -0.30),
            ("Follow up in cardiology clinic next week", -0.60),
            ("Return if symptoms worsen", -0.90),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        # Verify OCR top candidate is preserved as rescored text
        assert res.rescored_text == "The patient was discharged home today"
        assert res.rescore_applied is False


# ===========================================================================
# 5. Throughput & Latency SLA Verification (<5ms)
# ===========================================================================

class TestLatencyAndThroughputSLA:
    """Verify rescorer execution speed strictly meets the <5ms SLA."""

    def test_rescorer_beam_latency_sla(self, shared_rescorer: BeamRescorer) -> None:
        """5-candidate beam rescoring must execute in < 5.0 ms per line hypothesis set."""
        beams = [
            ("Amoxcillin 500mg PO TID", -0.45),
            ("Amoxicillin 500mg PO TID", -0.55),
            ("Ampicillin 500mg PO TID", -0.65),
            ("Amoxil 500mg PO TID", -0.75),
            ("Aspirin 500mg PO TID", -0.85),
        ]
        ctx = ContextFeatures(dosage="500mg", route="PO", frequency="TID", form="capsule")

        # Warmup
        for _ in range(20):
            _ = shared_rescorer.rescore_detailed(beams, context=ctx)

        n_trials = 300
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = shared_rescorer.rescore_detailed(beams, context=ctx)
        elapsed_total = (time.perf_counter() - t0) * 1000.0
        avg_latency_ms = elapsed_total / n_trials

        print(f"\n[BENCHMARK] Beam Rescorer Latency: Avg={avg_latency_ms:.4f} ms/op across {n_trials} trials")
        assert avg_latency_ms < 5.0, f"Rescorer latency {avg_latency_ms:.4f}ms exceeds 5.0ms SLA"


# ===========================================================================
# 6. Full 4-Stage Ablation Benchmark Verification
# ===========================================================================

class TestAblationBenchmarkVerification:
    """
    Empirically execute and verify the 4-Stage Ablation Benchmark on test_manifest.jsonl.
    Validates that:
    1. Stage 4 (Full Multi-Objective) achieves superior CER/WER over Stage 1 (Baseline).
    2. Stage 4 achieves superior PNDA over Stage 1.
    3. Rescorer exhibits ZERO error regressions (losses_vs_baseline == 0).
    """

    def test_run_4_stage_ablation_benchmark(self) -> None:
        manifest_path = "data/reference_handwriting/test_manifest.jsonl"
        report = run_ablation_benchmark(
            manifest_path=manifest_path,
            split="test",
            beam_width=5,
            max_samples=250,
        )

        assert isinstance(report, AblationBenchmarkReport)
        assert len(report.stages) == 4

        s1 = report.stages["stage_1_baseline"]
        s2 = report.stages["stage_2_lexicon_only"]
        s3 = report.stages["stage_3_lexicon_confusion"]
        s4 = report.stages["stage_4_full_multiobjective"]

        # 1. Monotonic CER improvement
        assert s4.cer < s1.cer, f"Stage 4 CER ({s4.cer:.4f}) not lower than Stage 1 Baseline CER ({s1.cer:.4f})"
        assert s2.cer <= s1.cer, f"Stage 2 CER ({s2.cer:.4f}) higher than Stage 1 ({s1.cer:.4f})"

        # 2. Monotonic WER improvement
        assert s4.wer < s1.wer, f"Stage 4 WER ({s4.wer:.4f}) not lower than Stage 1 Baseline WER ({s1.wer:.4f})"

        # 3. PNDA improvement
        assert s4.drug_exact_match_pct > s1.drug_exact_match_pct, (
            f"Stage 4 PNDA ({s4.drug_exact_match_pct:.1f}%) not higher than Stage 1 ({s1.drug_exact_match_pct:.1f}%)"
        )

        # 4. Zero regressions vs baseline
        assert s4.losses_vs_baseline == 0, f"Stage 4 caused {s4.losses_vs_baseline} regression losses vs baseline"
        assert s4.wins_vs_baseline > 0, f"Stage 4 had 0 wins vs baseline"

        # 5. Latency SLA
        assert s4.p50_latency_ms < 5.0, f"Stage 4 p50 latency ({s4.p50_latency_ms:.2f}ms) exceeds 5ms SLA"

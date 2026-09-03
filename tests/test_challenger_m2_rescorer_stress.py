"""
tests/test_challenger_m2_rescorer_stress.py
Adversarial Empirical Stress Harness for Milestone 2:
Online Self-Tuning Visual Confusion Matrix & Beam Rescorer Recalibration.

Author: challenger_m2_2 (Empirical Challenger)
Target Milestone: Milestone 2

Empirically stress-tests:
1. Adversarial adaptation on critical Look-Alike Sound-Alike (LASA) drug pairs
   under extreme learning rates (e.g. 0.99) and negative min_cost requests:
   proves that costs never drop below 0.15 and clinical safety is never violated.
2. Multi-candidate beam competition (5+ competing hypotheses with mixed lexicons
   and context features):
   proves that tuning a confusion pair specifically flips the targeted candidate
   while preserving the exact scores and relative ordering of unrelated candidates.
3. InferenceEngine live adaptation:
   proves that engine.adapt_confusion_matrix() immediately propagates into
   subsequent engine.rescorer.rescore_detailed() calls in real-time, supports
   persistence roundtrip, and handles concurrent multi-threaded requests safely.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
import pytest

from pipeline.rescorer.confusion_matrix import VisualConfusionMatrix
from pipeline.rescorer.beam_rescorer import (
    BeamCandidate,
    BeamRescorer,
    RescorerResult,
)
from pipeline.rescorer.trie import PrefixTrie
from backend.app.engine import InferenceEngine


# ===========================================================================
# 1. Adversarial LASA Pairs and Clinical Safety Bounds Stress Tests
# ===========================================================================

class TestAdversarialLASAPairsAndSafetyFloor:
    """Adversarial stress-testing of clinical LASA pairs and minimum cost floor."""

    CRITICAL_LASA_PAIRS: List[Tuple[str, str]] = [
        ("hydroxyzine", "hydralazine"),
        ("clonidine", "klonopin"),
        ("prednisone", "prednisolone"),
        ("celebrex", "celexa"),
        ("adderall", "inderal"),
        ("zantac", "xanax"),
        ("metformin", "metronidazole"),
        ("amoxicillin", "ampicillin"),
    ]

    @pytest.fixture
    def cm(self) -> VisualConfusionMatrix:
        return VisualConfusionMatrix(load_defaults=True)

    def test_extreme_learning_rate_adversarial_iterations(self, cm: VisualConfusionMatrix) -> None:
        """
        Adversarially adapt all 8 critical LASA pairs 50 times each with lr=0.99
        and adversarial min_cost=-50.0. Prove that no cost in the matrix ever
        collapses below 0.15, and string distance remains >= 0.15.
        """
        for pair_a, pair_b in self.CRITICAL_LASA_PAIRS:
            for _ in range(50):
                cm.adapt_from_correction(pair_a, pair_b, learning_rate=0.99, min_cost=-50.0)
                cm.adapt_from_correction(pair_b, pair_a, learning_rate=0.99, min_cost=0.0)

            # Check pairwise distance
            dist_ab = cm.compute_distance(pair_a, pair_b)
            dist_ba = cm.compute_distance(pair_b, pair_a)
            assert dist_ab >= 0.15, f"Distance between {pair_a} and {pair_b} collapsed to {dist_ab}"
            assert dist_ba >= 0.15, f"Distance between {pair_b} and {pair_a} collapsed to {dist_ba}"
            assert abs(dist_ab - dist_ba) < 1e-6, "Asymmetric distance between LASA pair!"

        # Exhaustive verification of entire internal matrix table
        for (src, tgt), cost in cm._matrix.items():
            assert cost >= 0.15, (
                f"Clinical safety violation! Cost for ({src}, {tgt}) fell below 0.15: {cost}"
            )

    def test_lasa_clinical_safety_preservation_in_beam_rescoring(self) -> None:
        """
        Verify that an adversarial attack trying to adapt 'hydroxyzine' into 'hydralazine'
        CANNOT cause clinical misidentification when OCR and clinical context support
        the true drug.
        """
        vocab_path = Path("data/reference_handwriting/vocabularies")
        trie = PrefixTrie()
        if vocab_path.exists():
            trie.load_vocabularies(vocab_path)
        cm = VisualConfusionMatrix(load_defaults=True)
        rescorer = BeamRescorer(
            trie=trie,
            confusion_matrix=cm,
            lambda_lexicon=1.0,
            lambda_context=0.8,
            lambda_confusion=8.0,
            vocab_dir=vocab_path,
        )

        # Baseline: OCR confidently identified hydroxyzine 25mg
        hyps = [
            BeamCandidate(text="hydroxyzine 25mg", log_prob=-0.05),
            BeamCandidate(text="hydralazine 25mg", log_prob=-0.40),
        ]

        # Initial rescore: hydroxyzine wins
        r1 = rescorer.rescore_detailed(hyps)
        assert r1.rescored_text == "hydroxyzine 25mg"

        # Attacker repeatedly submits adversarial feedback to force hydroxyzine -> hydralazine
        for _ in range(25):
            rescorer.adapt_confusion_matrix(
                original_prediction="hydroxyzine",
                operator_correction="hydralazine",
                learning_rate=0.99,
                min_cost=0.0,
            )

        # Even after extreme adversarial adaptation, hydroxyzine STILL correctly wins
        r2 = rescorer.rescore_detailed(hyps)
        assert r2.rescored_text == "hydroxyzine 25mg", (
            f"Clinical safety breach! Erroneously swapped to {r2.rescored_text}"
        )
        assert r2.all_candidates[0]["text"] == "hydroxyzine 25mg"
        assert r2.all_candidates[1]["text"] == "hydralazine 25mg"
        assert r2.all_candidates[0]["final_score"] > r2.all_candidates[1]["final_score"]

    def test_asymmetric_adversarial_adaptation_maintains_symmetry(self, cm: VisualConfusionMatrix) -> None:
        """
        Adapting in only one direction (e.g. A -> B 20 times) must maintain
        strict bidirectionality: cost(A, B) == cost(B, A) >= 0.15.
        """
        for _ in range(20):
            cm.adapt_from_correction("hydroxyzine", "hydralazine", learning_rate=0.80, min_cost=0.15)

        cost_ab = cm.get_cost("o", "a")
        cost_ba = cm.get_cost("a", "o")
        assert cost_ab >= 0.15
        assert cost_ba >= 0.15
        assert cost_ab == cost_ba, f"Asymmetry detected: {cost_ab} != {cost_ba}"

    def test_denial_of_service_string_length_ceiling(self, cm: VisualConfusionMatrix) -> None:
        """
        Adversarial payload with length > 1000 characters must be rejected immediately
        without performing $O(N \times M)$ DP alignment or mutating the matrix.
        """
        initial_pairs_count = len(cm._matrix)
        adversarial_long_a = "a" * 1500
        adversarial_long_b = "b" * 1500

        updates = cm.adapt_from_correction(adversarial_long_a, adversarial_long_b, learning_rate=0.50)
        assert updates == []
        assert len(cm._matrix) == initial_pairs_count


# ===========================================================================
# 2. Multi-Candidate Beam Competition and Isolation Tests
# ===========================================================================

class TestMultiCandidateBeamCompetitionAndIsolation:
    """
    Stress-test multi-candidate beam competition with 5+ competing hypotheses.
    Verifies that tuning a specific confusion pair flips the targeted candidate
    without distorting scores or relative rankings of unrelated candidates.
    """

    @pytest.fixture
    def rescorer(self) -> BeamRescorer:
        vocab_path = Path("data/reference_handwriting/vocabularies")
        trie = PrefixTrie()
        if vocab_path.exists():
            trie.load_vocabularies(vocab_path)
        cm = VisualConfusionMatrix(load_defaults=True)
        return BeamRescorer(
            trie=trie,
            confusion_matrix=cm,
            lambda_lexicon=1.0,
            lambda_context=0.8,
            lambda_confusion=8.0,
            vocab_dir=vocab_path,
        )

    def test_five_plus_candidates_targeted_flip_and_unrelated_invariance(
        self,
        rescorer: BeamRescorer,
    ) -> None:
        """
        Competition with 6 candidates:
        - Candidate 0: 'xlonopin 1mg' (erroneous OCR top beam, log_prob -0.10)
        - Candidate 1: 'klonopin 1mg' (valid pharmaceutical drug, log_prob -0.35)
        - Candidate 2: 'flonopin 1mg' (unrelated OCR noise, log_prob -0.90)
        - Candidate 3: 'clonazepam 1mg' (unrelated valid drug, log_prob -0.45)
        - Candidate 4: 'lorazepam 1mg' (unrelated valid drug, log_prob -0.60)
        - Candidate 5: 'diazepam 1mg' (unrelated valid drug, log_prob -0.80)

        Verify:
        1. Before adaptation: 'xlonopin 1mg' wins (#1).
        2. Adapt 'xlonopin' -> 'klonopin' (optical substitution 'x' <-> 'k').
        3. After adaptation: 'klonopin 1mg' flips to #1.
        4. Invariance:
           - The scores of ALL unrelated candidates remain EXACTLY identical.
           - The relative ordering of ALL unrelated candidates is 100% PRESERVED.
        """
        hypotheses = [
            BeamCandidate(text="xlonopin 1mg", log_prob=-0.10),
            BeamCandidate(text="klonopin 1mg", log_prob=-0.35),
            BeamCandidate(text="flonopin 1mg", log_prob=-0.90),
            BeamCandidate(text="clonazepam 1mg", log_prob=-0.45),
            BeamCandidate(text="lorazepam 1mg", log_prob=-0.60),
            BeamCandidate(text="diazepam 1mg", log_prob=-0.80),
        ]
        assert len(hypotheses) >= 5, "Must test at least 5 competing hypotheses"

        # 1. Rescore before adaptation
        r_before = rescorer.rescore_detailed(hypotheses)
        assert r_before.rescored_text == "xlonopin 1mg"

        # Record scores and ordering of unrelated candidates
        unrelated_texts = ["flonopin 1mg", "clonazepam 1mg", "lorazepam 1mg", "diazepam 1mg"]
        scores_before = {
            c["text"]: c["final_score"]
            for c in r_before.all_candidates
            if c["text"] in unrelated_texts
        }
        order_before = [
            c["text"]
            for c in r_before.all_candidates
            if c["text"] in unrelated_texts
        ]

        # 2. Perform targeted adaptation specifically for x <-> k
        updates = rescorer.adapt_confusion_matrix(
            original_prediction="xlonopin",
            operator_correction="klonopin",
            learning_rate=0.85,
            min_cost=0.15,
        )
        assert len(updates) >= 1
        assert any(u["source"] == "x" and u["target"] == "k" for u in updates)

        # 3. Rescore after adaptation on identical candidates
        r_after = rescorer.rescore_detailed(hypotheses)

        # A) Targeted candidate flipped to #1
        assert r_after.rescored_text == "klonopin 1mg", (
            f"Expected rank flip to 'klonopin 1mg', got '{r_after.rescored_text}'"
        )
        assert r_after.rescore_applied is True
        assert r_after.delta_score > 0.0

        # B) Unrelated candidates score invariance
        scores_after = {
            c["text"]: c["final_score"]
            for c in r_after.all_candidates
            if c["text"] in unrelated_texts
        }
        for text in unrelated_texts:
            assert abs(scores_after[text] - scores_before[text]) < 1e-9, (
                f"Unrelated candidate '{text}' score distorted! "
                f"Before: {scores_before[text]}, After: {scores_after[text]}"
            )

        # C) Relative ordering invariance among unrelated candidates
        order_after = [
            c["text"]
            for c in r_after.all_candidates
            if c["text"] in unrelated_texts
        ]
        assert order_after == order_before, (
            f"Relative order of unrelated candidates was distorted!\n"
            f"Before: {order_before}\nAfter: {order_after}"
        )

    def test_multi_candidate_with_contraction_isolation(
        self,
        rescorer: BeamRescorer,
    ) -> None:
        """
        Verify multi-candidate competition with contraction operation 'rn' -> 'm'.
        Ensures disjoint drug candidates (Ciprofloxacin, Doxycycline, Azithromycin)
        retain strictly unchanged scores and ranking.
        """
        hyps = [
            BeamCandidate(text="Arnoxicillin 500mg", log_prob=-0.05),
            BeamCandidate(text="Amoxicillin 500mg", log_prob=-0.35),
            BeamCandidate(text="Ampicillin 500mg", log_prob=-0.50),
            BeamCandidate(text="Ciprofloxacin 500mg", log_prob=-0.90),
            BeamCandidate(text="Doxycycline 100mg", log_prob=-1.10),
            BeamCandidate(text="Azithromycin 500mg", log_prob=-1.30),
        ]

        r_before = rescorer.rescore_detailed(hyps)
        disjoint_texts = ["Ciprofloxacin 500mg", "Doxycycline 100mg", "Azithromycin 500mg"]
        scores_before = {
            c["text"]: c["final_score"]
            for c in r_before.all_candidates
            if c["text"] in disjoint_texts
        }

        # Adapt contraction rn <-> m
        rescorer.adapt_confusion_matrix(
            original_prediction="Arnoxicillin",
            operator_correction="Amoxicillin",
            learning_rate=0.80,
            min_cost=0.15,
        )

        r_after = rescorer.rescore_detailed(hyps)
        scores_after = {
            c["text"]: c["final_score"]
            for c in r_after.all_candidates
            if c["text"] in disjoint_texts
        }

        # Target candidate score improved
        target_score_before = next(c["final_score"] for c in r_before.all_candidates if c["text"] == "Amoxicillin 500mg")
        target_score_after = next(c["final_score"] for c in r_after.all_candidates if c["text"] == "Amoxicillin 500mg")
        assert target_score_after > target_score_before

        # Disjoint candidates are 100% invariant
        for t in disjoint_texts:
            assert abs(scores_after[t] - scores_before[t]) < 1e-9, f"Distortion in disjoint candidate {t}"

    def test_mixed_context_and_lexicon_scores_are_preserved(
        self,
        rescorer: BeamRescorer,
    ) -> None:
        """
        Verify that confusion adaptation ONLY modifies Penalty_confusion and does NOT
        pollute or alter s_lex or s_ctx across any candidate.
        """
        hyps = [
            BeamCandidate(text="xlonopin 1mg", log_prob=-0.10),
            BeamCandidate(text="klonopin 1mg", log_prob=-0.35),
            BeamCandidate(text="clonazepam 1mg", log_prob=-0.45),
            BeamCandidate(text="lorazepam 2mg", log_prob=-0.60),
            BeamCandidate(text="diazepam 5mg", log_prob=-0.80),
        ]

        r_before = rescorer.rescore_detailed(hyps)
        features_before = {
            c["text"]: (c["s_lex"], c["s_ctx"])
            for c in r_before.all_candidates
        }

        rescorer.adapt_confusion_matrix("xlonopin", "klonopin", learning_rate=0.85, min_cost=0.15)

        r_after = rescorer.rescore_detailed(hyps)
        features_after = {
            c["text"]: (c["s_lex"], c["s_ctx"])
            for c in r_after.all_candidates
        }

        for text in features_before:
            assert features_before[text] == features_after[text], (
                f"Feature contamination for {text}! Before: {features_before[text]}, After: {features_after[text]}"
            )


# ===========================================================================
# 3. InferenceEngine Live Real-Time Adaptation and Concurrency Tests
# ===========================================================================

class TestInferenceEngineLiveAdaptationRealTime:
    """
    Stress-test live InferenceEngine adaptation:
    - Real-time propagation to subsequent rescore_detailed() calls.
    - Dynamic state persistence roundtrip.
    - Concurrent multi-threaded adaptation and rescoring.
    """

    def test_live_engine_rescorer_rank_flip(self) -> None:
        """
        Test that calling engine.adapt_confusion_matrix() immediately flips candidate
        ranking on the subsequent engine.rescorer.rescore_detailed() execution.
        """
        engine = InferenceEngine(mode="mock", enable_rescorer=True, confusion_weight=8.0)
        assert engine.rescorer is not None

        hyps = [
            BeamCandidate(text="xlonopin 1mg", log_prob=-0.10),
            BeamCandidate(text="klonopin 1mg", log_prob=-0.40),
        ]

        # 1. Before live adaptation
        res1 = engine.rescorer.rescore_detailed(hyps)
        assert res1.rescored_text == "xlonopin 1mg"

        # 2. Live adaptation via engine API
        updates = engine.adapt_confusion_matrix(
            original_prediction="xlonopin",
            operator_correction="klonopin",
            learning_rate=0.85,
            min_cost=0.15,
        )
        assert len(updates) >= 1

        # 3. Immediately rescore on engine
        res2 = engine.rescorer.rescore_detailed(hyps)
        assert res2.rescored_text == "klonopin 1mg"
        assert res2.rescore_applied is True
        assert res2.delta_score > 0.0

    def test_engine_live_adaptation_persistence_roundtrip(self, tmp_path: Path) -> None:
        """
        Verify that engine.adapt_confusion_matrix with auto_persist=True saves the state,
        and a new engine instance loading this file starts with the adapted weights.
        """
        persist_file = tmp_path / "dynamic_matrix.json"
        engine1 = InferenceEngine(mode="mock", enable_rescorer=True, confusion_weight=8.0)
        assert engine1.rescorer is not None

        updates = engine1.adapt_confusion_matrix(
            original_prediction="xlonopin",
            operator_correction="klonopin",
            learning_rate=0.85,
            min_cost=0.15,
            auto_persist=True,
            persist_filepath=persist_file,
        )
        assert len(updates) >= 1
        assert persist_file.exists()

        # Instantiate second engine and reload dynamic state
        engine2 = InferenceEngine(mode="mock", enable_rescorer=True, confusion_weight=8.0)
        assert engine2.rescorer is not None
        loaded_count = engine2.rescorer.reload_confusion_matrix(persist_file)
        assert loaded_count >= 1

        hyps = [
            BeamCandidate(text="xlonopin 1mg", log_prob=-0.10),
            BeamCandidate(text="klonopin 1mg", log_prob=-0.40),
        ]
        res = engine2.rescorer.rescore_detailed(hyps)
        assert res.rescored_text == "klonopin 1mg"

    def test_engine_concurrent_adaptation_and_rescoring(self) -> None:
        """
        Stress test engine under concurrent multi-threaded workload:
        Multiple worker threads perform simultaneous adapt_confusion_matrix()
        and rescore_detailed() calls. Verifies thread safety and no deadlock.
        """
        engine = InferenceEngine(mode="mock", enable_rescorer=True, confusion_weight=8.0)
        assert engine.rescorer is not None

        test_pairs = [
            ("prednlsone", "prednisone"),
            ("Arnoxicillin", "Amoxicillin"),
            ("take po daily", "take 10 daily"),
            ("xlonopin", "klonopin"),
            ("cydindamycin", "clindamycin"),
        ]

        def adapt_worker(pair: Tuple[str, str]) -> int:
            res = engine.adapt_confusion_matrix(pair[0], pair[1], learning_rate=0.20, min_cost=0.15)
            return len(res)

        def rescore_worker(n: int) -> str:
            hyps = [
                BeamCandidate(text="xlonopin 1mg", log_prob=-0.10),
                BeamCandidate(text="klonopin 1mg", log_prob=-0.40),
            ]
            res = engine.rescorer.rescore_detailed(hyps)
            return res.rescored_text

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            adapt_futures = [executor.submit(adapt_worker, p) for p in test_pairs * 4]
            rescore_futures = [executor.submit(rescore_worker, i) for i in range(20)]

            adapt_results = [f.result() for f in concurrent.futures.as_completed(adapt_futures)]
            rescore_results = [f.result() for f in concurrent.futures.as_completed(rescore_futures)]

        assert len(adapt_results) == len(test_pairs) * 4
        assert len(rescore_results) == 20
        # Post-concurrency verification: matrix minimum cost is strictly >= 0.15
        assert min(engine.rescorer.confusion_matrix._matrix.values()) >= 0.15

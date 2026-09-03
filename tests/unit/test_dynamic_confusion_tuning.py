"""
tests/unit/test_dynamic_confusion_tuning.py
Comprehensive unit test suite for Milestone 2:
Online Self-Tuning Visual Confusion Matrix & Beam Rescorer Recalibration.

Validates:
1. Multi-gram Dynamic Programming (DP) character alignment extraction across
   1:1, 2:1, 1:2, and 2:2 operations.
2. Cost discounting with strict lower bound enforcement (min_cost >= 0.15) and
   bidirectional symmetry.
3. Clinical Look-Alike Sound-Alike (LASA) safety bounds preservation under
   repeated adversarial adaptations.
4. Immediate candidate rank flipping without model retraining:
   Erroneous OCR top beam outranks valid drug candidate initially; after
   operator correction adaptation, candidate ranking flips immediately.
5. Dynamic confusion matrix serialization and deserialization (export/load).
6. InferenceEngine singleton delegation and graceful fallback.
7. Thread-safe concurrent execution under multi-threaded load.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import tempfile
import pytest

from pipeline.rescorer.confusion_matrix import (
    AlignmentResult,
    AlignmentStep,
    VisualConfusionMatrix,
)
from pipeline.rescorer.beam_rescorer import (
    BeamCandidate,
    BeamRescorer,
    RescorerResult,
)
from pipeline.rescorer.trie import PrefixTrie
from backend.app.engine import InferenceEngine


# ===========================================================================
# 1. DP Alignment Extraction (1:1, 2:1, 1:2, 2:2)
# ===========================================================================

class TestDPAlignmentExtraction:
    """Test DP character alignment extraction across all supported operations."""

    @pytest.fixture
    def matrix(self) -> VisualConfusionMatrix:
        return VisualConfusionMatrix(load_defaults=True)

    def test_1_to_1_substitution_alignment(self, matrix: VisualConfusionMatrix) -> None:
        """Verify 1:1 character substitution extraction (e.g. 'l' -> 'i')."""
        updates = matrix.adapt_from_correction(
            original_prediction="prednlsone",
            operator_correction="prednisone",
            learning_rate=0.20,
            min_cost=0.15,
        )
        assert len(updates) >= 1
        sub = next((u for u in updates if u["source"] == "l" and u["target"] == "i"), None)
        assert sub is not None, f"Expected 1:1 substitution for l->i, got {updates}"
        assert sub["operation"] == "substitution"
        assert sub["previous_cost"] == 0.30
        assert sub["updated_cost"] == 0.24

    def test_2_to_1_contraction_alignment(self, matrix: VisualConfusionMatrix) -> None:
        """Verify 2:1 contraction extraction (e.g. 'rn' -> 'm' in 'Arnoxicillin' -> 'Amoxicillin')."""
        updates = matrix.adapt_from_correction(
            original_prediction="Arnoxicillin",
            operator_correction="Amoxicillin",
            learning_rate=0.20,
            min_cost=0.15,
        )
        assert len(updates) >= 1
        contraction = next((u for u in updates if u["source"] == "rn" and u["target"] == "m"), None)
        assert contraction is not None, f"Expected 2:1 contraction for rn->m, got {updates}"
        assert contraction["operation"] == "contraction"
        assert contraction["previous_cost"] == 0.25
        assert contraction["updated_cost"] == 0.20

    def test_1_to_2_expansion_alignment(self, matrix: VisualConfusionMatrix) -> None:
        """Verify 1:2 expansion extraction (e.g. 'm' -> 'rn' in 'Amoxicillin' -> 'Arnoxicillin')."""
        updates = matrix.adapt_from_correction(
            original_prediction="Amoxicillin",
            operator_correction="Arnoxicillin",
            learning_rate=0.20,
            min_cost=0.15,
        )
        assert len(updates) >= 1
        expansion = next((u for u in updates if u["source"] == "m" and u["target"] == "rn"), None)
        assert expansion is not None, f"Expected 1:2 expansion for m->rn, got {updates}"
        assert expansion["operation"] == "expansion"
        assert expansion["previous_cost"] == 0.25
        assert expansion["updated_cost"] == 0.20

    def test_2_to_2_substitution_alignment(self, matrix: VisualConfusionMatrix) -> None:
        """Verify 2:2 substitution extraction (e.g. 'po' -> '10')."""
        updates = matrix.adapt_from_correction(
            original_prediction="take po daily",
            operator_correction="take 10 daily",
            learning_rate=0.20,
            min_cost=0.15,
        )
        assert len(updates) >= 1
        sub_22 = next((u for u in updates if u["source"] == "po" and u["target"] == "10"), None)
        assert sub_22 is not None, f"Expected 2:2 substitution for po->10, got {updates}"
        assert sub_22["operation"] == "substitution_2_2"
        assert sub_22["previous_cost"] == 0.40
        assert sub_22["updated_cost"] == 0.32

    def test_identical_and_empty_strings(self, matrix: VisualConfusionMatrix) -> None:
        """Verify identical and empty inputs yield no confusion updates."""
        assert matrix.adapt_from_correction("Amoxicillin", "Amoxicillin") == []
        assert matrix.adapt_from_correction("", "") == []
        assert matrix.adapt_from_correction("   ", "   ") == []

    def test_case_insensitive_handling(self, matrix: VisualConfusionMatrix) -> None:
        """Verify case normalization preserves symmetric mapping."""
        updates = matrix.adapt_from_correction(
            original_prediction="PREDNISONE",
            operator_correction="prednisone",
        )
        assert updates == []  # Same string case-insensitively


# ===========================================================================
# 2. Cost Discounting, Symmetry, and Min Cost Bounds
# ===========================================================================

class TestCostDiscountingAndSymmetry:
    """Test cost discounting math, min_cost bound clamping, and symmetry."""

    @pytest.fixture
    def matrix(self) -> VisualConfusionMatrix:
        return VisualConfusionMatrix(load_defaults=True)

    def test_cost_discounting_calculation(self, matrix: VisualConfusionMatrix) -> None:
        """Verify mathematical formula c_new = max(min_cost, round(curr * (1 - lr), 4))."""
        # 'c' <-> 'e' initially costs 0.30
        assert matrix.get_cost("c", "e") == 0.30
        updates = matrix.adapt_from_correction("c", "e", learning_rate=0.20, min_cost=0.15)
        assert len(updates) == 1
        u = updates[0]
        assert u["source"] == "c"
        assert u["target"] == "e"
        assert u["previous_cost"] == 0.30
        assert u["updated_cost"] == 0.24
        assert u["old_cost"] == 0.30
        assert u["new_cost"] == 0.24
        assert matrix.get_cost("c", "e") == 0.24

    def test_symmetry_enforcement(self, matrix: VisualConfusionMatrix) -> None:
        """Verify get_cost(u, v) == get_cost(v, u) after adaptation."""
        matrix.adapt_from_correction("c", "e", learning_rate=0.20, min_cost=0.15)
        assert matrix.get_cost("c", "e") == 0.24
        assert matrix.get_cost("e", "c") == 0.24

        # Multi-gram symmetry
        matrix.adapt_from_correction("rn", "m", learning_rate=0.20, min_cost=0.15)
        assert matrix.get_cost("rn", "m") == 0.20
        assert matrix.get_cost("m", "rn") == 0.20

    def test_min_cost_lower_bound_clamp(self, matrix: VisualConfusionMatrix) -> None:
        """Verify cost does not drop below min_cost even with high learning rate."""
        # 'l' <-> '1' starts at 0.20
        assert matrix.get_cost("l", "1") == 0.20
        updates = matrix.adapt_from_correction("l", "1", learning_rate=0.80, min_cost=0.15)
        assert updates[0]["updated_cost"] == 0.15
        assert matrix.get_cost("l", "1") == 0.15

        # Further adaptation remains clamped at 0.15
        updates2 = matrix.adapt_from_correction("l", "1", learning_rate=0.50, min_cost=0.15)
        assert updates2[0]["updated_cost"] == 0.15
        assert matrix.get_cost("l", "1") == 0.15

    def test_strict_minimum_cost_clinical_floor(self, matrix: VisualConfusionMatrix) -> None:
        """Verify that passing min_cost < 0.15 is overridden to >= 0.15 for clinical safety."""
        updates = matrix.adapt_from_correction("l", "1", learning_rate=0.99, min_cost=0.0)
        assert updates[0]["updated_cost"] == 0.15
        assert matrix.get_cost("l", "1") >= 0.15

        updates_neg = matrix.adapt_from_correction("l", "1", learning_rate=0.99, min_cost=-1.0)
        assert updates_neg[0]["updated_cost"] == 0.15
        assert matrix.get_cost("l", "1") >= 0.15

    def test_unmapped_token_discounting(self, matrix: VisualConfusionMatrix) -> None:
        """Verify that an unmapped optical confusion pair discounts from default_substitution_cost."""
        # 'x' <-> 'k' is initially unmapped (cost 1.20)
        assert matrix.get_cost("x", "k") == 1.20
        updates = matrix.adapt_from_correction("x", "k", learning_rate=0.25, min_cost=0.15)
        assert updates[0]["previous_cost"] == 1.20
        assert updates[0]["updated_cost"] == 0.90
        assert matrix.get_cost("x", "k") == 0.90
        assert matrix.get_cost("k", "x") == 0.90

    def test_multiple_iterations_asymptotic_decay(self, matrix: VisualConfusionMatrix) -> None:
        """Verify repeated feedback calls decay costs smoothly toward 0.15."""
        costs = []
        for _ in range(10):
            res = matrix.adapt_from_correction("s", "5", learning_rate=0.20, min_cost=0.15)
            costs.append(res[0]["updated_cost"])

        # Monotonically non-increasing
        for i in range(len(costs) - 1):
            assert costs[i] >= costs[i + 1]
        assert costs[-1] == 0.15


# ===========================================================================
# 3. Clinical Look-Alike Sound-Alike (LASA) Safety Bounds
# ===========================================================================

class TestClinicalLASAPairsSafetyBounds:
    """Verify that critical LASA clinical pairs maintain bounds >= 0.15."""

    CRITICAL_LASA_PAIRS = [
        ("amoxicillin", "ampicillin"),
        ("hydroxyzine", "hydralazine"),
        ("celebrex", "celexa"),
        ("prednisone", "prednisolone"),
        ("adderall", "inderal"),
        ("zantac", "xanax"),
        ("clonidine", "klonopin"),
        ("metformin", "metronidazole"),
    ]

    def test_lasa_clinical_pairs_never_drop_below_min_cost(self) -> None:
        """Adversarially adapt all 8 LASA pairs 20 times and verify bounds."""
        cm = VisualConfusionMatrix(load_defaults=True)

        for name_a, name_b in self.CRITICAL_LASA_PAIRS:
            for _ in range(20):
                cm.adapt_from_correction(name_a, name_b, learning_rate=0.50, min_cost=0.15)
                cm.adapt_from_correction(name_b, name_a, learning_rate=0.50, min_cost=0.15)

            # Check that distance between the pair never drops to zero
            dist = cm.compute_distance(name_a, name_b)
            assert dist >= 0.15, f"LASA pair {name_a}/{name_b} distance collapsed to {dist}"

            # Check all pairwise matrix entries remain >= 0.15
            for (src, tgt), cost in cm._matrix.items():
                assert cost >= 0.15, f"Cost for ({src}, {tgt}) fell below 0.15: {cost}"

    def test_lasa_adversarial_zero_min_cost(self) -> None:
        """Even if caller requests min_cost=0.0, LASA pairs preserve safety bound."""
        cm = VisualConfusionMatrix(load_defaults=True)
        for _ in range(15):
            cm.adapt_from_correction("prednisone", "prednisolone", learning_rate=0.90, min_cost=0.0)

        assert cm.get_cost("s", "l") >= 0.15
        assert cm.compute_distance("prednisone", "prednisolone") >= 0.15


# ===========================================================================
# 4. Immediate Candidate Rank Flipping Without Retraining
# ===========================================================================

class TestImmediateCandidateRankFlipping:
    """
    Test live beam rescorer immediate candidate rank flip:
    An erroneous OCR beam initially scores higher than a valid candidate.
    After adapt_from_correction() recalibrates confusion costs, subsequent
    rescore_detailed() call on the SAME hypotheses flips rank to the valid candidate!
    """

    @pytest.fixture
    def rescorer(self) -> BeamRescorer:
        trie = PrefixTrie()
        vocab_path = Path("data/reference_handwriting/vocabularies")
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

    def test_rank_flip_unmapped_optical_confusion(self, rescorer: BeamRescorer) -> None:
        """
        Scenario:
        - Candidate 1: 'xlonopin 1mg' (erroneous OCR top beam, higher log_prob -0.10)
        - Candidate 2: 'klonopin 1mg' (valid pharmaceutical drug, lower log_prob -0.40)
        Initially, 'x' <-> 'k' is unmapped (cost 1.20).
        Initial Rescoring: 'xlonopin 1mg' wins (#1).
        After adapt_from_correction('xlonopin', 'klonopin'), cost drops from 1.20 to 0.18.
        Subsequent Rescoring: 'klonopin 1mg' wins (#1) with positive delta_score!
        """
        cand1 = BeamCandidate(text="xlonopin 1mg", log_prob=-0.10)
        cand2 = BeamCandidate(text="klonopin 1mg", log_prob=-0.40)
        hypotheses = [cand1, cand2]

        # 1. Initial rescoring prior to adaptation
        res1 = rescorer.rescore_detailed(hypotheses)
        assert res1.rescored_text == "xlonopin 1mg", (
            f"Expected initial top beam 'xlonopin 1mg', got '{res1.rescored_text}'"
        )
        assert res1.all_candidates[0]["text"] == "xlonopin 1mg"

        # 2. Operator submits correction for the misread glyph
        updates = rescorer.adapt_confusion_matrix(
            original_prediction="xlonopin",
            operator_correction="klonopin",
            learning_rate=0.85,
            min_cost=0.15,
        )
        assert len(updates) >= 1
        assert updates[0]["source"] == "x"
        assert updates[0]["target"] == "k"
        assert updates[0]["previous_cost"] == 1.20
        assert updates[0]["updated_cost"] <= 0.20

        # 3. Subsequent rescoring on the EXACT SAME candidates
        res2 = rescorer.rescore_detailed(hypotheses)
        assert res2.rescored_text == "klonopin 1mg", (
            f"Expected rank flip to 'klonopin 1mg', got '{res2.rescored_text}'"
        )
        assert res2.rescore_applied is True
        assert res2.delta_score > 0.0
        assert res2.matched_lexicon_term is not None

    def test_rank_flip_delta_score_and_confidence(self, rescorer: BeamRescorer) -> None:
        """Verify delta_score and confidence metrics reflect the rank flip."""
        cand1 = BeamCandidate(text="xlonopin 1mg", log_prob=-0.10)
        cand2 = BeamCandidate(text="klonopin 1mg", log_prob=-0.40)

        # Before adaptation
        res1 = rescorer.rescore_detailed([cand1, cand2])
        assert res1.rescored_text == "xlonopin 1mg"

        # Adapt
        rescorer.adapt_confusion_matrix("xlonopin", "klonopin", learning_rate=0.85, min_cost=0.15)

        # After adaptation
        res2 = rescorer.rescore_detailed([cand1, cand2])
        assert res2.rescored_text == "klonopin 1mg"
        assert res2.confidence > 0.50
        assert res2.delta_score > 0.0

    def test_rescore_convenience_method_flips_ranking(self, rescorer: BeamRescorer) -> None:
        """Verify rescorer.rescore() list of tuples immediately reflects rank flip."""
        hyps = [("xlonopin 1mg", -0.10), ("klonopin 1mg", -0.40)]
        ranked_before = rescorer.rescore(hyps)
        assert ranked_before[0][0] == "xlonopin 1mg"

        rescorer.adapt_confusion_matrix("xlonopin", "klonopin", learning_rate=0.85, min_cost=0.15)

        ranked_after = rescorer.rescore(hyps)
        assert ranked_after[0][0] == "klonopin 1mg"


# ===========================================================================
# 5. Dynamic State Persistence (Export / Load)
# ===========================================================================

class TestDynamicStatePersistence:
    """Test dynamic confusion state export and load routines."""

    def test_export_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Export dynamic state to JSON, reload into fresh matrix, and verify weights."""
        cm1 = VisualConfusionMatrix(load_defaults=True)
        cm1.adapt_from_correction("x", "k", learning_rate=0.50, min_cost=0.15)
        cm1.adapt_from_correction("Arnoxicillin", "Amoxicillin", learning_rate=0.40, min_cost=0.15)

        export_file = tmp_path / "subdir" / "dynamic_matrix.json"
        exported_path = cm1.export_dynamic_state(export_file)
        assert Path(exported_path).exists()

        # Read raw json to verify structure
        with open(exported_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == "1.0"
        assert data["total_dynamic_pairs"] >= 2
        assert len(data["dynamic_pairs"]) >= 2
        assert len(data["pairs"]) >= 2

        # Load into fresh matrix
        cm2 = VisualConfusionMatrix(load_defaults=True)
        count = cm2.load_dynamic_state(exported_path)
        assert count >= 2
        assert cm2.get_cost("x", "k") == cm1.get_cost("x", "k")
        assert cm2.get_cost("k", "x") == cm1.get_cost("k", "x")
        assert cm2.get_cost("rn", "m") == cm1.get_cost("rn", "m")

    def test_load_nonexistent_file_returns_zero(self, tmp_path: Path) -> None:
        """Loading from nonexistent path returns 0 and does not crash."""
        cm = VisualConfusionMatrix()
        count = cm.load_dynamic_state(tmp_path / "nonexistent.json")
        assert count == 0

    def test_auto_persist_flag_writes_file(self, tmp_path: Path) -> None:
        """Verify auto_persist=True with persist_filepath saves automatically."""
        target_path = tmp_path / "auto_persisted.json"
        cm = VisualConfusionMatrix(load_defaults=True)
        cm.adapt_from_correction(
            "prednlsone",
            "prednisone",
            learning_rate=0.20,
            auto_persist=True,
            persist_filepath=target_path,
        )
        assert target_path.exists()
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_dynamic_pairs"] >= 1


# ===========================================================================
# 6. InferenceEngine Singleton Integration
# ===========================================================================

class TestInferenceEngineIntegration:
    """Test InferenceEngine adapt_confusion_matrix delegation."""

    def test_engine_adapt_confusion_matrix_live_update(self) -> None:
        """Verify engine.adapt_confusion_matrix updates engine.rescorer live."""
        engine = InferenceEngine(mode="mock")
        if engine.rescorer is None or engine.rescorer.confusion_matrix is None:
            pytest.skip("Rescorer not initialized in engine")

        initial_cost = engine.rescorer.confusion_matrix.get_cost("c", "e")
        updates = engine.adapt_confusion_matrix(
            original_prediction="c",
            operator_correction="e",
            learning_rate=0.20,
            min_cost=0.15,
            auto_persist=False,
        )
        assert len(updates) == 1
        assert updates[0]["previous_cost"] == initial_cost
        assert updates[0]["updated_cost"] < initial_cost
        assert engine.rescorer.confusion_matrix.get_cost("c", "e") == updates[0]["updated_cost"]

    def test_engine_adapt_confusion_matrix_when_rescorer_disabled(self) -> None:
        """When rescorer is None, adapt_confusion_matrix returns empty list."""
        engine = InferenceEngine(mode="mock")
        engine.rescorer = None
        updates = engine.adapt_confusion_matrix("abc", "def")
        assert updates == []


# ===========================================================================
# 7. Thread-Safety and Concurrency
# ===========================================================================

class TestThreadSafetyAndConcurrency:
    """Test multi-threaded concurrent adaptations without race conditions."""

    def test_concurrent_adaptations_thread_safe(self) -> None:
        """Run concurrent adaptations across 10 threads."""
        cm = VisualConfusionMatrix(load_defaults=True)
        pairs_to_adapt = [
            ("prednlsone", "prednisone"),
            ("Arnoxicillin", "Amoxicillin"),
            ("Amoxicillin", "Arnoxicillin"),
            ("take po daily", "take 10 daily"),
            ("xlonopin", "klonopin"),
            ("cydindamycin", "clindamycin"),
            ("amoxicitlin", "amoxicillin"),
            ("hydroxyzine", "hydralazine"),
        ]

        def worker(pair: tuple[str, str]) -> int:
            res = cm.adapt_from_correction(pair[0], pair[1], learning_rate=0.10, min_cost=0.15)
            return len(res)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, p) for p in pairs_to_adapt * 5]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == len(pairs_to_adapt) * 5
        assert all(r >= 1 for r in results)

        # Verify all costs are within valid bounds
        for (src, tgt), cost in cm._matrix.items():
            assert cost >= 0.15

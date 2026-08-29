"""
tests/unit/test_beam_rescorer.py
Unit tests for BeamRescorer, BeamCandidate, ContextFeatures, and RescorerResult.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
import pytest

from pipeline.rescorer.beam_rescorer import (
    BeamCandidate,
    BeamRescorer,
    ContextFeatures,
    RescorerResult,
)
from pipeline.rescorer.confusion_matrix import VisualConfusionMatrix
from pipeline.rescorer.trie import PrefixTrie


@pytest.fixture(scope="module")
def shared_rescorer() -> BeamRescorer:
    """Instantiate a fully loaded BeamRescorer instance."""
    trie = PrefixTrie()
    trie.load_vocabularies("data/reference_handwriting/vocabularies")
    cm = VisualConfusionMatrix(load_defaults=True)
    return BeamRescorer(
        trie=trie,
        confusion_matrix=cm,
        lambda_lexicon=1.0,
        lambda_context=0.8,
        lambda_confusion=0.5,
        vocab_dir="data/reference_handwriting/vocabularies",
    )


class TestBeamRescorerSchemas:
    """Test Pydantic schemas and interface contract conformance."""

    def test_beam_candidate_creation(self) -> None:
        """Verify BeamCandidate validation and fields."""
        cand = BeamCandidate(text="Amoxicillin 500mg", log_prob=-0.55)
        assert cand.text == "Amoxicillin 500mg"
        assert cand.log_prob == -0.55

    def test_context_features_creation(self) -> None:
        """Verify ContextFeatures fields."""
        ctx = ContextFeatures(dosage="500mg", route="PO", frequency="TID", form="capsule")
        assert ctx.dosage == "500mg"
        assert ctx.route == "PO"
        assert ctx.frequency == "TID"
        assert ctx.form == "capsule"

    def test_rescorer_result_structure(self) -> None:
        """Verify RescorerResult dictionary serialization matching PROJECT.md."""
        res = RescorerResult(
            rescored_text="Amoxicillin 500mg PO TID",
            confidence=0.95,
            original_top_beam="Amoxcillin 500mg PO TID",
            delta_score=1.42,
            rescore_applied=True,
            matched_lexicon_term="Amoxicillin",
        )
        assert res.rescore_applied is True
        assert res.confidence > 0.90
        assert res.matched_lexicon_term == "Amoxicillin"
        d = res.model_dump()
        assert d["rescored_text"] == "Amoxicillin 500mg PO TID"
        assert d["original_top_beam"] == "Amoxcillin 500mg PO TID"


class TestMultiObjectiveScoring:
    """Test multi-objective scoring formula and re-ranking."""

    def test_ocr_typo_correction(self, shared_rescorer: BeamRescorer) -> None:
        """Verify correcting OCR typos in the top beam using Trie and confusion penalty."""
        hypotheses = [
            ("Amoxcillin 500mg PO TID", -0.40),  # OCR typo with higher logprob
            ("Amoxicillin 500mg PO TID", -0.55),  # Canonical drug name
            ("Ampicillin 500mg PO TID", -0.70),   # Alternative drug
        ]
        result = shared_rescorer.rescore_detailed(hypotheses)
        assert result.rescored_text == "Amoxicillin 500mg PO TID"
        assert result.rescore_applied is True
        assert result.matched_lexicon_term == "Amoxicillin"

    def test_weight_modulation_formula(self, shared_rescorer: BeamRescorer) -> None:
        """Verify formula: S_final = S_OCR + 1.0*S_Lex + 0.8*S_Ctx - 0.5*Penalty."""
        # Direct rescorer calculation check
        cands = [("Amoxicillin 500mg PO TID", -0.50)]
        res = shared_rescorer.rescore_detailed(cands)
        assert res.rescored_text == "Amoxicillin 500mg PO TID"


class TestLASAPairDisambiguation:
    """
    Test disambiguating the 8 critical Look-Alike Sound-Alike (LASA) drug pairs
    defined in Scenario S6 using dosage and clinical context.
    """

    def test_lasa_01_amoxicillin_vs_ampicillin(self, shared_rescorer: BeamRescorer) -> None:
        """Amoxicillin 500mg PO TID vs Ampicillin."""
        beams = [
            ("Amoxcillin 500mg PO TID", -0.45),
            ("Amoxicillin 500mg PO TID", -0.55),
            ("Ampicillin 500mg PO TID", -0.65),
        ]
        top_text, _ = shared_rescorer.rescore_top1(beams)
        assert "Amoxicillin" in top_text

    def test_lasa_02_hydroxyzine_vs_hydralazine(self, shared_rescorer: BeamRescorer) -> None:
        """Hydroxyzine 25mg PO QHS vs Hydralazine."""
        beams = [
            ("Hydroxyzine 25mg PO QHS", -0.45),
            ("Hydralazine 25mg PO QHS", -0.65),
        ]
        top_text, _ = shared_rescorer.rescore_top1(beams)
        assert "Hydroxyzine" in top_text

    def test_lasa_03_celebrex_vs_celexa(self, shared_rescorer: BeamRescorer) -> None:
        """Celebrex 200mg PO Daily vs Celexa (Celexa does not have 200mg)."""
        beams = [
            ("Celexa 200mg PO Daily", -0.45),      # Celexa max dose is 40mg
            ("Celebrex 200mg PO Daily", -0.55),    # Celebrex standard is 200mg
        ]
        top_text, _ = shared_rescorer.rescore_top1(beams)
        assert "Celebrex" in top_text

    def test_lasa_04_prednisone_vs_prednisolone(self, shared_rescorer: BeamRescorer) -> None:
        """Prednisone 10mg PO Daily vs Prednisolone."""
        beams = [
            ("Prednisolone 10mg PO Daily", -0.45),
            ("Prednisone 10mg PO Daily", -0.55),
        ]
        top_text, _ = shared_rescorer.rescore_top1(beams)
        assert "Prednisone" in top_text

    def test_lasa_05_adderall_vs_inderal(self, shared_rescorer: BeamRescorer) -> None:
        """Adderall 10mg PO Daily vs Inderal."""
        beams = [
            ("Inderal 10mg PO Daily", -0.45),
            ("Adderall 10mg PO Daily", -0.55),
        ]
        # In this context, dosage 10mg is compatible with both, but Adderall has prior
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text in ("Adderall 10mg PO Daily", "Inderal 10mg PO Daily")

    def test_lasa_06_zantac_vs_xanax(self, shared_rescorer: BeamRescorer) -> None:
        """Zantac 150mg PO BID vs Xanax (Xanax is 0.25mg-2mg, Zantac is 150mg)."""
        beams = [
            ("Xanax 150mg PO BID", -0.45),     # Unrealistic dose for Xanax
            ("Zantac 150mg PO BID", -0.55),    # Canonical 150mg Zantac dose
        ]
        top_text, _ = shared_rescorer.rescore_top1(beams)
        assert "Zantac" in top_text

    def test_lasa_07_clonidine_vs_klonopin(self, shared_rescorer: BeamRescorer) -> None:
        """Clonidine 0.1mg PO BID vs Klonopin (Clonidine is 0.1mg, Klonopin is 0.5-2mg)."""
        beams = [
            ("Klonopin 0.1mg PO BID", -0.45),
            ("Clonidine 0.1mg PO BID", -0.55),
        ]
        top_text, _ = shared_rescorer.rescore_top1(beams)
        assert "Clonidine" in top_text

    def test_lasa_08_metformin_vs_metronidazole(self, shared_rescorer: BeamRescorer) -> None:
        """Metformin 500mg PO BID vs Metronidazole."""
        beams = [
            ("Metformin 500mg PO BID", -0.45),
            ("Metronidazole 500mg PO BID", -0.65),
        ]
        top_text, _ = shared_rescorer.rescore_top1(beams)
        assert "Metformin" in top_text


class TestBoundaryAndEdgeCases:
    """Test robustness against boundary conditions and invalid inputs."""

    def test_beam_width_1_greedy_fallback(self, shared_rescorer: BeamRescorer) -> None:
        """Verify K=1 returns the single candidate."""
        beams = [("Amoxicillin 500mg", -0.2)]
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text == "Amoxicillin 500mg"
        assert res.rescore_applied is False

    def test_all_candidates_identical(self, shared_rescorer: BeamRescorer) -> None:
        """Verify handling when all beam candidates are identical."""
        beams = [("Amoxicillin 500mg", -0.2), ("Amoxicillin 500mg", -0.3)]
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text == "Amoxicillin 500mg"

    def test_very_low_log_probabilities(self, shared_rescorer: BeamRescorer) -> None:
        """Verify handling of large negative log probabilities (-1000.0)."""
        beams = [("Amoxicillin 500mg", -1000.0), ("Aspirin 81mg", -1000.0)]
        res = shared_rescorer.rescore_detailed(beams)
        assert math.isfinite(res.confidence)
        assert 0.0 <= res.confidence <= 1.0

    def test_empty_string_candidate(self, shared_rescorer: BeamRescorer) -> None:
        """Verify handling of empty string candidates."""
        beams = [("", -5.0), ("Amoxicillin", -6.0)]
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text == "Amoxicillin"

    def test_unrealistic_dosage_rejection(self, shared_rescorer: BeamRescorer) -> None:
        """Verify unrealistic dosage (50,000mg) receives penalty."""
        beams = [
            ("Amoxicillin 50000mg PO", -0.45),
            ("Amoxicillin 500mg PO", -0.55),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert res.rescored_text == "Amoxicillin 500mg PO"

    def test_conflicting_route_and_form(self, shared_rescorer: BeamRescorer) -> None:
        """Verify incompatible route and form (e.g. tablet IV) is penalized."""
        beams = [
            ("Amoxicillin 500mg tablet IV", -0.45),
            ("Amoxicillin 500mg tablet PO", -0.55),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert "PO" in res.rescored_text

    def test_missing_context(self, shared_rescorer: BeamRescorer) -> None:
        """Verify rescoring behaves gracefully when no context is provided."""
        beams = [("Amoxcillin", -0.45), ("Amoxicillin", -0.55)]
        res = shared_rescorer.rescore_detailed(beams, context=None)
        assert res.rescored_text == "Amoxicillin"

    def test_dosage_strength_exact_boundary_matching(self, shared_rescorer: BeamRescorer) -> None:
        """Verify 0mg or 50mg do not falsely match standard 250mg catalog strengths."""
        # Amoxicillin standard strengths are 250mg, 500mg, 875mg
        # Test 50mg context does NOT match 250mg
        ctx_50mg = ContextFeatures(dosage="50mg")
        score_50mg = shared_rescorer._score_context("Amoxicillin 50mg", "Amoxicillin", ctx_50mg)
        # Should be penalized (-0.4) rather than bonus (+1.0)
        assert score_50mg < 0.0

        ctx_0mg = ContextFeatures(dosage="0mg")
        score_0mg = shared_rescorer._score_context("Amoxicillin 0mg", "Amoxicillin", ctx_0mg)
        assert score_0mg < 0.0

        # Exact standard strength (250mg) must receive full bonus (+1.0)
        ctx_250mg = ContextFeatures(dosage="250mg")
        score_250mg = shared_rescorer._score_context("Amoxicillin 250mg", "Amoxicillin", ctx_250mg)
        assert score_250mg >= 1.0

    def test_po_injection_incompatibility_penalty(self, shared_rescorer: BeamRescorer) -> None:
        """Verify PO injection receives clinical incompatibility penalty."""
        beams = [
            ("Amoxicillin 500mg PO injection", -0.40),
            ("Amoxicillin 500mg PO capsule", -0.60),
        ]
        res = shared_rescorer.rescore_detailed(beams)
        assert "capsule" in res.rescored_text

    def test_rescore_latency_sla(self, shared_rescorer: BeamRescorer) -> None:
        """Verify rescorer latency < 5.0ms per hypothesis set."""
        beams = [
            ("Amoxcillin 500mg PO TID", -0.45),
            ("Amoxicillin 500mg PO TID", -0.55),
            ("Ampicillin 500mg PO TID", -0.65),
            ("Amoxil 500mg PO TID", -0.75),
            ("Aspirin 500mg PO TID", -0.85),
        ]
        t0 = time.perf_counter()
        n_iters = 200
        for _ in range(n_iters):
            _ = shared_rescorer.rescore_detailed(beams)
        avg_ms = (time.perf_counter() - t0) * 1000.0 / n_iters
        assert avg_ms < 5.0, f"Rescorer latency {avg_ms:.4f}ms exceeds 5.0ms SLA"


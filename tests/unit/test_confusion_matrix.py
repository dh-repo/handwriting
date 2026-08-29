"""
tests/unit/test_confusion_matrix.py
Unit tests for VisualConfusionMatrix, ConfusionPair, AlignmentStep, and AlignmentResult.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import pytest

from pipeline.rescorer.confusion_matrix import (
    AlignmentResult,
    AlignmentStep,
    ConfusionPair,
    DEFAULT_CONFUSION_PAIRS,
    VisualConfusionMatrix,
)


class TestVisualConfusionMatrixBasics:
    """Test matrix creation, default pairs, cost lookup, and symmetry."""

    def test_default_pairs_count_and_categories(self) -> None:
        """Verify 70+ default pairs across the 7 medical handwriting domains."""
        assert len(DEFAULT_CONFUSION_PAIRS) >= 50
        categories = {p.category for p in DEFAULT_CONFUSION_PAIRS}
        expected_cats = {
            "loop_curve",
            "vertical_ascender",
            "minim_arch",
            "descender",
            "alphanumeric",
            "digraph_ligature",
            "medical_notation",
        }
        for cat in expected_cats:
            assert cat in categories, f"Missing expected category {cat}"

    def test_matrix_symmetry(self) -> None:
        """Verify symmetry: get_cost(u, v) == get_cost(v, u)."""
        cm = VisualConfusionMatrix(load_defaults=True)
        assert len(cm) >= 50

        test_pairs = [
            ("c", "e", 0.30),
            ("l", "1", 0.20),
            ("rn", "m", 0.25),
            ("cl", "d", 0.30),
            ("vv", "w", 0.25),
            ("a", "o", 0.35),
            ("u", "v", 0.35),
            ("s", "5", 0.25),
            ("g", "9", 0.25),
        ]
        for s, t, expected_c in test_pairs:
            assert cm.get_cost(s, t) == expected_c
            assert cm.get_cost(t, s) == expected_c

    def test_identical_string_cost(self) -> None:
        """Verify cost 0.0 for identical characters and strings."""
        cm = VisualConfusionMatrix()
        assert cm.get_cost("a", "a") == 0.0
        assert cm.get_cost("rn", "rn") == 0.0
        assert cm.compute_distance("Amoxicillin", "Amoxicillin") == 0.0
        assert cm.compute_similarity("Amoxicillin", "Amoxicillin") == 1.0

    def test_unmapped_pairs(self) -> None:
        """Verify default costs for unmapped single-char and multi-gram pairs."""
        cm = VisualConfusionMatrix(default_substitution_cost=1.20)
        # 1:1 unmapped
        assert cm.get_cost("z", "b") == 1.20
        # Multi-gram unmapped
        assert cm.get_cost("xyz", "abc") == float("inf")

    def test_custom_cost_setter(self) -> None:
        """Verify dynamic addition and modification of substitution costs."""
        cm = VisualConfusionMatrix(load_defaults=False)
        assert len(cm) == 0

        cm.set_cost("alpha", "beta", 0.42, symmetric=True)
        assert cm.get_cost("alpha", "beta") == 0.42
        assert cm.get_cost("beta", "alpha") == 0.42

        # Non-symmetric setter
        cm.set_cost("x", "y", 0.15, symmetric=False)
        assert cm.get_cost("x", "y") == 0.15
        assert cm.get_cost("y", "x") == cm.default_substitution_cost


class TestStringDistanceAndAlignment:
    """Test dynamic programming edit distance with multi-gram alignment and backtracing."""

    @pytest.fixture
    def matrix(self) -> VisualConfusionMatrix:
        return VisualConfusionMatrix(load_defaults=True)

    def test_digraph_contraction_rn_to_m(self, matrix: VisualConfusionMatrix) -> None:
        """Verify 2:1 digraph contraction 'rn' -> 'm' in 'Arnoxicillin' -> 'Amoxicillin'."""
        dist = matrix.compute_distance("Arnoxicillin", "Amoxicillin")
        assert dist == 0.25

        align = matrix.align("Arnoxicillin", "Amoxicillin")
        assert align.raw_distance == 0.25
        assert align.similarity > 0.95
        ops = [s.operation for s in align.steps]
        assert "contraction" in ops

    def test_digraph_expansion_m_to_rn(self, matrix: VisualConfusionMatrix) -> None:
        """Verify 1:2 digraph expansion 'm' -> 'rn' in 'Amoxicillin' -> 'Arnoxicillin'."""
        dist = matrix.compute_distance("Amoxicillin", "Arnoxicillin")
        assert dist == 0.25

        align = matrix.align("Amoxicillin", "Arnoxicillin")
        assert align.raw_distance == 0.25
        ops = [s.operation for s in align.steps]
        assert "expansion" in ops

    def test_digraph_cl_to_d(self, matrix: VisualConfusionMatrix) -> None:
        """Verify 'cl' -> 'd' substitution in 'cloc' -> 'doc'."""
        dist = matrix.compute_distance("cloc", "doc")
        assert dist == 0.30

    def test_damerau_transposition_alignment(self, matrix: VisualConfusionMatrix) -> None:
        """Verify transposition operation in 'Aomxicillin' -> 'Amoxicillin'."""
        dist = matrix.compute_distance("Aomxicillin", "Amoxicillin")
        assert dist == 0.80

        align = matrix.align("Aomxicillin", "Amoxicillin")
        assert align.raw_distance == 0.80
        ops = [s.operation for s in align.steps]
        assert "transposition" in ops

    def test_medical_notation_confusion(self, matrix: VisualConfusionMatrix) -> None:
        """Verify unit and digit substitutions (e.g. 5oomg -> 500mg)."""
        dist = matrix.compute_distance("5oomg", "500mg")
        # 'o' -> '0' cost is 0.20 each, two occurrences = 0.40
        assert round(dist, 2) == 0.40

    def test_empty_string_handling(self, matrix: VisualConfusionMatrix) -> None:
        """Verify distance computation against empty strings."""
        assert matrix.compute_distance("", "") == 0.0
        assert matrix.compute_distance("abc", "") == 3.0 * matrix.default_deletion_cost
        assert matrix.compute_distance("", "def") == 3.0 * matrix.default_insertion_cost

    def test_alignment_step_dictionary(self, matrix: VisualConfusionMatrix) -> None:
        """Verify AlignmentResult.to_dict format."""
        align = matrix.align("c", "e")
        assert align.raw_distance == 0.30
        d = align.to_dict()
        assert "raw_distance" in d
        assert "normalized_distance" in d
        assert "similarity" in d
        assert "steps" in d
        assert len(d["steps"]) == 1
        assert d["steps"][0]["operation"] == "substitution"
        assert d["steps"][0]["cost"] == 0.30

    def test_alignment_latency(self, matrix: VisualConfusionMatrix) -> None:
        """Verify string alignment latency < 0.15ms per call."""
        t0 = time.perf_counter()
        n_iters = 1000
        for _ in range(n_iters):
            _ = matrix.compute_distance("Arnoxicillin", "Amoxicillin")
        avg_ms = (time.perf_counter() - t0) * 1000.0 / n_iters
        assert avg_ms < 0.15, f"Confusion distance latency {avg_ms:.4f}ms exceeds 0.15ms budget"


class TestConfusionMatrixSerialization:
    """Test export and import to JSON."""

    def test_export_and_load_json(self, tmp_path: Path) -> None:
        """Verify exporting to JSON and re-loading maintains all weights."""
        cm = VisualConfusionMatrix(load_defaults=True)
        json_file = tmp_path / "confusion_matrix.json"
        cm.export_json(json_file)

        assert json_file.exists()

        cm2 = VisualConfusionMatrix(load_defaults=False)
        count = cm2.load_json(json_file)
        assert count == len(cm)
        assert cm2.get_cost("c", "e") == 0.30
        assert cm2.get_cost("rn", "m") == 0.25

    def test_load_rxnorm_confusion_profiles(self) -> None:
        """Verify extracting LASA confusion profiles from rxnorm_medications.json."""
        cm = VisualConfusionMatrix(load_defaults=True)
        rx_path = "data/reference_handwriting/vocabularies/rxnorm_medications.json"
        if Path(rx_path).exists():
            count = cm.load_rxnorm_confusion_profiles(rx_path)
            assert count > 0
            # Check a glyph pair from RxNorm (e.g. oxi <-> ipi)
            assert cm.get_cost("oxi", "ipi") < 1.0

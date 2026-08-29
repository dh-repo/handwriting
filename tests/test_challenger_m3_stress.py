"""
tests/test_challenger_m3_stress.py
Empirical Adversarial Stress Test Suite for Milestone 3 (Trie & Visual Confusion Matrix).

Comprehensive stress-testing across:
1. Adversarial & Pathological Inputs (empty, single-char, >100 chars, unicode, emojis, control chars, symbols).
2. Edit distance pruning & scaling (D=0, 1, 2, 3, 4) on PrefixTrie.
3. Damerau-Levenshtein corner cases (transposition at start, end, duplicate adjacent chars, overlapping transpositions).
4. Long repeating digraph sequences (e.g., 'rnrnrn', 'clclcl', 'vvvvvv', 'mgmgmg', 'oxioxioxi').
5. Symmetry & Mathematical Consistency of VisualConfusionMatrix:
   - Metric Symmetry: dist(s1, s2) == dist(s2, s1)
   - Backtrace Consistency: sum(step.cost for step in align(s1, s2).steps) == raw_distance
   - Step Application: applying alignment steps to s1 transforms it exactly into s2
   - Fast DP vs Full DP equivalence: _compute_distance_fast(s1, s2) == _align_internal(s1, s2)[0]
6. High-Volume QPS & SLA Benchmarking (10,000 exact lookups, prefix searches, D=2 fuzzy searches, confusion distances).
"""

from __future__ import annotations

import math
import random
import string
import time
from pathlib import Path
from typing import List, Tuple
import pytest

from pipeline.rescorer.confusion_matrix import (
    AlignmentResult,
    DEFAULT_CONFUSION_PAIRS,
    VisualConfusionMatrix,
)
from pipeline.rescorer.trie import FuzzyMatch, PrefixTrie, TrieMatch
from pipeline.rescorer.beam_rescorer import BeamCandidate, BeamRescorer, RescorerResult


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
def loaded_rescorer(loaded_trie: PrefixTrie, loaded_matrix: VisualConfusionMatrix) -> BeamRescorer:
    return BeamRescorer(
        trie=loaded_trie,
        confusion_matrix=loaded_matrix,
        vocab_dir="data/reference_handwriting/vocabularies",
    )


# ===========================================================================
# 1. Adversarial & Pathological Inputs
# ===========================================================================

class TestAdversarialInputs:
    """Stress test PrefixTrie, VisualConfusionMatrix, and BeamRescorer with hostile/edge inputs."""

    @pytest.mark.parametrize("bad_query", [
        "",
        " ",
        "   \t\n   ",
        "a",
        "Z",
        "!",
        "???///&&&$$$***",
        "💊💉📋🩺🏥",
        "وصفة_طبية_باللغة_العربية",
        "處方簽_手寫測試",
        "Café_naïve_résumé",
        "\x00\x01\x02\x1f\x7f",
        "A" * 150,
        "a-b-c-d-e-f-g-h-i-j",
        "1234567890" * 15,
        "Amoxicillin" + ("\u200b" * 10) + "500mg",  # Zero-width spaces
        "Robert'); DROP TABLE Medications;--",     # SQL injection string
        "<script>alert('xss')</script>",           # XSS payload
    ])
    def test_trie_adversarial_queries(self, loaded_trie: PrefixTrie, bad_query: str):
        """Verify Trie methods never throw unhandled exceptions or crash on adversarial strings."""
        # 1. Exact search
        res_exact = loaded_trie.search_exact(bad_query)
        assert res_exact is None or isinstance(res_exact, TrieMatch)

        # 2. Prefix search
        res_prefix = loaded_trie.search_prefix(bad_query)
        assert isinstance(res_prefix, list)

        # 3. Fuzzy search D=0, 1, 2, 3
        for d in (0, 1, 2, 3):
            res_fuzzy = loaded_trie.fuzzy_search(bad_query, max_distance=d)
            assert isinstance(res_fuzzy, list)
            for m in res_fuzzy:
                assert isinstance(m, FuzzyMatch)
                assert m.distance <= d
                assert 0.0 <= m.similarity <= 1.0

    @pytest.mark.parametrize("bad_str", [
        "",
        " ",
        "a",
        "!",
        "💊💉📋",
        "A" * 200,
        "\x00\x00\x00",
        "~`!@#$%^&*()_+=-{}[]|\\:;\"'<>,.?/",
    ])
    def test_confusion_matrix_adversarial_strings(self, loaded_matrix: VisualConfusionMatrix, bad_str: str):
        """Verify VisualConfusionMatrix handles hostile strings without exception or NaN."""
        dist = loaded_matrix.compute_distance(bad_str, "Amoxicillin")
        assert math.isfinite(dist)
        assert dist >= 0.0

        align = loaded_matrix.align(bad_str, "Amoxicillin")
        assert math.isfinite(align.raw_distance)
        assert math.isfinite(align.normalized_distance)
        assert 0.0 <= align.similarity <= 1.0

        # Self distance
        self_dist = loaded_matrix.compute_distance(bad_str, bad_str)
        assert self_dist == 0.0


# ===========================================================================
# 2. Trie Edit Distance Scaling & Pruning Verification (D=0, 1, 2, 3, 4)
# ===========================================================================

class TestTrieDistanceScalingAndPruning:
    """Verify branch pruning behavior and monotonicity across edit distances D=0..4."""

    @pytest.mark.parametrize("target_word, corruptions", [
        ("Amoxicillin", [
            ("Amoxicillin", 0),
            ("Amoxcillin", 1),
            ("Amoxcilin", 2),
            ("Amocilin", 3),
            ("Amoclin", 4),
        ]),
        ("Hydroxyzine", [
            ("Hydroxyzine", 0),
            ("Hydroxizine", 1),
            ("Hidroxizine", 2),
            ("Hidroxizin", 3),
            ("Hidroxisin", 4),
        ]),
        ("Prednisone", [
            ("Prednisone", 0),
            ("Prednisoe", 1),
            ("Prednsoe", 2),
            ("Prdnsoe", 3),
            ("Pdnsoe", 4),
        ]),
    ])
    def test_distance_scaling_and_monotonicity(
        self,
        loaded_trie: PrefixTrie,
        target_word: str,
        corruptions: List[Tuple[str, int]],
    ):
        """Verify fuzzy search finds target word at exact expected distance and respects D thresholds."""
        for query_str, expected_d in corruptions:
            for max_d in range(0, 5):
                matches = loaded_trie.fuzzy_search(query_str, max_distance=max_d, length_adaptive=False)
                words = [m.word for m in matches]
                if expected_d <= max_d:
                    assert target_word in words, (
                        f"Expected '{target_word}' in matches for query '{query_str}' at max_d={max_d}, "
                        f"expected_d={expected_d}, found {words}"
                    )
                    # Verify recorded distance
                    match_obj = next(m for m in matches if m.word == target_word)
                    assert match_obj.distance == expected_d
                else:
                    # Target word must NOT appear when expected_d > max_d
                    assert target_word not in words, (
                        f"Target '{target_word}' appeared in matches for query '{query_str}' at max_d={max_d} "
                        f"even though expected_d={expected_d}"
                    )

    def test_pruning_performance_deep_trie(self, loaded_trie: PrefixTrie):
        """Verify pruning stops tree traversal rapidly even for large max_distance queries."""
        # Non-matching string of length 20
        query = "Z" * 20
        t0 = time.perf_counter()
        matches = loaded_trie.fuzzy_search(query, max_distance=2)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert len(matches) == 0
        assert elapsed_ms < 2.0, f"Pruned search took {elapsed_ms:.3f}ms (expected <2ms)"


# ===========================================================================
# 3. Damerau-Levenshtein Transposition Corner Cases
# ===========================================================================

class TestDamerauLevenshteinCornerCases:
    """Stress test adjacent transposition logic in Trie and ConfusionMatrix."""

    def test_transposition_at_start_of_word(self, loaded_trie: PrefixTrie, loaded_matrix: VisualConfusionMatrix):
        """Transposition at index 0-1: 'mAoxicillin' -> 'Amoxicillin'."""
        matches = loaded_trie.fuzzy_search("mAoxicillin", max_distance=1, enable_damerau=True, length_adaptive=False)
        assert "Amoxicillin" in [m.word for m in matches]
        m = next(x for x in matches if x.word == "Amoxicillin")
        assert m.distance == 1

        dist = loaded_matrix.compute_distance("mAoxicillin", "Amoxicillin")
        assert round(dist, 4) == round(loaded_matrix.default_transposition_cost, 4)

    def test_transposition_at_end_of_word(self, loaded_trie: PrefixTrie, loaded_matrix: VisualConfusionMatrix):
        """Transposition at last two chars: 'Amoxicillni' -> 'Amoxicillin'."""
        matches = loaded_trie.fuzzy_search("Amoxicillni", max_distance=1, enable_damerau=True, length_adaptive=False)
        assert "Amoxicillin" in [m.word for m in matches]
        m = next(x for x in matches if x.word == "Amoxicillin")
        assert m.distance == 1

        dist = loaded_matrix.compute_distance("Amoxicillni", "Amoxicillin")
        assert round(dist, 4) == round(loaded_matrix.default_transposition_cost, 4)

    def test_adjacent_duplicate_characters(self, loaded_trie: PrefixTrie, loaded_matrix: VisualConfusionMatrix):
        """Transposing identical adjacent characters ('ll' -> 'll') is a cost-free match (distance 0)."""
        dist = loaded_matrix.compute_distance("Amoxicillin", "Amoxicillin")
        assert dist == 0.0

    def test_multiple_transpositions(self, loaded_matrix: VisualConfusionMatrix):
        """Multiple non-overlapping transpositions: 'mAoxcillni' -> 'Amoxcillin' (2 transpositions: mA->Am, ni->in)."""
        dist = loaded_matrix.compute_distance("mAoxcillni", "Amoxcillin")
        expected_cost = 2 * loaded_matrix.default_transposition_cost
        assert round(dist, 4) == round(expected_cost, 4)


# ===========================================================================
# 4. Long Digraph Sequences & Multi-Gram Stress in Confusion Matrix
# ===========================================================================

class TestDigraphSequencesAndMultiGramStress:
    """Stress test 2:1, 1:2, 2:2 multi-gram lookups under extreme repeating patterns."""

    def test_repeating_rn_to_m_chain(self, loaded_matrix: VisualConfusionMatrix):
        """Chains of 'rn' <-> 'm': 'rnrnrnrnrn' (10 chars) <-> 'mmmmm' (5 chars)."""
        s_rn = "rnrnrnrnrn"  # 5 * 'rn'
        s_m = "mmmmm"        # 5 * 'm'

        dist = loaded_matrix.compute_distance(s_rn, s_m)
        # Each 'rn' -> 'm' costs 0.25, total 5 * 0.25 = 1.25
        assert round(dist, 4) == 1.25

        align = loaded_matrix.align(s_rn, s_m)
        assert round(align.raw_distance, 4) == 1.25
        # All steps should be contraction
        ops = [step.operation for step in align.steps]
        assert all(op == "contraction" for op in ops)
        assert len(ops) == 5

    def test_repeating_cl_to_d_chain(self, loaded_matrix: VisualConfusionMatrix):
        """Chains of 'cl' <-> 'd': 'clclclcl' (8 chars) <-> 'dddd' (4 chars)."""
        s_cl = "clclclcl"
        s_d = "dddd"

        dist = loaded_matrix.compute_distance(s_cl, s_d)
        # 'cl' -> 'd' costs 0.30, total 4 * 0.30 = 1.20
        assert round(dist, 4) == 1.20

        align = loaded_matrix.align(s_cl, s_d)
        assert round(align.raw_distance, 4) == 1.20

    def test_repeating_vv_to_w_chain(self, loaded_matrix: VisualConfusionMatrix):
        """Chains of 'vv' <-> 'w': 'vvvvvv' (6 chars) <-> 'www' (3 chars)."""
        s_vv = "vvvvvv"
        s_w = "www"

        dist = loaded_matrix.compute_distance(s_vv, s_w)
        # 'vv' -> 'w' costs 0.25, total 3 * 0.25 = 0.75
        assert round(dist, 4) == 0.75

    def test_alternating_mixed_multigram_sentence(self, loaded_matrix: VisualConfusionMatrix):
        """Complex mixture of 1:1, 2:1, 1:2, and transpositions."""
        s1 = "Arnoxclln 5oomg po tid"
        s2 = "Amoxicillin 500mg PO TID"
        dist = loaded_matrix.compute_distance(s1, s2)
        assert math.isfinite(dist)
        # Expected distance is exactly 3.65 (rn->m: 0.25, 3 ins of 'i': 3.0, 2 o->0: 0.40)
        assert round(dist, 2) == 3.65

        align = loaded_matrix.align(s1, s2)
        assert round(align.raw_distance, 4) == round(dist, 4)


# ===========================================================================
# 5. Mathematical Consistency & Metric Axioms Verification
# ===========================================================================

class TestMathematicalConsistency:
    """Verify metric axioms and DP backtrace consistency across thousands of pairs."""

    def test_fast_dp_vs_full_dp_exact_equivalence(self, loaded_matrix: VisualConfusionMatrix):
        """Verify _compute_distance_fast() == _align_internal()[0] on 200 diverse strings."""
        test_strings = [
            "", "a", "rn", "m", "cl", "d", "vv", "w", "Amoxicillin", "Arnoxicillin",
            "Hydroxyzine", "Hydralazine", "500mg", "5oomg", "PO", "TID", "1000x",
            "Metformin", "Metronidazole", "Prednisone", "Prednisolone",
            "Supercalifragilisticexpialidocious", "quick brown fox jumps over lazy dog",
            "rnrnrn", "clclcl", "vvvvvv", "mgmgmg", "oxioxioxi",
        ]

        for s1 in test_strings:
            for s2 in test_strings:
                fast_d = loaded_matrix.compute_distance(s1, s2)
                align_res = loaded_matrix.align(s1, s2)
                assert round(fast_d, 6) == round(align_res.raw_distance, 6), (
                    f"Discrepancy between fast DP ({fast_d}) and align DP ({align_res.raw_distance}) "
                    f"for s1='{s1}', s2='{s2}'"
                )

    def test_metric_symmetry_across_pairs(self, loaded_matrix: VisualConfusionMatrix):
        """Verify symmetry axiom: dist(s1, s2) == dist(s2, s1) for all configured confusion pairs."""
        for pair in DEFAULT_CONFUSION_PAIRS:
            d_forward = loaded_matrix.compute_distance(pair.source, pair.target)
            d_reverse = loaded_matrix.compute_distance(pair.target, pair.source)
            assert round(d_forward, 6) == round(d_reverse, 6), (
                f"Symmetry violation for pair ('{pair.source}', '{pair.target}'): "
                f"forward={d_forward}, reverse={d_reverse}"
            )

    def test_random_string_pair_symmetry(self, loaded_matrix: VisualConfusionMatrix):
        """Verify symmetry on 500 randomly generated character sequences."""
        rng = random.Random(42)
        vocab = list("abcdefghijklmnopqrstuvwxyz0123456789 -./")

        for _ in range(500):
            len1 = rng.randint(0, 15)
            len2 = rng.randint(0, 15)
            s1 = "".join(rng.choices(vocab, k=len1))
            s2 = "".join(rng.choices(vocab, k=len2))

            d12 = loaded_matrix.compute_distance(s1, s2)
            d21 = loaded_matrix.compute_distance(s2, s1)
            assert round(d12, 6) == round(d21, 6), (
                f"Symmetry failed on random strings: '{s1}' vs '{s2}': {d12} != {d21}"
            )

    def test_backtrace_step_cost_sum_equals_distance(self, loaded_matrix: VisualConfusionMatrix):
        """Verify that the sum of atomic step costs in align() exactly equals raw_distance."""
        rng = random.Random(1337)
        vocab = list("abcdefghijklmnopqrstuvwxyz0123456789 -./")

        for _ in range(200):
            len1 = rng.randint(1, 12)
            len2 = rng.randint(1, 12)
            s1 = "".join(rng.choices(vocab, k=len1))
            s2 = "".join(rng.choices(vocab, k=len2))

            align = loaded_matrix.align(s1, s2)
            sum_step_costs = sum(step.cost for step in align.steps)
            assert round(sum_step_costs, 6) == round(align.raw_distance, 6), (
                f"Backtrace sum ({sum_step_costs}) != raw_distance ({align.raw_distance}) for '{s1}' -> '{s2}'"
            )

    def test_backtrace_step_transformation_integrity(self, loaded_matrix: VisualConfusionMatrix):
        """Verify that executing the sequence of alignment steps on s1 produces exactly s2."""
        rng = random.Random(999)
        vocab = list("abcdefghijklmnopqrstuvwxyz0123456789 -./")

        for _ in range(200):
            len1 = rng.randint(0, 10)
            len2 = rng.randint(0, 10)
            s1 = "".join(rng.choices(vocab, k=len1))
            s2 = "".join(rng.choices(vocab, k=len2))

            align = loaded_matrix.align(s1, s2)
            # Reconstruct target string from steps
            reconstructed_tgt = []
            for step in align.steps:
                if step.operation in ("match", "substitution", "contraction", "expansion", "substitution_2_2", "insertion"):
                    reconstructed_tgt.append(step.target_chars)
                elif step.operation == "transposition":
                    reconstructed_tgt.append(step.target_chars)
                elif step.operation == "deletion":
                    pass  # Deleted chars do not appear in target

            res_str = "".join(reconstructed_tgt)
            assert res_str == s2.lower(), (
                f"Reconstruction failed for s1='{s1}', s2='{s2}': reconstructed='{res_str}'"
            )


# ===========================================================================
# 6. Empirical Latency & Memory Stress Testing (10,000 Operations)
# ===========================================================================

class TestEmpiricalLatencyAndMemoryStress:
    """Benchmark high-throughput performance SLAs under 10,000 operations."""

    def test_10000_exact_lookups_benchmark(self, loaded_trie: PrefixTrie):
        """Benchmark 10,000 exact lookups. Target: > 50,000 QPS (latency < 0.05ms)."""
        vocab = ["Amoxicillin", "Ampicillin", "Hydroxyzine", "Celebrex", "BID", "PO", "TID", "Metformin", "Aspirin", "NonExistentMed"]
        n_queries = 10000

        t0 = time.perf_counter()
        for i in range(n_queries):
            word = vocab[i % len(vocab)]
            _ = loaded_trie.search_exact(word)
        total_time = time.perf_counter() - t0
        qps = n_queries / total_time
        avg_ms = (total_time / n_queries) * 1000.0

        print(f"\n[BENCHMARK] 10,000 Exact Lookups: QPS={qps:.0f}, Avg={avg_ms*1000:.2f} µs/op")
        assert qps > 50000, f"Exact lookup QPS {qps:.0f} below 50,000 threshold"
        assert avg_ms < 0.05, f"Exact lookup latency {avg_ms:.4f}ms exceeds 0.05ms SLA"

    def test_10000_prefix_searches_benchmark(self, loaded_trie: PrefixTrie):
        """Benchmark 10,000 prefix completions. Target: > 10,000 QPS (latency < 0.10ms)."""
        prefixes = ["Am", "Hyd", "Ce", "Pr", "Me", "Asp", "PO", "TI", "BI", "Zz"]
        n_queries = 10000

        t0 = time.perf_counter()
        for i in range(n_queries):
            p = prefixes[i % len(prefixes)]
            _ = loaded_trie.search_prefix(p, max_results=10)
        total_time = time.perf_counter() - t0
        qps = n_queries / total_time
        avg_ms = (total_time / n_queries) * 1000.0

        print(f"\n[BENCHMARK] 10,000 Prefix Searches: QPS={qps:.0f}, Avg={avg_ms*1000:.2f} µs/op")
        assert qps > 10000, f"Prefix search QPS {qps:.0f} below 10,000 threshold"
        assert avg_ms < 0.10, f"Prefix search latency {avg_ms:.4f}ms exceeds 0.10ms SLA"

    def test_10000_confusion_distance_benchmark(self, loaded_matrix: VisualConfusionMatrix):
        """Benchmark 10,000 confusion distance evaluations. Target: < 0.10ms (100 µs) per calculation."""
        pairs = [
            ("Arnoxicillin", "Amoxicillin"),
            ("Hidroxizine", "Hydroxyzine"),
            ("5oomg", "500mg"),
            ("cloc", "doc"),
            ("Aomxicillin", "Amoxicillin"),
            ("Celexa", "Celebrex"),
            ("Prednisoe", "Prednisone"),
            ("po tid", "PO TID"),
        ]
        n_evals = 10000

        t0 = time.perf_counter()
        for i in range(n_evals):
            s1, s2 = pairs[i % len(pairs)]
            _ = loaded_matrix.compute_distance(s1, s2)
        total_time = time.perf_counter() - t0
        qps = n_evals / total_time
        avg_ms = (total_time / n_evals) * 1000.0

        print(f"\n[BENCHMARK] 10,000 Confusion Distance Evals: QPS={qps:.0f}, Avg={avg_ms:.4f} ms/op ({avg_ms*1000:.1f} µs)")
        assert avg_ms < 0.10, f"Confusion distance latency {avg_ms:.4f}ms exceeds 0.10ms SLA"

    def test_fuzzy_searches_throughput_and_latency(self, loaded_trie: PrefixTrie):
        """Benchmark fuzzy search across D=1 (10,000 queries) and D=2 (1,000 queries)."""
        queries = [
            "Amoxcillin", "Hidroxizine", "Prednsoe", "Metformn", "Celebrex",
            "Aspirn", "Ampiclin", "Lisinoprl", "Atorvastatn", "Omeprazol"
        ]

        # 1. D=1 Benchmark (10,000 queries) - Target: < 0.5ms
        t0 = time.perf_counter()
        for i in range(10000):
            q = queries[i % len(queries)]
            _ = loaded_trie.fuzzy_search(q, max_distance=1, max_results=5)
        t_d1 = time.perf_counter() - t0
        avg_d1_ms = (t_d1 / 10000) * 1000.0
        print(f"\n[BENCHMARK] 10,000 Fuzzy Searches (D=1): QPS={10000/t_d1:.0f}, Avg={avg_d1_ms:.4f} ms/op")
        assert avg_d1_ms < 0.50, f"D=1 latency {avg_d1_ms:.4f}ms exceeds 0.50ms"

        # 2. D=2 Benchmark (1,000 queries)
        t0 = time.perf_counter()
        for i in range(1000):
            q = queries[i % len(queries)]
            _ = loaded_trie.fuzzy_search(q, max_distance=2, max_results=5)
        t_d2 = time.perf_counter() - t0
        avg_d2_ms = (t_d2 / 1000) * 1000.0
        print(f"[BENCHMARK] 1,000 Fuzzy Searches (D=2): QPS={1000/t_d2:.0f}, Avg={avg_d2_ms:.4f} ms/op")

    def test_trie_memory_footprint(self, loaded_trie: PrefixTrie):
        """Verify Trie memory consumption is compact (< 5MB for full RxNorm catalog)."""
        stats = loaded_trie.get_stats()
        print(f"\n[MEMORY STATS] Trie Nodes={stats['total_nodes']}, Terms={stats['total_terms']}, Memory={stats['estimated_memory_kb']} KB")
        assert stats["total_terms"] >= 1800
        assert stats["estimated_memory_kb"] < 5000.0  # Under 5 MB

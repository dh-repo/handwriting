"""
tests/test_challenger_m6_ablation_stress.py
Adversarial Stress Test Suite for Milestone 6:
Multi-Source Ablation Benchmark & Metric Calculation Integrity.

Covers:
1. Levenshtein backtrace calculations under empty strings, single-char mismatches, huge strings, unicode diacritics.
2. Latency percentiles calculation under degenerate distributions (constant, empty, zero, single, extreme outliers).
3. Mathematical invariants verification: S + D + H == N_ref and (S + D + I) / N_ref == CER.
4. Trie search & beam rescorer stress testing under corrupted / adversarial OCR candidates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import string
import time
import unicodedata
import pytest
import numpy as np

from pipeline.evaluation.metrics import (
    ErrorBreakdown,
    LatencyMetrics,
    LatencyTracker,
    MetricResult,
    NormalizationConfig,
    ThroughputMetrics,
    _levenshtein_ops,
    _normalize_input_sequences,
    compute_cer,
    compute_metrics,
    compute_wer,
    normalize_text,
)
from pipeline.rescorer.trie import FuzzyMatch, MatchType, PrefixTrie, TrieMatch
from pipeline.rescorer.confusion_matrix import VisualConfusionMatrix
from pipeline.rescorer.beam_rescorer import BeamCandidate, BeamRescorer, ContextFeatures, RescorerResult
from pipeline.evaluation.benchmark_ablation import (
    AblationBenchmarkReport,
    StageResult,
    calculate_pnda,
    run_ablation_benchmark,
)


# ===========================================================================
# 1. LEVENSHTEIN BACKTRACE STRESS TESTS
# ===========================================================================

class TestLevenshteinBacktraceStress:
    """Stress testing the dynamic programming and backtrace edit operations."""

    def test_empty_string_combinations(self):
        """Test empty vs empty, empty vs non-empty, non-empty vs empty."""
        # Empty vs Empty
        s, d, ins, h = _levenshtein_ops([], [])
        assert (s, d, ins, h) == (0, 0, 0, 0)
        assert compute_cer("", "") == 0.0
        assert compute_wer("", "") == 0.0

        # Empty ref vs Non-empty hyp (all insertions)
        s, d, ins, h = _levenshtein_ops([], list("abcde"))
        assert (s, d, ins, h) == (0, 0, 5, 0)
        assert compute_cer("", "abcde") == 1.0

        # Non-empty ref vs Empty hyp (all deletions)
        s, d, ins, h = _levenshtein_ops(list("abcde"), [])
        assert (s, d, ins, h) == (0, 5, 0, 0)
        assert compute_cer("abcde", "") == 1.0

    def test_single_character_variations(self):
        """Test all single-character boundary conditions."""
        # Exact match
        s, d, ins, h = _levenshtein_ops(["a"], ["a"])
        assert (s, d, ins, h) == (0, 0, 0, 1)

        # Single substitution
        s, d, ins, h = _levenshtein_ops(["a"], ["b"])
        assert (s, d, ins, h) == (1, 0, 0, 0)

        # Single insertion
        s, d, ins, h = _levenshtein_ops(["a"], ["a", "b"])
        assert (s, d, ins, h) == (0, 0, 1, 1)

        # Single deletion
        s, d, ins, h = _levenshtein_ops(["a", "b"], ["a"])
        assert (s, d, ins, h) == (0, 1, 0, 1)

    def test_huge_string_scaling(self):
        """Test performance and correctness on huge strings (1,000 - 3,000 chars)."""
        rng = random.Random(1337)
        chars = string.ascii_letters + string.digits + " "

        ref_str = "".join(rng.choices(chars, k=2500))
        # Introduce 100 substitutions, 50 deletions, 50 insertions
        hyp_list = list(ref_str)
        # 100 subs
        for _ in range(100):
            pos = rng.randint(0, len(hyp_list) - 1)
            hyp_list[pos] = rng.choice(chars)
        # 50 dels
        for _ in range(50):
            if hyp_list:
                pos = rng.randint(0, len(hyp_list) - 1)
                hyp_list.pop(pos)
        # 50 ins
        for _ in range(50):
            pos = rng.randint(0, len(hyp_list))
            hyp_list.insert(pos, rng.choice(chars))

        hyp_str = "".join(hyp_list)

        t0 = time.perf_counter()
        s, d, ins, h = _levenshtein_ops(list(ref_str), list(hyp_str))
        elapsed = time.perf_counter() - t0

        # Verification
        assert s + d + h == len(ref_str), f"Invariant broken: {s}+{d}+{h} != {len(ref_str)}"
        assert elapsed < 3.0, f"Levenshtein execution took too long: {elapsed:.2f}s"
        cer = compute_cer(ref_str, hyp_str)
        assert 0.0 < cer < 0.25

    def test_unicode_diacritics_and_combining_characters(self):
        """Test unicode normalizations, accents, CJK, and emojis."""
        test_pairs = [
            ("café", "cafe", 1, 0, 0, 3),                     # accent substitution
            ("résumé", "resume", 2, 0, 0, 4),                 # two accents
            ("naïve", "naive", 1, 0, 0, 4),                   # diaeresis
            ("español", "espanol", 1, 0, 0, 6),               # tilde
            ("München", "Munchen", 1, 0, 0, 6),               # umlaut
            ("São Paulo", "Sao Paulo", 1, 0, 0, 8),           # tilde
            ("Ångström", "Angstrom", 2, 0, 0, 6),             # ring above & umlaut
            ("αβγδε", "αβγδε", 0, 0, 0, 5),                   # Greek identical
            ("αβγδε", "αβxδε", 1, 0, 0, 4),                   # Greek substitution
            ("Медицина", "Медицина", 0, 0, 0, 8),             # Cyrillic
            ("処方箋", "処方箋", 0, 0, 0, 3),                   # CJK characters
            ("💊500mg", "💊500mg", 0, 0, 0, 6),               # Emoji prefix
            ("💊500mg", "💉500mg", 1, 0, 0, 5),               # Emoji substitution
        ]

        for ref, hyp, exp_s, exp_d, exp_i, exp_h in test_pairs:
            s, d, ins, h = _levenshtein_ops(list(ref), list(hyp))
            assert (s, d, ins, h) == (exp_s, exp_d, exp_i, exp_h), (
                f"Failed on pair ({ref!r}, {hyp!r}): expected ({exp_s}, {exp_d}, {exp_i}, {exp_h}), got ({s}, {d}, {ins}, {h})"
            )
            assert s + d + h == len(ref)

    def test_nfc_vs_nfd_normalization(self):
        """Test decomposed combining characters vs precomposed characters."""
        # 'e\u0301' is 'e' + combining acute (2 unicode codepoints)
        # 'é' is precomposed (1 codepoint)
        nfd_str = "cafe\u0301"  # len 5
        nfc_str = "café"       # len 4

        # Raw comparison without normalization sees 1 sub + 1 ins or similar
        s_raw, d_raw, i_raw, h_raw = _levenshtein_ops(list(nfc_str), list(nfd_str))
        assert s_raw + d_raw + h_raw == len(nfc_str)

        # With remove_accents normalization
        cfg = NormalizationConfig(remove_accents=True)
        assert normalize_text(nfd_str, cfg) == "cafe"
        assert normalize_text(nfc_str, cfg) == "cafe"
        assert compute_cer(nfd_str, nfc_str, normalization_config=cfg) == 0.0

    def test_inverted_and_repetitive_strings(self):
        """Test pathological worst-case alignments."""
        # Completely inverted string
        s1 = "abcdefghij"
        s2 = "jihgfedcba"
        s, d, ins, h = _levenshtein_ops(list(s1), list(s2))
        assert s + d + h == len(s1)

        # Repetitive sequences
        r1 = "aaaaaa"
        r2 = "aaaaa"
        s, d, ins, h = _levenshtein_ops(list(r1), list(r2))
        assert (s, d, ins, h) == (0, 1, 0, 5)
        assert s + d + h == len(r1)

        # Alternating sequence offset by 1
        alt1 = "abababab"
        alt2 = "babababa"
        s, d, ins, h = _levenshtein_ops(list(alt1), list(alt2))
        assert s + d + h == len(alt1)


# ===========================================================================
# 2. LATENCY PERCENTILES DEGENERATE DISTRIBUTIONS
# ===========================================================================

class TestLatencyPercentilesStress:
    """Stress test LatencyTracker with degenerate, empty, constant, and outlier data."""

    def test_empty_distribution(self):
        """Test empty tracker does not raise ZeroDivisionError and returns zeroed metrics."""
        tracker = LatencyTracker()
        lat, tp = tracker.compute_metrics()

        assert isinstance(lat, LatencyMetrics)
        assert isinstance(tp, ThroughputMetrics)
        assert lat.p50_ms == 0.0
        assert lat.p90_ms == 0.0
        assert lat.p95_ms == 0.0
        assert lat.p99_ms == 0.0
        assert lat.mean_ms == 0.0
        assert lat.std_ms == 0.0
        assert lat.min_ms == 0.0
        assert lat.max_ms == 0.0
        assert tp.total_samples == 0
        assert tp.samples_per_second == 0.0

    def test_single_sample_distribution(self):
        """Test distribution with exactly one sample."""
        tracker = LatencyTracker()
        tracker.start()
        tracker.record_sample(12.34, char_count=10, word_count=2)
        tracker.stop()

        lat, tp = tracker.compute_metrics()
        assert lat.p50_ms == 12.34
        assert lat.p90_ms == 12.34
        assert lat.p95_ms == 12.34
        assert lat.p99_ms == 12.34
        assert lat.mean_ms == 12.34
        assert lat.min_ms == 12.34
        assert lat.max_ms == 12.34
        assert lat.std_ms == 0.0
        assert tp.total_samples == 1
        assert tp.total_characters == 10
        assert tp.total_words == 2

    def test_constant_distribution(self):
        """Test distribution where all sample latencies are identical."""
        tracker = LatencyTracker()
        for _ in range(500):
            tracker.record_sample(5.50, char_count=20, word_count=4)

        lat, tp = tracker.compute_metrics()
        assert lat.p50_ms == 5.50
        assert lat.p90_ms == 5.50
        assert lat.p95_ms == 5.50
        assert lat.p99_ms == 5.50
        assert lat.mean_ms == 5.50
        assert lat.min_ms == 5.50
        assert lat.max_ms == 5.50
        assert lat.std_ms == 0.0
        assert tp.total_samples == 500

    def test_all_zero_latencies(self):
        """Test distribution where all latencies are 0.0 ms (e.g. baseline instantaneous lookup)."""
        tracker = LatencyTracker()
        for _ in range(100):
            tracker.record_sample(0.0)

        lat, tp = tracker.compute_metrics()
        assert lat.p50_ms == 0.0
        assert lat.p95_ms == 0.0
        assert lat.mean_ms == 0.0
        assert lat.min_ms == 0.0
        assert lat.max_ms == 0.0
        assert lat.std_ms == 0.0

    def test_extreme_outliers(self):
        """Test distribution with heavy tail and extreme outliers."""
        tracker = LatencyTracker()
        # 950 samples at 1.0 ms
        for _ in range(950):
            tracker.record_sample(1.0)
        # 40 samples at 10.0 ms
        for _ in range(40):
            tracker.record_sample(10.0)
        # 9 samples at 100.0 ms
        for _ in range(9):
            tracker.record_sample(100.0)
        # 1 extreme outlier at 10,000.0 ms
        tracker.record_sample(10000.0)

        lat, tp = tracker.compute_metrics()
        assert lat.p50_ms == 1.0
        assert lat.p90_ms == 1.0
        assert lat.p95_ms == 1.45
        assert lat.p99_ms == 10.9
        assert lat.min_ms == 1.0
        assert lat.max_ms == 10000.0
        assert lat.mean_ms > 10.0
        assert lat.std_ms > 100.0

    def test_batch_recording_consistency(self):
        """Test that record_batch yields identical total count and apportioned latencies."""
        tracker_sample = LatencyTracker()
        tracker_batch = LatencyTracker()

        # Record 4 batches of size 16 taking 0.032s (32ms total -> 2ms per sample)
        for _ in range(4):
            tracker_batch.record_batch(
                batch_size=16,
                char_count=320,
                word_count=64,
                batch_duration_sec=0.032,
            )

        for _ in range(64):
            tracker_sample.record_sample(2.0, char_count=20, word_count=4)

        lat_b, tp_b = tracker_batch.compute_metrics()
        lat_s, tp_s = tracker_sample.compute_metrics()

        assert tp_b.total_samples == tp_s.total_samples == 64
        assert tp_b.total_characters == tp_s.total_characters == 1280
        assert tp_b.total_words == tp_s.total_words == 256
        assert lat_b.p50_ms == lat_s.p50_ms == 2.0
        assert lat_b.mean_ms == lat_s.mean_ms == 2.0


# ===========================================================================
# 3. MATHEMATICAL INVARIANTS VERIFICATION
# ===========================================================================

class TestMathematicalInvariants:
    """Property-based verification of mathematical identities: S + D + H == N_ref and CER formula."""

    def test_randomized_invariants_across_thousands_of_pairs(self):
        """
        Generate 1,000 randomized reference and hypothesis pairs of varying lengths,
        alphabets, and error distributions to verify S + D + H == N_ref and (S+D+I)/N_ref == CER.
        """
        rng = random.Random(2026)
        alphabets = [
            string.ascii_lowercase,
            string.ascii_letters + string.digits,
            "abcdefghijklmnopqrstuvwxyz0123456789 -/.,()%+",
            "αβγδεζηθικλμνξοπρστυφχψω",
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        ]
        norm_cfg = NormalizationConfig(lowercase=True, normalize_whitespace=True, strip_surrounding_whitespace=True)

        for i in range(1000):
            alphabet = rng.choice(alphabets)
            ref_len = rng.randint(0, 150)
            hyp_len = rng.randint(0, 150)

            ref = "".join(rng.choices(alphabet, k=ref_len))
            hyp = "".join(rng.choices(alphabet, k=hyp_len))

            # 1. Pure raw sequence invariant test
            s, d, ins, h = _levenshtein_ops(list(ref), list(hyp))
            n_ref = len(ref)

            # Invariant 1: S + D + H == N_ref
            assert s + d + h == n_ref, f"Invariant broken on iteration {i}: S({s}) + D({d}) + H({h}) != N_ref({n_ref})"

            # Invariant 2: (S + D + I) / N_ref formula check
            if n_ref > 0:
                expected_raw_cer = float(s + d + ins) / float(n_ref)
                # Compute via compute_metrics
                res = compute_metrics([ref], [hyp], normalization_config=norm_cfg)
                assert res.char_breakdown.substitutions + res.char_breakdown.deletions + res.char_breakdown.hits == res.char_breakdown.total_reference_units
                if res.char_breakdown.total_reference_units > 0:
                    calc_cer = float(res.char_breakdown.substitutions + res.char_breakdown.deletions + res.char_breakdown.insertions) / float(res.char_breakdown.total_reference_units)
                    assert math.isclose(res.char_breakdown.error_rate, round(calc_cer, 4), abs_tol=1e-4)
                    assert math.isclose(res.normalized_cer, round(calc_cer, 4), abs_tol=1e-4)
            else:
                if len(hyp) == 0:
                    assert compute_cer(ref, hyp) == 0.0
                else:
                    assert compute_cer(ref, hyp) == 1.0

    def test_word_level_invariants(self):
        """Verify word-level invariants: S_w + D_w + H_w == N_words_ref."""
        rng = random.Random(42)
        vocab = ["Amoxicillin", "500mg", "PO", "TID", "x10d", "Hydroxyzine", "25mg", "QHS", "Hydralazine", "take", "daily"]

        for _ in range(500):
            n_words_ref = rng.randint(1, 20)
            n_words_hyp = rng.randint(0, 20)

            ref_words = rng.choices(vocab, k=n_words_ref)
            hyp_words = rng.choices(vocab, k=n_words_hyp)

            s, d, ins, h = _levenshtein_ops(ref_words, hyp_words)

            assert s + d + h == n_words_ref, f"Word invariant broken: {s}+{d}+{h} != {n_words_ref}"
            expected_wer = float(s + d + ins) / float(n_words_ref)

            ref_str = " ".join(ref_words)
            hyp_str = " ".join(hyp_words)
            computed_wer = compute_wer(ref_str, hyp_str)
            assert math.isclose(computed_wer, expected_wer, rel_tol=1e-5, abs_tol=1e-5)

    def test_ablation_benchmark_persisted_artifacts_invariants(self):
        """
        Verify mathematical invariants directly on the serialized M6 ablation benchmark results:
        checkpoints/ablation_benchmark_results.json and evaluation_multisource_report.json.
        """
        json_path = Path("checkpoints/ablation_benchmark_results.json")
        if not json_path.exists():
            pytest.skip("ablation_benchmark_results.json not found")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_samples"] == 5050
        assert data["split"] == "test"

        for stage_name, stage_data in data["stages"].items():
            cb = stage_data["char_breakdown"]
            wb = stage_data["word_breakdown"]

            # Character Breakdown Invariants
            c_s = cb["substitutions"]
            c_d = cb["deletions"]
            c_i = cb["insertions"]
            c_h = cb["hits"]
            c_total = cb["total_reference_units"]
            c_err = cb["error_rate"]

            assert c_s + c_d + c_h == c_total, (
                f"Stage {stage_name} Char Invariant broken: {c_s} + {c_d} + {c_h} != {c_total}"
            )
            calc_cer = round(float(c_s + c_d + c_i) / float(c_total), 4)
            assert abs(calc_cer - c_err) <= 0.0001, (
                f"Stage {stage_name} Char Error Rate formula mismatch: {calc_cer} vs {c_err}"
            )
            assert abs(calc_cer - stage_data["cer"]) <= 0.0001

            # Word Breakdown Invariants
            w_s = wb["substitutions"]
            w_d = wb["deletions"]
            w_i = wb["insertions"]
            w_h = wb["hits"]
            w_total = wb["total_reference_units"]
            w_err = wb["error_rate"]

            assert w_s + w_d + w_h == w_total, (
                f"Stage {stage_name} Word Invariant broken: {w_s} + {w_d} + {w_h} != {w_total}"
            )
            calc_wer = round(float(w_s + w_d + w_i) / float(w_total), 4)
            assert abs(calc_wer - w_err) <= 0.0001
            assert abs(calc_wer - stage_data["wer"]) <= 0.0001

            # Sample Count Conservation
            wins = stage_data["wins_vs_baseline"]
            ties = stage_data["ties_vs_baseline"]
            losses = stage_data["losses_vs_baseline"]
            assert wins + ties + losses == stage_data["sample_count"] == 5050


# ===========================================================================
# 4. TRIE SEARCH & BEAM RESCORER UNDER CORRUPTED/ADVERSARIAL INPUTS
# ===========================================================================

@pytest.fixture(scope="module")
def loaded_trie():
    trie = PrefixTrie()
    vocab_dir = Path("data/reference_handwriting/vocabularies")
    if vocab_dir.exists():
        trie.load_vocabularies(vocab_dir)
    else:
        trie.insert("Amoxicillin", weight=2.0)
        trie.insert("Ampicillin", weight=2.0)
        trie.insert("Hydroxyzine", weight=2.0)
        trie.insert("Hydralazine", weight=2.0)
        trie.insert("TID", weight=3.0)
    return trie


@pytest.fixture(scope="module")
def rescorer(loaded_trie):
    cm = VisualConfusionMatrix(load_defaults=True)
    return BeamRescorer(
        trie=loaded_trie,
        confusion_matrix=cm,
        lambda_lexicon=1.0,
        lambda_context=0.8,
        lambda_confusion=0.5,
    )


class TestTrieAndBeamRescorerAdversarialStress:
    """Stress test PrefixTrie, VisualConfusionMatrix, and BeamRescorer with hostile/corrupted inputs."""

    def test_trie_corrupted_strings_and_null_bytes(self, loaded_trie):
        """Test Trie exact and fuzzy search with null bytes, control chars, and extreme queries."""
        hostile_queries = [
            "\x00\x00\x00",
            "Amox\x00icillin",
            "\x01\x02\x03\x04\x05",
            "\t\n\r   \t\n",
            "A" * 5000,
            "💊💉🏥👨‍⚕️",
            "=CMD|'/C calc'!A0",
            "<script>alert(1)</script>",
            "'; DROP TABLE medications; --",
            r"\\\///:::;;;***",
        ]

        for q in hostile_queries:
            # Must not raise exceptions
            exact = loaded_trie.search_exact(q)
            prefix = loaded_trie.search_prefix(q, max_results=5)
            fuzzy = loaded_trie.fuzzy_search(q, max_distance=2, max_results=5)
            assert isinstance(prefix, list)
            assert isinstance(fuzzy, list)

    def test_trie_fuzz_mutations_latency(self, loaded_trie):
        """Fuzz Trie fuzzy search with 1,000 random corruptions of drug names, checking latency < 2ms."""
        rng = random.Random(999)
        base_drugs = ["Amoxicillin", "Metoprolol", "Hydrochlorothiazide", "Atorvastatin", "Levothyroxine"]

        latencies = []
        for _ in range(1000):
            drug = rng.choice(base_drugs)
            d_list = list(drug)
            # Apply 1 or 2 random mutations
            mutation_type = rng.choice(["sub", "del", "ins", "trans"])
            if mutation_type == "sub" and d_list:
                idx = rng.randint(0, len(d_list) - 1)
                d_list[idx] = rng.choice(string.ascii_lowercase)
            elif mutation_type == "del" and len(d_list) > 2:
                idx = rng.randint(0, len(d_list) - 1)
                d_list.pop(idx)
            elif mutation_type == "ins":
                idx = rng.randint(0, len(d_list))
                d_list.insert(idx, rng.choice(string.ascii_lowercase))
            elif mutation_type == "trans" and len(d_list) >= 2:
                idx = rng.randint(0, len(d_list) - 2)
                d_list[idx], d_list[idx + 1] = d_list[idx + 1], d_list[idx]

            query = "".join(d_list)
            t0 = time.perf_counter()
            matches = loaded_trie.fuzzy_search(query, max_distance=2, max_results=5)
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_ms)

        p95_lat = float(np.percentile(latencies, 95))
        assert p95_lat < 2.0, f"95th percentile fuzzy search latency {p95_lat:.2f}ms exceeds 2.0ms limit"

    def test_beam_rescorer_empty_and_single_candidate(self, rescorer):
        """Test edge cases with empty beam candidate list or single candidate."""
        # Empty beam
        res_empty = rescorer.rescore_detailed([])
        assert res_empty.rescored_text == ""
        assert res_empty.confidence == 0.0
        assert not res_empty.rescore_applied

        # Single candidate
        cand = BeamCandidate(text="Amoxicillin 500mg PO TID", log_prob=-0.2)
        res_single = rescorer.rescore_detailed([cand])
        assert res_single.rescored_text == "Amoxicillin 500mg PO TID"
        assert 0.0 <= res_single.confidence <= 1.0

    def test_beam_rescorer_degenerate_logprobs(self, rescorer):
        """Test candidates with NaN, -Inf, +Inf, or identical log probabilities."""
        # NaN log prob gets sanitized
        cand_nan = BeamCandidate(text="Amoxicillin 500mg", log_prob=float("nan"))
        assert cand_nan.log_prob == -1000.0

        # Identical log probs
        cands_equal = [
            BeamCandidate(text="Amoxicillin 500mg", log_prob=-0.5),
            BeamCandidate(text="Ampicillin 500mg", log_prob=-0.5),
            BeamCandidate(text="Arnoxicillin 500mg", log_prob=-0.5),
        ]
        res_equal = rescorer.rescore_detailed(cands_equal)
        assert res_equal.rescored_text in ["Amoxicillin 500mg", "Ampicillin 500mg"]
        assert 0.0 < res_equal.confidence <= 1.0

    def test_beam_rescorer_lasa_disambiguation_with_context(self, rescorer):
        """
        Stress test LASA drug disambiguation:
        E.g., Amoxicillin vs Ampicillin under oral dosage context (Amoxicillin standard 500mg PO TID).
        """
        candidates = [
            BeamCandidate(text="Ampicillin 500mg PO TID", log_prob=-0.40),  # Baseline top OCR beam (slightly higher logprob)
            BeamCandidate(text="Amoxicillin 500mg PO TID", log_prob=-0.45), # Ground truth
        ]

        # Explicit context matching Amoxicillin standard oral outpatient regimen
        ctx = ContextFeatures(dosage="500mg", route="PO", frequency="TID")
        res = rescorer.rescore_detailed(candidates, context=ctx)

        # Rescorer should select Amoxicillin
        assert res.rescored_text == "Amoxicillin 500mg PO TID"
        assert res.rescore_applied is True

    def test_beam_rescorer_extreme_dosage_and_contradictions(self, rescorer):
        """Test extreme dosage values (>50,000mg) and contradictory route/form combinations."""
        # Incompatible route / form: IV tablet
        cands = [
            BeamCandidate(text="Amoxicillin 500mg", log_prob=-0.5),
        ]
        ctx_bad = ContextFeatures(route="IV", form="tablet")
        score_bad = rescorer._score_context("Amoxicillin 500mg", "Amoxicillin", ctx_bad)

        ctx_good = ContextFeatures(route="PO", form="capsule")
        score_good = rescorer._score_context("Amoxicillin 500mg", "Amoxicillin", ctx_good)

        assert score_good > score_bad, "Context scoring failed to penalize contradictory IV tablet"

        # Extreme toxic dosage: 100,000mg
        ctx_toxic = ContextFeatures(dosage="100000mg")
        score_toxic = rescorer._score_context("Amoxicillin 100000mg", "Amoxicillin", ctx_toxic)
        assert score_toxic < 0.0, "Context scoring failed to penalize extreme 100g dosage"

    def test_visual_confusion_matrix_symmetry_and_non_negativity(self):
        """Verify that VisualConfusionMatrix is symmetric, non-negative, and fast."""
        cm = VisualConfusionMatrix(load_defaults=True)

        for (s, t), cost in cm._matrix.items():
            assert cost >= 0.0, f"Negative cost found for pair ({s}, {t}): {cost}"
            reverse_cost = cm.get_cost(t, s)
            assert math.isclose(cost, reverse_cost, rel_tol=1e-5), f"Asymmetric cost for ({s}, {t}): {cost} vs {reverse_cost}"

        # Fast DP distance vs Backtrace alignment consistency on 50 sample pairs
        test_pairs = [
            ("amoxicillin", "arnoxicillin"),
            ("prednisone", "prednisolone"),
            ("hydroxyzine", "hydralazine"),
            ("take 1 tab po tid", "take l tab po tid"),
            ("500mg", "500ing"),
            ("q4-6h", "q4-6h"),
        ]

        for s1, s2 in test_pairs:
            dist_fast = cm.compute_distance(s1, s2)
            align_res = cm.align(s1, s2)
            assert math.isclose(dist_fast, align_res.raw_distance, rel_tol=1e-5), (
                f"Distance mismatch between fast DP ({dist_fast}) and alignment trace ({align_res.raw_distance}) for ({s1}, {s2})"
            )

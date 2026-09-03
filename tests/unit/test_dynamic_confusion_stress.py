"""
tests/unit/test_dynamic_confusion_stress.py
Empirical stress-testing harness for Milestone 2:
Online Self-Tuning Visual Confusion Matrix & Beam Rescorer Recalibration.

Targets:
1. High-concurrency multithreading:
   - 16 concurrent threads adapting and reading distances simultaneously.
   - Concurrent rescoring + dynamic adaptation without race conditions.
   - Post-concurrency matrix symmetry validation across all pairs.
2. Boundary inputs:
   - String length ceiling at 999, 1000, and 1001 characters.
   - Zero-length strings, pure insertions, pure deletions, whitespace.
   - Special Unicode glyphs (Greek, Cyrillic, ligatures, diacritics, emoji).
   - Punctuation and symbol optical substitutions.
3. Persistence fuzzing:
   - Corrupted JSON syntax and truncated files.
   - Auto-creation of missing deeply nested parent directories.
   - Read-only file and directory permissions behavior.
   - Export-modify-reload consistency and data integrity.
4. Extreme discounting:
   - 100 consecutive adaptations across 1:1, 2:1, 1:2, 2:2, and unmapped pairs.
   - Strict monotonicity and asymptotic decay.
   - Absolute clinical safety floor preservation (cost NEVER drops below 0.15).
   - Adversarial learning rates and min_cost parameters.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import random
import stat
import tempfile
import time
from typing import Any, Dict, List

import pytest

from pipeline.rescorer.confusion_matrix import (
    VisualConfusionMatrix,
)
from pipeline.rescorer.beam_rescorer import (
    BeamCandidate,
    BeamRescorer,
)
from pipeline.rescorer.trie import PrefixTrie


# ===========================================================================
# 1. High-Concurrency Multithreading
# ===========================================================================

class TestConcurrencyAndThreadSafetyStress:
    """Stress test VisualConfusionMatrix under high-concurrency multi-threaded load."""

    def test_high_concurrency_threads_adapting_and_reading_simultaneously(self) -> None:
        """
        Run 16 concurrent threads:
        - 8 writer threads adapting various pairs with random learning rates
        - 8 reader threads computing distances simultaneously
        Total 800+ concurrent operations.
        Verifies thread safety, no deadlock, valid distance ranges, and symmetric integrity.
        """
        cm = VisualConfusionMatrix(load_defaults=True)

        pairs_pool = [
            ("Arnoxicillin", "Amoxicillin"),
            ("prednlsone", "prednisone"),
            ("take po daily", "take 10 daily"),
            ("xlonopin", "klonopin"),
            ("hydroxyzine", "hydralazine"),
            ("celebrex", "celexa"),
            ("clonidine", "klonopin"),
            ("metformin", "metronidazole"),
            ("cydindamycin", "clindamycin"),
            ("aspirin, 500mg", "aspirin. 500mg"),
        ]

        query_pool = [
            ("amoxicillin", "ampicillin"),
            ("prednisone", "prednisolone"),
            ("xlonopin", "klonopin"),
            ("take po daily", "take 10 daily"),
            ("hydroxyzine", "hydralazine"),
            ("aspirin, 500mg", "aspirin. 500mg"),
            ("randomtoken1", "randomtoken2"),
        ]

        num_threads = 16
        ops_per_thread = 50
        errors: List[tuple[int, str, Any]] = []

        def adapter_worker(worker_id: int) -> None:
            try:
                for _ in range(ops_per_thread):
                    pair = random.choice(pairs_pool)
                    lr = random.uniform(0.05, 0.40)
                    res = cm.adapt_from_correction(pair[0], pair[1], learning_rate=lr, min_cost=0.15)
                    assert isinstance(res, list)
                    time.sleep(0.0005)
            except Exception as e:
                errors.append((worker_id, "adapt", e))

        def reader_worker(worker_id: int) -> None:
            try:
                for _ in range(ops_per_thread):
                    q = random.choice(query_pool)
                    d = cm.compute_distance(q[0], q[1], normalize=True)
                    assert 0.0 <= d <= 2.0, f"Distance out of bounds: {d}"
                    time.sleep(0.0005)
            except Exception as e:
                errors.append((worker_id, "read", e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = []
            for i in range(num_threads // 2):
                futures.append(pool.submit(adapter_worker, i))
            for i in range(num_threads // 2, num_threads):
                futures.append(pool.submit(reader_worker, i))
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Encountered concurrency errors: {errors}"

        # Post-concurrency invariant: verify complete bidirectional symmetry
        symmetric_mismatches = []
        for (src, tgt), cost in cm._matrix.items():
            rev_cost = cm._matrix.get((tgt, src))
            if rev_cost is None:
                symmetric_mismatches.append((src, tgt, "missing reverse key"))
            elif abs(cost - rev_cost) > 1e-6:
                symmetric_mismatches.append((src, tgt, f"cost mismatch {cost} vs {rev_cost}"))

        assert len(symmetric_mismatches) == 0, f"Symmetry violations: {symmetric_mismatches}"
        assert all(c >= 0.15 for c in cm._matrix.values()), "A cost fell below 0.15 floor!"

    def test_concurrent_beam_rescorer_and_confusion_adaptation(self) -> None:
        """
        Stress test BeamRescorer rescoring candidates concurrently while confusion matrix is adapted.
        12 concurrent worker threads executing 40 iterations each.
        """
        trie = PrefixTrie()
        vocab_path = Path("data/reference_handwriting/vocabularies")
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

        cands_batch = [
            [
                BeamCandidate(text="xlonopin 1mg", log_prob=-0.10),
                BeamCandidate(text="klonopin 1mg", log_prob=-0.40),
            ],
            [
                BeamCandidate(text="prednlsone 10mg", log_prob=-0.15),
                BeamCandidate(text="prednisone 10mg", log_prob=-0.30),
            ],
            [
                BeamCandidate(text="Arnoxicillin 500mg", log_prob=-0.12),
                BeamCandidate(text="Amoxicillin 500mg", log_prob=-0.35),
            ],
        ]

        errors: List[tuple[str, Any]] = []

        def rescore_worker() -> None:
            try:
                for _ in range(40):
                    hyps = random.choice(cands_batch)
                    res = rescorer.rescore_detailed(hyps)
                    assert res.rescored_text in [h.text for h in hyps]
                    assert res.confidence > 0.0
            except Exception as e:
                errors.append(("rescore", e))

        def adapt_worker() -> None:
            try:
                for _ in range(40):
                    rescorer.adapt_confusion_matrix("xlonopin", "klonopin", learning_rate=0.15, min_cost=0.15)
                    rescorer.adapt_confusion_matrix("prednlsone", "prednisone", learning_rate=0.15, min_cost=0.15)
            except Exception as e:
                errors.append(("adapt", e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            futs = [pool.submit(rescore_worker) for _ in range(8)] + [pool.submit(adapt_worker) for _ in range(4)]
            concurrent.futures.wait(futs)

        assert len(errors) == 0, f"Encountered errors in concurrent rescore/adapt: {errors}"

    def test_concurrent_export_and_adaptation(self, tmp_path: Path) -> None:
        """
        Verify thread safety when one thread repeatedly exports dynamic state to disk
        while other threads are actively modifying the matrix.
        """
        cm = VisualConfusionMatrix(load_defaults=True)
        export_file = tmp_path / "concurrent_export.json"
        errors: List[tuple[str, Any]] = []

        def exporter() -> None:
            try:
                for _ in range(25):
                    cm.export_dynamic_state(export_file)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(("export", e))

        def adapter(pair: tuple[str, str]) -> None:
            try:
                for _ in range(25):
                    cm.adapt_from_correction(pair[0], pair[1], learning_rate=0.20, min_cost=0.15)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(("adapt", e))

        pairs = [
            ("xlonopin", "klonopin"),
            ("Arnoxicillin", "Amoxicillin"),
            ("prednlsone", "prednisone"),
            ("take po daily", "take 10 daily"),
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futs = [pool.submit(exporter), pool.submit(exporter)]
            for p in pairs:
                futs.append(pool.submit(adapter, p))
            concurrent.futures.wait(futs)

        assert len(errors) == 0, f"Encountered errors during concurrent export/adapt: {errors}"
        assert export_file.exists()
        with open(export_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_dynamic_pairs"] >= 4


# ===========================================================================
# 2. Boundary Inputs & Edge Cases
# ===========================================================================

class TestBoundaryInputStress:
    """Stress test boundary string lengths, special glyphs, punctuation, and empty inputs."""

    @pytest.fixture
    def cm(self) -> VisualConfusionMatrix:
        return VisualConfusionMatrix(load_defaults=True)

    def test_string_length_ceiling_boundary_999_1000_1001(self, cm: VisualConfusionMatrix) -> None:
        """
        Verify precise boundary behavior around the 1000 character DP safety ceiling:
        - len = 999: DP alignment runs and extracts substitution.
        - len = 1000: DP alignment runs and extracts substitution.
        - len = 1001: safety limit triggered, skips DP safely and returns [].
        """
        # 1. Length 999 (998 'a's + 1 differing char 'x' vs 'k')
        s999_orig = "a" * 998 + "x"
        s999_corr = "a" * 998 + "k"
        assert len(s999_orig) == 999
        res999 = cm.adapt_from_correction(s999_orig, s999_corr, learning_rate=0.20)
        assert len(res999) == 1
        assert res999[0]["source"] == "x"
        assert res999[0]["target"] == "k"

        # 2. Length 1000 (999 'a's + 1 differing char 'x' vs 'k')
        s1000_orig = "a" * 999 + "x"
        s1000_corr = "a" * 999 + "k"
        assert len(s1000_orig) == 1000
        res1000 = cm.adapt_from_correction(s1000_orig, s1000_corr, learning_rate=0.20)
        assert len(res1000) == 1
        assert res1000[0]["source"] == "x"
        assert res1000[0]["target"] == "k"

        # 3. Length 1001 (1000 'a's + 1 differing char 'x' vs 'k')
        s1001_orig = "a" * 1000 + "x"
        s1001_corr = "a" * 1000 + "k"
        assert len(s1001_orig) == 1001
        res1001 = cm.adapt_from_correction(s1001_orig, s1001_corr, learning_rate=0.20)
        assert res1001 == [], f"Expected empty list for length > 1000, got {res1001}"

        # 4. Asymmetric lengths: one <= 1000, other > 1000
        res_asym1 = cm.adapt_from_correction("short", "b" * 1001)
        assert res_asym1 == []
        res_asym2 = cm.adapt_from_correction("a" * 1001, "short")
        assert res_asym2 == []

    def test_compute_distance_large_strings(self, cm: VisualConfusionMatrix) -> None:
        """Verify compute_distance handles strings > 1000 characters without memory issues."""
        s1 = "amoxicillin 500mg " * 100  # 1800 chars
        s2 = "ampicillin 500mg " * 100   # 1700 chars
        dist = cm.compute_distance(s1, s2, normalize=True)
        assert 0.0 < dist < 1.0

    def test_zero_length_and_pure_edits(self, cm: VisualConfusionMatrix) -> None:
        """
        Verify empty and pure insertion/deletion strings do not create erroneous substitutions:
        - Pure insertions and pure deletions are not optical substitutions and return [].
        """
        assert cm.adapt_from_correction("", "") == []
        assert cm.adapt_from_correction("abc", "") == []
        assert cm.adapt_from_correction("", "abc") == []
        assert cm.adapt_from_correction("   ", "   ") == []

    def test_special_unicode_glyphs_and_diacritics(self, cm: VisualConfusionMatrix) -> None:
        """
        Verify robust handling of diverse Unicode alphabets and symbols:
        - Greek letters: 'α' vs 'β'
        - Ligatures: 'æ' vs 'a'
        - Accented characters: 'é' vs 'e'
        - Cyrillic lookalikes: 'а' (U+0430) vs 'a' (U+0061)
        - Emoji glyphs: '💊' vs '💉'
        """
        # Greek letters
        res_greek = cm.adapt_from_correction("α-blocker", "β-blocker", learning_rate=0.20)
        assert len(res_greek) == 1
        assert res_greek[0]["source"] == "α"
        assert res_greek[0]["target"] == "β"
        assert cm.get_cost("α", "β") == res_greek[0]["updated_cost"]
        assert cm.get_cost("β", "α") == res_greek[0]["updated_cost"]

        # Accents
        res_accent = cm.adapt_from_correction("café", "cafe", learning_rate=0.20)
        assert len(res_accent) == 1
        assert res_accent[0]["source"] == "é"
        assert res_accent[0]["target"] == "e"

        # Cyrillic homoglyph
        cyrillic_a = "\u0430"  # Cyrillic small letter a
        latin_a = "a"
        res_cyrillic = cm.adapt_from_correction(f"{cyrillic_a}mox", f"{latin_a}mox", learning_rate=0.20)
        assert len(res_cyrillic) == 1
        assert res_cyrillic[0]["source"] == cyrillic_a
        assert res_cyrillic[0]["target"] == latin_a

        # Emoji
        res_emoji = cm.adapt_from_correction("dose 💊", "dose 💉", learning_rate=0.20)
        assert len(res_emoji) == 1
        assert res_emoji[0]["source"] == "💊"
        assert res_emoji[0]["target"] == "💉"

    def test_punctuation_and_symbols_optical_substitutions(self, cm: VisualConfusionMatrix) -> None:
        """Verify optical substitution extraction and discounting for punctuation marks."""
        # Comma vs Period, Semicolon vs Colon
        res = cm.adapt_from_correction("500,mg; qd", "500.mg: qd", learning_rate=0.20)
        ops = {u["source"]: u["target"] for u in res}
        assert ops.get(",") == "."
        assert ops.get(";") == ":"
        assert cm.get_cost(",", ".") == cm.get_cost(".", ",")


# ===========================================================================
# 3. Persistence Fuzzing
# ===========================================================================

class TestPersistenceFuzzing:
    """Stress test dynamic persistence against corrupted files, permission errors, and deep paths."""

    def test_missing_parent_directories_created_on_export(self, tmp_path: Path) -> None:
        """Verify export_dynamic_state automatically creates nested parent directories."""
        cm = VisualConfusionMatrix(load_defaults=True)
        cm.adapt_from_correction("x", "k")

        deep_target = tmp_path / "deep" / "nested" / "feedback" / "cache" / "dynamic_matrix.json"
        assert not deep_target.parent.exists()

        result_path = cm.export_dynamic_state(deep_target)
        assert Path(result_path).exists()
        assert deep_target.is_file()

        with open(deep_target, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == "1.0"
        assert data["total_dynamic_pairs"] >= 1

    def test_corrupted_json_syntax_raises_decode_error(self, tmp_path: Path) -> None:
        """Verify loading malformed JSON raises json.JSONDecodeError rather than silently corrupting."""
        cm = VisualConfusionMatrix(load_defaults=True)
        corrupted_file = tmp_path / "corrupted.json"
        corrupted_file.write_text('{"dynamic_pairs": [{"source": "x", "target": "k", "cost": 0.2', encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            cm.load_dynamic_state(corrupted_file)

    def test_empty_file_raises_decode_error(self, tmp_path: Path) -> None:
        """Verify loading 0-byte file raises json.JSONDecodeError."""
        cm = VisualConfusionMatrix(load_defaults=True)
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            cm.load_dynamic_state(empty_file)

    def test_empty_json_object_loads_zero_safely(self, tmp_path: Path) -> None:
        """Verify loading an empty JSON object {} loads 0 pairs safely without exception."""
        cm = VisualConfusionMatrix(load_defaults=True)
        empty_obj_file = tmp_path / "empty_obj.json"
        empty_obj_file.write_text("{}", encoding="utf-8")

        count = cm.load_dynamic_state(empty_obj_file)
        assert count == 0

    def test_read_only_permissions_behavior(self, tmp_path: Path) -> None:
        """
        Verify permissions handling:
        - Loading from a read-only file (mode 0400) succeeds.
        - Exporting to a read-only file or directory raises PermissionError cleanly.
        """
        cm = VisualConfusionMatrix(load_defaults=True)
        cm.adapt_from_correction("x", "k")

        ro_file = tmp_path / "read_only.json"
        cm.export_dynamic_state(ro_file)

        # Make file read-only
        os.chmod(ro_file, stat.S_IRUSR)

        # 1. Loading read-only file succeeds
        cm2 = VisualConfusionMatrix(load_defaults=True)
        loaded = cm2.load_dynamic_state(ro_file)
        assert loaded >= 1
        assert cm2.get_cost("x", "k") == cm.get_cost("x", "k")

        # 2. Overwriting read-only file raises PermissionError
        with pytest.raises(PermissionError):
            cm.export_dynamic_state(ro_file)

        # 3. Exporting into a read-only directory raises PermissionError
        ro_dir = tmp_path / "ro_directory"
        ro_dir.mkdir()
        os.chmod(ro_dir, stat.S_IRUSR | stat.S_IXUSR)  # Read + execute only
        try:
            with pytest.raises(PermissionError):
                cm.export_dynamic_state(ro_dir / "out.json")
        finally:
            os.chmod(ro_dir, stat.S_IRWXU)

    def test_export_modify_reload_consistency(self, tmp_path: Path) -> None:
        """
        Verify export-modify-reload lifecycle:
        1. Adapt confusion pair ('x' <-> 'k').
        2. Export to JSON.
        3. Modify cost and add an extra pair in JSON.
        4. Reload into a fresh VisualConfusionMatrix instance.
        5. Verify modified cost and injected pair take effect in distance computation.
        """
        cm1 = VisualConfusionMatrix(load_defaults=True)
        cm1.adapt_from_correction("xlonopin", "klonopin", learning_rate=0.50)
        export_file = tmp_path / "state_lifecycle.json"
        cm1.export_dynamic_state(export_file)

        # Modify JSON on disk
        data = json.loads(export_file.read_text(encoding="utf-8"))
        for p in data["dynamic_pairs"]:
            if p["source"] == "x" and p["target"] == "k":
                p["cost"] = 0.185
        # Inject custom pair
        data["dynamic_pairs"].append({
            "source": "q",
            "target": "g",
            "cost": 0.22,
            "previous_cost": 1.20,
            "operation": "substitution",
            "update_count": 1,
        })
        export_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Reload into new instance
        cm2 = VisualConfusionMatrix(load_defaults=True)
        count = cm2.load_dynamic_state(export_file)
        assert count == len(data["dynamic_pairs"])
        assert cm2.get_cost("x", "k") == 0.185
        assert cm2.get_cost("k", "x") == 0.185
        assert cm2.get_cost("q", "g") == 0.22
        assert cm2.get_cost("g", "q") == 0.22


# ===========================================================================
# 4. Extreme Discounting Stress (100 Iterations & Floor Preservation)
# ===========================================================================

class TestExtremeDiscountingStress:
    """Stress test extreme adaptation iterations and mathematical floor guarantees."""

    @pytest.mark.parametrize(
        "pair_type, orig, corr, src, tgt",
        [
            ("1:1 substitution", "s", "5", "s", "5"),
            ("2:1 contraction", "Arnoxicillin", "Amoxicillin", "rn", "m"),
            ("1:2 expansion", "Amoxicillin", "Arnoxicillin", "m", "rn"),
            ("2:2 substitution", "take po daily", "take 10 daily", "po", "10"),
            ("unmapped 1:1", "xlonopin", "klonopin", "x", "k"),
        ],
    )
    def test_100_consecutive_adaptations_asymptotic_decay_and_floor(
        self,
        pair_type: str,
        orig: str,
        corr: str,
        src: str,
        tgt: str,
    ) -> None:
        """
        Adapt 100 consecutive times on the same pair.
        Verifies:
        1. Monotonically non-increasing: cost[i] >= cost[i+1].
        2. Absolute lower bound: cost NEVER drops below 0.15 at any step.
        3. Asymptotic floor convergence: reaches exactly 0.15 after sufficient steps.
        4. Bidirectional symmetry preserved at all 100 steps.
        """
        cm = VisualConfusionMatrix(load_defaults=True)
        history: List[float] = []

        for step in range(100):
            res = cm.adapt_from_correction(orig, corr, learning_rate=0.15, min_cost=0.15)
            assert len(res) >= 1, f"Step {step} yielded no updates for {pair_type}"
            cost = cm.get_cost(src, tgt)
            rev_cost = cm.get_cost(tgt, src)
            assert cost == rev_cost, f"Symmetry broken at step {step}: {cost} != {rev_cost}"
            assert cost >= 0.15, f"Cost dropped below 0.15 floor at step {step}: {cost}"
            history.append(cost)

        # Monotonicity check
        for i in range(len(history) - 1):
            assert history[i] >= history[i + 1], (
                f"{pair_type}: Non-monotonic at step {i}: {history[i]} -> {history[i+1]}"
            )

        # Asymptotic floor reached
        assert history[-1] == 0.15, f"{pair_type}: Did not reach 0.15, final cost: {history[-1]}"

    @pytest.mark.parametrize("lr", [0.0, 0.05, 0.50, 1.0, -0.5, 2.0])
    def test_extreme_and_out_of_bounds_learning_rates(self, lr: float) -> None:
        """
        Verify stability when learning_rate is at extreme boundaries or out of [0, 1] range:
        - lr = 0.0: cost remains unchanged across 100 iterations.
        - lr = 1.0: cost jumps immediately to 0.15 on iteration 1 and stays clamped.
        - lr < 0.0: clamped to 0.0, cost remains unchanged.
        - lr > 1.0: clamped to 1.0, cost jumps to 0.15 and stays clamped.
        """
        cm = VisualConfusionMatrix(load_defaults=True)
        initial_cost = cm.get_cost("c", "e")

        for _ in range(20):
            cm.adapt_from_correction("c", "e", learning_rate=lr, min_cost=0.15)
            current_cost = cm.get_cost("c", "e")
            assert current_cost >= 0.15

        final_cost = cm.get_cost("c", "e")
        if lr <= 0.0:
            assert final_cost == initial_cost, f"lr={lr} modified cost unexpectedly: {final_cost}"
        elif lr >= 1.0:
            assert final_cost == 0.15, f"lr={lr} did not clamp to min_cost 0.15: {final_cost}"

    @pytest.mark.parametrize("adversarial_min_cost", [0.0, -1.0, -100.0, 0.1499])
    def test_adversarial_min_cost_enforces_015_floor(self, adversarial_min_cost: float) -> None:
        """
        Verify that passing min_cost < 0.15 is strictly overridden to >= 0.15 for clinical safety,
        even after 50 aggressive adaptations.
        """
        cm = VisualConfusionMatrix(load_defaults=True)
        for _ in range(50):
            cm.adapt_from_correction("l", "1", learning_rate=0.90, min_cost=adversarial_min_cost)

        assert cm.get_cost("l", "1") >= 0.15
        assert cm.get_cost("1", "l") >= 0.15
        assert cm.get_cost("l", "1") == 0.15

    def test_higher_custom_min_cost_respected(self) -> None:
        """Verify that when min_cost > 0.15 (e.g. 0.40), the higher bound is respected."""
        cm = VisualConfusionMatrix(load_defaults=True)
        for _ in range(30):
            cm.adapt_from_correction("s", "5", learning_rate=0.50, min_cost=0.40)

        assert cm.get_cost("s", "5") == 0.40
        assert cm.get_cost("5", "s") == 0.40

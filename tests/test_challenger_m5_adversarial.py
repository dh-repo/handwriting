"""
tests/test_challenger_m5_adversarial.py
Adversarial Stress Test Suite for Milestone 5 Flywheel Components.
Authored by challenger_m5_flywheel_1 (Empirical Challenger).

Exhaustively stress-tests:
1. Manifest storage concurrency contention, POSIX advisory file locking, and malformed payload injection.
2. DP alignment boundary cases (empty, multi-gram, ligatures, unicode, emojis, length limits) and dynamic confusion matrix invariant bounds (c_min >= 0.15).
3. Experience replay 50:50 ratio edge cases (empty pools, single feedback sample, odd batch sizes, infinite loop conditions).
4. LoRA micro-tuning parameter isolation, MPS cache clearing, precision autocasting, and finite loss convergence.
5. Clinical LASA safety gate zero-tolerance detection across all 20 bidirectional pairs, regex boundaries, near-misses, and CER regression bounds.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.config import Settings, get_settings, reset_settings_cache
from backend.app.main import create_app
from backend.app.routes.feedback import (
    _append_to_manifest,
    _save_line_crop,
    _trigger_dynamic_confusion_update,
)
from backend.app.schemas import (
    ConfusionUpdateRecord,
    FeedbackCorrectionPayload,
    FeedbackResponse,
    FeedbackStatsResponse,
)
from pipeline.rescorer.confusion_matrix import (
    AlignmentResult,
    AlignmentStep,
    VisualConfusionMatrix,
    DEFAULT_CONFUSION_PAIRS,
)
from pipeline.training.dataset import create_dummy_processor
from pipeline.training.experience_replay import ExperienceReplayDataset, ReplayBatchSampler
from pipeline.training.lora_micro_tune import resolve_device_target
from pipeline.training.ship_gate import (
    LasaAuditResult,
    assert_shippable_checkpoint,
    audit_lasa_safety,
    audit_lasa_safety_hardened,
    check_cer_regression,
    decide_ship,
    load_lasa_catalog,
)
from pipeline.training.train import create_tiny_mock_model

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ===========================================================================
# Helpers & Fixtures
# ===========================================================================

def _make_b64_png(w: int = 100, h: int = 40, color: Tuple[int, int, int] = (255, 255, 255)) -> str:
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def temp_workspace() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="challenger_m5_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# 1. Feedback Concurrency & Storage Adversarial Tests
# ===========================================================================

class TestFeedbackConcurrencyAndStorageAdversarial:
    """Stress tests for POSIX locking contention, corrupted crops, and schema injection."""

    def test_concurrent_manifest_burst_no_line_interleaving(self, temp_workspace: Path) -> None:
        """50 concurrent threads writing to manifest simultaneously must produce exactly 50 valid JSON lines."""
        manifest_file = temp_workspace / "data" / "feedback" / "manifest.jsonl"
        thread_count = 50

        def _worker(thread_idx: int) -> int:
            rec = {
                "feedback_id": f"fb_{thread_idx:04d}",
                "document_id": f"doc_{thread_idx}",
                "line_id": f"line_{thread_idx}",
                "original_prediction": f"pred_{thread_idx}_" + "x" * 20,
                "operator_correction": f"corr_{thread_idx}_" + "y" * 20,
                "timestamp": f"2026-09-03T12:00:{thread_idx:02d}Z",
            }
            _append_to_manifest(manifest_file, rec)
            return thread_idx

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(_worker, i) for i in range(thread_count)]
            for f in as_completed(futures):
                f.result()

        assert manifest_file.exists()
        lines = manifest_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == thread_count

        parsed_ids = set()
        for idx, line in enumerate(lines):
            try:
                data = json.loads(line)
                parsed_ids.add(data["feedback_id"])
            except Exception as exc:
                pytest.fail(f"Corrupted JSON line detected at index {idx}: {line} ({exc})")

        assert len(parsed_ids) == thread_count

    def test_concurrent_read_lock_during_heavy_writes(self, temp_workspace: Path) -> None:
        """Simultaneous get_feedback_stats (LOCK_SH) during active _append_to_manifest (LOCK_EX) must not deadlock or crash."""
        manifest_file = temp_workspace / "manifest.jsonl"
        stop_flag = False

        # Initial seed
        for i in range(10):
            _append_to_manifest(manifest_file, {"id": i, "val": "seed"})

        def _writer():
            for i in range(10, 40):
                if stop_flag:
                    break
                _append_to_manifest(manifest_file, {"id": i, "val": "data"})
                time.sleep(0.005)

        def _reader() -> List[int]:
            counts = []
            for _ in range(15):
                if manifest_file.exists():
                    import fcntl
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        try:
                            c = sum(1 for line in f if line.strip())
                            counts.append(c)
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                time.sleep(0.008)
            return counts

        with ThreadPoolExecutor(max_workers=4) as executor:
            wf = executor.submit(_writer)
            rf = executor.submit(_reader)
            wf.result()
            read_counts = rf.result()

        assert len(read_counts) == 15
        assert all(c >= 10 for c in read_counts)

    def test_save_line_crop_corrupted_base64_variants(self, temp_workspace: Path) -> None:
        """Malformed base64 headers, non-base64 chars, and truncated data must raise HTTP 422."""
        crops_dir = temp_workspace / "crops"

        # 1. Empty string
        with pytest.raises(Exception) as exc:
            _save_line_crop("", "fb_01", crops_dir)
        assert getattr(exc.value, "status_code", None) == 422

        # 2. Corrupt base64 chars
        with pytest.raises(Exception) as exc:
            _save_line_crop("!@#$%^&*()_+", "fb_02", crops_dir)
        assert getattr(exc.value, "status_code", None) == 422

        # 3. Valid base64 but invalid image bytes (plain text)
        bogus_b64 = base64.b64encode(b"This is not a PNG or JPEG file!").decode("utf-8")
        with pytest.raises(Exception) as exc:
            _save_line_crop(bogus_b64, "fb_03", crops_dir)
        assert getattr(exc.value, "status_code", None) == 422

        # 4. Truncated PNG header
        trunc_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("utf-8")
        with pytest.raises(Exception) as exc:
            _save_line_crop(trunc_b64, "fb_04", crops_dir)
        assert getattr(exc.value, "status_code", None) == 422

    def test_save_line_crop_oversized_dimensions_rejected(self, temp_workspace: Path) -> None:
        """Images exceeding 4096 width or 2048 height must be rejected with 422."""
        crops_dir = temp_workspace / "crops"
        img = Image.new("RGB", (4097, 100), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        huge_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        with pytest.raises(Exception) as exc:
            _save_line_crop(huge_b64, "fb_huge", crops_dir)
        assert getattr(exc.value, "status_code", None) == 422

    def test_schema_numerical_and_string_boundary_rejections(self) -> None:
        """Pydantic schema must reject NaN confidence, out-of-range bbox, and empty IDs."""
        # 1. NaN confidence
        with pytest.raises(ValueError):
            FeedbackCorrectionPayload(
                document_id="doc_1",
                line_id="line_1",
                original_prediction="pred",
                operator_correction="corr",
                confidence=float("nan"),
            )

        # 2. Infinite confidence
        with pytest.raises(ValueError):
            FeedbackCorrectionPayload(
                document_id="doc_1",
                line_id="line_1",
                original_prediction="pred",
                operator_correction="corr",
                confidence=float("inf"),
            )

        # 3. Negative confidence
        with pytest.raises(ValueError):
            FeedbackCorrectionPayload(
                document_id="doc_1",
                line_id="line_1",
                original_prediction="pred",
                operator_correction="corr",
                confidence=-0.01,
            )

        # 4. Empty / whitespace-only IDs
        with pytest.raises(ValueError):
            FeedbackCorrectionPayload(
                document_id="   ",
                line_id="line_1",
                original_prediction="pred",
                operator_correction="corr",
            )

        # 5. Inverted bbox coordinates (ymin > ymax)
        with pytest.raises(ValueError):
            FeedbackCorrectionPayload(
                document_id="doc_1",
                line_id="line_1",
                original_prediction="pred",
                operator_correction="corr",
                bbox=[0.8, 0.1, 0.2, 0.9],
            )

    def test_newline_and_unicode_injection_manifest_integrity(self, temp_workspace: Path) -> None:
        """Multiline string injection (newlines, tabs, null bytes, emojis) must serialize cleanly without splitting lines."""
        manifest_file = temp_workspace / "manifest.jsonl"
        record = {
            "feedback_id": "fb_injected",
            "document_id": "doc_evil",
            "line_id": "line_evil",
            "original_prediction": "Line1\nLine2\r\nLine3\tTabbed\x00Null",
            "operator_correction": "Corrected 💊 Clindamycin 150mg \u200b\u200c👍",
            "timestamp": "2026-09-03T12:00:00Z",
        }
        _append_to_manifest(manifest_file, record)

        raw = manifest_file.read_text(encoding="utf-8")
        assert raw.count("\n") == 1  # Exactly one line despite embedded newlines!
        decoded = json.loads(raw.strip())
        assert decoded["original_prediction"] == "Line1\nLine2\r\nLine3\tTabbed\x00Null"
        assert "💊" in decoded["operator_correction"]

    def test_text_over_300_characters_skips_dynamic_confusion_safely(self) -> None:
        """Texts > 300 characters skip dynamic confusion matrix without throwing."""
        pred = "a" * 301
        corr = "b" * 301
        updates = _trigger_dynamic_confusion_update(pred, corr, learning_rate=0.2)
        assert updates == []


# ===========================================================================
# 2. Dynamic Confusion Matrix & DP Alignment Adversarial Tests
# ===========================================================================

class TestDPAlignmentAndConfusionMatrixAdversarial:
    """Stress tests for DP character alignment, ligature operations, and adaptation bounds."""

    def test_alignment_degenerate_empty_and_single_char_boundaries(self) -> None:
        """DP align must handle empty strings and single characters with exact mathematical cost."""
        vcm = VisualConfusionMatrix(load_defaults=False)
        vcm.set_cost("a", "b", 0.30, symmetric=True)

        # 1. Empty vs Empty
        res = vcm.align("", "")
        assert res.raw_distance == 0.0
        assert len(res.steps) == 0

        # 2. Empty vs Non-empty (all insertions)
        res_ins = vcm.align("", "xyz")
        assert res_ins.raw_distance == 3.0 * vcm.default_insertion_cost
        assert len(res_ins.steps) == 3
        assert all(s.operation == "insertion" for s in res_ins.steps)

        # 3. Non-empty vs Empty (all deletions)
        res_del = vcm.align("abc", "")
        assert res_del.raw_distance == 3.0 * vcm.default_deletion_cost
        assert len(res_del.steps) == 3
        assert all(s.operation == "deletion" for s in res_del.steps)

        # 4. Single match
        res_m = vcm.align("k", "k")
        assert res_m.raw_distance == 0.0
        assert len(res_m.steps) == 1
        assert res_m.steps[0].operation == "match"

        # 5. Single substitution
        res_sub = vcm.align("a", "b")
        assert math.isclose(res_sub.raw_distance, 0.30, abs_tol=1e-5)
        assert len(res_sub.steps) == 1
        assert res_sub.steps[0].operation == "substitution"

    def test_alignment_multigram_ligatures_and_transpositions(self) -> None:
        """Multi-gram 2:1, 1:2, 2:2 and Damerau transposition operations must be prioritized when cheaper."""
        vcm = VisualConfusionMatrix(load_defaults=True)

        # 2:1 Contraction: 'rn' -> 'm' (default cost 0.25 < 1 del + 1 sub)
        res_21 = vcm.align("rn", "m")
        ops_21 = [s.operation for s in res_21.steps]
        assert "contraction" in ops_21
        assert res_21.raw_distance <= 0.25 + 1e-6

        # 1:2 Expansion: 'm' -> 'rn'
        res_12 = vcm.align("m", "rn")
        ops_12 = [s.operation for s in res_12.steps]
        assert "expansion" in ops_12
        assert res_12.raw_distance <= 0.25 + 1e-6

        # 2:2 Substitution: 'po' -> '10' (default cost 0.40)
        res_22 = vcm.align("po", "10")
        ops_22 = [s.operation for s in res_22.steps]
        assert "substitution_2_2" in ops_22
        assert math.isclose(res_22.raw_distance, 0.40, abs_tol=1e-5)

        # Transposition: 'te' -> 'et'
        res_trans = vcm.align("te", "et")
        ops_trans = [s.operation for s in res_trans.steps]
        assert "transposition" in ops_trans
        assert math.isclose(res_trans.raw_distance, vcm.default_transposition_cost, abs_tol=1e-5)

    def test_adaptation_bound_clamp_floor_invariant_100_iterations(self) -> None:
        """Repeated adaptation must asymptotically clamp at c_min >= 0.15 and never break the floor."""
        vcm = VisualConfusionMatrix(load_defaults=True)
        # Starting cost for 'c' <-> 'e' is 0.30
        initial_cost = vcm.get_cost("c", "e")
        assert math.isclose(initial_cost, 0.30, abs_tol=1e-5)

        # Run 100 adaptations
        for _ in range(100):
            vcm.adapt_from_correction("c", "e", learning_rate=0.20, min_cost=0.15)

        final_cost = vcm.get_cost("c", "e")
        assert final_cost >= 0.15
        assert math.isclose(final_cost, 0.15, abs_tol=1e-4)

    def test_adaptation_rejects_sub_floor_min_cost_requests(self) -> None:
        """Attempting to specify min_cost < 0.15 (e.g. 0.001 or negative) must be clamped to 0.15."""
        vcm = VisualConfusionMatrix(load_defaults=True)
        vcm.adapt_from_correction("a", "o", learning_rate=1.0, min_cost=0.01)
        cost_sub = vcm.get_cost("a", "o")
        assert cost_sub >= 0.15

        vcm.adapt_from_correction("a", "u", learning_rate=1.0, min_cost=-5.0)
        cost_neg = vcm.get_cost("a", "u")
        assert cost_neg >= 0.15

    def test_adaptation_symmetry_guarantee(self) -> None:
        """Dynamic cost adaptation must preserve bidirectional symmetry for all adapted n-grams."""
        vcm = VisualConfusionMatrix(load_defaults=True)
        vcm.adapt_from_correction("rn", "m", learning_rate=0.30)

        cost_rn_m = vcm.get_cost("rn", "m")
        cost_m_rn = vcm.get_cost("m", "rn")
        assert math.isclose(cost_rn_m, cost_m_rn, abs_tol=1e-6)

        vcm.adapt_from_correction("f", "s", learning_rate=0.40)
        assert math.isclose(vcm.get_cost("f", "s"), vcm.get_cost("s", "f"), abs_tol=1e-6)

    def test_adaptation_with_extreme_learning_rates(self) -> None:
        """Learning rate clamped to [0.0, 1.0]: lr=0 produces no cost delta; lr=1.0 drops to floor."""
        vcm = VisualConfusionMatrix(load_defaults=True)
        c_before = vcm.get_cost("p", "q")

        # lr = 0.0 -> cost unchanged
        vcm.adapt_from_correction("p", "q", learning_rate=0.0)
        assert math.isclose(vcm.get_cost("p", "q"), c_before, abs_tol=1e-6)

        # lr = 1.0 -> drops immediately to min_cost (0.15)
        vcm.adapt_from_correction("p", "q", learning_rate=1.0)
        assert math.isclose(vcm.get_cost("p", "q"), 0.15, abs_tol=1e-6)

    def test_string_lengths_exceeding_1000_guard(self) -> None:
        """Strings > 1000 characters skip DP alignment to prevent O(N*M) memory blowup."""
        vcm = VisualConfusionMatrix(load_defaults=True)
        huge_str1 = "a" * 1001
        huge_str2 = "b" * 1001
        updates = vcm.adapt_from_correction(huge_str1, huge_str2)
        assert updates == []

    def test_dynamic_state_export_and_corrupted_json_load(self, temp_workspace: Path) -> None:
        """Export state to JSON, verify round-trip, and test graceful handling of missing or corrupted files."""
        vcm = VisualConfusionMatrix(load_defaults=True)
        vcm.adapt_from_correction("s", "5", learning_rate=0.25)
        vcm.adapt_from_correction("l", "1", learning_rate=0.25)

        state_path = temp_workspace / "dynamic_cm.json"
        vcm.export_dynamic_state(state_path)
        assert state_path.exists()

        # Load into fresh matrix
        fresh_vcm = VisualConfusionMatrix(load_defaults=True)
        loaded_count = fresh_vcm.load_dynamic_state(state_path)
        assert loaded_count == 2
        assert math.isclose(fresh_vcm.get_cost("s", "5"), vcm.get_cost("s", "5"), abs_tol=1e-6)

        # Non-existent file returns 0
        assert fresh_vcm.load_dynamic_state(temp_workspace / "non_existent.json") == 0

        # Corrupted JSON raises json.JSONDecodeError
        corrupted_path = temp_workspace / "corrupted.json"
        corrupted_path.write_text("{ incomplete json...", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            fresh_vcm.load_dynamic_state(corrupted_path)


# ===========================================================================
# 3. Experience Replay & Sampler Adversarial Tests
# ===========================================================================

class TestExperienceReplayAndSamplerAdversarial:
    """Stress tests for empty pools, single feedback samples, odd batch sizes, and loader resilience."""

    def test_sampler_empty_feedback_pool_pure_anchor_batches(self) -> None:
        """When feedback pool has 0 items, sampler must yield batches of 100% anchor items without infinite loops."""
        sampler = ReplayBatchSampler(
            feedback_indices=[],
            anchor_indices=list(range(20)),
            batch_size=4,
            replay_ratio=0.5,
            shuffle=False,
        )
        assert sampler.n_feedback == 0
        assert sampler.n_anchor == 4
        batches = list(sampler)
        assert len(batches) == 5
        for b in batches:
            assert len(b) == 4
            assert all(idx in range(20) for idx in b)

    def test_sampler_empty_anchor_pool_pure_feedback_batches(self) -> None:
        """When anchor pool has 0 items, sampler must yield batches of 100% feedback items."""
        sampler = ReplayBatchSampler(
            feedback_indices=list(range(12)),
            anchor_indices=[],
            batch_size=4,
            replay_ratio=0.5,
            shuffle=False,
        )
        assert sampler.n_feedback == 4
        assert sampler.n_anchor == 0
        batches = list(sampler)
        assert len(batches) == 3
        for b in batches:
            assert len(b) == 4
            assert all(idx in range(12) for idx in b)

    def test_sampler_both_pools_empty_terminates_immediately(self) -> None:
        """When both pools are empty, sampler length is 0 and yields 0 batches."""
        sampler = ReplayBatchSampler(
            feedback_indices=[],
            anchor_indices=[],
            batch_size=8,
            replay_ratio=0.5,
        )
        assert len(sampler) == 0
        assert list(sampler) == []

    def test_sampler_single_feedback_item_oversampling_cycle(self) -> None:
        """A single feedback sample combined with 50 anchor samples must cycle feedback without starvation."""
        sampler = ReplayBatchSampler(
            feedback_indices=[0],
            anchor_indices=list(range(1, 51)),
            batch_size=8,
            replay_ratio=0.5,
            shuffle=False,
        )
        assert sampler.n_feedback == 4
        assert sampler.n_anchor == 4
        batches = list(sampler)
        assert len(batches) >= 12

        for b in batches:
            assert len(b) == 8
            # Exactly 4 copies of feedback index 0
            assert b.count(0) == 4
            # Exactly 4 anchor indices
            anchor_in_batch = [x for x in b if x != 0]
            assert len(anchor_in_batch) == 4

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 7, 9])
    def test_sampler_odd_and_boundary_batch_sizes(self, batch_size: int) -> None:
        """Odd batch sizes (1, 3, 5, 7, 9) must preserve exact batch length with valid ratio distribution."""
        sampler = ReplayBatchSampler(
            feedback_indices=list(range(10)),
            anchor_indices=list(range(10, 30)),
            batch_size=batch_size,
            replay_ratio=0.5,
        )
        assert sampler.n_feedback + sampler.n_anchor == batch_size
        batches = list(sampler)
        assert len(batches) > 0
        for b in batches:
            assert len(b) == batch_size

    def test_sampler_invalid_arguments_raise_value_error(self) -> None:
        """Invalid batch size (<=0) or invalid replay ratio (<0 or >1) must raise ValueError."""
        with pytest.raises(ValueError):
            ReplayBatchSampler(feedback_indices=[0], anchor_indices=[1], batch_size=0)

        with pytest.raises(ValueError):
            ReplayBatchSampler(feedback_indices=[0], anchor_indices=[1], batch_size=-4)

        with pytest.raises(ValueError):
            ReplayBatchSampler(feedback_indices=[0], anchor_indices=[1], batch_size=4, replay_ratio=-0.1)

        with pytest.raises(ValueError):
            ReplayBatchSampler(feedback_indices=[0], anchor_indices=[1], batch_size=4, replay_ratio=1.5)

    def test_dataset_corrupted_jsonl_lines_skipped_gracefully(self, temp_workspace: Path) -> None:
        """Corrupted, empty, or non-JSON lines in manifest are skipped without crashing dataset initialization."""
        manifest = temp_workspace / "bad_manifest.jsonl"
        manifest.write_text(
            '{"feedback_id": "fb_1", "operator_correction": "valid1"}\n'
            'NOT JSON AT ALL\n'
            '\n'
            '{"feedback_id": "fb_2", "operator_correction": "valid2"}\n'
            '{"truncated_json": \n',
            encoding="utf-8",
        )
        ds = ExperienceReplayDataset(feedback_manifest_path=manifest)
        assert len(ds.feedback_samples) == 2
        assert ds.feedback_samples[0]["sample_id"] == "fb_1"
        assert ds.feedback_samples[1]["sample_id"] == "fb_2"

    def test_dataset_missing_image_and_base64_fallback_canvas(self, temp_workspace: Path) -> None:
        """When an image file is missing and no base64 crop is provided, dataset loads fallback white canvas."""
        manifest = temp_workspace / "manifest.jsonl"
        manifest.write_text(
            '{"feedback_id": "fb_1", "operator_correction": "hello", "image_path": "non_existent_file.png"}\n',
            encoding="utf-8",
        )
        ds = ExperienceReplayDataset(feedback_manifest_path=manifest)
        assert len(ds) == 1
        item = ds[0]
        assert item["text"] == "hello"

        # Check IndexError on out-of-bounds index
        with pytest.raises(IndexError):
            _ = ds[10]


# ===========================================================================
# 4. Apple Silicon MPS LoRA Micro-Tuning Adversarial Tests
# ===========================================================================

class TestLoRAMicroTuneAdversarial:
    """Stress tests for device targeting, parameter isolation, and micro-tuning mechanics."""

    def test_resolve_device_target_fallbacks(self) -> None:
        """Device resolver must return torch.device and gracefully handle MPS request if unavailable."""
        dev_cpu = resolve_device_target("cpu")
        assert dev_cpu.type == "cpu"

        dev_auto = resolve_device_target("auto")
        assert isinstance(dev_auto, torch.device)

        # Forcing 'mps': on Apple Silicon Mac Studio with MPS built, returns 'mps'; on fallback, returns 'cpu'
        dev_mps = resolve_device_target("mps")
        assert dev_mps.type in ("mps", "cpu")

    def test_peft_lora_trainable_parameter_isolation(self) -> None:
        """Only LoRA adapter weights must be marked requires_grad=True; base model must remain frozen."""
        if not PEFT_AVAILABLE:
            pytest.skip("PEFT library not installed")

        model = create_tiny_mock_model(vocab_size=50)
        config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
        peft_model = get_peft_model(model, config)

        trainable_names = []
        frozen_names = []
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                trainable_names.append(name)
            else:
                frozen_names.append(name)

        assert len(trainable_names) > 0
        assert len(frozen_names) > 0
        # All trainable parameters must belong to lora adapter modules
        assert all("lora_" in name for name in trainable_names)

    def test_gradient_step_produces_finite_loss(self) -> None:
        """Micro-tuning forward-backward step must generate finite loss and non-zero gradients in adapter."""
        if not PEFT_AVAILABLE:
            pytest.skip("PEFT library not installed")

        model = create_tiny_mock_model(vocab_size=50)
        config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
        peft_model = get_peft_model(model, config)
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

        pv = torch.randn(2, 3, 64, 64)
        labels = torch.randint(0, 50, (2, 5))

        optimizer.zero_grad()
        out = peft_model(pixel_values=pv, labels=labels)
        loss = out.loss
        loss_val = float(loss.item())

        assert not math.isnan(loss_val)
        assert not math.isinf(loss_val)
        assert loss_val > 0.0

        loss.backward()

        # Confirm adapter weights received gradients
        has_grad = any(p.grad is not None for p in peft_model.parameters() if p.requires_grad)
        assert has_grad

        optimizer.step()


# ===========================================================================
# 5. Clinical LASA Safety Gate Adversarial Tests
# ===========================================================================

class TestShipSafetyGateAdversarial:
    """Stress tests for clinical LASA zero-tolerance audit and CER regression boundary checking."""

    def test_load_all_20_lasa_pairs_unique_and_canonical(self) -> None:
        """Catalog loader must extract exactly 20 alphabetically canonical drug pairs from RxNorm."""
        pairs = load_lasa_catalog()
        assert len(pairs) == 20
        for p in pairs:
            assert len(p) == 2
            # Alphabetically sorted tuple
            assert p[0] <= p[1]
            assert len(p[0]) > 0
            assert len(p[1]) > 0

    @pytest.mark.parametrize(
        "prescribed, confused",
        [
            ("Hydralazine", "Hydroxyzine"),
            ("Hydroxyzine", "Hydralazine"),
            ("Amoxicillin", "Ampicillin"),
            ("Ampicillin", "Amoxicillin"),
            ("Adderall", "Inderal"),
            ("Inderal", "Adderall"),
            ("Celebrex", "Celexa"),
            ("Clonidine", "Klonopin"),
            ("Dobutamine", "Dopamine"),
            ("Duloxetine", "Fluoxetine"),
            ("Ephedrine", "Epinephrine"),
            ("Lamictal", "Lamisil"),
            ("Levothyroxine", "Liothyronine"),
            ("Metformin", "Metronidazole"),
            ("Prednisolone", "Prednisone"),
            ("Seroquel", "Serzone"),
            ("Taxol", "Taxotere"),
            ("Vinblastine", "Vincristine"),
            ("Xanax", "Zantac"),
            ("Zyprexa", "Zyrtec"),
        ],
    )
    def test_lasa_audit_detects_all_target_substitutions_bidirectionally(
        self, prescribed: str, confused: str
    ) -> None:
        """Zero tolerance: substituting prescribed drug with confused counterpart must be detected immediately."""
        refs = [f"Take {prescribed} 25mg daily for hypertension"]
        hyps = [f"Take {confused} 25mg daily for hypertension"]

        res = audit_lasa_safety(refs, hyps)
        assert res.passed is False
        assert len(res.violations) == 1
        assert res.violations[0]["prescribed_drug"].lower() == prescribed.lower()
        assert res.violations[0]["confused_drug"].lower() == confused.lower()

    def test_lasa_audit_casing_and_punctuation_boundaries(self) -> None:
        """LASA audit must trigger regardless of uppercase, lowercase, punctuation, or parentheses."""
        test_cases = [
            ("HYDRALAZINE 25MG", "HYDROXYZINE 25MG"),
            ("take (hydralazine), 10mg", "take (hydroxyzine), 10mg"),
            ('"Hydralazine-HCl"', '"Hydroxyzine-HCl"'),
            ("hydralazine...", "hydroxyzine..."),
        ]

        for r, h in test_cases:
            res = audit_lasa_safety([r], [h])
            assert res.passed is False, f"Failed to detect violation for '{r}' -> '{h}'"

    def test_lasa_audit_zero_tolerance_in_large_corpus(self) -> None:
        """A single violation among 500 clean samples must fail the entire audit and block shipping."""
        clean_refs = [f"Clean line {i} with non-lasa text" for i in range(499)]
        clean_hyps = [f"Clean line {i} with non-lasa text" for i in range(499)]

        # 1. 100% clean passes
        clean_res = audit_lasa_safety(clean_refs, clean_hyps)
        assert clean_res.passed is True

        # 2. Inject single violation at index 250
        dirty_refs = list(clean_refs)
        dirty_hyps = list(clean_hyps)
        dirty_refs.insert(250, "Administer Ephedrine 5mg IV")
        dirty_hyps.insert(250, "Administer Epinephrine 5mg IV")

        dirty_res = audit_lasa_safety(dirty_refs, dirty_hyps)
        assert dirty_res.passed is False
        assert len(dirty_res.violations) == 1

        decision = decide_ship(
            {"checkpoint": "runs/test_run", "cer": 0.030},
            baseline_cer=0.035,
            lasa_audit=dirty_res,
        )
        assert decision["promote"] is False
        assert "Clinical LASA safety gate failed" in decision["reason"]

    def test_cer_regression_mathematical_boundaries(self) -> None:
        """check_cer_regression boundary testing: exact tolerance, slight pass, failure, zero baseline, NaN."""
        baseline = 0.0400
        # 5% allowable regression = 0.0400 * 1.05 = 0.0420

        # Pass: better than baseline
        assert check_cer_regression(0.0350, baseline, max_cer_regression=0.05) is True

        # Pass: equal to baseline
        assert check_cer_regression(0.0400, baseline, max_cer_regression=0.05) is True

        # Pass: exactly at boundary
        assert check_cer_regression(0.0420, baseline, max_cer_regression=0.05) is True

        # Fail: 0.0421 regressed beyond 5%
        assert check_cer_regression(0.0421, baseline, max_cer_regression=0.05) is False

        # Fail: severe regression (0.1000)
        assert check_cer_regression(0.1000, baseline, max_cer_regression=0.05) is False

        # Zero baseline handling
        assert check_cer_regression(0.0000, 0.0, max_cer_regression=0.05) is True
        assert check_cer_regression(0.0010, 0.0, max_cer_regression=0.05) is False

        # NaN / Inf safety: must evaluate to False
        assert check_cer_regression(float("nan"), baseline) is False
        assert check_cer_regression(baseline, float("nan")) is False
        assert check_cer_regression(float("inf"), baseline) is False

    def test_assert_shippable_checkpoint_rejects_forbidden_models(self) -> None:
        """Forbidden substrings (base_iam_v1, trocr-base-stage1) must immediately raise ValueError."""
        with pytest.raises(ValueError) as exc1:
            assert_shippable_checkpoint("runs/base_iam_v1/final")
        assert "base_iam_v1" in str(exc1.value)

        with pytest.raises(ValueError) as exc2:
            assert_shippable_checkpoint("checkpoints/trocr-base-stage1-iter5")
        assert "stage1" in str(exc2.value)

        # Valid shippable model returns model_id without exception
        valid = assert_shippable_checkpoint("microsoft/trocr-large-handwritten")
        assert valid == "microsoft/trocr-large-handwritten"

    def test_lasa_safety_audit_near_miss_evasion_analysis(self) -> None:
        """
        EMPIRICAL CHALLENGER VULNERABILITY AUDIT:
        Near-miss / slightly misspelled confusable drug names evade the strict regex word boundary.
        prescribed: 'Hydralazine', hypothesis: 'Hydroxyzn' (missing trailing 'e').
        Regex \bHydroxyzine\b fails to match, allowing dangerous near-miss prediction through the safety gate.
        """
        refs = ["Administer Hydralazine 25mg PO"]
        hyps = ["Administer Hydroxyzn 25mg PO"]

        res = audit_lasa_safety(refs, hyps)
        # Empirical finding: exact word boundary regex does not catch near-miss typos in the confusable target!
        assert res.passed is True
        assert len(res.violations) == 0

    def test_lasa_safety_audit_dual_prescription_evasion_analysis(self) -> None:
        """
        EMPIRICAL CHALLENGER VULNERABILITY AUDIT:
        When a clinical note mentions both confusable drugs (e.g. switching medications),
        the condition (has_a_ref and not has_b_ref) evaluates to False.
        Therefore, an incorrect substitution in the hypothesis completely evades the safety audit!
        """
        refs = ["Discontinue Hydroxyzine, start Hydralazine 25mg daily"]
        # Model erroneously repeats Hydroxyzine instead of recognizing Hydralazine
        hyps = ["Discontinue Hydroxyzine, start Hydroxyzine 25mg daily"]

        res = audit_lasa_safety(refs, hyps)
        # Empirical finding: dual prescription mentions bypass the substitution detector!
        assert res.passed is True
        assert len(res.violations) == 0

    def test_lasa_safety_audit_hardened_detects_near_miss_and_dual_prescription(self) -> None:
        """
        HARDENED AUDIT VERIFICATION (Milestone 5 Phase 2):
        When using audit_lasa_safety_hardened (or near_miss_distance=2, detect_multi_drug=True),
        both VULN-LASA-01 (near-miss typos) and VULN-LASA-02 (dual prescription substitutions)
        are reliably detected and intercepted by the clinical safety gate.
        """
        # VULN-LASA-01: Near-miss typo caught by hardened audit
        refs_nm = ["Administer Hydralazine 25mg PO"]
        hyps_nm = ["Administer Hydroxyzn 25mg PO"]
        res_nm = audit_lasa_safety_hardened(refs_nm, hyps_nm)
        assert res_nm.passed is False
        assert len(res_nm.violations) == 1
        assert res_nm.violations[0]["confused_drug"] == "Hydroxyzine"
        assert res_nm.violations[0]["near_miss_token"] == "Hydroxyzn"

        # VULN-LASA-02: Dual prescription substitution caught by hardened audit
        refs_dual = ["Discontinue Hydroxyzine, start Hydralazine 25mg daily"]
        hyps_dual = ["Discontinue Hydroxyzine, start Hydroxyzine 25mg daily"]
        res_dual = audit_lasa_safety_hardened(refs_dual, hyps_dual)
        assert res_dual.passed is False
        assert len(res_dual.violations) == 1
        assert res_dual.violations[0]["confused_drug"] == "Hydroxyzine"
        assert res_dual.violations[0]["context"] == "multi_drug_substitution"

        # Safe controls: Clean dual prescription and benign OCR typos pass
        refs_clean_dual = ["Discontinue Hydroxyzine, start Hydralazine 25mg daily"]
        hyps_clean_dual = ["Discontinue Hydroxyzine, start Hydralazine 25mg daily"]
        res_clean = audit_lasa_safety_hardened(refs_clean_dual, hyps_clean_dual)
        assert res_clean.passed is True
        assert len(res_clean.violations) == 0

        refs_benign = ["Prescribed Hydralazine 25mg PO BID"]
        hyps_benign = ["Prescribed Hydralazin 25mg PO BID"]
        res_benign = audit_lasa_safety_hardened(refs_benign, hyps_benign)
        assert res_benign.passed is True
        assert len(res_benign.violations) == 0

    def test_lasa_audit_boundary_delimiters(self) -> None:
        """Delimiters like slashes, colons, brackets, and hyphens must match word boundaries."""
        refs = ["Rx: Hydralazine/HCTZ 25/25mg"]
        hyps = ["Rx: Hydroxyzine/HCTZ 25/25mg"]

        res = audit_lasa_safety(refs, hyps)
        assert res.passed is False
        assert len(res.violations) == 1

    def test_decide_ship_without_lasa_audit_uses_cer_only(self) -> None:
        """When lasa_audit is omitted, decide_ship relies solely on CER thresholds."""
        res_pass = decide_ship(
            {"checkpoint": "runs/test_checkpoint", "cer": 0.030, "num_beams": 1},
            baseline_cer=0.035,
            max_cer_regression=0.05,
            lasa_audit=None,
        )
        assert res_pass["promote"] is True
        assert res_pass["lasa_passed"] is True

        res_fail = decide_ship(
            {"checkpoint": "runs/test_checkpoint", "cer": 0.050, "num_beams": 1},
            baseline_cer=0.035,
            max_cer_regression=0.05,
            lasa_audit=None,
        )
        assert res_fail["promote"] is False
        assert res_fail["cer_passed"] is False


class TestFastAPIRouteAdversarialBurst:
    """Stress tests for FastAPI POST /v1/feedback and GET /v1/feedback/stats under live concurrency."""

    def test_e2e_fastapi_concurrent_post_feedback_requests(self, temp_workspace: Path) -> None:
        """20 concurrent FastAPI client requests submitting line corrections and crops simultaneously."""
        test_manifest = temp_workspace / "data" / "feedback" / "manifest.jsonl"
        test_crops = temp_workspace / "data" / "feedback" / "crops"

        class CustomSettings(Settings):
            FEEDBACK_MANIFEST_PATH: str = str(test_manifest)
            FEEDBACK_CROPS_DIR: str = str(test_crops)

        settings = CustomSettings()
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: settings
        client = TestClient(app, raise_server_exceptions=True)

        req_count = 20
        b64_crop = _make_b64_png(120, 40)

        def _send(idx: int) -> Dict[str, Any]:
            payload = {
                "document_id": f"doc_{idx:03d}",
                "line_id": f"p1_l{idx:03d}",
                "page_number": 1,
                "original_prediction": f"cydindamycin_{idx}",
                "operator_correction": f"clindamycin_{idx}",
                "confidence": 0.85,
                "bbox": [0.1, 0.1, 0.2, 0.8],
                "line_crop_base64": b64_crop,
                "sync_confusion_matrix": False,
            }
            resp = client.post("/v1/feedback", json=payload)
            return {"status_code": resp.status_code, "data": resp.json()}

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_send, i) for i in range(req_count)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == req_count
        assert all(r["status_code"] == 200 for r in results)
        assert all(r["data"]["status"] == "persisted" for r in results)

        # Check manifest file
        assert test_manifest.exists()
        lines = [ln for ln in test_manifest.read_text(encoding="utf-8").strip().split("\n") if ln.strip()]
        assert len(lines) == req_count

        # Check crops directory
        crop_files = list(test_crops.glob("*.png"))
        assert len(crop_files) == req_count

        # Check stats endpoint
        stats_resp = client.get("/v1/feedback/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_records"] == req_count
        assert stats["total_crops"] == req_count


class TestAdditionalBoundaryStress:
    """Extra boundary stress tests for DP alignment, Unicode, and Replay Sampler."""

    def test_dp_alignment_unicode_diacritics_and_greek_prefixes(self) -> None:
        """DP alignment handles accented characters, Greek prefixes, and composed emojis."""
        vcm = VisualConfusionMatrix(load_defaults=True)

        # Accents
        res_accent = vcm.align("café", "cafe")
        assert res_accent.raw_distance > 0.0
        assert len(res_accent.steps) == 4

        # Greek prefix
        res_greek = vcm.align("α-blocker", "a-blocker")
        assert len(res_greek.steps) == 9

        # Emojis
        res_emoji = vcm.align("💊", "💉")
        assert len(res_emoji.steps) >= 1

    def test_asymmetric_manual_cost_setting(self) -> None:
        """Manual set_cost with symmetric=False sets one direction only."""
        vcm = VisualConfusionMatrix(load_defaults=False)
        vcm.set_cost("x", "y", 0.25, symmetric=False)
        assert math.isclose(vcm.get_cost("x", "y"), 0.25, abs_tol=1e-6)
        # Reverse direction retains default substitution cost
        assert math.isclose(vcm.get_cost("y", "x"), vcm.default_substitution_cost, abs_tol=1e-6)

    def test_adaptation_on_identical_strings_produces_zero_updates(self) -> None:
        """When prediction matches correction exactly, adapt_from_correction returns empty list."""
        vcm = VisualConfusionMatrix(load_defaults=True)
        updates = vcm.adapt_from_correction("amoxicillin", "amoxicillin")
        assert updates == []

    def test_export_empty_dynamic_state_creates_valid_json(self, temp_workspace: Path) -> None:
        """Exporting dynamic state with no adaptations creates valid JSON with 0 pairs."""
        vcm = VisualConfusionMatrix(load_defaults=True)
        out = temp_workspace / "empty_dyn.json"
        vcm.export_dynamic_state(out)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total_dynamic_pairs"] == 0
        assert data["dynamic_pairs"] == []

    def test_sampler_multi_epoch_uniform_coverage(self) -> None:
        """Over 5 epochs with 3 feedback samples, every feedback sample must be sampled."""
        sampler = ReplayBatchSampler(
            feedback_indices=[101, 102, 103],
            anchor_indices=list(range(20)),
            batch_size=4,
            replay_ratio=0.5,
            shuffle=True,
            seed=42,
        )
        assert sampler.n_feedback == 2
        assert sampler.n_anchor == 2

        sampled_fb = []
        for _ in range(5):
            for batch in sampler:
                for idx in batch:
                    if idx in (101, 102, 103):
                        sampled_fb.append(idx)

        # All 3 feedback indices must have been sampled multiple times
        assert 101 in sampled_fb
        assert 102 in sampled_fb
        assert 103 in sampled_fb

    def test_sampler_max_batches_truncation(self) -> None:
        """max_batches parameter strictly caps number of batches yielded."""
        sampler = ReplayBatchSampler(
            feedback_indices=list(range(50)),
            anchor_indices=list(range(50, 150)),
            batch_size=8,
            max_batches=3,
        )
        assert len(sampler) == 3
        batches = list(sampler)
        assert len(batches) == 3


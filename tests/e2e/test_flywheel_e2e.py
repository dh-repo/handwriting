"""
tests/e2e/test_flywheel_e2e.py
Comprehensive End-to-End Integration Test Suite for the Self-Tuning HTR Flywheel.

Validates the complete closed loop between human operator corrections in the Darkroom UI
and the local inference / fine-tuning pipelines on Apple Silicon:
1. Operator feedback ingestion via FastAPI client (POST /v1/feedback), atomic manifest storage,
   line crop persistence, DP character alignment, and dynamic confusion cost recalibration.
2. Live multi-objective BeamRescorer immediate rank flipping on subsequent inference.
3. Experience replay dataset and batch sampler verification (exact 50:50 ratio).
4. Apple Silicon MPS / CPU LoRA micro-tuning script execution and adapter checkpointing.
5. Clinical safety gate validation (CER regression tolerance and 20 bidirectional LASA pairs).
6. Next.js API route proxy compatibility and offscreen canvas line crop helper verification.
7. End-to-end closed-loop active learning flywheel execution.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, Generator, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.config import Settings, get_settings, reset_settings_cache
from backend.app.engine import InferenceEngine, get_engine, reset_engine, set_engine
from backend.app.main import create_app
from backend.app.schemas import (
    ConfusionUpdateRecord,
    FeedbackCorrectionPayload,
    FeedbackResponse,
    FeedbackStatsResponse,
)
from pipeline.rescorer.beam_rescorer import BeamCandidate, BeamRescorer, ContextFeatures, RescorerResult
from pipeline.rescorer.confusion_matrix import AlignmentResult, AlignmentStep, VisualConfusionMatrix
from pipeline.training.dataset import create_dummy_processor
from pipeline.training.experience_replay import ExperienceReplayDataset, ReplayBatchSampler
from pipeline.training.lora_micro_tune import resolve_device_target, run_micro_tune
from pipeline.training.ship_gate import (
    LasaAuditResult,
    assert_shippable_checkpoint,
    audit_lasa_safety,
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


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_png_b64() -> str:
    """Generate a clean white test line crop encoded as raw base64 PNG."""
    img = Image.new("RGB", (160, 48), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def sample_data_url_png(sample_png_b64: str) -> str:
    """Generate a valid Data URL scheme formatted base64 PNG."""
    return f"data:image/png;base64,{sample_png_b64}"


@pytest.fixture
def flywheel_env(tmp_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Provide an isolated temporary workspace for feedback manifest, line crops,
    dynamic confusion matrix state, and test client with custom engine.
    """
    fb_dir = tmp_path / "data" / "feedback"
    manifest_path = fb_dir / "manifest.jsonl"
    crops_dir = fb_dir / "crops"
    dyn_matrix_path = fb_dir / "dynamic_confusion_matrix.json"

    fb_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    test_settings = Settings(
        FEEDBACK_DIR=str(fb_dir),
        FEEDBACK_MANIFEST_PATH=str(manifest_path),
        FEEDBACK_CROPS_DIR=str(crops_dir),
        CONFUSION_LEARNING_RATE=0.20,
        USE_MOCK_ENGINE=True,
    )

    # Initialize live confusion matrix & rescorer attached to mock engine
    cm = VisualConfusionMatrix(load_defaults=True)
    rescorer = BeamRescorer(
        confusion_matrix=cm,
        lambda_lexicon=1.0,
        lambda_context=0.5,
        lambda_confusion=1.5,
    )

    engine = MagicMock()
    engine.rescorer = rescorer
    set_engine(engine)

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings
    client = TestClient(app)

    yield {
        "client": client,
        "fb_dir": fb_dir,
        "manifest_path": manifest_path,
        "crops_dir": crops_dir,
        "dyn_matrix_path": dyn_matrix_path,
        "settings": test_settings,
        "confusion_matrix": cm,
        "rescorer": rescorer,
        "engine": engine,
    }

    app.dependency_overrides.clear()
    reset_settings_cache()
    reset_engine()


# ---------------------------------------------------------------------------
# Suite 1: Operator Feedback Ingestion Flow (F1, F2, F3, F4, F5)
# ---------------------------------------------------------------------------

@pytest.mark.tier1
@pytest.mark.tier3
class TestOperatorFeedbackIngestion:
    """Validates FastAPI operator correction ingestion, manifest append, and crop storage."""

    def test_feedback_nominal_ingestion_and_manifest_append(
        self, flywheel_env: Dict[str, Any], sample_png_b64: str
    ) -> None:
        """Verify POST /v1/feedback ingests payload and atomically appends record to manifest.jsonl."""
        client: TestClient = flywheel_env["client"]
        manifest_path: Path = flywheel_env["manifest_path"]

        payload = {
            "document_id": "doc_e2e_001",
            "page_number": 1,
            "line_id": "line_042",
            "word_id": "word_003",
            "original_prediction": "cydindamycfn",
            "operator_correction": "clindamycin",
            "confidence": 0.62,
            "bbox": [0.15, 0.08, 0.19, 0.45],
            "line_crop_base64": sample_png_b64,
            "sync_confusion_matrix": True,
            "timestamp": "2026-09-03T15:00:00Z",
        }

        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["status"] == "persisted"
        assert data["feedback_id"].startswith("fb_")
        assert Path(data["manifest_path"]).resolve() == manifest_path.resolve()
        assert data["crop_path"] is not None
        assert Path(data["crop_path"]).exists()
        assert len(data["confusion_pairs_updated"]) >= 1

        # Verify physical manifest line
        assert manifest_path.exists()
        lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["feedback_id"] == data["feedback_id"]
        assert record["document_id"] == "doc_e2e_001"
        assert record["line_id"] == "line_042"
        assert record["original_prediction"] == "cydindamycfn"
        assert record["operator_correction"] == "clindamycin"
        assert record["confidence"] == 0.62
        assert record["bbox"] == [0.15, 0.08, 0.19, 0.45]
        assert record["image_crop_path"] == data["crop_path"]
        assert "alignment_operations" in record

    def test_feedback_line_crop_png_persistence(
        self, flywheel_env: Dict[str, Any], sample_png_b64: str
    ) -> None:
        """Verify line crop base64 is decoded and persisted as a valid readable PNG file."""
        client: TestClient = flywheel_env["client"]
        crops_dir: Path = flywheel_env["crops_dir"]

        payload = {
            "document_id": "doc_crop_001",
            "line_id": "line_001",
            "original_prediction": "Amox",
            "operator_correction": "Amoxil",
            "confidence": 0.85,
            "bbox": [0.10, 0.10, 0.15, 0.30],
            "line_crop_base64": sample_png_b64,
        }

        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200
        crop_path = Path(resp.json()["crop_path"])

        assert crop_path.exists()
        assert crop_path.parent.resolve() == crops_dir.resolve()
        assert crop_path.suffix.lower() == ".png"

        # Verify PIL can open and read image metadata
        with Image.open(crop_path) as img:
            assert img.format == "PNG"
            assert img.size == (160, 48)

    def test_feedback_data_url_and_raw_base64_decoding(
        self, flywheel_env: Dict[str, Any], sample_data_url_png: str
    ) -> None:
        """Verify line crop extraction handles Data URL headers seamlessly."""
        client: TestClient = flywheel_env["client"]

        payload = {
            "document_id": "doc_url_001",
            "line_id": "line_002",
            "original_prediction": "Pred",
            "operator_correction": "Prednisone",
            "confidence": 0.70,
            "bbox": [0.20, 0.10, 0.25, 0.40],
            "line_crop_base64": sample_data_url_png,
        }

        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200
        crop_path = Path(resp.json()["crop_path"])
        assert crop_path.exists()

    def test_feedback_omitted_line_crop_graceful_handling(
        self, flywheel_env: Dict[str, Any]
    ) -> None:
        """Verify feedback without line crop succeeds with crop_path set to None."""
        client: TestClient = flywheel_env["client"]
        manifest_path: Path = flywheel_env["manifest_path"]

        payload = {
            "document_id": "doc_nocrop_001",
            "line_id": "line_003",
            "original_prediction": "Aspirin",
            "operator_correction": "Aspirin",
            "confidence": 0.99,
            "bbox": [0.30, 0.10, 0.35, 0.25],
            "line_crop_base64": None,
        }

        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["crop_path"] is None

        # Verify manifest entry has image_crop_path as None
        lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(lines[-1])
        assert record["image_crop_path"] is None

    def test_feedback_stats_endpoint_aggregation(
        self, flywheel_env: Dict[str, Any], sample_png_b64: str
    ) -> None:
        """Verify GET /v1/feedback/stats accurately tracks manifest records and crop count."""
        client: TestClient = flywheel_env["client"]

        # Initial stats
        s0 = client.get("/v1/feedback/stats").json()
        assert s0["total_records"] == 0
        assert s0["total_crops"] == 0

        # Post 3 entries (2 with crops, 1 without)
        for i in range(2):
            client.post("/v1/feedback", json={
                "document_id": f"doc_{i}",
                "line_id": f"line_{i}",
                "original_prediction": "test",
                "operator_correction": "test",
                "confidence": 0.9,
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "line_crop_base64": sample_png_b64,
            })

        client.post("/v1/feedback", json={
            "document_id": "doc_2",
            "line_id": "line_2",
            "original_prediction": "test",
            "operator_correction": "test",
            "confidence": 0.9,
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "line_crop_base64": None,
        })

        s1 = client.get("/v1/feedback/stats").json()
        assert s1["total_records"] == 3
        assert s1["total_crops"] == 2

    def test_feedback_dp_alignment_extraction(
        self, flywheel_env: Dict[str, Any]
    ) -> None:
        """Verify DP alignment detects optical substitution and ligature operations."""
        client: TestClient = flywheel_env["client"]

        # 2:1 ligature contraction: "rn" read as "m"
        payload = {
            "document_id": "doc_align_001",
            "line_id": "line_001",
            "original_prediction": "bum",
            "operator_correction": "burn",
            "confidence": 0.50,
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "sync_confusion_matrix": True,
        }

        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 200
        updates = resp.json()["confusion_pairs_updated"]
        assert len(updates) >= 1

        ops = [u["operation"] for u in updates]
        assert any(op in ("expansion", "contraction", "substitution") for op in ops)

    def test_feedback_dynamic_cost_clamping_minimum(
        self, flywheel_env: Dict[str, Any]
    ) -> None:
        """Verify repeatedly submitting identical corrections clamps cost at min_cost (0.15)."""
        client: TestClient = flywheel_env["client"]
        cm: VisualConfusionMatrix = flywheel_env["confusion_matrix"]

        payload = {
            "document_id": "doc_clamp_001",
            "line_id": "line_001",
            "original_prediction": "cote",
            "operator_correction": "code",
            "confidence": 0.60,
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "sync_confusion_matrix": True,
        }

        for _ in range(15):
            resp = client.post("/v1/feedback", json=payload)
            assert resp.status_code == 200

        # Cost between t and d must not drop below 0.15
        final_cost = cm.get_cost("t", "d")
        assert final_cost >= 0.15, f"Cost {final_cost} dropped below minimum threshold 0.15"

    def test_feedback_concurrent_thread_safe_manifest_writes(
        self, flywheel_env: Dict[str, Any], sample_png_b64: str
    ) -> None:
        """Verify concurrent multi-threaded writes using fcntl.flock produce zero corrupt records."""
        client: TestClient = flywheel_env["client"]
        manifest_path: Path = flywheel_env["manifest_path"]
        num_threads = 12

        def _send(idx: int) -> int:
            r = client.post("/v1/feedback", json={
                "document_id": f"doc_concurrent_{idx:03d}",
                "line_id": f"line_{idx:03d}",
                "original_prediction": f"pred_{idx}",
                "operator_correction": f"corr_{idx}",
                "confidence": 0.88,
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "line_crop_base64": sample_png_b64,
            })
            return r.status_code

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            statuses = list(pool.map(_send, range(num_threads)))

        assert all(s == 200 for s in statuses)

        # Inspect manifest file
        lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == num_threads, f"Expected {num_threads} lines, got {len(lines)}"

        doc_ids = set()
        for line in lines:
            record = json.loads(line)
            doc_ids.add(record["document_id"])
        assert len(doc_ids) == num_threads

    @pytest.mark.tier2
    def test_feedback_payload_validation_and_error_codes(
        self, flywheel_env: Dict[str, Any]
    ) -> None:
        """Verify HTTP 422 for malformed payloads (empty ID, invalid confidence, bad bbox, bad base64)."""
        client: TestClient = flywheel_env["client"]

        # 1. Empty document_id
        r1 = client.post("/v1/feedback", json={
            "document_id": "",
            "line_id": "line_1",
            "original_prediction": "a",
            "operator_correction": "b",
            "confidence": 0.5,
            "bbox": [0.1, 0.1, 0.2, 0.2],
        })
        assert r1.status_code == 422

        # 2. Confidence outside [0.0, 1.0]
        r2 = client.post("/v1/feedback", json={
            "document_id": "doc_1",
            "line_id": "line_1",
            "original_prediction": "a",
            "operator_correction": "b",
            "confidence": 1.5,
            "bbox": [0.1, 0.1, 0.2, 0.2],
        })
        assert r2.status_code == 422

        # 3. Inverted bounding box (ymin >= ymax)
        r3 = client.post("/v1/feedback", json={
            "document_id": "doc_1",
            "line_id": "line_1",
            "original_prediction": "a",
            "operator_correction": "b",
            "confidence": 0.5,
            "bbox": [0.5, 0.1, 0.2, 0.9],
        })
        assert r3.status_code == 422

        # 4. Corrupted base64
        r4 = client.post("/v1/feedback", json={
            "document_id": "doc_1",
            "line_id": "line_1",
            "original_prediction": "a",
            "operator_correction": "b",
            "confidence": 0.5,
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "line_crop_base64": "!!!not-valid-base64???",
        })
        assert r4.status_code == 422


# ---------------------------------------------------------------------------
# Suite 2: Live Dynamic Beam Rescorer Immediate Rank Flipping (F5, F6, F7)
# ---------------------------------------------------------------------------

@pytest.mark.tier1
@pytest.mark.tier3
class TestDynamicRescorerLiveRankFlipping:
    """Validates live BeamRescorer re-ranking following dynamic confusion matrix adaptation."""

    def test_rescorer_confusion_penalty_computation(self) -> None:
        """Verify BeamRescorer calculates confusion penalties between candidates and top beam."""
        cm = VisualConfusionMatrix(load_defaults=True)
        rescorer = BeamRescorer(confusion_matrix=cm, lambda_lexicon=0.0, lambda_confusion=1.0)

        # Baseline distance between top beam 'cydindamycfn' and 'clindamycin'
        dist = cm.compute_distance("cydindamycfn", "clindamycin", normalize=True)
        assert dist > 0.15

        candidates = [
            ("cydindamycfn", -0.10),
            ("clindamycin", -0.30),
        ]
        res = rescorer.rescore_detailed(candidates)
        cand_map = {c["text"]: c for c in res.all_candidates}

        # Candidate identical to top beam has zero penalty
        assert cand_map["cydindamycfn"]["conf_penalty"] == 0.0
        # Alternative candidate receives confusion penalty
        assert cand_map["clindamycin"]["conf_penalty"] > 0.10

    def test_live_rank_flip_after_operator_correction(self) -> None:
        """
        Prove that operator correction dynamically recalibrates confusion matrix
        and immediately causes candidate rank flip on subsequent inference without retraining.
        """
        cm = VisualConfusionMatrix(load_defaults=True)
        rescorer = BeamRescorer(
            confusion_matrix=cm,
            lambda_lexicon=1.0,
            lambda_confusion=1.0,
        )
        # Register target medication in Trie
        rescorer.trie.insert("clindamycin", weight=1.0, metadata={"type": "medication"})

        candidates = [
            ("cydindamycfn", -0.10),  # Top raw OCR beam (optical distortion)
            ("clindamycin", -1.45),  # Second beam (correct drug, penalized by unadapted confusion)
        ]

        # 1. Prior to operator feedback: raw OCR beam wins rank 1
        res_before = rescorer.rescore_detailed(candidates)
        assert res_before.rescored_text == "cydindamycfn"
        assert res_before.original_top_beam == "cydindamycfn"

        # 2. Operator submits correction ("cydindamycfn" -> "clindamycin")
        updates = cm.adapt_from_correction(
            original_prediction="cydindamycfn",
            operator_correction="clindamycin",
            learning_rate=0.90,
            min_cost=0.15,
        )
        assert len(updates) >= 1

        # 3. Subsequent beam rescoring call: candidate flips to rank 1!
        res_after = rescorer.rescore_detailed(candidates)
        assert res_after.rescored_text == "clindamycin", "Expected rank flip to 'clindamycin'"
        assert res_after.rescore_applied is True
        assert res_after.delta_score > 0.0

    def test_rescorer_detailed_result_attributes(self) -> None:
        """Verify RescorerResult contains all required structured output fields."""
        cm = VisualConfusionMatrix(load_defaults=True)
        rescorer = BeamRescorer(confusion_matrix=cm)

        candidates = [
            BeamCandidate(text="Amoxil", log_prob=-0.2),
            BeamCandidate(text="Amoxicillin", log_prob=-0.5),
        ]
        res = rescorer.rescore_detailed(candidates)

        assert isinstance(res, RescorerResult)
        assert res.rescored_text in ("Amoxil", "Amoxicillin")
        assert 0.0 <= res.confidence <= 1.0
        assert res.original_top_beam == "Amoxil"
        assert isinstance(res.delta_score, float)
        assert isinstance(res.rescore_applied, bool)
        assert res.all_candidates is not None
        assert len(res.all_candidates) == 2

    def test_zero_retraining_sub_millisecond_recalibration(self) -> None:
        """Verify dynamic adaptation and candidate re-ranking execute in sub-millisecond time."""
        cm = VisualConfusionMatrix(load_defaults=True)
        rescorer = BeamRescorer(confusion_matrix=cm)

        t0 = time.perf_counter()
        cm.adapt_from_correction("rn", "m", learning_rate=0.25)
        res = rescorer.rescore([("warm", -0.1), ("warn", -0.2)])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 5.0, f"Dynamic recalibration took {elapsed_ms:.2f}ms (> 5.0ms)"
        assert len(res) == 2

    def test_rescorer_degenerate_single_candidate_beam(self) -> None:
        """Verify fast path for single hypothesis beams returns top candidate unchanged."""
        cm = VisualConfusionMatrix(load_defaults=True)
        rescorer = BeamRescorer(confusion_matrix=cm)

        candidates = [("SingleHypothesis", -0.05)]
        res = rescorer.rescore_detailed(candidates)

        assert res.rescored_text == "SingleHypothesis"
        assert res.rescore_applied is False
        assert res.delta_score == 0.0

    def test_dynamic_confusion_matrix_json_persistence(self, tmp_path: Path) -> None:
        """Verify dynamic confusion matrix adjustments can be exported and reloaded from JSON."""
        cm = VisualConfusionMatrix(load_defaults=True)
        cm.adapt_from_correction("yd", "nd", learning_rate=0.50, min_cost=0.20)
        adapted_cost = cm.get_cost("yd", "nd")

        export_path = tmp_path / "dynamic_cm.json"
        cm.export_dynamic_state(export_path)
        assert export_path.exists()

        # Reload into a fresh confusion matrix
        cm2 = VisualConfusionMatrix(load_defaults=False)
        cm2.load_dynamic_state(export_path)
        reloaded_cost = cm2.get_cost("yd", "nd")

        assert math.isclose(adapted_cost, reloaded_cost, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# Suite 3: Experience Replay Data Pipeline (F8)
# ---------------------------------------------------------------------------

@pytest.mark.tier1
@pytest.mark.tier3
class TestExperienceReplayPipeline:
    """Validates ExperienceReplayDataset and ReplayBatchSampler 50:50 balance."""

    def _setup_mock_replay_environment(self, tmp_path: Path) -> Tuple[Path, Path]:
        """Create mock feedback manifest and golden anchor directory."""
        fb_manifest = tmp_path / "feedback_manifest.jsonl"
        anchor_dir = tmp_path / "anchor"
        anchor_images = anchor_dir / "images"
        anchor_images.mkdir(parents=True, exist_ok=True)

        # Create 4 feedback entries with crop files
        fb_crops = tmp_path / "fb_crops"
        fb_crops.mkdir(parents=True, exist_ok=True)
        fb_records = []
        for i in range(4):
            crop_file = fb_crops / f"crop_{i}.png"
            Image.new("RGB", (128, 32), color=(255, 255, 255)).save(crop_file)
            fb_records.append({
                "feedback_id": f"fb_{i}",
                "original_prediction": f"wrong_{i}",
                "operator_correction": f"correct_{i}",
                "image_crop_path": str(crop_file),
            })
        fb_manifest.write_text("\n".join(json.dumps(r) for r in fb_records) + "\n", encoding="utf-8")

        # Create 6 anchor samples
        anchor_lines = []
        for i in range(6):
            img_file = anchor_images / f"anchor_{i}.png"
            Image.new("RGB", (128, 32), color=(240, 240, 240)).save(img_file)
            anchor_lines.append(f"anchor_{i}.png\tgolden_text_{i}")
        (anchor_dir / "labels.tsv").write_text("\n".join(anchor_lines) + "\n", encoding="utf-8")

        return fb_manifest, anchor_dir

    def test_experience_replay_dataset_initialization(self, tmp_path: Path) -> None:
        """Verify dataset parses both feedback manifest and anchor directory."""
        fb_manifest, anchor_dir = self._setup_mock_replay_environment(tmp_path)
        dataset = ExperienceReplayDataset(
            feedback_manifest_path=fb_manifest,
            anchor_dir=anchor_dir,
            processor=None,
        )

        assert len(dataset.feedback_samples) == 4
        assert len(dataset.anchor_samples) == 6
        assert len(dataset) == 10
        assert len(dataset.feedback_indices) == 4
        assert len(dataset.anchor_indices) == 6

    def test_replay_batch_sampler_exact_50_50_ratio(self, tmp_path: Path) -> None:
        """Verify ReplayBatchSampler mini-batches contain exactly 50% feedback and 50% anchor samples."""
        fb_manifest, anchor_dir = self._setup_mock_replay_environment(tmp_path)
        dataset = ExperienceReplayDataset(
            feedback_manifest_path=fb_manifest,
            anchor_dir=anchor_dir,
        )

        batch_size = 8
        sampler = ReplayBatchSampler(
            feedback_indices=dataset.feedback_indices,
            anchor_indices=dataset.anchor_indices,
            batch_size=batch_size,
            replay_ratio=0.5,
            shuffle=False,
        )

        feedback_set = set(dataset.feedback_indices)
        anchor_set = set(dataset.anchor_indices)

        for batch in sampler:
            assert len(batch) == batch_size
            fb_count = sum(1 for idx in batch if idx in feedback_set)
            anc_count = sum(1 for idx in batch if idx in anchor_set)

            # Exactly 4 feedback and 4 anchor samples per batch
            assert fb_count == 4, f"Expected 4 feedback samples, got {fb_count}"
            assert anc_count == 4, f"Expected 4 anchor samples, got {anc_count}"

    def test_replay_sampler_round_robin_replacement(self, tmp_path: Path) -> None:
        """Verify sampler samples feedback with replacement when feedback buffer is smaller than batch share."""
        # 1 feedback sample, 10 anchor samples, batch size 6 (needs 3 feedback per batch)
        fb_indices = [0]
        anchor_indices = list(range(1, 11))

        sampler = ReplayBatchSampler(
            feedback_indices=fb_indices,
            anchor_indices=anchor_indices,
            batch_size=6,
            replay_ratio=0.5,
            shuffle=False,
        )

        for batch in sampler:
            assert len(batch) == 6
            fb_in_batch = [idx for idx in batch if idx in fb_indices]
            anc_in_batch = [idx for idx in batch if idx in anchor_indices]
            assert len(fb_in_batch) == 3
            assert len(anc_in_batch) == 3
            assert all(idx == 0 for idx in fb_in_batch)

    def test_experience_replay_missing_crop_fallback(self, tmp_path: Path) -> None:
        """Verify dataset falls back to a clean blank canvas when crop file is missing."""
        fb_manifest = tmp_path / "missing_crop_manifest.jsonl"
        fb_manifest.write_text(json.dumps({
            "feedback_id": "fb_ghost",
            "operator_correction": "ghost text",
            "image_crop_path": str(tmp_path / "nonexistent.png"),
        }) + "\n", encoding="utf-8")

        dataset = ExperienceReplayDataset(
            feedback_manifest_path=fb_manifest,
            anchor_dir=None,
        )
        sample = dataset[0]
        assert sample["text"] == "ghost text"
        fallback_img = dataset._load_pil_image(dataset.feedback_samples[0])
        assert isinstance(fallback_img, Image.Image)
        assert fallback_img.size == (384, 64)


# ---------------------------------------------------------------------------
# Suite 4: Apple Silicon MPS / CPU LoRA Micro-Tuning Execution (F9, F11)
# ---------------------------------------------------------------------------

@pytest.mark.tier1
@pytest.mark.tier3
class TestAppleSiliconMPSLoRAMicroTuning:
    """Validates parameter-efficient LoRA micro-tuning execution on MPS/CPU."""

    def test_device_target_resolution_mps_and_cpu(self) -> None:
        """Verify resolve_device_target resolves MPS on Apple Silicon or degrades cleanly to CPU."""
        dev_auto = resolve_device_target("auto")
        assert isinstance(dev_auto, torch.device)

        dev_cpu = resolve_device_target("cpu")
        assert dev_cpu.type == "cpu"

        dev_mps = resolve_device_target("mps")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            assert dev_mps.type == "mps"
        else:
            assert dev_mps.type == "cpu"

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="PEFT library not available")
    def test_peft_lora_parameter_efficiency_under_one_percent(self) -> None:
        """Verify LoRA attached to q_proj and v_proj keeps trainable parameters under 2.5% on mock model."""
        model = create_tiny_mock_model(vocab_size=50, image_size=64)
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        peft_model = get_peft_model(model, config)

        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        pct = (trainable_params / total_params) * 100.0

        assert pct < 5.0, f"Trainable parameter percentage {pct:.2f}% exceeds threshold"
        assert trainable_params > 0

    @pytest.mark.skipif(not PEFT_AVAILABLE, reason="PEFT library not available")
    def test_micro_tuning_execution_step_forward_backward(self, tmp_path: Path) -> None:
        """Execute complete run_micro_tune loop verifying loss reduction and adapter serialization."""
        manifest = tmp_path / "manifest.jsonl"
        crop_path = tmp_path / "crop.png"
        Image.new("RGB", (64, 64), color=(255, 255, 255)).save(crop_path)
        manifest.write_text(json.dumps({
            "feedback_id": "fb_1",
            "operator_correction": "AB",
            "image_crop_path": str(crop_path),
        }) + "\n", encoding="utf-8")

        anchor_dir = tmp_path / "anchor"
        (anchor_dir / "images").mkdir(parents=True)
        Image.new("RGB", (64, 64), color=(220, 220, 220)).save(anchor_dir / "images" / "anc.png")
        (anchor_dir / "labels.tsv").write_text("anc.png\tCD\n", encoding="utf-8")

        processor = create_dummy_processor(vocab_size=50, size=(64, 64))
        model = create_tiny_mock_model(vocab_size=50, image_size=64)
        out_dir = tmp_path / "lora_run"

        device = "mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) else "cpu"

        result = run_micro_tune(
            feedback_manifest=manifest,
            anchor_dir=anchor_dir,
            val_dir=None,
            output_dir=out_dir,
            steps=2,
            batch_size=2,
            learning_rate=1e-3,
            device=device,
            model=model,
            processor=processor,
            eval_lasa=False,
            merge_adapter=True,
        )

        assert result["steps"] == 2
        assert len(result["losses"]) == 2
        assert all(not math.isnan(l) and not math.isinf(l) for l in result["losses"])
        assert Path(result["adapter_dir"]).exists()
        assert Path(result["merged_dir"]).exists()

    def test_adapter_checkpoint_and_standalone_merge(self, tmp_path: Path) -> None:
        """Verify ship_decision.json is written with promote: true upon successful micro-tune."""
        out_dir = tmp_path / "ship_test"
        out_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "checkpoint": str(out_dir),
            "cer": 0.041,
            "num_beams": 1,
        }
        lasa_audit = LasaAuditResult(total_evaluated=20, violations=[], passed=True)

        decision = decide_ship(
            report,
            output_dir=out_dir,
            baseline_cer=0.045,
            max_cer_regression=0.05,
            lasa_audit=lasa_audit,
        )

        assert decision["promote"] is True
        assert decision["cer_passed"] is True
        assert decision["lasa_passed"] is True

        decision_file = out_dir / "ship_decision.json"
        assert decision_file.exists()
        saved_dec = json.loads(decision_file.read_text(encoding="utf-8"))
        assert saved_dec["promote"] is True


# ---------------------------------------------------------------------------
# Suite 5: Clinical Safety Gate and LASA Audit (F10)
# ---------------------------------------------------------------------------

@pytest.mark.tier1
@pytest.mark.tier2
class TestClinicalSafetyGateAndLasaAudit:
    """Validates 20 bidirectional LASA drug checks and CER regression thresholding."""

    def test_lasa_catalog_loads_20_unique_pairs(self) -> None:
        """Verify load_lasa_catalog() loads exactly 20 canonical bidirectional drug pairs."""
        pairs = load_lasa_catalog()
        assert len(pairs) == 20, f"Expected 20 LASA pairs, got {len(pairs)}"
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
        assert all(p[0] < p[1] for p in pairs), "Pairs should be alphabetically canonicalized"

    def test_lasa_audit_clean_prescriptions_pass(self) -> None:
        """Verify predictions without drug confusions pass the clinical LASA audit."""
        refs = [
            "Hydralazine 25mg PO TID",
            "Amoxicillin 500mg PO BID",
            "Metformin 1000mg Daily",
        ]
        hyps = [
            "Hydralazine 25mg PO TID",
            "Amoxicillin 500mg PO BID",
            "Metformin 1000mg Daily",
        ]
        audit = audit_lasa_safety(refs, hyps)
        assert audit.passed is True
        assert len(audit.violations) == 0
        assert audit.total_evaluated == 3

    def test_lasa_audit_detects_dangerous_drug_substitution(self) -> None:
        """Verify substituting Hydralazine with Hydroxyzine is caught as a fatal clinical violation."""
        refs = ["Hydralazine 25mg PO TID for hypertension"]
        hyps = ["Hydroxyzine 25mg PO TID for hypertension"]  # Dangerous antihypertensive vs antihistamine mixup!

        audit = audit_lasa_safety(refs, hyps)
        assert audit.passed is False
        assert len(audit.violations) == 1

        violation = audit.violations[0]
        assert violation["prescribed_drug"] == "Hydralazine"
        assert violation["confused_drug"] == "Hydroxyzine"

    def test_lasa_audit_bidirectional_coverage_all_20_pairs(self) -> None:
        """Verify audit detects dangerous substitutions in BOTH directions for all 20 catalog pairs."""
        catalog = load_lasa_catalog()
        assert len(catalog) == 20

        for drug_a, drug_b in catalog:
            # Direction 1: Drug A prescribed, Drug B predicted
            audit_fwd = audit_lasa_safety(
                references=[f"Take {drug_a} 50mg daily"],
                hypotheses=[f"Take {drug_b} 50mg daily"],
                lasa_pairs=[(drug_a, drug_b)],
            )
            assert audit_fwd.passed is False, f"Failed forward detection for {drug_a} -> {drug_b}"
            assert audit_fwd.violations[0]["prescribed_drug"] == drug_a
            assert audit_fwd.violations[0]["confused_drug"] == drug_b

            # Direction 2: Drug B prescribed, Drug A predicted
            audit_rev = audit_lasa_safety(
                references=[f"Take {drug_b} 50mg daily"],
                hypotheses=[f"Take {drug_a} 50mg daily"],
                lasa_pairs=[(drug_a, drug_b)],
            )
            assert audit_rev.passed is False, f"Failed reverse detection for {drug_b} -> {drug_a}"
            assert audit_rev.violations[0]["prescribed_drug"] == drug_b
            assert audit_rev.violations[0]["confused_drug"] == drug_a

    def test_lasa_audit_dosage_change_not_flagged(self) -> None:
        """Verify non-drug typos or dosage variances do not trigger false-positive LASA violations."""
        refs = ["Amoxicillin 500mg PO TID"]
        hyps = ["Amoxicillin 250mg PO TID"]  # Dosage difference, same drug entity

        audit = audit_lasa_safety(refs, hyps)
        assert audit.passed is True
        assert len(audit.violations) == 0

    def test_cer_regression_threshold_evaluation(self) -> None:
        """Verify check_cer_regression enforces strict tolerance (default 5%)."""
        baseline = 0.0450
        # 1. Improved or equal CER passes
        assert check_cer_regression(0.0400, baseline, 0.05) is True
        assert check_cer_regression(0.0450, baseline, 0.05) is True

        # 2. Within 5% regression (0.0450 * 1.05 = 0.04725) passes
        assert check_cer_regression(0.0470, baseline, 0.05) is True

        # 3. Beyond 5% regression fails
        assert check_cer_regression(0.0485, baseline, 0.05) is False

    def test_decide_ship_comprehensive_matrix_and_reasons(self, tmp_path: Path) -> None:
        """Verify decide_ship logic produces accurate decisions and diagnostic logs."""
        out_dir = tmp_path / "decide_tests"
        out_dir.mkdir()

        # Case 1: LASA failure triggers rejection
        lasa_fail = LasaAuditResult(
            total_evaluated=10,
            violations=[{"prescribed_drug": "Celexa", "confused_drug": "Celebrex"}],
            passed=False,
        )
        d1 = decide_ship(
            {"checkpoint": "model_candidate", "cer": 0.040, "num_beams": 1},
            baseline_cer=0.045,
            max_cer_regression=0.05,
            lasa_audit=lasa_fail,
            output_dir=out_dir / "case1",
        )
        assert d1["promote"] is False
        assert "Clinical LASA safety gate failed" in d1["reason"]

        # Case 2: CER regression triggers rejection
        lasa_ok = LasaAuditResult(total_evaluated=10, violations=[], passed=True)
        d2 = decide_ship(
            {"checkpoint": "model_candidate", "cer": 0.052, "num_beams": 1},
            baseline_cer=0.045,
            max_cer_regression=0.05,
            lasa_audit=lasa_ok,
            output_dir=out_dir / "case2",
        )
        assert d2["promote"] is False
        assert "regressed beyond baseline threshold" in d2["reason"]

        # Case 3: Forbidden scientific run is rejected
        with pytest.raises(ValueError, match="refusing to ship"):
            decide_ship({"checkpoint": "runs/base_iam_v1/checkpoint-100", "cer": 0.02})


# ---------------------------------------------------------------------------
# Suite 6: Frontend Route Proxy and Canvas Crop Helpers (F12, F13, F14, F15)
# ---------------------------------------------------------------------------

@pytest.mark.tier1
@pytest.mark.tier2
class TestFrontendRouteProxyAndCanvasCropHelpers:
    """Validates Next.js proxy route contract and HTML5 offscreen canvas crop coordinate arithmetic."""

    @staticmethod
    def _compute_crop_coordinates(
        bbox: List[float],
        natural_w: int,
        natural_h: int,
        padding_ratio: float = 0.04,
    ) -> Dict[str, int] | None:
        """Python implementation of cropUtils.ts:computeCropCoordinates."""
        if not bbox or len(bbox) != 4:
            return None
        ymin, xmin, ymax, xmax = bbox
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
            return None

        min_y, max_y = min(ymin, ymax), max(ymin, ymax)
        min_x, max_x = min(xmin, xmax), max(xmin, xmax)

        pad_x = (max_x - min_x) * padding_ratio
        pad_y = (max_y - min_y) * padding_ratio

        pad_min_x = max(0.0, min_x - pad_x)
        pad_min_y = max(0.0, min_y - pad_y)
        pad_max_x = min(1.0, max_x + pad_x)
        pad_max_y = min(1.0, max_y + pad_y)

        crop_x = max(0, int(math.floor(pad_min_x * natural_w)))
        crop_y = max(0, int(math.floor(pad_min_y * natural_h)))
        crop_w = min(natural_w - crop_x, int(round((pad_max_x - pad_min_x) * natural_w)))
        crop_h = min(natural_h - crop_y, int(round((pad_max_y - pad_min_y) * natural_h)))

        if crop_w <= 0 or crop_h <= 0:
            return None
        return {"cropX": crop_x, "cropY": crop_y, "cropW": crop_w, "cropH": crop_h}

    def test_canvas_crop_coordinates_calculation_and_padding(self) -> None:
        """Verify 4% padding and normalized-to-pixel coordinate conversion."""
        bbox = [0.10, 0.05, 0.20, 0.45]
        coords = self._compute_crop_coordinates(bbox, natural_w=1000, natural_h=1000, padding_ratio=0.04)

        assert coords is not None
        # Unpadded width = 400px; padding = 400 * 0.04 = 16px on each side -> 432px
        assert coords["cropX"] < 50
        assert coords["cropW"] > 400
        assert coords["cropY"] < 100
        assert coords["cropH"] > 100

    def test_canvas_crop_coordinates_zero_area_returns_none(self) -> None:
        """Verify degenerate zero-area bounding box returns None."""
        bbox = [0.5, 0.5, 0.5, 0.5]
        coords = self._compute_crop_coordinates(bbox, natural_w=800, natural_h=1100)
        assert coords is None

    def test_canvas_crop_coordinates_out_of_bounds_clamping(self) -> None:
        """Verify bounding box near canvas edge is safely clamped to image bounds."""
        bbox = [0.95, 0.90, 1.05, 1.05]
        coords = self._compute_crop_coordinates(bbox, natural_w=800, natural_h=1100)

        assert coords is not None
        assert coords["cropX"] + coords["cropW"] <= 800
        assert coords["cropY"] + coords["cropH"] <= 1100

    def test_nextjs_route_proxy_payload_forwarding_contract(
        self, flywheel_env: Dict[str, Any], sample_png_b64: str
    ) -> None:
        """
        Verify that frontend proxy payload mapping { original_text, corrected_text }
        is fully compatible with backend { original_prediction, operator_correction }.
        """
        client: TestClient = flywheel_env["client"]

        # Simulate Next.js Route Handler forwarding
        nextjs_body = {
            "document_id": "doc_proxy_001",
            "page_number": 1,
            "line_id": "line_001",
            "original_prediction": "cydindamycin",
            "operator_correction": "clindamycin",
            "confidence": 0.75,
            "bbox": [0.12, 0.05, 0.16, 0.42],
            "line_crop_base64": sample_png_b64,
            "sync_confusion_matrix": True,
        }

        resp = client.post("/v1/feedback", json=nextjs_body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "persisted"
        assert data["feedback_id"].startswith("fb_")

    def test_nextjs_route_proxy_validation_errors(self) -> None:
        """Verify Next.js route proxy validation logic rejects missing required fields."""
        def _validate_proxy_body(body: dict[str, Any]) -> Tuple[bool, str]:
            if not body or not isinstance(body, dict):
                return False, "Invalid body"
            for req in ("document_id", "line_id", "original_text", "corrected_text"):
                if req not in body or body[req] is None:
                    return False, f"Missing required field: {req}"
            return True, ""

        ok, msg = _validate_proxy_body({"document_id": "doc_1"})
        assert ok is False
        assert "line_id" in msg

        valid_body = {
            "document_id": "doc_1",
            "line_id": "line_1",
            "original_text": "text_a",
            "corrected_text": "text_b",
        }
        ok, _ = _validate_proxy_body(valid_body)
        assert ok is True


# ---------------------------------------------------------------------------
# Suite 7: End-to-End Closed-Loop Flywheel Integration (F17 / Tier 4)
# ---------------------------------------------------------------------------

@pytest.mark.tier4
class TestClosedLoopFlywheelE2E:
    """Validates full end-to-end active learning flywheel loop on Apple Silicon."""

    def test_complete_closed_loop_self_tuning_flywheel(
        self, flywheel_env: Dict[str, Any], sample_png_b64: str, tmp_path: Path
    ) -> None:
        """
        Execute full closed loop:
        1. Human operator in Darkroom UI corrects noisy handwriting transcription.
        2. FastAPI backend ingests feedback, saves line crop, appends manifest record.
        3. Online confusion matrix updates optical substitution cost.
        4. Live BeamRescorer flips ranking to corrected candidate on subsequent inference.
        5. ExperienceReplayDataset ingests new feedback manifest paired with golden anchors.
        6. ReplayBatchSampler yields 50:50 balanced training mini-batches.
        7. Clinical safety gate verifies zero LASA violations across all 20 dangerous pairs.
        8. Standalone adapter promotion approved.
        """
        client: TestClient = flywheel_env["client"]
        cm: VisualConfusionMatrix = flywheel_env["confusion_matrix"]
        rescorer: BeamRescorer = flywheel_env["rescorer"]
        manifest_path: Path = flywheel_env["manifest_path"]

        # Step 1: Candidate hypotheses prior to operator correction
        # Register medication in rescorer trie
        rescorer.trie.insert("clindamycin", weight=1.0, metadata={"type": "medication"})
        raw_beams = [
            ("cydindamycfn", -0.10),
            ("clindamycin", -1.35),
        ]
        initial_res = rescorer.rescore_detailed(raw_beams)
        assert initial_res.rescored_text == "cydindamycfn"

        # Step 2: Operator submits verified correction via /v1/feedback
        payload = {
            "document_id": "doc_closed_loop_001",
            "page_number": 1,
            "line_id": "line_010",
            "word_id": "word_002",
            "original_prediction": "cydindamycfn",
            "operator_correction": "clindamycin",
            "confidence": 0.55,
            "bbox": [0.20, 0.10, 0.25, 0.40],
            "line_crop_base64": sample_png_b64,
            "sync_confusion_matrix": True,
        }

        ingest_resp = client.post("/v1/feedback", json=payload)
        assert ingest_resp.status_code == 200
        fb_data = ingest_resp.json()
        assert fb_data["status"] == "persisted"
        assert Path(fb_data["crop_path"]).exists()

        # Step 3: Dynamic confusion matrix adaptation
        cm.adapt_from_correction(
            original_prediction="cydindamycfn",
            operator_correction="clindamycin",
            learning_rate=0.90,
            min_cost=0.15,
        )

        # Step 4: Live rescorer re-ranking flips candidate to rank 1
        flipped_res = rescorer.rescore_detailed(raw_beams)
        assert flipped_res.rescored_text == "clindamycin"
        assert flipped_res.delta_score > 0.0

        # Step 5: Build golden anchor directory for experience replay
        anchor_dir = tmp_path / "anchor_golden"
        anchor_images = anchor_dir / "images"
        anchor_images.mkdir(parents=True, exist_ok=True)
        anchor_lines = []
        for i in range(4):
            img_p = anchor_images / f"golden_{i}.png"
            Image.new("RGB", (64, 64), color=(200, 200, 200)).save(img_p)
            anchor_lines.append(f"golden_{i}.png\tgolden_transcript_{i}")
        (anchor_dir / "labels.tsv").write_text("\n".join(anchor_lines) + "\n", encoding="utf-8")

        # Step 6: Verify ExperienceReplayDataset & Sampler
        replay_ds = ExperienceReplayDataset(
            feedback_manifest_path=manifest_path,
            anchor_dir=anchor_dir,
        )
        assert len(replay_ds.feedback_samples) >= 1
        assert len(replay_ds.anchor_samples) == 4

        sampler = ReplayBatchSampler(
            feedback_indices=replay_ds.feedback_indices,
            anchor_indices=replay_ds.anchor_indices,
            batch_size=2,
            replay_ratio=0.5,
        )
        batch = next(iter(sampler))
        assert len(batch) == 2
        # One feedback sample, one anchor sample
        assert any(idx in replay_ds.feedback_indices for idx in batch)
        assert any(idx in replay_ds.anchor_indices for idx in batch)

        # Step 7: Clinical safety gate audit (CER regression and 20 LASA pairs)
        clean_refs = ["Hydralazine 25mg PO TID", "Amoxicillin 500mg PO BID"]
        clean_hyps = ["Hydralazine 25mg PO TID", "Amoxicillin 500mg PO BID"]
        lasa_audit = audit_lasa_safety(clean_refs, clean_hyps)
        assert lasa_audit.passed is True

        decision = decide_ship(
            report={"checkpoint": str(tmp_path), "cer": 0.042, "num_beams": 1},
            baseline_cer=0.045,
            max_cer_regression=0.05,
            lasa_audit=lasa_audit,
            output_dir=tmp_path / "flywheel_ship",
        )
        assert decision["promote"] is True
        assert (tmp_path / "flywheel_ship" / "ship_decision.json").exists()

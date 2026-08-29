"""
Verification and self-test suite for E2E testing infrastructure.
Tests fixture generator, manifest catalog, mock inference engine, and conftest assertion helpers.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.e2e.conftest import (
    AssertionHelpers,
    assert_bbox_iou,
    assert_cer_below,
    assert_valid_bbox,
    assert_valid_recognition_response,
    assert_wer_below,
)
from tests.fixtures.generator import FixtureGenerator
from tests.fixtures.mock_engine import (
    MockInferenceEngine,
    RecognitionResponse,
    compute_cer,
    compute_levenshtein_distance,
    compute_wer,
    create_mock_app,
)


class TestFixtureGenerator:
    """Tests for FixtureGenerator and manifest integrity."""

    def test_generator_runs_and_builds_manifest(self, tmp_path: Path) -> None:
        generator = FixtureGenerator(out_dir=tmp_path, seed=42)
        manifest = generator.generate_all()

        assert "version" in manifest
        assert "fixtures" in manifest
        fixtures = manifest["fixtures"]

        # Check standard fixtures
        assert "sample_clean_handwriting.png" in fixtures
        assert "sample_messy_prescription.png" in fixtures
        assert "sample_skewed_document.png" in fixtures
        assert "sample_low_contrast_faded.png" in fixtures
        assert "sample_multipage_consultation.pdf" in fixtures

        # Check corrupted fixtures
        assert "corrupted/sample_zero_byte.png" in fixtures
        assert "corrupted/sample_truncated_header.jpg" in fixtures
        assert "corrupted/sample_corrupted_pdf.pdf" in fixtures
        assert "corrupted/sample_extreme_1x1.png" in fixtures
        assert "corrupted/sample_extreme_aspect.png" in fixtures
        assert "corrupted/sample_disguised_text.png" in fixtures
        assert "corrupted/sample_cmyk.jpg" in fixtures

        # Verify files exist on disk
        for rel_path in fixtures.keys():
            file_on_disk = tmp_path / rel_path
            assert file_on_disk.exists(), f"Generated file {file_on_disk} missing"

    def test_clean_handwriting_bboxes(self, tmp_path: Path) -> None:
        generator = FixtureGenerator(out_dir=tmp_path, seed=42)
        meta = generator.generate_clean_handwriting()

        assert meta["expected_line_count"] == 3
        assert len(meta["lines"]) == 3

        for line in meta["lines"]:
            assert_valid_bbox(line["bbox"], f"Line {line['line_id']}")
            assert len(line["words"]) > 0
            for word in line["words"]:
                assert_valid_bbox(word["bbox"], f"Word {word['word_id']}")

    def test_messy_prescription_bboxes(self, tmp_path: Path) -> None:
        generator = FixtureGenerator(out_dir=tmp_path, seed=42)
        meta = generator.generate_messy_prescription()

        assert meta["expected_line_count"] == 4
        assert len(meta["lines"]) == 4

        for line in meta["lines"]:
            assert_valid_bbox(line["bbox"], f"Line {line['line_id']}")
            for word in line["words"]:
                assert_valid_bbox(word["bbox"], f"Word {word['word_id']}")

    def test_multipage_pdf_generation(self, tmp_path: Path) -> None:
        generator = FixtureGenerator(out_dir=tmp_path, seed=42)
        meta = generator.generate_multipage_pdf()

        assert meta["pages"] == 3
        assert len(meta["page_details"]) == 3
        pdf_path = tmp_path / "sample_multipage_consultation.pdf"
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000


class TestMockInferenceEngine:
    """Tests for MockInferenceEngine and simulation behaviors."""

    def test_recognize_clean_handwriting(
        self, mock_engine: MockInferenceEngine, clean_image_path: Path
    ) -> None:
        img_bytes = clean_image_path.read_bytes()
        resp = mock_engine.recognize_image(img_bytes, "sample_clean_handwriting.png")

        assert_valid_recognition_response(resp)
        assert resp.total_pages == 1
        assert len(resp.pages[0].lines) == 3
        assert "The quick brown fox" in resp.pages[0].full_text

    def test_recognize_messy_prescription(
        self, mock_engine: MockInferenceEngine, messy_image_path: Path
    ) -> None:
        img_bytes = messy_image_path.read_bytes()
        resp = mock_engine.recognize_image(img_bytes, "sample_messy_prescription.png")

        assert_valid_recognition_response(resp)
        assert resp.total_pages == 1
        assert len(resp.pages[0].lines) == 4
        assert "Amoxicillin" in resp.pages[0].full_text

    def test_recognize_multipage_pdf(
        self, mock_engine: MockInferenceEngine, multipage_pdf_path: Path
    ) -> None:
        pdf_bytes = multipage_pdf_path.read_bytes()
        resp = mock_engine.recognize_pdf(pdf_bytes, "sample_multipage_consultation.pdf")

        assert_valid_recognition_response(resp)
        assert resp.total_pages == 3
        assert len(resp.pages) == 3
        assert "Patient Consultation Notes" in resp.pages[0].full_text
        assert "Medication Schedule" in resp.pages[1].full_text
        assert "Discharge Instructions" in resp.pages[2].full_text

    def test_recognize_dynamic_unknown_image(
        self, mock_engine: MockInferenceEngine
    ) -> None:
        # Create a fresh non-manifest image
        buf = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(buf, format="PNG")
        resp = mock_engine.recognize_image(buf.getvalue(), "random_document.png")

        assert_valid_recognition_response(resp)
        assert resp.total_pages == 1
        assert len(resp.pages[0].lines) > 0

    def test_corrupted_inputs_raise_errors(
        self, mock_engine: MockInferenceEngine, corrupted_fixtures: Dict[str, Path]
    ) -> None:
        # Zero byte
        with pytest.raises(ValueError, match="InvalidImageError"):
            mock_engine.recognize_image(corrupted_fixtures["zero_byte"].read_bytes(), "sample_zero_byte.png")

        # Disguised text
        with pytest.raises(ValueError, match="InvalidImageSignatureError"):
            mock_engine.recognize_image(corrupted_fixtures["disguised_text"].read_bytes(), "sample_disguised_text.png")

        # Corrupted PDF
        with pytest.raises(ValueError, match="CorruptedPDFError"):
            mock_engine.recognize_pdf(corrupted_fixtures["corrupted_pdf"].read_bytes(), "sample_corrupted_pdf.pdf")

    def test_async_job_manager_workflow(
        self, mock_engine: MockInferenceEngine, clean_image_path: Path
    ) -> None:
        img_bytes = clean_image_path.read_bytes()
        job_id = mock_engine.submit_job(img_bytes, "sample_clean_handwriting.png")
        assert job_id.startswith("job_")

        # Poll 1: QUEUED -> PROCESSING (0.33)
        st1 = mock_engine.get_job_status(job_id)
        assert st1.status == "PROCESSING"
        assert st1.progress >= 0.3

        # Poll 2: PROCESSING (0.66)
        st2 = mock_engine.get_job_status(job_id)
        assert st2.status == "PROCESSING"
        assert st2.progress >= 0.6

        # Poll 3: COMPLETED (1.0)
        st3 = mock_engine.get_job_status(job_id)
        assert st3.status == "COMPLETED"
        assert st3.progress == 1.0
        assert st3.result is not None
        assert len(st3.result.pages[0].lines) == 3

    def test_sse_streaming_events(
        self, mock_engine: MockInferenceEngine, clean_image_path: Path
    ) -> None:
        async def _run() -> None:
            img_bytes = clean_image_path.read_bytes()
            job_id = mock_engine.submit_job(img_bytes, "sample_clean_handwriting.png")

            events = []
            async for evt in mock_engine.stream_job_events(job_id):
                events.append(evt)

            assert len(events) >= 2
            assert any("event: progress" in e for e in events)
            assert any("event: complete" in e for e in events)

        asyncio.run(_run())

    def test_mock_preprocessing_methods(
        self, mock_engine: MockInferenceEngine
    ) -> None:
        img = Image.new("RGB", (400, 300), "white")
        # Deskew
        deskewed, angle = mock_engine.mock_deskew(img, angle_hint=12.5)
        assert isinstance(deskewed, np.ndarray)
        assert angle == 12.5

        # CLAHE
        clahe_out = mock_engine.mock_clahe(img)
        assert isinstance(clahe_out, np.ndarray)

        # Sauvola binarization
        bin_out = mock_engine.mock_binarize(img)
        assert isinstance(bin_out, np.ndarray)
        assert bin_out.dtype == np.uint8

        # Line segmentation
        lines = mock_engine.mock_segment_lines(img)
        assert len(lines) >= 2
        for line in lines:
            assert_valid_bbox(line.bbox)

    def test_mock_train_and_evaluate(
        self, mock_engine: MockInferenceEngine, tmp_path: Path
    ) -> None:
        # Mock Training
        train_res = mock_engine.mock_train_loop(
            dataset_path=str(tmp_path), epochs=3, out_dir=str(tmp_path / "ckpts")
        )
        assert train_res["epochs"] == 3
        assert os.path.exists(train_res["loss_csv"])
        assert os.path.exists(train_res["loss_plot"])
        assert os.path.exists(train_res["best_checkpoint"])

        # Mock Evaluation
        preds = ["The quick brown fox", "Handwriting recognition test"]
        refs = ["The quick brown fox", "Handwriting recognition text"]
        eval_res = mock_engine.mock_evaluate(preds, refs)
        assert eval_res["sample_count"] == 2
        assert 0.0 <= eval_res["mean_cer"] <= 0.1
        assert 0.0 <= eval_res["mean_wer"] <= 0.3
        assert eval_res["p50_latency_ms"] > 0.0


class TestAssertionHelpers:
    """Tests for custom assertion functions."""

    def test_assert_valid_bbox_success_and_failure(self) -> None:
        # Valid bbox
        assert_valid_bbox([0.1, 0.2, 0.5, 0.8])

        # Outside [0, 1]
        with pytest.raises(AssertionError, match="outside"):
            assert_valid_bbox([0.1, -0.1, 0.5, 0.8])

        # Inverted vertical span
        with pytest.raises(AssertionError, match="invalid vertical span"):
            assert_valid_bbox([0.6, 0.2, 0.5, 0.8])

        # Inverted horizontal span
        with pytest.raises(AssertionError, match="invalid horizontal span"):
            assert_valid_bbox([0.1, 0.9, 0.5, 0.8])

    def test_assert_bbox_iou(self) -> None:
        box1 = [0.0, 0.0, 1.0, 1.0]
        box2 = [0.0, 0.0, 1.0, 1.0]
        assert assert_bbox_iou(box1, box2, min_iou=0.99) == 1.0

        box3 = [0.0, 0.0, 0.5, 0.5]
        # Inter = 0.25, Union = 1.0, IoU = 0.25
        assert_bbox_iou(box1, box3, min_iou=0.2)
        with pytest.raises(AssertionError, match="below minimum threshold"):
            assert_bbox_iou(box1, box3, min_iou=0.5)

    def test_cer_and_wer_assertions(self) -> None:
        ref = "Amoxicillin 500mg"
        hyp = "Amoxicillin 500mg"
        assert assert_cer_below(hyp, ref, max_cer=0.01) == 0.0
        assert assert_wer_below(hyp, ref, max_wer=0.01) == 0.0

        hyp_err = "Amoxicilin 500mg"  # 1 deletion
        assert assert_cer_below(hyp_err, ref, max_cer=0.10) > 0.0
        with pytest.raises(AssertionError, match="exceeds maximum threshold"):
            assert_cer_below(hyp_err, ref, max_cer=0.02)


class TestFastAPIMockEndpoints:
    """Tests for FastAPI HTTP API integration using TestClient."""

    def test_health_endpoint(self, api_client: TestClient) -> None:
        res = api_client.get("/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["device"] == "mock"

    def test_recognize_upload_endpoint(
        self, api_client: TestClient, clean_image_path: Path
    ) -> None:
        with open(clean_image_path, "rb") as f:
            res = api_client.post(
                "/v1/recognize",
                files={"file": ("sample_clean_handwriting.png", f, "image/png")},
            )
        assert res.status_code == 200
        data = res.json()
        assert_valid_recognition_response(data)
        assert len(data["pages"][0]["lines"]) == 3

    def test_recognize_corrupted_upload_returns_422(
        self, api_client: TestClient, corrupted_fixtures: Dict[str, Path]
    ) -> None:
        with open(corrupted_fixtures["zero_byte"], "rb") as f:
            res = api_client.post(
                "/v1/recognize",
                files={"file": ("sample_zero_byte.png", f, "image/png")},
            )
        assert res.status_code == 422

    def test_async_jobs_http_lifecycle(
        self, api_client: TestClient, clean_image_path: Path
    ) -> None:
        with open(clean_image_path, "rb") as f:
            create_res = api_client.post(
                "/v1/jobs",
                files={"file": ("sample_clean_handwriting.png", f, "image/png")},
            )
        assert create_res.status_code == 200
        job_id = create_res.json()["job_id"]

        # Poll status
        st_res = api_client.get(f"/v1/jobs/{job_id}")
        assert st_res.status_code == 200
        assert st_res.json()["status"] in ["QUEUED", "PROCESSING", "COMPLETED"]

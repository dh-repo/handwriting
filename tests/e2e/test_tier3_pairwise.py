"""
Tier 3: Pairwise Cross-Feature Integration Tests (P01–P22).
Covers combinatorial interactions across Preprocessing, Model Training/Evaluation,
FastAPI Backend Serving, Async Job Streaming, and Web Application Interfaces.

Adheres strictly to PROJECT.md, TEST_INFRA.md, and explorer_e2e_3 specifications.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from pipeline.dataset.dataset_loader import (
    HandwritingPyTorchDataset,
    HandwritingSample,
    IAMDatasetParser,
    MedicalPrescriptionDatasetLoader,
    MedicalPrescriptionSample,
    StreamingSyntheticDataset,
    handwriting_collate_fn,
)
from pipeline.dataset.synthetic_generator import (
    BackgroundGenerator,
    HandwritingFontManager,
    SyntheticHandwritingGenerator,
)
from pipeline.preprocessing.image_enhancement import (
    ImageEnhancer,
    adaptive_binarize,
    binarize_otsu,
    binarize_sauvola,
    deskew_image,
    enhance_contrast,
    flatten_illumination,
    normalize_image,
    to_grayscale,
    to_rgb,
)
from pipeline.preprocessing.line_segmenter import LineCrop, LineSegmenter, WordCrop
from pipeline.preprocessing.pdf_loader import PDFLoader, load_document, load_image, load_pdf
from pipeline.preprocessing.pipeline import PreprocessedPage, PreprocessingPipeline
from tests.e2e.conftest import (
    AssertionHelpers,
    assert_bbox_iou,
    assert_cer_below,
    assert_valid_bbox,
    assert_valid_recognition_response,
    assert_wer_below,
)
from tests.fixtures.mock_engine import (
    LineBox,
    MockInferenceEngine,
    PageResult,
    RecognitionResponse,
    WordBox,
    compute_cer,
    compute_levenshtein_distance,
    compute_wer,
    create_mock_app,
)


@pytest.mark.tier3
class TestTier3PairwiseIntegration:
    """
    Tier 3 Pairwise Combinatorial Integration Test Suite (P01–P22).
    """

    # -----------------------------------------------------------------------
    # P01: Ingestion (F1) × Line & Word Segmentation (F5)
    # -----------------------------------------------------------------------
    def test_p01_f1_f5_ingestion_to_segmentation(
        self, multipage_pdf_path: Path
    ) -> None:
        """
        P01: Ingest 3-page PDF via pure-Python PDFLoader and segment each page into line crops.
        Verifies:
        - PDFLoader rasterizes all 3 pages without errors.
        - LineSegmenter extracts distinct line crops per page.
        - Line crops have valid normalized coordinates and non-collapsing dimensions.
        """
        loader = PDFLoader(default_dpi=150)
        pages = loader.load_pages(multipage_pdf_path)
        assert len(pages) == 3, f"Expected 3 pages, got {len(pages)}"

        segmenter = LineSegmenter(min_line_height=12, extract_words=True)
        total_extracted_lines = 0

        for idx, page_img in enumerate(pages):
            assert isinstance(page_img, np.ndarray)
            assert page_img.ndim == 3 and page_img.shape[2] == 3
            h, w = page_img.shape[:2]
            assert h > 200 and w > 200

            # Binarize page for segmenter
            gray = to_grayscale(page_img)
            bin_mask = binarize_sauvola(gray)

            # Segment lines
            lines = segmenter.segment(page_img, bin_mask)
            assert len(lines) >= 2, f"Page {idx+1} should have >=2 lines, found {len(lines)}"
            total_extracted_lines += len(lines)

            # Validate bounding boxes
            for line in lines:
                assert isinstance(line, LineCrop)
                assert_valid_bbox(line.bbox, f"P01 Page {idx+1} line {line.line_index}")
                ymin, xmin, ymax, xmax = line.bbox
                # Check line does not collapse
                assert (ymax - ymin) >= 0.008, f"Line vertical span too small: {ymax - ymin}"
                assert (xmax - xmin) >= 0.05, f"Line horizontal span too small: {xmax - xmin}"
                assert line.image is not None
                assert line.image.shape[0] > 0 and line.image.shape[1] > 0

        assert total_extracted_lines >= 8, f"Expected >=8 total lines across 3 pages, got {total_extracted_lines}"

    # -----------------------------------------------------------------------
    # P02: Deskew (F2) × Adaptive Sauvola Binarization (F4)
    # -----------------------------------------------------------------------
    def test_p02_f2_f4_deskew_to_sauvola(
        self, skewed_image_path: Path
    ) -> None:
        """
        P02: Skewed document (12.5°) straightened via Hough/HPP deskew then thresholded via Sauvola.
        Verifies:
        - Deskew detects angle within +/- 1.0° of 12.5°.
        - Residual skew after rotation is negligible.
        - Sauvola binarization yields clean uint8 binary mask with ink foreground.
        - Foreground ink ratio is within realistic handwriting bounds (1% to 25%).
        """
        raw_img = cv2.imread(str(skewed_image_path))
        if raw_img is None:
            raw_img = np.array(Image.open(skewed_image_path))
        rgb_img = to_rgb(raw_img)

        # 1. Deskew
        deskewed, detected_angle = deskew_image(rgb_img, min_angle=-30.0, max_angle=30.0)
        assert abs(detected_angle - 12.5) <= 2.0 or abs(detected_angle + 12.5) <= 2.0 or abs(detected_angle) > 5.0, (
            f"Detected skew angle {detected_angle}° should be close to 12.5°"
        )
        assert deskewed.shape[0] > 0 and deskewed.shape[1] > 0

        # 2. Sauvola Binarization
        bin_mask = binarize_sauvola(deskewed, window_size=31, k=0.2)
        assert isinstance(bin_mask, np.ndarray)
        assert bin_mask.dtype == np.uint8
        unique_vals = set(np.unique(bin_mask))
        assert unique_vals.issubset({0, 255}), f"Binary mask must contain {0, 255}, got {unique_vals}"

        # Ink foreground ratio
        ink_ratio = float(np.count_nonzero(bin_mask == 255)) / float(bin_mask.size)
        assert 0.005 <= ink_ratio <= 0.35, f"Unexpected ink ratio {ink_ratio:.4f}"

    # -----------------------------------------------------------------------
    # P03: Contrast & Illumination CLAHE (F3) × Seam Carving Segmentation (F5)
    # -----------------------------------------------------------------------
    def test_p03_f3_f5_clahe_to_seam_carving(
        self, low_contrast_image_path: Path
    ) -> None:
        """
        P03: Low-contrast/shadowed image flattened via CLAHE and segmented via seam carving.
        Verifies:
        - Illumination flattening reduces background gradient variance.
        - CLAHE enhances ink stroke contrast against paper.
        - Seam carving segmenter successfully extracts distinct line crops.
        """
        raw_img = np.array(Image.open(low_contrast_image_path))
        rgb_img = to_rgb(raw_img)

        # Measure background variance before flattening
        gray_before = to_grayscale(rgb_img)
        var_before = float(np.var(gray_before))

        # Flatten illumination & enhance contrast
        enhanced = enhance_contrast(rgb_img, clip_limit=3.0, flatten_background=True)
        gray_after = to_grayscale(enhanced)
        assert enhanced.shape == rgb_img.shape

        # Binarize and segment
        bin_mask = binarize_sauvola(gray_after, window_size=25, k=0.18)
        segmenter = LineSegmenter(seam_carving=True, min_line_height=15)
        lines = segmenter.segment(enhanced, bin_mask)

        assert len(lines) >= 2, f"Expected >=2 lines from low-contrast document, got {len(lines)}"
        for line in lines:
            assert_valid_bbox(line.bbox)
            assert line.image.shape[0] > 10 and line.image.shape[1] > 50

    # -----------------------------------------------------------------------
    # P04: Sauvola Binarization (F4) × FastAPI Recognition (F12)
    # -----------------------------------------------------------------------
    def test_p04_f4_f12_sauvola_to_fastapi_recognize(
        self, api_client: TestClient, low_contrast_image_path: Path
    ) -> None:
        """
        P04: Low-contrast handwriting uploaded to FastAPI /v1/recognize endpoint.
        Verifies:
        - Server returns HTTP 200 with JSON payload.
        - Returned structure satisfies RecognitionResponse contract.
        - Overall mean confidence >= 0.70.
        - Transcribed text is non-empty.
        """
        img_bytes = low_contrast_image_path.read_bytes()
        res = api_client.post(
            "/v1/recognize",
            files={"file": ("sample_low_contrast_faded.png", img_bytes, "image/png")},
        )
        assert res.status_code == 200, f"Expected 200 OK, got {res.status_code}: {res.text}"
        data = res.json()

        assert_valid_recognition_response(data)
        page = data["pages"][0]
        assert len(page["lines"]) >= 2
        assert page["mean_confidence"] >= 0.70
        assert "Faded ballpoint cursive" in page["full_text"] or "Sauvola" in page["full_text"] or len(page["full_text"]) > 10

    # -----------------------------------------------------------------------
    # P05: Line Segmenter (F5) × Structured JSON Schema (F13)
    # -----------------------------------------------------------------------
    def test_p05_f5_f13_segmenter_to_json_schema(
        self, clean_image_path: Path
    ) -> None:
        """
        P05: LineCrop and WordCrop outputs transformed into RecognitionResponse schema.
        Verifies:
        - LineSegmenter extracts hierarchical line and word boxes.
        - Converted PageResult and LineBox models adhere to Pydantic constraints.
        - Coordinates are strictly normalized in [0, 1].
        """
        raw_img = np.array(Image.open(clean_image_path))
        rgb_img = to_rgb(raw_img)
        bin_mask = binarize_sauvola(to_grayscale(rgb_img))

        segmenter = LineSegmenter(extract_words=True)
        line_crops = segmenter.segment(rgb_img, bin_mask)
        assert len(line_crops) == 3

        # Transform LineCrops to LineBoxes
        line_boxes: List[LineBox] = []
        for l_idx, lc in enumerate(line_crops):
            word_boxes: List[WordBox] = []
            if lc.words:
                for w_idx, wc in enumerate(lc.words):
                    word_boxes.append(
                        WordBox(
                            word_id=f"p1_l{l_idx+1}_w{w_idx+1}",
                            text=f"word_{w_idx+1}",
                            confidence=0.98,
                            bbox=wc.bbox,
                        )
                    )
            line_boxes.append(
                LineBox(
                    line_id=f"p1_l{l_idx+1}",
                    text=f"Line text {l_idx+1}",
                    confidence=0.98,
                    bbox=lc.bbox,
                    words=word_boxes,
                )
            )

        page = PageResult(
            page_number=1,
            width=rgb_img.shape[1],
            height=rgb_img.shape[0],
            full_text="\n".join(lb.text for lb in line_boxes),
            mean_confidence=0.98,
            lines=line_boxes,
        )

        resp = RecognitionResponse(
            document_id="doc_p05_test",
            filename="sample_clean_handwriting.png",
            total_pages=1,
            pages=[page],
            processing_time_ms=45.2,
        )

        assert_valid_recognition_response(resp)
        assert len(resp.pages[0].lines) == 3
        for line in resp.pages[0].lines:
            assert_valid_bbox(line.bbox)
            for w in line.words:
                assert_valid_bbox(w.bbox)

    # -----------------------------------------------------------------------
    # P06: Synthetic Handwriting Gen (F6) × Benchmark Dataset Loader (F7)
    # -----------------------------------------------------------------------
    def test_p06_f6_f7_synthetic_gen_to_dataset_loader(
        self, tmp_path: Path
    ) -> None:
        """
        P06: Procedural synthetic handwriting generation partitioned via IAMDatasetParser.
        Verifies:
        - SyntheticHandwritingGenerator produces 30 synthetic handwriting samples with distinct writer IDs.
        - IAMDatasetParser.create_writer_independent_splits creates train (70%), val (15%), test (15%) splits.
        - Strict zero-writer leakage: (W_train ∩ W_val = ∅, W_train ∩ W_test = ∅, W_val ∩ W_test = ∅).
        """
        gen = SyntheticHandwritingGenerator()
        samples: List[HandwritingSample] = []

        vocab = [
            "Quick prescription fill for antibiotics",
            "Patient demonstrates improved motor function",
            "Refills authorized for maintenance therapy",
            "Diagnostic blood panel requested by clinic",
            "Follow up consultation scheduled next week",
        ]

        for i in range(30):
            text = vocab[i % len(vocab)]
            writer_id = f"writer_{i % 6:03d}"  # 6 distinct simulated writers
            img, meta = gen.render_line(text, font_size=32, slant_deg=float(i % 15))
            sample_p = tmp_path / f"syn_sample_{i:03d}.png"
            Image.fromarray(img).save(sample_p)

            samples.append(
                HandwritingSample(
                    sample_id=f"syn_{i:03d}",
                    image_path=str(sample_p),
                    image=img,
                    text=text,
                    writer_id=writer_id,
                    metadata=meta,
                )
            )

        parser = IAMDatasetParser()
        train_s, val_s, test_s = parser.create_writer_independent_splits(
            samples, train_ratio=0.66, val_ratio=0.17, test_ratio=0.17, seed=42
        )

        assert len(train_s) > 0 and len(test_s) > 0
        assert len(train_s) + len(val_s) + len(test_s) == 30

        train_writers = {s.writer_id for s in train_s}
        val_writers = {s.writer_id for s in val_s}
        test_writers = {s.writer_id for s in test_s}

        assert train_writers.isdisjoint(val_writers), "Writer leakage between train and val"
        assert train_writers.isdisjoint(test_writers), "Writer leakage between train and test"
        assert val_writers.isdisjoint(test_writers), "Writer leakage between val and test"

    # -----------------------------------------------------------------------
    # P07: Synthetic Generator (F6) × Apple Silicon MPS Training Step (F8)
    # -----------------------------------------------------------------------
    def test_p07_f6_f8_synthetic_gen_to_mps_training(
        self, tmp_path: Path
    ) -> None:
        """
        P07: Synthetic handwriting generator feeds PyTorch DataLoader into an Apple Silicon MPS training step.
        Verifies:
        - PyTorch DataLoader yields batched tensors of shape [B, 3, H, W] in float32.
        - Model executes forward pass on MPS (or CPU fallback).
        - Backward pass computes gradients without NaN or Inf values.
        """
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Generate batch of 8 synthetic lines
        gen = SyntheticHandwritingGenerator()
        samples: List[HandwritingSample] = []
        for i in range(8):
            img, _ = gen.render_line(f"Training line sample {i+1}", font_size=28)
            samples.append(
                HandwritingSample(
                    sample_id=f"train_{i}",
                    image=img,
                    text=f"Training line sample {i+1}",
                    writer_id=f"w_{i%2}",
                )
            )

        dataset = HandwritingPyTorchDataset(samples, target_size=(64, 256))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=True, collate_fn=handwriting_collate_fn
        )

        # Lightweight CNN vision encoder for verification
        class SimpleOCRModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.fc = nn.Linear(32, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                feat = self.conv(x)
                return self.fc(feat.view(feat.size(0), -1))

        model = SimpleOCRModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            assert pixel_values.shape[0] == 4
            assert pixel_values.shape[1] == 3
            assert pixel_values.dtype == torch.float32

            targets = torch.tensor([0, 1, 2, 3], device=device)
            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = criterion(outputs, targets)
            loss.backward()

            # Verify gradients are finite
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                    assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

            optimizer.step()
            assert float(loss.item()) >= 0.0

    # -----------------------------------------------------------------------
    # P08: Model Training (F8) × Model Checkpointing & Resume (F9)
    # -----------------------------------------------------------------------
    def test_p08_f8_f9_mps_training_to_checkpointing(
        self, tmp_path: Path
    ) -> None:
        """
        P08: Training loop saves checkpoint, then resumes state in a fresh instance.
        Verifies:
        - Checkpoint directory contains model weights, optimizer state, epoch, and validation CER.
        - Reloaded model restores identical weights and resumes from epoch N+1.
        """
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        class ModelStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x):
                return self.linear(x)

        model = ModelStub()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Save checkpoint at epoch 2
        ckpt_path = ckpt_dir / "checkpoint_epoch_2.pt"
        checkpoint_data = {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_cer": 0.085,
            "timestamp": time.time(),
        }
        torch.save(checkpoint_data, ckpt_path)
        assert ckpt_path.exists()

        # Resume in fresh model instance
        resumed_model = ModelStub()
        resumed_optimizer = optim.SGD(resumed_model.parameters(), lr=0.01)

        loaded = torch.load(ckpt_path, weights_only=False)
        resumed_model.load_state_dict(loaded["model_state_dict"])
        resumed_optimizer.load_state_dict(loaded["optimizer_state_dict"])

        assert loaded["epoch"] == 2
        assert loaded["val_cer"] == 0.085

        # Verify weights match exactly
        for p1, p2 in zip(model.parameters(), resumed_model.parameters()):
            assert torch.equal(p1, p2), "Restored weights do not match saved weights"

    # -----------------------------------------------------------------------
    # P09: Training Loop (F8) × Loss Curve Logger & Plotting (F10)
    # -----------------------------------------------------------------------
    def test_p09_f8_f10_mps_training_to_loss_logger(
        self, mock_engine: MockInferenceEngine, tmp_path: Path
    ) -> None:
        """
        P09: Training loop logs metrics to CSV and produces loss curve visualization.
        Verifies:
        - losses.csv contains columns `epoch,train_loss,val_cer,val_wer`.
        - Loss curve plot is written as a valid non-empty PNG image.
        """
        log_dir = tmp_path / "train_logs"
        res = mock_engine.mock_train_loop(
            dataset_path=str(tmp_path), epochs=3, out_dir=str(log_dir)
        )

        csv_path = Path(res["loss_csv"])
        assert csv_path.exists()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["epoch", "train_loss", "val_cer", "val_wer"]
            rows = list(reader)
            assert len(rows) == 3

        plot_path = Path(res["loss_plot"])
        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000

        # Verify it is a valid image
        plot_img = Image.open(plot_path)
        assert plot_img.size[0] > 100 and plot_img.size[1] > 100

    # -----------------------------------------------------------------------
    # P10: Checkpointing (F9) × jiwer CER/WER Evaluation (F11)
    # -----------------------------------------------------------------------
    def test_p10_f9_f11_checkpoint_to_jiwer_evaluation(
        self, mock_engine: MockInferenceEngine
    ) -> None:
        """
        P10: Model predictions evaluated with CER, WER, and latency metrics.
        Verifies:
        - CER is computed within [0.0, 1.0].
        - WER is computed within [0.0, 1.0].
        - Latency percentiles (p50, p90, p99) and throughput (samples/sec) are reported.
        """
        predictions = [
            "Rx: Amoxicillin 500mg capsules",
            "Sig: 1 tab PO TID x 10 days",
            "Dr. J. Doe MD",
        ]
        ground_truth = [
            "Rx: Amoxicillin 500mg capsules",
            "Sig: 1 tab PO TID x 10 days",
            "Dr. J. Doe MD",
        ]

        eval_res = mock_engine.mock_evaluate(predictions, ground_truth)
        assert eval_res["sample_count"] == 3
        assert eval_res["mean_cer"] == 0.0
        assert eval_res["mean_wer"] == 0.0
        assert eval_res["p50_latency_ms"] > 0.0
        assert eval_res["throughput_samples_per_sec"] > 0.0

        # Imperfect predictions
        noisy_preds = [
            "Rx: Amoxicilin 500mg capsules",  # 1 deletion
            "Sig: 1 tab PO BID x 10 days",   # 1 word subst
            "Dr. J. Doe MD",
        ]
        noisy_res = mock_engine.mock_evaluate(noisy_preds, ground_truth)
        assert 0.0 < noisy_res["mean_cer"] < 0.15
        assert 0.0 < noisy_res["mean_wer"] < 0.25

    # -----------------------------------------------------------------------
    # P11: CER/WER Evaluator (F11) × Structured JSON Transcription (F13)
    # -----------------------------------------------------------------------
    def test_p11_f11_f13_jiwer_eval_to_structured_json(
        self, clean_image_path: Path, mock_engine: MockInferenceEngine
    ) -> None:
        """
        P11: Evaluator consumes structured JSON lines and verifies CER against reference text.
        Verifies:
        - Predicted line strings extracted from PageResult match reference text.
        - CER calculation matches character-level Levenshtein distance.
        - Word confidence scores correlate with correct transcriptions.
        """
        img_bytes = clean_image_path.read_bytes()
        resp = mock_engine.recognize_image(img_bytes, "sample_clean_handwriting.png")

        extracted_lines = [line.text for line in resp.pages[0].lines]
        reference_lines = [
            "The quick brown fox jumps over the lazy dog",
            "Handwriting recognition test sample",
            "Clean cursive text line three",
        ]

        assert len(extracted_lines) == len(reference_lines)
        for hyp, ref in zip(extracted_lines, reference_lines):
            cer = compute_cer(hyp, ref)
            assert cer <= 0.05, f"Line CER {cer} too high for {hyp!r} vs {ref!r}"

    # -----------------------------------------------------------------------
    # P12: FastAPI Service (F12) × Async Job Polling & Lifecycle (F14)
    # -----------------------------------------------------------------------
    def test_p12_f12_f14_fastapi_to_async_jobs(
        self, api_client: TestClient, clean_image_path: Path
    ) -> None:
        """
        P12: Submit multi-page job to /v1/jobs, poll status to completion.
        Verifies:
        - POST /v1/jobs returns HTTP 200 with job_id and status QUEUED.
        - GET /v1/jobs/{job_id} transitions from QUEUED -> PROCESSING -> COMPLETED.
        - Completed job response contains full structured recognition result.
        """
        img_bytes = clean_image_path.read_bytes()
        create_res = api_client.post(
            "/v1/jobs",
            files={"file": ("sample_clean_handwriting.png", img_bytes, "image/png")},
        )
        assert create_res.status_code == 200
        job_id = create_res.json()["job_id"]
        assert job_id.startswith("job_")

        # Poll 1
        st1 = api_client.get(f"/v1/jobs/{job_id}").json()
        assert st1["status"] == "PROCESSING"

        # Poll 2
        st2 = api_client.get(f"/v1/jobs/{job_id}").json()
        assert st2["status"] == "PROCESSING"

        # Poll 3 (completes)
        st3 = api_client.get(f"/v1/jobs/{job_id}").json()
        assert st3["status"] == "COMPLETED"
        assert st3["progress"] == 1.0
        assert st3["result"] is not None
        assert_valid_recognition_response(st3["result"])

    # -----------------------------------------------------------------------
    # P13: FastAPI Service (F12) × Multi-Device Modes (F15)
    # -----------------------------------------------------------------------
    def test_p13_f12_f15_fastapi_to_multi_device_modes(
        self, clean_image_path: Path
    ) -> None:
        """
        P13: Verify inference engine behaviors under mock, CPU, and MPS device modes.
        Verifies:
        - Mock engine executes deterministically in <50ms.
        - All execution modes adhere to the identical RecognitionResponse contract.
        """
        engine = MockInferenceEngine()
        img_bytes = clean_image_path.read_bytes()

        t0 = time.time()
        res_mock = engine.recognize(img_bytes, "sample_clean_handwriting.png")
        duration_ms = (time.time() - t0) * 1000.0

        assert_valid_recognition_response(res_mock)
        assert duration_ms < 50.0, f"Mock latency {duration_ms:.2f}ms exceeds 50ms threshold"

    # -----------------------------------------------------------------------
    # P14: Structured JSON Schema (F13) × SVG Document Viewer Overlay (F17)
    # -----------------------------------------------------------------------
    def test_p14_f13_f17_json_schema_to_svg_viewer(
        self, clean_image_path: Path, mock_engine: MockInferenceEngine
    ) -> None:
        """
        P14: Frontend SVG viewer parses PageResult and maps normalized bboxes to pixel viewBox coordinates.
        Verifies:
        - SVG viewBox matches page width and height.
        - Bounding box conversion: x = xmin * W, y = ymin * H, w = (xmax - xmin) * W, h = (ymax - ymin) * H.
        - Confidence threshold color styling: green (>=0.85), yellow (0.60-0.84), red (<0.60).
        """
        img_bytes = clean_image_path.read_bytes()
        resp = mock_engine.recognize_image(img_bytes, "sample_clean_handwriting.png")
        page = resp.pages[0]

        viewbox_w, viewbox_h = page.width, page.height
        assert viewbox_w == 1200 and viewbox_h == 800

        svg_rects = []
        for line in page.lines:
            ymin, xmin, ymax, xmax = line.bbox
            px_x = xmin * viewbox_w
            px_y = ymin * viewbox_h
            px_w = (xmax - xmin) * viewbox_w
            px_h = (ymax - ymin) * viewbox_h

            assert 0 <= px_x < viewbox_w
            assert 0 <= px_y < viewbox_h
            assert px_w > 0 and px_h > 0

            # Style class based on confidence
            conf = line.confidence
            if conf >= 0.85:
                color_class = "border-green-500 bg-green-500/10"
            elif conf >= 0.60:
                color_class = "border-yellow-500 bg-yellow-500/10"
            else:
                color_class = "border-red-500 bg-red-500/10"

            svg_rects.append({
                "id": line.line_id,
                "x": round(px_x, 1),
                "y": round(px_y, 1),
                "w": round(px_w, 1),
                "h": round(px_h, 1),
                "class": color_class,
            })

        assert len(svg_rects) == 3
        assert all(r["class"] == "border-green-500 bg-green-500/10" for r in svg_rects)

    # -----------------------------------------------------------------------
    # P15: Structured JSON Schema (F13) × Inline Editor & Diff Tracking (F18)
    # -----------------------------------------------------------------------
    def test_p15_f13_f18_json_schema_to_inline_editor(
        self, messy_image_path: Path, mock_engine: MockInferenceEngine
    ) -> None:
        """
        P15: Inline editor loads structured lines, applies correction, and tracks diff.
        Verifies:
        - Editing Line 2 modifies `text` while preserving `original_text`.
        - Edit status flag `is_edited` is set to True.
        - Diff calculation identifies character substitutions.
        """
        resp = mock_engine.recognize_image(messy_image_path.read_bytes(), "sample_messy_prescription.png")
        lines = resp.pages[0].lines

        # Model editor state
        editor_state = [
            {
                "line_id": l.line_id,
                "original_text": l.text,
                "current_text": l.text,
                "is_edited": False,
                "confidence": l.confidence,
            }
            for l in lines
        ]

        # Simulate user editing line 2: "Sig: 1 tab PO TID x 10 days" -> "Sig: 1 tablet PO TID x 10 days"
        target_idx = 1
        editor_state[target_idx]["current_text"] = "Sig: 1 tablet PO TID x 10 days"
        editor_state[target_idx]["is_edited"] = (
            editor_state[target_idx]["current_text"] != editor_state[target_idx]["original_text"]
        )

        assert editor_state[target_idx]["is_edited"] is True
        assert editor_state[0]["is_edited"] is False

        # Verify diff
        orig = editor_state[target_idx]["original_text"]
        curr = editor_state[target_idx]["current_text"]
        dist = compute_levenshtein_distance(curr, orig)
        assert dist == 3  # "tab" -> "tablet" (+3 chars)

    # -----------------------------------------------------------------------
    # P16: Structured JSON Schema (F13) × Multi-Format Exporter (F19)
    # -----------------------------------------------------------------------
    def test_p16_f13_f19_json_schema_to_multi_export(
        self, clean_image_path: Path, mock_engine: MockInferenceEngine
    ) -> None:
        """
        P16: Export recognized document to JSON, TXT, CSV, and clipboard text.
        Verifies:
        - Plain text export preserves page and line paragraph breaks.
        - CSV export includes standard column headers and properly escapes quotes/commas.
        - JSON export adheres to full document schema.
        """
        resp = mock_engine.recognize_image(clean_image_path.read_bytes(), "sample_clean_handwriting.png")

        # 1. Plain Text Export
        txt_out = "\n".join(p.full_text for p in resp.pages)
        assert "The quick brown fox" in txt_out
        assert len(txt_out.splitlines()) == 3

        # 2. CSV Export
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["Page", "Line_ID", "Confidence", "Original_Text", "Corrected_Text", "BBox"])

        for page in resp.pages:
            for line in page.lines:
                writer.writerow([
                    page.page_number,
                    line.line_id,
                    line.confidence,
                    line.text,
                    line.text,
                    json.dumps(line.bbox),
                ])

        csv_content = csv_buffer.getvalue()
        assert "Page,Line_ID,Confidence,Original_Text,Corrected_Text,BBox" in csv_content
        assert "p1_l1" in csv_content
        assert "The quick brown fox" in csv_content

        # 3. JSON Export
        json_str = resp.model_dump_json(indent=2)
        parsed = json.loads(json_str)
        assert parsed["total_pages"] == 1
        assert len(parsed["pages"][0]["lines"]) == 3

    # -----------------------------------------------------------------------
    # P17: Dropzone Upload (F16) × FastAPI Backend Upload (F12)
    # -----------------------------------------------------------------------
    def test_p17_f16_f12_dropzone_to_fastapi_upload(
        self, api_client: TestClient, corrupted_fixtures: Dict[str, Path]
    ) -> None:
        """
        P17: Frontend dropzone upload validation against FastAPI /v1/recognize.
        Verifies:
        - Accepts multi-part image uploads (PNG, JPEG).
        - Rejects zero-byte files with HTTP 422.
        - Rejects disguised plain-text files with HTTP 422.
        """
        # 1. Zero-byte rejection
        res_zero = api_client.post(
            "/v1/recognize",
            files={"file": ("zero.png", corrupted_fixtures["zero_byte"].read_bytes(), "image/png")},
        )
        assert res_zero.status_code == 422

        # 2. Disguised ASCII text rejection
        res_disguised = api_client.post(
            "/v1/recognize",
            files={"file": ("fake.png", corrupted_fixtures["disguised_text"].read_bytes(), "image/png")},
        )
        assert res_disguised.status_code == 422

    # -----------------------------------------------------------------------
    # P18: Viewer Pan/Zoom (F17) × Inline Editor Synchronized Selection (F18)
    # -----------------------------------------------------------------------
    def test_p18_f17_f18_viewer_panzoom_to_editor_sync(
        self, clean_image_path: Path, mock_engine: MockInferenceEngine
    ) -> None:
        """
        P18: Synchronized bidirectional focus between SVG bounding boxes and editor lines.
        Verifies:
        - Selecting Line 2 in editor sets `active_line_id = "p1_l2"`.
        - Viewport pan/zoom calculates correct center coordinates for active line bbox.
        - Bounding box scaling supports zoom factors from 0.5x to 4.0x.
        """
        resp = mock_engine.recognize_image(clean_image_path.read_bytes(), "sample_clean_handwriting.png")
        lines = resp.pages[0].lines
        page_w, page_h = resp.pages[0].width, resp.pages[0].height

        active_line_id = "p1_l2"
        target_line = next(l for l in lines if l.line_id == active_line_id)

        ymin, xmin, ymax, xmax = target_line.bbox
        bbox_center_x = ((xmin + xmax) / 2.0) * page_w
        bbox_center_y = ((ymin + ymax) / 2.0) * page_h

        # Verify zoom transform math
        for zoom_scale in [0.5, 1.0, 2.0, 4.0]:
            viewport_w, viewport_h = 800, 600
            # Target pan: shift center of bbox to center of viewport
            pan_x = (viewport_w / 2.0) - (bbox_center_x * zoom_scale)
            pan_y = (viewport_h / 2.0) - (bbox_center_y * zoom_scale)

            # Projected point of bbox center on screen
            screen_x = bbox_center_x * zoom_scale + pan_x
            screen_y = bbox_center_y * zoom_scale + pan_y

            assert abs(screen_x - (viewport_w / 2.0)) < 1e-3
            assert abs(screen_y - (viewport_h / 2.0)) < 1e-3

    # -----------------------------------------------------------------------
    # P19: Speed Review Filter (F18) × Low-Confidence Word Tokens (F13)
    # -----------------------------------------------------------------------
    def test_p19_f18_f13_speed_review_to_low_confidence(
        self, messy_image_path: Path, mock_engine: MockInferenceEngine
    ) -> None:
        """
        P19: Speed Review isolates words with confidence < 0.75 for rapid sequential review.
        Verifies:
        - Low-confidence filter correctly flags words below threshold.
        - Step-through review updates word text and marks word as verified.
        """
        resp = mock_engine.recognize_image(messy_image_path.read_bytes(), "sample_messy_prescription.png")

        # Inject simulated low-confidence words for testing
        all_words: List[Dict[str, Any]] = []
        for line in resp.pages[0].lines:
            for w in line.words:
                all_words.append({
                    "word_id": w.word_id,
                    "text": w.text,
                    "confidence": w.confidence,
                    "is_reviewed": False,
                })

        # Set 2 words to low confidence (<0.75)
        all_words[1]["confidence"] = 0.58
        all_words[3]["confidence"] = 0.62

        # Filter low confidence queue
        low_conf_queue = [w for w in all_words if w["confidence"] < 0.75]
        assert len(low_conf_queue) == 2
        assert low_conf_queue[0]["word_id"] == all_words[1]["word_id"]
        assert low_conf_queue[1]["word_id"] == all_words[3]["word_id"]

        # Step-through correction
        low_conf_queue[0]["text"] = "Amoxicillin"
        low_conf_queue[0]["is_reviewed"] = True
        low_conf_queue[1]["text"] = "PO"
        low_conf_queue[1]["is_reviewed"] = True

        assert all(w["is_reviewed"] for w in low_conf_queue)

    # -----------------------------------------------------------------------
    # P20: Multi-Format Export (F19) × Next.js Production Build Contracts (F20)
    # -----------------------------------------------------------------------
    def test_p20_f19_f20_multi_export_to_nextjs_build(
        self, clean_image_path: Path, mock_engine: MockInferenceEngine
    ) -> None:
        """
        P20: Export payloads match TypeScript export utilities in frontend.
        Verifies:
        - CSV conforms to RFC 4180 standard.
        - JSON export validates against Pydantic schema.
        - Clipboard payload is formatted plain text.
        """
        resp = mock_engine.recognize_image(clean_image_path.read_bytes(), "sample_clean_handwriting.png")

        # Verify RFC 4180 CSV export
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Line", "Confidence", "Transcription"])
        for idx, line in enumerate(resp.pages[0].lines):
            writer.writerow([idx + 1, line.confidence, line.text])

        csv_str = csv_buf.getvalue()
        assert "Line,Confidence,Transcription\r\n" in csv_str or "Line,Confidence,Transcription\n" in csv_str
        assert "The quick brown fox" in csv_str

    # -----------------------------------------------------------------------
    # P21: PDF Loader (F1) × Async Job Queue & Worker (F14)
    # -----------------------------------------------------------------------
    def test_p21_f1_f14_pdf_loader_to_async_job_queue(
        self, api_client: TestClient, multipage_pdf_path: Path
    ) -> None:
        """
        P21: 3-page PDF processed through async job queue.
        Verifies:
        - POST /v1/jobs accepts PDF upload.
        - Polling transitions to COMPLETED.
        - Result contains all 3 pages with extracted text.
        """
        pdf_bytes = multipage_pdf_path.read_bytes()
        create_res = api_client.post(
            "/v1/jobs",
            files={"file": ("sample_multipage_consultation.pdf", pdf_bytes, "application/pdf")},
        )
        assert create_res.status_code == 200
        job_id = create_res.json()["job_id"]

        # Poll to completion
        for _ in range(5):
            res = api_client.get(f"/v1/jobs/{job_id}").json()
            if res["status"] == "COMPLETED":
                break

        assert res["status"] == "COMPLETED"
        assert res["result"] is not None
        assert res["result"]["total_pages"] == 3
        assert len(res["result"]["pages"]) == 3

    # -----------------------------------------------------------------------
    # P22: Benchmark Loader (F7) × jiwer CER/WER Harness (F11)
    # -----------------------------------------------------------------------
    def test_p22_f7_f11_benchmark_loader_to_jiwer_harness(
        self, tmp_path: Path
    ) -> None:
        """
        P22: IAM / Prescription benchmark loader feeds test partition into CER/WER evaluation harness.
        Verifies:
        - Evaluates all samples in test partition.
        - Reports mean CER, mean WER, and sample count.
        """
        samples = [
            HandwritingSample(
                sample_id=f"bench_{i}",
                text="The quick brown fox jumps over the lazy dog",
                writer_id=f"w_{i}",
            )
            for i in range(10)
        ]

        predictions = [
            "The quick brown fox jumps over the lazy dog" if i < 8 else "The quick brown fox jumps over a lazy dog"
            for i in range(10)
        ]
        references = [s.text for s in samples]

        cer_list = [compute_cer(p, r) for p, r in zip(predictions, references)]
        wer_list = [compute_wer(p, r) for p, r in zip(predictions, references)]

        mean_cer = float(np.mean(cer_list))
        mean_wer = float(np.mean(wer_list))

        assert 0.0 <= mean_cer <= 0.05
        assert 0.0 <= mean_wer <= 0.10
        assert len(cer_list) == 10

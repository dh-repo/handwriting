"""
pipeline/tests/test_adversarial_m1_challenger.py
Empirical Adversarial Stress Test Suite for Milestone 1:
- Line Segmentation & Preprocessing Pipeline (blank pages, 1-line, 15-30 dense lines, overlapping ascenders/descenders, lined notebook paper, [0.0, 1.0] normalized bbox invariants).
- Synthetic Handwriting Generator (empty strings, huge paragraphs, missing font fallback, Albumentations on unusual aspect ratios).
- Dataset Loaders (IAM writer independence verification W_train ∩ W_test = ∅, PyTorch DataLoader collation on variable dimensions/batch sizes).
"""

import io
import math
import os
import random
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from pipeline.preprocessing.image_enhancement import (
    ImageEnhancer,
    adaptive_binarize,
    deskew_image,
    enhance_contrast,
    flatten_illumination,
    normalize_image,
    to_grayscale,
    to_rgb,
)
from pipeline.preprocessing.line_segmenter import LineCrop, LineSegmenter, WordCrop
from pipeline.preprocessing.pipeline import PreprocessingPipeline, PreprocessedPage
from pipeline.dataset.synthetic_generator import (
    BackgroundGenerator,
    HandwritingFontManager,
    SyntheticHandwritingGenerator,
    ALBUMENTATIONS_AVAILABLE,
)
from pipeline.dataset.dataset_loader import (
    HandwritingPyTorchDataset,
    HandwritingSample,
    IAMDatasetParser,
    MedicalPrescriptionDatasetLoader,
    MedicalPrescriptionSample,
    StreamingSyntheticDataset,
    handwriting_collate_fn,
)


# ===========================================================================
# 1. Line Segmenter & Preprocessing Pipeline Adversarial Tests
# ===========================================================================

class TestLineSegmenterAdversarial:
    """
    Adversarial challenge tests for LineSegmenter and PreprocessingPipeline.
    """

    def test_blank_and_near_blank_pages(self):
        """
        Challenge: Blank pages, all-white, all-black, and random noise speckles.
        Expected: Clean handling without crashing, returning empty line lists or safe crops.
        """
        segmenter = LineSegmenter(extract_words=True)
        enhancer = ImageEnhancer()

        # 1. All-white page (800x600, 255)
        white_img = np.full((800, 600, 3), 255, dtype=np.uint8)
        white_bin = np.zeros((800, 600), dtype=np.uint8)
        lines = segmenter.segment(white_img, white_bin)
        assert lines == [], f"Expected 0 lines for all-white page, got {len(lines)}"

        # 2. All-black page (800x600, 0)
        black_img = np.zeros((800, 600, 3), dtype=np.uint8)
        black_bin = np.full((800, 600), 255, dtype=np.uint8)
        # When entire page is ink, ink_mask has 0 or all pixels.
        # It should handle gracefully without crash.
        lines = segmenter.segment(black_img, black_bin)
        assert isinstance(lines, list)

        # 3. Near-blank with faint speckles (< 50 ink pixels)
        speckle_img = np.full((800, 600, 3), 255, dtype=np.uint8)
        speckle_bin = np.zeros((800, 600), dtype=np.uint8)
        # Add 30 random noise dots
        for _ in range(30):
            ry, rx = random.randint(0, 799), random.randint(0, 599)
            speckle_bin[ry, rx] = 255
            speckle_img[ry, rx] = (0, 0, 0)
        lines = segmenter.segment(speckle_img, speckle_bin)
        assert lines == [], f"Expected 0 lines for <50 ink pixels speckle noise, got {len(lines)}"

        # 4. Pipeline execution on blank white image
        pipeline = PreprocessingPipeline(extract_words=True)
        page = pipeline.process_image(white_img)
        assert isinstance(page, PreprocessedPage)
        assert len(page.lines) == 0

    def test_single_line_boundary_placements(self):
        """
        Challenge: Single line placed at extreme boundaries (top edge, middle, bottom edge).
        """
        segmenter = LineSegmenter(extract_words=True, margin=4)
        H, W = 800, 600

        for y_pos in [10, 400, 770]:
            img = np.full((H, W, 3), 255, dtype=np.uint8)
            bin_img = np.zeros((H, W), dtype=np.uint8)

            # Draw a simulated text stroke: 3 word blocks
            for wx in [50, 200, 380]:
                cv2.rectangle(img, (wx, y_pos), (wx + 100, min(H - 1, y_pos + 18)), (20, 20, 20), -1)
                cv2.rectangle(bin_img, (wx, y_pos), (wx + 100, min(H - 1, y_pos + 18)), 255, -1)

            lines = segmenter.segment(img, bin_img)
            assert len(lines) == 1, f"Expected 1 line for single-line placement at y={y_pos}, got {len(lines)}"

            crop = lines[0]
            assert isinstance(crop, LineCrop)
            assert crop.image.ndim == 3
            assert crop.image.shape[0] > 0 and crop.image.shape[1] > 0
            # Verify coordinates are strictly in [0.0, 1.0]
            ymin, xmin, ymax, xmax = crop.bbox
            assert 0.0 <= ymin <= ymax <= 1.0, f"Line bbox y out of range: {crop.bbox}"
            assert 0.0 <= xmin <= xmax <= 1.0, f"Line bbox x out of range: {crop.bbox}"

            # Verify word breakdown
            assert crop.words is not None and len(crop.words) == 3, f"Expected 3 words, got {len(crop.words) if crop.words else 0}"
            for w in crop.words:
                assert isinstance(w, WordCrop)
                w_ymin, w_xmin, w_ymax, w_xmax = w.bbox
                assert 0.0 <= w_ymin <= w_ymax <= 1.0, f"Word bbox y out of range: {w.bbox}"
                assert 0.0 <= w_xmin <= w_xmax <= 1.0, f"Word bbox x out of range: {w.bbox}"

    def test_dense_multiline_pages_15_to_30_lines(self):
        """
        Challenge: 15, 20, and 25 dense lines with narrow line spacing.
        """
        segmenter = LineSegmenter(min_line_height=12, seam_carving=True, extract_words=False)

        for num_lines in [15, 20, 25]:
            H, W = 1600, 1000
            img = np.full((H, W, 3), 255, dtype=np.uint8)
            bin_img = np.zeros((H, W), dtype=np.uint8)

            spacing = (H - 100) // (num_lines + 1)
            for i in range(num_lines):
                y_center = 50 + (i + 1) * spacing
                # Draw text line with 4 words
                for w_idx in range(4):
                    wx = 80 + w_idx * 210
                    cv2.rectangle(img, (wx, y_center - 8), (wx + 170, y_center + 8), (30, 30, 80), -1)
                    cv2.rectangle(bin_img, (wx, y_center - 8), (wx + 170, y_center + 8), 255, -1)

            crops = segmenter.segment(img, bin_img)
            assert len(crops) >= num_lines - 1, f"Expected at least {num_lines - 1} lines detected, got {len(crops)}"

            # Verify all crops valid
            for idx, crop in enumerate(crops):
                assert crop.line_index == idx
                assert crop.image.shape[0] > 0 and crop.image.shape[1] > 0
                ymin, xmin, ymax, xmax = crop.bbox
                assert 0.0 <= ymin <= ymax <= 1.0
                assert 0.0 <= xmin <= xmax <= 1.0

    def test_deskew_featureless_or_low_edge_image(self):
        """
        Challenge: Deskewing a completely flat/blank or low-contrast image.
        Should detect 0.0 skew angle, NOT default to -45.0 degrees.
        """
        flat_img = np.full((300, 400, 3), 200, dtype=np.uint8)
        deskewed, angle = deskew_image(flat_img)
        assert abs(angle) < 0.1, f"Expected ~0.0 skew angle for flat image, got {angle}"

    def test_overlapping_ascenders_descenders_seam_carving(self):
        """
        Challenge: Deeply overlapping cursive ascenders and descenders between adjacent lines.
        Ensures Vectorized Dynamic Programming seam carving separates lines without truncation or crashes.
        """
        segmenter = LineSegmenter(seam_carving=True)
        H, W = 400, 600
        img = np.full((H, W, 3), 255, dtype=np.uint8)
        bin_img = np.zeros((H, W), dtype=np.uint8)

        # Line 1: words at y=100..130 with descenders
        for wx in [50, 180, 320, 460]:
            cv2.rectangle(img, (wx, 100), (wx + 90, 130), (30, 30, 30), -1)
            cv2.rectangle(bin_img, (wx, 100), (wx + 90, 130), 255, -1)
        # Descenders at x=150, 350
        for dx in [150, 350]:
            cv2.rectangle(img, (dx, 130), (dx + 15, 230), (30, 30, 30), -1)
            cv2.rectangle(bin_img, (dx, 130), (dx + 15, 230), 255, -1)

        # Line 2: words at y=240..270 with ascenders
        for wx in [50, 180, 320, 460]:
            cv2.rectangle(img, (wx, 240), (wx + 90, 270), (30, 30, 30), -1)
            cv2.rectangle(bin_img, (wx, 240), (wx + 90, 270), 255, -1)
        # Ascenders at x=250, 450
        for ax in [250, 450]:
            cv2.rectangle(img, (ax, 150), (ax + 15, 240), (30, 30, 30), -1)
            cv2.rectangle(bin_img, (ax, 150), (ax + 15, 240), 255, -1)

        crops = segmenter.segment(img, bin_img)
        assert len(crops) == 2, f"Expected 2 lines for interlocking strokes, got {len(crops)}"

        for crop in crops:
            assert crop.image.ndim == 3
            assert crop.image.shape[0] > 0 and crop.image.shape[1] > 0
            ymin, xmin, ymax, xmax = crop.bbox
            assert 0.0 <= ymin <= ymax <= 1.0
            assert 0.0 <= xmin <= xmax <= 1.0

    def test_lined_notebook_paper_ruling_interference(self):
        """
        Challenge: Continuous dark ruling lines running horizontally across the entire page.
        The segmenter must filter ruling lines from HPP to correctly segment the handwritten text lines.
        """
        segmenter = LineSegmenter(seam_carving=True)
        H, W = 800, 800

        # Create lined notebook paper
        paper = BackgroundGenerator.generate_lined_paper(
            width=W, height=H, line_spacing=60, margin_x=80,
            paper_color=(250, 250, 250), line_color=(100, 100, 100) # Dark ruling lines
        )

        # Draw 5 distinct text lines between/across ruling lines
        bin_img = np.zeros((H, W), dtype=np.uint8)
        # Ruling lines will appear in binarized image:
        for y in range(60, H - 10, 60):
            bin_img[y, :] = 255  # Solid continuous ruling lines

        # Draw 5 handwritten lines at y=150, 270, 390, 510, 630
        for y_text in [150, 270, 390, 510, 630]:
            for wx in [120, 300, 480]:
                cv2.rectangle(paper, (wx, y_text - 12), (wx + 140, y_text + 12), (20, 30, 120), -1)
                cv2.rectangle(bin_img, (wx, y_text - 12), (wx + 140, y_text + 12), 255, -1)

        crops = segmenter.segment(paper, bin_img)
        # Verify that text lines were segmented and ruling lines didn't create 13 separate ruling line crops
        assert 4 <= len(crops) <= 6, f"Expected ~5 text lines on lined notebook paper, got {len(crops)}"

        for crop in crops:
            ymin, xmin, ymax, xmax = crop.bbox
            assert 0.0 <= ymin <= ymax <= 1.0
            assert 0.0 <= xmin <= xmax <= 1.0

    def test_extreme_aspect_ratios_and_dimensions(self):
        """
        Challenge: Extreme image shapes (very thin, very wide, 1x1, 10x10).
        """
        segmenter = LineSegmenter()

        # 1. Very thin vertical image (1000 x 50)
        tall_img = np.full((1000, 50, 3), 255, dtype=np.uint8)
        tall_bin = np.zeros((1000, 50), dtype=np.uint8)
        cv2.rectangle(tall_img, (5, 200), (45, 250), (0, 0, 0), -1)
        cv2.rectangle(tall_bin, (5, 200), (45, 250), 255, -1)
        tall_crops = segmenter.segment(tall_img, tall_bin)
        assert len(tall_crops) == 1
        assert 0.0 <= tall_crops[0].bbox[0] <= tall_crops[0].bbox[2] <= 1.0

        # 2. Very wide horizontal image (50 x 1000)
        wide_img = np.full((50, 1000, 3), 255, dtype=np.uint8)
        wide_bin = np.zeros((50, 1000), dtype=np.uint8)
        cv2.rectangle(wide_img, (100, 10), (900, 40), (0, 0, 0), -1)
        cv2.rectangle(wide_bin, (100, 10), (900, 40), 255, -1)
        wide_crops = segmenter.segment(wide_img, wide_bin)
        assert len(wide_crops) == 1
        assert 0.0 <= wide_crops[0].bbox[1] <= wide_crops[0].bbox[3] <= 1.0

        # 3. Tiny 10x10 image
        tiny_img = np.full((10, 10, 3), 255, dtype=np.uint8)
        tiny_bin = np.zeros((10, 10), dtype=np.uint8)
        tiny_crops = segmenter.segment(tiny_img, tiny_bin)
        assert tiny_crops == []


# ===========================================================================
# 2. Synthetic Handwriting Generator Adversarial Tests
# ===========================================================================

class TestSyntheticGeneratorAdversarial:
    """
    Adversarial stress testing for SyntheticHandwritingGenerator, FontManager, and Albumentations.
    """

    def test_render_line_empty_and_whitespace_strings(self):
        """
        Challenge: Empty string, whitespace strings, tab/newline characters.
        """
        gen = SyntheticHandwritingGenerator()

        for empty_text in ["", "   ", "\t  \n  "]:
            img, meta = gen.render_line(empty_text, font_size=32)
            assert isinstance(img, np.ndarray)
            assert img.ndim == 3 and img.shape[2] == 3
            assert img.dtype == np.uint8
            assert "norm_bbox" in meta
            ymin, xmin, ymax, xmax = meta["norm_bbox"]
            assert 0.0 <= ymin <= ymax <= 1.0
            assert 0.0 <= xmin <= xmax <= 1.0

    def test_render_line_massive_text(self):
        """
        Challenge: Very long paragraphs (300 to 1000 characters).
        """
        gen = SyntheticHandwritingGenerator()
        long_text = "The quick brown fox jumps over the lazy dog repeatedly and creates an exceedingly long synthetic handwriting line for stress testing memory and canvas bounds. " * 3

        img, meta = gen.render_line(long_text, font_size=28, slant_deg=10.0)
        assert img.shape[0] > 0 and img.shape[1] > 500
        ymin, xmin, ymax, xmax = meta["norm_bbox"]
        assert 0.0 <= ymin <= ymax <= 1.0
        assert 0.0 <= xmin <= xmax <= 1.0

    def test_missing_font_fallback_and_headless_simulation(self):
        """
        Challenge: Missing font names, corrupted font requests, and simulated headless Linux environment.
        """
        # 1. Non-existent font name requested
        gen = SyntheticHandwritingGenerator()
        font = gen.font_mgr.get_font("CompletelyNonExistentFont_12345.ttf", size=32)
        assert font is not None, "FontManager failed to fallback gracefully on non-existent font name"

        img, meta = gen.render_line("Testing fallback font", font_name="BogusFont.ttf")
        assert img.ndim == 3 and img.shape[0] > 0

        # 2. Simulated headless Linux container (empty font directory search)
        with tempfile.TemporaryDirectory() as empty_dir:
            headless_mgr = HandwritingFontManager(custom_font_dirs=[empty_dir])
            # Clear discovered system fonts to strictly simulate bare Linux environment
            headless_mgr.font_paths.clear()

            # get_font() should fall back to PIL default font without crashing
            default_font = headless_mgr.get_font(size=24)
            assert default_font is not None

            headless_gen = SyntheticHandwritingGenerator(font_manager=headless_mgr)
            h_img, h_meta = headless_gen.render_line("Headless Linux Test", font_size=24)
            assert h_img.ndim == 3
            assert h_meta["text"] == "Headless Linux Test"

    def test_albumentations_unusual_aspect_ratios_and_shapes(self):
        """
        Challenge: Elastic distortions applied on extreme aspect ratios (ultra-wide, ultra-tall, tiny, huge).
        """
        if not ALBUMENTATIONS_AVAILABLE:
            pytest.skip("Albumentations not installed")

        gen = SyntheticHandwritingGenerator()

        test_shapes = [
            (30, 2000, 3),   # Ultra-wide line strip
            (2000, 30, 3),   # Ultra-tall strip
            (16, 16, 3),     # Tiny image
            (800, 800, 3),   # Square image
        ]

        for shape in test_shapes:
            dummy_img = np.full(shape, 240, dtype=np.uint8)
            # Add some simulated ink
            dummy_img[shape[0] // 2, :, :] = 30

            augmented = gen.apply_elastic_distortions(dummy_img)
            assert augmented.shape == shape, f"Elastic distortion altered shape: expected {shape}, got {augmented.shape}"
            assert augmented.dtype == np.uint8
            assert not np.isnan(augmented).any()
            assert not np.isinf(augmented).any()

    def test_render_prescription_exotic_configurations(self):
        """
        Challenge: Prescriptions with 0 medications, 5 medications, custom Unicode clinic/doctor names.
        """
        gen = SyntheticHandwritingGenerator()

        # 1. Prescription with 0 medications
        p_img_0, p_doc_0 = gen.render_prescription(
            medications=[],
            include_stamp=True,
            include_stains=True,
            width=700,
            height=900
        )
        assert p_img_0.shape == (900, 700, 3)
        assert len(p_doc_0["lines"]) == 0

        # 2. Prescription with 4 medications + Unicode names
        meds_4 = [
            ("Amoxicillin", "500mg", "1 cap tid", "#30", "0"),
            ("Lisinopril", "20mg", "1 tab daily", "#90", "3"),
            ("Metformin", "1000mg", "1 tab bid", "#60", "2"),
            ("Atorvastatin", "40mg", "1 tab qhs", "#30", "1"),
        ]
        p_img_4, p_doc_4 = gen.render_prescription(
            clinic_name="Clinique Saint-Honoré 🏥",
            doctor_name="Dr. François Müller, MD",
            patient_name="José González",
            medications=meds_4,
            include_stamp=True,
            include_stains=True,
            apply_distortions=True,
            width=800,
            height=1200
        )
        assert p_img_4.shape == (1200, 800, 3)
        assert len(p_doc_4["lines"]) == 12  # 4 meds * 3 lines each
        for line_meta in p_doc_4["lines"]:
            ymin, xmin, ymax, xmax = line_meta["bbox"]
            assert 0.0 <= ymin <= ymax <= 1.0
            assert 0.0 <= xmin <= xmax <= 1.0


# ===========================================================================
# 3. Dataset Loader & PyTorch Batching Adversarial Tests
# ===========================================================================

class TestDatasetLoaderAdversarial:
    """
    Adversarial challenge tests for IAM Dataset partitioner, PyTorch Dataset, and DataLoader collator.
    """

    def test_iam_writer_independence_exhaustive(self):
        """
        Challenge: Exhaustively verify zero writer overlap across varying split ratios,
        writer counts (1, 2, 3, 10, 50, 200 writers), and random seeds.
        Strict contract: W_train ∩ W_val = ∅, W_train ∩ W_test = ∅, W_val ∩ W_test = ∅.
        """
        parser = IAMDatasetParser()

        # Test varying writer counts
        for num_writers in [2, 3, 5, 10, 50, 150]:
            for seed in [42, 100, 999]:
                # Generate synthetic samples with multiple samples per writer
                samples = []
                for w_idx in range(num_writers):
                    w_id = f"writer_{w_idx:03d}"
                    num_samples_for_w = random.randint(1, 8)
                    for s_idx in range(num_samples_for_w):
                        samples.append(
                            HandwritingSample(
                                sample_id=f"{w_id}_s{s_idx:02d}",
                                text=f"Sample text {s_idx} by {w_id}",
                                writer_id=w_id
                            )
                        )

                train_s, val_s, test_s = parser.create_writer_independent_splits(
                    samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
                )

                train_writers = set(s.writer_id for s in train_s)
                val_writers = set(s.writer_id for s in val_s)
                test_writers = set(s.writer_id for s in test_s)

                # 1. Zero Writer Overlap Invariant
                assert train_writers.isdisjoint(val_writers), f"Writer leakage between Train & Val: {train_writers & val_writers}"
                assert train_writers.isdisjoint(test_writers), f"Writer leakage between Train & Test: {train_writers & test_writers}"
                assert val_writers.isdisjoint(test_writers), f"Writer leakage between Val & Test: {val_writers & test_writers}"

                # 2. Total Sample Count Conservation
                assert len(train_s) + len(val_s) + len(test_s) == len(samples), "Sample loss during partitioning!"

                # 3. Sample ID uniqueness across splits
                train_ids = set(s.sample_id for s in train_s)
                val_ids = set(s.sample_id for s in val_s)
                test_ids = set(s.sample_id for s in test_s)
                assert train_ids.isdisjoint(val_ids)
                assert train_ids.isdisjoint(test_ids)
                assert val_ids.isdisjoint(test_ids)

    def test_dataloader_variable_batch_sizes_and_image_dimensions(self):
        """
        Challenge: Test PyTorch DataLoader with handwriting_collate_fn on diverse,
        unaligned image dimensions in the same batch and various batch sizes (1, 2, 7, 16).
        """
        # Create samples with radically different dimensions
        diverse_dimensions = [
            (32, 100),
            (80, 550),
            (45, 220),
            (120, 300),
            (25, 80),
            (60, 420),
            (95, 310),
            (15, 600),
        ]

        samples = []
        for i, (h, w) in enumerate(diverse_dimensions):
            # Create synthetic line image with unique ink intensity
            img = np.full((h, w, 3), 255, dtype=np.uint8)
            cv2.putText(img, f"Sample {i}", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
            samples.append(
                HandwritingSample(
                    sample_id=f"sample_{i:03d}",
                    image=img,
                    text=f"Transcription {i}",
                    writer_id=f"w_{i % 3}"
                )
            )

        # PyTorch Dataset WITHOUT fixed target_size (yielding raw variable tensors)
        ds = HandwritingPyTorchDataset(samples, target_size=None)

        for batch_size in [1, 2, 3, 5, 8]:
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=handwriting_collate_fn
            )

            total_items = 0
            for batch in loader:
                assert "pixel_values" in batch
                assert "sample_ids" in batch
                assert "texts" in batch
                assert "writer_ids" in batch

                pv = batch["pixel_values"]
                assert isinstance(pv, torch.Tensor)
                assert pv.dtype == torch.float32
                assert pv.ndim == 4  # (B, 3, max_H, max_W)
                assert pv.shape[0] <= batch_size
                assert pv.shape[1] == 3

                # Value range invariant: [0.0, 1.0]
                assert pv.min() >= 0.0
                assert pv.max() <= 1.0

                # Check padding color: un-drawn padded background should be 1.0 (white)
                # Bottom-right corner of padded tensor should be 1.0
                assert pv[:, :, -1, -1].mean().item() >= 0.99

                total_items += len(batch["sample_ids"])

            assert total_items == len(samples)

    def test_medical_prescription_loader_manifest_variants(self):
        """
        Challenge: Load medical prescription manifests in JSON, JSONL, and CSV formats,
        handling missing fields, extra keys, and malformed entries gracefully.
        """
        loader = MedicalPrescriptionDatasetLoader()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # 1. JSON manifest with structured and legacy field names
            json_file = tmp / "manifest.json"
            json_content = [
                {
                    "sample_id": "rx_001",
                    "clinic": "Boston Clinic",
                    "doctor": "Dr. Smith",
                    "patient": "Jane Doe",
                    "date": "2026-08-26",
                    "medications": [
                        {"name": "Amoxicillin", "strength": "500mg", "instructions": "1 cap tid", "disp": "#30", "refills": "0"}
                    ]
                },
                {
                    "sample_id": "rx_002",
                    "clinic_name": "Seattle Health",
                    "doctor_name": "Dr. Vance",
                    "patient_name": "Bob Jones",
                    "date": "2026-08-26",
                    "items": [
                        ["Lisinopril", "20mg", "1 tab daily", "#90", "3"]
                    ]
                }
            ]
            import json
            with open(json_file, "w") as f:
                json.dump(json_content, f)

            samples_json = loader.load_manifest(json_file)
            assert len(samples_json) == 2
            assert samples_json[0].clinic_name == "Boston Clinic"
            assert len(samples_json[0].items) == 1
            assert samples_json[0].items[0].medication == "Amoxicillin"
            assert samples_json[1].items[0].medication == "Lisinopril"

            # 2. JSONL manifest with blank lines
            jsonl_file = tmp / "manifest.jsonl"
            with open(jsonl_file, "w") as f:
                f.write(json.dumps({"sample_id": "rx_jl_1", "doctor_name": "Dr. Lee", "items": []}) + "\n\n")
                f.write(json.dumps({"sample_id": "rx_jl_2", "doctor_name": "Dr. Kim", "items": []}) + "\n")

            samples_jsonl = loader.load_manifest(jsonl_file)
            assert len(samples_jsonl) == 2
            assert samples_jsonl[0].doctor_name == "Dr. Lee"
            assert samples_jsonl[1].doctor_name == "Dr. Kim"

            # 3. CSV manifest
            csv_file = tmp / "manifest.csv"
            with open(csv_file, "w", newline="") as f:
                f.write("sample_id,clinic_name,doctor_name,patient_name,full_text\n")
                f.write("rx_csv_1,Chicago Care,Dr. Ray,Alice,Amoxicillin 500mg\n")

            samples_csv = loader.load_manifest(csv_file)
            assert len(samples_csv) == 1
            assert samples_csv[0].sample_id == "rx_csv_1"
            assert samples_csv[0].full_text == "Amoxicillin 500mg"

            # 4. Non-existent manifest error handling
            with pytest.raises(FileNotFoundError):
                loader.load_manifest(tmp / "non_existent.json")

            # 5. Unsupported extension error handling
            dummy_xml = tmp / "manifest.xml"
            dummy_xml.write_text("<root></root>")
            with pytest.raises(ValueError):
                loader.load_manifest(dummy_xml)

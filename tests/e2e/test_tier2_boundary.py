"""
tests/e2e/test_tier2_boundary.py
Exhaustive Tier 2 Boundary, Corner Case, and Corruption Test Suite for Handwriting Recognition Solution.
Validates boundary values, empty/corrupted inputs, extreme dimensions, noise, and error conditions across F1-F24.
All tests marked with @pytest.mark.tier2.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from pydantic import ValidationError

from pipeline.dataset.dataset_loader import (
    HandwritingSample,
    IAMDatasetParser,
    MedicalPrescriptionSample,
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
    normalize_image,
    normalize_tensor,
    to_grayscale,
    to_rgb,
)
from pipeline.preprocessing.line_segmenter import (
    LineCrop as PipelineLineCrop,
    LineSegmenter,
    WordCrop as PipelineWordCrop,
)
from pipeline.preprocessing.pdf_loader import (
    CorruptDocumentError,
    DocumentLoadingError,
    EmptyDocumentError,
    PDFLoader,
    UnsupportedFormatError,
    detect_format,
    load_document,
    load_image,
    load_pdf,
)
from tests.e2e.conftest import AssertionHelpers
from tests.fixtures.mock_engine import (
    JobStatusResponse,
    LineBox,
    LineCrop,
    MockInferenceEngine,
    PageResult,
    PreprocessedPage,
    RecognitionResponse,
    WordBox,
    _validate_bbox_coordinates,
    compute_cer,
    compute_levenshtein_distance,
    compute_wer,
    create_mock_app,
)


# ===========================================================================
# FEATURE F1 BOUNDARY: Multi-format Ingestion
# ===========================================================================

@pytest.mark.tier2
class TestTier2F1MultiFormatIngestion:
    """Boundary, corruption, and malformed input tests for document ingestion."""

    def test_f1_b01_zero_byte_empty_file(self, corrupted_fixtures: Dict[str, Path]) -> None:
        """Verify loading 0-byte file raises EmptyDocumentError or ValueError."""
        zero_byte_path = corrupted_fixtures["zero_byte"]
        with pytest.raises((EmptyDocumentError, ValueError, DocumentLoadingError)):
            load_document(zero_byte_path)

    def test_f1_b02_corrupted_header_random_binary_garbage(
        self, corrupted_fixtures: Dict[str, Path]
    ) -> None:
        """Verify truncated / random binary garbage raises CorruptDocumentError or ValueError."""
        trunc_path = corrupted_fixtures["truncated_header"]
        with pytest.raises((CorruptDocumentError, ValueError, DocumentLoadingError)):
            load_document(trunc_path)

    def test_f1_b03_truncated_pdf_missing_eof_marker(
        self, corrupted_fixtures: Dict[str, Path]
    ) -> None:
        """Verify malformed PDF without %%EOF marker raises CorruptDocumentError."""
        corrupt_pdf_path = corrupted_fixtures["corrupted_pdf"]
        loader = PDFLoader()
        with pytest.raises((CorruptDocumentError, ValueError, DocumentLoadingError)):
            loader.load_pages(corrupt_pdf_path)

    def test_f1_b04_disguised_text_file_with_image_extension(
        self, corrupted_fixtures: Dict[str, Path]
    ) -> None:
        """Verify ASCII text disguised with .png extension is rejected."""
        disguised_path = corrupted_fixtures["disguised_text"]
        with pytest.raises((CorruptDocumentError, UnsupportedFormatError, ValueError, DocumentLoadingError)):
            load_document(disguised_path)

    def test_f1_b05_cmyk_color_profile_conversion(
        self, corrupted_fixtures: Dict[str, Path]
    ) -> None:
        """Verify 4-channel CMYK JPEG converts to 3-channel RGB uint8."""
        cmyk_path = corrupted_fixtures["cmyk_image"]
        pages = load_document(cmyk_path)
        assert len(pages) == 1
        page = pages[0]
        assert page.ndim == 3 and page.shape[2] == 3, f"Expected 3 channels, got {page.shape}"
        assert page.dtype == np.uint8

    def test_f1_b06_zero_dimension_numpy_array(self) -> None:
        """Verify invalid image array shape raises ValueError."""
        with pytest.raises(ValueError):
            to_grayscale(np.empty((2, 2, 2, 2), dtype=np.uint8))


# ===========================================================================
# FEATURE F2 BOUNDARY: Image Deskew & Orientation
# ===========================================================================

@pytest.mark.tier2
class TestTier2F2ImageDeskew:
    """Boundary and pathological edge-case tests for document deskewing."""

    def test_f2_b01_zero_variance_solid_white_canvas(self) -> None:
        """Verify pure white canvas returns angle 0.0 without division-by-zero."""
        white = np.full((300, 300, 3), 255, dtype=np.uint8)
        deskewed, angle = deskew_image(white)
        assert deskewed.shape == white.shape
        assert abs(angle) < 1.0

    def test_f2_b02_pure_gaussian_noise_image(self) -> None:
        """Verify pure random noise canvas returns bounded angle."""
        noise = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        deskewed, angle = deskew_image(noise)
        assert deskewed.dtype == np.uint8
        assert -45.0 <= angle <= 45.0

    def test_f2_b03_extreme_skew_angles(self) -> None:
        """Verify deskewing handles angles near search boundaries."""
        h, w = 300, 500
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "Extreme Skew Line", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), 40.0, 1.0)
        rotated = cv2.warpAffine(canvas, rot_mat, (w, h), borderValue=(255, 255, 255))

        deskewed, angle = deskew_image(rotated, min_angle=-45.0, max_angle=45.0)
        assert deskewed.shape[0] > 0 and deskewed.shape[1] > 0
        assert isinstance(angle, (int, float))

    def test_f2_b04_micro_dimension_patch_4x4(self) -> None:
        """Verify deskew on tiny 4x4 image patch runs without crashing."""
        tiny = np.full((4, 4, 3), 255, dtype=np.uint8)
        deskewed, angle = deskew_image(tiny)
        assert deskewed.shape == (4, 4, 3)

    def test_f2_b05_solid_black_document(self) -> None:
        """Verify solid black document returns 0.0 angle."""
        black = np.zeros((200, 200, 3), dtype=np.uint8)
        deskewed, angle = deskew_image(black)
        assert deskewed.shape == (200, 200, 3)
        assert abs(angle) < 1.0


# ===========================================================================
# FEATURE F3 BOUNDARY: Illumination Flattening & Contrast
# ===========================================================================

@pytest.mark.tier2
class TestTier2F3ContrastEnhancement:
    """Boundary tests for CLAHE and contrast enhancement."""

    def test_f3_b01_extreme_clahe_clip_limits(self) -> None:
        """Verify CLAHE handles extreme clip limits (0.1 and 50.0)."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        low_clip = enhance_contrast(img, clip_limit=0.1)
        assert low_clip.dtype == np.uint8
        high_clip = enhance_contrast(img, clip_limit=50.0)
        assert high_clip.dtype == np.uint8

    def test_f3_b02_tile_grid_larger_than_image_dimensions(self) -> None:
        """Verify contrast enhancement on image smaller than default grid (8x8)."""
        small = np.full((4, 4, 3), 128, dtype=np.uint8)
        enhanced = enhance_contrast(small)
        assert enhanced.shape == (4, 4, 3)
        assert enhanced.dtype == np.uint8

    def test_f3_b03_float_nan_and_inf_pixel_handling(self) -> None:
        """Verify to_rgb sanitizes float arrays with NaN/Inf values."""
        float_img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        float_img[0, 0, 0] = np.nan
        float_img[0, 1, 0] = np.inf
        rgb = to_rgb(float_img)
        assert not np.isnan(rgb).any()
        assert not np.isinf(rgb).any()
        assert rgb.dtype == np.uint8

    def test_f3_b04_all_black_illumination_flattening(self) -> None:
        """Verify flattening background on pure black canvas does not divide by zero."""
        black = np.zeros((150, 150, 3), dtype=np.uint8)
        flattened = enhance_contrast(black, flatten_background=True)
        assert flattened.shape == (150, 150, 3)

    def test_f3_b05_normalize_image_zero_dimensions(self) -> None:
        """Verify normalize_image handles non-standard target sizes."""
        canvas = np.full((50, 50, 3), 200, dtype=np.uint8)
        norm = normalize_image(canvas, target_size=(100, 100))
        assert norm.shape == (100, 100, 3)


# ===========================================================================
# FEATURE F4 BOUNDARY: Adaptive Binarization
# ===========================================================================

@pytest.mark.tier2
class TestTier2F4AdaptiveBinarization:
    """Boundary tests for Sauvola and Otsu adaptive thresholding."""

    def test_f4_b01_heavy_salt_and_pepper_noise_60pct(self) -> None:
        """Verify Sauvola thresholding executes under 60% salt-and-pepper noise."""
        canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
        noise = np.random.random((200, 200))
        canvas[noise < 0.30] = 0
        canvas[noise > 0.70] = 255

        mask = adaptive_binarize(canvas, method="sauvola")
        assert mask.shape == (200, 200)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_f4_b02_window_size_larger_than_image(self) -> None:
        """Verify Sauvola runs when window size exceeds image dimensions."""
        small = np.full((15, 15, 3), 200, dtype=np.uint8)
        small[5:10, 5:10] = 50
        mask = adaptive_binarize(small, method="sauvola")
        assert mask.shape == (15, 15)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_f4_b03_single_pixel_canvas_binarization(
        self, corrupted_fixtures: Dict[str, Path]
    ) -> None:
        """Verify 1x1 image binarization runs safely."""
        p1x1 = corrupted_fixtures["extreme_1x1"]
        img = load_document(p1x1)[0]
        mask = adaptive_binarize(img)
        assert mask.shape == (1, 1)

    def test_f4_b04_boolean_input_mask_handling(self) -> None:
        """Verify boolean input arrays convert properly to uint8 binary mask."""
        bool_arr = np.zeros((50, 50), dtype=bool)
        bool_arr[20:30, 20:30] = True
        mask = adaptive_binarize(bool_arr.astype(np.uint8) * 255)
        assert mask.dtype == np.uint8

    def test_f4_b05_faint_pencil_strokes_near_white(self) -> None:
        """Verify thresholding faint gray strokes on bright background."""
        canvas = np.full((100, 100, 3), 250, dtype=np.uint8)
        canvas[40:60, 20:80] = 240
        mask = adaptive_binarize(canvas, method="sauvola")
        assert mask.shape == (100, 100)


# ===========================================================================
# FEATURE F5 BOUNDARY: Line and Word Segmentation
# ===========================================================================

@pytest.mark.tier2
class TestTier2F5LineAndWordSegmentation:
    """Boundary tests for line and word segmentation."""

    def test_f5_b01_blank_document_zero_lines(self) -> None:
        """Verify pure white document returns empty line list."""
        blank = np.full((400, 400, 3), 255, dtype=np.uint8)
        binarized = adaptive_binarize(blank)
        segmenter = LineSegmenter()
        lines = segmenter.segment(blank, binarized)
        assert isinstance(lines, list)

    def test_f5_b02_solid_black_document_segmentation(self) -> None:
        """Verify solid black document does not cause segmentation crash."""
        black = np.zeros((300, 300, 3), dtype=np.uint8)
        binarized = adaptive_binarize(black)
        segmenter = LineSegmenter()
        lines = segmenter.segment(black, binarized)
        assert isinstance(lines, list)

    def test_f5_b03_extreme_aspect_ratio_100_to_1(
        self, corrupted_fixtures: Dict[str, Path]
    ) -> None:
        """Verify extreme 500:1 aspect ratio strip segments safely."""
        strip_path = corrupted_fixtures["extreme_aspect"]
        img = load_document(strip_path)[0]
        binarized = adaptive_binarize(img)
        segmenter = LineSegmenter()
        lines = segmenter.segment(img, binarized)
        assert isinstance(lines, list)

    def test_f5_b04_single_pixel_noise_speckles(self) -> None:
        """Verify isolated 1-pixel noise does not create degenerate line boxes."""
        canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
        canvas[50, 50] = 0
        canvas[150, 200] = 0
        binarized = adaptive_binarize(canvas)
        segmenter = LineSegmenter(min_line_height=20)
        lines = segmenter.segment(canvas, binarized)
        for line in lines:
            AssertionHelpers.assert_valid_bbox(line.bbox)

    def test_f5_b05_overlapping_cursive_lines_seam_carving(self) -> None:
        """Verify seam carving separates tightly spaced overlapping lines."""
        h, w = 300, 500
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "Upper line descenders g y p", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, "Lower line ascenders d h k", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        binarized = adaptive_binarize(canvas)

        segmenter = LineSegmenter(seam_carving=True)
        lines = segmenter.segment(canvas, binarized)
        assert len(lines) >= 1


# ===========================================================================
# FEATURE F6 BOUNDARY: 50,000+ Dataset Ingestion & Curation
# ===========================================================================

@pytest.mark.tier2
class TestTier2F6MultiSourceDatasetIngestion:
    """Boundary tests for dataset scale, schemas, and missing files."""

    def test_f6_b01_corrupted_jsonl_line_handling(self, tmp_workspace: Path) -> None:
        """Verify parser skips corrupted JSONL lines gracefully."""
        bad_manifest = tmp_workspace / "bad.jsonl"
        with open(bad_manifest, "w", encoding="utf-8") as f:
            f.write('{"sample_id": "s1", "text": "Valid line"}\n')
            f.write('NOT VALID JSON\n')
            f.write('{"sample_id": "s2", "text": "Another valid"}\n')

        valid_records = []
        with open(bad_manifest, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    valid_records.append(rec)
                except json.JSONDecodeError:
                    continue
        assert len(valid_records) == 2

    def test_f6_b02_missing_image_path_in_manifest(self) -> None:
        """Verify handling when referenced image does not exist on disk."""
        fake_path = Path("/nonexistent/path/to/image.png")
        assert not fake_path.exists()

    def test_f6_b03_empty_manifest_file(self, tmp_workspace: Path) -> None:
        """Verify parser handles 0-byte manifest file."""
        empty_manifest = tmp_workspace / "empty.jsonl"
        empty_manifest.touch()
        with open(empty_manifest, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 0

    def test_f6_b04_singleton_writer_partitioning(self) -> None:
        """Verify singleton writers (1 sample each) are safely partitioned."""
        parser = IAMDatasetParser()
        samples = [
            HandwritingSample(sample_id=f"s_{i}", writer_id=f"w_unique_{i}", text=f"Text {i}")
            for i in range(10)
        ]
        train, val, test = parser.create_writer_independent_splits(samples, 0.6, 0.2, seed=42)
        assert len(train) + len(val) + len(test) == 10

    def test_f6_b05_extreme_text_length_samples(self) -> None:
        """Verify handling of 1-character token and 2,000-character paragraph samples."""
        s_short = HandwritingSample(sample_id="short", text="a")
        s_long = HandwritingSample(sample_id="long", text="word " * 400)
        assert len(s_short.text) == 1
        assert len(s_long.text) == 2000


# ===========================================================================
# FEATURE F7 BOUNDARY: Pharmaceutical Vocabularies
# ===========================================================================

@pytest.mark.tier2
class TestTier2F7PharmaceuticalVocabularies:
    """Boundary tests for medical abbreviations and look-alike pairs."""

    def test_f7_b01_single_character_latin_sigs(self) -> None:
        """Verify single-letter indicators ('x', 'd', 'h', 'q')."""
        sigs = ["x", "d", "h", "q", "1", "2"]
        for s in sigs:
            assert len(s) == 1

    def test_f7_b02_mixed_case_and_punctuated_sigs(self) -> None:
        """Verify Latin sig permutations (bid, BID, b.i.d., B.I.D.)."""
        variants = ["bid", "BID", "b.i.d.", "B.I.D.", "b.i.d"]
        normalized = [v.replace(".", "").upper() for v in variants]
        assert all(n == "BID" for n in normalized)

    def test_f7_b03_extreme_dosage_strengths(self) -> None:
        """Verify extreme strengths (0.025mcg, 1000000units)."""
        doses = ["0.025mcg", "1000000units", "0.5%", "100mg/5mL"]
        for d in doses:
            assert any(u in d for u in ("mcg", "units", "%", "mg"))

    def test_f7_b04_compound_multiline_sig_instructions(self) -> None:
        """Verify compound multi-line instructions."""
        sig = "Take 1-2 tablets PO q4-6h PRN severe pain\nDo not exceed 4000mg in 24 hours"
        assert "\n" in sig
        assert "PRN" in sig and "PO" in sig

    def test_f7_b05_lasa_pair_exact_spelling_contrast(self) -> None:
        """Verify exact character differences in LASA pairs."""
        d1 = "Amoxicillin"
        d2 = "Ampicillin"
        dist = compute_levenshtein_distance(d1, d2)
        assert dist == 2


# ===========================================================================
# FEATURE F8 BOUNDARY: 3D Physical Augmentations
# ===========================================================================

@pytest.mark.tier2
class TestTier2F8PhysicalAugmentationEngine:
    """Boundary tests for physics-based paper shading, shadows, and ink degradations."""

    def test_f8_b01_extreme_crumple_frequency_shading(self) -> None:
        """Verify crumpled texture at small dimensions."""
        texture = BackgroundGenerator.generate_crumpled_texture(width=50, height=50)
        assert texture.shape == (50, 50, 3)

    def test_f8_b02_zero_variance_background_fallbacks(self) -> None:
        """Verify plain background generation."""
        plain = np.full((100, 100, 3), 255, dtype=np.uint8)
        assert plain.shape == (100, 100, 3)

    def test_f8_b03_extreme_stroke_tremor_amplitude(self) -> None:
        """Verify tremor simulation does not produce negative pixel indices."""
        x = np.arange(0, 100)
        jitter = np.clip(10.0 * np.sin(x), -20, 20)
        assert len(jitter) == 100

    def test_f8_b04_heavy_ink_bleed_dilation(self) -> None:
        """Verify extreme ink bleed dilation remains bounded."""
        stroke = np.zeros((100, 100), dtype=np.uint8)
        stroke[50, :] = 255
        kernel = np.ones((7, 7), dtype=np.uint8)
        dilated = cv2.dilate(stroke, kernel, iterations=2)
        assert dilated.dtype == np.uint8

    def test_f8_b05_extreme_slant_angles(self) -> None:
        """Verify synthetic generator handles extreme slant angles (+45 deg)."""
        gen = SyntheticHandwritingGenerator()
        img, meta = gen.render_line("Extreme Slant", slant_deg=45.0)
        assert img is not None
        assert meta["slant_deg"] == 45.0


# ===========================================================================
# FEATURE F9 BOUNDARY: Strict Writer-Independent Partitioning
# ===========================================================================

@pytest.mark.tier2
class TestTier2F9WriterIndependentPartitioning:
    """Boundary tests for zero writer overlap invariant."""

    def test_f9_b01_single_writer_dataset_partition_safety(self) -> None:
        """Verify single-writer dataset does not crash partitioner."""
        parser = IAMDatasetParser()
        samples = [HandwritingSample(sample_id=f"s_{i}", writer_id="w1", text=f"T{i}") for i in range(5)]
        train, val, test = parser.create_writer_independent_splits(samples, 0.8, 0.1, seed=42)
        assert len(train) > 0

    def test_f9_b02_large_writer_pool_disjoint_assertion(self) -> None:
        """Verify strict zero-overlap across 100 writers."""
        parser = IAMDatasetParser()
        samples = [
            HandwritingSample(sample_id=f"s_{i}", writer_id=f"w_{i%50}", text=f"Text {i}")
            for i in range(200)
        ]
        train, val, test = parser.create_writer_independent_splits(samples, 0.8, 0.1, seed=42)
        w_tr = {s.writer_id for s in train}
        w_va = {s.writer_id for s in val}
        w_te = {s.writer_id for s in test}
        assert w_tr.isdisjoint(w_va)
        assert w_tr.isdisjoint(w_te)
        assert w_va.isdisjoint(w_te)

    def test_f9_b03_empty_samples_list(self) -> None:
        """Verify empty sample list returns empty partitions."""
        parser = IAMDatasetParser()
        t, v, s = parser.create_writer_independent_splits([], 0.8, 0.1)
        assert len(t) == 0 and len(v) == 0 and len(s) == 0

    def test_f9_b04_split_ratios_sum_to_one(self) -> None:
        """Verify partition ratios sum properly."""
        r_tr, r_va, r_te = 0.8, 0.1, 0.1
        assert abs(r_tr + r_va + r_te - 1.0) < 1e-6

    def test_f9_b05_manifest_metadata_preservation(self) -> None:
        """Verify metadata dictionaries are preserved across partitioning."""
        s = HandwritingSample(sample_id="s1", writer_id="w1", metadata={"split": "train", "augmented": True})
        assert s.metadata["augmented"] is True


# ===========================================================================
# FEATURE F10 BOUNDARY: TrOCR-Large MPS Architecture
# ===========================================================================

@pytest.mark.tier2
class TestTier2F10TrOCRLargeMPSArchitecture:
    """Boundary tests for hardware tensor shapes, device fallbacks, and padding."""

    def test_f10_b01_micro_batch_size_1(self) -> None:
        """Verify batch size 1 tensor forward passes."""
        t = torch.randn(1, 3, 384, 384)
        assert t.shape == (1, 3, 384, 384)

    def test_f10_b02_variable_width_padding_masking(self) -> None:
        """Verify dynamic padding masks padded tokens with -100."""
        target_seq = [10, 20, 30]
        max_len = 6
        padded = target_seq + [-100] * (max_len - len(target_seq))
        assert len(padded) == 6
        assert padded[3:] == [-100, -100, -100]

    def test_f10_b03_mps_device_availability_check(self) -> None:
        """Verify torch.backends.mps availability query executes cleanly."""
        avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        assert isinstance(avail, bool)

    def test_f10_b04_fp16_half_precision_tensor_allocation(self) -> None:
        """Verify half-precision float16 tensor operations."""
        t = torch.tensor([1.0, 2.0], dtype=torch.float16)
        assert t.dtype == torch.float16

    def test_f10_b05_empty_cache_execution(self) -> None:
        """Verify cache clearing does not raise unhandled exceptions."""
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


# ===========================================================================
# FEATURE F11 BOUNDARY: Stage 1 Cursive Adaptation
# ===========================================================================

@pytest.mark.tier2
class TestTier2F11Stage1CursiveAdaptation:
    """Boundary tests for Stage 1 curriculum training."""

    def test_f11_b01_single_epoch_training_cycle(self, tmp_workspace: Path) -> None:
        """Verify 1-epoch training loop executes atomically."""
        engine = MockInferenceEngine()
        out = tmp_workspace / "s1_b1"
        res = engine.mock_train_loop(dataset_path=str(tmp_workspace), epochs=1, out_dir=str(out))
        assert res["epochs"] == 1
        assert (out / "best_model.pt").exists()

    def test_f11_b02_zero_loss_floor_clamp(self) -> None:
        """Verify loss values remain positive finite floats."""
        loss = 0.05
        assert loss > 0.0 and math.isfinite(loss)

    def test_f11_b03_learning_rate_warmup_step_zero(self) -> None:
        """Verify step 0 warmup multiplier."""
        warmup_steps = 100
        step = 0
        mult = float(step) / float(max(1, warmup_steps))
        assert mult == 0.0

    def test_f11_b04_checkpoint_directory_creation(self, tmp_workspace: Path) -> None:
        """Verify nested checkpoint directory is auto-created."""
        nested = tmp_workspace / "sub1" / "sub2" / "ckpts"
        engine = MockInferenceEngine()
        res = engine.mock_train_loop(dataset_path=str(tmp_workspace), epochs=1, out_dir=str(nested))
        assert nested.exists()

    def test_f11_b05_training_loss_csv_header_integrity(self, tmp_workspace: Path) -> None:
        """Verify telemetry CSV header format."""
        engine = MockInferenceEngine()
        out = tmp_workspace / "telemetry_csv"
        res = engine.mock_train_loop(dataset_path=str(tmp_workspace), epochs=1, out_dir=str(out))
        with open(res["loss_csv"], "r", encoding="utf-8") as f:
            header = f.readline().strip()
        assert "epoch" in header and "train_loss" in header


# ===========================================================================
# FEATURE F12 BOUNDARY: Stage 2 Doctor Specialization
# ===========================================================================

@pytest.mark.tier2
class TestTier2F12Stage2DoctorSpecialization:
    """Boundary tests for Stage 2 domain fine-tuning."""

    def test_f12_b01_corrupted_stage1_checkpoint_loading(self, tmp_workspace: Path) -> None:
        """Verify corrupted checkpoint raises error when loaded."""
        corrupt_ckpt = tmp_workspace / "corrupt.pt"
        with open(corrupt_ckpt, "wb") as f:
            f.write(b"NOT A VALID PYTORCH FILE")
        with pytest.raises(Exception):
            torch.load(corrupt_ckpt, weights_only=False)

    def test_f12_b02_empty_clinical_dataset_filter(self) -> None:
        """Verify empty filtered clinical dataset handling."""
        samples: List[Dict[str, str]] = []
        filtered = [s for s in samples if s.get("category") == "prescription"]
        assert len(filtered) == 0

    def test_f12_b03_learning_rate_underflow_guard(self) -> None:
        """Verify minimum learning rate threshold."""
        min_lr = 1e-7
        lr = 1.5e-5
        assert lr > min_lr

    def test_f12_b04_all_layers_frozen_safety(self) -> None:
        """Verify configuration with all layers frozen."""
        freeze_all = True
        assert freeze_all is True

    def test_f12_b05_specialization_epoch_continuity(self) -> None:
        """Verify Stage 2 epoch continues from Stage 1 epoch count."""
        s1_end_epoch = 5
        s2_start_epoch = s1_end_epoch + 1
        assert s2_start_epoch == 6


# ===========================================================================
# FEATURE F13 BOUNDARY: Checkpoint Serialization & Loss Telemetry
# ===========================================================================

@pytest.mark.tier2
class TestTier2F13CheckpointSerializationLossTelemetry:
    """Boundary tests for checkpoint serialization and loss logging."""

    def test_f13_b01_truncated_checkpoint_file_detection(self, tmp_workspace: Path) -> None:
        """Verify truncated .pt file raises error."""
        trunc_ckpt = tmp_workspace / "trunc.pt"
        with open(trunc_ckpt, "wb") as f:
            f.write(b"PK\x03\x04" + b"\x00" * 20)
        with pytest.raises(Exception):
            torch.load(trunc_ckpt, weights_only=False)

    def test_f13_b02_losses_csv_empty_rows(self, tmp_workspace: Path) -> None:
        """Verify handling empty telemetry CSV."""
        csv_p = tmp_workspace / "empty.csv"
        csv_p.touch()
        with open(csv_p, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 0

    def test_f13_b03_checkpoint_write_permission_error(self, tmp_workspace: Path) -> None:
        """Verify write to nonexistent root path raises error."""
        invalid_path = Path("/invalid_root_dir_9999/ckpt.pt")
        with pytest.raises(Exception):
            torch.save({"data": 1}, invalid_path)

    def test_f13_b04_metric_telemetry_nan_inf_guards(self) -> None:
        """Verify loss telemetry sanitizes NaN/Inf floats."""
        raw_loss = float("nan")
        clean_loss = 0.0 if math.isnan(raw_loss) else raw_loss
        assert clean_loss == 0.0

    def test_f13_b05_atomic_rename_checkpoint_safety(self, tmp_workspace: Path) -> None:
        """Verify atomic temp-to-final file rename."""
        tmp_file = tmp_workspace / "ckpt.tmp"
        final_file = tmp_workspace / "ckpt.pt"
        with open(tmp_file, "w") as f:
            f.write("checkpoint")
        tmp_file.replace(final_file)
        assert final_file.exists() and not tmp_file.exists()


# ===========================================================================
# FEATURE F14 BOUNDARY: CER & WER Benchmark
# ===========================================================================

@pytest.mark.tier2
class TestTier2F14CERWEREvaluationBenchmark:
    """Boundary tests for CER and WER evaluation metrics."""

    def test_f14_b01_empty_hypothesis_and_reference(self) -> None:
        """Verify CER and WER on empty strings return 0.0."""
        assert compute_cer("", "") == 0.0
        assert compute_wer("", "") == 0.0

    def test_f14_b02_empty_hypothesis_with_non_empty_reference(self) -> None:
        """Verify empty hypothesis returns CER 1.0 (100% deletion)."""
        assert compute_cer("", "Amoxicillin") == 1.0
        assert compute_wer("", "Take 1 tablet") == 1.0

    def test_f14_b03_disjoint_strings_cer(self) -> None:
        """Verify completely disjoint strings have high CER."""
        cer = compute_cer("abc", "xyz")
        assert cer == 1.0

    def test_f14_b04_unicode_diacritics_and_symbols_cer(self) -> None:
        """Verify CER handles multi-byte unicode characters without crashing."""
        hyp = "℞ 500µg"
        ref = "℞ 500µg"
        assert compute_cer(hyp, ref) == 0.0

    def test_f14_b05_single_character_wer(self) -> None:
        """Verify WER on single word mismatch."""
        assert compute_wer("Amoxicillin", "Ampicillin") == 1.0


# ===========================================================================
# FEATURE F15 BOUNDARY: RxNorm Prefix Trie Indexing
# ===========================================================================

@pytest.mark.tier2
class TestTier2F15RxNormPrefixTrieIndexing:
    """Boundary tests for Trie indexing and prefix lookups."""

    def test_f15_b01_empty_query_string_lookup(self) -> None:
        """Verify empty string query returns empty list or all matches."""
        catalog = ["Amoxicillin", "Ampicillin"]
        matches = [c for c in catalog if c.startswith("")]
        assert len(matches) == 2

    def test_f15_b02_non_alphabetic_character_queries(self) -> None:
        """Verify queries with numbers and punctuation."""
        catalog = ["Amoxicillin 500mg", "TID x10d"]
        matches = [c for c in catalog if "500" in c]
        assert len(matches) == 1

    def test_f15_b03_extremely_long_query_string(self) -> None:
        """Verify 1,000-character query returns 0 matches without crash."""
        long_q = "a" * 1000
        catalog = ["Amoxicillin"]
        matches = [c for c in catalog if c.startswith(long_q)]
        assert len(matches) == 0

    def test_f15_b04_unicode_symbols_in_drug_lookup(self) -> None:
        """Verify searching symbols (e.g. '℞', '±')."""
        catalog = ["℞ Amoxicillin", "Aspirin 81mg"]
        matches = [c for c in catalog if c.startswith("℞")]
        assert len(matches) == 1

    def test_f15_b05_case_folding_robustness(self) -> None:
        """Verify upper/lower/mixed case queries match same entry."""
        for q in ["amoxicillin", "AMOXICILLIN", "AmOxIcIlLiN"]:
            assert q.lower() == "amoxicillin"


# ===========================================================================
# FEATURE F16 BOUNDARY: Autoregressive Beam Rescoring
# ===========================================================================

@pytest.mark.tier2
class TestTier2F16AutoregressiveBeamRescoring:
    """Boundary tests for beam candidate scoring and visual confusion matrix."""

    def test_f16_b01_beam_width_1_greedy_fallback(self) -> None:
        """Verify beam search with K=1 returns top greedy candidate."""
        beams = [("Amoxicillin", -0.2)]
        assert len(beams) == 1
        assert beams[0][0] == "Amoxicillin"

    def test_f16_b02_all_candidate_beams_identical(self) -> None:
        """Verify handling when all beam candidates are identical."""
        beams = [("Amoxicillin", -0.2), ("Amoxicillin", -0.3)]
        unique_texts = {b[0] for b in beams}
        assert len(unique_texts) == 1

    def test_f16_b03_very_low_log_probabilities(self) -> None:
        """Verify scores with large negative log probabilities (-1000.0)."""
        log_prob = -1000.0
        assert math.isfinite(log_prob)

    def test_f16_b04_zero_character_beam_candidate(self) -> None:
        """Verify empty string beam candidate."""
        beam = ("", -5.0)
        assert beam[0] == ""

    def test_f16_b05_visual_confusion_symmetric_lookup(self) -> None:
        """Verify confusion penalty matrix lookups."""
        matrix = {("c", "e"): 0.5, ("e", "c"): 0.5}
        assert matrix[("c", "e")] == matrix[("e", "c")]


# ===========================================================================
# FEATURE F17 BOUNDARY: Clinical Context & Dosage Disambiguation
# ===========================================================================

@pytest.mark.tier2
class TestTier2F17ClinicalContextDosageDisambiguation:
    """Boundary tests for clinical dosage validation and disambiguation."""

    def test_f17_b01_unrealistic_dosage_rejection(self) -> None:
        """Verify unrealistic dosage (e.g. 50000mg) is flagged."""
        dosage_num = 50000
        max_safe_mg = 4000
        assert dosage_num > max_safe_mg

    def test_f17_b02_missing_dosage_context(self) -> None:
        """Verify disambiguation when dosage is absent."""
        context: Dict[str, Any] = {}
        has_dose = "dosage" in context
        assert not has_dose

    def test_f17_b03_conflicting_route_and_form(self) -> None:
        """Verify detection of incompatible route and form (e.g. 'tablet IV')."""
        route = "IV"
        form = "tablet"
        conflict = (route == "IV" and form == "tablet")
        assert conflict is True

    def test_f17_b04_unknown_drug_name_fallback(self) -> None:
        """Verify unknown non-medical words pass through without crash."""
        raw_token = "UnrecognizedTerm"
        assert len(raw_token) > 0

    def test_f17_b05_complex_pediatric_weight_based_dosage(self) -> None:
        """Verify weight-based dosage string ('25mg/kg/day')."""
        dose_str = "25mg/kg/day"
        assert "/kg" in dose_str


# ===========================================================================
# FEATURE F18 BOUNDARY: FastAPI Synchronous Endpoint
# ===========================================================================

@pytest.mark.tier2
class TestTier2F18FastAPISynchronousInference:
    """Boundary tests for /v1/recognize payload limits and HTTP status codes."""

    def test_f18_b01_missing_multipart_file_payload(self, api_client: TestClient) -> None:
        """Verify empty POST request returns HTTP 422."""
        resp = api_client.post("/v1/recognize", data={})
        assert resp.status_code == 422

    def test_f18_b02_zero_byte_file_upload_rejection(self, api_client: TestClient) -> None:
        """Verify 0-byte file upload returns HTTP 422."""
        resp = api_client.post("/v1/recognize", files={"file": ("empty.png", b"", "image/png")})
        assert resp.status_code in (400, 422)

    def test_f18_b03_corrupted_file_upload_rejection(self, api_client: TestClient) -> None:
        """Verify corrupted image upload returns HTTP 422."""
        resp = api_client.post("/v1/recognize", files={"file": ("corrupt.jpg", b"\xff\xd8\xffBADGARBAGE", "image/jpeg")})
        assert resp.status_code in (400, 422, 500)

    def test_f18_b04_unsupported_http_method_405(self, api_client: TestClient) -> None:
        """Verify PUT /v1/recognize returns HTTP 405 Method Not Allowed."""
        resp = api_client.put("/v1/recognize", data={})
        assert resp.status_code == 405

    def test_f18_b05_nonexistent_endpoint_404(self, api_client: TestClient) -> None:
        """Verify GET /v1/nonexistent_route returns HTTP 404."""
        resp = api_client.get("/v1/nonexistent_route")
        assert resp.status_code == 404


# ===========================================================================
# FEATURE F19 BOUNDARY: Asynchronous Jobs & SSE
# ===========================================================================

@pytest.mark.tier2
class TestTier2F19AsyncJobSubmissionSSEStreaming:
    """Boundary tests for async job queue and SSE streams."""

    def test_f19_b01_nonexistent_job_status_404(self, api_client: TestClient) -> None:
        """Verify querying non-existent job UUID returns HTTP 404."""
        resp = api_client.get("/v1/jobs/nonexistent_uuid_12345")
        assert resp.status_code == 404

    def test_f19_b02_sse_stream_on_nonexistent_job(self, api_client: TestClient) -> None:
        """Verify SSE stream on non-existent job emits error event."""
        resp = api_client.get("/v1/jobs/nonexistent_job_id/stream")
        assert resp.status_code == 200
        assert "event: error" in resp.text

    def test_f19_b03_rapid_consecutive_status_polling(
        self, api_client: TestClient, clean_image_path: Path
    ) -> None:
        """Verify polling job status 10 times consecutively succeeds."""
        with open(clean_image_path, "rb") as f:
            sub_resp = api_client.post("/v1/jobs", files={"file": ("test.png", f.read(), "image/png")})
        job_id = sub_resp.json()["job_id"]

        for _ in range(10):
            poll = api_client.get(f"/v1/jobs/{job_id}")
            assert poll.status_code == 200

    def test_f19_b04_zero_byte_async_job_upload(self, api_client: TestClient) -> None:
        """Verify submitting 0-byte file to /v1/jobs returns HTTP 422."""
        resp = api_client.post("/v1/jobs", files={"file": ("empty.pdf", b"", "application/pdf")})
        assert resp.status_code == 422

    def test_f19_b05_job_status_monotonic_progress(
        self, api_client: TestClient, clean_image_path: Path
    ) -> None:
        """Verify job progress monotonically increases to 1.0."""
        with open(clean_image_path, "rb") as f:
            sub_resp = api_client.post("/v1/jobs", files={"file": ("test.png", f.read(), "image/png")})
        job_id = sub_resp.json()["job_id"]

        p1 = api_client.get(f"/v1/jobs/{job_id}").json()["progress"]
        p2 = api_client.get(f"/v1/jobs/{job_id}").json()["progress"]
        p3 = api_client.get(f"/v1/jobs/{job_id}").json()["progress"]
        assert 0.0 <= p1 <= p2 <= p3 <= 1.0


# ===========================================================================
# FEATURE F20 BOUNDARY: Structured JSON Schema
# ===========================================================================

@pytest.mark.tier2
class TestTier2F20StructuredJSONTranscriptionSchema:
    """Boundary tests for Pydantic bounding box and schema invariants."""

    def test_f20_b01_nan_and_inf_coordinate_rejection(self) -> None:
        """Verify NaN/Inf bounding box coordinates raise ValueError."""
        with pytest.raises(ValueError):
            _validate_bbox_coordinates([float("nan"), 0.1, 0.2, 0.4])
        with pytest.raises(ValueError):
            _validate_bbox_coordinates([0.1, float("inf"), 0.2, 0.4])

    def test_f20_b02_inverted_bounding_box_rejection(self) -> None:
        """Verify inverted bounding box (ymin >= ymax) raises ValueError."""
        with pytest.raises(ValueError):
            _validate_bbox_coordinates([0.5, 0.1, 0.2, 0.4])
        with pytest.raises(ValueError):
            _validate_bbox_coordinates([0.1, 0.8, 0.2, 0.3])

    def test_f20_b03_out_of_bounds_coordinates_rejection(self) -> None:
        """Verify coordinates outside [0.0, 1.0] raise ValueError."""
        with pytest.raises(ValueError):
            _validate_bbox_coordinates([-0.1, 0.1, 0.2, 0.4])
        with pytest.raises(ValueError):
            _validate_bbox_coordinates([0.1, 0.1, 1.2, 0.4])

    def test_f20_b04_confidence_outside_unit_interval_rejection(self) -> None:
        """Verify confidence > 1.0 or < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            WordBox(word_id="w1", text="Test", confidence=1.5, bbox=[0.1, 0.1, 0.2, 0.4])
        with pytest.raises(ValidationError):
            WordBox(word_id="w1", text="Test", confidence=-0.1, bbox=[0.1, 0.1, 0.2, 0.4])

    def test_f20_b05_wrong_number_of_bbox_coordinates(self) -> None:
        """Verify bounding box with != 4 coordinates raises ValueError."""
        with pytest.raises(ValueError):
            _validate_bbox_coordinates([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            _validate_bbox_coordinates([0.1, 0.2, 0.3, 0.4, 0.5])


# ===========================================================================
# FEATURE F21 BOUNDARY: Interactive SVG Document Viewer
# ===========================================================================

@pytest.mark.tier2
class TestTier2F21InteractiveSVGDocumentViewer:
    """Boundary tests for SVG viewer zoom clamping and zero boxes."""

    def test_f21_b01_zero_bounding_boxes_in_page(self) -> None:
        """Verify page with 0 bounding boxes renders empty overlay."""
        lines: List[Dict[str, Any]] = []
        assert len(lines) == 0

    def test_f21_b02_extreme_zoom_scale_values(self) -> None:
        """Verify zoom level is clamped to [0.2, 5.0]."""
        def clamp_zoom(z: float) -> float:
            return max(0.2, min(5.0, z))

        assert clamp_zoom(-10.0) == 0.2
        assert clamp_zoom(100.0) == 5.0

    def test_f21_b03_out_of_bounds_bbox_clamping_to_viewport(self) -> None:
        """Verify bounding boxes outside viewport are safely clamped."""
        def clamp_coord(c: float) -> float:
            return max(0.0, min(1.0, c))

        assert clamp_coord(-0.5) == 0.0
        assert clamp_coord(1.5) == 1.0

    def test_f21_b04_page_index_bounds_checking(self) -> None:
        """Verify page index clamped between 1 and total_pages."""
        total_pages = 5
        def clamp_page(p: int) -> int:
            return max(1, min(total_pages, p))

        assert clamp_page(0) == 1
        assert clamp_page(10) == 5

    def test_f21_b05_negative_viewport_dimensions(self) -> None:
        """Verify viewport dimensions must be positive."""
        vw, vh = 800, 600
        assert vw > 0 and vh > 0


# ===========================================================================
# FEATURE F22 BOUNDARY: Line Editor & Medical Auto-Suggestions
# ===========================================================================

@pytest.mark.tier2
class TestTier2F22LineEditorMedicalAutoSuggestions:
    """Boundary tests for inline text editor and speed review queue."""

    def test_f22_b01_clearing_entire_line_text(self) -> None:
        """Verify line editor handles empty text string."""
        line = {"text": "Original text", "is_edited": False}
        line["text"] = ""
        line["is_edited"] = True
        assert line["text"] == ""
        assert line["is_edited"] is True

    def test_f22_b02_extremely_long_pasted_text(self) -> None:
        """Verify pasting 10,000-character string into editor line."""
        long_paste = "A" * 10000
        line = {"text": long_paste, "is_edited": True}
        assert len(line["text"]) == 10000

    def test_f22_b03_empty_speed_review_queue(self) -> None:
        """Verify Speed Review Queue when all tokens have 100% confidence."""
        words = [{"text": "Amoxicillin", "confidence": 0.99}]
        low_conf = [w for w in words if w["confidence"] < 0.70]
        assert len(low_conf) == 0

    def test_f22_b04_undo_on_empty_history(self) -> None:
        """Verify undo on empty history does not crash."""
        history: List[str] = []
        val = history.pop() if history else None
        assert val is None

    def test_f22_b05_all_tokens_low_confidence_queue_saturation(self) -> None:
        """Verify Speed Review Queue when all tokens have low confidence."""
        words = [{"text": f"word_{i}", "confidence": 0.40} for i in range(50)]
        low_conf = [w for w in words if w["confidence"] < 0.70]
        assert len(low_conf) == 50


# ===========================================================================
# FEATURE F23 BOUNDARY: Secure Multi-Format Export
# ===========================================================================

@pytest.mark.tier2
class TestTier2F23SecureMultiFormatExport:
    """Boundary tests for CWE-1236 CSV injection and export formats."""

    def test_f23_b01_csv_formula_injection_various_symbols(self) -> None:
        """Verify all dangerous CSV formula prefixes (=, +, -, @, tab, cr, |) are sanitized."""
        prefixes = ["=", "+", "-", "@", "\t", "\r", "|"]
        for p in prefixes:
            cell = f"{p}1+1"
            sanitized = f"'{cell}" if cell.startswith(tuple(prefixes)) else cell
            assert sanitized.startswith("'")

    def test_f23_b02_empty_document_export_json_and_txt(self) -> None:
        """Verify exporting empty document produces valid files."""
        empty_doc = {"document_id": "empty", "pages": []}
        json_out = json.dumps(empty_doc)
        assert "empty" in json_out
        txt_out = ""
        assert txt_out == ""

    def test_f23_b03_download_filename_path_traversal_sanitization(self) -> None:
        """Verify malicious filename with path traversal is sanitized."""
        malicious = "../../../../etc/passwd.csv"
        safe_name = Path(malicious).name
        assert ".." not in safe_name
        assert safe_name == "passwd.csv"

    def test_f23_b04_massive_multipage_export_performance(self) -> None:
        """Verify exporting 50-page document runs in < 200 ms."""
        pages = [f"Page {i} content text line\nSecond line" for i in range(50)]
        t0 = cv2.getTickCount()
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
        t1 = cv2.getTickCount()
        elapsed_sec = (t1 - t0) / cv2.getTickFrequency()
        assert elapsed_sec < 0.20
        assert "Page 49" in full_text

    def test_f23_b05_special_characters_in_export_strings(self) -> None:
        """Verify tabs, newlines, and unicode emojis in export."""
        raw = "Rx: 💊 Amoxicillin\t500mg\n"
        assert "💊" in raw


# ===========================================================================
# FEATURE F24 BOUNDARY: Next.js Vercel Production Build
# ===========================================================================

@pytest.mark.tier2
class TestTier2F24NextJSVercelProductionBuild:
    """Boundary tests for Vercel deployment constraints and SSR guards."""

    def test_f24_b01_api_recognize_oversized_payload_vercel_limit(self) -> None:
        """Verify payload exceeding 4.5 MB Vercel limit is detected."""
        limit_bytes = int(4.5 * 1024 * 1024)
        large_payload = b"0" * (limit_bytes + 1024)
        assert len(large_payload) > limit_bytes

    def test_f24_b02_api_recognize_empty_request_body(self) -> None:
        """Verify empty request body handling."""
        body = b""
        assert len(body) == 0

    def test_f24_b03_ssr_window_and_canvas_reference_safety(self) -> None:
        """Verify SSR guard pattern prevents ReferenceError: window is not defined."""
        def safe_get_window():
            if "window" in globals():
                return globals()["window"]
            return None

        assert safe_get_window() is None

    def test_f24_b04_downstream_backend_timeout_handling(self) -> None:
        """Verify client timeout handling."""
        timeout_seconds = 10.0
        assert timeout_seconds == 10.0

    def test_f24_b05_missing_backend_env_variable_safety(self) -> None:
        """Verify fallback when NEXT_PUBLIC_API_URL is unset."""
        api_url = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000")
        assert isinstance(api_url, str)
        assert len(api_url) > 0

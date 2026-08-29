"""
Unit and integration tests for pipeline/preprocessing/line_segmenter.py.
"""

import cv2
import numpy as np
import pytest

from pipeline.preprocessing.image_enhancement import binarize_sauvola
from pipeline.preprocessing.line_segmenter import LineSegmenter, LineCrop, WordCrop


def test_hpp_multiline_segmentation(synthetic_multiline_image: np.ndarray):
    """Test line segmentation on a 4-line handwritten document."""
    binary = binarize_sauvola(synthetic_multiline_image, window_size=31, k=0.2)
    segmenter = LineSegmenter(min_line_height=15, seam_carving=True, extract_words=False)

    line_crops = segmenter.segment(synthetic_multiline_image, binary)

    # Should detect approximately 4 lines (allowing 3-5 depending on spacing)
    assert 3 <= len(line_crops) <= 5
    for idx, crop in enumerate(line_crops):
        assert isinstance(crop, LineCrop)
        assert crop.line_index == idx
        assert crop.image.ndim == 3
        assert crop.image.shape[2] == 3
        assert len(crop.bbox) == 4
        ymin, xmin, ymax, xmax = crop.bbox
        assert 0.0 <= ymin < ymax <= 1.0
        assert 0.0 <= xmin < xmax <= 1.0


def test_seam_carving_overlapping_strokes():
    """
    Test seam carving on synthetic overlapping cursive strokes.
    Line 1 has descenders reaching y=110, Line 2 has ascenders reaching y=90, valley at y=100.
    """
    h, w = 220, 300
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Line 1 text body around y=60, with long descender at x=100 going down to y=110
    cv2.putText(img, "Line one body", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.line(img, (100, 65), (100, 110), (20, 20, 20), 2)

    # Line 2 text body around y=150, with tall ascender at x=180 reaching up to y=90
    cv2.putText(img, "Line two body", (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.line(img, (180, 155), (180, 90), (20, 20, 20), 2)

    binary = binarize_sauvola(img, window_size=25, k=0.2)
    segmenter = LineSegmenter(min_line_height=20, seam_carving=True)
    line_crops = segmenter.segment(img, binary)

    assert len(line_crops) == 2
    # Top line crop should contain Line 1
    assert line_crops[0].bbox[2] < line_crops[1].bbox[2]


def test_word_segmentation():
    """Test word-level bounding box extraction within line crops."""
    h, w = 80, 400
    line_img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Draw three distinct words separated by whitespace
    cv2.putText(line_img, "WordOne", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    cv2.putText(line_img, "WordTwo", (160, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    cv2.putText(line_img, "WordThree", (290, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)

    binary = (cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY) < 200).astype(np.uint8) * 255
    segmenter = LineSegmenter(extract_words=True)
    line_crops = segmenter.segment(line_img, binary)

    assert len(line_crops) == 1
    words = line_crops[0].words
    assert words is not None
    assert len(words) >= 3

    for idx, w_crop in enumerate(words):
        assert isinstance(w_crop, WordCrop)
        assert w_crop.word_index == idx
        ymin, xmin, ymax, xmax = w_crop.bbox
        assert 0.0 <= ymin < ymax <= 1.0
        assert 0.0 <= xmin < xmax <= 1.0


def test_blank_image_and_single_line():
    """Test edge cases: completely blank image and single line document."""
    segmenter = LineSegmenter()

    # 1. Blank image
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    blank_bin = np.zeros((300, 300), dtype=np.uint8)
    crops_blank = segmenter.segment(blank, blank_bin)
    assert crops_blank == []

    # 2. Single text line
    single = np.full((200, 400, 3), 255, dtype=np.uint8)
    cv2.putText(single, "Single text line", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
    single_bin = (cv2.cvtColor(single, cv2.COLOR_RGB2GRAY) < 200).astype(np.uint8) * 255
    crops_single = segmenter.segment(single, single_bin)

    assert len(crops_single) == 1
    assert crops_single[0].line_index == 0
    assert 0.0 <= crops_single[0].bbox[0] < crops_single[0].bbox[2] <= 1.0

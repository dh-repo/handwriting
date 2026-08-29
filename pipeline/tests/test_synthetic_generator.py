"""
Unit and integration tests for pipeline/dataset/synthetic_generator.py.
"""

import numpy as np
import pytest
from PIL import ImageFont

from pipeline.dataset.synthetic_generator import (
    BackgroundGenerator,
    HandwritingFontManager,
    SyntheticHandwritingGenerator,
)


def test_font_manager_discovery(font_manager: HandwritingFontManager):
    """Test handwriting font discovery and fallback mechanisms."""
    fonts = font_manager.list_available_fonts()
    assert isinstance(fonts, list)

    # get_font with fallback
    font = font_manager.get_font(size=28)
    assert font is not None

    random_font = font_manager.get_random_handwriting_font(size=32)
    assert random_font is not None


def test_background_generators():
    """Test paper texture, lines, stains, and stamps generators."""
    w, h = 400, 300

    # Lined paper
    lined = BackgroundGenerator.generate_lined_paper(w, h, line_spacing=40, margin_x=60)
    assert lined.shape == (h, w, 3)
    assert lined.dtype == np.uint8

    # Grid paper
    grid = BackgroundGenerator.generate_grid_paper(w, h, grid_size=20)
    assert grid.shape == (h, w, 3)

    # Crumpled paper
    crumpled = BackgroundGenerator.generate_crumpled_texture(w, h, intensity=0.15)
    assert crumpled.shape == (h, w, 3)

    # Coffee stain
    stain = BackgroundGenerator.add_coffee_stain(crumpled, center=(150, 150), radius=40)
    assert stain.shape == (h, w, 3)

    # Stamp
    stamped = BackgroundGenerator.add_stamp(stain, text="APPROVED", position=(50, 50))
    assert stamped.shape == (h, w, 3)


def test_render_line_physics(synthetic_generator: SyntheticHandwritingGenerator):
    """Test character jitter, slant shear, baseline wave, and ink models."""
    text = "The quick brown fox jumps"
    for ink in ["blue", "black", "fountain", "pencil"]:
        img, meta = synthetic_generator.render_line(
            text=text,
            font_size=30,
            ink_color=ink,
            slant_deg=14.0,
            tremor_sigma=1.5,
            wave_amplitude=3.0
        )
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8
        assert meta["text"] == text
        assert len(meta["bbox"]) == 4
        assert len(meta["norm_bbox"]) == 4


def test_render_prescription(synthetic_generator: SyntheticHandwritingGenerator):
    """Test complete medical prescription generation."""
    rx_img, rx_dict = synthetic_generator.render_prescription(
        clinic_name="Alpha Medical Center",
        doctor_name="Dr. Gregory House, MD",
        patient_name="James Wilson",
        medications=[
            ("Amoxicillin", "500mg capsules", "Take 1 cap tid x 10 days", "#30", "0")
        ],
        include_stamp=True,
        include_stains=True,
        apply_distortions=False,
        width=700,
        height=900
    )

    assert rx_img.shape == (900, 700, 3)
    assert rx_dict["clinic"] == "Alpha Medical Center"
    assert rx_dict["doctor"] == "Dr. Gregory House, MD"
    assert rx_dict["patient"] == "James Wilson"
    assert len(rx_dict["lines"]) >= 3
    assert len(rx_dict["full_text"]) > 0


def test_elastic_distortions(synthetic_generator: SyntheticHandwritingGenerator):
    """Test Albumentations elastic transformation."""
    img = np.full((200, 300, 3), 240, dtype=np.uint8)
    distorted = synthetic_generator.apply_elastic_distortions(img)
    assert distorted.shape == img.shape
    assert distorted.dtype == np.uint8

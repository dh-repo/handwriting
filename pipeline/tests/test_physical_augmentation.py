"""
pipeline/tests/test_physical_augmentation.py
Unit tests verifying the 3D Physical Augmentation Engine:
- 3D Lambertian & Blinn-Phong normal-mapped shading
- Non-uniform shadow gradients (directional ramps, vignetting, blob occlusion, spine curvature)
- Ink capillary paper fiber bleeding & vertex pooling
- Ornstein-Uhlenbeck continuous stochastic tremor jitter
- Multi-harmonic baseline sine wave deformation (cv2.remap)
- VocabularyManager template filling & slot syntax integrity
"""

import math
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline.dataset.synthetic_generator import (
    PhysicalAugmenter,
    VocabularyManager,
    SyntheticHandwritingGenerator,
    BackgroundGenerator
)


def test_heightmap_generation():
    """Test multi-octave continuous elevation field generation."""
    w, h = 400, 300
    hm = PhysicalAugmenter.generate_heightmap(w, h, roughness=1.2, include_ridges=True, num_folds=2)

    assert hm.shape == (h, w)
    assert hm.dtype == np.float32
    assert 0.0 <= hm.min() <= hm.max() <= 1.0
    # Elevation map must have non-zero gradient variation
    gx = cv2.Sobel(hm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    assert np.mean(grad_mag) > 0.001


def test_3d_lambertian_and_blinn_phong_shading():
    """Test 3D surface normal mapping and Blinn-Phong lighting modulation."""
    w, h = 500, 400
    base_img = np.full((h, w, 3), 245, dtype=np.uint8)

    # Test top-left lighting
    shaded_top_left = PhysicalAugmenter.apply_3d_lambertian_shading(
        base_img,
        intensity=0.30,
        light_theta=math.radians(45.0),
        light_phi=math.radians(60.0),
        specular_weight=0.10,
        shininess=32.0
    )
    assert shaded_top_left.shape == (h, w, 3)
    assert shaded_top_left.dtype == np.uint8
    assert np.std(shaded_top_left) > 1.0

    # Test bottom-right lighting
    shaded_bottom_right = PhysicalAugmenter.apply_3d_lambertian_shading(
        base_img,
        intensity=0.30,
        light_theta=math.radians(225.0),
        light_phi=math.radians(45.0)
    )
    assert shaded_bottom_right.shape == (h, w, 3)
    # Shading patterns under opposite lighting angles must differ
    diff = np.abs(shaded_top_left.astype(np.float32) - shaded_bottom_right.astype(np.float32))
    assert np.mean(diff) > 0.5


def test_shadow_gradients_composite():
    """Test directional linear ramps, radial vignetting, and ambient blob shadows."""
    w, h = 600, 800
    flat_img = np.full((h, w, 3), 250, dtype=np.uint8)

    rng = np.random.default_rng(42)
    shadowed = PhysicalAugmenter.apply_shadow_gradients(
        flat_img,
        linear_intensity=0.25,
        vignette_intensity=0.30,
        include_spine_shadow=True,
        include_blob=True,
        rng=rng
    )
    assert shadowed.shape == (h, w, 3)
    assert shadowed.dtype == np.uint8
    # Center should generally be brighter than extreme corners due to vignetting
    center_val = np.mean(shadowed[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20])
    corner_val = np.mean(shadowed[:40, :40])
    assert center_val >= corner_val - 5.0
    assert shadowed.min() >= 50  # Lower bound safety clamp


def test_ink_capillary_bleeding():
    """Test anisotropic fiber diffusion feathering on handwritten strokes."""
    w, h = 300, 100
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Draw horizontal stroke line
    rgba[48:52, 20:280, :3] = (30, 40, 150)
    rgba[48:52, 20:280, 3] = 255

    initial_alpha_sum = np.sum(rgba[:, :, 3] > 0)

    bled_rgba = PhysicalAugmenter.apply_ink_capillary_bleeding(
        rgba, bleed_intensity=0.60, fringe_blur_sigma=0.80
    )
    assert bled_rgba.shape == (h, w, 4)
    final_alpha_sum = np.sum(bled_rgba[:, :, 3] > 0)
    # Capillary bleeding must expand ink perimeter into paper fibers
    assert final_alpha_sum > initial_alpha_sum


def test_stroke_vertex_pooling():
    """Test ink pooling and darkening at stroke cusps and turnarounds."""
    w, h = 200, 200
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Draw a sharp "V" shape with acute vertex at (100, 150)
    cv2.line(rgba, (50, 50), (100, 150), (40, 60, 180, 255), 4)
    cv2.line(rgba, (100, 150), (150, 50), (40, 60, 180, 255), 4)

    pooled_rgba = PhysicalAugmenter.apply_stroke_vertex_pooling(
        rgba, corner_threshold=0.01, pooling_darken=0.40
    )
    assert pooled_rgba.shape == (h, w, 4)

    # Vertex region (around 100, 150) should be darker than mid-stroke (around 75, 100)
    vertex_rgb_mean = np.mean(pooled_rgba[145:155, 95:105, :3])
    stroke_rgb_mean = np.mean(pooled_rgba[95:105, 70:80, :3])
    assert vertex_rgb_mean <= stroke_rgb_mean + 5.0


def test_ornstein_uhlenbeck_stroke_tremor():
    """Test continuous mean-reverting stochastic trajectory jitter."""
    points = [(float(x), 50.0) for x in range(0, 200, 5)]

    displaced_steady = PhysicalAugmenter.simulate_stroke_tremor_ou(points, sigma=0.3, theta=0.8)
    displaced_tremor = PhysicalAugmenter.simulate_stroke_tremor_ou(points, sigma=2.5, theta=0.3)

    assert len(displaced_steady) == len(points)
    assert len(displaced_tremor) == len(points)

    dy_steady = [abs(p[1] - 50.0) for p in displaced_steady]
    dy_tremor = [abs(p[1] - 50.0) for p in displaced_tremor]

    assert np.mean(dy_tremor) > np.mean(dy_steady)


def test_multiharmonic_baseline_warp():
    """Test 3-harmonic sine wave baseline deformation via cv2.remap."""
    w, h = 600, 80
    line_img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Draw horizontal text baseline
    line_img[40:44, 20:580] = (20, 20, 20)

    warped, dy = PhysicalAugmenter.apply_multiharmonic_baseline_warp(
        line_img,
        amplitudes=(4.0, 1.5, 0.5),
        drift_deg=1.2
    )

    assert warped.shape == (h, w, 3)
    assert dy.shape == (w,)
    # Output image must reflect sinusoidal warping
    assert np.std(dy) > 0.5


def test_vocabulary_manager_integration():
    """Test VocabularyManager loading and dynamic template slot resolution."""
    vm = VocabularyManager()

    med = vm.get_random_medication()
    assert "generic_name" in med
    assert "therapeutic_class" in med

    doc = vm.get_random_doctor_profile()
    assert "full_name" in doc
    assert "npi" in doc

    sig = vm.get_random_sig_code()
    assert "code" in sig

    # Test template filling across all 5 categories
    categories = [
        "outpatient_encounter",
        "discharge_summary",
        "soap_note",
        "emergency_triage",
        "prescription_slip"
    ]
    for cat in categories:
        filled_text, meta = vm.fill_template(category=cat)
        assert len(filled_text) > 0
        assert meta["category"] == cat
        # Ensure zero lingering unresolved slot brackets {SLOT}
        import re
        unresolved = re.findall(r"\{[A-Z0-9_]+\}", filled_text)
        assert len(unresolved) == 0, f"Unresolved slots in filled template: {unresolved}"

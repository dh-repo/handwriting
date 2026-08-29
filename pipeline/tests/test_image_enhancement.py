"""
Unit and integration tests for pipeline/preprocessing/image_enhancement.py.
"""

from typing import Tuple
import cv2
import numpy as np
import pytest

from pipeline.preprocessing.image_enhancement import (
    ImageEnhancer,
    deskew_image,
    flatten_illumination,
    enhance_contrast,
    binarize_sauvola,
    binarize_otsu,
    adaptive_binarize,
    normalize_image,
    pad_to_size,
    to_rgb,
    to_grayscale,
    normalize_tensor,
)


def test_to_rgb_and_to_grayscale():
    """Test color conversions between grayscale, RGB, and RGBA."""
    gray = np.full((50, 50), 100, dtype=np.uint8)
    rgb = to_rgb(gray)
    assert rgb.shape == (50, 50, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb == 100)

    # RGBA with alpha
    rgba = np.full((50, 50, 4), 200, dtype=np.uint8)
    rgba[:, :, 3] = 128  # 50% opacity
    rgb_comp = to_rgb(rgba)
    assert rgb_comp.shape == (50, 50, 3)

    # Grayscale conversion
    back_to_gray = to_grayscale(rgb)
    assert back_to_gray.shape == (50, 50)
    assert back_to_gray.dtype == np.uint8


def test_deskew_zero_angle(synthetic_multiline_image: np.ndarray):
    """Test deskewing on a perfectly straight horizontal document."""
    deskewed, angle = deskew_image(synthetic_multiline_image)
    assert deskewed.shape == synthetic_multiline_image.shape
    assert abs(angle) < 1.0


def test_deskew_rotated_angles(synthetic_rotated_image: Tuple[np.ndarray, float]):
    """Test deskewing detection and correction on rotated documents."""
    rotated_img, true_angle = synthetic_rotated_image
    deskewed, detected_angle = deskew_image(rotated_img)

    # Detected angle magnitude should match ground truth rotation within 1.0 degree
    assert abs(detected_angle) == pytest.approx(abs(true_angle), abs=1.0)
    assert deskewed.shape == rotated_img.shape

    # Re-running deskew on the corrected image should yield near-zero residual skew
    _, residual_angle = deskew_image(deskewed)
    assert abs(residual_angle) < 1.0


def test_flatten_illumination():
    """Test morphological background division on uneven illumination."""
    # Create synthetic image with a heavy linear shadow gradient
    h, w = 200, 300
    base = np.full((h, w, 3), 220, dtype=np.uint8)
    # Add dark text strokes
    cv2.putText(base, "Illumination Test", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

    # Add dark linear gradient shadow across image (left=1.0, right=0.3)
    gradient = np.linspace(1.0, 0.3, w, dtype=np.float32).reshape(1, w, 1)
    shadowed = np.clip(base.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    flattened = flatten_illumination(shadowed)
    assert flattened.shape == shadowed.shape
    assert flattened.dtype == np.uint8

    # Background on the right side should be significantly brighter after flattening
    right_bg_shadowed = float(np.mean(shadowed[20:40, 220:280]))
    right_bg_flattened = float(np.mean(flattened[20:40, 220:280]))
    assert right_bg_flattened > right_bg_shadowed


def test_enhance_contrast():
    """Test CLAHE contrast enhancement."""
    # Create faint low-contrast text on gray background
    img = np.full((150, 250, 3), 200, dtype=np.uint8)
    # Very faint text
    cv2.putText(img, "Faint Ink Text", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)

    enhanced = enhance_contrast(img, clip_limit=3.0, flatten_background=False)
    assert enhanced.shape == img.shape

    # Standard deviation of intensities should increase after CLAHE
    std_orig = np.std(to_grayscale(img))
    std_enh = np.std(to_grayscale(enhanced))
    assert std_enh > std_orig


def test_binarize_sauvola(synthetic_multiline_image: np.ndarray):
    """Test fast Sauvola binarization output."""
    binary = binarize_sauvola(synthetic_multiline_image, window_size=31, k=0.2)
    assert binary.ndim == 2
    assert binary.shape == synthetic_multiline_image.shape[:2]
    assert binary.dtype == np.uint8

    # Should only contain 0 and 255
    unique_vals = set(np.unique(binary))
    assert unique_vals.issubset({0, 255})

    # Should detect ink foreground (non-zero pixels)
    ink_count = np.count_nonzero(binary == 255)
    total_pixels = binary.size
    # Handwriting typically occupies 2% to 25% of the page
    assert 0.01 * total_pixels < ink_count < 0.40 * total_pixels


def test_binarize_otsu_and_adaptive_dispatcher():
    """Test Otsu binarization and adaptive dispatcher."""
    img = np.full((100, 100, 3), 240, dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (30, 30, 30), -1)

    bin_otsu = binarize_otsu(img)
    assert bin_otsu.shape == (100, 100)
    assert bin_otsu[50, 50] == 255  # ink foreground
    assert bin_otsu[5, 5] == 0  # background

    bin_disp = adaptive_binarize(img, method="otsu")
    assert np.array_equal(bin_otsu, bin_disp)


def test_normalize_image_and_pad_to_size():
    """Test aspect-ratio preserving resizing and canvas padding."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)

    # Resize to target canvas (384, 384) with aspect ratio preserved
    norm = normalize_image(img, target_size=(384, 384), keep_aspect_ratio=True, fill_value=255)
    assert norm.shape == (384, 384, 3)
    assert norm.dtype == np.uint8
    # Outer corners should be padded with white (255)
    assert norm[0, 0, 0] == 255
    assert norm[-1, -1, 0] == 255

    # Direct pad_to_size
    small = np.full((50, 60), 100, dtype=np.uint8)
    padded = pad_to_size(small, 80, 90, fill_value=255)
    assert padded.shape == (80, 90)
    assert padded[10, 10] == 100
    assert padded[70, 80] == 255


def test_normalize_tensor():
    """Test conversion to (3, H, W) normalized float tensor."""
    img = np.full((64, 128, 3), 128, dtype=np.uint8)
    tensor = normalize_tensor(img)
    assert tensor.shape == (3, 64, 128)
    assert tensor.dtype == np.float32


def test_image_enhancer_class_wrapper(synthetic_multiline_image: np.ndarray):
    """Test ImageEnhancer wrapper interface."""
    enhancer = ImageEnhancer(deskew=True, flatten_bg=True, clahe_clip=2.5)

    deskewed, angle = enhancer.deskew(synthetic_multiline_image)
    assert deskewed.shape == synthetic_multiline_image.shape

    enhanced = enhancer.enhance(synthetic_multiline_image)
    assert enhanced.shape == synthetic_multiline_image.shape

    binary = enhancer.binarize(enhanced, method="sauvola")
    assert binary.shape == synthetic_multiline_image.shape[:2]

    normalized = enhancer.normalize(enhanced, target_size=(384, 384))
    assert normalized.shape == (384, 384, 3)

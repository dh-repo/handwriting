"""
pipeline/preprocessing/image_enhancement.py
Comprehensive image enhancement pipeline for difficult, messy handwriting.
Implements:
- Hybrid Hough + HPP Variance/Entropy document deskewing for [-45°, +45°]
- Morphological background division and CIE-LAB CLAHE contrast enhancement
- Fast O(1) OpenCV boxFilter Sauvola adaptive binarization with Otsu fallback
- Aspect-ratio preserving normalization, padding, and tensor formatting
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color Space Utilities
# ---------------------------------------------------------------------------

def to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Ensure input numpy array is 3-channel RGB uint8.

    Args:
        image: (H, W), (H, W, 1), (H, W, 3), or (H, W, 4) uint8 numpy array.

    Returns:
        (H, W, 3) uint8 numpy array in RGB channel order.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif channels == 3:
            return image
        elif channels == 4:
            # Alpha composite over white background
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            rgb = image[:, :, :3].astype(np.float32)
            white = np.ones_like(rgb) * 255.0
            return (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)

    raise ValueError(f"Invalid image array shape: {image.shape}")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB or multi-channel image to single-channel 2D uint8 grayscale (H, W).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")

    if image.ndim == 2:
        return image.astype(np.uint8)
    elif image.ndim == 3:
        if image.shape[2] == 1:
            return image[:, :, 0].astype(np.uint8)
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            rgb = to_rgb(image)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    raise ValueError(f"Invalid image array shape for grayscale conversion: {image.shape}")


# ---------------------------------------------------------------------------
# Deskew & Orientation Correction
# ---------------------------------------------------------------------------

def _detect_skew_hough(gray: np.ndarray, max_dim: int = 1000) -> Optional[float]:
    """
    Detect document skew angle via Probabilistic Hough Line Transform.
    Returns angle in degrees if consistent lines found, else None.
    """
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        target_w = max(1, int(round(w * scale)))
        target_h = max(1, int(round(h * scale)))
        small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    min_line_len = max(20, int(min(small.shape[:2]) * 0.15))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_line_len, maxLineGap=10)

    if lines is None or len(lines) < 4:
        return None

    angles = []
    weights = []
    for line in lines:
        coords = np.asarray(line).reshape(-1)
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.sqrt(dx * dx + dy * dy))
        if length < min_line_len:
            continue
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Normalize to [-45, +45]
        while angle_deg > 45.0:
            angle_deg -= 90.0
        while angle_deg < -45.0:
            angle_deg += 90.0

        if abs(angle_deg) <= 45.0:
            angles.append(angle_deg)
            weights.append(length)

    if len(angles) < 4:
        return None

    angles_arr = np.array(angles)
    weights_arr = np.array(weights)
    weighted_mean = np.sum(angles_arr * weights_arr) / np.sum(weights_arr)
    std_dev = np.sqrt(np.sum(weights_arr * (angles_arr - weighted_mean) ** 2) / np.sum(weights_arr))

    # If lines are highly consistent (std < 1.5 deg), return weighted mean
    if std_dev < 1.5:
        return float(weighted_mean)

    return None


def _detect_skew_hpp(
    gray: np.ndarray,
    min_angle: float = -45.0,
    max_angle: float = 45.0,
    coarse_step: float = 1.0,
    fine_step: float = 0.1,
    max_dim: int = 800
) -> float:
    """
    Detect document skew angle via Horizontal Projection Profile (HPP)
    variance maximization on edge-filtered image.
    """
    h, w = gray.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        target_w = max(1, int(round(w * scale)))
        target_h = max(1, int(round(h * scale)))
        small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    sh, sw = small.shape[:2]

    # Preprocess: Canny edges to isolate text line structure from illumination gradients
    edges = cv2.Canny(small, 50, 150)
    if np.count_nonzero(edges) < 20:
        return 0.0

    center = (sw / 2.0, sh / 2.0)

    # Margin crop to avoid rotation boundary artifacts
    crop_my = int(sh * 0.1)
    crop_mx = int(sw * 0.1)

    def evaluate_angle(angle: float) -> float:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(edges, M, (sw, sh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inner = rotated[crop_my:sh - crop_my, crop_mx:sw - crop_mx]
        proj = np.sum(inner, axis=1, dtype=np.float32)
        # Objective: Maximizing variance of projection profile
        return float(np.var(proj))

    # Coarse search
    coarse_angles = np.arange(min_angle, max_angle + coarse_step, coarse_step)
    coarse_scores = [evaluate_angle(a) for a in coarse_angles]
    if len(coarse_scores) == 0 or max(coarse_scores) <= 1e-6:
        return 0.0

    best_coarse_idx = int(np.argmax(coarse_scores))
    best_coarse_angle = float(coarse_angles[best_coarse_idx])

    # Fine search around best coarse angle
    fine_min = max(min_angle, best_coarse_angle - coarse_step)
    fine_max = min(max_angle, best_coarse_angle + coarse_step)
    fine_angles = np.arange(fine_min, fine_max + fine_step, fine_step)
    fine_scores = [evaluate_angle(a) for a in fine_angles]
    if len(fine_scores) == 0 or max(fine_scores) <= 1e-6:
        return 0.0

    best_fine_angle = float(np.clip(fine_angles[int(np.argmax(fine_scores))], min_angle, max_angle))

    return best_fine_angle


def deskew_image(
    image: np.ndarray,
    min_angle: float = -45.0,
    max_angle: float = 45.0,
    border_value: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[np.ndarray, float]:
    """
    Detect document skew and rotate the image to straighten horizontal text lines.

    Args:
        image: RGB or Grayscale numpy array.
        min_angle: Minimum skew search angle in degrees (default: -45.0).
        max_angle: Maximum skew search angle in degrees (default: +45.0).
        border_value: Color to fill outer corners created by rotation (default: white).

    Returns:
        Tuple of (deskewed_image, detected_skew_angle_in_degrees).
    """
    rgb = to_rgb(image)
    gray = to_grayscale(rgb)

    # 1. Try Hough Line Detection
    hough_angle = _detect_skew_hough(gray)

    # 2. If Hough is uncertain or returns None, use edge HPP variance
    if hough_angle is not None and abs(hough_angle) <= 45.0:
        skew_angle = hough_angle
    else:
        skew_angle = _detect_skew_hpp(gray, min_angle=min_angle, max_angle=max_angle)

    # If angle is negligible (< 0.1 deg), return original image
    if abs(skew_angle) < 0.1:
        return (rgb if image.ndim == 3 else gray, 0.0)

    # Rotate image by skew_angle to straighten it
    h, w = rgb.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    deskewed_rgb = cv2.warpAffine(
        rgb,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    if image.ndim == 2:
        return (to_grayscale(deskewed_rgb), skew_angle)
    return (deskewed_rgb, skew_angle)


# ---------------------------------------------------------------------------
# Illumination Flattening & Contrast Enhancement
# ---------------------------------------------------------------------------

def flatten_illumination(
    image: np.ndarray,
    kernel_size: int = 51,
    blur_size: int = 51
) -> np.ndarray:
    """
    Remove uneven shadow gradients and yellowed/dirty paper backgrounds
    using morphological dilation background estimation and division.

    Args:
        image: RGB or Grayscale numpy array.
        kernel_size: Size of dilation structuring element (odd integer, default: 51).
        blur_size: Size of Gaussian smoothing kernel (odd integer, default: 51).

    Returns:
        Illumination-flattened image of same shape and uint8 type.
    """
    is_rgb = (image.ndim == 3 and image.shape[2] == 3)
    rgb = to_rgb(image)

    # Convert to LAB to flatten illumination strictly on L (lightness) channel
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    bsize = blur_size if blur_size % 2 == 1 else blur_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    # Background estimation via morphological dilation
    bg_raw = cv2.morphologyEx(l_channel, cv2.MORPH_DILATE, kernel)
    bg_smooth = cv2.GaussianBlur(bg_raw, (bsize, bsize), 0)

    # Normalized division
    l_float = l_channel.astype(np.float32)
    bg_float = np.maximum(bg_smooth.astype(np.float32), 1.0)
    l_flat = np.clip((l_float / bg_float) * 255.0, 0, 255).astype(np.uint8)

    lab_flat = cv2.merge([l_flat, a_channel, b_channel])
    rgb_flat = cv2.cvtColor(lab_flat, cv2.COLOR_LAB2RGB)

    if not is_rgb and image.ndim == 2:
        return to_grayscale(rgb_flat)
    return rgb_flat


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
    flatten_background: bool = True
) -> np.ndarray:
    """
    Boost contrast of faint pencil, faded ballpoint, and messy handwriting strokes
    using CIE-LAB CLAHE with optional morphological illumination flattening.

    Args:
        image: RGB or Grayscale uint8 numpy array.
        clip_limit: CLAHE contrast limiting parameter (default: 2.5).
        tile_grid_size: CLAHE grid partition dimensions (default: (8, 8)).
        flatten_background: Whether to apply illumination background division first.

    Returns:
        Enhanced RGB or Grayscale uint8 numpy array.
    """
    is_rgb = (image.ndim == 3 and image.shape[2] == 3)
    rgb = to_rgb(image)

    if flatten_background:
        rgb = flatten_illumination(rgb)

    # Apply CLAHE strictly to L channel in LAB color space
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_clahe, a_channel, b_channel])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    if not is_rgb and image.ndim == 2:
        return to_grayscale(rgb_enhanced)
    return rgb_enhanced


# ---------------------------------------------------------------------------
# Fast Sauvola Adaptive Binarization
# ---------------------------------------------------------------------------

def binarize_sauvola(
    image: np.ndarray,
    window_size: int = 31,
    k: float = 0.2,
    r: float = 128.0,
    low_var_threshold: float = 5.0,
    bright_bg_threshold: float = 180.0
) -> np.ndarray:
    """
    O(1) Fast Sauvola adaptive binarization using OpenCV boxFilter.

    T(x, y) = m(x, y) * [1 + k * (s(x, y) / R - 1)]

    Includes low-variance noise guard to eliminate salt-and-pepper artifacts on white margins.

    Args:
        image: RGB or Grayscale uint8 numpy array.
        window_size: Local neighborhood window dimension (odd integer, default: 31).
        k: Sauvola sensitivity parameter in range [0.1, 0.5] (default: 0.2).
        r: Dynamic range of standard deviation (default: 128.0 for 8-bit images).
        low_var_threshold: Noise guard threshold for flat background areas.
        bright_bg_threshold: Mean brightness cutoff for white paper background guard.

    Returns:
        (H, W) uint8 binary mask where 255 represents ink foreground and 0 represents paper background.
    """
    gray = to_grayscale(image)
    f_gray = gray.astype(np.float32)

    w = window_size if window_size % 2 == 1 else window_size + 1
    ksize = (w, w)

    # Local mean E[X] via boxFilter in O(1)
    mean = cv2.boxFilter(f_gray, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)
    # Local second moment E[X^2]
    mean_sq = cv2.boxFilter(f_gray * f_gray, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)

    # Local standard deviation: s = sqrt(max(E[X^2] - E[X]^2, 0))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    std_dev = np.sqrt(variance)

    # Sauvola threshold: T = m * (1 + k * (s / R - 1))
    threshold = mean * (1.0 + k * ((std_dev / float(r)) - 1.0))

    # Low-variance background guard: suppress noise on clean, flat paper backgrounds
    bg_guard = (std_dev < low_var_threshold) & (mean > bright_bg_threshold)
    threshold[bg_guard] = 0.0

    # Binary mask: Ink is darker than threshold (f_gray <= threshold)
    # Return 255 for ink foreground, 0 for background
    binary_mask = np.zeros_like(gray, dtype=np.uint8)
    binary_mask[f_gray <= threshold] = 255

    return binary_mask


def binarize_otsu(image: np.ndarray) -> np.ndarray:
    """
    Global Otsu adaptive thresholding fallback.

    Returns:
        (H, W) uint8 binary mask where 255 is ink foreground and 0 is background.
    """
    gray = to_grayscale(image)
    # THRESH_BINARY_INV so ink (dark) becomes 255 foreground
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_mask


def adaptive_binarize(
    image: np.ndarray,
    method: str = "sauvola",
    window_size: int = 31,
    k: float = 0.2
) -> np.ndarray:
    """
    High-level binarization dispatcher supporting Sauvola with automatic Otsu fallback.

    Args:
        image: RGB or Grayscale numpy array.
        method: Binarization algorithm ("sauvola", "otsu").
        window_size: Window size for Sauvola.
        k: Parameter k for Sauvola.

    Returns:
        (H, W) uint8 binary mask (255=ink, 0=background).
    """
    gray = to_grayscale(image)
    global_std = float(np.std(gray))
    gray_mean = float(np.mean(gray))

    # Guard against flat white/near-white pages (e.g. blank background noise from CLAHE)
    if global_std < 2.0 and gray_mean > 200.0:
        return np.zeros_like(gray, dtype=np.uint8)

    # Fallback to Otsu if image is near-uniform or explicitly requested
    if method.lower() == "otsu" or global_std < 8.0:
        return binarize_otsu(gray)

    return binarize_sauvola(gray, window_size=window_size, k=k)


# ---------------------------------------------------------------------------
# Image Normalization & Tensor Preparation
# ---------------------------------------------------------------------------

def pad_to_size(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    fill_value: Union[int, Tuple[int, int, int]] = 255
) -> np.ndarray:
    """
    Pad image to exact target dimensions placing content at top-left.

    Args:
        image: (H, W) or (H, W, 3) numpy array.
        target_height: Desired canvas height.
        target_width: Desired canvas width.
        fill_value: Fill value for padded regions (default: 255 for white).

    Returns:
        Padded numpy array of shape (target_height, target_width, [C]).
    """
    h, w = image.shape[:2]
    if h > target_height or w > target_width:
        raise ValueError(f"Image ({h}, {w}) is larger than target canvas ({target_height}, {target_width})")

    if image.ndim == 2:
        canvas = np.full((target_height, target_width), fill_value, dtype=image.dtype)
        canvas[:h, :w] = image
        return canvas
    elif image.ndim == 3:
        fill = fill_value if isinstance(fill_value, tuple) else (fill_value, fill_value, fill_value)
        canvas = np.full((target_height, target_width, image.shape[2]), fill, dtype=image.dtype)
        canvas[:h, :w, :] = image
        return canvas

    raise ValueError(f"Invalid image dimensions: {image.shape}")


def normalize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    keep_aspect_ratio: bool = True,
    fill_value: int = 255
) -> np.ndarray:
    """
    Resize and pad image with aspect ratio preservation.

    Args:
        image: RGB or Grayscale numpy array.
        target_size: Optional (target_height, target_width).
        keep_aspect_ratio: If True, preserves aspect ratio and pads with fill_value.
        fill_value: Background padding color.

    Returns:
        Normalized uint8 numpy array.
    """
    rgb = to_rgb(image)
    if target_size is None:
        return rgb

    target_h, target_w = target_size
    orig_h, orig_w = rgb.shape[:2]

    if not keep_aspect_ratio:
        interp = cv2.INTER_AREA if (orig_h > target_h or orig_w > target_w) else cv2.INTER_CUBIC
        return cv2.resize(rgb, (target_w, target_h), interpolation=interp)

    scale = min(float(target_w) / orig_w, float(target_h) / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)

    canvas = np.full((target_h, target_w, 3), fill_value, dtype=np.uint8)
    # Center the resized image within the canvas
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = resized

    return canvas


def normalize_tensor(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Convert (H, W, 3) uint8 image to (3, H, W) float32 normalized tensor.
    """
    rgb = to_rgb(image).astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (rgb - mean_arr) / std_arr
    return np.transpose(normalized, (2, 0, 1))


# ---------------------------------------------------------------------------
# High-Level ImageEnhancer Class Wrapper
# ---------------------------------------------------------------------------

class ImageEnhancer:
    """
    Unified Image Enhancement Engine orchestrating deskewing, illumination
    flattening, contrast enhancement, and Sauvola adaptive binarization.
    """

    def __init__(
        self,
        deskew: bool = True,
        flatten_bg: bool = True,
        clahe_clip: float = 2.5,
        binarize_method: str = "sauvola"
    ):
        self.enable_deskew = deskew
        self.flatten_bg = flatten_bg
        self.clahe_clip = clahe_clip
        self.binarize_method = binarize_method

    def deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Straighten rotated document."""
        return deskew_image(image)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Flatten shadows and boost ink contrast via CLAHE."""
        return enhance_contrast(
            image,
            clip_limit=self.clahe_clip,
            flatten_background=self.flatten_bg
        )

    def binarize(self, image: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """Generate high-quality adaptive binary mask."""
        m = method if method is not None else self.binarize_method
        return adaptive_binarize(image, method=m)

    def normalize(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Normalize aspect-ratio and scale."""
        return normalize_image(image, target_size=target_size)

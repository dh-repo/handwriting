"""
pipeline/preprocessing/line_segmenter.py
Line and Word Segmentation Module for Handwritten Text Documents.
Implements:
- Horizontal Projection Profile (HPP) with adaptive Gaussian smoothing
- Dynamic peak and valley detection
- Vectorized Dynamic Programming Seam Carving across energy maps
- Word segmentation within line crops via horizontal dilation and CCA
- Strict Pydantic interface contracts for LineCrop and WordCrop
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
import scipy.ndimage as ndi
from scipy.signal import find_peaks
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Pydantic Interface Models
# ---------------------------------------------------------------------------

class WordCrop(BaseModel):
    """
    Word token bounding box and optional image patch.
    """
    word_index: int
    image: Optional[np.ndarray] = None
    bbox: List[float]  # [ymin, xmin, ymax, xmax] in [0, 1] relative to page

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LineCrop(BaseModel):
    """
    Handwritten line crop with normalized coordinates and optional word breakdown.
    """
    line_index: int
    image: np.ndarray  # line patch HxWxC uint8
    bbox: List[float]  # [ymin, xmin, ymax, xmax] in [0, 1] relative to page
    words: Optional[List[WordCrop]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Line and Word Segmenter Engine
# ---------------------------------------------------------------------------

class LineSegmenter:
    """
    Robust Line and Word Segmenter using Horizontal Projection Profile (HPP)
    and Dynamic Programming Seam Carving for separating entangled cursive text.
    """

    def __init__(
        self,
        min_line_height: int = 15,
        smoothing_sigma: Optional[float] = None,
        peak_prominence_factor: float = 0.08,
        seam_carving: bool = True,
        margin: int = 4,
        extract_words: bool = False
    ):
        self.min_line_height = min_line_height
        self.smoothing_sigma = smoothing_sigma
        self.peak_prominence_factor = peak_prominence_factor
        self.seam_carving = seam_carving
        self.margin = margin
        self.extract_words = extract_words

    def segment(
        self,
        image: np.ndarray,
        binarized_image: np.ndarray
    ) -> List[LineCrop]:
        """
        Segment a document image into individual LineCrop instances.

        Args:
            image: RGB image (HxWxC uint8).
            binarized_image: Binary image (HxW uint8), where ink is non-zero (e.g. 255) or 0.

        Returns:
            List of LineCrop objects with normalized coordinates [ymin, xmin, ymax, xmax].
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3D RGB image array, got ndim={image.ndim}")

        H, W = image.shape[:2]
        if H == 0 or W == 0:
            return []

        # Ensure standard binary mask: 1 for ink, 0 for background
        if binarized_image.dtype == bool:
            ink_mask = binarized_image.astype(np.uint8)
        else:
            ink_count_255 = int(np.count_nonzero(binarized_image == 255))
            ink_count_0 = int(np.count_nonzero(binarized_image == 0))
            if ink_count_255 < ink_count_0:
                ink_mask = (binarized_image == 255).astype(np.uint8)
            else:
                ink_mask = (binarized_image == 0).astype(np.uint8)

        # Handle blank image edge case
        total_ink = int(np.sum(ink_mask))
        if total_ink < 50:
            return []

        # 1. Filter out continuous thin ruling lines (e.g. notebook ruling lines) for HPP computation
        ruling_kw = max(35, int(W * 0.15))
        ruling_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ruling_kw, 1))
        opened_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, ruling_kernel)
        
        # Verify vertical thinness of ruling lines
        ruling_lines = cv2.morphologyEx(opened_lines, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
        # Subtract only pure thin horizontal ruling features
        clean_ruling = cv2.subtract(opened_lines, ruling_lines)
        text_ink = cv2.subtract(ink_mask, clean_ruling)

        # Use text_ink if sufficient strokes remain, otherwise use ink_mask
        hpp_mask = text_ink if int(np.sum(text_ink)) > 50 else ink_mask

        # Compute Horizontal Projection Profile (HPP)
        hpp = np.sum(hpp_mask, axis=1).astype(np.float32)

        # Adaptive Gaussian smoothing
        sigma = self.smoothing_sigma if self.smoothing_sigma is not None else max(2.5, float(H) / 90.0)
        smoothed_hpp = ndi.gaussian_filter1d(hpp, sigma=sigma)

        # 2. Find Peaks (Line Centroids)
        min_dist = max(self.min_line_height, int(H * 0.020))
        max_val = float(np.max(smoothed_hpp))
        prominence = max(0.5, max_val * min(0.05, self.peak_prominence_factor))
        peaks, _ = find_peaks(smoothed_hpp, distance=min_dist, prominence=prominence)

        # Fallback: If 0 or 1 peak found, treat whole foreground as single line
        if len(peaks) <= 1:
            return self._single_line_fallback(image, ink_mask, H, W)

        # 3. Find Valleys between consecutive peaks
        valleys: List[int] = []
        for i in range(len(peaks) - 1):
            p1, p2 = int(peaks[i]), int(peaks[i + 1])
            if p2 <= p1:
                valleys.append(p1)
            else:
                v = p1 + int(np.argmin(smoothed_hpp[p1:p2]))
                valleys.append(v)

        # 4. Compute Seams between lines
        seams: List[np.ndarray] = []
        for i, v in enumerate(valleys):
            p1, p2 = int(peaks[i]), int(peaks[i + 1])
            if not self.seam_carving or smoothed_hpp[v] == 0:
                # Straight cut fallback if clear white space
                seams.append(np.full(W, v, dtype=np.int32))
            else:
                seam = self._compute_seam(ink_mask, v, p1, p2, H, W)
                seams.append(seam)

        # 5. Extract Line Crops with Seam Boundary Masking
        num_lines = len(peaks)
        line_crops: List[LineCrop] = []

        for i in range(num_lines):
            top_seam = np.zeros(W, dtype=np.int32) if i == 0 else seams[i - 1]
            bot_seam = np.full(W, H - 1, dtype=np.int32) if i == num_lines - 1 else seams[i]

            y_min_line = int(np.min(top_seam))
            y_max_line = int(np.max(bot_seam))

            if y_max_line <= y_min_line:
                continue

            # Crop vertical band
            patch = np.copy(image[y_min_line:y_max_line + 1, :, :])
            patch_ink = np.copy(ink_mask[y_min_line:y_max_line + 1, :])

            # Mask out pixels outside seam boundaries with clean white
            for col in range(W):
                t_y = top_seam[col] - y_min_line
                b_y = bot_seam[col] - y_min_line
                if t_y > 0:
                    patch[:t_y, col, :] = 255
                    patch_ink[:t_y, col] = 0
                if b_y < patch.shape[0] - 1:
                    patch[b_y + 1:, col, :] = 255
                    patch_ink[b_y + 1:, col] = 0

            # Compute tight ink bounding box
            y_indices, x_indices = np.where(patch_ink > 0)
            if len(y_indices) == 0:
                continue

            tight_ymin = y_min_line + int(np.min(y_indices))
            tight_ymax = y_min_line + int(np.max(y_indices))
            tight_xmin = int(np.min(x_indices))
            tight_xmax = int(np.max(x_indices))

            # Add safety margin
            crop_ymin = max(0, tight_ymin - self.margin)
            crop_ymax = min(H - 1, tight_ymax + self.margin)
            crop_xmin = max(0, tight_xmin - self.margin)
            crop_xmax = min(W - 1, tight_xmax + self.margin)

            rel_ymin = max(0, crop_ymin - y_min_line)
            rel_ymax = min(patch.shape[0] - 1, crop_ymax - y_min_line)
            rel_xmin = crop_xmin
            rel_xmax = crop_xmax

            cropped_patch = patch[rel_ymin:rel_ymax + 1, rel_xmin:rel_xmax + 1]

            norm_bbox = [
                round(float(crop_ymin) / float(H), 5),
                round(float(crop_xmin) / float(W), 5),
                round(float(crop_ymax) / float(H), 5),
                round(float(crop_xmax) / float(W), 5)
            ]

            # Optional Word Segmentation
            words = None
            if self.extract_words:
                cropped_ink = patch_ink[rel_ymin:rel_ymax + 1, rel_xmin:rel_xmax + 1]
                words = self.segment_words(
                    cropped_patch,
                    cropped_ink,
                    page_offset=(crop_ymin, crop_xmin),
                    page_shape=(H, W)
                )

            line_crops.append(
                LineCrop(
                    line_index=len(line_crops),
                    image=cropped_patch,
                    bbox=norm_bbox,
                    words=words
                )
            )

        return line_crops

    def _compute_seam(
        self,
        ink_mask: np.ndarray,
        valley_y: int,
        peak1_y: int,
        peak2_y: int,
        H: int,
        W: int
    ) -> np.ndarray:
        """
        Compute minimal energy seam using Vectorized Dynamic Programming.
        """
        y_min = max(0, int(peak1_y + 0.35 * (valley_y - peak1_y)))
        y_max = min(H - 1, int(valley_y + 0.65 * (peak2_y - valley_y)))
        band_h = y_max - y_min + 1

        if band_h < 3:
            return np.full(W, valley_y, dtype=np.int32)

        band_ink = ink_mask[y_min:y_max + 1, :].astype(np.float32)

        # Energy formulation: Ink penalty + guide penalty + distance penalty
        energy = band_ink * 100.0
        y_coords = np.arange(y_min, y_max + 1, dtype=np.float32)[:, None]
        guide_cost = ((y_coords - float(valley_y)) / float(band_h)) ** 2 * 10.0
        energy += guide_cost

        # Distance transform penalty to favor open white background
        dist = ndi.distance_transform_edt(1.0 - band_ink)
        energy += np.exp(-dist / 2.0) * 5.0

        # Dynamic programming matrix
        cost = np.copy(energy)
        backtrack = np.zeros((band_h, W), dtype=np.int32)

        for x in range(1, W):
            prev = cost[:, x - 1]
            p_up = np.pad(prev[:-1], (1, 0), constant_values=1e9)
            p_mid = prev
            p_down = np.pad(prev[1:], (0, 1), constant_values=1e9)

            stacked = np.stack([p_up, p_mid, p_down], axis=0)
            choice = np.argmin(stacked, axis=0)  # 0: up, 1: mid, 2: down
            cost[:, x] += np.min(stacked, axis=0)
            backtrack[:, x] = np.arange(band_h) + (choice - 1)

        seam_y = np.zeros(W, dtype=np.int32)
        seam_y[-1] = int(np.argmin(cost[:, -1]))
        for x in range(W - 2, -1, -1):
            next_idx = seam_y[x + 1]
            seam_y[x] = backtrack[next_idx, x + 1]

        return np.clip(seam_y + y_min, 0, H - 1)

    def segment_words(
        self,
        line_image: np.ndarray,
        line_ink: np.ndarray,
        page_offset: Tuple[int, int],
        page_shape: Tuple[int, int]
    ) -> List[WordCrop]:
        """
        Extract word bounding boxes within a line patch using horizontal dilation and CCA.
        """
        lh, lw = line_ink.shape[:2]
        H_page, W_page = page_shape
        y_off, x_off = page_offset

        if lh == 0 or lw == 0 or np.sum(line_ink) == 0:
            return []

        k_w = max(3, int(lh * 0.25))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        dilated = cv2.dilate((line_ink > 0).astype(np.uint8) * 255, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        word_boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 15 or w < 3:
                continue

            comp_mask = (labels == i)
            comp_ink = (line_ink > 0) & comp_mask
            y_idx, x_idx = np.where(comp_ink)
            if len(y_idx) == 0:
                continue

            w_ymin, w_ymax = int(np.min(y_idx)), int(np.max(y_idx))
            w_xmin, w_xmax = int(np.min(x_idx)), int(np.max(x_idx))

            page_ymin = y_off + w_ymin
            page_ymax = y_off + w_ymax
            page_xmin = x_off + w_xmin
            page_xmax = x_off + w_xmax

            word_boxes.append({
                'xmin': w_xmin,
                'xmax': w_xmax,
                'ymin': w_ymin,
                'ymax': w_ymax,
                'page_bbox': [
                    round(float(page_ymin) / float(H_page), 5),
                    round(float(page_xmin) / float(W_page), 5),
                    round(float(page_ymax) / float(H_page), 5),
                    round(float(page_xmax) / float(W_page), 5)
                ]
            })

        word_boxes.sort(key=lambda b: b['xmin'])

        word_crops: List[WordCrop] = []
        for idx, wb in enumerate(word_boxes):
            word_patch = line_image[wb['ymin']:wb['ymax'] + 1, wb['xmin']:wb['xmax'] + 1]
            word_crops.append(
                WordCrop(
                    word_index=idx,
                    image=word_patch,
                    bbox=wb['page_bbox']
                )
            )
        return word_crops

    def _single_line_fallback(
        self,
        image: np.ndarray,
        ink_mask: np.ndarray,
        H: int,
        W: int
    ) -> List[LineCrop]:
        """Fallback when document contains only a single text line."""
        y_indices, x_indices = np.where(ink_mask > 0)
        if len(y_indices) == 0:
            return []

        # Ink density guard against scattered noise
        box_area = (int(np.max(y_indices)) - int(np.min(y_indices)) + 1) * (int(np.max(x_indices)) - int(np.min(x_indices)) + 1)
        ink_density = len(y_indices) / float(max(1, box_area))
        if len(y_indices) < 150 and ink_density < 0.01:
            return []
        if ink_density < 0.002:
            return []

        ymin = max(0, int(np.min(y_indices)) - self.margin)
        ymax = min(H - 1, int(np.max(y_indices)) + self.margin)
        xmin = max(0, int(np.min(x_indices)) - self.margin)
        xmax = min(W - 1, int(np.max(x_indices)) + self.margin)

        crop = image[ymin:ymax + 1, xmin:xmax + 1]
        bbox = [
            round(float(ymin) / float(H), 5),
            round(float(xmin) / float(W), 5),
            round(float(ymax) / float(H), 5),
            round(float(xmax) / float(W), 5)
        ]

        words = None
        if self.extract_words:
            line_ink = ink_mask[ymin:ymax + 1, xmin:xmax + 1]
            words = self.segment_words(crop, line_ink, page_offset=(ymin, xmin), page_shape=(H, W))

        return [LineCrop(line_index=0, image=crop, bbox=bbox, words=words)]

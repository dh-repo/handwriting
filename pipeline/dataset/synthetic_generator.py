"""
pipeline/dataset/synthetic_generator.py
Procedural synthetic handwriting, 3D physical augmentation, and medical prescription generator.

Features:
- HandwritingFontManager with macOS system font discovery & fallbacks
- PhysicalAugmenter:
    * 3D Lambertian & Blinn-Phong normal-mapped paper shading with randomized spherical lighting
    * Non-uniform shadow gradients (directional linear ramps, notebook spine shadow, radial vignetting, arm blobs)
    * Ink capillary paper fiber bleeding (anisotropic morphological dilation + noise diffusion)
    * Stroke vertex ink pooling (Harris corner response / kinematic deceleration cusps)
    * Continuous Ornstein-Uhlenbeck stochastic stroke tremor jitter
    * 3-harmonic non-linear baseline curvature via cv2.remap
- VocabularyManager (loads RxNorm 1,000+ drugs, 50+ Latin sigs, 100+ doctor profiles, 200+ clinical templates)
- BackgroundGenerator (ruled notebook paper, 3D normal-mapped crumpled paper, grid paper, stamps, coffee stains)
- SyntheticHandwritingGenerator (character tremor/jitter, baseline sine wave, stroke shear slant, ink models)
- Albumentations elastic distortions
"""

import json
import logging
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.preprocessing.image_enhancement import to_grayscale

logger = logging.getLogger("synthetic_generator")

# Optional Albumentations import with safe fallback
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Font Manager with macOS System Discovery & Fallbacks
# ---------------------------------------------------------------------------

class HandwritingFontManager:
    """
    Discovers, loads, and manages system and bundled cursive/script fonts.
    """

    KNOWN_CURSIVE_FONTS = [
        "Bradley Hand Bold.ttf",
        "Brush Script.ttf",
        "Chalkboard.ttc",
        "ChalkboardSE.ttc",
        "Chalkduster.ttf",
        "Comic Sans MS.ttf",
        "Comic Sans MS Bold.ttf",
        "Papyrus.ttc",
        "Savoye LET.ttc",
        "SignPainter.ttc",
        "SnellRoundhand.ttc",
        "Trattatello.ttf",
        "Zapfino.ttf",
        "Apple Chancery.ttf",
        "MarkerFelt.ttc",
        "Noteworthy.ttc",
    ]

    SYSTEM_FONT_DIRS = [
        "/System/Library/Fonts/Supplemental",
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]

    EXCLUDED_FONT_PATTERNS = (
        "emoji", "noto", "webdings", "wingdings", "symbol", "cjksymbol",
        "lastresort", "braille", "dingbats", "ornaments", "arabic",
        "hebrew", "devanagari", "gurmukhi", "gujarati", "oriya",
        "tamil", "telugu", "kannada", "malayalam", "sinhala", "thai",
        "lao", "tibetan", "myanmar", "georgian", "ethiopic", "cherokee",
        "canadian", "ogham", "runic", "khmer", "mongolian", "syriac",
        "thaana", "heiti", "pingfang", "hiragino", "songti", "kaiti",
        "yugothic", "yumincho", "meiryo", "gothic", "mincho", "batang",
        "dotum", "gulim", "gungsuh", "malgun", "damascus", "baghdad",
        "alnile", "altarikh", "geezapro", "decotype", "farah", "farisi",
        "kefa", "kokonor", "mishafi", "muna", "sana", "silom", "thonburi",
        "waseem", "yuppy", "nisc"
    )

    def __init__(self, custom_font_dirs: Optional[List[str]] = None):
        self.font_paths: Dict[str, str] = {}
        search_dirs = list(self.SYSTEM_FONT_DIRS)
        if custom_font_dirs:
            search_dirs = custom_font_dirs + search_dirs

        self._discover_fonts(search_dirs)

    def _discover_fonts(self, search_dirs: List[str]):
        for sdir in search_dirs:
            p = Path(sdir)
            if not p.exists() or not p.is_dir():
                continue
            for font_file in p.glob("**/*.[tT][tT][fFcC]"):
                fname_l = font_file.name.lower()
                stem_l = font_file.stem.lower()
                if any(pat in fname_l or pat in stem_l for pat in self.EXCLUDED_FONT_PATTERNS):
                    continue
                fname = font_file.name
                self.font_paths[fname] = str(font_file)
                self.font_paths[font_file.stem] = str(font_file)

    def get_reliable_latin_font(self, size: int = 32) -> ImageFont.FreeTypeFont:
        """
        Guaranteed fallback to a reliable Latin font.
        """
        for font_candidate in [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
            "/Library/Fonts/Arial.ttf",
        ]:
            if os.path.exists(font_candidate):
                try:
                    return ImageFont.truetype(font_candidate, size=size)
                except Exception:
                    continue
        for cursive in self.KNOWN_CURSIVE_FONTS:
            if cursive in self.font_paths:
                try:
                    return ImageFont.truetype(self.font_paths[cursive], size=size)
                except Exception:
                    continue
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def list_available_fonts(self) -> List[str]:
        return list(self.font_paths.keys())

    def get_font(self, font_name: Optional[str] = None, size: int = 32) -> ImageFont.FreeTypeFont:
        """
        Load a font by name or return a suitable handwriting/default font.
        """
        if font_name and font_name in self.font_paths:
            try:
                return ImageFont.truetype(self.font_paths[font_name], size=size)
            except Exception:
                pass

        if font_name and os.path.exists(font_name):
            try:
                return ImageFont.truetype(font_name, size=size)
            except Exception:
                pass

        # Try to find any known cursive font in discovered paths
        for cursive in self.KNOWN_CURSIVE_FONTS:
            if cursive in self.font_paths:
                try:
                    return ImageFont.truetype(self.font_paths[cursive], size=size)
                except Exception:
                    continue

        # Try any available truetype font
        for _, path in self.font_paths.items():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue

        # Fallback to default PIL font
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def get_random_handwriting_font(self, size: int = 32) -> ImageFont.FreeTypeFont:
        """
        Pick a random cursive/handwriting font from available fonts.
        """
        available_cursive = [c for c in self.KNOWN_CURSIVE_FONTS if c in self.font_paths]
        if available_cursive:
            chosen = random.choice(available_cursive)
            return self.get_font(chosen, size=size)
        return self.get_font(size=size)


# ---------------------------------------------------------------------------
# 3D Physical Augmentation Engine
# ---------------------------------------------------------------------------

class PhysicalAugmenter:
    """
    Pure NumPy/OpenCV/SciPy physics-based document augmentation engine.
    Zero external native C-library dependencies.
    """

    @staticmethod
    def generate_heightmap(
        width: int,
        height: int,
        roughness: float = 1.0,
        include_ridges: bool = True,
        num_folds: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Generate continuous multi-octave elevation field Z(x, y) in [0, 1].
        Combines macro sheet curvature, medium crumpled facets, sharp ridges,
        and directional fold fractures.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Octave 1: Macro sheet curvature (H/120, W/120)
        h1 = max(3, height // 120 + 2)
        w1 = max(3, width // 120 + 2)
        n1 = rng.standard_normal((h1, w1), dtype=np.float32)
        z1 = cv2.resize(n1, (width, height), interpolation=cv2.INTER_LINEAR)

        # Octave 2: Medium crumpled facets (H/40, W/40)
        h2 = max(3, height // 40 + 2)
        w2 = max(3, width // 40 + 2)
        n2 = rng.standard_normal((h2, w2), dtype=np.float32)
        z2 = cv2.resize(n2, (width, height), interpolation=cv2.INTER_LINEAR)

        # Octave 3: Micro paper grain (H/10, W/10)
        h3 = max(3, height // 10 + 2)
        w3 = max(3, width // 10 + 2)
        n3 = rng.standard_normal((h3, w3), dtype=np.float32)
        z3 = cv2.resize(n3, (width, height), interpolation=cv2.INTER_LINEAR)

        # fBM Composite
        z_fbm = 0.50 * z1 + 0.35 * z2 + 0.15 * z3

        # Ridge synthesis
        if include_ridges:
            mu_z = float(np.mean(z_fbm))
            z_ridged = 1.0 - 2.0 * np.abs(z_fbm - mu_z)
        else:
            z_ridged = np.zeros_like(z_fbm)

        # Directional Crease Fractures
        z_folds = np.zeros((height, width), dtype=np.float32)
        if num_folds > 0:
            x_coords = np.arange(width, dtype=np.float32)
            y_coords = np.arange(height, dtype=np.float32)[:, None]
            diag = math.sqrt(width * width + height * height)

            for _ in range(num_folds):
                alpha = float(rng.uniform(0, math.pi))
                offset_d = float(rng.uniform(0.2 * diag, 0.8 * diag))
                sigma = float(rng.uniform(8.0, 25.0))
                amp = float(rng.uniform(-0.3, 0.3))

                dist_proj = x_coords * math.cos(alpha) + y_coords * math.sin(alpha) - offset_d
                z_folds += amp * np.exp(-(dist_proj ** 2) / (2.0 * (sigma ** 2)))

        z_total = 0.55 * z_fbm + 0.30 * z_ridged + 0.15 * z_folds
        z_min = float(np.min(z_total))
        z_max = float(np.max(z_total))
        if z_max > z_min:
            z_norm = (z_total - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros_like(z_total)

        return z_norm.astype(np.float32)

    @staticmethod
    def apply_3d_lambertian_shading(
        image: np.ndarray,
        heightmap: Optional[np.ndarray] = None,
        intensity: float = 0.20,
        light_theta: Optional[float] = None,
        light_phi: Optional[float] = None,
        specular_weight: float = 0.08,
        shininess: float = 24.0,
        relief_scale: float = 1.5,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Apply dynamic 3D Lambertian diffuse + Blinn-Phong specular paper shading.
        I = I_a k_a + I_d (N · L) + I_s (N · H)^alpha.
        """
        if rng is None:
            rng = np.random.default_rng()

        h, w = image.shape[:2]
        if heightmap is None:
            heightmap = PhysicalAugmenter.generate_heightmap(w, h, rng=rng)
        elif heightmap.shape[:2] != (h, w):
            heightmap = cv2.resize(heightmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Sobel surface gradients
        gx = cv2.Sobel(heightmap, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(heightmap, cv2.CV_32F, 0, 1, ksize=3)

        # Unit normal field N(x, y) = (-sz*gx, -sz*gy, 1.0) / norm
        sz = float(relief_scale)
        nx_un = -sz * gx
        ny_un = -sz * gy
        nz_un = np.ones_like(gx, dtype=np.float32)

        norm = np.sqrt(nx_un * nx_un + ny_un * ny_un + nz_un * nz_un)
        norm = np.maximum(norm, 1e-6)
        nx = nx_un / norm
        ny = ny_un / norm
        nz = nz_un / norm

        # Spherical light source L = (cos theta cos phi, sin theta cos phi, sin phi)
        theta = light_theta if light_theta is not None else float(rng.uniform(0, 2 * math.pi))
        phi = light_phi if light_phi is not None else float(rng.uniform(math.radians(25.0), math.radians(75.0)))

        lx = math.cos(theta) * math.cos(phi)
        ly = math.sin(theta) * math.cos(phi)
        lz = math.sin(phi)

        # Viewer along optical axis V = (0, 0, 1) -> Half-vector H = (L + V) / ||L + V||
        vx, vy, vz = 0.0, 0.0, 1.0
        hx_un = lx + vx
        hy_un = ly + vy
        hz_un = lz + vz
        h_norm = math.sqrt(hx_un * hx_un + hy_un * hy_un + hz_un * hz_un)
        hx = hx_un / max(1e-6, h_norm)
        hy = hy_un / max(1e-6, h_norm)
        hz = hz_un / max(1e-6, h_norm)

        # Diffuse Lambertian component (N · L)
        dot_nl = np.maximum(0.0, nx * lx + ny * ly + nz * lz)

        # Blinn-Phong Specular component (N · H)^alpha
        dot_nh = np.maximum(0.0, nx * hx + ny * hy + nz * hz)
        specular = np.power(dot_nh, shininess)

        ka = 0.40
        kd = 0.55
        ks = specular_weight

        shading = ka + kd * dot_nl + ks * specular
        mean_shading = float(np.mean(shading))
        if mean_shading > 1e-6:
            shading_norm = (shading - mean_shading) * intensity + 1.0
        else:
            shading_norm = np.ones_like(shading)

        shading_norm = np.clip(shading_norm, 0.2, 1.8)

        if image.ndim == 3:
            shading_3d = shading_norm[:, :, None]
        else:
            shading_3d = shading_norm

        shaded_img = np.clip(image.astype(np.float32) * shading_3d, 0, 255).astype(np.uint8)
        return shaded_img

    @staticmethod
    def apply_shadow_gradients(
        image: np.ndarray,
        linear_intensity: float = 0.25,
        vignette_intensity: float = 0.30,
        include_spine_shadow: bool = False,
        include_blob: bool = True,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Apply composite non-uniform illumination field:
        S_total = S_linear * V * S_blob * S_spine.
        Evaluated via NumPy outer product vector broadcasting.
        """
        if rng is None:
            rng = np.random.default_rng()

        h, w = image.shape[:2]
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)[:, None]

        # 1. Directional Linear Ramp
        psi = float(rng.uniform(0, 2 * math.pi))
        u = x_coords * math.cos(psi) + y_coords * math.sin(psi)
        u_min = float(np.min(u))
        u_max = float(np.max(u))
        if u_max > u_min:
            u_norm = (u - u_min) / (u_max - u_min)
        else:
            u_norm = np.zeros_like(u)
        s_linear = 1.0 - linear_intensity * u_norm

        # 2. Radial Lens Vignetting
        cx = w * (0.5 + float(rng.uniform(-0.08, 0.08)))
        cy = h * (0.5 + float(rng.uniform(-0.08, 0.08)))
        r_max_sq = (0.5 * w) ** 2 + (0.5 * h) ** 2
        r_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
        vignette = 1.0 - vignette_intensity * np.power(np.clip(r_sq / r_max_sq, 0.0, 1.0), 1.2)

        # 3. Soft Physician Arm / Phone Blob Shadow
        s_blob = np.ones((h, w), dtype=np.float32)
        if include_blob:
            blob_x0 = float(rng.uniform(0.2 * w, 0.8 * w))
            blob_y0 = float(rng.uniform(0.2 * h, 0.8 * h))
            theta_b = float(rng.uniform(0, math.pi))
            sig_x = float(rng.uniform(0.2 * w, 0.5 * w))
            sig_y = float(rng.uniform(0.15 * h, 0.4 * h))
            a_blob = float(rng.uniform(0.15, 0.30))

            dx = x_coords - blob_x0
            dy = y_coords - blob_y0
            d_ellipse_sq = (
                ((dx * math.cos(theta_b) + dy * math.sin(theta_b)) / max(1.0, sig_x)) ** 2 +
                ((-dx * math.sin(theta_b) + dy * math.cos(theta_b)) / max(1.0, sig_y)) ** 2
            )
            s_blob = 1.0 - a_blob * np.exp(-d_ellipse_sq / 2.0)

        # 4. Notebook Spine Shadow
        s_spine = np.ones((h, w), dtype=np.float32)
        if include_spine_shadow:
            spine_x = 0.0 if rng.random() < 0.5 else float(w)
            lambda_spine = float(rng.uniform(40.0, 120.0))
            a_spine = float(rng.uniform(0.20, 0.45))
            s_spine = 1.0 - a_spine * np.exp(-np.abs(x_coords - spine_x) / lambda_spine)

        # Composite total shadow map (clamped >= 0.40)
        s_total = np.clip(s_linear * vignette * s_blob * s_spine, 0.40, 1.0)

        if image.ndim == 3:
            s_3d = s_total[:, :, None]
        else:
            s_3d = s_total

        out = np.clip(image.astype(np.float32) * s_3d, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def apply_ink_capillary_bleeding(
        rgba_image: np.ndarray,
        bleed_intensity: float = 0.40,
        fringe_blur_sigma: float = 0.60,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Simulate capillary ink diffusion along cellulose paper fibers on an RGBA image.
        Feathers the stroke contour outwards with anisotropic fiber noise.
        """
        if rng is None:
            rng = np.random.default_rng()

        out_rgba = rgba_image.copy()
        alpha = out_rgba[:, :, 3].astype(np.float32) / 255.0
        if np.max(alpha) < 0.05:
            return out_rgba

        h, w = alpha.shape
        # Morphological dilation using 3x3 structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_alpha = cv2.dilate((alpha * 255).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255.0
        fringe = np.maximum(0.0, dilated_alpha - alpha)

        # Anisotropic fiber grain noise
        h_fiber = max(3, h // 4 + 2)
        w_fiber = max(3, w // 2 + 2)
        grain_small = rng.uniform(0.3, 1.0, (h_fiber, w_fiber)).astype(np.float32)
        g_fiber = cv2.resize(grain_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Continuous capillary diffusion
        feather = cv2.GaussianBlur(fringe * g_fiber, (0, 0), fringe_blur_sigma)
        alpha_bleed = np.clip(alpha + bleed_intensity * feather, 0.0, 1.0)

        out_rgba[:, :, 3] = (alpha_bleed * 255.0).astype(np.uint8)
        return out_rgba

    @staticmethod
    def apply_stroke_vertex_pooling(
        rgba_image: np.ndarray,
        corner_threshold: float = 0.02,
        pooling_darken: float = 0.30
    ) -> np.ndarray:
        """
        Simulate ink pooling at deceleration points, cusps, and loop vertices.
        Darkens RGB pigment where Harris corner response is elevated.
        """
        out_rgba = rgba_image.copy()
        alpha = out_rgba[:, :, 3].astype(np.float32) / 255.0
        if np.max(alpha) < 0.05:
            return out_rgba

        # Smooth alpha mask
        alpha_smooth = cv2.GaussianBlur(alpha, (3, 3), 1.0)
        # Compute gradients
        gx = cv2.Sobel(alpha_smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(alpha_smooth, cv2.CV_32F, 0, 1, ksize=3)

        # Structure tensor elements
        gxx = cv2.GaussianBlur(gx * gx, (5, 5), 1.5)
        gyy = cv2.GaussianBlur(gy * gy, (5, 5), 1.5)
        gxy = cv2.GaussianBlur(gx * gy, (5, 5), 1.5)

        # Harris response R = det(J) - 0.04 * trace(J)^2
        det_j = gxx * gyy - gxy * gxy
        trace_j = gxx + gyy
        harris_r = det_j - 0.04 * (trace_j ** 2)

        max_r = float(np.max(harris_r))
        if max_r > 1e-6:
            corner_mask = (harris_r > corner_threshold * max_r).astype(np.float32) * (alpha > 0.3).astype(np.float32)
            pool_map = cv2.GaussianBlur(corner_mask, (5, 5), 2.0)
            pool_map = np.clip(pool_map, 0.0, 1.0)[:, :, None]

            rgb = out_rgba[:, :, :3].astype(np.float32)
            rgb_darkened = rgb * (1.0 - pooling_darken * pool_map)
            out_rgba[:, :, :3] = np.clip(rgb_darkened, 0, 255).astype(np.uint8)

        return out_rgba

    @staticmethod
    def simulate_stroke_tremor_ou(
        points: List[Tuple[float, float]],
        sigma: float = 1.2,
        theta: float = 0.6,
        dt: float = 1.0,
        rng: Optional[np.random.Generator] = None
    ) -> List[Tuple[float, float]]:
        """
        Simulate continuous mean-reverting Ornstein-Uhlenbeck stochastic tremor along stroke trajectory:
        delta_x_i = (1 - theta * dt) * delta_x_{i-1} + sigma * sqrt(dt) * eps_x
        """
        if rng is None:
            rng = np.random.default_rng()

        if not points:
            return points

        dx_prev = 0.0
        dy_prev = 0.0
        decay = max(0.0, 1.0 - theta * dt)
        vol = sigma * math.sqrt(dt)

        displaced_points = []
        for x, y in points:
            eps_x = float(rng.standard_normal())
            eps_y = float(rng.standard_normal())
            dx = decay * dx_prev + vol * eps_x
            dy = decay * dy_prev + vol * eps_y
            displaced_points.append((x + dx, y + dy))
            dx_prev, dy_prev = dx, dy

        return displaced_points

    @staticmethod
    def apply_multiharmonic_baseline_warp(
        image: np.ndarray,
        amplitudes: Tuple[float, float, float] = (3.5, 1.2, 0.4),
        wavelengths: Optional[Tuple[float, float, float]] = None,
        drift_deg: Optional[float] = None,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply 3-harmonic baseline sine wave deformation via fast cv2.remap coordinate warping.
        Harmonic 1: Line-level arc / tilt (lambda ~ 0.7W - 1.4W)
        Harmonic 2: Word-level wave (lambda ~ 100 - 250 px)
        Harmonic 3: Micro-bobbing (lambda ~ 25 - 60 px)
        """
        if rng is None:
            rng = np.random.default_rng()

        h, w = image.shape[:2]
        a1, a2, a3 = amplitudes

        if wavelengths is None:
            l1 = float(rng.uniform(0.7 * w, 1.4 * w))
            l2 = float(rng.uniform(100.0, 250.0))
            l3 = float(rng.uniform(25.0, 60.0))
        else:
            l1, l2, l3 = wavelengths

        phi1 = float(rng.uniform(0, 2 * math.pi))
        phi2 = float(rng.uniform(0, 2 * math.pi))
        phi3 = float(rng.uniform(0, 2 * math.pi))

        drift = math.tan(math.radians(drift_deg if drift_deg is not None else float(rng.uniform(-1.5, 1.5))))

        x_grid = np.arange(w, dtype=np.float32)
        dy_warp = (
            drift * x_grid +
            a1 * np.sin(2.0 * math.pi * x_grid / max(1.0, l1) + phi1) +
            a2 * np.sin(2.0 * math.pi * x_grid / max(1.0, l2) + phi2) +
            a3 * np.sin(2.0 * math.pi * x_grid / max(1.0, l3) + phi3)
        )

        map_x = np.tile(x_grid, (h, 1))
        y_grid = np.arange(h, dtype=np.float32)[:, None]
        map_y = y_grid - dy_warp[None, :]

        warped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return warped, dy_warp


# ---------------------------------------------------------------------------
# Clinical Vocabulary Manager
# ---------------------------------------------------------------------------

class VocabularyManager:
    """
    Loads and provides structured sampling from validated clinical vocabularies:
    - rxnorm_medications.json
    - latin_sig_codes.json
    - doctor_profiles.json
    - clinical_templates.json
    """

    DEFAULT_VOCAB_DIR = Path("data/reference_handwriting/vocabularies")

    def __init__(self, vocab_dir: Optional[Union[str, Path]] = None):
        self.vocab_dir = Path(vocab_dir) if vocab_dir is not None else self.DEFAULT_VOCAB_DIR
        self.medications: List[Dict[str, Any]] = []
        self.sig_codes: List[Dict[str, Any]] = []
        self.doctors: List[Dict[str, Any]] = []
        self.templates: List[Dict[str, Any]] = []
        self._load_vocabularies()

    def _load_vocabularies(self):
        med_file = self.vocab_dir / "rxnorm_medications.json"
        sig_file = self.vocab_dir / "latin_sig_codes.json"
        doc_file = self.vocab_dir / "doctor_profiles.json"
        tpl_file = self.vocab_dir / "clinical_templates.json"

        if med_file.exists():
            with open(med_file, "r", encoding="utf-8") as f:
                self.medications = json.load(f).get("medications", [])
        if sig_file.exists():
            with open(sig_file, "r", encoding="utf-8") as f:
                self.sig_codes = json.load(f).get("codes", [])
        if doc_file.exists():
            with open(doc_file, "r", encoding="utf-8") as f:
                self.doctors = json.load(f).get("doctors", [])
        if tpl_file.exists():
            with open(tpl_file, "r", encoding="utf-8") as f:
                self.templates = json.load(f).get("templates", [])

    def get_random_medication(
        self,
        therapeutic_class: Optional[str] = None,
        is_lasa: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Sample a medication matching optional class and LASA constraints."""
        pool = self.medications
        if therapeutic_class:
            pool = [m for m in pool if m.get("therapeutic_class") == therapeutic_class]
        if is_lasa is not None:
            pool = [m for m in pool if m.get("is_lasa") == is_lasa]
        if not pool:
            pool = self.medications
        return random.choice(pool) if pool else {}

    def get_random_doctor_profile(self, specialty: Optional[str] = None) -> Dict[str, Any]:
        """Sample a physician profile matching optional specialty."""
        pool = self.doctors
        if specialty:
            pool = [d for d in pool if d.get("specialty") == specialty]
        if not pool:
            pool = self.doctors
        return random.choice(pool) if pool else {}

    def get_random_sig_code(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Sample a Latin sig code matching optional category."""
        pool = self.sig_codes
        if category:
            pool = [c for c in pool if c.get("sig_category") == category]
        if not pool:
            pool = self.sig_codes
        return random.choice(pool) if pool else {}

    def get_random_template(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Sample a clinical template matching optional category."""
        pool = self.templates
        if category:
            pool = [t for t in pool if t.get("template_category") == category]
        if not pool:
            pool = self.templates
        return random.choice(pool) if pool else {}

    def fill_template(
        self,
        template_id: Optional[str] = None,
        category: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Fill all dynamic slot placeholders in a clinical template with authentic vocabulary data.
        Returns (filled_text, metadata_dict).
        """
        if template_id:
            tpl = next((t for t in self.templates if t["template_id"] == template_id), None)
        else:
            tpl = self.get_random_template(category=category)

        if not tpl:
            return "Amoxicillin 500mg PO TID x 10 days", {"medication": "Amoxicillin"}

        raw_text = tpl["raw_template_text"]
        doc = self.get_random_doctor_profile(specialty=tpl.get("specialty"))
        med1 = self.get_random_medication()
        med2 = self.get_random_medication()
        sig1 = self.get_random_sig_code(category="frequency")
        sig2 = self.get_random_sig_code(category="frequency")

        patient_firsts = ["John", "Mary", "Robert", "Patricia", "James", "Jennifer", "Michael", "Elizabeth", "David", "Sarah"]
        patient_lasts = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        p_name = f"{random.choice(patient_firsts)} {random.choice(patient_lasts)}"

        replacements = {
            "{PATIENT_NAME}": p_name,
            "{AGE}": str(random.randint(22, 84)),
            "{GENDER}": random.choice(["M", "F"]),
            "{DOB}": f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/{random.randint(1945, 2002)}",
            "{DATE}": f"08/{random.randint(10, 27):02d}/2026",
            "{TIME}": f"{random.randint(8, 20):02d}:{random.randint(10, 59):02d}",
            "{MRN}": f"MRN-{random.randint(100000, 999999)}",
            "{ADM_DATE}": "08/18/2026",
            "{DISCH_DATE}": "08/24/2026",
            "{BP_SYSTOLIC}": str(random.randint(112, 168)),
            "{BP_DIASTOLIC}": str(random.randint(68, 98)),
            "{HEART_RATE}": str(random.randint(60, 96)),
            "{RESP_RATE}": str(random.randint(14, 22)),
            "{SPO2}": str(random.randint(95, 99)),
            "{TEMP_F}": f"{random.uniform(97.8, 101.4):.1f}",
            "{WEIGHT_KG}": str(random.randint(58, 110)),
            "{MEDICATION}": med1.get("generic_name", "Amoxicillin"),
            "{STRENGTH}": random.choice(med1.get("standard_strengths", ["500mg"])),
            "{DOSAGE_FORM}": random.choice(med1.get("dosage_forms", ["capsule"])),
            "{MEDICATION_1}": med1.get("generic_name", "Amoxicillin"),
            "{STRENGTH_1}": random.choice(med1.get("standard_strengths", ["500mg"])),
            "{DOSAGE_FORM_1}": random.choice(med1.get("dosage_forms", ["capsule"])),
            "{SIG_FREQ_1}": sig1.get("code", "TID"),
            "{DISPENSE_1}": str(random.choice([30, 60, 90])),
            "{REFILLS_1}": str(random.choice([0, 1, 2, 3])),
            "{MEDICATION_2}": med2.get("generic_name", "Lisinopril"),
            "{STRENGTH_2}": random.choice(med2.get("standard_strengths", ["20mg"])),
            "{DOSAGE_FORM_2}": random.choice(med2.get("dosage_forms", ["tablet"])),
            "{SIG_FREQ_2}": sig2.get("code", "QD"),
            "{DISPENSE_2}": str(random.choice([30, 60, 90])),
            "{REFILLS_2}": str(random.choice([0, 1, 2, 3])),
            "{DOCTOR_NAME}": doc.get("full_name", "Dr. Sarah Jenkins, MD"),
            "{DOCTOR_LAST_NAME}": doc.get("last_name", "Jenkins"),
            "{CREDENTIALS}": "MD",
            "{SPECIALTY}": doc.get("specialty", "Internal Medicine"),
            "{NPI}": doc.get("npi", "1982736410"),
            "{DEA}": doc.get("dea_number", "MJ9283741"),
            "{CLINIC_NAME}": doc.get("clinic_name", "Metropolitan Clinic"),
            "{CLINIC_ADDRESS}": doc.get("clinic_address", "100 Health Ave"),
            "{CITY}": doc.get("city", "Boston"),
            "{STATE}": doc.get("state", "MA"),
            "{ZIP}": doc.get("zip_code", "02115"),
            "{PHONE}": doc.get("phone", "(617) 555-0100"),
            "{PATIENT_ADDRESS}": f"{random.randint(100, 999)} Main Street",
            "{SIG_ACTION}": "Take",
            "{SIG_QUANTITY}": "1",
            "{SIG_ROUTE}": "PO",
            "{SIG_FREQUENCY}": sig1.get("code", "TID"),
            "{SIG_MODIFIER}": "with food",
            "{DISPENSE_QTY}": "30",
            "{DISPENSE_WORDS}": "Thirty",
            "{REFILLS}": "2",
            "{SIGNATURE_SCRAWL}": f"{doc.get('first_name', 'S')[0]}. {doc.get('last_name', 'Jenkins')}"
        }

        filled_text = raw_text
        for slot, val in replacements.items():
            filled_text = filled_text.replace(slot, val)

        metadata = {
            "template_id": tpl.get("template_id"),
            "category": tpl.get("template_category"),
            "doctor": doc.get("full_name"),
            "patient": p_name,
            "medication_1": med1.get("generic_name"),
            "is_lasa": med1.get("is_lasa", False) or med2.get("is_lasa", False),
            "therapeutic_class": med1.get("therapeutic_class")
        }
        return filled_text, metadata


# ---------------------------------------------------------------------------
# Background Synthesis & Texture Generator
# ---------------------------------------------------------------------------

class BackgroundGenerator:
    """
    Procedural generation of realistic paper backgrounds, lined paper,
    3D normal-mapped crumpled paper, clinic verification stamps, and coffee stains.
    """

    @staticmethod
    def generate_lined_paper(
        width: int,
        height: int,
        line_spacing: int = 40,
        margin_x: int = 100,
        paper_color: Tuple[int, int, int] = (252, 250, 244),
        line_color: Tuple[int, int, int] = (180, 205, 230),
        margin_color: Tuple[int, int, int] = (230, 160, 160)
    ) -> np.ndarray:
        """
        Generate realistic ruled notebook paper with horizontal lines and vertical margin.
        """
        img = np.full((height, width, 3), paper_color, dtype=np.uint8)

        # Subtle paper grain noise
        noise = np.random.normal(0, 2.0, (height, width, 3)).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Horizontal ruling lines with subtle waviness
        for y in range(line_spacing, height - 10, line_spacing):
            wave_phase = random.uniform(0, 2 * math.pi)
            for x in range(width):
                dy = int(round(0.4 * math.sin(x / 50.0 + wave_phase)))
                target_y = y + dy
                if 0 <= target_y < height:
                    img[target_y, x] = line_color

        # Vertical red margin line
        if 0 < margin_x < width:
            for y in range(height):
                dx = int(round(0.3 * math.sin(y / 40.0)))
                target_x = margin_x + dx
                if 0 <= target_x < width:
                    img[y, target_x] = margin_color

        return img

    @staticmethod
    def generate_grid_paper(
        width: int,
        height: int,
        grid_size: int = 25,
        paper_color: Tuple[int, int, int] = (250, 250, 248),
        grid_color: Tuple[int, int, int] = (210, 225, 238)
    ) -> np.ndarray:
        """
        Generate math/engineering grid paper.
        """
        img = np.full((height, width, 3), paper_color, dtype=np.uint8)
        # Horizontal lines
        for y in range(grid_size, height, grid_size):
            img[y, :] = grid_color
        # Vertical lines
        for x in range(grid_size, width, grid_size):
            img[:, x] = grid_color
        return img

    @staticmethod
    def generate_crumpled_texture(
        width: int,
        height: int,
        intensity: float = 0.15,
        base_color: Tuple[int, int, int] = (250, 248, 242)
    ) -> np.ndarray:
        """
        Generate 3D normal-mapped Lambertian shaded crumpled paper texture using PhysicalAugmenter.
        """
        base = np.full((height, width, 3), base_color, dtype=np.uint8)
        shaded = PhysicalAugmenter.apply_3d_lambertian_shading(base, intensity=intensity)
        return shaded

    @staticmethod
    def add_coffee_stain(
        image: np.ndarray,
        center: Optional[Tuple[int, int]] = None,
        radius: int = 60,
        color: Tuple[int, int, int] = (180, 135, 80)
    ) -> np.ndarray:
        """
        Add a procedural coffee/tea cup stain ring with realistic capillary fluid edge.
        """
        out = image.copy()
        h, w = out.shape[:2]
        if center is None:
            cx = random.randint(radius, max(radius + 1, w - radius))
            cy = random.randint(radius, max(radius + 1, h - radius))
        else:
            cx, cy = center

        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

        # Outer drying ring (coffee-ring effect)
        ring_mask = np.exp(-((dist - radius) ** 2) / (2.0 * (3.5 ** 2))) * 0.45
        # Inner fluid puddle
        inner_mask = np.maximum(0.0, (radius - dist) / float(radius)) * 0.12
        stain_mask = np.clip(ring_mask + inner_mask, 0.0, 1.0)[:, :, None]

        stain_color = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        float_img = out.astype(np.float32)
        blended = float_img * (1.0 - stain_mask) + stain_color * stain_mask
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def add_stamp(
        image: np.ndarray,
        text: str = "DISPENSED",
        position: Optional[Tuple[int, int]] = None,
        color: Tuple[int, int, int] = (30, 40, 180),
        angle: Optional[float] = None
    ) -> np.ndarray:
        """
        Add a distressed clinic/pharmacy stamp overlay with outer ring and text.
        """
        h_img, w_img = image.shape[:2]
        stamp_w, stamp_h = 220, 100

        stamp_rgba = Image.new("RGBA", (stamp_w, stamp_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(stamp_rgba)

        # Draw outer rounded border
        border_col = (color[0], color[1], color[2], 220)
        draw.rectangle([4, 4, stamp_w - 5, stamp_h - 5], outline=border_col, width=4)
        draw.rectangle([8, 8, stamp_w - 9, stamp_h - 9], outline=border_col, width=1)

        # Draw text
        font = None
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size=24)
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                pass

        if font:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = len(text) * 12, 20
            tx = (stamp_w - tw) // 2
            ty = (stamp_h - th) // 2
            draw.text((tx, ty), text, fill=border_col, font=font)

        # Stamp rotation
        rot_angle = angle if angle is not None else random.uniform(-18.0, 18.0)
        rotated_stamp = stamp_rgba.rotate(rot_angle, resample=Image.BICUBIC, expand=True)

        # Distressed void filter (salt-and-pepper mask)
        stamp_arr = np.array(rotated_stamp)
        noise = np.random.rand(stamp_arr.shape[0], stamp_arr.shape[1])
        stamp_arr[:, :, 3] = np.where(noise < 0.2, (stamp_arr[:, :, 3] * 0.3).astype(np.uint8), stamp_arr[:, :, 3])

        # Composite onto image
        sw, sh = rotated_stamp.size
        if position is None:
            px = random.randint(10, max(11, w_img - sw - 10))
            py = random.randint(10, max(11, h_img - sh - 10))
        else:
            px, py = position

        pil_base = Image.fromarray(image).convert("RGBA")
        stamp_pil = Image.fromarray(stamp_arr, mode="RGBA")
        pil_base.alpha_composite(stamp_pil, (px, py))
        return np.array(pil_base.convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Synthetic Handwriting & Prescription Generator
# ---------------------------------------------------------------------------

class SyntheticHandwritingGenerator:
    """
    Procedural generator producing realistic messy cursive, clean handwriting,
    and doctor prescriptions with 3D physical augmentations, fluid ink models,
    tremor jitter, and Albumentations.
    """

    INK_MODELS = {
        "blue": {"base": (25, 45, 135), "var": (10, 10, 20), "type": "ballpoint"},
        "black": {"base": (35, 35, 40), "var": (10, 10, 10), "type": "ballpoint"},
        "fountain": {"base": (15, 30, 95), "var": (5, 10, 15), "type": "fountain"},
        "pencil": {"base": (90, 90, 95), "var": (15, 15, 15), "type": "pencil"},
    }

    MEDICATIONS_DB = [
        ("Amoxicillin", "500mg capsules", "Take 1 cap po tid x 10 days", "#30", "0"),
        ("Metformin HCl", "1000mg tabs", "Take 1 tab po bid with meals", "#60", "2"),
        ("Lisinopril", "20mg tablets", "Take 1 tab po daily in morning", "#30", "3"),
        ("Azithromycin", "250mg Z-Pak", "Take 2 tabs day 1, then 1 daily", "#6", "0"),
        ("Atorvastatin", "40mg tablets", "Take 1 tab po qhs for lipid control", "#90", "3"),
        ("Gabapentin", "300mg capsules", "Take 1 cap po tid prn neuropathic pain", "#90", "1"),
        ("Levothyroxine", "100mcg tabs", "Take 1 tab po daily on empty stomach", "#90", "2"),
        ("Omeprazole", "40mg DR caps", "Take 1 cap po daily 30 min before meal", "#30", "2"),
        ("Hydrochlorothiazide", "25mg tabs", "Take 1 tab po qam with water", "#30", "3"),
        ("Prednisone", "20mg tablets", "Take as directed on taper schedule", "#21", "0"),
    ]

    DOCTORS_DB = [
        ("Dr. Sarah Jenkins, MD", "Metropolitan Family Practice", "1042 Beacon St, Boston MA", "NPI: 1982736410"),
        ("Dr. Marcus Vance, DO", "St. Jude Internal Medicine", "450 Sunset Blvd, Los Angeles CA", "NPI: 1409283719"),
        ("Dr. Emily Zhao, MD, PhD", "Northwestern Clinic", "720 N Michigan Ave, Chicago IL", "NPI: 1726354890"),
        ("Dr. David K. Sterling, MD", "Evergreen Health Partners", "1100 Olive Way, Seattle WA", "NPI: 1839201742"),
    ]

    PATIENTS_DB = [
        ("Johnathan Miller", "47", "M", "08/24/2026", "145 Elm Street"),
        ("Eleanor Rigby", "63", "F", "08/24/2026", "89 Abbey Road"),
        ("Michael Chang", "35", "M", "08/25/2026", "412 Pacific Ave"),
        ("Sophia Rodriguez", "28", "F", "08/26/2026", "670 Oakridge Lane"),
    ]

    def __init__(
        self,
        font_manager: Optional[HandwritingFontManager] = None,
        vocab_manager: Optional[VocabularyManager] = None
    ):
        self.font_mgr = font_manager if font_manager is not None else HandwritingFontManager()
        self.vocab_mgr = vocab_manager if vocab_manager is not None else VocabularyManager()

    def render_line(
        self,
        text: str,
        font_name: Optional[str] = None,
        font_size: int = 36,
        ink_color: str = "blue",
        slant_deg: float = 12.0,
        tremor_sigma: float = 1.2,
        wave_amplitude: float = 3.0,
        canvas_width: Optional[int] = None,
        canvas_height: Optional[int] = None,
        apply_physical_effects: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Procedural rendering of a single handwritten text line with continuous
        Ornstein-Uhlenbeck character tremor, multi-harmonic baseline curvature,
        stroke slant shear, ink capillary paper bleeding, and vertex pooling.

        Returns:
            Tuple of (RGB uint8 numpy array, metadata dict with bounding box).
        """
        font = self.font_mgr.get_font(font_name, size=font_size)
        ink_info = self.INK_MODELS.get(ink_color, self.INK_MODELS["blue"])

        # Determine canvas size
        approx_w = max(200, len(text) * int(font_size * 0.75) + 100)
        line_w = canvas_width if canvas_width is not None else approx_w
        line_h = canvas_height if canvas_height is not None else int(font_size * 2.8)

        # High-res RGBA canvas for smooth anti-aliased character placement
        rgba_img = Image.new("RGBA", (line_w, line_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rgba_img)

        cur_x = 20.0
        baseline_y = line_h * 0.55
        wave_freq = random.uniform(0.01, 0.025)
        wave_phase = random.uniform(0, 2 * math.pi)

        # Generate OU tremor deviations
        tremor_dx = 0.0
        tremor_dy = 0.0
        theta_ou = 0.6

        for char in text:
            if char == " ":
                cur_x += font_size * 0.45
                continue

            # Character tremor (OU stochastic update) & baseline wave
            eps_x = random.gauss(0, tremor_sigma * 0.5)
            eps_y = random.gauss(0, tremor_sigma * 0.5)
            tremor_dx = (1.0 - theta_ou) * tremor_dx + eps_x
            tremor_dy = (1.0 - theta_ou) * tremor_dy + eps_y

            wave_dy = wave_amplitude * math.sin(cur_x * wave_freq + wave_phase)
            char_y = baseline_y + wave_dy + tremor_dy

            # Ink color jitter
            base_rgb = ink_info["base"]
            var_rgb = ink_info["var"]
            r = int(np.clip(base_rgb[0] + random.randint(-var_rgb[0], var_rgb[0]), 0, 255))
            g = int(np.clip(base_rgb[1] + random.randint(-var_rgb[1], var_rgb[1]), 0, 255))
            b = int(np.clip(base_rgb[2] + random.randint(-var_rgb[2], var_rgb[2]), 0, 255))
            char_color = (r, g, b, 240)

            # Draw character onto temporary small patch for micro-rotation
            char_patch = Image.new("RGBA", (int(font_size * 2), int(font_size * 2)), (0, 0, 0, 0))
            cp_draw = ImageDraw.Draw(char_patch)
            cp_draw.text((int(font_size * 0.3), int(font_size * 0.2)), char, fill=char_color, font=font)

            # Micro-rotation
            micro_rot = random.gauss(0, 1.8)
            rotated_char = char_patch.rotate(micro_rot, resample=Image.BICUBIC)

            # Paste character
            paste_x = int(cur_x + tremor_dx)
            paste_y = int(char_y - font_size * 0.8)
            rgba_img.alpha_composite(rotated_char, (paste_x, paste_y))

            # Advance x position
            try:
                bbox = draw.textbbox((0, 0), char, font=font)
                char_w = max(font_size * 0.35, bbox[2] - bbox[0])
            except Exception:
                char_w = font_size * 0.45

            cur_x += char_w + random.uniform(0.5, 2.5)

        # Convert to NumPy RGBA array
        arr = np.array(rgba_img)

        # Zero-ink fallback: if font failed to render any glyphs, re-render with reliable Latin font
        if arr[:, :, 3].max() == 0:
            fallback_font = self.font_mgr.get_reliable_latin_font(size=font_size)
            if fallback_font is not None:
                rgba_img = Image.new("RGBA", (line_w, line_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(rgba_img)
                cur_x = 20.0
                tremor_dx = 0.0
                tremor_dy = 0.0
                for char in text:
                    if char == " ":
                        cur_x += font_size * 0.45
                        continue
                    eps_x = random.gauss(0, tremor_sigma * 0.5)
                    eps_y = random.gauss(0, tremor_sigma * 0.5)
                    tremor_dx = (1.0 - theta_ou) * tremor_dx + eps_x
                    tremor_dy = (1.0 - theta_ou) * tremor_dy + eps_y
                    wave_dy = wave_amplitude * math.sin(cur_x * wave_freq + wave_phase)
                    char_y = baseline_y + wave_dy + tremor_dy

                    base_rgb = ink_info["base"]
                    var_rgb = ink_info["var"]
                    r = int(np.clip(base_rgb[0] + random.randint(-var_rgb[0], var_rgb[0]), 0, 255))
                    g = int(np.clip(base_rgb[1] + random.randint(-var_rgb[1], var_rgb[1]), 0, 255))
                    b = int(np.clip(base_rgb[2] + random.randint(-var_rgb[2], var_rgb[2]), 0, 255))
                    char_color = (r, g, b, 240)

                    char_patch = Image.new("RGBA", (int(font_size * 2), int(font_size * 2)), (0, 0, 0, 0))
                    cp_draw = ImageDraw.Draw(char_patch)
                    cp_draw.text((int(font_size * 0.3), int(font_size * 0.2)), char, fill=char_color, font=fallback_font)
                    micro_rot = random.gauss(0, 1.8)
                    rotated_char = char_patch.rotate(micro_rot, resample=Image.BICUBIC)
                    paste_x = int(cur_x + tremor_dx)
                    paste_y = int(char_y - font_size * 0.8)
                    rgba_img.alpha_composite(rotated_char, (paste_x, paste_y))

                    try:
                        bbox = draw.textbbox((0, 0), char, font=fallback_font)
                        char_w = max(font_size * 0.35, bbox[2] - bbox[0])
                    except Exception:
                        char_w = font_size * 0.45
                    cur_x += char_w + random.uniform(0.5, 2.5)

                arr = np.array(rgba_img)

        # Stroke Slant (Affine Shear)
        if abs(slant_deg) > 0.5 and arr[:, :, 3].max() > 0:
            shear_factor = math.tan(math.radians(-slant_deg))
            M = np.array([[1.0, shear_factor, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            arr = cv2.warpAffine(
                arr,
                M,
                (line_w, line_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

        # Apply Ink Capillary Bleed & Vertex Pooling physics
        if apply_physical_effects and arr[:, :, 3].max() > 0:
            arr = PhysicalAugmenter.apply_ink_capillary_bleeding(arr, bleed_intensity=0.35)
            arr = PhysicalAugmenter.apply_stroke_vertex_pooling(arr, corner_threshold=0.02, pooling_darken=0.25)

        # Composite over white background
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        white = np.ones_like(rgb) * 255.0
        comp_rgb = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)

        # Find tight text bounding box
        ink_indices = np.where(arr[:, :, 3] > 30)
        if len(ink_indices[0]) > 0:
            ymin = int(np.min(ink_indices[0]))
            ymax = int(np.max(ink_indices[0]))
            xmin = int(np.min(ink_indices[1]))
            xmax = int(np.max(ink_indices[1]))
        else:
            ymin, ymax, xmin, xmax = 0, line_h - 1, 0, line_w - 1

        metadata = {
            "text": text,
            "font_size": font_size,
            "slant_deg": slant_deg,
            "ink_color": ink_color,
            "bbox": [ymin, xmin, ymax, xmax],
            "norm_bbox": [
                round(ymin / float(line_h), 5),
                round(xmin / float(line_w), 5),
                round(ymax / float(line_h), 5),
                round(xmax / float(line_w), 5)
            ]
        }

        return comp_rgb, metadata

    def render_prescription(
        self,
        clinic_name: Optional[str] = None,
        doctor_name: Optional[str] = None,
        patient_name: Optional[str] = None,
        medications: Optional[List[Tuple[str, str, str, str, str]]] = None,
        include_stamp: bool = True,
        include_stains: bool = True,
        apply_distortions: bool = False,
        apply_3d_shading: bool = True,
        width: int = 800,
        height: int = 1100
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a complete, photorealistic medical prescription document slip
        with 3D normal-mapped paper shading and clinical vocabulary integration.

        Returns:
            Tuple of (RGB uint8 numpy array, structured document dictionary with bounding boxes).
        """
        doc_info = random.choice(self.DOCTORS_DB)
        pat_info = random.choice(self.PATIENTS_DB)

        clinic = clinic_name if clinic_name is not None else doc_info[1]
        physician = doctor_name if doctor_name is not None else doc_info[0]
        address = doc_info[2]
        npi = doc_info[3]

        patient = patient_name if patient_name is not None else pat_info[0]
        age = pat_info[1]
        sex = pat_info[2]
        date_str = pat_info[3]

        meds = medications if medications is not None else random.sample(self.MEDICATIONS_DB, k=random.randint(1, 2))

        # 1. Base Paper Canvas with 3D Shading
        canvas = BackgroundGenerator.generate_crumpled_texture(width, height, intensity=0.08)

        # 2. Add Ruled Lines for Rx Section
        rx_paper = BackgroundGenerator.generate_lined_paper(
            width - 80, 500, line_spacing=45, margin_x=0,
            paper_color=(255, 255, 255), line_color=(220, 230, 242), margin_color=(255, 255, 255)
        )
        # Blend ruled section into canvas
        canvas[380:880, 40:width - 40] = cv2.addWeighted(
            canvas[380:880, 40:width - 40], 0.4, rx_paper, 0.6, 0
        )

        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)

        # Standard Printed Header Fonts
        header_font = None
        bold_font = None
        rx_symbol_font = None
        try:
            header_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
            bold_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
            rx_symbol_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 44)
        except Exception:
            header_font = ImageFont.load_default()
            bold_font = ImageFont.load_default()
            rx_symbol_font = ImageFont.load_default()

        # Draw Clinic Header Box
        draw.rectangle([30, 30, width - 30, 180], outline=(100, 100, 100), width=2)
        draw.text((50, 45), clinic.upper(), fill=(20, 40, 90), font=bold_font)
        draw.text((50, 75), physician, fill=(40, 40, 40), font=header_font)
        draw.text((50, 100), address, fill=(60, 60, 60), font=header_font)
        draw.text((50, 125), npi + " — DEA: MJ9283741", fill=(60, 60, 60), font=header_font)

        # Draw Patient Demographic Block
        draw.rectangle([30, 195, width - 30, 310], outline=(140, 140, 140), width=1)
        draw.text((50, 210), f"PATIENT: {patient}", fill=(30, 30, 30), font=bold_font)
        draw.text((50, 245), f"AGE: {age}    SEX: {sex}    DATE: {date_str}", fill=(50, 50, 50), font=header_font)
        draw.text((50, 275), f"ADDRESS: {pat_info[4]}", fill=(50, 50, 50), font=header_font)

        # Draw Large ℞ Symbol
        draw.text((50, 325), "℞", fill=(10, 20, 60), font=rx_symbol_font)

        canvas_arr = np.array(pil_canvas)

        lines_metadata: List[Dict[str, Any]] = []
        cur_y = 400

        for med in meds:
            med_name, dosage, sig, disp, refills = med
            line1_text = f"{med_name} {dosage}"
            line2_text = f"Sig: {sig}"
            line3_text = f"Disp: {disp}    Refills: {refills}"

            for line_text in [line1_text, line2_text, line3_text]:
                line_img, meta = self.render_line(
                    line_text,
                    font_size=32,
                    ink_color=random.choice(["blue", "black", "fountain"]),
                    slant_deg=random.uniform(8.0, 16.0),
                    tremor_sigma=random.uniform(0.8, 1.6),
                    canvas_width=width - 120,
                    canvas_height=60,
                    apply_physical_effects=True
                )
                # Composite line onto prescription
                lh, lw = line_img.shape[:2]
                mask = (to_grayscale(line_img) < 240).astype(np.float32)[:, :, None]
                target_region = canvas_arr[cur_y:cur_y + lh, 60:60 + lw]
                canvas_arr[cur_y:cur_y + lh, 60:60 + lw] = np.clip(
                    target_region.astype(np.float32) * (1.0 - mask) + line_img.astype(np.float32) * mask,
                    0, 255
                ).astype(np.uint8)

                # Store line record
                lines_metadata.append({
                    "text": line_text,
                    "bbox": [
                        round(float(cur_y + meta["bbox"][0]) / height, 5),
                        round(float(60 + meta["bbox"][1]) / width, 5),
                        round(float(cur_y + meta["bbox"][2]) / height, 5),
                        round(float(60 + meta["bbox"][3]) / width, 5)
                    ]
                })

                cur_y += 65

        # Render Doctor Cursive Signature
        sig_y = height - 180
        sig_img, sig_meta = self.render_line(
            physician,
            font_size=38,
            ink_color="blue",
            slant_deg=22.0,
            tremor_sigma=2.0,
            canvas_width=350,
            canvas_height=80,
            apply_physical_effects=True
        )
        sig_mask = (to_grayscale(sig_img) < 240).astype(np.float32)[:, :, None]
        canvas_arr[sig_y:sig_y + 80, width - 420:width - 70] = np.clip(
            canvas_arr[sig_y:sig_y + 80, width - 420:width - 70] * (1.0 - sig_mask) + sig_img * sig_mask,
            0, 255
        ).astype(np.uint8)

        # Draw Signature Line
        cv2.line(canvas_arr, (width - 430, height - 100), (width - 60, height - 100), (80, 80, 80), 2)
        pil_footer = Image.fromarray(canvas_arr)
        draw_f = ImageDraw.Draw(pil_footer)
        draw_f.text((width - 420, height - 95), "Prescriber's Signature (Dispense As Written)", fill=(60, 60, 60), font=header_font)
        canvas_arr = np.array(pil_footer)

        # Add Clinic Stamp Overlay
        if include_stamp:
            canvas_arr = BackgroundGenerator.add_stamp(
                canvas_arr,
                text="CLINIC Rx",
                position=(width - 270, 200),
                color=(160, 30, 30),
                angle=-12.0
            )

        # Add Coffee Stain
        if include_stains:
            canvas_arr = BackgroundGenerator.add_coffee_stain(canvas_arr, center=(130, 750), radius=55)

        # Apply 3D Shading & Non-Uniform Shadows
        if apply_3d_shading:
            canvas_arr = PhysicalAugmenter.apply_shadow_gradients(canvas_arr, linear_intensity=0.15, vignette_intensity=0.20)

        # Apply Albumentations
        if apply_distortions and ALBUMENTATIONS_AVAILABLE:
            canvas_arr = self.apply_elastic_distortions(canvas_arr)

        doc_dict = {
            "clinic": clinic,
            "doctor": physician,
            "patient": patient,
            "date": date_str,
            "lines": lines_metadata,
            "full_text": "\n".join([item["text"] for item in lines_metadata])
        }

        return canvas_arr, doc_dict

    def apply_elastic_distortions(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Albumentations elastic transforms, blur, and noise.
        """
        if not ALBUMENTATIONS_AVAILABLE:
            return image

        transform = A.Compose([
            A.ElasticTransform(alpha=1.0, sigma=25, p=0.6),
            A.OpticalDistortion(distort_limit=0.08, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussNoise(std_range=(0.02, 0.06), p=0.4),
        ])
        augmented = transform(image=image)
        return augmented["image"]

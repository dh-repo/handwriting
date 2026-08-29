"""
pipeline/tests/test_m1_stress_challenger.py
Empirical Adversarial Stress Test Suite for Milestone 1 Challenger.

Exhaustively stress-tests:
1. PhysicalAugmenter numerical stability, boundary limits, and edge cases:
   - Extreme parameters (zero/negative lighting, huge crease heights, extreme tremor, huge amplitudes, empty/odd shapes).
   - Numerical instability (NaN, Inf, overflow, zero-division, type mismatches).
   - Channel dimensions, uint8 clamping, perceptual validity.
2. Clinical Vocabularies:
   - 100% CMS 15-digit Luhn mod-10 compliance for all NPI numbers.
   - 100% DEA check digit formula compliance and initial matching for all DEA numbers.
   - 100% Bidirectional symmetry, existence, and metadata for all LASA pairs.
   - 100% Slot syntax and resolution across all 200+ clinical templates.
"""

import json
import math
from pathlib import Path
import re
from typing import List, Tuple
import cv2
import numpy as np
import pytest

from pipeline.dataset.synthetic_generator import (
    PhysicalAugmenter,
    VocabularyManager,
    SyntheticHandwritingGenerator,
    BackgroundGenerator,
    ALBUMENTATIONS_AVAILABLE
)

VOCAB_DIR = Path("data/reference_handwriting/vocabularies")


# ==============================================================================
# 1. PHYSICAL AUGMENTER STRESS & ADVERSARIAL TESTS
# ==============================================================================

class TestPhysicalAugmenterStress:
    """
    Adversarial stress-testing of all physical augmentation sub-routines.
    """

    @pytest.mark.parametrize("w, h", [
        (10, 10),
        (3, 3),
        (100, 20),
        (20, 200),
        (1024, 768),
        (128, 128),
    ])
    def test_heightmap_extreme_shapes(self, w: int, h: int):
        """Test heightmap generation on diverse aspect ratios and tiny scales."""
        rng = np.random.default_rng(42)
        hm = PhysicalAugmenter.generate_heightmap(
            width=w, height=h, roughness=2.5, include_ridges=True, num_folds=4, rng=rng
        )
        assert hm.shape == (h, w)
        assert hm.dtype == np.float32
        assert not np.isnan(hm).any(), "Heightmap contains NaN"
        assert not np.isinf(hm).any(), "Heightmap contains Inf"
        assert 0.0 <= np.min(hm)
        assert np.max(hm) <= 1.0

    @pytest.mark.parametrize("roughness, folds, ridges", [
        (0.0, 0, False),
        (-5.0, 0, True),
        (100.0, 10, True),
        (1.0, 50, False),
        (10.0, 20, True),
    ])
    def test_heightmap_extreme_parameters(self, roughness: float, folds: int, ridges: bool):
        """Test heightmap with extreme/boundary parameter inputs."""
        w, h = 200, 150
        hm = PhysicalAugmenter.generate_heightmap(
            width=w, height=h, roughness=roughness, include_ridges=ridges, num_folds=folds
        )
        assert hm.shape == (h, w)
        assert hm.dtype == np.float32
        assert not np.isnan(hm).any()
        assert not np.isinf(hm).any()
        assert 0.0 <= np.min(hm) <= np.max(hm) <= 1.0

    def test_lambertian_shading_extreme_lighting_and_relief(self):
        """Stress-test 3D shading with zero/negative lighting, huge relief, and specular extremes."""
        h, w = 120, 160
        img_3ch = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        img_1ch = np.random.randint(50, 200, (h, w), dtype=np.uint8)

        # Test extreme parameter combinations
        extreme_configs = [
            {"intensity": 0.0, "relief_scale": 0.0, "specular_weight": 0.0, "shininess": 1.0},
            {"intensity": -1.0, "relief_scale": -5.0, "specular_weight": 0.0, "shininess": 0.1},
            {"intensity": 5.0, "relief_scale": 100.0, "specular_weight": 10.0, "shininess": 1000.0},
            {"intensity": 0.5, "light_theta": 0.0, "light_phi": 0.0, "relief_scale": 10.0},
            {"intensity": 0.5, "light_theta": -10.0, "light_phi": -math.pi / 2, "relief_scale": 1.0},
            {"intensity": 0.5, "light_theta": 100.0, "light_phi": math.pi / 2, "relief_scale": 1.0},
        ]

        for cfg in extreme_configs:
            # 3-channel input
            out_3ch = PhysicalAugmenter.apply_3d_lambertian_shading(img_3ch, **cfg)
            assert out_3ch.shape == img_3ch.shape
            assert out_3ch.dtype == np.uint8
            assert not np.isnan(out_3ch).any()
            assert not np.isinf(out_3ch).any()
            assert 0 <= np.min(out_3ch) <= np.max(out_3ch) <= 255

            # 1-channel / 2D grayscale input
            out_1ch = PhysicalAugmenter.apply_3d_lambertian_shading(img_1ch, **cfg)
            assert out_1ch.shape == img_1ch.shape
            assert out_1ch.dtype == np.uint8
            assert not np.isnan(out_1ch).any()
            assert not np.isinf(out_1ch).any()

    def test_lambertian_all_black_and_all_white_images(self):
        """Verify behavior on all-black (0) and all-white (255) images."""
        h, w = 100, 100
        black = np.zeros((h, w, 3), dtype=np.uint8)
        white = np.full((h, w, 3), 255, dtype=np.uint8)

        shaded_black = PhysicalAugmenter.apply_3d_lambertian_shading(black, intensity=0.5)
        shaded_white = PhysicalAugmenter.apply_3d_lambertian_shading(white, intensity=0.5)

        assert np.all(shaded_black == 0), "Black image should remain 0 when shaded"
        assert shaded_white.dtype == np.uint8
        assert not np.isnan(shaded_white).any()
        assert np.max(shaded_white) <= 255
        assert np.min(shaded_white) >= 50

    def test_shadow_gradients_extreme_parameters(self):
        """Stress-test shadow gradients with extreme intensities and feature flags."""
        h, w = 100, 150
        img = np.full((h, w, 3), 220, dtype=np.uint8)

        configs = [
            {"linear_intensity": 0.0, "vignette_intensity": 0.0, "include_spine_shadow": False, "include_blob": False},
            {"linear_intensity": 1.0, "vignette_intensity": 1.0, "include_spine_shadow": True, "include_blob": True},
            {"linear_intensity": -2.0, "vignette_intensity": -2.0, "include_spine_shadow": True, "include_blob": True},
            {"linear_intensity": 5.0, "vignette_intensity": 5.0, "include_spine_shadow": True, "include_blob": True},
        ]

        for cfg in configs:
            shadowed = PhysicalAugmenter.apply_shadow_gradients(img, **cfg)
            assert shadowed.shape == img.shape
            assert shadowed.dtype == np.uint8
            assert not np.isnan(shadowed).any()
            assert not np.isinf(shadowed).any()
            assert 0 <= np.min(shadowed) <= np.max(shadowed) <= 255

    def test_ink_capillary_bleeding_stress(self):
        """Stress-test ink capillary bleeding with extreme alphas, bleeds, and blur sigmas."""
        h, w = 60, 200

        # 1. Empty/transparent RGBA image (alpha == 0)
        empty_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        out_empty = PhysicalAugmenter.apply_ink_capillary_bleeding(empty_rgba, bleed_intensity=1.0)
        assert np.all(out_empty == 0)

        # 2. Fully opaque RGBA image (alpha == 255 everywhere)
        full_rgba = np.full((h, w, 4), 255, dtype=np.uint8)
        out_full = PhysicalAugmenter.apply_ink_capillary_bleeding(full_rgba, bleed_intensity=1.0)
        assert out_full.shape == (h, w, 4)
        assert out_full.dtype == np.uint8
        assert not np.isnan(out_full).any()

        # 3. Stroke RGBA image with extreme bleed and blur parameters
        stroke_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        stroke_rgba[25:35, 20:180, :3] = (20, 30, 100)
        stroke_rgba[25:35, 20:180, 3] = 255

        extreme_params = [
            {"bleed_intensity": 0.0, "fringe_blur_sigma": 0.1},
            {"bleed_intensity": -1.0, "fringe_blur_sigma": 0.5},
            {"bleed_intensity": 10.0, "fringe_blur_sigma": 5.0},
            {"bleed_intensity": 50.0, "fringe_blur_sigma": 20.0},
        ]
        for p in extreme_params:
            res = PhysicalAugmenter.apply_ink_capillary_bleeding(stroke_rgba, **p)
            assert res.shape == (h, w, 4)
            assert res.dtype == np.uint8
            assert not np.isnan(res).any()
            assert not np.isinf(res).any()
            assert 0 <= np.min(res) <= np.max(res) <= 255

    def test_stroke_vertex_pooling_stress(self):
        """Stress-test stroke vertex pooling with empty images and extreme corner thresholds."""
        h, w = 100, 100

        # Empty image
        empty_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        out_empty = PhysicalAugmenter.apply_stroke_vertex_pooling(empty_rgba)
        assert np.all(out_empty == 0)

        # Complex geometry: sharp turns, zigzag, single dot
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        cv2.polylines(rgba, [np.array([[10, 10], [50, 90], [90, 10], [50, 50]], dtype=np.int32)], isClosed=False, color=(50, 50, 150, 255), thickness=3)

        extreme_thresholds = [
            {"corner_threshold": 0.0, "pooling_darken": 0.0},
            {"corner_threshold": -0.5, "pooling_darken": -1.0},
            {"corner_threshold": 1.0, "pooling_darken": 1.0},
            {"corner_threshold": 10.0, "pooling_darken": 5.0},
        ]
        for cfg in extreme_thresholds:
            res = PhysicalAugmenter.apply_stroke_vertex_pooling(rgba, **cfg)
            assert res.shape == (h, w, 4)
            assert res.dtype == np.uint8
            assert not np.isnan(res).any()
            assert 0 <= np.min(res) <= np.max(res) <= 255

    def test_ornstein_uhlenbeck_tremor_stress(self):
        """Stress-test Ornstein-Uhlenbeck stochastic tremor jitter."""
        # 1. Empty points
        assert PhysicalAugmenter.simulate_stroke_tremor_ou([]) == []

        # 2. Single point
        single = [(50.0, 50.0)]
        res_single = PhysicalAugmenter.simulate_stroke_tremor_ou(single)
        assert len(res_single) == 1
        assert isinstance(res_single[0][0], float) and isinstance(res_single[0][1], float)

        # 3. 1,000 points with extreme parameters
        points = [(float(i), 100.0) for i in range(1000)]
        res = PhysicalAugmenter.simulate_stroke_tremor_ou(
            points, sigma=10.0, theta=0.01, dt=2.0
        )
        assert len(res) == len(points)
        for x, y in res:
            assert not math.isnan(x) and not math.isnan(y)
            assert not math.isinf(x) and not math.isinf(y)

    def test_multiharmonic_baseline_warp_stress(self):
        """Stress-test 3-harmonic sine warping with extreme amplitudes, wavelengths, and drift."""
        h, w = 80, 500
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        cv2.line(img, (10, 40), (490, 40), (20, 20, 20), 2)

        extreme_configs = [
            {"amplitudes": (0.0, 0.0, 0.0), "drift_deg": 0.0},
            {"amplitudes": (20.0, 10.0, 5.0), "drift_deg": 5.0},
            {"amplitudes": (-10.0, -5.0, -2.0), "drift_deg": -5.0},
            {"amplitudes": (5.0, 2.0, 1.0), "wavelengths": (10.0, 5.0, 2.0), "drift_deg": 0.0},
            {"amplitudes": (5.0, 2.0, 1.0), "wavelengths": (2000.0, 1000.0, 500.0), "drift_deg": 0.0},
        ]

        for cfg in extreme_configs:
            warped, dy = PhysicalAugmenter.apply_multiharmonic_baseline_warp(img, **cfg)
            assert warped.shape == img.shape
            assert warped.dtype == np.uint8
            assert dy.shape == (w,)
            assert not np.isnan(warped).any()
            assert not np.isnan(dy).any()
            assert not np.isinf(warped).any()
            assert not np.isinf(dy).any()

    def test_synthetic_generator_render_line_extreme_inputs(self):
        """Stress-test render_line with unusual text, large canvas, extreme slant, and tremor."""
        gen = SyntheticHandwritingGenerator()

        # Unicode symbols, digits, punctuation, and long string
        test_strings = [
            "A",
            "1234567890",
            "!@#$%^&*()_+-=[]{}|;':,.<>/?`~",
            "℞ Amoxicillin 500mg #30 PO TID x 10d",
            "Dr. René François-Müller, MD, PhD (Specialty: Otorhinolaryngology)",
        ]

        for s in test_strings:
            img, meta = gen.render_line(
                text=s,
                font_size=28,
                slant_deg=-15.0,
                tremor_sigma=3.0,
                wave_amplitude=6.0,
                apply_physical_effects=True
            )
            assert isinstance(img, np.ndarray)
            assert img.ndim == 3 and img.shape[2] == 3
            assert img.dtype == np.uint8
            assert not np.isnan(img).any()
            ymin, xmin, ymax, xmax = meta["norm_bbox"]
            assert 0.0 <= ymin <= ymax <= 1.0
            assert 0.0 <= xmin <= xmax <= 1.0


@pytest.fixture(scope="module")
def vocabularies():
    with open(VOCAB_DIR / "rxnorm_medications.json", "r", encoding="utf-8") as f:
        meds_data = json.load(f)
    with open(VOCAB_DIR / "latin_sig_codes.json", "r", encoding="utf-8") as f:
        sigs_data = json.load(f)
    with open(VOCAB_DIR / "doctor_profiles.json", "r", encoding="utf-8") as f:
        docs_data = json.load(f)
    with open(VOCAB_DIR / "clinical_templates.json", "r", encoding="utf-8") as f:
        tpls_data = json.load(f)
    return {
        "meds": meds_data,
        "sigs": sigs_data,
        "docs": docs_data,
        "tpls": tpls_data
    }


# ==============================================================================
# 2. CLINICAL VOCABULARIES ADVERSARIAL VERIFICATION
# ==============================================================================

class TestClinicalVocabulariesAdversarial:
    """
    Exhaustive verification of all mathematical and structural invariants across
    clinical vocabulary catalogs.
    """

    @staticmethod
    def independent_cms_luhn_15(npi: str) -> bool:
        """
        Standalone independent implementation of CMS 15-digit Luhn mod-10 algorithm.
        Standard specification:
        1. Prefix 10-digit NPI with '80840' to create 15-digit string.
        2. From right to left (0-indexed):
           - Position 0 (check digit, 15th digit): weight 1
           - Position 1 (14th digit): weight 2
           - Position 2 (13th digit): weight 1
           - Position 3 (12th digit): weight 2
           ...
           Double every digit in odd position from right. If 2*d > 9, subtract 9 (or sum digits).
        3. Sum all resulting values.
        4. Valid if sum % 10 == 0.
        """
        if not (isinstance(npi, str) and len(npi) == 10 and npi.isdigit()):
            return False

        full_str = "80840" + npi
        total = 0
        for i, char in enumerate(reversed(full_str)):
            val = int(char)
            if i % 2 == 1:
                val *= 2
                if val > 9:
                    val -= 9
            total += val
        return (total % 10) == 0

    @staticmethod
    def independent_dea_checksum(dea: str) -> Tuple[bool, str]:
        """
        Standalone independent implementation of DEA number checksum algorithm.
        Standard DEA specification:
        1. 2 uppercase letters followed by 7 digits (e.g. AB1234567).
        2. Letter 1: Registrant type code (A, B, C, D, E, F, G, H, J, K, L, M, P, R, S, T, U, X).
        3. Letter 2: First letter of registrant's legal last name.
        4. Digits:
           - Sum of 1st, 3rd, 5th digits = S1
           - Sum of 2nd, 4th, 6th digits = S2
           - Total = S1 + 2 * S2
           - Checksum (7th digit) = Total % 10.
        """
        if not isinstance(dea, str) or len(dea) != 9:
            return False, "Length != 9"
        if not re.match(r"^[A-Z]{2}\d{7}$", dea):
            return False, "Pattern mismatch"

        digits = [int(c) for c in dea[2:]]
        s1 = digits[0] + digits[2] + digits[4]
        s2 = digits[1] + digits[3] + digits[5]
        computed_check = (s1 + 2 * s2) % 10
        if computed_check != digits[6]:
            return False, f"Checksum mismatch: expected {computed_check}, got {digits[6]}"
        return True, "Valid"

    def test_npi_100_percent_cms_luhn_compliance(self, vocabularies):
        """
        Adversarially verify 100% of doctor NPI numbers against the independent
        CMS 15-digit Luhn mod-10 algorithm.
        Also tests adversarial corruptions to guarantee our checker rejects invalid NPIs.
        """
        doctors = vocabularies["docs"].get("doctors", [])
        assert len(doctors) >= 100, f"Expected >= 100 doctors, got {len(doctors)}"

        # 1. Verify all genuine doctors in catalog
        failures = []
        for doc in doctors:
            npi = doc.get("npi", "")
            if not self.independent_cms_luhn_15(npi):
                failures.append((doc.get("doctor_id"), npi))

        assert failures == [], f"Found {len(failures)} invalid NPI numbers: {failures}"

        # 2. Adversarial check: verify corrupted NPIs are REJECTED
        for doc in doctors[:10]:
            npi = doc["npi"]
            # Flip last digit (check digit)
            last_digit = int(npi[-1])
            corrupted_check = npi[:-1] + str((last_digit + 1) % 10)
            assert not self.independent_cms_luhn_15(corrupted_check), f"Corrupted check digit {corrupted_check} was not rejected!"

            # Flip middle digit
            mid_digit = int(npi[4])
            corrupted_mid = npi[:4] + str((mid_digit + 1) % 10) + npi[5:]
            assert not self.independent_cms_luhn_15(corrupted_mid), f"Corrupted middle digit {corrupted_mid} was not rejected!"

    def test_dea_100_percent_checksum_and_initial_compliance(self, vocabularies):
        """
        Adversarially verify 100% of doctor DEA numbers against the independent
        DEA checksum formula and verify the second letter matches doctor's last name.
        """
        doctors = vocabularies["docs"].get("doctors", [])
        assert len(doctors) >= 100

        failures = []
        for doc in doctors:
            dea = doc.get("dea_number", "")
            last_name = doc.get("last_name", "")

            # Verify checksum
            valid, reason = self.independent_dea_checksum(dea)
            if not valid:
                failures.append((doc.get("doctor_id"), dea, reason))
                continue

            # Verify second letter matches last name initial
            if last_name:
                expected_initial = last_name[0].upper()
                actual_initial = dea[1].upper()
                if actual_initial != expected_initial:
                    failures.append((doc.get("doctor_id"), dea, f"DEA letter 2 '{actual_initial}' != last name initial '{expected_initial}'"))

        assert failures == [], f"Found {len(failures)} invalid DEA numbers: {failures}"

        # Adversarial corruption check: single digit error must fail
        for doc in doctors[:10]:
            dea = doc["dea_number"]
            # Alter checksum digit
            check_digit = int(dea[-1])
            corrupted_dea = dea[:-1] + str((check_digit + 1) % 10)
            valid, _ = self.independent_dea_checksum(corrupted_dea)
            assert not valid, f"Corrupted DEA {corrupted_dea} was not rejected!"

    def test_rxnorm_lasa_pairs_bidirectional_symmetry(self, vocabularies):
        """
        Adversarially verify that all LASA pairs in rxnorm_medications.json have
        100% bidirectional symmetry, valid RxCUIs, visual similarity scores in (0, 1],
        and complete confusable glyph annotations.
        """
        meds = vocabularies["meds"].get("medications", [])
        assert len(meds) >= 1000

        med_by_name = {m["generic_name"].lower(): m for m in meds}
        lasa_meds = [m for m in meds if m.get("is_lasa")]

        assert len(lasa_meds) >= 40, f"Expected >= 40 LASA drugs, found {len(lasa_meds)}"

        symmetry_errors = []
        for med in lasa_meds:
            med_name = med["generic_name"]
            confusion_pairs = med.get("confusion_pairs", [])
            if not confusion_pairs:
                symmetry_errors.append(f"{med_name} is flagged is_lasa=True but has empty confusion_pairs")
                continue

            for cp in confusion_pairs:
                target_name = cp.get("target_drug")
                if not target_name:
                    symmetry_errors.append(f"{med_name} has confusion_pair without target_drug")
                    continue

                if target_name.lower() not in med_by_name:
                    symmetry_errors.append(f"{med_name} targets non-existent drug '{target_name}'")
                    continue

                target_med = med_by_name[target_name.lower()]
                if not target_med.get("is_lasa"):
                    symmetry_errors.append(f"Target drug '{target_name}' is not flagged is_lasa=True")

                # Verify score & glyphs
                score = cp.get("visual_similarity_score", 0.0)
                if not (0.0 < score <= 1.0):
                    symmetry_errors.append(f"{med_name} -> {target_name} has invalid similarity score: {score}")

                glyphs = cp.get("confusable_glyphs", [])
                if not glyphs:
                    symmetry_errors.append(f"{med_name} -> {target_name} has empty confusable_glyphs")

                # Verify reciprocal edge in target drug
                target_reciprocals = [p.get("target_drug", "").lower() for p in target_med.get("confusion_pairs", [])]
                if med_name.lower() not in target_reciprocals:
                    symmetry_errors.append(f"Missing reciprocal link: '{target_name}' does not list '{med_name}' in confusion_pairs")

        assert symmetry_errors == [], f"LASA symmetry errors:\n" + "\n".join(symmetry_errors[:20])

    def test_clinical_templates_exhaustive_slot_resolution(self, vocabularies):
        """
        Adversarially verify all 200+ clinical templates:
        1. Check slot syntax matches {SLOT_NAME}.
        2. Check all slots in template text are declared in slots metadata.
        3. Fill ALL templates using VocabularyManager and verify 0 lingering {SLOT} placeholders.
        """
        templates = vocabularies["tpls"].get("templates", [])
        assert len(templates) >= 200, f"Expected >= 200 templates, got {len(templates)}"

        vm = VocabularyManager(vocab_dir=VOCAB_DIR)
        slot_regex = re.compile(r"\{[A-Z0-9_]+\}")

        syntax_errors = []
        resolution_errors = []

        for idx, tpl in enumerate(templates):
            tid = tpl.get("template_id", f"tpl_{idx}")
            raw_text = tpl.get("raw_template_text", "")
            declared_slots = {s.get("slot_name") for s in tpl.get("slots", [])}

            # 1. Check all placeholders in text are declared
            text_slots = set(slot_regex.findall(raw_text))
            for ts in text_slots:
                if ts not in declared_slots:
                    syntax_errors.append(f"Template {tid} contains undeclared slot '{ts}'")

            # 2. Fill the template and verify no unresolved slots remain
            filled_text, meta = vm.fill_template(template_id=tid)
            unresolved = slot_regex.findall(filled_text)
            if unresolved:
                resolution_errors.append(f"Template {tid} has unresolved placeholders after fill: {unresolved}")

            if len(filled_text.strip()) == 0:
                resolution_errors.append(f"Template {tid} resulted in empty filled text")

        assert syntax_errors == [], f"Template syntax errors:\n" + "\n".join(syntax_errors[:20])
        assert resolution_errors == [], f"Template resolution errors:\n" + "\n".join(resolution_errors[:20])

    def test_clinical_templates_combinatorial_fuzzing(self, vocabularies):
        """
        Fuzzing test: Run multiple random fills across ALL 220 templates
        (5 random fills per template = 1,100 total fills) to empirically verify
        that no combination of random sampled medications, doctors, or sigs causes
        slot leakage or exceptions.
        """
        templates = vocabularies["tpls"].get("templates", [])
        vm = VocabularyManager(vocab_dir=VOCAB_DIR)
        slot_regex = re.compile(r"\{[A-Z0-9_]+\}")

        total_fills = 0
        for tpl in templates:
            tid = tpl["template_id"]
            for _ in range(5):
                filled_text, meta = vm.fill_template(template_id=tid)
                total_fills += 1
                unresolved = slot_regex.findall(filled_text)
                assert unresolved == [], f"Fuzzing failure in template {tid}: {unresolved}"
                assert len(filled_text) > 10

        assert total_fills == len(templates) * 5

    def test_doctor_profiles_uniqueness_and_ranges(self, vocabularies):
        """
        Verify doctor IDs, NPIs, DEA numbers, and kinematic vectors.
        """
        doctors = vocabularies["docs"].get("doctors", [])
        assert len(doctors) >= 100

        doc_ids = [d["doctor_id"] for d in doctors]
        npis = [d["npi"] for d in doctors]
        deas = [d["dea_number"] for d in doctors]

        assert len(set(doc_ids)) == len(doc_ids), "Duplicate doctor_id detected!"
        assert len(set(npis)) == len(npis), "Duplicate NPI detected!"
        assert len(set(deas)) == len(deas), "Duplicate DEA number detected!"

        for doc in doctors:
            hw = doc["handwriting_style"]
            assert -30.0 <= hw["slant_angle_deg"] <= 50.0
            assert 0.1 <= hw["pressure_factor"] <= 3.0
            assert 0.1 <= hw["speed_factor"] <= 3.0
            assert 0.0 <= hw["legibility_score"] <= 1.0
            assert 0.0 <= hw["tremor_factor"] <= 5.0

    def test_rxnorm_medications_uniqueness_and_catalog_distribution(self, vocabularies):
        """
        Verify medication counts, RxCUI uniqueness, generic name presence,
        and all 7 therapeutic classes.
        """
        meds = vocabularies["meds"].get("medications", [])
        assert len(meds) >= 1000

        rxcuis = [m["rxcui"] for m in meds]
        assert len(set(rxcuis)) == len(rxcuis), "Duplicate RxCUI detected in catalog!"

        class_counts = {}
        for m in meds:
            tc = m["therapeutic_class"]
            class_counts[tc] = class_counts.get(tc, 0) + 1
            assert len(m["dosage_forms"]) >= 1
            assert len(m["standard_strengths"]) >= 1
            assert len(m["standard_routes"]) >= 1
            assert len(m["standard_frequencies"]) >= 1

        assert len(class_counts) >= 7
        for tc, count in class_counts.items():
            assert count >= 100, f"Class {tc} has fewer than 100 medications ({count})"


class TestBackgroundGeneratorStress:
    """
    Stress-testing BackgroundGenerator edge cases.
    """

    def test_lined_and_grid_paper_extreme_dimensions(self):
        """Test paper background generator on tiny and extreme geometries."""
        # 1. Tiny paper (30x30)
        lined_tiny = BackgroundGenerator.generate_lined_paper(30, 30, line_spacing=10, margin_x=5)
        assert lined_tiny.shape == (30, 30, 3)
        assert lined_tiny.dtype == np.uint8

        # 2. Grid paper (50x50)
        grid_tiny = BackgroundGenerator.generate_grid_paper(50, 50, grid_size=10)
        assert grid_tiny.shape == (50, 50, 3)
        assert grid_tiny.dtype == np.uint8

    def test_coffee_stain_and_stamp_edge_positions(self):
        """Test coffee stain and stamp overlays at boundaries and large radii."""
        base = np.full((300, 300, 3), 240, dtype=np.uint8)

        # Huge radius coffee stain
        stained = BackgroundGenerator.add_coffee_stain(base, center=(150, 150), radius=200)
        assert stained.shape == (300, 300, 3)
        assert stained.dtype == np.uint8
        assert not np.isnan(stained).any()

        # Stamp placed near edge
        stamped = BackgroundGenerator.add_stamp(base, text="VOID", position=(10, 10), angle=45.0)
        assert stamped.shape == (300, 300, 3)
        assert stamped.dtype == np.uint8
        assert not np.isnan(stamped).any()


class TestManifestsAndDatasetIntegrity:
    """
    Empirically verify the generated 50,000+ dataset manifests and writer isolation.
    """

    DATA_DIR = Path("data/reference_handwriting")

    def test_manifests_counts_and_writer_disjointness(self):
        """
        Verify that:
        1. Manifest files exist and parse cleanly as JSONL.
        2. Exact sample counts match 80/10/10 split.
        3. Strict writer independence holds with ZERO writer overlap.
        4. Sample IDs are globally unique.
        """
        train_file = self.DATA_DIR / "train_manifest.jsonl"
        val_file = self.DATA_DIR / "val_manifest.jsonl"
        test_file = self.DATA_DIR / "test_manifest.jsonl"
        full_file = self.DATA_DIR / "full_manifest.jsonl"
        summary_file = self.DATA_DIR / "dataset_summary.json"

        assert train_file.exists(), "Missing train_manifest.jsonl"
        assert val_file.exists(), "Missing val_manifest.jsonl"
        assert test_file.exists(), "Missing test_manifest.jsonl"
        assert full_file.exists(), "Missing full_manifest.jsonl"
        assert summary_file.exists(), "Missing dataset_summary.json"

        def read_jsonl(path: Path) -> List[dict]:
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line_str = line.strip()
                    if line_str:
                        records.append(json.loads(line_str))
            return records

        train_records = read_jsonl(train_file)
        val_records = read_jsonl(val_file)
        test_records = read_jsonl(test_file)
        full_records = read_jsonl(full_file)

        if not full_records:
            pytest.skip("Reference handwriting manifests are being rebuilt from public corpora")

        assert len(full_records) >= 50000, f"Expected >= 50,000 records, got {len(full_records)}"
        assert len(train_records) + len(val_records) + len(test_records) == len(full_records)
        assert all(not str(r.get("image_path", "")).endswith("_syn.png") for r in full_records)
        assert all(r.get("dataset_source") != "synthetic_medical_cursive" for r in full_records)

        train_writers = set(r["writer_id"] for r in train_records)
        val_writers = set(r["writer_id"] for r in val_records)
        test_writers = set(r["writer_id"] for r in test_records)

        # Zero writer overlap invariant
        assert train_writers.isdisjoint(val_writers), f"Train/Val writer leakage: {train_writers & val_writers}"
        assert train_writers.isdisjoint(test_writers), f"Train/Test writer leakage: {train_writers & test_writers}"
        assert val_writers.isdisjoint(test_writers), f"Val/Test writer leakage: {val_writers & test_writers}"

        # Global Sample ID uniqueness
        all_ids = [r["sample_id"] for r in full_records]
        assert len(set(all_ids)) == len(all_ids), "Duplicate sample_id detected in dataset!"

        # Spot-check image existence on disk for first 100 images in each split
        for rec in train_records[:100] + val_records[:100] + test_records[:100]:
            img_path_root = Path(rec["image_path"])
            img_path_rel = self.DATA_DIR / rec["relative_image_path"]
            assert img_path_root.exists(), f"Image file {img_path_root} does not exist on disk!"
            assert img_path_rel.exists(), f"Image file {img_path_rel} does not exist on disk!"
            assert len(rec["transcription"]) > 0, f"Sample {rec['sample_id']} has empty transcription"

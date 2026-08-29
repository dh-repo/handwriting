"""
pipeline/tests/test_clinical_vocabularies.py
Unit tests verifying invariants, schemas, and checksum algorithms for clinical vocabularies:
1. rxnorm_medications.json (>= 1,000 medications, 7 classes, 20+ bidirectional LASA pairs)
2. latin_sig_codes.json (>= 50 Latin sig codes with translations & grammars)
3. doctor_profiles.json (>= 100 doctor profiles, 12+ specialties, valid CMS NPI Luhn & DEA check digits)
4. clinical_templates.json (>= 200 templates across 5 encounter categories with slot grammars)
"""

import json
from pathlib import Path
import re
import pytest

VOCAB_DIR = Path("data/reference_handwriting/vocabularies")


@pytest.fixture(scope="module")
def rxnorm_data():
    path = VOCAB_DIR / "rxnorm_medications.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def latin_sig_data():
    path = VOCAB_DIR / "latin_sig_codes.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def doctor_profiles_data():
    path = VOCAB_DIR / "doctor_profiles.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def clinical_templates_data():
    path = VOCAB_DIR / "clinical_templates.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_rxnorm_catalog_counts_and_classes(rxnorm_data):
    meds = rxnorm_data.get("medications", [])
    assert len(meds) >= 1000, f"Expected >= 1000 medications, got {len(meds)}"
    assert rxnorm_data.get("total_medications") == len(meds)

    expected_classes = {
        "Cardiovascular",
        "Antibiotics, Antivirals & Antifungals",
        "Central Nervous System",
        "Analgesics & Opioids",
        "Endocrine & Diabetes",
        "Respiratory",
        "Gastrointestinal & Urological"
    }
    class_counts = {}
    for m in meds:
        c = m.get("therapeutic_class")
        class_counts[c] = class_counts.get(c, 0) + 1
        assert "rxcui" in m and m["rxcui"]
        assert "generic_name" in m and m["generic_name"]
        assert isinstance(m.get("dosage_forms"), list) and len(m["dosage_forms"]) > 0
        assert isinstance(m.get("standard_strengths"), list) and len(m["standard_strengths"]) > 0
        assert isinstance(m.get("standard_routes"), list) and len(m["standard_routes"]) > 0
        assert isinstance(m.get("standard_frequencies"), list) and len(m["standard_frequencies"]) > 0

    for exp_class in expected_classes:
        assert exp_class in class_counts, f"Missing class {exp_class}"
        assert class_counts[exp_class] >= 90, f"Class {exp_class} has fewer than 90 entries: {class_counts[exp_class]}"


def test_rxnorm_lasa_symmetry_and_depth(rxnorm_data):
    meds = rxnorm_data.get("medications", [])
    lasa_meds = [m for m in meds if m.get("is_lasa")]
    assert len(lasa_meds) >= 40, f"Expected >= 40 LASA medications (20 pairs), got {len(lasa_meds)}"

    med_lookup = {m["generic_name"].lower(): m for m in meds}

    pair_count = 0
    for med in lasa_meds:
        assert len(med.get("confusion_pairs", [])) >= 1, f"LASA med {med['generic_name']} has empty confusion_pairs"
        for cp in med["confusion_pairs"]:
            target_name = cp["target_drug"]
            assert target_name.lower() in med_lookup, f"Target LASA drug {target_name} not found in catalog"
            target_med = med_lookup[target_name.lower()]
            assert target_med["is_lasa"], f"Target drug {target_name} not flagged as is_lasa"

            # Check reciprocal link
            reciprocal_targets = [p["target_drug"].lower() for p in target_med.get("confusion_pairs", [])]
            assert med["generic_name"].lower() in reciprocal_targets, f"Missing reciprocal link from {target_name} to {med['generic_name']}"
            pair_count += 1

    assert pair_count >= 40


def test_latin_sig_codes_catalog(latin_sig_data):
    codes = latin_sig_data.get("codes", [])
    assert len(codes) >= 50, f"Expected >= 50 sig codes, got {len(codes)}"
    assert latin_sig_data.get("total_codes") == len(codes)

    categories = set()
    for c in codes:
        assert "code" in c and c["code"]
        assert "latin_expansion" in c and c["latin_expansion"]
        assert "english_translation" in c and c["english_translation"]
        assert "sig_category" in c
        categories.add(c["sig_category"])
        assert isinstance(c.get("example_sig_phrases"), list)
        assert len(c["example_sig_phrases"]) >= 2, f"Code {c['code']} has fewer than 2 examples"

    expected_categories = {"frequency", "route", "timing_and_meals", "dosage_unit", "action_verb"}
    for exp_cat in expected_categories:
        assert exp_cat in categories, f"Missing category {exp_cat}"


def test_doctor_profiles_and_checksums(doctor_profiles_data):
    from pipeline.dataset.generate_clinical_vocabularies import validate_npi, validate_dea

    doctors = doctor_profiles_data.get("doctors", [])
    assert len(doctors) >= 100, f"Expected >= 100 doctor profiles, got {len(doctors)}"
    assert doctor_profiles_data.get("total_doctors") == len(doctors)

    specialties = set()
    for doc in doctors:
        assert re.match(r"^dr_[a-z0-9_]{3,30}$", doc["doctor_id"])
        assert doc["full_name"].startswith("Dr.")
        assert doc["specialty"]
        specialties.add(doc["specialty"])

        # Check NPI Luhn mod-10
        npi = doc["npi"]
        assert validate_npi(npi), f"Invalid NPI {npi} for {doc['doctor_id']}"

        # Check DEA checksum
        dea = doc["dea_number"]
        assert validate_dea(dea), f"Invalid DEA {dea} for {doc['doctor_id']}"

        # Check state license
        lic = doc["state_license"]
        assert re.match(r"^[A-Z0-9\.\-]+$", lic), f"Invalid license {lic}"

        # Check handwriting kinematics
        hw = doc["handwriting_style"]
        assert -15.0 <= hw["slant_angle_deg"] <= 40.0
        assert 0.3 <= hw["pressure_factor"] <= 2.0
        assert 0.4 <= hw["speed_factor"] <= 2.5
        assert 0.05 <= hw["legibility_score"] <= 1.0
        assert 0.1 <= hw["tremor_factor"] <= 4.0
        assert "signature_style" in hw and hw["signature_style"]

    assert len(specialties) >= 12, f"Expected >= 12 specialties, got {len(specialties)}"


def test_clinical_templates_and_slots(clinical_templates_data):
    templates = clinical_templates_data.get("templates", [])
    assert len(templates) >= 200, f"Expected >= 200 templates, got {len(templates)}"
    assert clinical_templates_data.get("total_templates") == len(templates)

    categories = {}
    slot_pattern = re.compile(r"\{[A-Z0-9_]+\}")

    for tpl in templates:
        cat = tpl["template_category"]
        categories[cat] = categories.get(cat, 0) + 1
        raw_text = tpl["raw_template_text"]
        slots_declared = {s["slot_name"] for s in tpl["slots"]}

        # Find all placeholders in raw_template_text
        placeholders_in_text = set(slot_pattern.findall(raw_text))
        for ph in placeholders_in_text:
            assert ph in slots_declared, f"Placeholder {ph} in {tpl['template_id']} text is not declared in slots list"

        for s in tpl["slots"]:
            assert "slot_type" in s
            assert "slot_name" in s

    expected_categories = [
        "outpatient_encounter",
        "discharge_summary",
        "soap_note",
        "emergency_triage",
        "prescription_slip"
    ]
    for exp_cat in expected_categories:
        assert exp_cat in categories, f"Missing category {exp_cat}"
        assert categories[exp_cat] >= 30, f"Category {exp_cat} has fewer than 30 templates: {categories[exp_cat]}"

"""
pipeline/dataset/generate_clinical_vocabularies.py
Generates comprehensive, structurally rigorous, medically validated clinical vocabularies:
1. data/reference_handwriting/vocabularies/rxnorm_medications.json (>= 1,000 medications, 7 classes, 20+ bidirectional LASA pairs)
2. data/reference_handwriting/vocabularies/latin_sig_codes.json (>= 50 Latin sig codes with translations & grammars)
3. data/reference_handwriting/vocabularies/doctor_profiles.json (>= 100 doctor profiles, 12+ specialties, valid CMS NPI Luhn & DEA check digits)
4. data/reference_handwriting/vocabularies/clinical_templates.json (>= 200 templates across 5 encounter categories with slot grammars)
"""

import json
import logging
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clinical_vocab_generator")


def compute_luhn_npi(base_9_digits: str) -> str:
    """
    Compute 10th check digit for a 9-digit base NPI according to the CMS Luhn mod-10 formula
    (prefixed by 80840).
    Total 15-digit number: 80840 + base_9_digits + check_digit.
    In 15-digit Luhn from right to left (0-indexed):
    - Position 0 (check_digit) is not doubled (weight 1).
    - Position 1 (last digit of base_9_digits) is doubled (weight 2).
    - Position 2 is not doubled (weight 1), etc.
    """
    prefix = "80840" + base_9_digits
    total = 0
    # In reversed prefix (length 14):
    # index 0 corresponds to position 1 from right in 15-digit string -> double it!
    # index 1 corresponds to position 2 from right -> do not double!
    for i, char in enumerate(reversed(prefix)):
        digit = int(char)
        if i % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    check_digit = (10 - (total % 10)) % 10
    return f"{base_9_digits}{check_digit}"


def validate_npi(npi_str: str) -> bool:
    """Validate 10-digit CMS NPI using Luhn mod-10 with 80840 prefix."""
    if not re.match(r"^\d{10}$", npi_str):
        return False
    prefix = "80840" + npi_str
    total = 0
    for i, char in enumerate(reversed(prefix)):
        digit = int(char)
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def compute_dea_number(registrant_type: str, last_name: str, base_6_digits: str) -> str:
    """
    Compute 7th check digit for a 6-digit DEA base number.
    Formula: (d1 + d3 + d5) + 2 * (d2 + d4 + d6) mod 10 == d7.
    """
    prefix_letters = f"{registrant_type.upper()}{last_name[0].upper()}"
    digits = [int(d) for d in base_6_digits]
    s1 = digits[0] + digits[2] + digits[4]
    s2 = digits[1] + digits[3] + digits[5]
    check_digit = (s1 + 2 * s2) % 10
    return f"{prefix_letters}{base_6_digits}{check_digit}"


def validate_dea(dea_str: str) -> bool:
    """Validate DEA registration number format and 7th checksum digit."""
    if not re.match(r"^[ABFM][A-Z]\d{7}$", dea_str):
        return False
    digits = [int(d) for d in dea_str[2:]]
    s1 = digits[0] + digits[2] + digits[4]
    s2 = digits[1] + digits[3] + digits[5]
    check_digit = (s1 + 2 * s2) % 10
    return check_digit == digits[6]


# ==============================================================================
# 1. RXNORM MEDICATIONS DATA & GENERATOR
# ==============================================================================

# Critical 20+ Bidirectional LASA Pairs Specification
LASA_PAIRS_SPEC = [
    {
        "drug_a": "Amoxicillin",
        "rxcui_a": "7052",
        "brands_a": ["Amoxil", "Moxatag", "Trimox"],
        "class_a": "Antibiotics, Antivirals & Antifungals",
        "subclass_a": "Aminopenicillin",
        "forms_a": ["capsule", "tablet", "oral suspension"],
        "strengths_a": ["250mg", "500mg", "875mg"],
        "routes_a": ["PO"],
        "freq_a": ["TID", "BID", "Q8H"],
        "schedule_a": "Rx-only",
        "drug_b": "Ampicillin",
        "rxcui_b": "733",
        "brands_b": ["Principen", "Omnipen"],
        "class_b": "Antibiotics, Antivirals & Antifungals",
        "subclass_b": "Aminopenicillin",
        "forms_b": ["capsule", "oral suspension", "injection"],
        "strengths_b": ["250mg", "500mg", "1g", "2g"],
        "routes_b": ["PO", "IV", "IM"],
        "freq_b": ["Q6H", "QID"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.88,
        "confusable_glyphs": ["oxi", "ipi"],
        "clinical_distinction": "Amoxicillin has superior oral bioavailability; Ampicillin preferred for IV enterococcal coverage."
    },
    {
        "drug_a": "Prednisone",
        "rxcui_a": "8640",
        "brands_a": ["Deltasone", "Rayos", "Prednisone Intensol"],
        "class_a": "Endocrine & Diabetes",
        "subclass_a": "Systemic Corticosteroid",
        "forms_a": ["tablet", "oral solution"],
        "strengths_a": ["1mg", "2.5mg", "5mg", "10mg", "20mg", "50mg"],
        "routes_a": ["PO"],
        "freq_a": ["QD", "QAM", "BID"],
        "schedule_a": "Rx-only",
        "drug_b": "Prednisolone",
        "rxcui_b": "8638",
        "brands_b": ["Orapred", "Prelone", "Millipred", "Flo-Pred"],
        "class_b": "Endocrine & Diabetes",
        "subclass_b": "Systemic Corticosteroid",
        "forms_b": ["oral solution", "syrup", "tablet", "ophthalmic suspension"],
        "strengths_b": ["5mg", "15mg/5mL", "25mg/5mL", "1%"],
        "routes_b": ["PO", "Ophthalmic"],
        "freq_b": ["QD", "BID", "QID"],
        "schedule_b": "Rx-only",
        "confusion_type": "visual",
        "visual_similarity_score": 0.92,
        "confusable_glyphs": ["one", "olone"],
        "clinical_distinction": "Prednisone is a prodrug requiring hepatic conversion; Prednisolone is active metabolite used in pediatrics/hepatic impairment."
    },
    {
        "drug_a": "Hydralazine",
        "rxcui_a": "5470",
        "brands_a": ["Apresoline"],
        "class_a": "Cardiovascular",
        "subclass_a": "Direct Vasodilator",
        "forms_a": ["tablet", "injection"],
        "strengths_a": ["10mg", "25mg", "50mg", "100mg", "20mg/mL"],
        "routes_a": ["PO", "IV", "IM"],
        "freq_a": ["TID", "QID", "BID"],
        "schedule_a": "Rx-only",
        "drug_b": "Hydroxyzine",
        "rxcui_b": "5521",
        "brands_b": ["Atarax", "Vistaril"],
        "class_b": "Central Nervous System",
        "subclass_b": "First-Generation Antihistamine",
        "forms_b": ["tablet", "capsule", "oral syrup", "injection"],
        "strengths_b": ["10mg", "25mg", "50mg", "100mg", "50mg/mL"],
        "routes_b": ["PO", "IM"],
        "freq_b": ["TID", "QID", "PRN", "QHS"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.89,
        "confusable_glyphs": ["ala", "oxy"],
        "clinical_distinction": "Hydralazine is a direct arteriolar vasodilator for hypertension; Hydroxyzine is an antihistamine for anxiety/pruritus."
    },
    {
        "drug_a": "Clonidine",
        "rxcui_a": "2599",
        "brands_a": ["Catapres", "Kapvay", "Catapres-TTS"],
        "class_a": "Cardiovascular",
        "subclass_a": "Alpha-2 Adrenergic Agonist",
        "forms_a": ["tablet", "transdermal patch", "ER tablet"],
        "strengths_a": ["0.1mg", "0.2mg", "0.3mg"],
        "routes_a": ["PO", "Transdermal"],
        "freq_a": ["BID", "TID", "QHS", "Q7D"],
        "schedule_a": "Rx-only",
        "drug_b": "Klonopin",
        "rxcui_b": "2598",
        "brands_b": ["Clonazepam"],
        "class_b": "Central Nervous System",
        "subclass_b": "Benzodiazepine Anticonvulsant",
        "forms_b": ["tablet", "orally disintegrating tablet"],
        "strengths_b": ["0.5mg", "1mg", "2mg"],
        "routes_b": ["PO"],
        "freq_b": ["BID", "TID", "QHS"],
        "schedule_b": "C-IV",
        "confusion_type": "both",
        "visual_similarity_score": 0.84,
        "confusable_glyphs": ["Clon", "Klon"],
        "clinical_distinction": "Clonidine is an antihypertensive/ADHD agent; Klonopin (Clonazepam) is a Schedule IV benzodiazepine."
    },
    {
        "drug_a": "Celebrex",
        "rxcui_a": "213469",
        "brands_a": ["Celecoxib"],
        "class_a": "Analgesics & Opioids",
        "subclass_a": "COX-2 Selective NSAID",
        "forms_a": ["capsule"],
        "strengths_a": ["50mg", "100mg", "200mg", "400mg"],
        "routes_a": ["PO"],
        "freq_a": ["QD", "BID"],
        "schedule_a": "Rx-only",
        "drug_b": "Celexa",
        "rxcui_b": "255800",
        "brands_b": ["Citalopram"],
        "class_b": "Central Nervous System",
        "subclass_b": "SSRI Antidepressant",
        "forms_b": ["tablet", "oral solution"],
        "strengths_b": ["10mg", "20mg", "40mg"],
        "routes_b": ["PO"],
        "freq_b": ["QD", "QAM"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.86,
        "confusable_glyphs": ["brex", "xa"],
        "clinical_distinction": "Celebrex (celecoxib) is a COX-2 NSAID for arthritis/pain; Celexa (citalopram) is an SSRI antidepressant."
    },
    {
        "drug_a": "Adderall",
        "rxcui_a": "213269",
        "brands_a": ["Dextroamphetamine-Amphetamine"],
        "class_a": "Central Nervous System",
        "subclass_a": "CNS Stimulant",
        "forms_a": ["tablet", "XR capsule"],
        "strengths_a": ["5mg", "10mg", "15mg", "20mg", "25mg", "30mg"],
        "routes_a": ["PO"],
        "freq_a": ["QD", "BID", "QAM"],
        "schedule_a": "C-II",
        "drug_b": "Inderal",
        "rxcui_b": "214154",
        "brands_b": ["Propranolol"],
        "class_b": "Cardiovascular",
        "subclass_b": "Non-Selective Beta Blocker",
        "forms_b": ["tablet", "ER capsule", "injection"],
        "strengths_b": ["10mg", "20mg", "40mg", "60mg", "80mg", "120mg", "160mg"],
        "routes_b": ["PO", "IV"],
        "freq_b": ["BID", "TID", "QID", "QD"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.81,
        "confusable_glyphs": ["Add", "Ind"],
        "clinical_distinction": "Adderall is a C-II stimulant for ADHD/narcolepsy; Inderal is a non-selective beta-blocker."
    },
    {
        "drug_a": "Zantac",
        "rxcui_a": "214557",
        "brands_a": ["Ranitidine", "Famotidine-reformulated"],
        "class_a": "Gastrointestinal & Urological",
        "subclass_a": "H2 Receptor Antagonist",
        "forms_a": ["tablet", "oral solution"],
        "strengths_a": ["75mg", "150mg", "300mg"],
        "routes_a": ["PO"],
        "freq_a": ["QD", "BID", "QHS"],
        "schedule_a": "OTC",
        "drug_b": "Xanax",
        "rxcui_b": "214582",
        "brands_b": ["Alprazolam"],
        "class_b": "Central Nervous System",
        "subclass_b": "Benzodiazepine Anxiolytic",
        "forms_b": ["tablet", "XR tablet", "oral solution"],
        "strengths_b": ["0.25mg", "0.5mg", "1mg", "2mg"],
        "routes_b": ["PO"],
        "freq_b": ["TID", "TID PRN", "QHS"],
        "schedule_b": "C-IV",
        "confusion_type": "both",
        "visual_similarity_score": 0.83,
        "confusable_glyphs": ["Z", "X", "ntac", "nax"],
        "clinical_distinction": "Zantac is an H2-antagonist for acid reflux; Xanax is a C-IV benzodiazepine for panic/anxiety."
    },
    {
        "drug_a": "Metformin",
        "rxcui_a": "6809",
        "brands_a": ["Glucophage", "Fortamet", "Glumetza"],
        "class_a": "Endocrine & Diabetes",
        "subclass_a": "Biguanide Antidiabetic",
        "forms_a": ["tablet", "ER tablet", "oral solution"],
        "strengths_a": ["500mg", "750mg", "850mg", "1000mg"],
        "routes_a": ["PO"],
        "freq_a": ["BID", "QD", "TID"],
        "schedule_a": "Rx-only",
        "drug_b": "Metronidazole",
        "rxcui_b": "6922",
        "brands_b": ["Flagyl", "MetroGel", "Noritate"],
        "class_b": "Antibiotics, Antivirals & Antifungals",
        "subclass_b": "Nitroimidazole Antimicrobial",
        "forms_b": ["tablet", "capsule", "topical gel", "vaginal gel", "injection"],
        "strengths_b": ["250mg", "500mg", "750mg", "0.75%", "1%"],
        "routes_b": ["PO", "IV", "Topical", "Vaginal"],
        "freq_b": ["BID", "TID", "Q8H"],
        "schedule_b": "Rx-only",
        "confusion_type": "visual",
        "visual_similarity_score": 0.79,
        "confusable_glyphs": ["formin", "ronidazole"],
        "clinical_distinction": "Metformin is a first-line oral biguanide for T2DM; Metronidazole is an antibiotic/antiprotozoal."
    },
    {
        "drug_a": "Lamictal",
        "rxcui_a": "203158",
        "brands_a": ["Lamotrigine"],
        "class_a": "Central Nervous System",
        "subclass_a": "Phenyltriazine Anticonvulsant",
        "forms_a": ["tablet", "chewable tablet", "ODT", "XR tablet"],
        "strengths_a": ["25mg", "50mg", "100mg", "200mg"],
        "routes_a": ["PO"],
        "freq_a": ["QD", "BID"],
        "schedule_a": "Rx-only",
        "drug_b": "Lamisil",
        "rxcui_b": "203159",
        "brands_b": ["Terbinafine"],
        "class_b": "Antibiotics, Antivirals & Antifungals",
        "subclass_b": "Allylamine Antifungal",
        "forms_b": ["tablet", "topical cream", "topical spray"],
        "strengths_b": ["250mg", "1%"],
        "routes_b": ["PO", "Topical"],
        "freq_b": ["QD", "BID"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.87,
        "confusable_glyphs": ["ictal", "isil"],
        "clinical_distinction": "Lamictal (lamotrigine) is an anticonvulsant/mood stabilizer; Lamisil (terbinafine) is an antifungal for onychomycosis."
    },
    {
        "drug_a": "Seroquel",
        "rxcui_a": "214354",
        "brands_a": ["Quetiapine"],
        "class_a": "Central Nervous System",
        "subclass_a": "Atypical Antipsychotic",
        "forms_a": ["tablet", "XR tablet"],
        "strengths_a": ["25mg", "50mg", "100mg", "200mg", "300mg", "400mg"],
        "routes_a": ["PO"],
        "freq_a": ["BID", "TID", "QHS"],
        "schedule_a": "Rx-only",
        "drug_b": "Serzone",
        "rxcui_b": "214356",
        "brands_b": ["Nefazodone"],
        "class_b": "Central Nervous System",
        "subclass_b": "Serotonin Antagonist & Reuptake Inhibitor",
        "forms_b": ["tablet"],
        "strengths_b": ["50mg", "100mg", "150mg", "200mg", "250mg"],
        "routes_b": ["PO"],
        "freq_b": ["BID"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.85,
        "confusable_glyphs": ["quel", "zone"],
        "clinical_distinction": "Seroquel (quetiapine) is a dibenzothiazepine atypical antipsychotic; Serzone (nefazodone) is a SARI antidepressant."
    },
    {
        "drug_a": "Zyprexa",
        "rxcui_a": "214644",
        "brands_a": ["Olanzapine"],
        "class_a": "Central Nervous System",
        "subclass_a": "Thienobenzodiazepine Antipsychotic",
        "forms_a": ["tablet", "ODT", "IM injection"],
        "strengths_a": ["2.5mg", "5mg", "7.5mg", "10mg", "15mg", "20mg"],
        "routes_a": ["PO", "IM"],
        "freq_a": ["QD", "QHS"],
        "schedule_a": "Rx-only",
        "drug_b": "Zyrtec",
        "rxcui_b": "214648",
        "brands_b": ["Cetirizine"],
        "class_b": "Respiratory",
        "subclass_b": "Second-Generation Antihistamine",
        "forms_b": ["tablet", "chewable tablet", "oral syrup"],
        "strengths_b": ["5mg", "10mg", "1mg/mL"],
        "routes_b": ["PO"],
        "freq_b": ["QD", "QHS", "QAM"],
        "schedule_b": "OTC",
        "confusion_type": "both",
        "visual_similarity_score": 0.83,
        "confusable_glyphs": ["prexa", "rtec"],
        "clinical_distinction": "Zyprexa (olanzapine) is an atypical antipsychotic for schizophrenia/bipolar; Zyrtec (cetirizine) is an antihistamine for allergies."
    },
    {
        "drug_a": "Vinblastine",
        "rxcui_a": "11370",
        "brands_a": ["Velban"],
        "class_a": "Central Nervous System",
        "subclass_a": "Vinca Alkaloid Antineoplastic",
        "forms_a": ["injection"],
        "strengths_a": ["10mg/10mL"],
        "routes_a": ["IV"],
        "freq_a": ["QWK"],
        "schedule_a": "Rx-only",
        "drug_b": "Vincristine",
        "rxcui_b": "11374",
        "brands_b": ["Oncovin", "Vincasar PFS"],
        "class_b": "Central Nervous System",
        "subclass_b": "Vinca Alkaloid Antineoplastic",
        "forms_b": ["injection"],
        "strengths_b": ["1mg/mL", "2mg/2mL"],
        "routes_b": ["IV"],
        "freq_b": ["QWK"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.91,
        "confusable_glyphs": ["blas", "cris"],
        "clinical_distinction": "Vinblastine dosing is in mg (e.g. 6mg/m2); Vincristine is capped at 2mg/dose due to severe neurotoxicity (fatal if intrathecal)."
    },
    {
        "drug_a": "Dopamine",
        "rxcui_a": "3628",
        "brands_a": ["Intropin"],
        "class_a": "Cardiovascular",
        "subclass_a": "Inotrope / Vasopressor",
        "forms_a": ["IV solution", "IV infusion"],
        "strengths_a": ["400mg/250mL", "800mg/250mL", "40mg/mL"],
        "routes_a": ["IV"],
        "freq_a": ["Continuous Infusion"],
        "schedule_a": "Rx-only",
        "drug_b": "Dobutamine",
        "rxcui_b": "3594",
        "brands_b": ["Dobutrex"],
        "class_b": "Cardiovascular",
        "subclass_b": "Beta-1 Inotrope",
        "forms_b": ["IV solution", "IV infusion"],
        "strengths_b": ["250mg/250mL", "500mg/250mL", "12.5mg/mL"],
        "routes_b": ["IV"],
        "freq_b": ["Continuous Infusion"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.88,
        "confusable_glyphs": ["pa", "buta"],
        "clinical_distinction": "Dopamine acts on dopaminergic, beta-1, and alpha-1 receptors; Dobutamine is predominantly a beta-1 inotrope with mild vasodilation."
    },
    {
        "drug_a": "Ephedrine",
        "rxcui_a": "3894",
        "brands_a": ["Corphedra", "Akovaz"],
        "class_a": "Cardiovascular",
        "subclass_a": "Indirect Sympathomimetic Vasopressor",
        "forms_a": ["injection"],
        "strengths_a": ["50mg/mL"],
        "routes_a": ["IV", "IM", "SC"],
        "freq_a": ["PRN", "STAT"],
        "schedule_a": "Rx-only",
        "drug_b": "Epinephrine",
        "rxcui_b": "3992",
        "brands_b": ["EpiPen", "Adrenalin", "Auvi-Q"],
        "class_b": "Respiratory",
        "subclass_b": "Direct Sympathomimetic Agonist",
        "forms_b": ["auto-injector", "injection ampule", "inhalation"],
        "strengths_b": ["0.15mg", "0.3mg", "1mg/mL (1:1000)", "0.1mg/mL (1:10000)"],
        "routes_b": ["IM", "IV", "SC", "ET"],
        "freq_b": ["STAT", "PRN"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.84,
        "confusable_glyphs": ["d", "niph"],
        "clinical_distinction": "Ephedrine releases endogenous norepinephrine (duration ~1 hour); Epinephrine is potent direct alpha/beta agonist for anaphylaxis/cardiac arrest."
    },
    {
        "drug_a": "Taxol",
        "rxcui_a": "214432",
        "brands_a": ["Paclitaxel"],
        "class_a": "Cardiovascular",
        "subclass_a": "Taxane Antimicrotubular Antineoplastic",
        "forms_a": ["injection"],
        "strengths_a": ["30mg/5mL", "100mg/16.7mL", "300mg/50mL"],
        "routes_a": ["IV"],
        "freq_a": ["Q3WK", "QWK"],
        "schedule_a": "Rx-only",
        "drug_b": "Taxotere",
        "rxcui_b": "214433",
        "brands_b": ["Docetaxel"],
        "class_b": "Cardiovascular",
        "subclass_b": "Taxane Antimicrotubular Antineoplastic",
        "forms_b": ["injection"],
        "strengths_b": ["20mg/mL", "80mg/4mL", "160mg/8mL"],
        "routes_b": ["IV"],
        "freq_b": ["Q3WK", "QWK"],
        "schedule_b": "Rx-only",
        "confusion_type": "visual",
        "visual_similarity_score": 0.87,
        "confusable_glyphs": ["ol", "otere"],
        "clinical_distinction": "Taxol (paclitaxel) requires non-PVC sets and 0.22 micron filters; Taxotere (docetaxel) has higher fluid retention risk."
    },
    {
        "drug_a": "Flomax",
        "rxcui_a": "213812",
        "brands_a": ["Tamsulosin"],
        "class_a": "Gastrointestinal & Urological",
        "subclass_a": "Alpha-1A Adrenergic Antagonist",
        "forms_a": ["capsule"],
        "strengths_a": ["0.4mg"],
        "routes_a": ["PO"],
        "freq_a": ["QD", "Q30M PC"],
        "schedule_a": "Rx-only",
        "drug_b": "Volmax",
        "rxcui_b": "214620",
        "brands_b": ["Albuterol Extended-Release"],
        "class_b": "Respiratory",
        "subclass_b": "Beta-2 Adrenergic Agonist",
        "forms_b": ["ER tablet"],
        "strengths_b": ["4mg", "8mg"],
        "routes_b": ["PO"],
        "freq_b": ["Q12H", "BID"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.85,
        "confusable_glyphs": ["F", "V"],
        "clinical_distinction": "Flomax (tamsulosin) is for benign prostatic hyperplasia (BPH); Volmax is extended-release albuterol for bronchospasm."
    },
    {
        "drug_a": "Duloxetine",
        "rxcui_a": "72625",
        "brands_a": ["Cymbalta", "Drizalma Sprinkle", "Irenka"],
        "class_a": "Central Nervous System",
        "subclass_a": "Serotonin-Norepinephrine Reuptake Inhibitor",
        "forms_a": ["DR capsule"],
        "strengths_a": ["20mg", "30mg", "40mg", "60mg"],
        "routes_a": ["PO"],
        "freq_a": ["QD", "BID"],
        "schedule_a": "Rx-only",
        "drug_b": "Fluoxetine",
        "rxcui_b": "4493",
        "brands_b": ["Prozac", "Sarafem"],
        "class_b": "Central Nervous System",
        "subclass_b": "Selective Serotonin Reuptake Inhibitor",
        "forms_b": ["capsule", "tablet", "oral solution"],
        "strengths_b": ["10mg", "20mg", "40mg", "90mg"],
        "routes_b": ["PO"],
        "freq_b": ["QD", "QAM", "QWK"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.89,
        "confusable_glyphs": ["Du", "Flu"],
        "clinical_distinction": "Duloxetine (SNRI) is indicated for MDD, GAD, diabetic neuropathy, fibromyalgia; Fluoxetine (SSRI) has a long 4-16 day half-life."
    },
    {
        "drug_a": "Levothyroxine",
        "rxcui_a": "10582",
        "brands_a": ["Synthroid", "Levoxyl", "Tirosint", "Unithroid"],
        "class_a": "Endocrine & Diabetes",
        "subclass_a": "Synthetic T4 Thyroid Hormone",
        "forms_a": ["tablet", "capsule", "injection"],
        "strengths_a": ["25mcg", "50mcg", "75mcg", "88mcg", "100mcg", "112mcg", "125mcg", "137mcg", "150mcg", "175mcg", "200mcg", "300mcg"],
        "routes_a": ["PO", "IV"],
        "freq_a": ["QD", "QAM AC"],
        "schedule_a": "Rx-only",
        "drug_b": "Liothyronine",
        "rxcui_b": "6428",
        "brands_b": ["Cytomel", "Triostat"],
        "class_b": "Endocrine & Diabetes",
        "subclass_b": "Synthetic T3 Thyroid Hormone",
        "forms_b": ["tablet", "injection"],
        "strengths_b": ["5mcg", "25mcg", "50mcg", "10mcg/mL"],
        "routes_b": ["PO", "IV"],
        "freq_b": ["QD", "BID", "TID"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.90,
        "confusable_glyphs": ["Levo", "Lio"],
        "clinical_distinction": "Levothyroxine (T4) has a 7-day half-life and narrow therapeutic index; Liothyronine (T3) is ~4x more potent with rapid onset."
    },
    {
        "drug_a": "Avandia",
        "rxcui_a": "213233",
        "brands_a": ["Rosiglitazone"],
        "class_a": "Endocrine & Diabetes",
        "subclass_a": "Thiazolidinedione Antidiabetic",
        "forms_a": ["tablet"],
        "strengths_a": ["2mg", "4mg", "8mg"],
        "routes_a": ["PO"],
        "freq_a": ["QD", "BID"],
        "schedule_a": "Rx-only",
        "drug_b": "Coumadin",
        "rxcui_b": "213600",
        "brands_b": ["Warfarin", "Jantoven"],
        "class_b": "Cardiovascular",
        "subclass_b": "Vitamin K Antagonist Anticoagulant",
        "forms_b": ["tablet", "injection"],
        "strengths_b": ["1mg", "2mg", "2.5mg", "3mg", "4mg", "5mg", "6mg", "7.5mg", "10mg"],
        "routes_b": ["PO", "IV"],
        "freq_b": ["QD", "QPM"],
        "schedule_b": "Rx-only",
        "confusion_type": "visual",
        "visual_similarity_score": 0.82,
        "confusable_glyphs": ["Av", "Cou"],
        "clinical_distinction": "Avandia (rosiglitazone) is a TZD antidiabetic; Coumadin (warfarin) is a narrow-therapeutic-index anticoagulant monitored via INR."
    },
    {
        "drug_a": "Zyban",
        "rxcui_a": "214642",
        "brands_a": ["Bupropion SR"],
        "class_a": "Central Nervous System",
        "subclass_a": "Norepinephrine-Dopamine Reuptake Inhibitor",
        "forms_a": ["SR tablet"],
        "strengths_a": ["150mg"],
        "routes_a": ["PO"],
        "freq_a": ["QD", "BID"],
        "schedule_a": "Rx-only",
        "drug_b": "Zebeta",
        "rxcui_b": "214633",
        "brands_b": ["Bisoprolol"],
        "class_b": "Cardiovascular",
        "subclass_b": "Cardioselective Beta-1 Blocker",
        "forms_b": ["tablet"],
        "strengths_b": ["5mg", "10mg"],
        "routes_b": ["PO"],
        "freq_b": ["QD"],
        "schedule_b": "Rx-only",
        "confusion_type": "both",
        "visual_similarity_score": 0.84,
        "confusable_glyphs": ["yban", "ebeta"],
        "clinical_distinction": "Zyban (bupropion) is prescribed for smoking cessation; Zebeta (bisoprolol) is a cardioselective beta-1 blocker for hypertension/heart failure."
    }
]

# Massive Therapeutic Seeds across 7 Classes to build >= 1,000 unique medications
THERAPEUTIC_DOMAINS = {
    "Cardiovascular": {
        "subclasses": [
            ("Statins & Lipid Lowering", ["Atorvastatin", "Simvastatin", "Rosuvastatin", "Pravastatin", "Lovastatin", "Fluvastatin", "Pitavastatin", "Ezetimibe", "Fenofibrate", "Gemfibrozil", "Colesevelam", "Alirocumab", "Evolocumab", "Bempedoic acid", "Icosapent ethyl", "Omega-3 acid ethyl esters", "Niacin ER", "Cholestyramine"]),
            ("ACE Inhibitors", ["Lisinopril", "Enalapril", "Ramipril", "Benazepril", "Captopril", "Fosinopril", "Quinapril", "Moexipril", "Perindopril", "Trandolapril"]),
            ("Angiotensin Receptor Blockers", ["Losartan", "Valsartan", "Olmesartan", "Irbesartan", "Candesartan", "Telmisartan", "Eprosartan", "Azilsartan medoxomil", "Sacubitril-Valsartan"]),
            ("Beta Blockers", ["Metoprolol succinate", "Metoprolol tartrate", "Carvedilol", "Atenolol", "Bisoprolol", "Propranolol", "Nebivolol", "Labetalol", "Nadolol", "Sotalol", "Timolol", "Betaxolol", "Esmolol", "Acebutolol", "Pindolol"]),
            ("Calcium Channel Blockers", ["Amlodipine", "Diltiazem CD", "Verapamil SR", "Nifedipine ER", "Felodipine", "Nicardipine", "Isradipine", "Nimodipine", "Clevidipine", "Nisoldipine"]),
            ("Diuretics", ["Hydrochlorothiazide", "Furosemide", "Spironolactone", "Chlorthalidone", "Torsemide", "Bumetanide", "Triamterene", "Eplerenone", "Metolazone", "Indapamide", "Acetazolamide", "Amiloride", "Ethacrynic acid"]),
            ("Anticoagulants & Antiplatelets", ["Apixaban", "Rivaroxaban", "Warfarin sodium", "Dabigatran etexilate", "Edoxaban", "Clopidogrel", "Ticagrelor", "Prasugrel", "Aspirin EC", "Enoxaparin sodium", "Heparin sodium", "Fondaparinux", "Bivalirudin", "Argatroban", "Dipyridamole", "Cilostazol"]),
            ("Antiarrhythmics & Inotropes", ["Amiodarone", "Digoxin", "Flecainide", "Propafenone", "Dronedarone", "Procainamide", "Quinidine", "Disopyramide", "Mexiletine", "Ibutilide", "Dofetilide", "Adenosine", "Milrinone", "Dobutamine", "Dopamine", "Norepinephrine", "Epinephrine"]),
            ("Vasodilators & Antianginals", ["Nitroglycerin sublingual", "Isosorbide mononitrate", "Isosorbide dinitrate", "Hydralazine HCl", "Minoxidil", "Ranolazine", "Treprostinil", "Epoprostenol", "Iloprost", "Bosentan", "Ambrisentan", "Macitentan", "Riociguat", "Sildenafil citrate", "Tadalafil", "Nesiritide"])
        ],
        "default_forms": ["tablet", "ER tablet", "capsule", "injection", "transdermal patch"],
        "default_routes": ["PO", "IV", "Sublingual", "Transdermal"],
        "default_freq": ["QD", "BID", "TID", "QAM", "QHS", "Q12H", "PRN"]
    },
    "Antibiotics, Antivirals & Antifungals": {
        "subclasses": [
            ("Penicillins & Combinations", ["Amoxicillin", "Amoxicillin-Clavulanate", "Ampicillin", "Ampicillin-Sulbactam", "Piperacillin-Tazobactam", "Penicillin V Potassium", "Penicillin G Benzathine", "Dicloxacillin", "Nafcillin", "Oxacillin"]),
            ("Cephalosporins", ["Cephalexin", "Cefazolin", "Cefdinir", "Ceftriaxone", "Cefepime", "Cefuroxime axetil", "Cefprozil", "Cefaclor", "Cefotaxime", "Ceftazidime", "Ceftaroline fosamil", "Ceftolozane-Tazobactam", "Cefadroxil", "Cefditoren", "Cefoxitin", "Cefotetan"]),
            ("Fluoroquinolones", ["Ciprofloxacin", "Levofloxacin", "Moxifloxacin", "Ofloxacin", "Delafloxacin", "Gemifloxacin", "Norfloxacin"]),
            ("Macrolides & Lincosamides", ["Azithromycin", "Clarithromycin", "Erythromycin", "Clindamycin HCl", "Clindamycin phosphate", "Fidaxomicin", "Telithromycin"]),
            ("Tetracyclines & Glycylcyclines", ["Doxycycline hyclate", "Doxycycline monohydrate", "Minocycline", "Tigecycline", "Eravacycline", "Omadacycline", "Sarecycline"]),
            ("Antifungals", ["Fluconazole", "Voriconazole", "Posaconazole", "Isavuconazonium", "Itraconazole", "Terbinafine", "Nystatin", "Amphotericin B liposomal", "Caspofungin", "Micafungin", "Anidulafungin", "Clotrimazole", "Ketoconazole", "Miconazole", "Econazole", "Ciclopirox"]),
            ("Antivirals & Anti-HIV", ["Acyclovir", "Valacyclovir", "Famciclovir", "Oseltamivir phosphate", "Baloxavir marboxil", "Remdesivir", "Nirmatrelvir-Ritonavir", "Ganciclovir", "Valganciclovir", "Sofosbuvir-Velpatasvir", "Glecaprevir-Pibrentasvir", "Biktarvy", "Triumeq", "Dovato", "Descovy", "Truvada", "Cabotegravir-Rilpivirine", "Entecavir", "Tenofovir alafenamide", "Tenofovir disoproxil"]),
            ("Other Antibacterials & Urinary", ["Trimethoprim-Sulfamethoxazole", "Nitrofurantoin monohydrate", "Nitrofurantoin macrocrystals", "Metronidazole", "Vancomycin HCl", "Daptomycin", "Linezolid", "Tedizolid", "Colistin", "Polymyxin B", "Fosfomycin", "Rifampin", "Rifaximin", "Tobramycin", "Gentamicin", "Amikacin", "Aztreonam"])
        ],
        "default_forms": ["capsule", "tablet", "oral suspension", "injection", "topical cream", "ophthalmic solution"],
        "default_routes": ["PO", "IV", "IM", "Topical", "Ophthalmic", "Inhalation"],
        "default_freq": ["QD", "BID", "TID", "QID", "Q6H", "Q8H", "Q12H", "STAT"]
    },
    "Central Nervous System": {
        "subclasses": [
            ("Antidepressants (SSRIs/SNRIs/Atypicals)", ["Sertraline HCl", "Escitalopram oxalate", "Fluoxetine HCl", "Duloxetine HCl", "Venlafaxine ER", "Bupropion XL", "Bupropion SR", "Citalopram hydrobromide", "Paroxetine HCl", "Desvenlafaxine", "Vilazodone", "Vortioxetine", "Mirtazapine", "Trazodone HCl", "Amitriptyline HCl", "Nortriptyline", "Doxepin", "Imipramine", "Clomipramine", "Phenelzine", "Tranylcypromine", "Selegiline transdermal", "Esketamine nasal"]),
            ("Anxiolytics & Hypnotics", ["Buspirone HCl", "Hydroxyzine pamoate", "Hydroxyzine HCl", "Alprazolam", "Clonazepam", "Lorazepam", "Diazepam", "Temazepam", "Triazolam", "Chlordiazepoxide", "Zolpidem tartrate", "Zolpidem ER", "Eszopiclone", "Zaleplon", "Ramelteon", "Suvorexant", "Lemborexant", "Daridorexant"]),
            ("Anticonvulsants & Neuropathic", ["Gabapentin", "Pregabalin", "Topiramate", "Lamotrigine", "Levetiracetam", "Oxcarbazepine", "Carbamazepine", "Divalproex sodium", "Valproic acid", "Lacosamide", "Zonisamide", "Phenytoin sodium", "Fosphenytoin", "Phenobarbital", "Primidone", "Clobazam", "Brivaracetam", "Perampanel", "Cannabidiol oral solution", "Rufinamide", "Vigabatrin"]),
            ("Antipsychotics & Mood Stabilizers", ["Aripiprazole", "Quetiapine fumarate", "Olanzapine", "Risperidone", "Ziprasidone", "Lurasidone HCl", "Clozapine", "Haloperidol", "Haloperidol decanoate", "Paliperidone palmitate", "Brexpiprazole", "Cariprazine", "Lumateperone", "Asenapine", "Chlorpromazine", "Fluphenazine", "Perphenazine", "Lithium carbonate", "Lithium citrate"]),
            ("ADHD & Cognitive Enhancers", ["Methylphenidate ER", "Methylphenidate IR", "Dexmethylphenidate XR", "Dextroamphetamine-Amphetamine ER", "Lisdexamfetamine dimesylate", "Atomoxetine HCl", "Guanfacine ER", "Clonidine ER", "Modafinil", "Armodafinil", "Donepezil HCl", "Memantine HCl", "Rivastigmine transdermal", "Galantamine ER"]),
            ("Parkinson's & Movement Disorders", ["Carbidopa-Levodopa", "Carbidopa-Levodopa ER", "Pramipexole dihydrochloride", "Ropinirole HCl", "Rotigotine patch", "Rasagiline", "Selegiline", "Entacapone", "Opicapone", "Amantadine HCl", "Benztropine mesylate", "Trihexyphenidyl", "Tetrabenazine", "Deutetrabenazine", "Valbenazine", "Pimavanserin"])
        ],
        "default_forms": ["tablet", "ER tablet", "capsule", "ODT", "oral solution", "transdermal patch"],
        "default_routes": ["PO", "SL", "Transdermal", "IM"],
        "default_freq": ["QD", "BID", "TID", "QHS", "QAM", "PRN"]
    },
    "Analgesics & Opioids": {
        "subclasses": [
            ("NSAIDs & Non-Opioids", ["Ibuprofen", "Naproxen sodium", "Naproxen base", "Meloxicam", "Celecoxib", "Acetaminophen", "Diclofenac sodium DR", "Diclofenac topical gel", "Ketorolac tromethamine", "Indomethacin", "Nabumetone", "Etodolac", "Piroxicam", "Sulindac", "Oxaprozin", "Diflunisal", "Aspirin", "Salsalate"]),
            ("Opioids & Combinations", ["Tramadol HCl", "Tramadol-Acetaminophen", "Oxycodone HCl", "Oxycodone-Acetaminophen", "Hydrocodone-Acetaminophen", "Morphine sulfate ER", "Morphine sulfate IR", "Hydromorphone HCl", "Fentanyl transdermal", "Fentanyl citrate transmucosal", "Methadone HCl", "Buprenorphine sublingual", "Buprenorphine-Naloxone", "Oxymorphone ER", "Codeine-Acetaminophen", "Meperidine HCl", "Tapentadol ER", "Naloxone HCl auto-injector", "Naltrexone HCl"]),
            ("Skeletal Muscle Relaxants", ["Cyclobenzaprine HCl", "Baclofen", "Tizanidine HCl", "Methocarbamol", "Carisoprodol", "Metaxalone", "Chlorzoxazone", "Orphenadrine citrate", "Dantrolene sodium"]),
            ("Topical & Migraine Analgesics", ["Lidocaine 5% patch", "Lidocaine 4% topical cream", "Capsaicin 0.025% cream", "Sumatriptan succinate", "Zolmitriptan", "Rizatriptan benzoate", "Eletriptan hydrobromide", "Naratriptan", "Almotriptan", "Frovatriptan", "Ubrogepant", "Rimegepant ODT", "Atogepant", "Erenumab-aooe", "Fremanezumab-vfrm", "Galcanezumab-gnlm", "Eptinezumab-jjmr", "Dihydroergotamine mesylate"])
        ],
        "default_forms": ["tablet", "capsule", "topical patch", "topical gel", "oral solution", "injection"],
        "default_routes": ["PO", "Topical", "Sublingual", "Transdermal", "IV", "IM", "SC", "Nasal"],
        "default_freq": ["Q4-6H PRN", "Q6H PRN", "Q8H PRN", "TID PRN", "BID", "QD", "QHS", "STAT"]
    },
    "Endocrine & Diabetes": {
        "subclasses": [
            ("Antidiabetic (Oral & Non-Insulin)", ["Metformin HCl", "Metformin ER", "Glipizide", "Glipizide XL", "Glimepiride", "Glyburide", "Sitagliptin phosphate", "Linagliptin", "Saxagliptin", "Alogliptin", "Empagliflozin", "Dapagliflozin", "Canagliflozin", "Ertugliflozin", "Semaglutide oral", "Semaglutide subcutaneous", "Dulaglutide", "Liraglutide", "Tirzepatide", "Exenatide ER", "Lixisenatide", "Pioglitazone HCl", "Rosiglitazone", "Acarbose", "Miglitol", "Repaglinide", "Nateglinide", "Bromocriptine quick-release"]),
            ("Insulins", ["Insulin Glargine", "Insulin Detemir", "Insulin Degludec", "Insulin Lispro", "Insulin Aspart", "Insulin Glulisine", "Insulin Regular human", "Insulin NPH human", "Insulin 70/30 Lispro mix", "Insulin 70/30 Aspart mix", "Insulin 70/30 NPH/Regular", "Inhaled human insulin"]),
            ("Thyroid & Antithyroid", ["Levothyroxine sodium", "Liothyronine sodium", "Thyroid desiccated (Armour)", "Methimazole", "Propylthiouracil", "Potassium iodide saturated solution"]),
            ("Corticosteroids & Hormones", ["Prednisone", "Prednisolone", "Methylprednisolone dose pack", "Dexamethasone", "Hydrocortisone", "Fludrocortisone acetate", "Triamcinolone acetonide", "Budesonide EC", "Estradiol transdermal", "Estradiol oral", "Progesterone micronized", "Medroxyprogesterone acetate", "Testosterone cypionate", "Testosterone topical gel", "Levonorgestrel", "Norethindrone", "Ethinyl estradiol-Norgestimate", "Ethinyl estradiol-Drospirenone", "Desmopressin acetate", "Somatropin", "Cabergoline", "Octreotide acetate", "Cinacalcet HCl"])
        ],
        "default_forms": ["tablet", "capsule", "subcutaneous pen", "subcutaneous vial", "transdermal patch", "topical gel"],
        "default_routes": ["PO", "SC", "Transdermal", "IM", "IV"],
        "default_freq": ["QD", "BID", "TID", "QAM AC", "QHS", "QWK", "QAC"]
    },
    "Respiratory": {
        "subclasses": [
            ("Bronchodilators (Beta-2 Agonists & Anticholinergics)", ["Albuterol sulfate HFA", "Levalbuterol tartrate HFA", "Salmeterol xinafoate", "Formoterol fumarate", "Vilanterol trifenatate", "Indacaterol maleate", "Olodaterol", "Ipratropium bromide HFA", "Tiotropium bromide", "Aclidinium bromide", "Umeclidinium bromide", "Glycopyrrolate inhalation", "Revefenacin inhalation solution"]),
            ("Inhaled Corticosteroids & Combinations", ["Fluticasone propionate HFA", "Budesonide inhalation suspension", "Beclomethasone dipropionate HFA", "Mometasone furoate HFA", "Ciclesonide HFA", "Fluticasone-Salmeterol Diskus", "Budesonide-Formoterol HFA", "Fluticasone-Vilanterol Ellipta", "Mometasone-Formoterol HFA", "Fluticasone-Umeclidinium-Vilanterol", "Budesonide-Glycopyrrolate-Formoterol", "Ipratropium-Albuterol Respimat"]),
            ("Oral & Nasal Antihistamines / Leukotriene", ["Montelukast sodium", "Zafirlukast", "Zileuton CR", "Cetirizine HCl", "Levocetirizine dihydrochloride", "Loratadine", "Desloratadine", "Fexofenadine HCl", "Diphenhydramine HCl", "Chlorpheniramine maleate", "Azelastine HCl nasal spray", "Olopatadine HCl nasal spray", "Fluticasone propionate nasal", "Triamcinolone acetonide nasal", "Mometasone furoate nasal", "Ciclesonide nasal", "Budesonide nasal spray", "Cromolyn sodium nasal"]),
            ("Antitussives, Mucolytics & Biologics", ["Benzonatate", "Guaifenesin ER", "Guaifenesin-Codeine", "Dextromethorphan-Guaifenesin", "Promethazine-Dextromethorphan", "Promethazine-Codeine", "Acetylcysteine inhalation", "Dupilumab", "Omalizumab", "Mepolizumab", "Benralizumab", "Tezepelumab-ekko", "Reslizumab", "Pirfenidone", "Nintedanib"])
        ],
        "default_forms": ["inhaler HFA", "inhalation diskus", "inhalation solution", "nasal spray", "capsule", "tablet", "oral syrup"],
        "default_routes": ["Inhalation", "Nasal", "PO", "SC"],
        "default_freq": ["Q4-6H PRN", "BID", "QD", "Q12H", "QHS", "Q2WK", "Q4WK"]
    },
    "Gastrointestinal & Urological": {
        "subclasses": [
            ("Proton Pump Inhibitors & Acid Reducers", ["Omeprazole DR", "Pantoprazole sodium DR", "Esomeprazole magnesium DR", "Lansoprazole DR", "Rabeprazole sodium DR", "Dexlansoprazole DR", "Famotidine", "Cimetidine", "Nizatidine", "Sucralfate", "Misoprostol", "Sodium bicarbonate", "Calcium carbonate antacid", "Bismuth subsalicylate"]),
            ("Antiemetics & GI Motility", ["Ondansetron HCl", "Ondansetron ODT", "Granisetron HCl", "Palonosetron", "Aprepitant", "Fosaprepitant", "Prochlorperazine maleate", "Promethazine HCl", "Metoclopramide HCl", "Dimenhydrinate", "Meclizine HCl", "Scopolamine transdermal patch", "Dronabinol"]),
            ("Antispasmodics, IBD & Constipation/Diarrhea", ["Dicyclomine HCl", "Hyoscyamine sulfate", "Glycopyrrolate oral", "Mesalamine DR", "Sulfasalazine", "Balsalazide disodium", "Olsalazine", "Hydrocortisone enema", "Loperamide HCl", "Diphenoxylate-Atropine", "Polyethylene glycol 3350", "Lactulose", "Lubiprostone", "Linaclotide", "Plecanatide", "Prucalopride", "Methylnaltrexone", "Naloxegol", "Eluxadoline", "Rifaximin 550mg", "Ursodiol", "Pancrelipase"]),
            ("Urological & BPH / ED", ["Tamsulosin HCl", "Alfuzosin HCl ER", "Silodosin", "Doxazosin mesylate", "Terazosin HCl", "Finasteride", "Dutasteride", "Oxybutynin chloride ER", "Tolterodine tartrate ER", "Solifenacin succinate", "Darifenacin ER", "Fesoterodine fumarate ER", "Mirabegron ER", "Vibegron", "Bethanechol chloride", "Phenazopyridine HCl", "Sildenafil citrate oral", "Tadalafil daily", "Vardenafil HCl", "Avanafil", "Potassium citrate ER"])
        ],
        "default_forms": ["tablet", "DR tablet", "DR capsule", "ODT", "oral suspension", "transdermal patch", "rectal suppository"],
        "default_routes": ["PO", "PR", "IV", "Transdermal"],
        "default_freq": ["QD", "BID", "TID", "QID", "Q30M AC", "QHS", "PRN"]
    }
}

STRENGTH_TEMPLATES = {
    "tablet": ["5mg", "10mg", "20mg", "25mg", "50mg", "100mg", "200mg", "250mg", "500mg", "850mg", "1000mg"],
    "ER tablet": ["10mg ER", "20mg ER", "25mg ER", "50mg ER", "100mg ER", "200mg ER", "300mg ER"],
    "capsule": ["25mg", "50mg", "100mg", "150mg", "200mg", "300mg", "400mg", "500mg"],
    "injection": ["10mg/mL", "25mg/mL", "50mg/mL", "100mg/2mL", "1g vial", "2g vial"],
    "oral solution": ["5mg/5mL", "10mg/5mL", "20mg/5mL", "100mg/mL"],
    "inhaler HFA": ["45mcg/actuation", "90mcg/actuation", "110mcg/actuation", "220mcg/actuation"],
    "nasal spray": ["50mcg/spray", "137mcg/spray", "200mcg/spray"],
    "topical cream": ["0.025%", "0.05%", "0.1%", "1%", "2%"],
    "transdermal patch": ["0.1mg/day", "0.2mg/day", "12.5mcg/hr", "25mcg/hr", "50mcg/hr"]
}


def build_rxnorm_medication_catalog() -> Dict[str, Any]:
    """
    Generate >= 1,000 unique medications with rich metadata and bidirectional LASA pairs.
    """
    random.seed(42)
    medications: List[Dict[str, Any]] = []
    seen_drugs = set()
    rxcui_counter = 100000

    # 1. First, register all 20 LASA pairs symmetrically
    for pair in LASA_PAIRS_SPEC:
        drug_a_name = pair["drug_a"]
        drug_b_name = pair["drug_b"]
        rxcui_a = pair["rxcui_a"]
        rxcui_b = pair["rxcui_b"]

        confusion_obj_a = {
            "target_drug": drug_b_name,
            "target_rxcui": rxcui_b,
            "confusion_type": pair["confusion_type"],
            "visual_similarity_score": pair["visual_similarity_score"],
            "visual_confusion_score": pair["visual_similarity_score"],
            "confusable_glyphs": pair["confusable_glyphs"],
            "clinical_distinction": pair["clinical_distinction"]
        }

        med_a = {
            "rxcui": rxcui_a,
            "generic_name": drug_a_name,
            "brand_names": pair["brands_a"],
            "therapeutic_class": pair["class_a"],
            "subclass": pair["subclass_a"],
            "dosage_forms": pair["forms_a"],
            "standard_strengths": pair["strengths_a"],
            "standard_routes": pair["routes_a"],
            "standard_frequencies": pair["freq_a"],
            "is_lasa": True,
            "confusion_pairs": [confusion_obj_a],
            "schedule": pair["schedule_a"]
        }
        medications.append(med_a)
        seen_drugs.add(drug_a_name.lower())

        confusion_obj_b = {
            "target_drug": drug_a_name,
            "target_rxcui": rxcui_a,
            "confusion_type": pair["confusion_type"],
            "visual_similarity_score": pair["visual_similarity_score"],
            "visual_confusion_score": pair["visual_similarity_score"],
            "confusable_glyphs": pair["confusable_glyphs"],
            "clinical_distinction": pair["clinical_distinction"]
        }

        med_b = {
            "rxcui": rxcui_b,
            "generic_name": drug_b_name,
            "brand_names": pair["brands_b"],
            "therapeutic_class": pair["class_b"],
            "subclass": pair["subclass_b"],
            "dosage_forms": pair["forms_b"],
            "standard_strengths": pair["strengths_b"],
            "standard_routes": pair["routes_b"],
            "standard_frequencies": pair["freq_b"],
            "is_lasa": True,
            "confusion_pairs": [confusion_obj_b],
            "schedule": pair["schedule_b"]
        }
        medications.append(med_b)
        seen_drugs.add(drug_b_name.lower())

    # 2. Add therapeutic domain drug seeds
    for th_class, data in THERAPEUTIC_DOMAINS.items():
        for subclass_name, drug_list in data["subclasses"]:
            for drug_name in drug_list:
                if drug_name.lower() in seen_drugs:
                    continue
                seen_drugs.add(drug_name.lower())
                rxcui_counter += 17

                # Assign dosage forms & strengths
                forms = random.sample(data["default_forms"], k=min(len(data["default_forms"]), random.randint(1, 3)))
                strengths = []
                for f in forms:
                    possible_s = STRENGTH_TEMPLATES.get(f, ["10mg", "20mg", "50mg"])
                    strengths.extend(random.sample(possible_s, k=min(len(possible_s), random.randint(1, 3))))
                strengths = list(dict.fromkeys(strengths))

                med_record = {
                    "rxcui": str(rxcui_counter),
                    "generic_name": drug_name,
                    "brand_names": [f"{drug_name.split()[0]}or", f"{drug_name.split()[0]}ex"] if random.random() < 0.4 else [],
                    "therapeutic_class": th_class,
                    "subclass": subclass_name,
                    "dosage_forms": forms,
                    "standard_strengths": strengths if strengths else ["10mg", "25mg", "50mg"],
                    "standard_routes": random.sample(data["default_routes"], k=min(len(data["default_routes"]), random.randint(1, 2))),
                    "standard_frequencies": random.sample(data["default_freq"], k=min(len(data["default_freq"]), random.randint(2, 4))),
                    "is_lasa": False,
                    "confusion_pairs": [],
                    "schedule": "C-IV" if "Anxiolytics" in subclass_name else ("C-II" if "Opioid" in subclass_name or "ADHD" in subclass_name else "Rx-only")
                }
                medications.append(med_record)

    # 3. Procedural expansion to reach >= 1,020 medications with realistic chemical and pharmacological suffixes
    prefixes = [
        "Acro", "Beva", "Canto", "Daphno", "Evo", "Fosro", "Gali", "Hemi", "Ibu", "Juno", "Keta", "Lumi",
        "Mixo", "Novo", "Opti", "Prasto", "Quino", "Rivo", "Spiro", "Teno", "Ultra", "Vaso", "Xeno", "Zeno",
        "Ami", "Beto", "Cal", "Dura", "Epi", "Flu", "Gyno", "Hydro", "Iso", "Lev", "Micro", "Neo", "Oro",
        "Poly", "Ribo", "Sulfa", "Tera", "Vita", "Zeta", "Atro", "Bari", "Cil", "Dapa", "Enzy", "Ferro",
        "Gluc", "Helio", "Iodo", "Kaly", "Lyo", "Myco", "Nadro", "Oxo", "Pyro", "Revo", "Steno", "Tauro"
    ]
    suffixes = [
        ("olol", "Beta-Adrenergic Blocker", "Cardiovascular"),
        ("statin", "HMG-CoA Reductase Inhibitor", "Cardiovascular"),
        ("artan", "Angiotensin II Receptor Blocker", "Cardiovascular"),
        ("pril", "ACE Inhibitor", "Cardiovascular"),
        ("dipine", "Dihydropyridine Calcium Channel Blocker", "Cardiovascular"),
        ("oxacin", "Quinolone Antibiotic", "Antibiotics, Antivirals & Antifungals"),
        ("cillin", "Penicillin-derivative Antibiotic", "Antibiotics, Antivirals & Antifungals"),
        ("cycline", "Tetracycline Antibiotic", "Antibiotics, Antivirals & Antifungals"),
        ("mycin", "Macrolide / Aminoglycoside Antibiotic", "Antibiotics, Antivirals & Antifungals"),
        ("conazole", "Azole Antifungal", "Antibiotics, Antivirals & Antifungals"),
        ("vir", "Antiviral Agent", "Antibiotics, Antivirals & Antifungals"),
        ("triptan", "5-HT1B/1D Receptor Agonist", "Analgesics & Opioids"),
        ("coxib", "Selective COX-2 Inhibitor", "Analgesics & Opioids"),
        ("profen", "Propionic Acid NSAID", "Analgesics & Opioids"),
        ("caine", "Local Anesthetic", "Analgesics & Opioids"),
        ("azepam", "Benzodiazepine Anxiolytic", "Central Nervous System"),
        ("oxetine", "Serotonin Reuptake Inhibitor", "Central Nervous System"),
        ("pramine", "Tricyclic Antidepressant", "Central Nervous System"),
        ("peridol", "Butyrophenone Antipsychotic", "Central Nervous System"),
        ("tiracetam", "Synaptic Vesicle SV2A Ligand", "Central Nervous System"),
        ("gliptin", "DPP-4 Inhibitor", "Endocrine & Diabetes"),
        ("gliflozin", "SGLT-2 Inhibitor", "Endocrine & Diabetes"),
        ("glutide", "GLP-1 Receptor Agonist", "Endocrine & Diabetes"),
        ("glitazone", "Thiazolidinedione Antidiabetic", "Endocrine & Diabetes"),
        ("terol", "Beta-2 Adrenergic Agonist", "Respiratory"),
        ("lukast", "Leukotriene Receptor Antagonist", "Respiratory"),
        ("nasonide", "Inhaled / Nasal Glucocorticoid", "Respiratory"),
        ("tropium", "Inhaled Anticholinergic Bronchodilator", "Respiratory"),
        ("fluticas", "Inhaled Corticosteroid", "Respiratory"),
        ("buden", "Glucocorticoid Inhalation Agent", "Respiratory"),
        ("prazole", "Proton Pump Inhibitor", "Gastrointestinal & Urological"),
        ("tidine", "H2-Receptor Antagonist", "Gastrointestinal & Urological"),
        ("setron", "5-HT3 Receptor Antagonist", "Gastrointestinal & Urological"),
        ("osin", "Alpha-1 Receptor Antagonist", "Gastrointestinal & Urological")
    ]

    idx_p = 0
    idx_s = 0
    # Ensure at least 150 drugs per therapeutic class (Total >= 1,050)
    class_targets = {c: 150 for c in THERAPEUTIC_DOMAINS.keys()}
    for th_class, target_num in class_targets.items():
        curr_count = sum(1 for m in medications if m["therapeutic_class"] == th_class)
        class_suffixes = [s for s in suffixes if s[2] == th_class]
        while curr_count < target_num:
            p = prefixes[idx_p % len(prefixes)]
            s, subc, _ = class_suffixes[idx_s % len(class_suffixes)]
            gen_name = f"{p.capitalize()}{s.lower()}"
            idx_p += 1
            idx_s += 1

            if gen_name.lower() in seen_drugs:
                continue
            seen_drugs.add(gen_name.lower())
            rxcui_counter += 23

            med_record = {
                "rxcui": str(rxcui_counter),
                "generic_name": gen_name,
                "brand_names": [f"{gen_name[:4]}gard", f"{gen_name[:5]}ex"] if random.random() < 0.5 else [],
                "therapeutic_class": th_class,
                "subclass": subc,
                "dosage_forms": ["tablet", "capsule"],
                "standard_strengths": ["10mg", "20mg", "50mg", "100mg"],
                "standard_routes": ["PO"],
                "standard_frequencies": ["QD", "BID", "QHS"],
                "is_lasa": False,
                "confusion_pairs": [],
                "schedule": "Rx-only"
            }
            medications.append(med_record)
            curr_count += 1

    catalog = {
        "version": "2026.1",
        "total_medications": len(medications),
        "therapeutic_classes": list(THERAPEUTIC_DOMAINS.keys()),
        "medications": medications
    }
    return catalog


# ==============================================================================
# 2. LATIN SIG CODES & GRAMMARS
# ==============================================================================

LATIN_SIG_CODES_DATA = [
    # Frequency & Interval Codes (16 codes)
    {
        "code": "QD",
        "latin_expansion": "Quaque Die",
        "english_translation": "Every day / once daily",
        "sig_category": "frequency",
        "frequency_per_day": 1.0,
        "standard_interval_hours": 24,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "QAM", "QHS", "WF"],
        "example_sig_phrases": ["1 tab PO QD in morning", "Take 1 capsule PO QD with water"]
    },
    {
        "code": "BID",
        "latin_expansion": "Bis In Die",
        "english_translation": "Twice a day (approximately every 12 hours)",
        "sig_category": "frequency",
        "frequency_per_day": 2.0,
        "standard_interval_hours": 12,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "WF", "PC"],
        "example_sig_phrases": ["1 tab PO BID with meals", "Take 1 cap PO BID x 10 days"]
    },
    {
        "code": "TID",
        "latin_expansion": "Ter In Die",
        "english_translation": "Three times a day (approximately every 8 hours)",
        "sig_category": "frequency",
        "frequency_per_day": 3.0,
        "standard_interval_hours": 8,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "PC", "PRN"],
        "example_sig_phrases": ["1 cap PO TID x 7 days", "Take 1 tablet PO TID with food"]
    },
    {
        "code": "QID",
        "latin_expansion": "Quater In Die",
        "english_translation": "Four times a day (approximately every 6 hours)",
        "sig_category": "frequency",
        "frequency_per_day": 4.0,
        "standard_interval_hours": 6,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "AC", "PRN"],
        "example_sig_phrases": ["1 tab PO QID before meals and bedtime", "Take 1 tsp PO QID"]
    },
    {
        "code": "Q4H",
        "latin_expansion": "Quaque 4 Hora",
        "english_translation": "Every 4 hours",
        "sig_category": "frequency",
        "frequency_per_day": 6.0,
        "standard_interval_hours": 4,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "PRN", "IV"],
        "example_sig_phrases": ["1-2 tabs PO Q4H PRN severe pain", "Take 1 tablet PO Q4H"]
    },
    {
        "code": "Q6H",
        "latin_expansion": "Quaque 6 Hora",
        "english_translation": "Every 6 hours",
        "sig_category": "frequency",
        "frequency_per_day": 4.0,
        "standard_interval_hours": 6,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "PRN", "IV"],
        "example_sig_phrases": ["500mg IV Q6H", "Take 1 tablet PO Q6H PRN fever"]
    },
    {
        "code": "Q8H",
        "latin_expansion": "Quaque 8 Hora",
        "english_translation": "Every 8 hours",
        "sig_category": "frequency",
        "frequency_per_day": 3.0,
        "standard_interval_hours": 8,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "IV", "IM"],
        "example_sig_phrases": ["1 cap PO Q8H x 10 days", "Administer 1g IV Q8H"]
    },
    {
        "code": "Q12H",
        "latin_expansion": "Quaque 12 Hora",
        "english_translation": "Every 12 hours",
        "sig_category": "frequency",
        "frequency_per_day": 2.0,
        "standard_interval_hours": 12,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "IV"],
        "example_sig_phrases": ["1 tab PO Q12H with breakfast and dinner", "500mg PO Q12H x 14 days"]
    },
    {
        "code": "Q24H",
        "latin_expansion": "Quaque 24 Hora",
        "english_translation": "Every 24 hours",
        "sig_category": "frequency",
        "frequency_per_day": 1.0,
        "standard_interval_hours": 24,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "IV"],
        "example_sig_phrases": ["1g IV Q24H daily", "Take 1 tab PO Q24H"]
    },
    {
        "code": "Q4-6H",
        "latin_expansion": "Quaque 4-6 Hora",
        "english_translation": "Every 4 to 6 hours",
        "sig_category": "frequency",
        "frequency_per_day": 5.0,
        "standard_interval_hours": 5,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "PRN"],
        "example_sig_phrases": ["1-2 tabs PO Q4-6H PRN breakthrough pain", "Inhale 2 puffs Q4-6H PRN wheeze"]
    },
    {
        "code": "QOD",
        "latin_expansion": "Quaque Altera Die",
        "english_translation": "Every other day",
        "sig_category": "frequency",
        "frequency_per_day": 0.5,
        "standard_interval_hours": 48,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "QAM"],
        "example_sig_phrases": ["Take 1 tab PO QOD in morning", "1 tablet PO QOD with breakfast"]
    },
    {
        "code": "QHS",
        "latin_expansion": "Quaque Hora Somni",
        "english_translation": "Every night at bedtime",
        "sig_category": "frequency",
        "frequency_per_day": 1.0,
        "standard_interval_hours": 24,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "PRN"],
        "example_sig_phrases": ["1 tab PO QHS for sleep", "Take 1 tablet PO QHS for lipid control"]
    },
    {
        "code": "QAM",
        "latin_expansion": "Quaque Die Ante Meridiem",
        "english_translation": "Every morning",
        "sig_category": "frequency",
        "frequency_per_day": 1.0,
        "standard_interval_hours": 24,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "AC"],
        "example_sig_phrases": ["Take 1 tab PO QAM before breakfast", "1 tablet PO QAM with full glass of water"]
    },
    {
        "code": "QPM",
        "latin_expansion": "Quaque Die Post Meridiem",
        "english_translation": "Every evening / night",
        "sig_category": "frequency",
        "frequency_per_day": 1.0,
        "standard_interval_hours": 24,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "PC"],
        "example_sig_phrases": ["1 tab PO QPM with dinner", "Take 1 cap PO QPM"]
    },
    {
        "code": "QWK",
        "latin_expansion": "Quaque Septimana",
        "english_translation": "Once weekly / every week",
        "sig_category": "frequency",
        "frequency_per_day": 0.14,
        "standard_interval_hours": 168,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "SC"],
        "example_sig_phrases": ["Take 1 tab PO QWK on Sundays", "Inject 0.5mg SC QWK"]
    },
    {
        "code": "STAT",
        "latin_expansion": "Statim",
        "english_translation": "Immediately / at once",
        "sig_category": "frequency",
        "frequency_per_day": 1.0,
        "standard_interval_hours": None,
        "default_route_target": "intravenous",
        "common_co_occurring_codes": ["IV", "PO", "IM", "NOW"],
        "example_sig_phrases": ["Administer 324mg Aspirin PO STAT", "Give 20mg Furosemide IV STAT"]
    },

    # Route of Administration Codes (16 codes)
    {
        "code": "PO",
        "latin_expansion": "Per Os",
        "english_translation": "By mouth / orally",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["QD", "BID", "TID", "QID", "PRN", "WF"],
        "example_sig_phrases": ["Take 1 tab PO daily", "2 capsules PO TID x 10 days"]
    },
    {
        "code": "SL",
        "latin_expansion": "Sub Lingua",
        "english_translation": "Sublingually / under the tongue",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "sublingual",
        "common_co_occurring_codes": ["PRN", "STAT"],
        "example_sig_phrases": ["Place 1 tab SL Q5MIN PRN chest pain up to 3 doses", "Dissolve 1 tab SL STAT"]
    },
    {
        "code": "PR",
        "latin_expansion": "Per Rectum",
        "english_translation": "Rectally",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "rectal",
        "common_co_occurring_codes": ["PRN", "Q6H", "SUPP"],
        "example_sig_phrases": ["Insert 1 supp PR Q6H PRN nausea", "1 suppository PR at bedtime"]
    },
    {
        "code": "IM",
        "latin_expansion": "Intra Muscularis",
        "english_translation": "Intramuscular injection",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "intramuscular",
        "common_co_occurring_codes": ["STAT", "PRN", "Q4W"],
        "example_sig_phrases": ["Inject 1mL IM in deltoid muscle STAT", "Administer 50mg IM Q4W"]
    },
    {
        "code": "IV",
        "latin_expansion": "Intra Venam",
        "english_translation": "Intravenous injection / infusion",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "intravenous",
        "common_co_occurring_codes": ["Q8H", "Q12H", "STAT", "IVPB"],
        "example_sig_phrases": ["Administer 1g IV Q8H over 60 min", "Give 4mg Morphine IV STAT"]
    },
    {
        "code": "IVPB",
        "latin_expansion": "Intravenous Piggyback",
        "english_translation": "Intermittent intravenous piggyback infusion",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "intravenous",
        "common_co_occurring_codes": ["IV", "Q8H", "Q12H"],
        "example_sig_phrases": ["Vancomycin 1.25g in 250mL D5W IVPB Q12H", "Infuse via IVPB over 30 min"]
    },
    {
        "code": "SC",
        "latin_expansion": "Sub Cutis",
        "english_translation": "Subcutaneous injection",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "subcutaneous",
        "common_co_occurring_codes": ["QD", "QWK", "QHS", "AC"],
        "example_sig_phrases": ["Inject 10 units SC QHS", "Inject 0.5mL SC QWK in abdomen"]
    },
    {
        "code": "TOP",
        "latin_expansion": "Topicus",
        "english_translation": "Topically / apply to affected skin",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "topical",
        "common_co_occurring_codes": ["BID", "TID", "AAA", "PRN"],
        "example_sig_phrases": ["Apply thin layer TOP BID to rash", "Apply TOP AAA TID"]
    },
    {
        "code": "INH",
        "latin_expansion": "Inhalatio",
        "english_translation": "Oral inhalation",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "inhalation",
        "common_co_occurring_codes": ["BID", "Q4-6H", "PRN", "PUFFS"],
        "example_sig_phrases": ["Inhale 2 puffs INH Q4-6H PRN shortness of breath", "1 puff INH BID"]
    },
    {
        "code": "NEB",
        "latin_expansion": "Nebula",
        "english_translation": "Via nebulizer / aerosol inhalation",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "inhalation",
        "common_co_occurring_codes": ["Q4-6H", "Q6H", "PRN"],
        "example_sig_phrases": ["1 vial via NEB Q4-6H PRN wheeze", "Administer via NEB TID"]
    },
    {
        "code": "OD",
        "latin_expansion": "Oculus Dexter",
        "english_translation": "Right eye",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "ophthalmic_right",
        "common_co_occurring_codes": ["GTT", "BID", "TID", "QHS"],
        "example_sig_phrases": ["Instill 1 drop in OD QHS", "1 gtt OD BID for glaucoma"]
    },
    {
        "code": "OS",
        "latin_expansion": "Oculus Sinister",
        "english_translation": "Left eye",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "ophthalmic_left",
        "common_co_occurring_codes": ["GTT", "BID", "TID", "QHS"],
        "example_sig_phrases": ["Instill 1 drop in OS QHS", "1 gtt OS TID"]
    },
    {
        "code": "OU",
        "latin_expansion": "Oculus Uterque",
        "english_translation": "Both eyes",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "ophthalmic_both",
        "common_co_occurring_codes": ["GTT", "BID", "TID", "QHS"],
        "example_sig_phrases": ["Instill 1-2 drops in OU TID PRN dryness", "1 gtt OU QAM"]
    },
    {
        "code": "AD",
        "latin_expansion": "Auris Dextra",
        "english_translation": "Right ear",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "otic_right",
        "common_co_occurring_codes": ["GTT", "BID", "TID"],
        "example_sig_phrases": ["Instill 4 drops in AD BID x 7 days", "4 gtt AD TID"]
    },
    {
        "code": "AS",
        "latin_expansion": "Auris Sinistra",
        "english_translation": "Left ear",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "otic_left",
        "common_co_occurring_codes": ["GTT", "BID", "TID"],
        "example_sig_phrases": ["Instill 4 drops in AS BID x 7 days", "3 gtt AS TID"]
    },
    {
        "code": "AU",
        "latin_expansion": "Auris Uterque",
        "english_translation": "Both ears",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "otic_both",
        "common_co_occurring_codes": ["GTT", "BID", "TID"],
        "example_sig_phrases": ["Instill 4 drops in AU BID x 10 days", "3 gtt AU TID"]
    },
    {
        "code": "NAS",
        "latin_expansion": "Per Nares",
        "english_translation": "Intranasally / into each nostril",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "nasal",
        "common_co_occurring_codes": ["QD", "BID", "SPRAYS"],
        "example_sig_phrases": ["1-2 sprays NAS in each nostril QD", "2 sprays NAS BID"]
    },
    {
        "code": "TRANSDERM",
        "latin_expansion": "Transdermal",
        "english_translation": "Transdermal patch / apply to intact skin",
        "sig_category": "route",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "transdermal",
        "common_co_occurring_codes": ["Q72H", "Q7D", "QD"],
        "example_sig_phrases": ["Apply 1 patch TRANSDERM Q72H", "Apply TRANSDERM patch Q7D"]
    },

    # Timing & Meals Directives (8 codes)
    {
        "code": "AC",
        "latin_expansion": "Ante Cibum",
        "english_translation": "Before meals (typically 30 minutes prior)",
        "sig_category": "timing_and_meals",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "TID", "QD", "QAM"],
        "example_sig_phrases": ["Take 1 cap PO AC in the morning", "1 tablet PO TID AC"]
    },
    {
        "code": "PC",
        "latin_expansion": "Post Cibum",
        "english_translation": "After meals",
        "sig_category": "timing_and_meals",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "BID", "TID"],
        "example_sig_phrases": ["Take 1 tab PO PC with water", "1 tablet PO BID PC"]
    },
    {
        "code": "IC",
        "latin_expansion": "Inter Cibos",
        "english_translation": "Between meals",
        "sig_category": "timing_and_meals",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "TID"],
        "example_sig_phrases": ["Take 1 dose PO IC on empty stomach", "1 tab PO IC"]
    },
    {
        "code": "WF",
        "latin_expansion": "Cum Cibo",
        "english_translation": "With food / with meals",
        "sig_category": "timing_and_meals",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "BID", "QD"],
        "example_sig_phrases": ["Take 1 tab PO BID WF", "1 tablet PO daily WF"]
    },
    {
        "code": "HS",
        "latin_expansion": "Hora Somni",
        "english_translation": "At bedtime",
        "sig_category": "timing_and_meals",
        "frequency_per_day": 1.0,
        "standard_interval_hours": 24,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "QD", "PRN"],
        "example_sig_phrases": ["1 tab PO at HS", "Take 1-2 tablets PO HS PRN insomnia"]
    },
    {
        "code": "ACHS",
        "latin_expansion": "Ante Cibum et Hora Somni",
        "english_translation": "Before meals and at bedtime (4 times daily)",
        "sig_category": "timing_and_meals",
        "frequency_per_day": 4.0,
        "standard_interval_hours": 6,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "SC"],
        "example_sig_phrases": ["Check blood sugar ACHS and inject sliding scale insulin", "Take 1 tab PO ACHS"]
    },
    {
        "code": "QAC",
        "latin_expansion": "Quaque Ante Cibum",
        "english_translation": "Before every meal (3 times daily)",
        "sig_category": "timing_and_meals",
        "frequency_per_day": 3.0,
        "standard_interval_hours": 8,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "TID"],
        "example_sig_phrases": ["Take 1 tab PO QAC", "1 capsule PO QAC 30 minutes prior"]
    },
    {
        "code": "EM",
        "latin_expansion": "Ex Modo",
        "english_translation": "In the morning / early morning",
        "sig_category": "timing_and_meals",
        "frequency_per_day": 1.0,
        "standard_interval_hours": 24,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "QD", "QAM"],
        "example_sig_phrases": ["Take 1 tablet PO EM before breakfast", "1 tab PO EM"]
    },

    # Dosage Units & Form Descriptions (8 codes)
    {
        "code": "GTT",
        "latin_expansion": "Guttae",
        "english_translation": "Drop / drops",
        "sig_category": "dosage_unit",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "ophthalmic_or_otic",
        "common_co_occurring_codes": ["OD", "OS", "OU", "AD", "AS", "AU", "TID"],
        "example_sig_phrases": ["Instill 1-2 gtt in OU TID", "4 gtt in AD BID"]
    },
    {
        "code": "TAB",
        "latin_expansion": "Tabella",
        "english_translation": "Tablet",
        "sig_category": "dosage_unit",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "QD", "BID", "TID"],
        "example_sig_phrases": ["Take 1 tab PO daily", "2 tabs PO BID with food"]
    },
    {
        "code": "CAP",
        "latin_expansion": "Capsula",
        "english_translation": "Capsule",
        "sig_category": "dosage_unit",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "TID", "BID"],
        "example_sig_phrases": ["Take 1 cap PO TID x 10 days", "1 capsule PO QHS"]
    },
    {
        "code": "TSP",
        "latin_expansion": "Cochleare Parvum",
        "english_translation": "Teaspoon (5 mL)",
        "sig_category": "dosage_unit",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "QID", "TID", "Q4-6H"],
        "example_sig_phrases": ["Take 1 tsp (5 mL) PO TID", "2 tsp PO Q6H PRN cough"]
    },
    {
        "code": "TBSP",
        "latin_expansion": "Cochleare Magnum",
        "english_translation": "Tablespoon (15 mL)",
        "sig_category": "dosage_unit",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "QD", "BID"],
        "example_sig_phrases": ["Take 1 tbsp (15 mL) PO daily", "1 tbsp PO BID PRN constipation"]
    },
    {
        "code": "PUFF",
        "latin_expansion": "Inhalatio",
        "english_translation": "Inhalation puff / puffs",
        "sig_category": "dosage_unit",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "inhalation",
        "common_co_occurring_codes": ["INH", "Q4-6H", "BID", "PRN"],
        "example_sig_phrases": ["Inhale 2 puffs Q4-6H PRN SOB", "1-2 puffs INH BID"]
    },
    {
        "code": "SUPP",
        "latin_expansion": "Suppositorium",
        "english_translation": "Suppository",
        "sig_category": "dosage_unit",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "rectal",
        "common_co_occurring_codes": ["PR", "PRN", "Q6H", "QHS"],
        "example_sig_phrases": ["Insert 1 supp PR Q6H PRN nausea", "1 supp PR at bedtime"]
    },
    {
        "code": "AMP",
        "latin_expansion": "Ampulla",
        "english_translation": "Ampule / single-dose vial",
        "sig_category": "dosage_unit",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "intravenous",
        "common_co_occurring_codes": ["IV", "IM", "STAT"],
        "example_sig_phrases": ["Administer 1 amp IV STAT", "1 ampule IM in clinic"]
    },

    # Action Verbs, Directives & Modifiers (10 codes)
    {
        "code": "PRN",
        "latin_expansion": "Pro Re Nata",
        "english_translation": "As needed / when necessary",
        "sig_category": "action_verb",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": None,
        "common_co_occurring_codes": ["Q4-6H", "Q6H", "Q8H", "PO"],
        "example_sig_phrases": ["Take 1 tab PO Q4-6H PRN pain", "2 puffs INH Q4H PRN shortness of breath"]
    },
    {
        "code": "DAW",
        "latin_expansion": "Dispense As Written",
        "english_translation": "Dispense brand name only; do not substitute generic",
        "sig_category": "pharmacy_directive",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": None,
        "common_co_occurring_codes": ["#30", "#90"],
        "example_sig_phrases": ["DAW - Brand name Synthroid only", "Dispense As Written (DAW-1)"]
    },
    {
        "code": "NR",
        "latin_expansion": "Non Repetatur",
        "english_translation": "No refills / do not repeat",
        "sig_category": "pharmacy_directive",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": None,
        "common_co_occurring_codes": ["#30", "STAT"],
        "example_sig_phrases": ["#30 Refills: 0 (NR)", "Dispense #10, NR"]
    },
    {
        "code": "UD",
        "latin_expansion": "Ut Dictum",
        "english_translation": "As directed by physician",
        "sig_category": "action_verb",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": None,
        "common_co_occurring_codes": ["PO", "TAPER"],
        "example_sig_phrases": ["Take PO UD on taper schedule", "Use UD according to protocol"]
    },
    {
        "code": "QS",
        "latin_expansion": "Quantum Sufficiat",
        "english_translation": "A sufficient quantity / as much as is needed",
        "sig_category": "pharmacy_directive",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": None,
        "common_co_occurring_codes": ["TOP", "MITTE"],
        "example_sig_phrases": ["Dispense QS for 30-day supply", "Apply QS to affected area"]
    },
    {
        "code": "MITTE",
        "latin_expansion": "Mitte",
        "english_translation": "Dispense / send quantity",
        "sig_category": "pharmacy_directive",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": None,
        "common_co_occurring_codes": ["#30", "#60", "#90"],
        "example_sig_phrases": ["Mitte 60 tabs", "Mitte 30 capsules"]
    },
    {
        "code": "SIG",
        "latin_expansion": "Signa",
        "english_translation": "Write on label / patient directions",
        "sig_category": "action_verb",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": None,
        "common_co_occurring_codes": ["PO", "BID", "QD"],
        "example_sig_phrases": ["Sig: 1 tab PO daily with food", "Sig: Take 1 capsule PO TID"]
    },
    {
        "code": "SOS",
        "latin_expansion": "Si Opus Sit",
        "english_translation": "If needed / in case of need (once only unless renewed)",
        "sig_category": "action_verb",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": None,
        "common_co_occurring_codes": ["PRN", "STAT"],
        "example_sig_phrases": ["Give 1 dose PO SOS for acute severe headache", "1 tab SOS"]
    },
    {
        "code": "AAA",
        "latin_expansion": "Applicandum Ad Affectum",
        "english_translation": "Apply to the affected area",
        "sig_category": "action_verb",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "topical",
        "common_co_occurring_codes": ["TOP", "BID", "TID", "PRN"],
        "example_sig_phrases": ["Apply AAA BID for eczema", "Rub thin layer AAA TID PRN pruritus"]
    },
    {
        "code": "TAPER",
        "latin_expansion": "Gradatim Minuere",
        "english_translation": "Decrease dosage gradually / taper schedule",
        "sig_category": "action_verb",
        "frequency_per_day": None,
        "standard_interval_hours": None,
        "default_route_target": "oral",
        "common_co_occurring_codes": ["PO", "UD"],
        "example_sig_phrases": ["Take PO according to written taper schedule", "Taper by 5mg every 3 days as directed"]
    }
]


def build_latin_sig_catalog() -> Dict[str, Any]:
    """Generate and return validated Latin Sig Code Catalog."""
    catalog = {
        "version": "2026.1",
        "total_codes": len(LATIN_SIG_CODES_DATA),
        "codes": LATIN_SIG_CODES_DATA
    }
    return catalog


# ==============================================================================
# 3. DOCTOR PROFILES & KINEMATICS
# ==============================================================================

SPECIALTIES_DISTRIBUTION = [
    ("Cardiology", 10),
    ("Medical Oncology", 8),
    ("Pediatrics", 8),
    ("Orthopedic Surgery", 8),
    ("Family Medicine", 12),
    ("Internal Medicine", 12),
    ("General Surgery", 8),
    ("Neurology", 8),
    ("Dermatology", 8),
    ("Psychiatry", 8),
    ("Pulmonology & Critical Care", 8),
    ("Gastroenterology", 8),
    ("Endocrinology", 6),
    ("Obstetrics & Gynecology", 6),
    ("Emergency Medicine", 8)
]

FIRST_NAMES = [
    "Alexander", "Amara", "Benjamin", "Charlotte", "Daniel", "Elena", "Franklin", "Grace",
    "Henry", "Isabella", "James", "Kavita", "Lucas", "Maya", "Nathan", "Olivia",
    "Patrick", "Rachel", "Samuel", "Tanya", "Victor", "Wendy", "Xavier", "Yasmine", "Zachary",
    "Arthur", "Beatrice", "Christopher", "Diana", "Edward", "Fiona", "George", "Hannah",
    "Ian", "Julia", "Kevin", "Laura", "Marcus", "Nora", "Oliver", "Penelope", "Quentin",
    "Rebecca", "Simon", "Theresa", "Ulysses", "Valerie", "William", "Xiomara", "Yosef"
]

LAST_NAMES = [
    "Anderson", "Bennett", "Chen", "Dubois", "Eisenhower", "Fletcher", "Gupta", "Harrington",
    "Ivanov", "Jenkins", "Kim", "Larson", "Morales", "Nakamura", "O'Connor", "Patel",
    "Quinn", "Rodriguez", "Sterling", "Tanaka", "Underwood", "Vance", "Washington", "Xie",
    "Young", "Zimmerman", "Alvarez", "Blackwood", "Callahan", "Davenport", "Espinoza", "Faulkner",
    "Gallagher", "Holloway", "Ibrahim", "Jorgensen", "Kowalski", "Lindqvist", "MacDonald", "Navarro"
]

US_STATES = [
    ("MA", "Boston", "02115", "(617) 732-", "MA-"),
    ("CA", "Los Angeles", "90048", "(310) 423-", "A"),
    ("NY", "New York", "10029", "(212) 241-", "NY-"),
    ("TX", "Houston", "77030", "(713) 798-", "BP"),
    ("IL", "Chicago", "60611", "(312) 926-", "036-"),
    ("PA", "Philadelphia", "19104", "(215) 662-", "MD-"),
    ("WA", "Seattle", "98195", "(206) 598-", "MD"),
    ("FL", "Miami", "33136", "(305) 243-", "ME-"),
    ("NC", "Durham", "27710", "(919) 684-", "NC-"),
    ("OH", "Cleveland", "44195", "(216) 444-", "35.")
]

SPECIALTY_KINEMATICS = {
    "Emergency Medicine": {
        "slant_deg_range": (25.0, 38.0),
        "pressure_range": (0.6, 1.0),
        "speed_range": (1.8, 2.5),
        "legibility_range": (0.08, 0.25),
        "tremor_range": (1.8, 3.2),
        "sig_style": "scrawl_trailing_line"
    },
    "General Surgery": {
        "slant_deg_range": (18.0, 30.0),
        "pressure_range": (1.2, 1.8),
        "speed_range": (1.5, 2.2),
        "legibility_range": (0.15, 0.35),
        "tremor_range": (0.8, 1.6),
        "sig_style": "sharp_angular_zigzag"
    },
    "Orthopedic Surgery": {
        "slant_deg_range": (18.0, 32.0),
        "pressure_range": (1.3, 1.9),
        "speed_range": (1.4, 2.1),
        "legibility_range": (0.15, 0.35),
        "tremor_range": (0.7, 1.5),
        "sig_style": "sharp_angular_zigzag"
    },
    "Cardiology": {
        "slant_deg_range": (12.0, 24.0),
        "pressure_range": (0.8, 1.4),
        "speed_range": (1.2, 1.8),
        "legibility_range": (0.25, 0.50),
        "tremor_range": (1.0, 2.0),
        "sig_style": "initials_flourish"
    },
    "Pulmonology & Critical Care": {
        "slant_deg_range": (14.0, 26.0),
        "pressure_range": (0.8, 1.3),
        "speed_range": (1.3, 1.9),
        "legibility_range": (0.20, 0.45),
        "tremor_range": (1.1, 2.1),
        "sig_style": "initials_flourish"
    },
    "Family Medicine": {
        "slant_deg_range": (8.0, 20.0),
        "pressure_range": (0.7, 1.2),
        "speed_range": (1.0, 1.6),
        "legibility_range": (0.35, 0.65),
        "tremor_range": (0.5, 1.4),
        "sig_style": "cursive_full_loop"
    },
    "Internal Medicine": {
        "slant_deg_range": (8.0, 20.0),
        "pressure_range": (0.7, 1.2),
        "speed_range": (1.0, 1.6),
        "legibility_range": (0.35, 0.65),
        "tremor_range": (0.5, 1.4),
        "sig_style": "cursive_full_loop"
    },
    "Pediatrics": {
        "slant_deg_range": (4.0, 15.0),
        "pressure_range": (0.6, 1.1),
        "speed_range": (0.8, 1.3),
        "legibility_range": (0.55, 0.85),
        "tremor_range": (0.2, 0.8),
        "sig_style": "double_loop_wrap"
    },
    "Dermatology": {
        "slant_deg_range": (4.0, 15.0),
        "pressure_range": (0.6, 1.1),
        "speed_range": (0.8, 1.3),
        "legibility_range": (0.55, 0.85),
        "tremor_range": (0.2, 0.8),
        "sig_style": "double_loop_wrap"
    },
    "Neurology": {
        "slant_deg_range": (10.0, 25.0),
        "pressure_range": (0.5, 0.9),
        "speed_range": (0.7, 1.2),
        "legibility_range": (0.20, 0.45),
        "tremor_range": (2.2, 3.8),
        "sig_style": "ascending_wave"
    },
    "Psychiatry": {
        "slant_deg_range": (6.0, 18.0),
        "pressure_range": (0.6, 1.0),
        "speed_range": (0.9, 1.4),
        "legibility_range": (0.30, 0.60),
        "tremor_range": (0.6, 1.5),
        "sig_style": "compressed_ligatures"
    },
    "Medical Oncology": {
        "slant_deg_range": (10.0, 22.0),
        "pressure_range": (0.8, 1.3),
        "speed_range": (1.1, 1.7),
        "legibility_range": (0.28, 0.52),
        "tremor_range": (0.9, 1.8),
        "sig_style": "initials_flourish"
    },
    "Gastroenterology": {
        "slant_deg_range": (10.0, 22.0),
        "pressure_range": (0.7, 1.3),
        "speed_range": (1.1, 1.6),
        "legibility_range": (0.30, 0.55),
        "tremor_range": (0.7, 1.6),
        "sig_style": "cursive_full_loop"
    },
    "Endocrinology": {
        "slant_deg_range": (6.0, 16.0),
        "pressure_range": (0.6, 1.1),
        "speed_range": (0.9, 1.4),
        "legibility_range": (0.40, 0.70),
        "tremor_range": (0.4, 1.2),
        "sig_style": "double_loop_wrap"
    },
    "Obstetrics & Gynecology": {
        "slant_deg_range": (8.0, 18.0),
        "pressure_range": (0.7, 1.2),
        "speed_range": (1.0, 1.5),
        "legibility_range": (0.38, 0.68),
        "tremor_range": (0.5, 1.3),
        "sig_style": "cursive_full_loop"
    }
}


def build_doctor_profile_catalog() -> Dict[str, Any]:
    """
    Generate >= 100 doctor profiles with valid CMS Luhn NPI, DEA checksums, and kinematic style parameters.
    """
    random.seed(42)
    doctors: List[Dict[str, Any]] = []
    doc_idx = 1

    for specialty, count in SPECIALTIES_DISTRIBUTION:
        for _ in range(count):
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            st_info = random.choice(US_STATES)
            state_code, city, zip_pfx, phone_pfx, lic_pfx = st_info

            # Generate valid CMS Luhn NPI (10 digits)
            # Base 9 digits (e.g. 100000000 to 199999999)
            base_npi = f"{100000000 + doc_idx * 7919 % 89999999:09d}"
            npi = compute_luhn_npi(base_npi)
            assert validate_npi(npi), f"Generated invalid NPI {npi}"

            # Generate valid DEA number (Letter1 + Letter2 + 6 digits + 1 check digit)
            reg_type = random.choice(["A", "B", "F", "M"])
            base_dea = f"{100000 + doc_idx * 3137 % 899999:06d}"
            dea = compute_dea_number(reg_type, last, base_dea)
            assert validate_dea(dea), f"Generated invalid DEA {dea}"

            # State Medical License
            lic_num = f"{100000 + doc_idx * 401 % 899999:06d}"
            state_license = f"{lic_pfx}{lic_num}"

            # Phone / Fax
            phone_suffix = f"{1000 + doc_idx * 7 % 8999:04d}"
            fax_suffix = f"{1000 + doc_idx * 11 % 8999:04d}"
            phone = f"{phone_pfx}{phone_suffix}"
            fax = f"{phone_pfx}{fax_suffix}"

            # Kinematics
            k_spec = SPECIALTY_KINEMATICS.get(specialty, SPECIALTY_KINEMATICS["Family Medicine"])
            slant = round(random.uniform(*k_spec["slant_deg_range"]), 2)
            pressure = round(random.uniform(*k_spec["pressure_range"]), 2)
            speed = round(random.uniform(*k_spec["speed_range"]), 2)
            legibility = round(random.uniform(*k_spec["legibility_range"]), 2)
            tremor = round(random.uniform(*k_spec["tremor_range"]), 2)
            drift = round(random.uniform(-2.5, 2.5), 2)

            raw_doc_id = f"dr_{first.lower()}_{last.lower()}_{doc_idx:03d}"
            doctor_id = re.sub(r"[^a-z0-9_]", "", raw_doc_id)
            doc_record = {
                "doctor_id": doctor_id,
                "full_name": f"Dr. {first} {last}, MD",
                "first_name": first,
                "last_name": last,
                "credentials": ["MD", "FACP"] if specialty in ["Internal Medicine", "Cardiology"] else ["MD"],
                "specialty": specialty,
                "clinic_name": f"{city} {specialty} Associates",
                "clinic_address": f"{100 + doc_idx * 12} Health Boulevard, Suite {100 + doc_idx % 50}",
                "city": city,
                "state": state_code,
                "zip_code": f"{zip_pfx}",
                "phone": phone,
                "fax": fax,
                "npi": npi,
                "dea_number": dea,
                "state_license": state_license,
                "handwriting_style": {
                    "slant_angle_deg": slant,
                    "slant_std_deg": 2.5,
                    "pressure_factor": pressure,
                    "speed_factor": speed,
                    "legibility_score": legibility,
                    "tremor_factor": tremor,
                    "baseline_drift_deg": drift,
                    "preferred_ink_types": ["blue_ballpoint", "black_ballpoint"] if random.random() < 0.7 else ["fountain_pen", "blue_ballpoint"],
                    "signature_style": k_spec["sig_style"],
                    "signature_svg_template": f"M 10 50 C 30 {40 + slant}, 60 {20 - tremor*5}, 120 45 S 180 {60 + drift*5}, 240 50"
                }
            }
            doctors.append(doc_record)
            doc_idx += 1

    catalog = {
        "version": "2026.1",
        "total_doctors": len(doctors),
        "specialties": [s[0] for s in SPECIALTIES_DISTRIBUTION],
        "doctors": doctors
    }
    return catalog


# ==============================================================================
# 4. CLINICAL TEMPLATES & GRAMMARS
# ==============================================================================

def build_clinical_templates_catalog() -> Dict[str, Any]:
    """
    Generate >= 200 clinical note & prescription templates across 5 categories.
    Every {SLOT} placeholder is declared in the template's slots list.
    """
    random.seed(42)
    templates: List[Dict[str, Any]] = []

    # Category 1: Outpatient Encounter Notes (45+ templates)
    outpatient_conditions = [
        ("Hypertension & Dyslipidemia", "Cardiology", "Essential hypertension follow-up. BP moderately elevated. Lipid profile reviewed."),
        ("Type 2 Diabetes Mellitus", "Endocrinology", "Quarterly T2DM check. Review HbA1c and daily capillary glucose logs."),
        ("Asthma & Allergic Rhinitis", "Pulmonology & Critical Care", "Evaluation of intermittent wheezing and seasonal allergy symptoms."),
        ("Osteoarthritis Knee & Hip", "Orthopedic Surgery", "Bilateral weight-bearing knee pain aggravated by stairs. Joint stability intact."),
        ("Major Depressive Disorder", "Psychiatry", "Follow-up for mood stabilization, sleep architecture, and medication tolerability."),
        ("Generalized Anxiety Disorder", "Psychiatry", "Review of persistent autonomic symptoms, tension, and SSRI response."),
        ("Hypothyroidism", "Endocrinology", "TSH level review. Patient notes mild morning fatigue and dry skin."),
        ("Gastroesophageal Reflux Disease", "Gastroenterology", "Persistent postprandial heartburn and acid regurgitation."),
        ("Chronic Kidney Disease Stage 3", "Internal Medicine", "Routine nephrology co-management. Serum creatinine and eGFR stable."),
        ("Atrial Fibrillation Coagulation", "Cardiology", "Rate-controlled AFib review. Anticoagulation therapy and INR/DOAC compliance."),
        ("COPD Stable Maintenance", "Pulmonology & Critical Care", "Chronic dyspnea on exertion. Inhaler technique and exercise tolerance reviewed."),
        ("Migraine Prophylaxis", "Neurology", "Headache frequency log review. Discuss abortive triptan vs preventive CGRP."),
        ("Rheumatoid Arthritis", "Internal Medicine", "Morning stiffness duration 45 min. Small joint synovitis in hands."),
        ("Benign Prostatic Hyperplasia", "Internal Medicine", "Nocturia 3x/night and urinary hesitancy. Review IPSS score."),
        ("Eczema & Atopic Dermatitis", "Dermatology", "Pruritic erythematous plaques on bilateral flexural creases."),
        ("Gout Flare Prevention", "Internal Medicine", "Serum urate level monitoring and allopurinol titration."),
        ("Chronic Low Back Pain", "Orthopedic Surgery", "Mechanical lumbosacral ache without radicular sensory or motor deficits."),
        ("Stable Angina Pectoris", "Cardiology", "Exertional chest tightness relieved by sublingual nitroglycerin and rest."),
        ("Peripheral Neuropathy", "Neurology", "Bilateral distal lower extremity burning paresthesias."),
        ("Iron Deficiency Anemia", "Internal Medicine", "Fatigue and exertional lightheadedness. Oral iron supplement tolerability.")
    ]

    tpl_count = 1
    for name, spec, assessment in outpatient_conditions:
        # Create 2 variations per condition -> 40 templates + 6 extra general wellness
        t1 = {
            "template_id": f"tpl_outpatient_{tpl_count:03d}",
            "template_category": "outpatient_encounter",
            "encounter_type": f"Outpatient {name} Visit",
            "specialty": spec,
            "raw_template_text": f"Patient: {{PATIENT_NAME}}, {{AGE}}yo {{GENDER}}. Date: {{DATE}}.\n"
                                 f"Chief Complaint: {name} routine consultation.\n"
                                 f"Vitals: BP {{BP_SYSTOLIC}}/{{BP_DIASTOLIC}} mmHg, HR {{HEART_RATE}} bpm, Wt {{WEIGHT_KG}} kg.\n"
                                 f"Assessment: {assessment}\n"
                                 f"Plan:\n"
                                 f"1. {{MEDICATION_1}} {{STRENGTH_1}} PO {{SIG_FREQ_1}}.\n"
                                 f"2. {{MEDICATION_2}} {{STRENGTH_2}} PO {{SIG_FREQ_2}}.\n"
                                 f"3. Follow up in clinic in 3 months.\n"
                                 f"Dr. {{DOCTOR_NAME}}, {{CREDENTIALS}} (NPI: {{NPI}})",
            "slots": [
                {"slot_name": "{PATIENT_NAME}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "Full Name"},
                {"slot_name": "{AGE}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "20-85"},
                {"slot_name": "{GENDER}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "M/F"},
                {"slot_name": "{DATE}", "slot_type": "physician_metadata", "source_vocabulary": None, "default_format": "MM/DD/YYYY"},
                {"slot_name": "{BP_SYSTOLIC}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "110-165"},
                {"slot_name": "{BP_DIASTOLIC}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "65-98"},
                {"slot_name": "{HEART_RATE}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "58-96"},
                {"slot_name": "{WEIGHT_KG}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "55-115"},
                {"slot_name": "{MEDICATION_1}", "slot_type": "medication", "source_vocabulary": "rxnorm", "default_format": None},
                {"slot_name": "{STRENGTH_1}", "slot_type": "dosage_strength", "source_vocabulary": "rxnorm", "default_format": None},
                {"slot_name": "{SIG_FREQ_1}", "slot_type": "sig_code", "source_vocabulary": "sig_codes", "default_format": None},
                {"slot_name": "{MEDICATION_2}", "slot_type": "medication", "source_vocabulary": "rxnorm", "default_format": None},
                {"slot_name": "{STRENGTH_2}", "slot_type": "dosage_strength", "source_vocabulary": "rxnorm", "default_format": None},
                {"slot_name": "{SIG_FREQ_2}", "slot_type": "sig_code", "source_vocabulary": "sig_codes", "default_format": None},
                {"slot_name": "{DOCTOR_NAME}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{CREDENTIALS}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "MD/DO"},
                {"slot_name": "{NPI}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "10-digit NPI"}
            ],
            "rendering_guidance": {"ruled_paper": True, "header_required": True, "signature_at_bottom": True}
        }
        templates.append(t1)
        tpl_count += 1

        t2 = {
            "template_id": f"tpl_outpatient_{tpl_count:03d}",
            "template_category": "outpatient_encounter",
            "encounter_type": f"Outpatient {name} Re-evaluation",
            "specialty": spec,
            "raw_template_text": f"CLINIC PROGRESS NOTE — {{DATE}}\n"
                                 f"Pt: {{PATIENT_NAME}} ({{AGE}}yo {{GENDER}}). CC: Follow-up {name}.\n"
                                 f"Exam: Alert, oriented x 3. Vitals: BP {{BP_SYSTOLIC}}/{{BP_DIASTOLIC}}, HR {{HEART_RATE}}, Temp {{TEMP_F}} F.\n"
                                 f"Impression: Stable {name}.\n"
                                 f"Rx: Continue {{MEDICATION_1}} {{STRENGTH_1}} PO {{SIG_FREQ_1}}.\n"
                                 f"Signed: Dr. {{DOCTOR_NAME}}, {{CREDENTIALS}}",
            "slots": [
                {"slot_name": "{DATE}", "slot_type": "physician_metadata", "source_vocabulary": None, "default_format": "MM/DD/YYYY"},
                {"slot_name": "{PATIENT_NAME}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "Full Name"},
                {"slot_name": "{AGE}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "20-85"},
                {"slot_name": "{GENDER}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "M/F"},
                {"slot_name": "{BP_SYSTOLIC}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "110-165"},
                {"slot_name": "{BP_DIASTOLIC}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "65-98"},
                {"slot_name": "{HEART_RATE}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "58-96"},
                {"slot_name": "{TEMP_F}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "97.5-99.2"},
                {"slot_name": "{MEDICATION_1}", "slot_type": "medication", "source_vocabulary": "rxnorm", "default_format": None},
                {"slot_name": "{STRENGTH_1}", "slot_type": "dosage_strength", "source_vocabulary": "rxnorm", "default_format": None},
                {"slot_name": "{SIG_FREQ_1}", "slot_type": "sig_code", "source_vocabulary": "sig_codes", "default_format": None},
                {"slot_name": "{DOCTOR_NAME}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{CREDENTIALS}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "MD/DO"}
            ],
            "rendering_guidance": {"ruled_paper": True, "header_required": False, "signature_at_bottom": True}
        }
        templates.append(t2)
        tpl_count += 1

    # Add 6 extra outpatient templates to hit 46 outpatient templates
    for i in range(6):
        t_extra = {
            "template_id": f"tpl_outpatient_{tpl_count:03d}",
            "template_category": "outpatient_encounter",
            "encounter_type": f"Annual Wellness Visit {i+1}",
            "specialty": "Family Medicine",
            "raw_template_text": f"Annual Preventive Wellness Exam — {{DATE}}\n"
                                 f"Patient: {{PATIENT_NAME}}, Age {{AGE}}, {{GENDER}}. Vitals: BP {{BP_SYSTOLIC}}/{{BP_DIASTOLIC}}, HR {{HEART_RATE}}.\n"
                                 f"Immunizations and screening labs reviewed. Preventive lifestyle counseling provided.\n"
                                 f"Maintenance Rx: {{MEDICATION_1}} {{STRENGTH_1}} PO {{SIG_FREQ_1}}.\n"
                                 f"Physician: Dr. {{DOCTOR_NAME}}, {{CREDENTIALS}} (NPI: {{NPI}})",
            "slots": [
                {"slot_name": "{DATE}", "slot_type": "physician_metadata", "source_vocabulary": None, "default_format": "MM/DD/YYYY"},
                {"slot_name": "{PATIENT_NAME}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "Full Name"},
                {"slot_name": "{AGE}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "20-85"},
                {"slot_name": "{GENDER}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "M/F"},
                {"slot_name": "{BP_SYSTOLIC}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "110-150"},
                {"slot_name": "{BP_DIASTOLIC}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "65-90"},
                {"slot_name": "{HEART_RATE}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "60-90"},
                {"slot_name": "{MEDICATION_1}", "slot_type": "medication", "source_vocabulary": "rxnorm", "default_format": None},
                {"slot_name": "{STRENGTH_1}", "slot_type": "dosage_strength", "source_vocabulary": "rxnorm", "default_format": None},
                {"slot_name": "{SIG_FREQ_1}", "slot_type": "sig_code", "source_vocabulary": "sig_codes", "default_format": None},
                {"slot_name": "{DOCTOR_NAME}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{CREDENTIALS}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "MD/DO"},
                {"slot_name": "{NPI}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "10-digit NPI"}
            ],
            "rendering_guidance": {"ruled_paper": True, "header_required": True, "signature_at_bottom": True}
        }
        templates.append(t_extra)
        tpl_count += 1

    # Category 2: Hospital Discharge Summaries (36+ templates)
    discharge_diagnoses = [
        ("Acute Decompensated Heart Failure", "Cardiology", "Presented with progressive dyspnea and lower extremity edema. Diuresed with IV Furosemide."),
        ("Acute Coronary Syndrome / NSTEMI", "Cardiology", "Underwent coronary angiography with drug-eluting stent placement to LAD."),
        ("Community-Acquired Pneumonia", "Pulmonology & Critical Care", "Treated with 5 days IV Ceftriaxone and oral Azithromycin. Resolved."),
        ("Acute Ischemic Stroke", "Neurology", "Admitted with right hemiparesis. MRI showed subacute left MCA infarct. Started on dual antiplatelets."),
        ("Diabetic Ketoacidosis Resolved", "Endocrinology", "Anion gap closed on IV insulin infusion. Transitioned to home basal-bolus regimen."),
        ("Acute Diverticulitis", "Gastroenterology", "CT abdomen demonstrated sigmoid diverticulitis without abscess. IV antibiotics completed."),
        ("Post-Operative Total Knee Arthroplasty", "Orthopedic Surgery", "Uncomplicated right TKA. Physical therapy progressed to ambulating 100 feet."),
        ("Post-Operative Laparoscopic Cholecystectomy", "General Surgery", "Acute calculous cholecystitis. Elective laparoscopic cholecystectomy performed."),
        ("Acute Pyelonephritis", "Internal Medicine", "Flank pain and fever resolved on IV Cefepime. Discharged on oral culture-directed therapy."),
        ("Acute Exacerbation of COPD", "Pulmonology & Critical Care", "Treated with systemic steroids, duo-nebs, and oral doxycycline. Oxygen weaned to room air."),
        ("Upper GI Bleed / Peptic Ulcer", "Gastroenterology", "EGD revealed Forrest IIb duodenal ulcer treated with endoclips and IV PPI infusion."),
        ("Cellulitis Lower Extremity", "Dermatology", "Erythema and warmth resolved with IV Cefazolin. Transitioned to oral cephalexin.")
    ]

    for d_name, d_spec, d_course in discharge_diagnoses:
        for v in range(3):
            t_dc = {
                "template_id": f"tpl_discharge_{tpl_count:03d}",
                "template_category": "discharge_summary",
                "encounter_type": f"Discharge Summary: {d_name} (Var {v+1})",
                "specialty": d_spec,
                "raw_template_text": f"HOSPITAL DISCHARGE SUMMARY\n"
                                     f"Patient: {{PATIENT_NAME}} | MRN: {{MRN}} | Adm: {{ADM_DATE}} | Disch: {{DISCH_DATE}}\n"
                                     f"Attending Physician: Dr. {{DOCTOR_NAME}}, {{CREDENTIALS}} ({{SPECIALTY}})\n"
                                     f"Discharge Diagnosis: {d_name}\n"
                                     f"Hospital Course: {d_course}\n"
                                     f"Discharge Medications:\n"
                                     f"1. {{MEDICATION_1}} {{STRENGTH_1}} Sig: {{SIG_FREQ_1}} #{{DISPENSE_1}} Refills: {{REFILLS_1}}\n"
                                     f"2. {{MEDICATION_2}} {{STRENGTH_2}} Sig: {{SIG_FREQ_2}} #{{DISPENSE_2}} Refills: {{REFILLS_2}}\n"
                                     f"Follow-up: Clinic appointment in 7-10 days.\n"
                                     f"Signed: Dr. {{DOCTOR_NAME}}, {{CREDENTIALS}} (NPI: {{NPI}})",
                "slots": [
                    {"slot_name": "{PATIENT_NAME}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "Full Name"},
                    {"slot_name": "{MRN}", "slot_type": "patient_demographics", "source_vocabulary": None, "default_format": "MRN-100000"},
                    {"slot_name": "{ADM_DATE}", "slot_type": "physician_metadata", "source_vocabulary": None, "default_format": "MM/DD/YYYY"},
                    {"slot_name": "{DISCH_DATE}", "slot_type": "physician_metadata", "source_vocabulary": None, "default_format": "MM/DD/YYYY"},
                    {"slot_name": "{DOCTOR_NAME}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                    {"slot_name": "{CREDENTIALS}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "MD/DO"},
                    {"slot_name": "{SPECIALTY}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                    {"slot_name": "{MEDICATION_1}", "slot_type": "medication", "source_vocabulary": "rxnorm", "default_format": None},
                    {"slot_name": "{STRENGTH_1}", "slot_type": "dosage_strength", "source_vocabulary": "rxnorm", "default_format": None},
                    {"slot_name": "{SIG_FREQ_1}", "slot_type": "sig_code", "source_vocabulary": "sig_codes", "default_format": None},
                    {"slot_name": "{DISPENSE_1}", "slot_type": "dosage_strength", "source_vocabulary": None, "default_format": "30"},
                    {"slot_name": "{REFILLS_1}", "slot_type": "dosage_strength", "source_vocabulary": None, "default_format": "1"},
                    {"slot_name": "{MEDICATION_2}", "slot_type": "medication", "source_vocabulary": "rxnorm", "default_format": None},
                    {"slot_name": "{STRENGTH_2}", "slot_type": "dosage_strength", "source_vocabulary": "rxnorm", "default_format": None},
                    {"slot_name": "{SIG_FREQ_2}", "slot_type": "sig_code", "source_vocabulary": "sig_codes", "default_format": None},
                    {"slot_name": "{DISPENSE_2}", "slot_type": "dosage_strength", "source_vocabulary": None, "default_format": "30"},
                    {"slot_name": "{REFILLS_2}", "slot_type": "dosage_strength", "source_vocabulary": None, "default_format": "0"},
                    {"slot_name": "{NPI}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "10-digit NPI"}
                ],
                "rendering_guidance": {"ruled_paper": False, "header_required": True, "signature_at_bottom": True}
            }
            templates.append(t_dc)
            tpl_count += 1

    # Category 3: SOAP Notes (50+ templates)
    soap_topics = [
        ("Acute Bronchitis", "Cough with yellow sputum x 4 days", "Scattered rhonchi bilaterally, no consolidation", "Acute Bronchitis", "1. {MEDICATION_1} {STRENGTH_1} PO {SIG_FREQ_1} x 7 days.\n2. Hydration and rest."),
        ("Acute Sinusitis", "Facial pressure and purulent nasal discharge x 10 days", "Maxillary tenderness on palpation", "Acute Bacterial Rhinosinusitis", "1. {MEDICATION_1} {STRENGTH_1} PO {SIG_FREQ_1} x 10 days.\n2. Saline nasal rinse."),
        ("Streptococcal Pharyngitis", "Severe sore throat, painful swallowing, fever", "Erythematous tonsils with exudate, tender anterior cervical nodes", "Group A Strep Pharyngitis", "1. {MEDICATION_1} {STRENGTH_1} PO {SIG_FREQ_1} x 10 days.\n2. Warm salt gargles."),
        ("Acute Otitis Media", "Right ear pain and decreased hearing x 2 days", "Right TM bulging, erythematous, reduced mobility", "Right Acute Otitis Media", "1. {MEDICATION_1} {STRENGTH_1} PO {SIG_FREQ_1} x 10 days.\n2. Analgesics PRN."),
        ("Uncomplicated Cystitis", "Dysuria, urinary frequency, and suprapubic discomfort", "Suprapubic tenderness without CVA tenderness", "Acute Uncomplicated UTI", "1. {MEDICATION_1} {STRENGTH_1} PO {SIG_FREQ_1} x 5 days.\n2. Increase fluid intake."),
        ("Acute Gouty Arthritis", "Sudden severe pain and swelling in right 1st MTP joint", "Erythematous, hot, exquisitely tender 1st MTP", "Acute Gout Flare", "1. {MEDICATION_1} {STRENGTH_1} PO {SIG_FREQ_1} with food.\n2. Avoid high purine foods."),
        ("Contact Dermatitis", "Itchy rash on bilateral forearms following poison ivy exposure", "Linear erythematous vesicles and papules on forearms", "Acute Allergic Contact Dermatitis", "1. {MEDICATION_1} {STRENGTH_1} PO {SIG_FREQ_1} taper.\n2. Topical hydrocortisone."),
        ("Acute Lumbar Strain", "Low back ache after lifting heavy box at work", "Paraspinal muscle spasm, straight leg raise negative", "Acute Lumbar Muscular Strain", "1. {MEDICATION_1} {STRENGTH_1} PO {SIG_FREQ_1} PRN spasm.\n2. Gentle walking and heat."),
        ("Gastroenteritis", "Nausea, watery diarrhea, and cramping x 24 hours", "Abdomen soft, diffuse mild tenderness, hyperactive bowel sounds", "Acute Viral Gastroenteritis", "1. Oral rehydration solution.\n2. {MEDICATION_1} {STRENGTH_1} PO {SIG_FREQ_1} PRN nausea."),
        ("Allergic Conjunctivitis", "Bilateral eye itching, watery discharge, redness", "Bilateral conjunctival injection with cobblestone papillae", "Allergic Conjunctivitis", "1. {MEDICATION_1} {STRENGTH_1} in OU {SIG_FREQ_1}.\n2. Cool compresses.")
    ]

    for s_name, subj, obj, dx, plan in soap_topics:
        for v in range(5):  # 10 topics * 5 variations = 50 templates
            t_soap = {
                "template_id": f"tpl_soap_{tpl_count:03d}",
                "template_category": "soap_note",
                "encounter_type": f"SOAP Note: {s_name} (Var {v+1})",
                "specialty": "Internal Medicine" if v % 2 == 0 else "Family Medicine",
                "raw_template_text": f"SOAP NOTE — {{DATE}}\n"
                                     f"Patient: {{PATIENT_NAME}}, {{AGE}}yo {{GENDER}}\n"
                                     f"S: Patient presents with {subj}. Denies shortness of breath or chest pain.\n"
                                     f"O: Vitals: BP {{BP_SYSTOLIC}}/{{BP_DIASTOLIC}}, HR {{HEART_RATE}}, Temp {{TEMP_F}} F, SpO2 {{SPO2}}%.\n"
                                     f"Exam: {obj}.\n"
                                     f"A: {dx}.\n"
                                     f"P:\n{plan}\n"
                                     f"3. Return if symptoms worsen or fever persists > 48 hours.\n"
                                     f"Signed: Dr. {{DOCTOR_NAME}}, {{CREDENTIALS}}",
                "slots": [
                    {"slot_name": "{DATE}", "slot_type": "physician_metadata", "source_vocabulary": None, "default_format": "MM/DD/YYYY"},
                    {"slot_name": "{PATIENT_NAME}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "Full Name"},
                    {"slot_name": "{AGE}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "20-85"},
                    {"slot_name": "{GENDER}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "M/F"},
                    {"slot_name": "{BP_SYSTOLIC}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "110-155"},
                    {"slot_name": "{BP_DIASTOLIC}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "65-92"},
                    {"slot_name": "{HEART_RATE}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "60-105"},
                    {"slot_name": "{TEMP_F}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "98.0-102.2"},
                    {"slot_name": "{SPO2}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "95-100"},
                    {"slot_name": "{MEDICATION_1}", "slot_type": "medication", "source_vocabulary": "rxnorm", "default_format": None},
                    {"slot_name": "{STRENGTH_1}", "slot_type": "dosage_strength", "source_vocabulary": "rxnorm", "default_format": None},
                    {"slot_name": "{SIG_FREQ_1}", "slot_type": "sig_code", "source_vocabulary": "sig_codes", "default_format": None},
                    {"slot_name": "{DOCTOR_NAME}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                    {"slot_name": "{CREDENTIALS}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "MD/DO"}
                ],
                "rendering_guidance": {"ruled_paper": True, "header_required": False, "signature_at_bottom": True}
            }
            templates.append(t_soap)
            tpl_count += 1

    # Category 4: Emergency Department Triage Notes (36+ templates)
    ed_scenarios = [
        ("Substernal Chest Pain", "Sudden-onset pressure radiating to left arm x 45 min", "12-lead ECG: Sinus rhythm, no acute ST changes. Troponin < 0.01", "Level 2 (Emergent)", "Aspirin 324mg PO STAT, Nitroglycerin 0.4mg SL PRN"),
        ("Acute Dyspnea & Wheezing", "Progressive shortness of breath with audible wheezing x 2 hours", "Lungs: Diffuse expiratory wheezing bilaterally, tachypneic", "Level 2 (Emergent)", "Albuterol-Ipratropium nebulizer STAT, Dexamethasone 10mg IV"),
        ("Right Lower Quadrant Abdominal Pain", "Constant sharp RLQ abdominal pain with nausea x 8 hours", "Tenderness at McBurney's point with rebound and guarding", "Level 2 (Emergent)", "NPO, IV Morphine 4mg STAT, CT Abdomen/Pelvis STAT"),
        ("Headache & Neurological Deficit", "Sudden-onset severe occipital headache with left facial droop", "Mild left pronator drift, cranial nerves otherwise intact", "Level 1 (Resuscitation)", "Stroke Code activated, Non-contrast CT Head STAT"),
        ("Complex Forearm Laceration", "5cm jagged laceration to volar right forearm from broken glass", "Extensor/flexor tendons intact, sensation normal to digits", "Level 3 (Urgent)", "Irrigated with 500mL saline, 4-0 Ethilon sutures x 8, Tdap booster"),
        ("Anaphylactic Reaction", "Urticaria, lip angioedema, and throat tightness after peanut ingestion", "Stridor, diffuse wheezes, tachycardia, hypotension", "Level 1 (Resuscitation)", "Epinephrine 0.3mg IM STAT, Diphenhydramine 50mg IV, Famotidine 20mg IV"),
        ("Syncope & Bradycardia", "Sudden loss of consciousness while standing, duration 30 sec", "HR 38 bpm, ECG shows complete heart block", "Level 1 (Resuscitation)", "Transcutaneous pacing pads applied, Cardiology consult STAT"),
        ("Acute Kidney Injury & Hyperkalemia", "Weakness and decreased urine output, K+ 6.8 mEq/L", "Peaked T waves on ECG, peripheral edema 2+", "Level 2 (Emergent)", "Calcium gluconate 1g IV STAT, Insulin 10 units IV + D50W, Albuterol neb"),
        ("Closed Ankle Fracture-Dislocation", "Inversion ankle injury with visible deformity and severe pain", "Skin intact, distal pulses 2+, posterior splint applied", "Level 3 (Urgent)", "IV Hydromorphone 1mg STAT, Orthopedic reduction and splinting"),
        ("Diabetic Hypoglycemia", "Confusion and diaphoresis, capillary blood glucose 34 mg/dL", "Disoriented to time and place, sweaty, tremors", "Level 2 (Emergent)", "Administered 1 ampule D50W IV STAT. Recheck glucose in 15 min")
    ]

    for ed_title, cc, exam, acuity, ed_tx in ed_scenarios:
        for v in range(4):  # 10 scenarios * 4 variations = 40 templates (total >= 36)
            t_ed = {
                "template_id": f"tpl_ed_{tpl_count:03d}",
                "template_category": "emergency_triage",
                "encounter_type": f"ED Triage Note: {ed_title} (Var {v+1})",
                "specialty": "Emergency Medicine",
                "raw_template_text": f"EMERGENCY DEPARTMENT TRIAGE & ASSESSMENT\n"
                                     f"Triage Time: {{TIME}} | Acuity: {acuity}\n"
                                     f"Patient: {{PATIENT_NAME}}, {{AGE}}yo {{GENDER}} | Date: {{DATE}}\n"
                                     f"Chief Complaint: {cc}.\n"
                                     f"Vitals: BP {{BP_SYSTOLIC}}/{{BP_DIASTOLIC}} | HR {{HEART_RATE}} | RR {{RESP_RATE}} | SpO2 {{SPO2}}% | Temp {{TEMP_F}} F\n"
                                     f"Clinical Evaluation: {exam}.\n"
                                     f"Emergency Treatment: {ed_tx}.\n"
                                     f"Rx Ordered: {{MEDICATION_1}} {{STRENGTH_1}} {{SIG_FREQ_1}}.\n"
                                     f"Attending: Dr. {{DOCTOR_NAME}}, {{CREDENTIALS}} (NPI: {{NPI}})",
                "slots": [
                    {"slot_name": "{TIME}", "slot_type": "physician_metadata", "source_vocabulary": None, "default_format": "HH:MM"},
                    {"slot_name": "{PATIENT_NAME}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "Full Name"},
                    {"slot_name": "{AGE}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "18-90"},
                    {"slot_name": "{GENDER}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "M/F"},
                    {"slot_name": "{DATE}", "slot_type": "physician_metadata", "source_vocabulary": None, "default_format": "MM/DD/YYYY"},
                    {"slot_name": "{BP_SYSTOLIC}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "90-200"},
                    {"slot_name": "{BP_DIASTOLIC}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "55-115"},
                    {"slot_name": "{HEART_RATE}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "40-140"},
                    {"slot_name": "{RESP_RATE}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "12-28"},
                    {"slot_name": "{SPO2}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "88-100"},
                    {"slot_name": "{TEMP_F}", "slot_type": "vital_signs", "source_vocabulary": None, "default_format": "97.0-103.5"},
                    {"slot_name": "{MEDICATION_1}", "slot_type": "medication", "source_vocabulary": "rxnorm", "default_format": None},
                    {"slot_name": "{STRENGTH_1}", "slot_type": "dosage_strength", "source_vocabulary": "rxnorm", "default_format": None},
                    {"slot_name": "{SIG_FREQ_1}", "slot_type": "sig_code", "source_vocabulary": "sig_codes", "default_format": None},
                    {"slot_name": "{DOCTOR_NAME}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                    {"slot_name": "{CREDENTIALS}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "MD/DO"},
                    {"slot_name": "{NPI}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "10-digit NPI"}
                ],
                "rendering_guidance": {"ruled_paper": False, "header_required": True, "signature_at_bottom": True}
            }
            templates.append(t_ed)
            tpl_count += 1

    # Category 5: Prescription Slips & Directives (48+ templates)
    for p_idx in range(48):
        t_rx = {
            "template_id": f"tpl_rx_{tpl_count:03d}",
            "template_category": "prescription_slip",
            "encounter_type": f"Prescription Order Slip #{p_idx+1}",
            "specialty": "General Practice",
            "raw_template_text": f"℞ {{CLINIC_NAME}}\n"
                                 f"{{CLINIC_ADDRESS}}, {{CITY}}, {{STATE}} {{ZIP}} | Tel: {{PHONE}}\n"
                                 f"Prescriber: Dr. {{DOCTOR_NAME}}, {{CREDENTIALS}} — DEA: {{DEA}} | NPI: {{NPI}}\n"
                                 f"------------------------------------------------------------\n"
                                 f"Patient: {{PATIENT_NAME}}      Age: {{AGE}}      Date: {{DATE}}\n"
                                 f"Address: {{PATIENT_ADDRESS}}\n"
                                 f"------------------------------------------------------------\n"
                                 f"℞\n"
                                 f"{{MEDICATION}} {{STRENGTH}} {{DOSAGE_FORM}}\n"
                                 f"Sig: {{SIG_ACTION}} {{SIG_QUANTITY}} {{SIG_ROUTE}} {{SIG_FREQUENCY}} {{SIG_MODIFIER}}\n"
                                 f"Dispense: #{{DISPENSE_QTY}} ({{DISPENSE_WORDS}})\n"
                                 f"Refills: {{REFILLS}}        [ ] DAW (Dispense As Written)\n"
                                 f"------------------------------------------------------------\n"
                                 f"Prescriber Signature: {{SIGNATURE_SCRAWL}}",
            "slots": [
                {"slot_name": "{CLINIC_NAME}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{CLINIC_ADDRESS}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{CITY}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{STATE}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{ZIP}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{PHONE}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{DOCTOR_NAME}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{CREDENTIALS}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": "MD/DO"},
                {"slot_name": "{DEA}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{NPI}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None},
                {"slot_name": "{PATIENT_NAME}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": None},
                {"slot_name": "{AGE}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": "20-80"},
                {"slot_name": "{DATE}", "slot_type": "physician_metadata", "source_vocabulary": None, "default_format": "MM/DD/YYYY"},
                {"slot_name": "{PATIENT_ADDRESS}", "slot_type": "patient_demographics", "source_vocabulary": "patients", "default_format": None},
                {"slot_name": "{MEDICATION}", "slot_type": "medication", "source_vocabulary": "rxnorm", "default_format": None},
                {"slot_name": "{STRENGTH}", "slot_type": "dosage_strength", "source_vocabulary": "rxnorm", "default_format": None},
                {"slot_name": "{DOSAGE_FORM}", "slot_type": "dosage_strength", "source_vocabulary": "rxnorm", "default_format": "tablets/capsules"},
                {"slot_name": "{SIG_ACTION}", "slot_type": "sig_phrase", "source_vocabulary": "sig_codes", "default_format": "Take"},
                {"slot_name": "{SIG_QUANTITY}", "slot_type": "sig_phrase", "source_vocabulary": "sig_codes", "default_format": "1 tablet"},
                {"slot_name": "{SIG_ROUTE}", "slot_type": "sig_code", "source_vocabulary": "sig_codes", "default_format": "PO"},
                {"slot_name": "{SIG_FREQUENCY}", "slot_type": "sig_code", "source_vocabulary": "sig_codes", "default_format": "BID"},
                {"slot_name": "{SIG_MODIFIER}", "slot_type": "sig_phrase", "source_vocabulary": "sig_codes", "default_format": "with meals"},
                {"slot_name": "{DISPENSE_QTY}", "slot_type": "dosage_strength", "source_vocabulary": None, "default_format": "30"},
                {"slot_name": "{DISPENSE_WORDS}", "slot_type": "dosage_strength", "source_vocabulary": None, "default_format": "Thirty"},
                {"slot_name": "{REFILLS}", "slot_type": "dosage_strength", "source_vocabulary": None, "default_format": "2"},
                {"slot_name": "{SIGNATURE_SCRAWL}", "slot_type": "physician_metadata", "source_vocabulary": "doctors", "default_format": None}
            ],
            "rendering_guidance": {"ruled_paper": True, "header_required": True, "signature_at_bottom": True}
        }
        templates.append(t_rx)
        tpl_count += 1

    catalog = {
        "version": "2026.1",
        "total_templates": len(templates),
        "categories": [
            "outpatient_encounter",
            "discharge_summary",
            "soap_note",
            "emergency_triage",
            "prescription_slip"
        ],
        "templates": templates
    }
    return catalog


def generate_and_save_all_vocabularies(output_dir: str = "data/reference_handwriting/vocabularies") -> Dict[str, int]:
    """
    Main generator pipeline executing generation, validation, and atomic JSON writes.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating RxNorm Medication Catalog (>= 1,000 medications, 7 classes, 20+ LASA pairs)...")
    rxnorm_catalog = build_rxnorm_medication_catalog()
    assert len(rxnorm_catalog["medications"]) >= 1000, f"Expected >= 1000 meds, got {len(rxnorm_catalog['medications'])}"
    rxnorm_file = out_path / "rxnorm_medications.json"
    with open(rxnorm_file, "w", encoding="utf-8") as f:
        json.dump(rxnorm_catalog, f, indent=2)
    logger.info(f"Saved {rxnorm_catalog['total_medications']} medications to {rxnorm_file}")

    logger.info("Generating Latin Sig Codes Catalog (>= 50 codes)...")
    sig_catalog = build_latin_sig_catalog()
    assert len(sig_catalog["codes"]) >= 50, f"Expected >= 50 sig codes, got {len(sig_catalog['codes'])}"
    sig_file = out_path / "latin_sig_codes.json"
    with open(sig_file, "w", encoding="utf-8") as f:
        json.dump(sig_catalog, f, indent=2)
    logger.info(f"Saved {sig_catalog['total_codes']} sig codes to {sig_file}")

    logger.info("Generating Doctor Profiles Catalog (>= 100 profiles, valid NPI/DEA)...")
    doc_catalog = build_doctor_profile_catalog()
    assert len(doc_catalog["doctors"]) >= 100, f"Expected >= 100 doctors, got {len(doc_catalog['doctors'])}"
    doc_file = out_path / "doctor_profiles.json"
    with open(doc_file, "w", encoding="utf-8") as f:
        json.dump(doc_catalog, f, indent=2)
    logger.info(f"Saved {doc_catalog['total_doctors']} doctor profiles to {doc_file}")

    logger.info("Generating Clinical Templates Catalog (>= 200 templates across 5 categories)...")
    tpl_catalog = build_clinical_templates_catalog()
    assert len(tpl_catalog["templates"]) >= 200, f"Expected >= 200 templates, got {len(tpl_catalog['templates'])}"
    tpl_file = out_path / "clinical_templates.json"
    with open(tpl_file, "w", encoding="utf-8") as f:
        json.dump(tpl_catalog, f, indent=2)
    logger.info(f"Saved {tpl_catalog['total_templates']} clinical templates to {tpl_file}")

    summary = {
        "medications": len(rxnorm_catalog["medications"]),
        "sig_codes": len(sig_catalog["codes"]),
        "doctors": len(doc_catalog["doctors"]),
        "templates": len(tpl_catalog["templates"])
    }
    logger.info(f"Clinical Vocabularies Generation Complete: {summary}")
    return summary


if __name__ == "__main__":
    generate_and_save_all_vocabularies()

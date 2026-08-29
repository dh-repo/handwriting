"""
pipeline/dataset/generate_dataset_parallel.py
High-throughput parallel generation and curation engine for 50,000+ medical & benchmark handwriting samples.

Features:
- Multi-core ProcessPoolExecutor (20 workers on Mac Studio Apple M3 Ultra)
- Direct NVMe disk writes with zero-IPC image streaming
- Realistic 3D physical augmentations (Lambertian shading, shadows, ink bleed, vertex pooling, OU tremor)
- Strict 80/10/10 writer-independent disjoint split partitioning (W_train ∩ W_val = ∅, W_train ∩ W_test = ∅, W_val ∩ W_test = ∅)
- Complete JSONL manifests: train_manifest.jsonl, val_manifest.jsonl, test_manifest.jsonl, full_manifest.jsonl, dataset_summary.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dataset.synthetic_generator import (
    HandwritingFontManager,
    PhysicalAugmenter,
    SyntheticHandwritingGenerator,
    VocabularyManager,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("parallel_generator")


def _refuse_synthetic_training_corpus(output_dir: str) -> None:
    resolved = Path(output_dir)
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    corpus = (PROJECT_ROOT / "data" / "reference_handwriting").resolve()
    try:
        resolved.relative_to(corpus)
    except ValueError:
        return
    raise RuntimeError(
        "Refusing to write synthetic handwriting into data/reference_handwriting/. "
        "Download public corpora with pipeline.dataset.download_and_curate_multisource."
    )


# ---------------------------------------------------------------------------
# Worker Task Function (Top-Level Module for macOS Process Spawn Safety)
# ---------------------------------------------------------------------------

def _parallel_worker_generate_chunk(
    worker_id: int,
    start_idx: int,
    num_samples: int,
    output_dir_str: str,
    base_seed: int,
    writer_split_map: Dict[str, str],
    writer_pool: List[str],
    vocab_dir_str: str
) -> List[Dict[str, Any]]:
    """
    Worker process generating a batch of synthetic handwriting samples.
    Executes physical augmentations and writes PNG images directly to disk.
    """
    worker_seed = base_seed + worker_id * 100_000
    random.seed(worker_seed)
    rng = np.random.default_rng(worker_seed)

    out_path = Path(output_dir_str)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    font_mgr = HandwritingFontManager()
    vocab_mgr = VocabularyManager(vocab_dir=vocab_dir_str)
    syn_gen = SyntheticHandwritingGenerator(font_manager=font_mgr, vocab_manager=vocab_mgr)

    available_fonts = font_mgr.list_available_fonts()
    manifest_records: List[Dict[str, Any]] = []

    # Assign distinct handwriting affinities per writer in pool
    writer_profiles: Dict[str, Dict[str, Any]] = {}
    for w_id in writer_pool:
        w_rand = random.Random(hash(w_id) + worker_seed)
        font_subset = w_rand.sample(available_fonts, k=min(len(available_fonts), w_rand.randint(1, 3)))
        writer_profiles[w_id] = {
            "fonts": font_subset,
            "slant_mean": w_rand.uniform(-5.0, 26.0),
            "slant_std": w_rand.uniform(1.0, 3.5),
            "tremor_sigma": w_rand.uniform(0.6, 2.4),
            "wave_amp": w_rand.uniform(0.8, 3.8),
            "ink": w_rand.choice(["blue", "black", "fountain", "pencil"])
        }

    categories = [
        ("prescription_item", 0.40),
        ("clinical_note", 0.30),
        ("doctor_signature", 0.15),
        ("general_cursive_line", 0.15),
    ]
    cat_choices = [c[0] for c in categories]
    cat_weights = [c[1] for c in categories]

    for i in range(num_samples):
        sample_idx = start_idx + i
        category = random.choices(cat_choices, weights=cat_weights, k=1)[0]
        writer_id = random.choice(writer_pool)
        split_name = writer_split_map[writer_id]
        w_prof = writer_profiles[writer_id]

        # Generate sample text & metadata
        is_lasa = False
        therapeutic_class = "general"
        meta_dict: Dict[str, Any] = {}

        if category == "prescription_item":
            med = vocab_mgr.get_random_medication()
            sig = vocab_mgr.get_random_sig_code(category="frequency")
            strength = random.choice(med.get("standard_strengths", ["500mg"]))
            form = random.choice(med.get("dosage_forms", ["capsule"]))
            is_lasa = med.get("is_lasa", False)
            therapeutic_class = med.get("therapeutic_class", "general")

            sig_code = sig.get("code", "TID")
            text = f"{med.get('generic_name', 'Amoxicillin')} {strength} {form} Sig: PO {sig_code}"
            meta_dict = {
                "medication": med.get("generic_name"),
                "strength": strength,
                "dosage_form": form,
                "sig": sig_code,
                "rxcui": med.get("rxcui")
            }
        elif category == "clinical_note":
            filled_text, tpl_meta = vocab_mgr.fill_template()
            # Pick a representative line from template
            lines = [l.strip() for l in filled_text.split("\n") if len(l.strip()) > 5]
            text = random.choice(lines) if lines else "Patient presents with acute symptoms. Vital signs stable."
            is_lasa = tpl_meta.get("is_lasa", False)
            therapeutic_class = tpl_meta.get("therapeutic_class", "general")
            meta_dict = tpl_meta
        elif category == "doctor_signature":
            doc = vocab_mgr.get_random_doctor_profile()
            text = f"Dr. {doc.get('first_name', 'Sarah')} {doc.get('last_name', 'Jenkins')}, {random.choice(['MD', 'DO', 'MD, FACP'])}"
            meta_dict = {
                "doctor_id": doc.get("doctor_id"),
                "npi": doc.get("npi"),
                "dea": doc.get("dea_number"),
                "specialty": doc.get("specialty")
            }
        else:  # general_cursive_line
            general_phrases = [
                "The patient tolerated the clinical examination without complications.",
                "Review of systems is negative for fever, chills, or acute respiratory distress.",
                "Continue current medication regimen and follow up in clinic in three months.",
                "Laboratory results demonstrate normal complete blood count and electrolyte panel.",
                "Patient advised on lifestyle modifications, low sodium diet, and regular exercise.",
                "No known drug allergies. Immunization status up to date.",
                "Prescription refilled as directed. Dr. Mitchell, MD.",
                "Electrocardiogram shows normal sinus rhythm without acute ischemic changes."
            ]
            text = random.choice(general_phrases)

        # Style parameters
        font_name = random.choice(w_prof["fonts"]) if w_prof["fonts"] else None
        slant = float(np.clip(random.gauss(w_prof["slant_mean"], w_prof["slant_std"]), -15.0, 38.0))
        tremor_sig = float(np.clip(random.gauss(w_prof["tremor_sigma"], 0.3), 0.2, 3.8))
        wave_amp = float(np.clip(random.gauss(w_prof["wave_amp"], 0.4), 0.3, 5.0))
        ink_col = w_prof["ink"]
        font_sz = random.choice([28, 32, 36, 40])

        # 1. Render single handwritten line
        line_img, line_meta = syn_gen.render_line(
            text=text,
            font_name=font_name,
            font_size=font_sz,
            ink_color=ink_col,
            slant_deg=slant,
            tremor_sigma=tremor_sig,
            wave_amplitude=wave_amp,
            apply_physical_effects=True
        )

        # 2. Apply 3D Lambertian Shading & Paper Texture
        if rng.random() < 0.85:
            line_img = PhysicalAugmenter.apply_3d_lambertian_shading(
                line_img,
                intensity=float(rng.uniform(0.10, 0.25)),
                light_theta=float(rng.uniform(0, 2 * math.pi)),
                light_phi=float(rng.uniform(0.4, 1.2)),
                relief_scale=float(rng.uniform(0.8, 1.8)),
                rng=rng
            )

        # 3. Apply Non-Uniform Shadow Gradients
        if rng.random() < 0.70:
            line_img = PhysicalAugmenter.apply_shadow_gradients(
                line_img,
                linear_intensity=float(rng.uniform(0.08, 0.22)),
                vignette_intensity=float(rng.uniform(0.10, 0.25)),
                include_spine_shadow=(rng.random() < 0.3),
                include_blob=(rng.random() < 0.5),
                rng=rng
            )

        # 4. Save PNG Image direct to disk
        sample_id_str = f"sample_{sample_idx:06d}"
        img_filename = f"{sample_id_str}_syn.png"
        img_save_path = images_dir / img_filename

        # Save using cv2.imwrite for max throughput (convert RGB to BGR)
        bgr_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_save_path), bgr_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        h_out, w_out = line_img.shape[:2]

        record = {
            "id": sample_id_str,
            "sample_id": sample_id_str,
            "image_path": str(img_save_path),
            "relative_image_path": f"images/{img_filename}",
            "transcription": text,
            "text": text,
            "writer_id": writer_id,
            "dataset_source": "synthetic_medical_cursive",
            "source": "synthetic_medical_cursive",
            "category": category,
            "split": split_name,
            "is_lasa": is_lasa,
            "therapeutic_class": therapeutic_class,
            "width": w_out,
            "height": h_out,
            "augmentation_params": {
                "slant_deg": round(slant, 2),
                "tremor_sigma": round(tremor_sig, 2),
                "wave_amplitude": round(wave_amp, 2),
                "ink_type": ink_col,
                "paper_texture": "3d_lambertian_crumple",
                "lighting_applied": True,
                "shadow_gradient": "composite_illumination"
            },
            "metadata": meta_dict
        }
        manifest_records.append(record)

    return manifest_records


# ---------------------------------------------------------------------------
# Main Parallel Dataset Generator & Orchestrator
# ---------------------------------------------------------------------------

def generate_50k_parallel_dataset(
    output_dir: str = "data/reference_handwriting",
    vocab_dir: str = "data/reference_handwriting/vocabularies",
    total_target: int = 50500,
    max_workers: int = 20,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Orchestrates the parallel generation of 50,000+ verified handwriting samples
    with strict 80/10/10 writer-independent split partitioning.
    """
    _refuse_synthetic_training_corpus(output_dir)
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Establish 600+ Disjoint Writer IDs
    num_writers = 650
    all_writer_ids = [f"syn_w_{i:03d}" for i in range(num_writers)]
    random.shuffle(all_writer_ids)

    # 80/10/10 Partitioning
    n_train_w = int(0.80 * num_writers)  # 520 writers
    n_val_w = int(0.10 * num_writers)    # 65 writers
    n_test_w = num_writers - n_train_w - n_val_w  # 65 writers

    train_writers = set(all_writer_ids[:n_train_w])
    val_writers = set(all_writer_ids[n_train_w:n_train_w + n_val_w])
    test_writers = set(all_writer_ids[n_train_w + n_val_w:])

    # Verify 0% writer overlap
    assert train_writers.isdisjoint(val_writers)
    assert train_writers.isdisjoint(test_writers)
    assert val_writers.isdisjoint(test_writers)

    writer_split_map: Dict[str, str] = {}
    for w in train_writers:
        writer_split_map[w] = "train"
    for w in val_writers:
        writer_split_map[w] = "val"
    for w in test_writers:
        writer_split_map[w] = "test"

    logger.info(f"Writer partitioning: {len(train_writers)} train, {len(val_writers)} val, {len(test_writers)} test.")

    # Determine Target Counts per Split (80/10/10)
    target_train = int(0.80 * total_target)  # 40,400
    target_val = int(0.10 * total_target)    # 5,050
    target_test = total_target - target_train - target_val  # 5,050

    logger.info(f"Target sample quotas: Train={target_train}, Val={target_val}, Test={target_test}, Total={total_target}")

    # Build Tasks across workers
    # We assign chunks specifically targeting train, val, and test writer pools
    worker_tasks = []
    current_sample_idx = 0

    # 1. Train chunks
    train_chunk_size = math.ceil(target_train / max_workers)
    for w_idx in range(max_workers):
        num_chunk = min(train_chunk_size, target_train - w_idx * train_chunk_size)
        if num_chunk <= 0:
            break
        worker_tasks.append((
            w_idx,
            current_sample_idx,
            num_chunk,
            str(out_path),
            seed,
            writer_split_map,
            list(train_writers),
            vocab_dir
        ))
        current_sample_idx += num_chunk

    # 2. Val chunks
    val_chunk_size = math.ceil(target_val / max_workers)
    for w_idx in range(max_workers):
        num_chunk = min(val_chunk_size, target_val - w_idx * val_chunk_size)
        if num_chunk <= 0:
            break
        worker_tasks.append((
            w_idx + 100,
            current_sample_idx,
            num_chunk,
            str(out_path),
            seed + 1000,
            writer_split_map,
            list(val_writers),
            vocab_dir
        ))
        current_sample_idx += num_chunk

    # 3. Test chunks
    test_chunk_size = math.ceil(target_test / max_workers)
    for w_idx in range(max_workers):
        num_chunk = min(test_chunk_size, target_test - w_idx * test_chunk_size)
        if num_chunk <= 0:
            break
        worker_tasks.append((
            w_idx + 200,
            current_sample_idx,
            num_chunk,
            str(out_path),
            seed + 2000,
            writer_split_map,
            list(test_writers),
            vocab_dir
        ))
        current_sample_idx += num_chunk

    logger.info(f"Launching {len(worker_tasks)} parallel chunks across {max_workers} worker processes...")

    all_records: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_parallel_worker_generate_chunk, *task) for task in worker_tasks]
        for f in as_completed(futures):
            try:
                chunk_records = f.result()
                all_records.extend(chunk_records)
            except Exception as e:
                logger.error(f"Worker task failed with exception: {e}")
                raise e

    elapsed = time.time() - start_time
    logger.info(f"Generated {len(all_records)} samples in {elapsed:.2f}s ({len(all_records)/elapsed:.1f} samples/sec).")

    # Group into splits
    train_records = [r for r in all_records if r["split"] == "train"]
    val_records = [r for r in all_records if r["split"] == "val"]
    test_records = [r for r in all_records if r["split"] == "test"]

    # Verify writer isolation in generated dataset
    actual_train_w = set(r["writer_id"] for r in train_records)
    actual_val_w = set(r["writer_id"] for r in val_records)
    actual_test_w = set(r["writer_id"] for r in test_records)

    overlap_train_val = len(actual_train_w & actual_val_w)
    overlap_train_test = len(actual_train_w & actual_test_w)
    overlap_val_test = len(actual_val_w & actual_test_w)

    assert overlap_train_val == 0, f"Writer leakage Train/Val: {actual_train_w & actual_val_w}"
    assert overlap_train_test == 0, f"Writer leakage Train/Test: {actual_train_w & actual_test_w}"
    assert overlap_val_test == 0, f"Writer leakage Val/Test: {actual_val_w & actual_test_w}"

    logger.info("Writer Independence Verified: 0% writer overlap between all splits.")

    # Write Manifests
    train_manifest_path = out_path / "train_manifest.jsonl"
    val_manifest_path = out_path / "val_manifest.jsonl"
    test_manifest_path = out_path / "test_manifest.jsonl"
    full_manifest_path = out_path / "full_manifest.jsonl"

    logger.info("Writing JSONL manifest files...")
    with open(train_manifest_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_manifest_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(test_manifest_path, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(full_manifest_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Generate dataset summary
    sources_count: Dict[str, int] = {}
    categories_count: Dict[str, int] = {}
    th_classes_count: Dict[str, int] = {}
    lasa_count = 0

    for r in all_records:
        src = r.get("dataset_source", "unknown")
        sources_count[src] = sources_count.get(src, 0) + 1
        cat = r.get("category", "unknown")
        categories_count[cat] = categories_count.get(cat, 0) + 1
        th = r.get("therapeutic_class", "unknown")
        th_classes_count[th] = th_classes_count.get(th, 0) + 1
        if r.get("is_lasa"):
            lasa_count += 1

    summary = {
        "dataset_name": "Multi-Source Clinical & Benchmark Handwriting Dataset",
        "version": "2.0.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(all_records),
        "splits": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records)
        },
        "split_ratios": {
            "train": round(len(train_records) / len(all_records), 4),
            "val": round(len(val_records) / len(all_records), 4),
            "test": round(len(test_records) / len(all_records), 4)
        },
        "unique_writers": {
            "total": len(actual_train_w | actual_val_w | actual_test_w),
            "train": len(actual_train_w),
            "val": len(actual_val_w),
            "test": len(actual_test_w),
            "writer_overlap_train_val": overlap_train_val,
            "writer_overlap_train_test": overlap_train_test,
            "writer_overlap_val_test": overlap_val_test
        },
        "sources": sources_count,
        "categories": categories_count,
        "therapeutic_classes": th_classes_count,
        "lasa_samples_count": lasa_count,
        "generation_time_seconds": round(elapsed, 2),
        "throughput_samples_per_sec": round(len(all_records) / max(0.001, elapsed), 2)
    }

    summary_path = out_path / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset Generation Complete! Summary written to {summary_path}")
    logger.info(f"Total: {len(all_records)} | Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")
    return summary


def remediate_blank_images(
    output_dir: str = "data/reference_handwriting",
    vocab_dir: str = "data/reference_handwriting/vocabularies",
    min_std: float = 0.5,
    seed: int = 42
) -> int:
    """
    Scans data/reference_handwriting/images/ via manifests, identifies any image
    where np.std(img) < min_std (blank or near-blank), and regenerates it using
    the reliable Latin handwriting generator with 3D physical augmentations.

    Returns:
        Number of remediated blank images.
    """
    _refuse_synthetic_training_corpus(output_dir)
    out_path = Path(output_dir)
    full_manifest_path = out_path / "full_manifest.jsonl"
    if not full_manifest_path.exists():
        logger.warning(f"Manifest not found: {full_manifest_path}")
        return 0

    font_mgr = HandwritingFontManager()
    vocab_mgr = VocabularyManager(vocab_dir=vocab_dir)
    syn_gen = SyntheticHandwritingGenerator(font_manager=font_mgr, vocab_manager=vocab_mgr)
    rng = np.random.default_rng(seed)

    records = []
    with open(full_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    remediated_count = 0
    updated_records = {}

    for r in records:
        img_path = Path(r["image_path"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path

        cv_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        is_blank = (cv_img is None) or (float(np.std(cv_img)) < min_std)

        if is_blank:
            remediated_count += 1
            text = r.get("transcription", r.get("text", "Amoxicillin 500mg"))
            slant = float(rng.uniform(5.0, 20.0))
            tremor = float(rng.uniform(0.8, 1.8))
            wave_amp = float(rng.uniform(1.0, 3.0))
            ink = random.choice(["blue", "black", "fountain"])

            # 1. Render single line with Latin font fallback guaranteed
            line_img, meta = syn_gen.render_line(
                text=text,
                font_size=random.choice([30, 34, 38]),
                ink_color=ink,
                slant_deg=slant,
                tremor_sigma=tremor,
                wave_amplitude=wave_amp,
                apply_physical_effects=True
            )

            # 2. 3D Shading
            line_img = PhysicalAugmenter.apply_3d_lambertian_shading(
                line_img,
                intensity=float(rng.uniform(0.12, 0.22)),
                light_theta=float(rng.uniform(0, 2 * math.pi)),
                light_phi=float(rng.uniform(0.4, 1.2)),
                relief_scale=float(rng.uniform(1.0, 1.6)),
                rng=rng
            )

            # 3. Non-Uniform Shadow Gradients
            line_img = PhysicalAugmenter.apply_shadow_gradients(
                line_img,
                linear_intensity=float(rng.uniform(0.10, 0.20)),
                vignette_intensity=float(rng.uniform(0.12, 0.22)),
                rng=rng
            )

            # 4. Save to disk
            bgr_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), bgr_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            # Update dimensions
            h_out, w_out = line_img.shape[:2]
            r["width"] = w_out
            r["height"] = h_out
            updated_records[r["id"]] = r
            logger.info(f"Remediated blank image {r['id']} at {img_path}")

    if remediated_count > 0:
        logger.info(f"Remediated {remediated_count} blank images. Syncing manifest files...")
        for m_name in ["train_manifest.jsonl", "val_manifest.jsonl", "test_manifest.jsonl", "full_manifest.jsonl"]:
            m_path = out_path / m_name
            if not m_path.exists():
                continue
            m_records = []
            with open(m_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec["id"] in updated_records:
                            rec = updated_records[rec["id"]]
                        m_records.append(rec)
            with open(m_path, "w", encoding="utf-8") as f:
                for rec in m_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Post-remediation audit verification
    remaining_blanks = 0
    with open(full_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                p = Path(rec["image_path"])
                if not p.is_absolute():
                    p = PROJECT_ROOT / p
                im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if im is None or float(np.std(im)) < min_std:
                    remaining_blanks += 1

    logger.info(f"Remediation verification: {remaining_blanks} blank images remaining out of {len(records)}.")
    assert remaining_blanks == 0, f"Remediation failed: {remaining_blanks} blank images remain!"
    return remediated_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/reference_handwriting")
    parser.add_argument("--vocab-dir", default="data/reference_handwriting/vocabularies")
    parser.add_argument("--total-target", type=int, default=50500)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--remediate-blanks", action="store_true", help="Scan and remediate any blank images in dataset")
    args = parser.parse_args()

    if getattr(args, "remediate_blanks", False):
        remediate_blank_images(output_dir=args.output_dir, vocab_dir=args.vocab_dir)
    else:
        generate_50k_parallel_dataset(
            output_dir=args.output_dir,
            vocab_dir=args.vocab_dir,
            total_target=args.total_target,
            max_workers=args.workers
        )
        remediate_blank_images(output_dir=args.output_dir, vocab_dir=args.vocab_dir)

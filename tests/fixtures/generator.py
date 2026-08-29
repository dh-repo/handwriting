#!/usr/bin/env python3
"""
Test Fixture Generator for Handwriting Recognition E2E Test Suite.
Generates synthetic handwriting samples, medical prescriptions, skewed documents,
low-contrast faded manuscripts, multi-page consultation PDFs, and corrupted test inputs.
Produces tests/fixtures/manifest.json cataloging ground truth and bounding boxes.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


# Candidate system fonts for handwriting rendering
CANDIDATE_FONTS = [
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/Brush Script.ttf",
    "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
    "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
    "/System/Library/Fonts/Supplemental/Zapfino.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def get_font(size: int = 36) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Attempts to load a cursive/handwriting font, falling back gracefully."""
    for font_path in CANDIDATE_FONTS:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


class FixtureGenerator:
    """Generator for all E2E test fixtures and ground-truth manifest."""

    def __init__(self, out_dir: Path | str = "tests/fixtures", seed: int = 42) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.corrupted_dir = self.out_dir / "corrupted"
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def ensure_directories(self) -> None:
        """Create target fixture directories if they don't exist."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.corrupted_dir.mkdir(parents=True, exist_ok=True)

    def _measure_text(
        self, draw: ImageDraw.ImageDraw, text: str, font: Any
    ) -> Tuple[int, int]:
        """Measure text width and height across Pillow versions."""
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                return draw.textsize(text, font=font)
            except Exception:
                return len(text) * 18, 30

    def generate_clean_handwriting(
        self, out_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate clean cursive handwriting sample with 3 lines."""
        width, height = 1200, 800
        image = Image.new("RGB", (width, height), color=(251, 251, 249))
        draw = ImageDraw.Draw(image)
        font = get_font(size=44)

        lines_text = [
            "The quick brown fox jumps over the lazy dog",
            "Handwriting recognition test sample",
            "Clean cursive text line three",
        ]

        lines_data: List[Dict[str, Any]] = []
        start_y = 140
        line_spacing = 180

        for line_idx, text in enumerate(lines_text):
            curr_y = start_y + line_idx * line_spacing
            curr_x = 100
            words = text.split(" ")
            words_data: List[Dict[str, Any]] = []

            line_w, line_h = self._measure_text(draw, text, font)
            # Draw line text
            draw.text((curr_x, curr_y), text, fill=(28, 37, 54), font=font)

            # Compute line bounding box in normalized [ymin, xmin, ymax, xmax]
            line_ymin = max(0.0, (curr_y - 10) / height)
            line_ymax = min(1.0, (curr_y + line_h + 15) / height)
            line_xmin = max(0.0, (curr_x - 10) / width)
            line_xmax = min(1.0, (curr_x + line_w + 15) / width)

            # Calculate word-level bounding boxes
            word_x = curr_x
            for word_idx, word in enumerate(words):
                w_w, w_h = self._measure_text(draw, word, font)
                w_ymin = max(0.0, (curr_y - 8) / height)
                w_ymax = min(1.0, (curr_y + w_h + 12) / height)
                w_xmin = max(0.0, (word_x - 6) / width)
                w_xmax = min(1.0, (word_x + w_w + 6) / width)

                words_data.append(
                    {
                        "word_id": f"p1_l{line_idx+1}_w{word_idx+1}",
                        "word_index": word_idx,
                        "text": word,
                        "confidence": 0.985,
                        "bbox": [
                            round(line_ymin, 4),
                            round(w_xmin, 4),
                            round(line_ymax, 4),
                            round(w_xmax, 4),
                        ],
                    }
                )
                space_w, _ = self._measure_text(draw, " ", font)
                word_x += w_w + space_w

            lines_data.append(
                {
                    "line_id": f"p1_l{line_idx+1}",
                    "line_index": line_idx,
                    "text": text,
                    "confidence": 0.982,
                    "bbox": [
                        round(line_ymin, 4),
                        round(line_xmin, 4),
                        round(line_ymax, 4),
                        round(line_xmax, 4),
                    ],
                    "words": words_data,
                }
            )

        target_file = out_path or (self.out_dir / "sample_clean_handwriting.png")
        image.save(target_file, format="PNG")

        return {
            "filename": target_file.name,
            "format": "PNG",
            "width": width,
            "height": height,
            "pages": 1,
            "category": "clean",
            "skew_angle": 0.0,
            "ground_truth_text": "\n".join(lines_text),
            "expected_line_count": len(lines_text),
            "expected_word_count": sum(len(l.split(" ")) for l in lines_text),
            "lines": lines_data,
        }

    def generate_messy_prescription(
        self, out_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate messy medical prescription with slant, jitter, ruled paper texture."""
        width, height = 1200, 1600
        image = Image.new("RGB", (width, height), color=(245, 247, 250))
        draw = ImageDraw.Draw(image)

        # Draw subtle blue ruled lines
        for y in range(80, height, 60):
            draw.line([(60, y), (width - 60, y)], fill=(226, 232, 240), width=1)

        # Draw Rx symbol header
        header_font = get_font(size=56)
        draw.text((100, 100), "℞", fill=(15, 23, 42), font=header_font)
        draw.line([(100, 180), (width - 100, 180)], fill=(203, 213, 225), width=2)

        body_font = get_font(size=40)
        lines_text = [
            "Rx: Amoxicillin 500mg",
            "Sig: 1 tab PO TID x 10 days",
            "Refills: 2",
            "Dr. J. Doe MD",
        ]

        lines_data: List[Dict[str, Any]] = []
        start_y = 260
        line_spacing = 160

        for line_idx, text in enumerate(lines_text):
            curr_y = start_y + line_idx * line_spacing
            curr_x = 120
            words = text.split(" ")
            words_data: List[Dict[str, Any]] = []

            # Simulated pen color with slight jitter
            ink_color = (20 + random.randint(0, 15), 30 + random.randint(0, 15), 60 + random.randint(0, 20))

            word_x = curr_x
            line_min_y = curr_y
            line_max_y = curr_y + 40
            line_max_x = curr_x

            for word_idx, word in enumerate(words):
                w_w, w_h = self._measure_text(draw, word, body_font)
                jitter_y = random.randint(-4, 4)
                w_y = curr_y + jitter_y

                # Draw word with jitter
                draw.text((word_x, w_y), word, fill=ink_color, font=body_font)

                w_ymin = max(0.0, (w_y - 8) / height)
                w_ymax = min(1.0, (w_y + w_h + 12) / height)
                w_xmin = max(0.0, (word_x - 6) / width)
                w_xmax = min(1.0, (word_x + w_w + 6) / width)

                words_data.append(
                    {
                        "word_id": f"p1_l{line_idx+1}_w{word_idx+1}",
                        "word_index": word_idx,
                        "text": word,
                        "confidence": round(0.92 + random.random() * 0.06, 3),
                        "bbox": [
                            round(w_ymin, 4),
                            round(w_xmin, 4),
                            round(w_ymax, 4),
                            round(w_xmax, 4),
                        ],
                    }
                )

                line_min_y = min(line_min_y, w_y)
                line_max_y = max(line_max_y, w_y + w_h)
                line_max_x = word_x + w_w

                space_w, _ = self._measure_text(draw, " ", body_font)
                word_x += w_w + space_w + random.randint(2, 6)

            line_ymin = max(0.0, (line_min_y - 12) / height)
            line_ymax = min(1.0, (line_max_y + 16) / height)
            line_xmin = max(0.0, (curr_x - 10) / width)
            line_xmax = min(1.0, (line_max_x + 10) / width)

            lines_data.append(
                {
                    "line_id": f"p1_l{line_idx+1}",
                    "line_index": line_idx,
                    "text": text,
                    "confidence": round(0.93 + random.random() * 0.05, 3),
                    "bbox": [
                        round(line_ymin, 4),
                        round(line_xmin, 4),
                        round(line_ymax, 4),
                        round(line_xmax, 4),
                    ],
                    "words": words_data,
                }
            )

        # Doctor signature flourish at bottom
        sig_y = start_y + len(lines_text) * line_spacing + 40
        draw.line([(120, sig_y + 60), (450, sig_y + 60)], fill=(148, 163, 184), width=1)
        draw.text((130, sig_y + 20), "J. Doe MD", fill=(30, 41, 59), font=body_font)

        target_file = out_path or (self.out_dir / "sample_messy_prescription.png")
        image.save(target_file, format="PNG")

        return {
            "filename": target_file.name,
            "format": "PNG",
            "width": width,
            "height": height,
            "pages": 1,
            "category": "messy",
            "skew_angle": -3.2,
            "ground_truth_text": "\n".join(lines_text),
            "expected_line_count": len(lines_text),
            "expected_word_count": sum(len(l.split(" ")) for l in lines_text),
            "lines": lines_data,
        }

    def generate_skewed_document(
        self, out_path: Optional[Path] = None, angle: float = 12.5
    ) -> Dict[str, Any]:
        """Generate document rotated by specified skew angle with background expansion."""
        base_w, base_h = 1200, 800
        base_img = Image.new("RGB", (base_w, base_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(base_img)
        font = get_font(size=46)

        lines_text = [
            "Skewed document line 1",
            "Skewed document line 2",
        ]

        draw.text((150, 250), lines_text[0], fill=(20, 20, 20), font=font)
        draw.text((150, 420), lines_text[1], fill=(20, 20, 20), font=font)

        # Rotate image with expand=True so corners are preserved
        rotated_img = base_img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        rot_w, rot_h = rotated_img.size

        target_file = out_path or (self.out_dir / "sample_skewed_document.png")
        rotated_img.save(target_file, format="PNG")

        return {
            "filename": target_file.name,
            "format": "PNG",
            "width": rot_w,
            "height": rot_h,
            "pages": 1,
            "category": "skewed",
            "skew_angle": angle,
            "ground_truth_text": "\n".join(lines_text),
            "expected_line_count": len(lines_text),
            "expected_word_count": sum(len(l.split(" ")) for l in lines_text),
        }

    def generate_low_contrast_faded(
        self, out_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate faded, low-contrast document with shadow gradient."""
        width, height = 1200, 800
        image = Image.new("RGB", (width, height), color=(240, 238, 230))
        draw = ImageDraw.Draw(image)
        font = get_font(size=40)

        lines_text = [
            "Faded ballpoint cursive line one",
            "Low contrast degraded historical manuscript",
            "Sauvola binarization verification test",
        ]

        # Draw text in faint grey ink
        y_pos = 180
        for text in lines_text:
            draw.text((120, y_pos), text, fill=(130, 130, 135), font=font)
            y_pos += 180

        # Apply lighting gradient: multiply array from top-left (1.0) to bottom-right (0.45)
        np_img = np.array(image, dtype=np.float32)
        y_grad = np.linspace(1.0, 0.55, height).reshape(height, 1, 1)
        x_grad = np.linspace(1.0, 0.70, width).reshape(1, width, 1)
        grad = y_grad * x_grad
        degraded = np.clip(np_img * grad, 0, 255).astype(np.uint8)

        faded_image = Image.fromarray(degraded)
        target_file = out_path or (self.out_dir / "sample_low_contrast_faded.png")
        faded_image.save(target_file, format="PNG")

        return {
            "filename": target_file.name,
            "format": "PNG",
            "width": width,
            "height": height,
            "pages": 1,
            "category": "low_contrast",
            "skew_angle": 0.0,
            "ground_truth_text": "\n".join(lines_text),
            "expected_line_count": len(lines_text),
            "expected_word_count": sum(len(l.split(" ")) for l in lines_text),
        }

    def generate_multipage_pdf(
        self, out_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate 3-page clinical consultation PDF."""
        # Page 1: Portrait Notes (1200x1600)
        p1_w, p1_h = 1200, 1600
        p1_img = Image.new("RGB", (p1_w, p1_h), color=(255, 255, 255))
        d1 = ImageDraw.Draw(p1_img)
        f_title = get_font(size=48)
        f_body = get_font(size=38)

        d1.text((100, 100), "Patient Consultation Notes", fill=(15, 23, 42), font=f_title)
        d1.line([(100, 170), (p1_w - 100, 170)], fill=(203, 213, 225), width=2)
        p1_lines = [
            "Date: 2026-08-26",
            "Chief Complaint: Persistent cough and fatigue",
            "Assessment: Acute bronchitis, mild fever",
            "Plan: Rest, fluids, and targeted antibiotic",
        ]
        y = 240
        for l in p1_lines:
            d1.text((100, y), l, fill=(30, 41, 59), font=f_body)
            y += 120

        # Page 2: Landscape Medication Schedule (1600x1200)
        p2_w, p2_h = 1600, 1200
        p2_img = Image.new("RGB", (p2_w, p2_h), color=(252, 252, 250))
        d2 = ImageDraw.Draw(p2_img)
        d2.text((100, 80), "Medication Schedule & Dosage Chart", fill=(15, 23, 42), font=f_title)
        d2.line([(100, 150), (p2_w - 100, 150)], fill=(203, 213, 225), width=2)
        p2_lines = [
            "1. Azithromycin 250mg PO QD x 5 days",
            "2. Guaifenesin 400mg every 4 hours PRN",
            "3. Albuterol inhaler 2 puffs q4h PRN wheezing",
        ]
        y = 220
        for l in p2_lines:
            d2.text((100, y), l, fill=(30, 41, 59), font=f_body)
            y += 130

        # Page 3: Portrait Follow-up (1200x1600)
        p3_w, p3_h = 1200, 1600
        p3_img = Image.new("RGB", (p3_w, p3_h), color=(255, 255, 255))
        d3 = ImageDraw.Draw(p3_img)
        d3.text((100, 100), "Follow-up & Discharge Instructions", fill=(15, 23, 42), font=f_title)
        d3.line([(100, 170), (p3_w - 100, 170)], fill=(203, 213, 225), width=2)
        p3_lines = [
            "Follow up in clinic in 2 weeks if symptoms persist",
            "Signed: Dr. Smith MD",
        ]
        y = 260
        for l in p3_lines:
            d3.text((100, y), l, fill=(30, 41, 59), font=f_body)
            y += 150

        target_file = out_path or (self.out_dir / "sample_multipage_consultation.pdf")
        # Save as multi-page PDF via Pillow
        p1_img.save(
            target_file,
            format="PDF",
            save_all=True,
            append_images=[p2_img, p3_img],
        )

        return {
            "filename": target_file.name,
            "format": "PDF",
            "pages": 3,
            "category": "multipage",
            "page_details": [
                {
                    "page_number": 1,
                    "width": p1_w,
                    "height": p1_h,
                    "expected_line_count": len(p1_lines) + 1,
                    "text": "Patient Consultation Notes\n" + "\n".join(p1_lines),
                },
                {
                    "page_number": 2,
                    "width": p2_w,
                    "height": p2_h,
                    "expected_line_count": len(p2_lines) + 1,
                    "text": "Medication Schedule & Dosage Chart\n" + "\n".join(p2_lines),
                },
                {
                    "page_number": 3,
                    "width": p3_w,
                    "height": p3_h,
                    "expected_line_count": len(p3_lines) + 1,
                    "text": "Follow-up & Discharge Instructions\n" + "\n".join(p3_lines),
                },
            ],
            "ground_truth_text": "\n\n--- PAGE BREAK ---\n\n".join(
                [
                    "Patient Consultation Notes\n" + "\n".join(p1_lines),
                    "Medication Schedule & Dosage Chart\n" + "\n".join(p2_lines),
                    "Follow-up & Discharge Instructions\n" + "\n".join(p3_lines),
                ]
            ),
        }

    def generate_corrupted_fixtures(self) -> Dict[str, Dict[str, Any]]:
        """Generate corrupted, malformed, and adversarial test inputs."""
        corrupted_meta: Dict[str, Dict[str, Any]] = {}

        # 1. 0-byte PNG file
        p_zero = self.corrupted_dir / "sample_zero_byte.png"
        p_zero.write_bytes(b"")
        corrupted_meta["corrupted/sample_zero_byte.png"] = {
            "format": "corrupted",
            "expected_error": "InvalidImageError",
            "expected_status_code": 422,
            "description": "Zero byte empty file",
        }

        # 2. Truncated JPEG header
        p_trunc = self.corrupted_dir / "sample_truncated_header.jpg"
        p_trunc.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00")
        corrupted_meta["corrupted/sample_truncated_header.jpg"] = {
            "format": "corrupted",
            "expected_error": "UnidentifiedImageError",
            "expected_status_code": 422,
            "description": "Truncated 12-byte JPEG header",
        }

        # 3. Corrupted PDF with broken xref and trailer
        p_bad_pdf = self.corrupted_dir / "sample_corrupted_pdf.pdf"
        p_bad_pdf.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"xref\n0 1\n0000000000 65535 f\ntrailer\n<<>>\n%%EOF\nBROKEN_BYTES"
        )
        corrupted_meta["corrupted/sample_corrupted_pdf.pdf"] = {
            "format": "corrupted",
            "expected_error": "CorruptedPDFError",
            "expected_status_code": 422,
            "description": "PDF with damaged cross-reference stream",
        }

        # 4. Extreme 1x1 pixel image
        p_1x1 = self.corrupted_dir / "sample_extreme_1x1.png"
        img_1x1 = Image.new("RGB", (1, 1), color=(255, 255, 255))
        img_1x1.save(p_1x1, format="PNG")
        corrupted_meta["corrupted/sample_extreme_1x1.png"] = {
            "format": "PNG",
            "width": 1,
            "height": 1,
            "pages": 1,
            "category": "boundary",
            "expected_line_count": 0,
            "expected_word_count": 0,
            "ground_truth_text": "",
            "description": "Minimal boundary 1x1 image",
        }

        # 5. Extreme aspect ratio (5000x10 px)
        p_aspect = self.corrupted_dir / "sample_extreme_aspect.png"
        img_aspect = Image.new("RGB", (5000, 10), color=(255, 255, 255))
        img_aspect.save(p_aspect, format="PNG")
        corrupted_meta["corrupted/sample_extreme_aspect.png"] = {
            "format": "PNG",
            "width": 5000,
            "height": 10,
            "pages": 1,
            "category": "boundary",
            "expected_line_count": 0,
            "ground_truth_text": "",
            "description": "Extreme aspect ratio 500:1 strip",
        }

        # 6. Disguised ASCII text file with .png extension
        p_disguised = self.corrupted_dir / "sample_disguised_text.png"
        p_disguised.write_bytes(b"This is plain text disguised as a PNG file.\nNot a valid image format.")
        corrupted_meta["corrupted/sample_disguised_text.png"] = {
            "format": "corrupted",
            "expected_error": "InvalidImageSignatureError",
            "expected_status_code": 422,
            "description": "Plain ASCII text disguised as PNG",
        }

        # 7. 4-Channel CMYK JPEG image
        p_cmyk = self.corrupted_dir / "sample_cmyk.jpg"
        img_cmyk = Image.new("CMYK", (600, 400), color=(0, 128, 128, 0))
        d_cmyk = ImageDraw.Draw(img_cmyk)
        font = get_font(size=28)
        d_cmyk.text((40, 180), "CMYK Color Space Document", fill=(0, 255, 255, 100), font=font)
        img_cmyk.save(p_cmyk, format="JPEG")
        corrupted_meta["corrupted/sample_cmyk.jpg"] = {
            "format": "JPEG",
            "width": 600,
            "height": 400,
            "color_mode": "CMYK",
            "pages": 1,
            "category": "color_space",
            "ground_truth_text": "CMYK Color Space Document",
            "expected_line_count": 1,
            "description": "4-channel CMYK JPEG image requiring RGB conversion",
        }

        return corrupted_meta

    def generate_all(self, out_dir: Optional[Path | str] = None) -> Dict[str, Any]:
        """Generate all test fixtures and build the manifest catalog."""
        if out_dir:
            self.out_dir = Path(out_dir).resolve()
            self.corrupted_dir = self.out_dir / "corrupted"

        self.ensure_directories()

        manifest: Dict[str, Any] = {
            "version": "1.0.0",
            "generated_at": "2026-08-26T20:25:00Z",
            "fixtures": {},
        }

        # 1. Clean Handwriting
        clean_meta = self.generate_clean_handwriting()
        manifest["fixtures"]["sample_clean_handwriting.png"] = clean_meta

        # 2. Messy Prescription
        messy_meta = self.generate_messy_prescription()
        manifest["fixtures"]["sample_messy_prescription.png"] = messy_meta

        # 3. Skewed Document
        skew_meta = self.generate_skewed_document(angle=12.5)
        manifest["fixtures"]["sample_skewed_document.png"] = skew_meta

        # 4. Low Contrast Faded Document
        faded_meta = self.generate_low_contrast_faded()
        manifest["fixtures"]["sample_low_contrast_faded.png"] = faded_meta

        # 5. Multi-Page Consultation PDF
        pdf_meta = self.generate_multipage_pdf()
        manifest["fixtures"]["sample_multipage_consultation.pdf"] = pdf_meta

        # 6. Corrupted & Adversarial Inputs
        corrupted_meta = self.generate_corrupted_fixtures()
        manifest["fixtures"].update(corrupted_meta)

        # Write manifest.json
        manifest_path = self.out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate E2E handwriting recognition test fixtures")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="tests/fixtures",
        help="Output directory for generated fixture assets (default: tests/fixtures)",
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        default=True,
        help="Generate all standard and corrupted fixture assets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible fixture generation (default: 42)",
    )

    args = parser.parse_args()
    generator = FixtureGenerator(out_dir=args.out_dir, seed=args.seed)
    print(f"Generating E2E fixtures in {generator.out_dir} ...")
    manifest = generator.generate_all()
    print(f"Successfully generated {len(manifest['fixtures'])} fixture definitions.")
    print(f"Manifest written to: {generator.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

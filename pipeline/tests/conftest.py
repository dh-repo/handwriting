"""
pipeline/tests/conftest.py
Pytest fixtures and test doubles for Milestone 1 test suite.
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, Tuple

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from pipeline.dataset.synthetic_generator import (
    BackgroundGenerator,
    SyntheticHandwritingGenerator,
    HandwritingFontManager,
)


def write_tiny_handwriting_manifest(
    directory: Path,
    name: str = "train_manifest.jsonl",
    count: int = 8,
    categories: tuple[str, ...] = (
        "prescription_item",
        "clinical_note",
        "doctor_signature",
        "general_cursive_line",
    ),
) -> Path:
    """Write a small on-disk image+JSONL fixture for curriculum/eval tests."""
    directory.mkdir(parents=True, exist_ok=True)
    img_dir = directory / "images"
    img_dir.mkdir(exist_ok=True)
    manifest = directory / name
    with open(manifest, "w", encoding="utf-8") as handle:
        for i in range(count):
            png = img_dir / f"{Path(name).stem}_{i:03d}.png"
            Image.fromarray(np.full((32, 128, 3), 180 + (i % 40), dtype=np.uint8)).save(png)
            rec = {
                "id": f"{Path(name).stem}_{i:03d}",
                "sample_id": f"{Path(name).stem}_{i:03d}",
                "image_path": str(png),
                "text": f"hello world {i}",
                "transcription": f"hello world {i}",
                "writer_id": f"w_{i}",
                "category": categories[i % len(categories)],
                "is_lasa": i % 5 == 0,
                "dataset_source": "iam_line",
                "source": "Teklia/IAM-line",
            }
            handle.write(json.dumps(rec) + "\n")
    return manifest


@pytest.fixture(scope="session")
def font_manager() -> HandwritingFontManager:
    return HandwritingFontManager()


@pytest.fixture(scope="session")
def synthetic_generator(font_manager: HandwritingFontManager) -> SyntheticHandwritingGenerator:
    return SyntheticHandwritingGenerator(font_manager=font_manager)


@pytest.fixture
def synthetic_multiline_image(synthetic_generator: SyntheticHandwritingGenerator) -> np.ndarray:
    """
    Generate a 4-line clean handwritten document on ruled paper.
    """
    w, h = 600, 450
    canvas = BackgroundGenerator.generate_lined_paper(w, h, line_spacing=70, margin_x=70)
    lines = [
        "First line of test handwriting.",
        "Second line with cursive descenders.",
        "Third line showing clear ascenders.",
        "Fourth final line of text."
    ]
    cur_y = 70
    for text in lines:
        line_img, meta = synthetic_generator.render_line(
            text, font_size=26, ink_color="blue", slant_deg=8.0,
            tremor_sigma=0.5, canvas_width=w - 90, canvas_height=55
        )
        lh, lw = line_img.shape[:2]
        gray = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
        mask = (gray < 240).astype(np.float32)[:, :, None]
        canvas[cur_y:cur_y + lh, 80:80 + lw] = np.clip(
            canvas[cur_y:cur_y + lh, 80:80 + lw].astype(np.float32) * (1.0 - mask) + line_img.astype(np.float32) * mask,
            0, 255
        ).astype(np.uint8)
        cur_y += 85

    return canvas


@pytest.fixture
def synthetic_rotated_image(synthetic_multiline_image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Synthesize a document rotated by exactly +12.0 degrees.
    """
    angle = 12.0
    h, w = synthetic_multiline_image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        synthetic_multiline_image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return rotated, angle


@pytest.fixture
def temp_pdf_multipage() -> Generator[str, None, None]:
    """
    Create a temporary 3-page PDF file and yield its path.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    imgs = []
    for i in range(3):
        arr = np.full((500, 400, 3), 250 - i * 15, dtype=np.uint8)
        pil_img = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil_img)
        draw.text((50, 50), f"Page {i + 1} Content", fill=(20, 20, 20))
        imgs.append(pil_img)

    imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:])
    yield pdf_path

    if os.path.exists(pdf_path):
        os.remove(pdf_path)


@pytest.fixture
def sample_iam_lines_txt() -> str:
    """
    Sample text matching IAM lines.txt format.
    """
    return """# Lines format: id status threshold components x y w h transcription
a01-000u-00 ok 154 19 408 768 27 51 A|MOVE|to|stop|Mr.|Gaitskell|from
a01-000u-01 ok 154 22 395 820 30 52 nominating|any|more|Labour|life|peers
a01-000u-02 ok 154 15 410 880 25 48 is|to|be|made|at|a|meeting
a01-001u-00 ok 160 20 405 770 28 50 The|government|announced|new|reforms
a01-001u-01 ok 160 18 412 830 32 55 for|national|education|standards
b02-005a-00 ok 145 14 380 750 26 49 A|distinct|writer|sample|here
b02-005a-01 ok 145 16 390 810 29 51 with|different|handwriting|style
"""


@pytest.fixture
def sample_prescription_json() -> str:
    """
    Sample medical prescription manifest JSON string.
    """
    return json.dumps({
        "samples": [
            {
                "sample_id": "rx_00001",
                "clinic_name": "Metro Clinic",
                "doctor_name": "Dr. House, MD",
                "patient_name": "Jane Doe",
                "date": "2026-08-26",
                "items": [
                    {
                        "medication": "Amoxicillin",
                        "dosage": "500mg",
                        "sig": "1 tab tid",
                        "dispense": "#30",
                        "refills": "0"
                    }
                ]
            }
        ]
    })

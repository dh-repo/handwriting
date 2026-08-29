"""
pipeline/preprocessing/pipeline.py
Unified Preprocessing Pipeline for Digital Images and Multi-Page PDFs.
Orchestrates PDF ingestion, deskewing, illumination flattening, Sauvola
adaptive binarization, and HPP seam-carving line segmentation.
Conforms strictly to PreprocessedPage and LineCrop Pydantic models.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from pipeline.preprocessing.pdf_loader import PDFLoader
from pipeline.preprocessing.image_enhancement import ImageEnhancer
from pipeline.preprocessing.line_segmenter import LineSegmenter, LineCrop, WordCrop


class PreprocessedPage(BaseModel):
    """
    Standard preprocessed page contract connecting preprocessing to TrOCR
    training and FastAPI serving.
    """
    page_index: int
    original_image: np.ndarray  # HxWxC uint8 RGB
    enhanced_image: np.ndarray  # HxWxC uint8 RGB
    binarized_image: np.ndarray  # HxW uint8 (255=ink, 0=bg or vice versa)
    lines: List[LineCrop]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PreprocessingPipeline:
    """
    End-to-End unified pipeline orchestrator.
    """

    def __init__(
        self,
        dpi: int = 300,
        deskew: bool = True,
        enhance_contrast: bool = True,
        binarization_method: str = "sauvola",
        seam_carving: bool = True,
        min_line_height: int = 15,
        extract_words: bool = False,
        margin: int = 4
    ):
        self.dpi = dpi
        self.deskew = deskew
        self.enhance_contrast = enhance_contrast
        self.binarization_method = binarization_method

        self.loader = PDFLoader(default_dpi=dpi)
        self.enhancer = ImageEnhancer(deskew=deskew, flatten_bg=enhance_contrast)
        self.segmenter = LineSegmenter(
            min_line_height=min_line_height,
            seam_carving=seam_carving,
            margin=margin,
            extract_words=extract_words
        )

    def process_image(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path, bytes],
        page_index: int = 0
    ) -> PreprocessedPage:
        """
        Executes end-to-end preprocessing on a single image.

        Args:
            image_input: Array, PIL Image, file path, or raw image bytes.
            page_index: Index of the current page (default: 0).

        Returns:
            PreprocessedPage instance conforming to PROJECT.md interface.
        """
        # 1. Load to standard RGB uint8 array
        original_rgb = self.loader.load_single_image(image_input)

        # 2. Deskew & Orientation Correction
        if self.deskew:
            deskewed_rgb, _ = self.enhancer.deskew(original_rgb)
        else:
            deskewed_rgb = original_rgb

        # 3. Illumination Flattening & Contrast Enhancement (CLAHE + Top-Hat)
        if self.enhance_contrast:
            enhanced_rgb = self.enhancer.enhance(deskewed_rgb)
        else:
            enhanced_rgb = deskewed_rgb

        # 4. Adaptive Binarization (Sauvola with Otsu fallback)
        binarized_mask = self.enhancer.binarize(
            enhanced_rgb,
            method=self.binarization_method
        )

        # 5. Line and Word Segmentation
        lines = self.segmenter.segment(enhanced_rgb, binarized_mask)

        return PreprocessedPage(
            page_index=page_index,
            original_image=original_rgb,
            enhanced_image=enhanced_rgb,
            binarized_image=binarized_mask,
            lines=lines
        )

    def process_document(
        self,
        file_path_or_bytes: Union[str, Path, bytes],
        dpi: Optional[int] = None
    ) -> List[PreprocessedPage]:
        """
        Executes preprocessing across all pages of an image or multi-page PDF document.

        Args:
            file_path_or_bytes: Path to PDF/image or raw document bytes.
            dpi: Resolution for PDF rasterization (defaults to pipeline configured DPI).

        Returns:
            List of PreprocessedPage instances, one per page.
        """
        target_dpi = dpi if dpi is not None else self.dpi
        page_images = self.loader.load_pages(file_path_or_bytes, dpi=target_dpi)

        pages: List[PreprocessedPage] = []
        for idx, page_img in enumerate(page_images):
            page_res = self.process_image(page_img, page_index=idx)
            pages.append(page_res)

        return pages

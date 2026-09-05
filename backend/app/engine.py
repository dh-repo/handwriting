"""
backend/app/engine.py
Unified Inference Engine orchestrating Preprocessing, TrOCR MPS/CPU, and Mock Fallback.
"""

from __future__ import annotations
import base64
import hashlib
import io
import json
import concurrent.futures
import httpx
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import uuid

import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from backend.app.onnx_engine import (
    is_onnx_available,
    load_onnx_htr_model,
    ONNXRUNTIME_AVAILABLE,
)
from backend.app.ship_gate import assert_shippable_checkpoint
from pipeline.training.english_beam import (
    is_name_or_title,
    looks_like_word_crop,
    pick_crop_hypothesis,
    strip_word_decoder_punct,
)
from pipeline.training.model_contract import (
    apply_trocr_generation_config,
    load_htr_model,
    load_htr_processor,
)
from pipeline.training.vlm_refine import (
    fuse_line,
    is_page_chrome_line,
    is_page_crumb_line,
    is_stray_page_tail,
    refine_line,
    repair_line_continuations,
    should_refine_with_vlm,
    vlm_refine_available,
)

HTR_MAX_NEW_TOKENS = 64

if hasattr(torch, "set_num_threads"):
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 4))
    except Exception:
        pass

try:
    from transformers import (
        AutoTokenizer,
        RobertaTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        ViTImageProcessor,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TrOCRProcessor = Any
    VisionEncoderDecoderModel = Any

try:
    from pipeline.rescorer import (
        BeamCandidate,
        BeamRescorer,
        ContextFeatures,
        PrefixTrie,
        RescorerResult,
        VisualConfusionMatrix,
    )
    RESCORER_AVAILABLE = True
except ImportError:
    try:
        from pipeline.rescorer.beam_rescorer import (
            BeamCandidate,
            BeamRescorer,
            ContextFeatures,
            RescorerResult,
        )
        from pipeline.rescorer.confusion_matrix import VisualConfusionMatrix
        from pipeline.rescorer.trie import PrefixTrie
        RESCORER_AVAILABLE = True
    except ImportError:
        RESCORER_AVAILABLE = False
        BeamRescorer = Any
        BeamCandidate = Any
        RescorerResult = Any
        ContextFeatures = Any
        VisualConfusionMatrix = Any
        PrefixTrie = Any

try:
    from pipeline.preprocessing.image_enhancement import suppress_ruling_lines
    from pipeline.preprocessing.pipeline import PreprocessingPipeline, PreprocessedPage
    from pipeline.preprocessing.pdf_loader import (
        PDFLoader,
        load_document,
        detect_format,
        DocumentLoadingError,
        EmptyDocumentError,
        CorruptDocumentError,
        PasswordProtectedPDFError,
        UnsupportedFormatError,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    PreprocessingPipeline = Any
    PreprocessedPage = Any

    class DocumentLoadingError(Exception):
        pass

    class EmptyDocumentError(DocumentLoadingError):
        pass

    class CorruptDocumentError(DocumentLoadingError):
        pass

    class PasswordProtectedPDFError(DocumentLoadingError):
        pass

    class UnsupportedFormatError(DocumentLoadingError):
        pass

from backend.app.config import get_settings
from backend.app.schemas import (
    LineBox,
    PageResult,
    RecognitionOptions,
    RecognitionResponse,
    WordBox,
)

logger = logging.getLogger("handwriting_backend.engine")


def is_sliver_bbox(bbox: List[float]) -> bool:
    """True for leftover segmenter crumbs too thin or top/bottom binder holes."""
    if len(bbox) < 4:
        return True
    ymin, xmin, ymax, xmax = [float(value) for value in bbox[:4]]
    if (xmax - xmin) < 0.05 or (ymax - ymin) < 0.03:
        return True
    if ymax <= 0.075:
        return True
    return False


def is_page_echo(text: str, page_so_far: str) -> bool:
    """True when a later crop restates a sentence already read on this page."""
    candidate = " ".join((text or "").split()).rstrip(".,;:!?")
    prior = " ".join((page_so_far or "").split()).rstrip(".,;:!?")
    if not candidate or not prior or len(candidate.split()) < 6:
        return False
    if len(prior.split()) < 6:
        return False
    return candidate == prior or candidate in prior or prior in candidate


def _finalize_page_lines(lines_data: List[LineBox]) -> None:
    """Drop leftover UI strings first so they cannot steal the final period."""
    kept: List[LineBox] = []
    seen: set[str] = set()
    for line in lines_data:
        key = line.text.strip()
        previous = kept[-1].text if kept else ""
        page_so_far = " ".join(item.text for item in kept)
        if (
            is_page_chrome_line(line.text)
            or is_page_crumb_line(line.text, previous)
            or is_page_echo(line.text, page_so_far)
            or is_stray_page_tail(line.text, page_so_far)
            or not key
            or key in seen
        ):
            continue
        seen.add(key)
        kept.append(line)
    continued = repair_line_continuations([line.text for line in kept])
    for line, text in zip(kept, continued):
        line.text = text
    lines_data[:] = kept


def _sequence_scores(model, outputs, count):
    if getattr(outputs, "sequences_scores", None) is not None:
        return [float(x.item()) for x in outputs.sequences_scores]
    if getattr(outputs, "scores", None) and hasattr(model, "compute_transition_scores"):
        try:
            values = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            return [float(row.mean().item()) for row in values]
        except (AttributeError, ValueError, RuntimeError):
            pass
    return [None] * count


def _page_image_url(array):
    import base64, io
    buffer = io.BytesIO()
    Image.fromarray(array).convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class InferenceEngine:
    """
    Unified Handwriting Recognition Engine with Apple Silicon MPS acceleration,
    RxNorm Prefix Trie Beam Rescorer, and deterministic mock fallback.
    """

    def __init__(
        self,
        execution_mode: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        dpi: Optional[int] = None,
        enable_rescorer: Optional[bool] = None,
        beam_width: Optional[int] = None,
        vocab_dir: Optional[str] = None,
        lexicon_path: Optional[str] = None,
        confusion_matrix_path: Optional[str] = None,
        rescorer_weight: Optional[float] = None,
        context_weight: Optional[float] = None,
        confusion_weight: Optional[float] = None,
        max_safe_mg: Optional[float] = None,
        line_batch_size: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.mode = execution_mode or mode or settings.resolve_device()
        self.model_name = assert_shippable_checkpoint(model_name_or_path or settings.resolve_model_path())
        self.use_fp16 = use_fp16 if use_fp16 is not None else settings.USE_FP16
        self.dpi = dpi or settings.DEFAULT_DPI

        # Beam Search & Rescorer configuration
        if settings.AZURE_OPENAI_ENDPOINT:
            os.environ.setdefault("AZURE_OPENAI_ENDPOINT", settings.AZURE_OPENAI_ENDPOINT)
        if settings.AZURE_OPENAI_API_KEY:
            os.environ.setdefault("AZURE_OPENAI_API_KEY", settings.AZURE_OPENAI_API_KEY)
        if settings.AZURE_OPENAI_DEPLOYMENT:
            os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", settings.AZURE_OPENAI_DEPLOYMENT)

        self.enable_rescorer = enable_rescorer if enable_rescorer is not None else settings.ENABLE_RESCORER
        self.enable_vlm_refine = bool(getattr(settings, "ENABLE_VLM_REFINE", True)) and vlm_refine_available()
        self.beam_width = beam_width if beam_width is not None else settings.BEAM_WIDTH
        self.vocab_dir = vocab_dir or lexicon_path or settings.LEXICON_PATH or settings.VOCAB_DIR
        self.confusion_matrix_path = confusion_matrix_path or settings.CONFUSION_MATRIX_PATH
        self.rescorer_weight = rescorer_weight if rescorer_weight is not None else settings.RESCORER_WEIGHT
        self.context_weight = context_weight if context_weight is not None else settings.CONTEXT_WEIGHT
        self.confusion_weight = confusion_weight if confusion_weight is not None else settings.CONFUSION_WEIGHT
        self.max_safe_mg = max_safe_mg if max_safe_mg is not None else settings.MAX_SAFE_MG
        self.line_batch_size = line_batch_size or settings.LINE_BATCH_SIZE
        self.empty_cache_interval = settings.MPS_EMPTY_CACHE_INTERVAL
        self.adaptive_beam_search = getattr(settings, "ADAPTIVE_BEAM_SEARCH", True)
        self.adaptive_threshold = float(getattr(settings, "ADAPTIVE_CONFIDENCE_THRESHOLD", 0.88))
        self.htr_max_new_tokens = int(getattr(settings, "HTR_MAX_NEW_TOKENS", 64))
        self.vlm_concurrency = int(getattr(settings, "VLM_CONCURRENCY", 4))

        # Preprocessing pipeline
        if PIPELINE_AVAILABLE:
            self.pipeline = PreprocessingPipeline(dpi=self.dpi, extract_words=True)
        else:
            self.pipeline = None

        # Rescorer Instance
        self.rescorer: Optional[BeamRescorer] = None
        if self.enable_rescorer and RESCORER_AVAILABLE:
            self._init_rescorer()

        # Device & Model State
        self.device = self._resolve_torch_device()
        self.model: Optional[VisionEncoderDecoderModel] = None
        self.processor: Optional[TrOCRProcessor] = None

        if self.mode != "mock" and TRANSFORMERS_AVAILABLE:
            self._load_model()
        elif self.mode != "mock":
            raise RuntimeError("Recognition model dependencies are unavailable")

    def _init_rescorer(self) -> None:
        """Initialize PrefixTrie, VisualConfusionMatrix, and BeamRescorer."""
        try:
            vocab_dir_path = Path(self.vocab_dir) if self.vocab_dir else Path("data/reference_handwriting/vocabularies")
            trie = PrefixTrie()
            if vocab_dir_path.exists():
                trie.load_vocabularies(vocab_dir_path)
                logger.info(f"Loaded {len(trie)} vocabulary terms into PrefixTrie from {vocab_dir_path}")

            cm = VisualConfusionMatrix(load_defaults=True)
            if self.confusion_matrix_path and Path(self.confusion_matrix_path).exists():
                cm.load_json(self.confusion_matrix_path)

            dynamic_state_path = Path("data/feedback/dynamic_confusion_matrix.json")
            if dynamic_state_path.exists():
                loaded_dynamic = cm.load_dynamic_state(dynamic_state_path)
                logger.info(f"Loaded {loaded_dynamic} dynamic confusion pairs from {dynamic_state_path}")

            self.rescorer = BeamRescorer(
                trie=trie,
                confusion_matrix=cm,
                lambda_lexicon=self.rescorer_weight,
                lambda_context=self.context_weight,
                lambda_confusion=self.confusion_weight,
                vocab_dir=vocab_dir_path,
                max_safe_mg=self.max_safe_mg,
            )
            logger.info("RxNorm BeamRescorer initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize BeamRescorer ({e}). Continuing without rescoring.")
            self.rescorer = None

    def _resolve_torch_device(self) -> torch.device:
        """Resolve PyTorch target device."""
        if self.mode == "mock":
            return torch.device("cpu")
        if self.mode == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        if self.mode == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cpu")

    def _load_model(self) -> None:
        """Load TrOCR processor and VisionEncoderDecoderModel (or ONNX Runtime Engine) with polymorphic checkpoint support."""
        try:
            settings = get_settings()
            onnx_dir = getattr(settings, "ONNX_MODEL_DIR", "export/trocr_base_iam_onnx")
            use_onnx = getattr(settings, "USE_ONNX_ENGINE", False) or self.mode in ("onnx", "cpu_onnx")

            if use_onnx and onnx_dir and is_onnx_available(onnx_dir):
                logger.info(f"Loading ONNX Runtime TrOCR model from '{onnx_dir}' (threads={settings.ONNX_NUM_THREADS})...")
                self.model, self.processor = load_onnx_htr_model(
                    onnx_dir,
                    num_threads=getattr(settings, "ONNX_NUM_THREADS", 4),
                )
                self.device = torch.device("cpu")
                logger.info(f"Successfully loaded ONNX Runtime TrOCR engine from {onnx_dir}.")
                return

            logger.info(f"Loading TrOCR model '{self.model_name}' on device '{self.device}'...")

            if self.device.type == "mps":
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.7"
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(get_settings().MPS_HIGH_WATERMARK_RATIO)

            ckpt_path = Path(self.model_name)
            base_model_id = "microsoft/trocr-large-handwritten"

            if ckpt_path.is_file() and ckpt_path.suffix in (".pt", ".pth", ".bin"):
                # Standalone serialized PyTorch state dict checkpoint
                state = torch.load(str(ckpt_path), map_location="cpu")
                model_state = state.get("model_state_dict", state) if isinstance(state, dict) else state
                if isinstance(state, dict) and "config" in state and isinstance(state["config"], dict):
                    base_model_id = state["config"].get("model_name_or_path", base_model_id)

                self.processor = load_htr_processor(base_model_id)
                self.model = load_htr_model(base_model_id)
                self.model.load_state_dict(model_state, strict=False)
            else:
                self.processor = load_htr_processor(self.model_name)
                self.model = load_htr_model(self.model_name)

            # Do not force SDPA. On TrOCR-Large, flipping _attn_implementation
            # after load makes generate loop ("I don't think I've I've...") on
            # both MPS and CPU. Leave the checkpoint's default attention.

            for attr in ["max_length", "early_stopping", "no_repeat_ngram_size", "length_penalty", "num_beams"]:
                if hasattr(self.model.config, attr):
                    try:
                        delattr(self.model.config, attr)
                    except Exception:
                        pass

            if self.processor is not None:
                apply_trocr_generation_config(
                    self.model,
                    self.processor,
                    max_length=HTR_MAX_NEW_TOKENS,
                    num_beams=max(1, int(self.beam_width)),
                )

            self.model.to(self.device)

            # Precision conversion
            if self.device.type in ("mps", "cuda") and self.use_fp16:
                self.model = self.model.half()
                logger.info(f"Model converted to FP16 on {self.device.type}.")

            self.model.eval()
            logger.info(f"Successfully loaded '{self.model_name}' on {self.device}.")
        except Exception as e:
            logger.warning(f"Could not load TrOCR model ({e}). Recognition unavailable.")
            self.model = None
            self.processor = None
            raise RuntimeError("Recognition model could not be loaded") from e

    def recognize_single_crop(self, image: Image.Image, num_beams: int | None = None) -> str:
        """Same generate path as page recognition, one line crop, no VLM."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Recognition model is unavailable")
        search_beams = max(1, int(num_beams or self.beam_width or 10))
        pil = image.convert("RGB")
        inputs = self.processor(images=pil, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        if self.device.type in ("mps", "cuda") and self.use_fp16:
            pixel_values = pixel_values.half()
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": HTR_MAX_NEW_TOKENS,
        }
        if search_beams > 1:
            generate_kwargs.update(
                {
                    "num_beams": search_beams,
                    "early_stopping": True,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "num_return_sequences": search_beams,
                }
            )
        else:
            generate_kwargs.update(
                {
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
            )
        with torch.no_grad():
            outputs = self.model.generate(pixel_values, **generate_kwargs)
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        texts = [t.strip() for t in self.processor.batch_decode(sequences, skip_special_tokens=True)]
        return pick_crop_hypothesis(texts, pil.width, pil.height)

    def recognize_stream(
        self,
        file_bytes: bytes,
        filename: str = "document.png",
        options: Optional[RecognitionOptions] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream recognition events: metadata -> line events -> complete event.
        Yields JSON-compatible dictionaries suitable for SSE.
        """
        for item in self._recognize_gen(file_bytes, filename, options, is_stream=True):
            event_type = item[0]
            data = item[1]
            extra = item[2:]
            if event_type == "metadata":
                yield {"event": "metadata", "data": data}
            elif event_type == "line":
                line_box = data
                page_num = extra[0] if extra else 1
                line_dict = line_box.model_dump() if hasattr(line_box, "model_dump") else dict(line_box)
                line_dict["page_number"] = page_num
                yield {"event": "line", "data": line_dict}
            elif event_type == "complete":
                resp_dict = data.model_dump() if hasattr(data, "model_dump") else dict(data)
                yield {"event": "complete", "data": resp_dict}

    def recognize(
        self,
        file_bytes: bytes,
        filename: str = "document.png",
        options: Optional[RecognitionOptions] = None,
    ) -> RecognitionResponse:
        """
        Recognize handwritten document from raw bytes synchronously.
        Supports PNG, JPEG, TIFF, BMP, WebP, and multi-page PDFs.
        """
        for item in self._recognize_gen(file_bytes, filename, options):
            if item[0] == "complete":
                return item[1]
        raise RuntimeError("Recognition generator terminated without complete event")

    def _recognize_turbo_gen(
        self,
        file_bytes: bytes,
        filename: str,
        doc_id: str,
        t0: float,
        opts: RecognitionOptions,
        is_stream: bool = False,
    ) -> Iterator[Tuple[Any, ...]]:
        """
        Sub-3-second high-accuracy full-page cursive recognition using Azure OpenAI (gpt-4o-mini).
        Combines fast OpenCV line bounding box segmentation with holistic VLM transcription.
        """
        pages = self._load_and_preprocess(file_bytes, opts)
        if not pages:
            raise EmptyDocumentError("No readable pages detected in input document.")

        # Emit metadata event immediately
        yield ("metadata", {
            "document_id": doc_id,
            "filename": Path(filename).name,
            "total_pages": len(pages),
            "pages": [
                {
                    "page_number": p_idx + 1,
                    "image_url": _page_image_url(p.original_image),
                    "width": p.original_image.shape[1],
                    "height": p.original_image.shape[0],
                    "total_lines": len([
                        l for l in p.lines
                        if l.image is not None and l.image.size > 0 and not is_sliver_bbox(l.bbox)
                    ]),
                }
                for p_idx, p in enumerate(pages)
            ],
        })

        pages_result: List[PageResult] = []
        deployment = getattr(self.settings, "TURBO_MODEL_DEPLOYMENT", "gpt-4o-mini")
        api_key = self.settings.AZURE_OPENAI_API_KEY
        endpoint = str(self.settings.AZURE_OPENAI_ENDPOINT).rstrip("/")
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-08-01-preview"

        for page_idx, p_page in enumerate(pages):
            h, w = p_page.original_image.shape[:2]
            valid_lines = [
                line
                for line in p_page.lines
                if line.image is not None and line.image.size > 0 and not is_sliver_bbox(line.bbox)
            ]

            pil_page = Image.fromarray(p_page.original_image).convert("RGB")
            buf = io.BytesIO()
            pil_page.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            prompt = (
                "Transcribe this handwritten English document verbatim line by line. "
                "Preserve exact line breaks, spelling, punctuation, proper nouns, and signatures. "
                "Do NOT autocorrect unusual spellings, surnames, or uncommon names. "
                "Output ONLY the transcribed lines, one line per line with no extra commentary."
            )

            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert handwriting paleographer. Transcribe exact handwritten characters verbatim.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 1200,
            }

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers={"api-key": api_key}, json=payload)
                resp.raise_for_status()
                data = resp.json()

            vlm_text = data["choices"][0]["message"]["content"].strip()
            raw_lines = [l.strip() for l in vlm_text.split("\n") if l.strip()]
            if not raw_lines:
                raw_lines = [""]

            lines_data: List[LineBox] = []
            num_vlm = len(raw_lines)
            num_boxes = len(valid_lines)

            for l_idx, line_text in enumerate(raw_lines):
                line_id = f"p{page_idx + 1}_l{l_idx + 1}"
                if l_idx < num_boxes:
                    bbox = list(valid_lines[l_idx].bbox)
                elif num_boxes > 0:
                    last_bbox = valid_lines[-1].bbox
                    span = max(0.04, last_bbox[2] - last_bbox[0])
                    bbox = [
                        min(1.0, last_bbox[0] + span * 0.5),
                        last_bbox[1],
                        min(1.0, last_bbox[2] + span * 0.5),
                        last_bbox[3],
                    ]
                else:
                    bbox = [
                        float(l_idx) / max(1, num_vlm),
                        0.05,
                        float(l_idx + 1) / max(1, num_vlm),
                        0.95,
                    ]

                w_tokens = line_text.split()
                words_data: List[WordBox] = []
                w_count = len(w_tokens)
                ymin, xmin, ymax, xmax = bbox
                line_w = max(0.01, xmax - xmin)

                for w_i, w_text in enumerate(w_tokens):
                    w_xmin = xmin + line_w * (w_i / max(1, w_count))
                    w_xmax = xmin + line_w * ((w_i + 1) / max(1, w_count))
                    words_data.append(
                        WordBox(
                            word_id=f"{line_id}_w{w_i + 1}",
                            text=w_text,
                            confidence=None,
                            bbox=[ymin, w_xmin, ymax, w_xmax],
                            is_proper_noun=is_name_or_title(w_text),
                        )
                    )

                line_box = LineBox(
                    line_id=line_id,
                    text=line_text,
                    confidence=None,
                    bbox=bbox,
                    words=words_data,
                )
                lines_data.append(line_box)
                yield ("line", line_box, page_idx + 1)

            page_full_text = "\n".join(l.text for l in lines_data)
            page_res = PageResult(
                page_number=page_idx + 1,
                width=w,
                height=h,
                full_text=page_full_text,
                image_url=_page_image_url(p_page.original_image),
                mean_confidence=None,
                lines=lines_data,
            )
            pages_result.append(page_res)

        elapsed_ms = (time.time() - t0) * 1000
        total_response = RecognitionResponse(
            document_id=doc_id,
            filename=Path(filename).name,
            total_pages=len(pages_result),
            pages=pages_result,
            processing_time_ms=elapsed_ms,
            preprocessing_flags={
                "deskew": opts.deskew,
                "enhance_contrast": opts.enhance_contrast,
                "binarization": opts.binarization_method,
                "engine": "turbo-vlm",
                "deployment": deployment,
            },
            engine_used="turbo-vlm",
            model_id=deployment, processing_location="cloud",
        )
        yield ("complete", total_response)

    def _recognize_gen(
        self,
        file_bytes: bytes,
        filename: str = "document.png",
        options: Optional[RecognitionOptions] = None,
        is_stream: bool = False,
    ) -> Iterator[Tuple[Any, ...]]:
        t0 = time.time()

        if not file_bytes or len(file_bytes) == 0:
            raise EmptyDocumentError("Input file is empty (0 bytes).")

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        opts = options or RecognitionOptions()

        if opts.processing_mode == "cloud":
            if not self.settings.ENABLE_TURBO_MODE or not (
                self.settings.AZURE_OPENAI_ENDPOINT and self.settings.AZURE_OPENAI_API_KEY
            ):
                raise ValueError("Cloud recognition is not enabled on this server")
            yield from self._recognize_turbo_gen(file_bytes, filename, doc_id, t0, opts, is_stream=is_stream)
            return
        if self.mode != "mock" and (self.model is None or self.processor is None):
            raise RuntimeError("Recognition model is unavailable")

        # If running in mock mode or model not loaded
        if self.mode == "mock" or self.model is None or self.processor is None:
            mock_res = self._recognize_mock(file_bytes, filename, doc_id, t0, opts)
            mock_res.is_demo = True
            mock_res.engine_used = "demo"
            yield ("metadata", {
                "document_id": doc_id,
                "filename": Path(filename).name,
                "total_pages": mock_res.total_pages,
                "pages": [
                    {
                        "page_number": p.page_number,
                        "width": p.width,
                        "height": p.height,
                        "total_lines": len(p.lines),
                    }
                    for p in mock_res.pages
                ],
            })
            for p in mock_res.pages:
                for l in p.lines:
                    yield ("line", l, p.page_number)
            yield ("complete", mock_res)
            return

        # Real TrOCR Pipeline Routing
        try:
            pages = self._load_and_preprocess(file_bytes, opts)
            pages_result: List[PageResult] = []

            # Determine rescoring & beam search parameters
            opt_beam_width = getattr(opts, "beam_width", self.beam_width) or self.beam_width
            opt_rescore = getattr(opts, "rescore", False)
            effective_rescore = bool(self.enable_rescorer and opt_rescore and (self.rescorer is not None))
            search_beams = max(1, int(opt_beam_width))
            effective_k = search_beams if search_beams > 1 else 1
            use_adaptive = (
                bool(getattr(opts, "adaptive", True))
                and getattr(self, "adaptive_beam_search", True)
                and search_beams > 1
                and not effective_rescore
            )
            max_tokens = min(self.htr_max_new_tokens, 64)

            # Emit metadata event immediately after preprocessing
            yield ("metadata", {
                "document_id": doc_id,
                "filename": Path(filename).name,
                "total_pages": len(pages),
                "pages": [
                    {
                        "page_number": p_idx + 1,
                        "image_url": _page_image_url(p.original_image),
                    "width": p.original_image.shape[1],
                        "height": p.original_image.shape[0],
                        "total_lines": len([
                            l for l in p.lines
                            if l.image is not None and l.image.size > 0 and not is_sliver_bbox(l.bbox)
                        ]),
                    }
                    for p_idx, p in enumerate(pages)
                ],
            })

            for page_idx, p_page in enumerate(pages):
                h, w = p_page.original_image.shape[:2]
                lines_data: List[LineBox] = []
                page_text_so_far = ""

                valid_lines = [
                    line
                    for line in p_page.lines
                    if line.image is not None
                    and line.image.size > 0
                    and not is_sliver_bbox(line.bbox)
                ]
                batch_size = 1 if is_stream else (min(2, max(1, self.line_batch_size)) if self.device.type == "cpu" else max(1, self.line_batch_size))

                for b_start in range(0, len(valid_lines), batch_size):
                    b_lines = valid_lines[b_start : b_start + batch_size]
                    pil_crops = [
                        Image.fromarray(suppress_ruling_lines(l.image)).convert("RGB")
                        for l in b_lines
                    ]

                    inputs = self.processor(images=pil_crops, return_tensors="pt")
                    pixel_values = inputs.pixel_values.to(self.device)

                    if self.device.type in ("mps", "cuda") and self.use_fp16:
                        pixel_values = pixel_values.half()

                    batch_decoded_texts: List[str] = []
                    batch_scores: List[float] = []

                    if use_adaptive:
                        # Pass 1: Fast greedy decoding
                        with torch.no_grad():
                            outputs_greedy = self.model.generate(
                                pixel_values,
                                max_new_tokens=max_tokens,
                                return_dict_in_generate=True,
                                output_scores=True,
                            )
                        greedy_texts = [
                            t.strip()
                            for t in self.processor.batch_decode(outputs_greedy.sequences, skip_special_tokens=True)
                        ]
                        greedy_seq_scores = _sequence_scores(self.model, outputs_greedy, len(greedy_texts))

                        batch_decoded_texts = list(greedy_texts)
                        batch_scores = list(greedy_seq_scores)

                        # Check which lines require beam escalation
                        escalate_indices = []
                        for s_idx, (g_txt, g_score) in enumerate(zip(greedy_texts, greedy_seq_scores)):
                            conf = math.exp(min(0.0, g_score)) if g_score is not None else 0.0
                            if (conf < self.adaptive_threshold) or should_refine_with_vlm(g_txt):
                                escalate_indices.append(s_idx)

                        if escalate_indices:
                            esc_pixels = pixel_values[escalate_indices]
                            with torch.no_grad():
                                outputs_beam = self.model.generate(
                                    esc_pixels,
                                    num_beams=search_beams,
                                    num_return_sequences=effective_k,
                                    max_new_tokens=max_tokens,
                                    early_stopping=True,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                )
                            beam_texts = [
                                t.strip()
                                for t in self.processor.batch_decode(outputs_beam.sequences, skip_special_tokens=True)
                            ]
                            beam_seq_scores = _sequence_scores(self.model, outputs_beam, len(beam_texts))

                            for esc_pos, orig_idx in enumerate(escalate_indices):
                                sub_cands = beam_texts[esc_pos * effective_k : (esc_pos + 1) * effective_k]
                                sub_scores = beam_seq_scores[esc_pos * effective_k : (esc_pos + 1) * effective_k]
                                chosen = pick_crop_hypothesis(
                                    sub_cands,
                                    pil_crops[orig_idx].width,
                                    pil_crops[orig_idx].height,
                                    page_text_so_far,
                                )
                                batch_decoded_texts[orig_idx] = chosen
                                if sub_scores:
                                    batch_scores[orig_idx] = sub_scores[0]
                    else:
                        with torch.no_grad():
                            if search_beams > 1:
                                generate_kwargs: Dict[str, Any] = {
                                    "num_beams": search_beams,
                                    "max_new_tokens": max_tokens,
                                    "early_stopping": True,
                                    "return_dict_in_generate": True,
                                    "output_scores": True,
                                }
                                if effective_k > 1:
                                    generate_kwargs["num_return_sequences"] = effective_k
                                outputs = self.model.generate(pixel_values, **generate_kwargs)
                            else:
                                outputs = self.model.generate(
                                    pixel_values,
                                    max_new_tokens=max_tokens,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                )

                        decoded_texts = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
                        seq_scores = _sequence_scores(self.model, outputs, len(decoded_texts))

                        for l_sub_idx in range(len(b_lines)):
                            if effective_k > 1 and effective_rescore:
                                line_beam_texts = decoded_texts[l_sub_idx * effective_k : (l_sub_idx + 1) * effective_k]
                                line_beam_scores = seq_scores[l_sub_idx * effective_k : (l_sub_idx + 1) * effective_k]
                                candidates = [
                                    BeamCandidate(text=b_txt.strip(), log_prob=float(b_s))
                                    for b_txt, b_s in zip(line_beam_texts, line_beam_scores)
                                ]
                                rescore_res = self.rescorer.rescore_detailed(candidates)
                                text = rescore_res.rescored_text
                                if looks_like_word_crop(pil_crops[l_sub_idx].width, pil_crops[l_sub_idx].height):
                                    text = strip_word_decoder_punct(text)
                                batch_decoded_texts.append(text)
                                batch_scores.append(float(rescore_res.confidence))
                            elif effective_k > 1:
                                line_beam_texts = [
                                    t.strip()
                                    for t in decoded_texts[l_sub_idx * effective_k : (l_sub_idx + 1) * effective_k]
                                ]
                                text = pick_crop_hypothesis(
                                    line_beam_texts,
                                    pil_crops[l_sub_idx].width,
                                    pil_crops[l_sub_idx].height,
                                    page_text_so_far,
                                )
                                batch_decoded_texts.append(text)
                                batch_scores.append(seq_scores[l_sub_idx * effective_k] if seq_scores else 0.0)
                            else:
                                text = pick_crop_hypothesis(
                                    [decoded_texts[l_sub_idx]],
                                    pil_crops[l_sub_idx].width,
                                    pil_crops[l_sub_idx].height,
                                )
                                batch_decoded_texts.append(text)
                                batch_scores.append(seq_scores[l_sub_idx] if seq_scores else 0.0)

                    # Step 2: Parallel VLM Refinement
                    vlm_candidates = []
                    for l_sub_idx, t_text in enumerate(batch_decoded_texts):
                        if (
                            False  # General local recognition never invokes a remote referee.
                            and self.mode != "mock"
                            and should_refine_with_vlm(t_text)
                        ):
                            vlm_candidates.append((l_sub_idx, pil_crops[l_sub_idx], t_text, page_text_so_far))

                    if vlm_candidates:
                        max_workers = min(self.vlm_concurrency, len(vlm_candidates))
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                            fut_map = {
                                pool.submit(refine_line, crop, hypothesis="", previous_text=p_text): (idx, orig_text)
                                for (idx, crop, orig_text, p_text) in vlm_candidates
                            }
                            for fut in concurrent.futures.as_completed(fut_map):
                                idx, orig_text = fut_map[fut]
                                try:
                                    vlm_text = fut.result()
                                    fused = fuse_line(orig_text, vlm_text)
                                    if vlm_text and fused != orig_text:
                                        logger.info("vlm fuse trocr=%r vlm=%r out=%r", orig_text, vlm_text, fused)
                                    batch_decoded_texts[idx] = fused
                                except Exception:
                                    logger.exception("vlm refine failed")

                    # Step 3: Format line boxes and word boxes
                    for l_sub_idx, line in enumerate(b_lines):
                        global_l_idx = b_start + l_sub_idx
                        crop = pil_crops[l_sub_idx]
                        text = batch_decoded_texts[l_sub_idx]
                        top_s = batch_scores[l_sub_idx]
                        line_conf = math.exp(min(0.0, top_s)) if top_s is not None else None

                        if not text:
                            continue

                        ymin, xmin, ymax, xmax = [max(0.0, min(1.0, float(c))) for c in line.bbox]
                        if ymin >= ymax:
                            ymax = min(1.0, ymin + 0.05)
                        if xmin >= xmax:
                            xmax = min(1.0, xmin + 0.1)

                        word_tokens = text.split()
                        words_data: List[WordBox] = []
                        if line.words and len(line.words) == len(word_tokens):
                            for w_idx, wc in enumerate(line.words):
                                w_text = word_tokens[w_idx]
                                w_ymin, w_xmin, w_ymax, w_xmax = [max(0.0, min(1.0, float(c))) for c in wc.bbox]
                                if w_ymin >= w_ymax:
                                    w_ymax = min(1.0, w_ymin + 0.04)
                                if w_xmin >= w_xmax:
                                    w_xmax = min(1.0, w_xmin + 0.05)
                                words_data.append(
                                    WordBox(
                                        word_id=f"p{page_idx+1}_l{global_l_idx+1}_w{w_idx+1}",
                                        text=w_text,
                                        confidence=round(line_conf, 3) if line_conf is not None else None,
                                        bbox=[w_ymin, w_xmin, w_ymax, w_xmax],
                                        is_proper_noun=is_name_or_title(w_text),
                                    )
                                )
                        else:
                            n_w = max(1, len(word_tokens))
                            w_span = (xmax - xmin) / n_w
                            for w_idx, wt in enumerate(word_tokens):
                                w_xmin = round(xmin + w_idx * w_span, 4)
                                w_xmax = round(min(xmax, w_xmin + w_span * 0.95), 4)
                                if w_xmin >= w_xmax:
                                    w_xmax = min(1.0, w_xmin + 0.02)
                                words_data.append(
                                    WordBox(
                                        word_id=f"p{page_idx+1}_l{global_l_idx+1}_w{w_idx+1}",
                                        text=wt,
                                        confidence=round(line_conf, 3) if line_conf is not None else None,
                                        bbox=[ymin, w_xmin, ymax, w_xmax],
                                        is_proper_noun=is_name_or_title(wt),
                                    )
                                )

                        if text and lines_data and lines_data[-1].text.strip() == text.strip():
                            continue
                        if is_page_echo(text, page_text_so_far):
                            continue

                        box = LineBox(
                            line_id=f"p{page_idx+1}_l{global_l_idx+1}",
                            text=text,
                            confidence=round(line_conf, 3) if line_conf is not None else None,
                            bbox=[ymin, xmin, ymax, xmax],
                            words=words_data,
                        )
                        lines_data.append(box)
                        page_text_so_far = " ".join(item.text for item in lines_data)

                        # Emit line event
                        yield ("line", box, page_idx + 1)

                    if self.device.type == "mps" and hasattr(torch, "mps"):
                        torch.mps.empty_cache()

                _finalize_page_lines(lines_data)
                full_text = "\n".join(l.text for l in lines_data)
                mean_conf = float(np.mean([l.confidence for l in lines_data])) if lines_data and all(l.confidence is not None for l in lines_data) else None
                mean_conf = max(0.0, min(1.0, mean_conf)) if mean_conf is not None else None

                pages_result.append(
                    PageResult(
                        page_number=page_idx + 1,
                        width=w,
                        height=h,
                        full_text=full_text,
                        mean_confidence=round(mean_conf, 3) if mean_conf is not None else None,
                        image_url=_page_image_url(p_page.original_image),
                        lines=lines_data,
                    )
                )

            elapsed_ms = (time.time() - t0) * 1000.0
            resp = RecognitionResponse(
                document_id=doc_id,
                filename=Path(filename).name,
                total_pages=len(pages_result),
                pages=pages_result,
                processing_time_ms=round(elapsed_ms, 2),
                model_id=self.model_name, processing_location="local",
                preprocessing_flags=opts.model_dump(),
            )
            yield ("complete", resp)
        except (DocumentLoadingError, EmptyDocumentError, CorruptDocumentError, PasswordProtectedPDFError, UnsupportedFormatError):
            raise
        except Exception as exc:
            logger.error(f"Inference error on {filename}: {exc}", exc_info=True)
            raise RuntimeError("Recognition failed; no transcript was produced") from exc

    def _load_and_preprocess(self, file_bytes: bytes, opts: RecognitionOptions) -> List[PreprocessedPage]:
        """Load and preprocess document using M1 PreprocessingPipeline."""
        if not PIPELINE_AVAILABLE:
            raise DocumentLoadingError("Preprocessing pipeline unavailable.")
        pipeline = PreprocessingPipeline(
            dpi=opts.dpi,
            deskew=opts.deskew,
            enhance_contrast=opts.enhance_contrast,
            binarization_method=opts.binarization_method,
            extract_words=opts.extract_words,
        )
        return pipeline.process_document(file_bytes, dpi=opts.dpi)

    def _recognize_mock(
        self,
        file_bytes: bytes,
        filename: str,
        doc_id: str,
        t0: float,
        opts: RecognitionOptions,
    ) -> RecognitionResponse:
        """Deterministic mock recognition engine for tests, CI, and serverless preview."""
        # Check if test fixture MockInferenceEngine exists and can handle this file
        try:
            from tests.fixtures.mock_engine import MockInferenceEngine
            mock_eng = MockInferenceEngine()
            resp = mock_eng.recognize(file_bytes, filename)
            resp.document_id = doc_id
            resp.processing_time_ms = round((time.time() - t0) * 1000.0, 2)
            resp.preprocessing_flags = opts.model_dump()
            return resp
        except Exception:
            pass

        # Standalone Internal Deterministic Mock
        return self._generate_internal_mock(file_bytes, filename, doc_id, t0, opts)

    def _generate_internal_mock(
        self,
        file_bytes: bytes,
        filename: str,
        doc_id: str,
        t0: float,
        opts: RecognitionOptions,
    ) -> RecognitionResponse:
        """Generate high-fidelity deterministic transcription without external dependencies."""
        # Attempt to inspect with PIL / pypdfium2 to detect page count and dimensions
        pages_meta: List[Tuple[int, int]] = []
        is_pdf = file_bytes.startswith(b"%PDF-")

        if is_pdf:
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(file_bytes)
                n_pages = len(pdf)
                for p_i in range(n_pages):
                    page = pdf[p_i]
                    pw = int(page.get_width() * (opts.dpi / 72.0))
                    ph = int(page.get_height() * (opts.dpi / 72.0))
                    pages_meta.append((pw, ph))
            except Exception as e:
                # Corrupt PDF detection
                raise CorruptDocumentError(f"Corrupt or invalid PDF file: {e}")
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes))
                pages_meta.append((img.width, img.height))
            except Exception as e:
                raise CorruptDocumentError(f"Cannot decode image file: {e}")

        if len(pages_meta) == 0:
            raise EmptyDocumentError("Input document has 0 readable pages.")

        file_hash = hashlib.sha256(file_bytes[:1024]).hexdigest()
        seed_val = int(file_hash[:8], 16)
        rng = np.random.RandomState(seed_val)

        sample_lexicon = [
            ["Rx:", "Amoxicillin", "500mg", "capsules"],
            ["Sig:", "Take", "1", "capsule", "every", "8", "hours"],
            ["Dispense:", "#30", "thirty", "capsules"],
            ["Refills:", "2", "times", "PRN"],
            ["Dr.", "Sarah", "Smith,", "MD"],
            ["License:", "ME-94821", "DEA:", "BS82109"],
            ["Patient:", "John", "Doe", "DOB:", "1980-05-14"],
            ["Date:", "2026-08-26", "Clinic:", "Metro", "Health"],
        ]

        pages_result: List[PageResult] = []
        for p_idx, (pw, ph) in enumerate(pages_meta):
            lines_data: List[LineBox] = []
            num_lines = min(len(sample_lexicon), max(2, int(ph / 200)))
            y_start = 0.08
            y_step = min(0.10, (0.85 - y_start) / max(1, num_lines))

            for l_idx in range(num_lines):
                line_words = sample_lexicon[l_idx % len(sample_lexicon)]
                line_text = " ".join(line_words)
                ymin = round(y_start + l_idx * y_step, 4)
                ymax = round(ymin + y_step * 0.75, 4)
                xmin = 0.08
                xmax = 0.88
                l_conf = round(float(rng.uniform(0.92, 0.98)), 3)

                words_data: List[WordBox] = []
                w_span = (xmax - xmin) / len(line_words)
                for w_idx, w_str in enumerate(line_words):
                    w_xmin = round(xmin + w_idx * w_span, 4)
                    w_xmax = round(w_xmin + w_span * 0.9, 4)
                    w_conf = round(float(np.clip(l_conf + rng.uniform(-0.02, 0.02), 0.85, 0.99)), 3)
                    words_data.append(
                        WordBox(
                            word_id=f"p{p_idx+1}_l{l_idx+1}_w{w_idx+1}",
                            text=w_str,
                            confidence=w_conf,
                            bbox=[ymin, w_xmin, ymax, w_xmax],
                            is_proper_noun=is_name_or_title(w_str),
                        )
                    )

                lines_data.append(
                    LineBox(
                        line_id=f"p{p_idx+1}_l{l_idx+1}",
                        text=line_text,
                        confidence=l_conf,
                        bbox=[ymin, xmin, ymax, xmax],
                        words=words_data,
                    )
                )

            _finalize_page_lines(lines_data)
            full_text = "\n".join(l.text for l in lines_data)
            mean_conf = float(np.mean([l.confidence for l in lines_data])) if lines_data else 1.0

            pages_result.append(
                PageResult(
                    page_number=p_idx + 1,
                    width=pw,
                    height=ph,
                    full_text=full_text,
                    mean_confidence=round(mean_conf, 3),
                    lines=lines_data,
                )
            )

        elapsed_ms = (time.time() - t0) * 1000.0
        return RecognitionResponse(
            document_id=doc_id,
            filename=Path(filename).name,
            total_pages=len(pages_result),
            pages=pages_result,
            processing_time_ms=round(elapsed_ms, 2),
            preprocessing_flags=opts.model_dump(),
        )

    def adapt_confusion_matrix(
        self,
        original_prediction: str,
        operator_correction: str,
        learning_rate: float = 0.20,
        min_cost: float = 0.15,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Adapt live visual confusion matrix costs based on human operator correction.
        Delegates directly to self.rescorer.confusion_matrix.adapt_from_correction() if rescorer is active.
        """
        if self.rescorer is None or not hasattr(self.rescorer, "confusion_matrix") or self.rescorer.confusion_matrix is None:
            logger.warning("BeamRescorer or VisualConfusionMatrix is not active; dynamic adaptation skipped.")
            return []

        return self.rescorer.confusion_matrix.adapt_from_correction(
            original_prediction=original_prediction,
            operator_correction=operator_correction,
            learning_rate=learning_rate,
            min_cost=min_cost,
            **kwargs,
        )


_engine_singleton: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """Return or initialize global InferenceEngine singleton."""
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = InferenceEngine()
    return _engine_singleton


def set_engine(engine: InferenceEngine) -> None:
    """Explicitly set the engine singleton (used in test harnesses)."""
    global _engine_singleton
    _engine_singleton = engine


def reset_engine() -> None:
    """Reset the engine singleton."""
    global _engine_singleton
    _engine_singleton = None

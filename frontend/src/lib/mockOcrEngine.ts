import { DocumentOCRResult, LineItem, PageResult, WordToken, BoundingBoxTuple } from "../types/ocr";
import { SAMPLE_PRESETS, SAMPLE_CLEAN_CURSIVE, SAMPLE_MESSY_CURSIVE, SAMPLE_PRESCRIPTION, SAMPLE_MULTIPAGE } from "./sampleDocuments";
import { bboxToPolygon } from "./transformUtils";

interface MockRecognitionOptions {
  filename?: string;
  sampleId?: string;
  mimeType?: string;
  fileSize?: number;
  imageUrl?: string;
}

const SYNTHETIC_SENTENCES = [
  "Comprehensive patient assessment and clinical notes recorded during session.",
  "Follow-up examination reveals steady recovery and normalized parameters.",
  "Continue regular treatment protocol and monitor vital indicators weekly.",
  "Consultation summary validated with clinical staff and attending physician.",
  "Prescribed therapeutic regimen to be maintained until subsequent review.",
];

export async function runMockOcr(options: MockRecognitionOptions): Promise<DocumentOCRResult> {
  const startTime = Date.now();

  // 1. Check for explicit sample ID
  if (options.sampleId && SAMPLE_PRESETS[options.sampleId]) {
    const matched = structuredClone(SAMPLE_PRESETS[options.sampleId]);
    matched.processing_time_ms = Math.max(12, Date.now() - startTime + 85);
    matched.is_mock = true;
    return matched;
  }

  // 2. Check filename for known patterns
  const fn = (options.filename || "").toLowerCase();
  if (fn.includes("clean") && fn.includes("cursive")) {
    const matched = structuredClone(SAMPLE_CLEAN_CURSIVE);
    matched.filename = options.filename || matched.filename;
    matched.processing_time_ms = Math.max(15, Date.now() - startTime + 90);
    matched.is_mock = true;
    return matched;
  }
  if (fn.includes("messy") || fn.includes("consult")) {
    const matched = structuredClone(SAMPLE_MESSY_CURSIVE);
    matched.filename = options.filename || matched.filename;
    matched.processing_time_ms = Math.max(15, Date.now() - startTime + 95);
    matched.is_mock = true;
    return matched;
  }
  if (fn.includes("prescript") || fn.includes("rx")) {
    const matched = structuredClone(SAMPLE_PRESCRIPTION);
    matched.filename = options.filename || matched.filename;
    matched.processing_time_ms = Math.max(15, Date.now() - startTime + 110);
    matched.is_mock = true;
    return matched;
  }
  if (fn.includes("multi") || fn.endsWith(".pdf")) {
    const matched = structuredClone(SAMPLE_MULTIPAGE);
    matched.filename = options.filename || matched.filename;
    matched.processing_time_ms = Math.max(25, Date.now() - startTime + 210);
    matched.is_mock = true;
    return matched;
  }

  // 3. Synthesize structured OCR layout for arbitrary uploaded image
  const docId = `doc_${Math.random().toString(36).substring(2, 9)}`;
  const width = 1200;
  const height = 1600;
  const numLines = Math.min(SYNTHETIC_SENTENCES.length, 5);

  const lines: LineItem[] = [];
  const lineYStart = 0.15;
  const lineSpacing = 0.14;
  const lineHeight = 0.06;

  for (let i = 0; i < numLines; i++) {
    const lineId = `p1_l${i + 1}`;
    const text = SYNTHETIC_SENTENCES[i];
    const ymin = +(lineYStart + i * lineSpacing).toFixed(4);
    const ymax = +(ymin + lineHeight).toFixed(4);
    const xmin = 0.10;
    const xmax = 0.90;
    const lineBbox: BoundingBoxTuple = [ymin, xmin, ymax, xmax];

    const wordsRaw = text.split(" ");
    const totalChars = wordsRaw.reduce((sum, w) => sum + w.length + 1, 0);
    let curFrac = 0;

    const words: WordToken[] = wordsRaw.map((word, wIdx) => {
      const wordId = `${lineId}_w${wIdx + 1}`;
      const frac = (word.length + 0.5) / totalChars;
      const wxMin = +(xmin + curFrac * (xmax - xmin)).toFixed(4);
      const wxMax = +(Math.min(xmax, wxMin + frac * (xmax - xmin))).toFixed(4);
      curFrac += frac;

      const wordBbox: BoundingBoxTuple = [ymin, wxMin, ymax, wxMax];
      const isLow = (i === 1 && wIdx === 2) || (i === 3 && wIdx === 4);
      const confidence = isLow ? +(0.64 + Math.random() * 0.05).toFixed(3) : +(0.92 + Math.random() * 0.07).toFixed(3);

      return {
        word_id: wordId,
        text: word,
        original_text: word,
        confidence,
        bbox: wordBbox,
        polygon: bboxToPolygon(wordBbox),
        is_low_confidence: confidence < 0.70,
        is_edited: false,
      };
    });

    const lineConf = +(words.reduce((sum, w) => sum + (w.confidence ?? 0), 0) / words.length).toFixed(3);

    lines.push({
      line_id: lineId,
      line_index: i,
      line_number: i + 1,
      text,
      original_text: text,
      confidence: lineConf,
      bbox: lineBbox,
      polygon: bboxToPolygon(lineBbox),
      words,
      is_edited: false,
    });
  }

  const meanConf = +(lines.reduce((sum, l) => sum + (l.confidence ?? 0), 0) / lines.length).toFixed(3);
  const fullText = lines.map((l) => l.text).join("\n");

  const page: PageResult = {
    page_number: 1,
    width,
    height,
    mean_confidence: meanConf,
    confidence: meanConf,
    image_url: options.imageUrl || "/samples/sample_clean_cursive.png",
    full_text: fullText,
    lines,
  };

  return {
    document_id: docId,
    filename: options.filename || "handwriting_upload.png",
    mime_type: options.mimeType || "image/png",
    total_pages: 1,
    processing_time_ms: Math.max(18, Date.now() - startTime + 140),
    model_version: "trocr-handwritten-mps-v1",
    mean_confidence: meanConf,
    overall_confidence: meanConf,
    full_text: fullText,
    pages: [page],
    is_mock: true,
  };
}

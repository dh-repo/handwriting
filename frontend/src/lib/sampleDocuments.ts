import { DocumentOCRResult, LineItem, WordToken, BoundingBoxTuple } from "../types/ocr";
import { bboxToPolygon } from "./transformUtils";

function buildWordsFromLine(
  lineId: string,
  lineText: string,
  lineBbox: BoundingBoxTuple,
  baseConfidence: number,
  wordOverrides?: Record<string, number>
): WordToken[] {
  const rawWords = lineText.trim().split(/\s+/);
  if (rawWords.length === 0) return [];

  const [ymin, xmin, ymax, xmax] = lineBbox;
  const lineWidth = xmax - xmin;
  const totalChars = rawWords.reduce((sum, w) => sum + w.length + 1, 0);

  let currentFraction = 0;

  return rawWords.map((word, idx) => {
    const wordFraction = (word.length + 0.5) / totalChars;
    const wordXMin = xmin + currentFraction * lineWidth;
    const wordXMax = Math.min(xmax, wordXMin + wordFraction * lineWidth);
    currentFraction += wordFraction;

    const wordBbox: BoundingBoxTuple = [ymin, wordXMin, ymax, wordXMax];
    const confidence = wordOverrides && wordOverrides[word] !== undefined
      ? wordOverrides[word]
      : Math.max(0.60, Math.min(0.99, +(baseConfidence + (Math.sin(idx * 1.5) * 0.04)).toFixed(3)));

    return {
      word_id: `${lineId}_w${idx + 1}`,
      text: word,
      original_text: word,
      confidence,
      bbox: wordBbox,
      polygon: bboxToPolygon(wordBbox),
      is_low_confidence: confidence < 0.70,
      is_edited: false,
    };
  });
}

function createLine(
  lineId: string,
  lineIndex: number,
  text: string,
  bbox: BoundingBoxTuple,
  confidence: number,
  wordOverrides?: Record<string, number>
): LineItem {
  const words = buildWordsFromLine(lineId, text, bbox, confidence, wordOverrides);
  return {
    line_id: lineId,
    line_index: lineIndex,
    line_number: lineIndex + 1,
    text,
    original_text: text,
    confidence,
    bbox,
    polygon: bboxToPolygon(bbox),
    words,
    is_edited: false,
  };
}

// 1. Clean Cursive Sample
export const SAMPLE_CLEAN_CURSIVE: DocumentOCRResult = {
  document_id: "sample_clean_cursive",
  filename: "sample_clean_cursive.png",
  mime_type: "image/png",
  total_pages: 1,
  processing_time_ms: 184.2,
  model_version: "trocr-handwritten-mps-v1",
  mean_confidence: 0.962,
  overall_confidence: 0.962,
  is_mock: true,
  full_text: [
    "The quick brown fox jumps over the lazy dog.",
    "Careful cursive handwriting practice improves clarity.",
    "Every stroke and curve conveys personal character.",
    "Signed with precision and steady flow.",
  ].join("\n"),
  pages: [
    {
      page_number: 1,
      width: 800,
      height: 600,
      mean_confidence: 0.962,
      confidence: 0.962,
      image_url: "/samples/sample_clean_cursive.png",
      full_text: [
        "The quick brown fox jumps over the lazy dog.",
        "Careful cursive handwriting practice improves clarity.",
        "Every stroke and curve conveys personal character.",
        "Signed with precision and steady flow.",
      ].join("\n"),
      lines: [
        createLine("p1_l1", 0, "The quick brown fox jumps over the lazy dog.", [0.23667, 0.17875, 0.29667, 0.9675], 0.978),
        createLine("p1_l2", 1, "Careful cursive handwriting practice improves clarity.", [0.40333, 0.1775, 0.46, 0.96625], 0.965),
        createLine("p1_l3", 2, "Every stroke and curve conveys personal character.", [0.57333, 0.17875, 0.63333, 0.965], 0.958),
        createLine("p1_l4", 3, "Signed with precision and steady flow.", [0.735, 0.175, 0.8, 0.9675], 0.947),
      ],
    },
  ],
};

// 2. Messy Cursive Consultation Note Sample (with low-confidence tokens for Speed Review)
export const SAMPLE_MESSY_CURSIVE: DocumentOCRResult = {
  document_id: "sample_messy_cursive",
  filename: "sample_messy_cursive.png",
  mime_type: "image/png",
  total_pages: 1,
  processing_time_ms: 228.6,
  model_version: "trocr-handwritten-mps-v1",
  mean_confidence: 0.835,
  overall_confidence: 0.835,
  is_mock: true,
  full_text: [
    "Patient reported recurring headaches since Tuesday",
    "Neurological exam unremarkable but symptoms continue",
    "Recommend MRI scan of brain and cervical spine",
    "Review results in follow-up appointment next week",
  ].join("\n"),
  pages: [
    {
      page_number: 1,
      width: 800,
      height: 600,
      mean_confidence: 0.835,
      confidence: 0.835,
      image_url: "/samples/sample_messy_cursive.png",
      full_text: [
        "Patient reported recurring headaches since Tuesday",
        "Neurological exam unremarkable but symptoms continue",
        "Recommend MRI scan of brain and cervical spine",
        "Review results in follow-up appointment next week",
      ].join("\n"),
      lines: [
        createLine("p1_l1", 0, "Patient reported recurring headaches since Tuesday", [0.21667, 0.12375, 0.28167, 0.91125], 0.865, { headaches: 0.68, Tuesday: 0.89 }),
        createLine("p1_l2", 1, "Neurological exam unremarkable but symptoms continue", [0.405, 0.125, 0.46, 0.9125], 0.842, { unremarkable: 0.72, symptoms: 0.66 }),
        createLine("p1_l3", 2, "Recommend MRI scan of brain and cervical spine", [0.58167, 0.1175, 0.63667, 0.91], 0.812, { cervical: 0.64, MRI: 0.91 }),
        createLine("p1_l4", 3, "Review results in follow-up appointment next week", [0.76667, 0.1175, 0.835, 0.91], 0.821, { "follow-up": 0.67 }),
      ],
    },
  ],
};

// 3. Medical Prescription Slip Sample
export const SAMPLE_PRESCRIPTION: DocumentOCRResult = {
  document_id: "sample_prescription",
  filename: "sample_prescription.png",
  mime_type: "image/png",
  total_pages: 1,
  processing_time_ms: 256.4,
  model_version: "trocr-handwritten-mps-v1",
  mean_confidence: 0.938,
  overall_confidence: 0.938,
  is_mock: true,
  full_text: [
    "Amoxicillin 500mg capsules",
    "Sig: Take 1 cap po tid x 10 days",
    "Disp: #30    Refills: 0",
    "Ibuprofen 600mg tablets",
    "Sig: Take 1 tab po q6h prn pain / fever",
    "Disp: #20    Refills: 1",
  ].join("\n"),
  pages: [
    {
      page_number: 1,
      width: 800,
      height: 1100,
      mean_confidence: 0.938,
      confidence: 0.938,
      image_url: "/samples/sample_prescription.png",
      full_text: [
        "Amoxicillin 500mg capsules",
        "Sig: Take 1 cap po tid x 10 days",
        "Disp: #30    Refills: 0",
        "Ibuprofen 600mg tablets",
        "Sig: Take 1 tab po q6h prn pain / fever",
        "Disp: #20    Refills: 1",
      ].join("\n"),
      lines: [
        createLine("p1_l1", 0, "Amoxicillin 500mg capsules", [0.37909, 0.095, 0.40818, 0.68375], 0.952),
        createLine("p1_l2", 1, "Sig: Take 1 cap po tid x 10 days", [0.43909, 0.0975, 0.46909, 0.78375], 0.915, { po: 0.67, tid: 0.69 }),
        createLine("p1_l3", 2, "Disp: #30    Refills: 0", [0.49455, 0.10125, 0.52818, 0.565], 0.962),
        createLine("p1_l4", 3, "Ibuprofen 600mg tablets", [0.55545, 0.10125, 0.58636, 0.59375], 0.948),
        createLine("p1_l5", 4, "Sig: Take 1 tab po q6h prn pain / fever", [0.61273, 0.09875, 0.64727, 0.91625], 0.922, { q6h: 0.68, prn: 0.69 }),
        createLine("p1_l6", 5, "Disp: #20    Refills: 1", [0.67455, 0.1, 0.70636, 0.56875], 0.931),
      ],
    },
  ],
};

// 4. Legal Agreement & Signature Sample
export const SAMPLE_LEGAL_CONTRACT: DocumentOCRResult = {
  document_id: "sample_legal_contract",
  filename: "sample_legal_agreement.png",
  mime_type: "image/png",
  total_pages: 1,
  processing_time_ms: 210.5,
  model_version: "trocr-handwritten-mps-v1",
  mean_confidence: 0.952,
  overall_confidence: 0.952,
  is_mock: true,
  full_text: [
    "IN WITNESS WHEREOF, the parties hereto have executed this Agreement.",
    "Party A: Arthur J. Pendelton (Managing Partner)",
    "Executed and delivered on the 14th day of October, 2024.",
    "Authorized Signature: Arthur Pendelton [Verified Signature]",
  ].join("\n"),
  pages: [
    {
      page_number: 1,
      width: 800,
      height: 600,
      mean_confidence: 0.952,
      confidence: 0.952,
      image_url: "/samples/sample_clean_cursive.png",
      full_text: [
        "IN WITNESS WHEREOF, the parties hereto have executed this Agreement.",
        "Party A: Arthur J. Pendelton (Managing Partner)",
        "Executed and delivered on the 14th day of October, 2024.",
        "Authorized Signature: Arthur Pendelton [Verified Signature]",
      ].join("\n"),
      lines: [
        createLine("p1_l1", 0, "IN WITNESS WHEREOF, the parties hereto have executed this Agreement.", [0.2, 0.1, 0.26, 0.95], 0.965),
        createLine("p1_l2", 1, "Party A: Arthur J. Pendelton (Managing Partner)", [0.38, 0.1, 0.44, 0.92], 0.958),
        createLine("p1_l3", 2, "Executed and delivered on the 14th day of October, 2024.", [0.56, 0.1, 0.62, 0.93], 0.949),
        createLine("p1_l4", 3, "Authorized Signature: Arthur Pendelton [Verified Signature]", [0.74, 0.1, 0.81, 0.95], 0.936),
      ],
    },
  ],
};

// 5. Multi-Page Comprehensive Document Sample
export const SAMPLE_MULTIPAGE: DocumentOCRResult = {
  document_id: "sample_multipage",
  filename: "sample_multipage.pdf",
  mime_type: "application/pdf",
  total_pages: 3,
  processing_time_ms: 612.8,
  model_version: "trocr-handwritten-mps-v1",
  mean_confidence: 0.941,
  overall_confidence: 0.941,
  is_mock: true,
  full_text: [
    "Chief Complaint: Sudden onset acute migraine with aura",
    "Past Medical History: Controlled hypertension on Lisinopril",
    "No known drug allergies reported by patient today",
    "Vital Signs: BP 128/82, HR 72, SpO2 99% on room air",
    "Plan: Administer non-steroidal anti-inflammatory therapy",
    "",
    "Amoxicillin 500mg capsules",
    "Sig: Take 1 cap po tid x 10 days",
    "Disp: #30    Refills: 0",
    "Ibuprofen 600mg tablets",
    "Sig: Take 1 tab po q6h prn pain / fever",
    "Disp: #20    Refills: 1",
    "",
    "Rest in bed for next 48 hours and avoid heavy lifting",
    "Stay well hydrated with minimum 2 liters of water daily",
    "Take prescribed medications with food to prevent nausea",
    "Contact emergency department if fever exceeds 101.5 F",
    "Return for wound check and suture removal in 10 days",
  ].join("\n"),
  pages: [
    {
      page_number: 1,
      title: "Clinical Intake Questionnaire",
      width: 800,
      height: 1100,
      mean_confidence: 0.948,
      confidence: 0.948,
      image_url: "/samples/sample_prescription.png",
      full_text: [
        "Chief Complaint: Sudden onset acute migraine with aura",
        "Past Medical History: Controlled hypertension on Lisinopril",
        "No known drug allergies reported by patient today",
        "Vital Signs: BP 128/82, HR 72, SpO2 99% on room air",
        "Plan: Administer non-steroidal anti-inflammatory therapy",
      ].join("\n"),
      lines: [
        createLine("p1_l1", 0, "Chief Complaint: Sudden onset acute migraine with aura", [0.16364, 0.125, 0.21818, 0.975], 0.965),
        createLine("p1_l2", 1, "Past Medical History: Controlled hypertension on Lisinopril", [0.28182, 0.125, 0.33636, 0.975], 0.952),
        createLine("p1_l3", 2, "No known drug allergies reported by patient today", [0.4, 0.125, 0.45455, 0.975], 0.961),
        createLine("p1_l4", 3, "Vital Signs: BP 128/82, HR 72, SpO2 99% on room air", [0.51818, 0.125, 0.57273, 0.975], 0.934),
        createLine("p1_l5", 4, "Plan: Administer non-steroidal anti-inflammatory therapy", [0.63636, 0.125, 0.69091, 0.975], 0.928),
      ],
    },
    {
      page_number: 2,
      title: "Medical Prescription Slip",
      width: 800,
      height: 1100,
      mean_confidence: 0.938,
      confidence: 0.938,
      image_url: "/samples/sample_prescription.png",
      full_text: [
        "Amoxicillin 500mg capsules",
        "Sig: Take 1 cap po tid x 10 days",
        "Disp: #30    Refills: 0",
        "Ibuprofen 600mg tablets",
        "Sig: Take 1 tab po q6h prn pain / fever",
        "Disp: #20    Refills: 1",
      ].join("\n"),
      lines: [
        createLine("p2_l1", 0, "Amoxicillin 500mg capsules", [0.37909, 0.095, 0.40818, 0.68375], 0.952),
        createLine("p2_l2", 1, "Sig: Take 1 cap po tid x 10 days", [0.43909, 0.0975, 0.46909, 0.78375], 0.915, { po: 0.67, tid: 0.69 }),
        createLine("p2_l3", 2, "Disp: #30    Refills: 0", [0.49455, 0.10125, 0.52818, 0.565], 0.962),
        createLine("p2_l4", 3, "Ibuprofen 600mg tablets", [0.55545, 0.10125, 0.58636, 0.59375], 0.948),
        createLine("p2_l5", 4, "Sig: Take 1 tab po q6h prn pain / fever", [0.61273, 0.09875, 0.64727, 0.91625], 0.922, { q6h: 0.68, prn: 0.69 }),
        createLine("p2_l6", 5, "Disp: #20    Refills: 1", [0.67455, 0.1, 0.70636, 0.56875], 0.931),
      ],
    },
    {
      page_number: 3,
      title: "Discharge Instructions",
      width: 800,
      height: 1100,
      mean_confidence: 0.937,
      confidence: 0.937,
      image_url: "/samples/sample_prescription.png",
      full_text: [
        "Rest in bed for next 48 hours and avoid heavy lifting",
        "Stay well hydrated with minimum 2 liters of water daily",
        "Take prescribed medications with food to prevent nausea",
        "Contact emergency department if fever exceeds 101.5 F",
        "Return for wound check and suture removal in 10 days",
      ].join("\n"),
      lines: [
        createLine("p3_l1", 0, "Rest in bed for next 48 hours and avoid heavy lifting", [0.16364, 0.075, 0.21818, 0.925], 0.955),
        createLine("p3_l2", 1, "Stay well hydrated with minimum 2 liters of water daily", [0.28182, 0.075, 0.33636, 0.925], 0.945),
        createLine("p3_l3", 2, "Take prescribed medications with food to prevent nausea", [0.4, 0.075, 0.45455, 0.925], 0.938),
        createLine("p3_l4", 3, "Contact emergency department if fever exceeds 101.5 F", [0.51818, 0.075, 0.57273, 0.925], 0.922, { "101.5": 0.73 }),
        createLine("p3_l5", 4, "Return for wound check and suture removal in 10 days", [0.63636, 0.075, 0.69091, 0.925], 0.925),
      ],
    },
  ],
};

export const SAMPLE_PRESETS: Record<string, DocumentOCRResult> = {
  sample_clean_cursive: SAMPLE_CLEAN_CURSIVE,
  sample_messy_cursive: SAMPLE_MESSY_CURSIVE,
  sample_prescription: SAMPLE_PRESCRIPTION,
  sample_multipage: SAMPLE_MULTIPAGE,
};

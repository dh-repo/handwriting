import { DocumentOCRResult } from "../types/ocr";

/**
 * Serializes the complete Document OCR result (including lines, words, confidences, and edits) to formatted JSON.
 */
export function exportDocumentAsJson(doc: DocumentOCRResult): string {
  const exportPayload = {
    export_timestamp: new Date().toISOString(),
    document_id: doc.document_id,
    filename: doc.filename,
    total_pages: doc.total_pages,
    mean_confidence: doc.mean_confidence ?? doc.overall_confidence ?? 0,
    processing_time_ms: doc.processing_time_ms,
    model_version: doc.model_version || "trocr-mps-v1",
    signature_reviews: doc.signature_reviews ?? [],
    pages: doc.pages.map((p) => ({
      page_number: p.page_number,
      width: p.width,
      height: p.height,
      mean_confidence: p.mean_confidence,
      full_text: p.full_text,
      lines: p.lines.map((l) => ({
        line_id: l.line_id,
        line_number: l.line_number ?? (l.line_index !== undefined ? l.line_index + 1 : 1),
        text: l.text,
        original_text: l.original_text || l.text,
        confidence: l.confidence,
        bbox: l.bbox,
        polygon: l.polygon,
        is_edited: !!l.is_edited,
        words: l.words.map((w) => ({
          word_id: w.word_id,
          text: w.text,
          original_text: w.original_text || w.text,
          confidence: w.confidence,
          bbox: w.bbox,
          polygon: w.polygon,
          is_edited: !!w.is_edited,
        })),
      })),
    })),
  };

  return JSON.stringify(exportPayload, null, 2);
}

/**
 * Exports document text as plain text with standard page break dividers.
 */
export function exportDocumentAsTxt(doc: DocumentOCRResult): string {
  if (!doc.pages || doc.pages.length === 0) {
    return doc.full_text || "";
  }

  return doc.pages
    .map((page) => {
      const pageText = page.lines.map((l) => l.text).join("\n");
      return pageText;
    })
    .join("\n\n--- PAGE BREAK ---\n\n");
}

/**
 * Exports token and line level data as RFC 4180 compliant CSV.
 */
export function exportDocumentAsCsv(doc: DocumentOCRResult): string {
  const escapeCsv = (str: string | number | boolean | undefined | null): string => {
    if (str === undefined || str === null) return '""';
    let val = String(str);
    // Neutralize CSV formula injection (CWE-1236)
    if (/^[=+\-@\t\r]/.test(val)) {
      val = `'${val}`;
    }
    val = val.replace(/"/g, '""');
    return `"${val}"`;
  };

  const headers = [
    "page",
    "line_number",
    "line_id",
    "word_id",
    "confidence",
    "text",
    "original_text",
    "is_edited",
    "ymin",
    "xmin",
    "ymax",
    "xmax",
  ];

  const rows: string[] = [headers.join(",")];

  doc.pages.forEach((page) => {
    page.lines.forEach((line, lineIdx) => {
      const lineNum = line.line_number ?? (line.line_index !== undefined ? line.line_index + 1 : lineIdx + 1);

      if (line.words && line.words.length > 0) {
        line.words.forEach((word) => {
          rows.push(
            [
              page.page_number,
              lineNum,
              escapeCsv(line.line_id),
              escapeCsv(word.word_id),
              word.confidence.toFixed(4),
              escapeCsv(word.text),
              escapeCsv(word.original_text || word.text),
              word.is_edited ? "TRUE" : "FALSE",
              word.bbox[0].toFixed(5),
              word.bbox[1].toFixed(5),
              word.bbox[2].toFixed(5),
              word.bbox[3].toFixed(5),
            ].join(",")
          );
        });
      } else {
        rows.push(
          [
            page.page_number,
            lineNum,
            escapeCsv(line.line_id),
            escapeCsv(""),
            line.confidence.toFixed(4),
            escapeCsv(line.text),
            escapeCsv(line.original_text || line.text),
            line.is_edited ? "TRUE" : "FALSE",
            line.bbox[0].toFixed(5),
            line.bbox[1].toFixed(5),
            line.bbox[2].toFixed(5),
            line.bbox[3].toFixed(5),
          ].join(",")
        );
      }
    });
  });

  return rows.join("\n");
}

/**
 * Copies text to user's clipboard using navigator.clipboard or execCommand fallback.
 */
export async function copyTextToClipboard(text: string): Promise<boolean> {
  if (typeof window === "undefined") return false;

  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fallback below
  }

  try {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-999999px";
    textArea.style.top = "-999999px";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    const successful = document.execCommand("copy");
    textArea.remove();
    return successful;
  } catch (err) {
    console.error("Clipboard copy failed:", err);
    return false;
  }
}

/**
 * Triggers a browser file download from string content.
 */
export function downloadFile(content: string, filename: string, mimeType: string): void {
  if (typeof window === "undefined") return;

  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

/**
 * Exports document text formatted cleanly as structured Markdown.
 */
export function exportDocumentAsMarkdown(doc: DocumentOCRResult): string {
  const parts: string[] = [];
  parts.push(`# ${doc.filename || 'Handwriting Transcription'}`);
  parts.push(`*Mean Confidence: ${((doc.mean_confidence ?? doc.overall_confidence ?? 0.95) * 100).toFixed(1)}%*`);
  parts.push(`*Total Pages: ${doc.total_pages || doc.pages?.length || 1}*\n`);

  doc.pages.forEach((page) => {
    if (doc.pages.length > 1) {
      parts.push(`## Page ${page.page_number}`);
    }
    const pageLines = page.lines.map((l) => l.text).filter(Boolean);
    parts.push(pageLines.join('\n'));
    parts.push('');
  });

  return parts.join('\n').trim();
}

/**
 * Combines multiple batch documents into a single consolidated Markdown file.
 */
export function exportBatchDocumentsAsMarkdown(docs: DocumentOCRResult[]): string {
  const parts: string[] = [
    '# Consolidated Batch Handwriting Transcription',
    `*Generated on ${new Date().toLocaleDateString()} • Total Documents: ${docs.length}*\n`,
    '---\n',
  ];

  docs.forEach((doc, idx) => {
    parts.push(`## Document ${idx + 1}: ${doc.filename || `Document_${idx + 1}`}`);
    parts.push(exportDocumentAsMarkdown(doc));
    parts.push('\n---\n');
  });

  return parts.join('\n').trim();
}

/**
 * Combines multiple batch documents into a single plain text file.
 */
export function exportBatchDocumentsAsTxt(docs: DocumentOCRResult[]): string {
  return docs
    .map((doc, idx) => `=== DOCUMENT ${idx + 1}: ${doc.filename || `doc_${idx + 1}`} ===\n\n${exportDocumentAsTxt(doc)}`)
    .join('\n\n========================================\n\n');
}

/**
 * Combines multiple batch documents into a structured batch JSON result.
 */
export function exportBatchDocumentsAsJson(docs: DocumentOCRResult[]): string {
  const payload = {
    batch_export_timestamp: new Date().toISOString(),
    total_documents: docs.length,
    documents: docs.map((doc) => JSON.parse(exportDocumentAsJson(doc))),
  };
  return JSON.stringify(payload, null, 2);
}


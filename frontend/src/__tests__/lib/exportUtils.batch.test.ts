import { describe, it, expect, vi } from 'vitest';
import {
  exportDocumentAsMarkdown,
  exportBatchDocumentsAsMarkdown,
  exportBatchDocumentsAsTxt,
  exportBatchDocumentsAsJson,
  exportDocumentAsSearchableHtml,
  openSearchablePdfPrint,
} from '@/lib/exportUtils';
import { SAMPLE_CLEAN_CURSIVE, SAMPLE_LEGAL_CONTRACT } from '@/lib/sampleDocuments';

describe('exportUtils Batch & Markdown Extensions', () => {
  it('exports a document cleanly to structured Markdown', () => {
    const md = exportDocumentAsMarkdown(SAMPLE_CLEAN_CURSIVE);
    expect(md).toContain('# sample_clean_cursive.png');
    expect(md).toContain('The quick brown fox jumps over the lazy dog.');
    expect(md).toContain('Mean Confidence:');
  });

  it('combines multiple documents into a single consolidated Markdown export', () => {
    const batchMd = exportBatchDocumentsAsMarkdown([
      SAMPLE_CLEAN_CURSIVE,
      SAMPLE_LEGAL_CONTRACT,
    ]);
    expect(batchMd).toContain('# Consolidated Batch Handwriting Transcription');
    expect(batchMd).toContain('Total Documents: 2');
    expect(batchMd).toContain('sample_clean_cursive.png');
    expect(batchMd).toContain('sample_legal_agreement.png');
    expect(batchMd).toContain('IN WITNESS WHEREOF');
  });

  it('combines multiple documents into a single plain text export', () => {
    const batchTxt = exportBatchDocumentsAsTxt([
      SAMPLE_CLEAN_CURSIVE,
      SAMPLE_LEGAL_CONTRACT,
    ]);
    expect(batchTxt).toContain('=== DOCUMENT 1: sample_clean_cursive.png ===');
    expect(batchTxt).toContain('=== DOCUMENT 2: sample_legal_agreement.png ===');
  });

  it('combines multiple documents into a structured batch JSON export', () => {
    const batchJsonStr = exportBatchDocumentsAsJson([
      SAMPLE_CLEAN_CURSIVE,
      SAMPLE_LEGAL_CONTRACT,
    ]);
    const parsed = JSON.parse(batchJsonStr);
    expect(parsed.total_documents).toBe(2);
    expect(parsed.documents.length).toBe(2);
    expect(parsed.documents[0].filename).toBe('sample_clean_cursive.png');
    expect(parsed.documents[1].filename).toBe('sample_legal_agreement.png');
  });

  it('generates a 1:1 aligned Searchable PDF HTML layer with transparent OCR tokens', () => {
    const html = exportDocumentAsSearchableHtml(SAMPLE_CLEAN_CURSIVE);
    expect(html).toContain('<!DOCTYPE html>');
    expect(html).toContain('ocr-layer');
    expect(html).toContain('ocr-token');
    expect(html).toContain('The');
    expect(html).toContain('quick');
    expect(html).toContain('sample_clean_cursive.png');
  });

  it('triggers window.open print window when openSearchablePdfPrint is called', () => {
    const mockPrint = vi.fn();
    const mockWrite = vi.fn();
    const mockClose = vi.fn();
    const mockFocus = vi.fn();

    const mockWindow = {
      document: { write: mockWrite, close: mockClose },
      focus: mockFocus,
      print: mockPrint,
    };

    const spyOpen = vi.spyOn(window, 'open').mockReturnValue(mockWindow as any);
    openSearchablePdfPrint(SAMPLE_CLEAN_CURSIVE);

    expect(spyOpen).toHaveBeenCalledWith('', '_blank');
    expect(mockWrite).toHaveBeenCalled();
    expect(mockClose).toHaveBeenCalled();
    expect(mockFocus).toHaveBeenCalled();
    spyOpen.mockRestore();
  });
});

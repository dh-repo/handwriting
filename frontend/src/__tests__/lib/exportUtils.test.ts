import { describe, it, expect, vi } from 'vitest';
import {
  exportDocumentAsJson,
  exportDocumentAsTxt,
  exportDocumentAsCsv,
  copyTextToClipboard,
  downloadFile,
} from '@/lib/exportUtils';
import { SAMPLE_CLEAN_CURSIVE, SAMPLE_MULTIPAGE } from '@/lib/sampleDocuments';

describe('exportUtils', () => {
  it('exports structured JSON matching ground truth schema', () => {
    const jsonStr = exportDocumentAsJson(SAMPLE_CLEAN_CURSIVE);
    const parsed = JSON.parse(jsonStr);

    expect(parsed.document_id).toBe('sample_clean_cursive');
    expect(parsed.total_pages).toBe(1);
    expect(parsed.pages[0].lines).toHaveLength(4);
    expect(parsed.pages[0].lines[0].text).toContain('quick brown fox');
    expect(parsed.pages[0].lines[0].words).toHaveLength(9);
    expect(parsed.pages[0].lines[0].bbox).toHaveLength(4);
  });

  it('exports plain text with page break delimiters for multi-page documents', () => {
    const txt = exportDocumentAsTxt(SAMPLE_MULTIPAGE);
    expect(txt).toContain('Chief Complaint: Sudden onset acute migraine with aura');
    expect(txt).toContain('--- PAGE BREAK ---');
    expect(txt).toContain('Amoxicillin 500mg capsules');
    expect(txt).toContain('Rest in bed for next 48 hours');
  });

  it('exports RFC 4180 compliant CSV table with tokens and bboxes', () => {
    const csv = exportDocumentAsCsv(SAMPLE_CLEAN_CURSIVE);
    const lines = csv.split('\n');

    expect(lines[0]).toBe(
      'page,line_number,line_id,word_id,confidence,text,original_text,is_edited,ymin,xmin,ymax,xmax'
    );
    expect(lines.length).toBeGreaterThan(5);
    expect(lines[1]).toContain('"The"');
  });

  it('copies text to clipboard via navigator.clipboard', async () => {
    const success = await copyTextToClipboard('Test clipboard content');
    expect(success).toBe(true);
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('Test clipboard content');
  });

  it('triggers browser file download using DOM anchor element', () => {
    const clickSpy = vi.fn();
    const originalCreateElement = document.createElement.bind(document);

    vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      const el = originalCreateElement(tagName);
      if (tagName === 'a') {
        el.click = clickSpy;
      }
      return el;
    });

    downloadFile('Sample text', 'test.txt', 'text/plain');
    expect(clickSpy).toHaveBeenCalled();
  });
});

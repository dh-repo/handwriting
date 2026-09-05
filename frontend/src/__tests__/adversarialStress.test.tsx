import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import {
  bboxToSvgRect,
  bboxToPolygon,
  polygonToSvgPoints,
  polygonToBbox,
  clampBbox,
} from '@/lib/transformUtils';
import {
  exportDocumentAsJson,
  exportDocumentAsTxt,
  exportDocumentAsCsv,
  copyTextToClipboard,
} from '@/lib/exportUtils';
import { runMockOcr } from '@/lib/mockOcrEngine';
import { SAMPLE_PRESETS, SAMPLE_CLEAN_CURSIVE, SAMPLE_MULTIPAGE } from '@/lib/sampleDocuments';
import { DocumentViewer } from '@/components/DocumentViewer';
import { InlineEditor } from '@/components/InlineEditor';
import { DocumentProvider, useDocumentContext } from '@/context/DocumentContext';
import { DocumentOCRResult, PageResult, LineItem, WordToken, BoundingBoxTuple } from '@/types/ocr';

describe('Adversarial Stress Test Suite: Milestone 4 Frontend', () => {
  // =========================================================================
  // 1. COORDINATE MATH & GEOMETRY TRANSFORMS
  // =========================================================================
  describe('1. Coordinate Math & Geometry Transforms', () => {
    it('handles extreme zoom scaling boundaries (0.2x min, 5.0x max)', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const zoomInBtn = screen.getByTestId('btn-zoom-in');
      const zoomOutBtn = screen.getByTestId('btn-zoom-out');
      const zoomText = screen.getByTestId('zoom-level-text');

      // Zoom in 30 times -> must clamp at 500% (5.0x)
      for (let i = 0; i < 30; i++) {
        fireEvent.click(zoomInBtn);
      }
      expect(zoomText).toHaveTextContent('500%');

      // Zoom out 40 times -> must clamp at 20% (0.2x)
      for (let i = 0; i < 40; i++) {
        fireEvent.click(zoomOutBtn);
      }
      expect(zoomText).toHaveTextContent('20%');
    });

    it('handles 4x 90-degree rotations in complete 360-degree cycle', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const rotateBtn = screen.getByTestId('btn-rotate');
      const canvas = screen.getByTestId('viewer-canvas-wrapper');

      expect(canvas.style.transform).toContain('rotate(0deg)');

      fireEvent.click(rotateBtn);
      expect(canvas.style.transform).toContain('rotate(90deg)');

      fireEvent.click(rotateBtn);
      expect(canvas.style.transform).toContain('rotate(180deg)');

      fireEvent.click(rotateBtn);
      expect(canvas.style.transform).toContain('rotate(270deg)');

      fireEvent.click(rotateBtn);
      expect(canvas.style.transform).toContain('rotate(0deg)');
    });

    it('safely handles zero, empty, negative, and out-of-bounds bounding boxes', () => {
      // 1. Zero bbox [0, 0, 0, 0] -> dimensions should be at least 1px to avoid zero-division / collapsing
      const zeroRect = bboxToSvgRect([0, 0, 0, 0], 1000, 2000);
      expect(zeroRect.x).toBe(0);
      expect(zeroRect.y).toBe(0);
      expect(zeroRect.width).toBeGreaterThanOrEqual(1);
      expect(zeroRect.height).toBeGreaterThanOrEqual(1);

      // 2. Negative out-of-bounds bbox [-0.5, -0.8, 1.8, 2.5]
      const outOfBoundsRect = bboxToSvgRect([-0.5, -0.8, 1.8, 2.5], 1000, 2000);
      expect(outOfBoundsRect.x).toBe(0);
      expect(outOfBoundsRect.y).toBe(0);
      expect(outOfBoundsRect.width).toBe(1000);
      expect(outOfBoundsRect.height).toBe(2000);

      // 3. Inverted coordinates [0.8, 0.9, 0.2, 0.1]
      const clampedInverted = clampBbox([0.8, 0.9, 0.2, 0.1]);
      expect(clampedInverted[0]).toBeLessThanOrEqual(clampedInverted[2]);
      expect(clampedInverted[1]).toBeLessThanOrEqual(clampedInverted[3]);

      // 4. Empty polygon to bbox
      const emptyBbox = polygonToBbox([]);
      expect(emptyBbox).toEqual([0, 0, 0, 0]);

      // 5. Huge document dimensions (e.g. 10000x20000)
      const hugeRect = bboxToSvgRect([0.1, 0.2, 0.3, 0.4], 10000, 20000);
      expect(hugeRect.x).toBeCloseTo(2000);
      expect(hugeRect.y).toBeCloseTo(2000);
      expect(hugeRect.width).toBeCloseTo(2000);
      expect(hugeRect.height).toBeCloseTo(4000);

      // 6. Polygon to SVG string
      const poly = bboxToPolygon([0.1, 0.2, 0.3, 0.4]);
      const svgPoints = polygonToSvgPoints(poly, 1000, 2000);
      expect(svgPoints).toBe('200.0,200.0 400.0,200.0 400.0,600.0 200.0,600.0');
    });
  });

  // =========================================================================
  // 2. EXPORT UTILITIES (CSV, JSON, TXT, CLIPBOARD)
  // =========================================================================
  describe('2. Export Utilities Adversarial Inputs', () => {
    it('handles empty documents without throwing runtime exceptions', () => {
      const emptyDoc: DocumentOCRResult = {
        document_id: 'doc_empty',
        filename: 'empty.png',
        total_pages: 0,
        pages: [],
        mean_confidence: 0,
        processing_time_ms: 10,
        full_text: '',
      };

      const jsonStr = exportDocumentAsJson(emptyDoc);
      expect(() => JSON.parse(jsonStr)).not.toThrow();
      const parsed = JSON.parse(jsonStr);
      expect(parsed.document_id).toBe('doc_empty');
      expect(parsed.pages).toEqual([]);

      const txtStr = exportDocumentAsTxt(emptyDoc);
      expect(txtStr).toBe('');

      const csvStr = exportDocumentAsCsv(emptyDoc);
      const csvLines = csvStr.trim().split('\n');
      expect(csvLines.length).toBe(1); // Header only
    });

    it('correctly escapes special characters (commas, quotes, newlines, emojis, HTML)', () => {
      const adversarialDoc: DocumentOCRResult = {
        document_id: 'doc_adversarial',
        filename: 'rx_adversarial.png',
        total_pages: 1,
        mean_confidence: 0.88,
        processing_time_ms: 120,
        full_text: 'Adversarial Text',
        pages: [
          {
            page_number: 1,
            width: 1200,
            height: 1600,
            mean_confidence: 0.88,
            full_text: 'Adversarial Line',
            lines: [
              {
                line_id: 'l_adv_1',
                line_number: 1,
                text: 'Rx: "Amoxicillin, 500mg" <p>Take 1 & 2</p> 💊 \n Newline',
                original_text: 'Rx: "Amoxicillin, 500mg" <p>Take 1 & 2</p> 💊 \n Newline',
                confidence: 0.88,
                bbox: [0.1, 0.1, 0.2, 0.9],
                polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.9]),
                words: [
                  {
                    word_id: 'w_adv_1',
                    text: '"Amoxicillin, 500mg"',
                    original_text: '"Amoxicillin, 500mg"',
                    confidence: 0.95,
                    bbox: [0.1, 0.1, 0.2, 0.5],
                    polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.5]),
                  },
                  {
                    word_id: 'w_adv_2',
                    text: '💊 emoji & <tags>',
                    original_text: '💊 emoji & <tags>',
                    confidence: 0.81,
                    bbox: [0.1, 0.5, 0.2, 0.9],
                    polygon: bboxToPolygon([0.1, 0.5, 0.2, 0.9]),
                  },
                ],
              },
            ],
          },
        ],
      };

      // 1. JSON Export
      const jsonStr = exportDocumentAsJson(adversarialDoc);
      const parsedJson = JSON.parse(jsonStr);
      expect(parsedJson.pages[0].lines[0].text).toContain('"Amoxicillin, 500mg"');
      expect(parsedJson.pages[0].lines[0].words[1].text).toContain('💊 emoji & <tags>');

      // 2. CSV Export (RFC 4180 Escaping Check)
      const csvStr = exportDocumentAsCsv(adversarialDoc);
      expect(csvStr).toContain('"""Amoxicillin, 500mg"""');
      expect(csvStr).toContain('"💊 emoji & <tags>"');

      // 3. TXT Export
      const txtStr = exportDocumentAsTxt(adversarialDoc);
      expect(txtStr).toContain('Rx: "Amoxicillin, 500mg"');
    });

    it('handles lines without words gracefully in CSV export', () => {
      const lineOnlyDoc: DocumentOCRResult = {
        document_id: 'doc_line_only',
        filename: 'lines.png',
        total_pages: 1,
        mean_confidence: 0.92,
        processing_time_ms: 80,
        full_text: 'Line with no words array',
        pages: [
          {
            page_number: 1,
            width: 1000,
            height: 1000,
            mean_confidence: 0.92,
            full_text: 'Line with no words array',
            lines: [
              {
                line_id: 'l_nowords_1',
                line_number: 1,
                text: 'Line with no words array',
                original_text: 'Line with no words array',
                confidence: 0.92,
                bbox: [0.1, 0.1, 0.2, 0.8],
                polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.8]),
                words: [], // empty words
              },
            ],
          },
        ],
      };

      const csvStr = exportDocumentAsCsv(lineOnlyDoc);
      const rows = csvStr.split('\n');
      expect(rows.length).toBe(2); // header + 1 line row
      expect(rows[1]).toContain('"l_nowords_1"');
    });

    it('falls back to execCommand when navigator.clipboard is unavailable', async () => {
      // Mock navigator.clipboard throwing an error
      const originalClipboard = navigator.clipboard;
      Object.defineProperty(navigator, 'clipboard', {
        value: {
          writeText: vi.fn().mockRejectedValue(new Error('Permission denied')),
        },
        configurable: true,
      });

      // Mock document.execCommand
      document.execCommand = vi.fn().mockReturnValue(true);

      const result = await copyTextToClipboard('Test clipboard text');
      expect(result).toBe(true);
      expect(document.execCommand).toHaveBeenCalledWith('copy');

      // Restore
      Object.defineProperty(navigator, 'clipboard', {
        value: originalClipboard,
        configurable: true,
      });
    });
  });

  // =========================================================================
  // 3. MOCK OCR ENGINE ARBITRARY INPUTS & PAGINATION
  // =========================================================================
  describe('3. Mock OCR Engine Stress & Synthetic Layouts', () => {
    it('generates synthetic document layout for arbitrary uploaded image', async () => {
      const result = await runMockOcr({
        filename: 'random_custom_upload_99.png',
        mimeType: 'image/png',
      });

      expect(result.document_id).toMatch(/^doc_/);
      expect(result.is_mock).toBe(true);
      expect(result.pages.length).toBe(1);
      expect(result.pages[0].lines.length).toBeGreaterThan(0);

      // Verify all words have strictly valid non-overlapping coordinates in [0, 1]
      for (const line of result.pages[0].lines) {
        expect(line.bbox[0]).toBeGreaterThanOrEqual(0);
        expect(line.bbox[2]).toBeLessThanOrEqual(1);
        expect(line.bbox[0]).toBeLessThanOrEqual(line.bbox[2]);

        let lastXMax = 0;
        for (const word of line.words) {
          expect(word.bbox[1]).toBeGreaterThanOrEqual(lastXMax - 0.001); // no major overlap
          expect(word.bbox[3]).toBeLessThanOrEqual(1);
          expect(word.confidence).toBeGreaterThan(0);
          expect(word.confidence).toBeLessThanOrEqual(1);
          lastXMax = word.bbox[3];
        }
      }
    });

    it('returns multi-page document for PDF uploads and supports page navigation', async () => {
      const result = await runMockOcr({
        filename: 'historical_archive.pdf',
        mimeType: 'application/pdf',
      });

      expect(result.total_pages).toBe(3);
      expect(result.pages.length).toBe(3);
      expect(result.pages[0].page_number).toBe(1);
      expect(result.pages[1].page_number).toBe(2);
      expect(result.pages[2].page_number).toBe(3);
    });

    it('guarantees preset immutability (structuredClone protection)', async () => {
      const preset1 = await runMockOcr({ sampleId: 'sample_clean_cursive' });
      // Mutate returned object
      preset1.pages[0].lines[0].text = 'MUTATED TEXT VALUE';

      // Re-fetch preset
      const preset2 = await runMockOcr({ sampleId: 'sample_clean_cursive' });
      expect(preset2.pages[0].lines[0].text).not.toBe('MUTATED TEXT VALUE');
      expect(preset2.pages[0].lines[0].text).toBe(SAMPLE_PRESETS['sample_clean_cursive'].pages[0].lines[0].text);
    });
  });

  // =========================================================================
  // 4. IN-APP SPEED REVIEW: 0% VS 100% LOW CONFIDENCE WORDS
  // =========================================================================
  describe('4. In-App Speed Review: 0% vs 100% Low Confidence Documents', () => {
    it('handles document with 0 low-confidence words (all >= 90% confidence)', () => {
      const highConfPage: PageResult = {
        page_number: 1,
        width: 1000,
        height: 1000,
        mean_confidence: 0.98,
        full_text: 'Perfect high confidence transcription',
        lines: [
          {
            line_id: 'l_hi_1',
            line_number: 1,
            text: 'Perfect high confidence transcription',
            original_text: 'Perfect high confidence transcription',
            confidence: 0.98,
            bbox: [0.1, 0.1, 0.2, 0.9],
            polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.9]),
            words: [
              {
                word_id: 'w_hi_1',
                text: 'Perfect',
                original_text: 'Perfect',
                confidence: 0.99,
                bbox: [0.1, 0.1, 0.2, 0.3],
                polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.3]),
              },
              {
                word_id: 'w_hi_2',
                text: 'transcription',
                original_text: 'transcription',
                confidence: 0.97,
                bbox: [0.1, 0.3, 0.2, 0.9],
                polygon: bboxToPolygon([0.1, 0.3, 0.2, 0.9]),
              },
            ],
          },
        ],
      };

      render(<InlineEditor enableMedicalSuggestions page={highConfPage} />);

      const speedTab = screen.getByTestId('tab-speed-review');
      expect(speedTab).toHaveTextContent('Speed Review (0)');

      fireEvent.click(speedTab);

      // Should show the "All Words Verified!" empty state
      expect(screen.getByText('All Words Verified!')).toBeInTheDocument();
      expect(screen.getByTestId('btn-return-lines')).toBeInTheDocument();

      // Clicking return to lines switches back to structured mode
      fireEvent.click(screen.getByTestId('btn-return-lines'));
      expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();
    });

    it('handles document with 100% low-confidence words and steps through entire queue', () => {
      const onWordChange = vi.fn();
      const lowConfPage: PageResult = {
        page_number: 1,
        width: 1000,
        height: 1000,
        mean_confidence: 0.45,
        full_text: 'messy doctor scrawl',
        lines: [
          {
            line_id: 'l_low_1',
            line_number: 1,
            text: 'messy doctor scrawl',
            original_text: 'messy doctor scrawl',
            confidence: 0.45,
            bbox: [0.1, 0.1, 0.2, 0.9],
            polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.9]),
            words: [
              {
                word_id: 'w_low_1',
                text: 'messy',
                original_text: 'messy',
                confidence: 0.42,
                bbox: [0.1, 0.1, 0.2, 0.3],
                polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.3]),
              },
              {
                word_id: 'w_low_2',
                text: 'doctor',
                original_text: 'doctor',
                confidence: 0.48,
                bbox: [0.1, 0.3, 0.2, 0.6],
                polygon: bboxToPolygon([0.1, 0.3, 0.2, 0.6]),
              },
              {
                word_id: 'w_low_3',
                text: 'scrawl',
                original_text: 'scrawl',
                confidence: 0.44,
                bbox: [0.1, 0.6, 0.2, 0.9],
                polygon: bboxToPolygon([0.1, 0.6, 0.2, 0.9]),
              },
            ],
          },
        ],
      };

      render(<InlineEditor enableMedicalSuggestions page={lowConfPage} onWordChange={onWordChange} />);

      const speedTab = screen.getByTestId('tab-speed-review');
      expect(speedTab).toHaveTextContent('Speed Review (3)');

      fireEvent.click(speedTab);
      const panel = screen.getByTestId('speed-review-panel');
      expect(panel).toBeInTheDocument();
      expect(panel).toHaveTextContent(/Reviewing Uncertain Word 1 of 3/);
      expect(panel).toHaveTextContent('42.0% Conf');

      // Correct Word 1 and Accept
      const input = screen.getByTestId('speed-review-input');
      fireEvent.change(input, { target: { value: 'Clean' } });
      const acceptBtn = screen.getByTestId('btn-speed-accept');
      fireEvent.click(acceptBtn);
      expect(onWordChange).toHaveBeenCalledWith('l_low_1', 'w_low_1', 'Clean');

      // Word 2: Skip
      const skipBtn = screen.getByTestId('btn-speed-skip');
      fireEvent.click(skipBtn);

      // Word 3: Accept via Enter Keydown
      fireEvent.change(input, { target: { value: 'Handwriting' } });
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
      expect(onWordChange).toHaveBeenCalledWith('l_low_1', 'w_low_3', 'Handwriting');

      // Queue finished -> Automatically returns to structured mode
      expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // 5. DOCUMENT CONTEXT STATE & UNDO/REDO CHURN
  // =========================================================================
  describe('5. Document Context State & Undo/Redo Churn', () => {
    const TestConsumer = () => {
      const ctx = useDocumentContext();
      return (
        <div>
          <span data-testid="doc-text">{ctx.activePage?.full_text}</span>
          <span data-testid="can-undo">{ctx.canUndo ? 'true' : 'false'}</span>
          <span data-testid="can-redo">{ctx.canRedo ? 'true' : 'false'}</span>
          <button data-testid="btn-test-edit-line" onClick={() => ctx.updateLineText('p1_l1', 'Edited First Line')}>
            Edit Line
          </button>
          <button data-testid="btn-test-edit-word" onClick={() => ctx.updateWordText('p1_l1', 'p1_l1_w1', 'SuperWord')}>
            Edit Word
          </button>
          <button data-testid="btn-test-undo" onClick={() => ctx.undo()}>
            Undo
          </button>
          <button data-testid="btn-test-redo" onClick={() => ctx.redo()}>
            Redo
          </button>
          <button data-testid="btn-test-revert-all" onClick={() => ctx.revertAll()}>
            Revert All
          </button>
        </div>
      );
    };

    it('maintains consistent state and history across multiple edit, undo, redo cycles', () => {
      render(
        <DocumentProvider initialDocument={SAMPLE_CLEAN_CURSIVE}>
          <TestConsumer />
        </DocumentProvider>
      );

      const docText = screen.getByTestId('doc-text');
      const canUndo = screen.getByTestId('can-undo');
      const canRedo = screen.getByTestId('can-redo');

      expect(canUndo).toHaveTextContent('false');
      expect(canRedo).toHaveTextContent('false');

      // 1. Edit Line
      fireEvent.click(screen.getByTestId('btn-test-edit-line'));
      expect(docText.textContent).toContain('Edited First Line');
      expect(canUndo).toHaveTextContent('true');

      // 2. Edit Word
      fireEvent.click(screen.getByTestId('btn-test-edit-word'));
      expect(docText.textContent).toContain('SuperWord');

      // 3. Undo twice
      fireEvent.click(screen.getByTestId('btn-test-undo'));
      expect(docText.textContent).toContain('Edited First Line');
      expect(canRedo).toHaveTextContent('true');

      fireEvent.click(screen.getByTestId('btn-test-undo'));
      expect(docText.textContent).not.toContain('Edited First Line');
      expect(canUndo).toHaveTextContent('false');

      // 4. Redo
      fireEvent.click(screen.getByTestId('btn-test-redo'));
      expect(docText.textContent).toContain('Edited First Line');
      expect(canUndo).toHaveTextContent('true');

      // 5. Revert All
      fireEvent.click(screen.getByTestId('btn-test-revert-all'));
      expect(docText.textContent).not.toContain('Edited First Line');
    });
  });

  // =========================================================================
  // 6. INTERACTIVE VIEWER DRAG & WHEEL STRESS
  // =========================================================================
  describe('6. Interactive Viewer Drag & Wheel Physics Stress', () => {
    it('handles mouse drag panning with large negative coordinate shifts', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const canvas = screen.getByTestId('viewer-canvas-wrapper');
      const viewport = canvas.parentElement!;

      // Mouse down to start drag
      fireEvent.mouseDown(viewport, { button: 0, clientX: 500, clientY: 500 });

      // Drag to negative offset
      fireEvent.mouseMove(viewport, { clientX: -200, clientY: -300 });

      expect(canvas.style.transform).toContain('translate(-700px, -800px)');

      // Mouse up to end drag
      fireEvent.mouseUp(viewport);

      // Subsequent mouse move does not alter pan
      fireEvent.mouseMove(viewport, { clientX: 1000, clientY: 1000 });
      expect(canvas.style.transform).toContain('translate(-700px, -800px)');
    });

    it('handles wheel events with ctrlKey zoom and standard pan', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const canvas = screen.getByTestId('viewer-canvas-wrapper');
      const viewport = canvas.parentElement!;
      const zoomText = screen.getByTestId('zoom-level-text');

      // Wheel scroll panning
      fireEvent.wheel(viewport, { deltaX: 50, deltaY: 80, ctrlKey: false });
      expect(canvas.style.transform).toContain('translate(-50px, -80px)');

      // Wheel pinch zoom (ctrlKey: true)
      fireEvent.wheel(viewport, { deltaY: -100, ctrlKey: true });
      expect(zoomText).toHaveTextContent('115%');
    });

    it('gracefully renders document viewer when image_url is missing and lines are empty', () => {
      const barePage: PageResult = {
        page_number: 1,
        width: 800,
        height: 1000,
        mean_confidence: 0,
        full_text: '',
        lines: [],
      };

      render(<DocumentViewer page={barePage} />);
      expect(screen.getByTestId('document-viewer-container')).toBeInTheDocument();
      expect(screen.getByText('[No Image Preview Available]')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // 7. INLINE EDITOR ESCAPE SHORTCUT & DEFENSIVE PROPERTY GUARDS
  // =========================================================================
  describe('7. Inline Editor Escape Shortcut & Defensive Property Guards', () => {
    it('switches back to structured mode when Escape key is pressed in Speed Review', () => {
      const pageWithLowConf: PageResult = {
        page_number: 1,
        width: 1000,
        height: 1000,
        mean_confidence: 0.5,
        full_text: 'low conf word',
        lines: [
          {
            line_id: 'l1',
            line_number: 1,
            text: 'low conf word',
            original_text: 'low conf word',
            confidence: 0.5,
            bbox: [0.1, 0.1, 0.2, 0.8],
            polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.8]),
            words: [
              {
                word_id: 'w1',
                text: 'low',
                confidence: 0.5,
                bbox: [0.1, 0.1, 0.2, 0.4],
                polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.4]),
              },
            ],
          },
        ],
      };

      render(<InlineEditor enableMedicalSuggestions page={pageWithLowConf} />);
      fireEvent.click(screen.getByTestId('tab-speed-review'));
      expect(screen.getByTestId('speed-review-panel')).toBeInTheDocument();

      const input = screen.getByTestId('speed-review-input');
      fireEvent.keyDown(input, { key: 'Escape', code: 'Escape' });

      expect(screen.getByTestId('structured-lines-list')).toBeInTheDocument();
    });
  });
});


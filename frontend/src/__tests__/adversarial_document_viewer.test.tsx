/**
 * frontend/src/__tests__/adversarial_document_viewer.test.tsx
 * Comprehensive Empirical Adversarial Stress Test for Document Viewer & Coordinate Math
 */

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
import { DocumentViewer } from '@/components/DocumentViewer';
import { PageResult, LineItem, WordToken, BoundingBoxTuple, PolygonPoint } from '@/types/ocr';
import { SAMPLE_CLEAN_CURSIVE, SAMPLE_MULTIPAGE } from '@/lib/sampleDocuments';

describe('Adversarial Document Viewer & Coordinate Math Stress Tests', () => {
  // =========================================================================
  // 1. TRANSFORM UTILS ADVERSARIAL GEOMETRY & BOUNDARY MATHEMATICS
  // =========================================================================
  describe('1. transformUtils Geometry Transformations & Degenerate Coordinates', () => {
    it('handles inverted bounding box coordinates [ymin > ymax, xmin > xmax]', () => {
      const invertedBbox: BoundingBoxTuple = [0.85, 0.95, 0.15, 0.25];
      const clamped = clampBbox(invertedBbox);

      expect(clamped[0]).toBe(0.15); // min y
      expect(clamped[1]).toBe(0.25); // min x
      expect(clamped[2]).toBe(0.85); // max y
      expect(clamped[3]).toBe(0.95); // max x

      const rect = bboxToSvgRect(invertedBbox, 1000, 2000);
      expect(rect.x).toBeCloseTo(250);
      expect(rect.y).toBeCloseTo(300);
      expect(rect.width).toBeCloseTo(700);
      expect(rect.height).toBeCloseTo(1400);

      const poly = bboxToPolygon(invertedBbox);
      expect(poly).toEqual([
        [0.25, 0.15],
        [0.95, 0.15],
        [0.95, 0.85],
        [0.25, 0.85],
      ]);
    });

    it('handles extreme negative and out-of-bounds bounding boxes', () => {
      // Both negative and > 1.0
      const outOfBounds: BoundingBoxTuple = [-5.0, -10.0, 15.0, 20.0];
      const clamped = clampBbox(outOfBounds);
      expect(clamped).toEqual([0, 0, 1, 1]);

      const rect = bboxToSvgRect(outOfBounds, 800, 1200);
      expect(rect.x).toBe(0);
      expect(rect.y).toBe(0);
      expect(rect.width).toBe(800);
      expect(rect.height).toBe(1200);

      // Entirely negative out of bounds
      const allNegative: BoundingBoxTuple = [-2.0, -3.0, -1.0, -0.5];
      const rectNeg = bboxToSvgRect(allNegative, 800, 1200);
      expect(rectNeg.x).toBe(0);
      expect(rectNeg.y).toBe(0);
      expect(rectNeg.width).toBe(1); // Min 1px fallback
      expect(rectNeg.height).toBe(1);
    });

    it('handles zero-dimension, point, and degenerate bounding boxes', () => {
      // Point bbox
      const point: BoundingBoxTuple = [0.5, 0.5, 0.5, 0.5];
      const rect = bboxToSvgRect(point, 1000, 1000);
      expect(rect.x).toBe(500);
      expect(rect.y).toBe(500);
      expect(rect.width).toBe(1); // Non-zero fallback
      expect(rect.height).toBe(1);

      // Zero bbox [0, 0, 0, 0]
      const zero: BoundingBoxTuple = [0, 0, 0, 0];
      const zeroRect = bboxToSvgRect(zero, 1000, 1000);
      expect(zeroRect.x).toBe(0);
      expect(zeroRect.y).toBe(0);
      expect(zeroRect.width).toBe(1);
      expect(zeroRect.height).toBe(1);
    });

    it('handles NaN, Infinity, -Infinity and invalid types without crashing', () => {
      const nanBbox = [NaN, Infinity, -Infinity, 'invalid' as any] as unknown as BoundingBoxTuple;
      const rect = bboxToSvgRect(nanBbox, 1000, 1000);

      expect(Number.isFinite(rect.x)).toBe(true);
      expect(Number.isFinite(rect.y)).toBe(true);
      expect(Number.isFinite(rect.width)).toBe(true);
      expect(Number.isFinite(rect.height)).toBe(true);
      expect(rect.width).toBeGreaterThanOrEqual(1);
      expect(rect.height).toBeGreaterThanOrEqual(1);

      const clamped = clampBbox(nanBbox);
      clamped.forEach((v) => {
        expect(Number.isFinite(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      });
    });

    it('handles document width and height being 0, negative, NaN, or Infinity', () => {
      const bbox: BoundingBoxTuple = [0.1, 0.2, 0.3, 0.4];

      const zeroDoc = bboxToSvgRect(bbox, 0, 0);
      expect(zeroDoc.x).toBe(0);
      expect(zeroDoc.y).toBe(0);
      expect(zeroDoc.width).toBe(1);
      expect(zeroDoc.height).toBe(1);

      const negDoc = bboxToSvgRect(bbox, -1000, -2000);
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

    it('handles polygon conversions with corrupt or out-of-bounds coordinates', () => {
      expect(polygonToBbox([])).toEqual([0, 0, 0, 0]);

      const corruptPoly: PolygonPoint[] = [
        [NaN, -2.0] as unknown as PolygonPoint,
        [Infinity, 5.0] as unknown as PolygonPoint,
      ];
      const polyBbox = polygonToBbox(corruptPoly);
      expect(polyBbox).toEqual([0, 0, 1, 0]);

      const svgPoints = polygonToSvgPoints(
        [[0.1, 0.2], [NaN, Infinity] as unknown as PolygonPoint],
        1000,
        1000
      );
      expect(svgPoints).toBe('100.0,200.0 0.0,0.0');
    });
  });

  // =========================================================================
  // 2. DOCUMENT VIEWER ZOOM EXTREMES (0.1x to 10.0x) & SLIDER ADVERSARIAL INPUTS
  // =========================================================================
  describe('2. DocumentViewer Zoom Extremes & Slider Boundaries', () => {
    it('strictly clamps zoom level between 0.2x (20%) and 5.0x (500%) via zoom buttons', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const zoomInBtn = screen.getByTestId('btn-zoom-in');
      const zoomOutBtn = screen.getByTestId('btn-zoom-out');
      const zoomText = screen.getByTestId('zoom-level-text');

      // Click Zoom In 50 times -> Should never exceed 500% (5.0x)
      for (let i = 0; i < 50; i++) {
        fireEvent.click(zoomInBtn);
      }
      expect(zoomText).toHaveTextContent('500%');

      // Click Zoom Out 60 times -> Should never drop below 20% (0.2x)
      for (let i = 0; i < 60; i++) {
        fireEvent.click(zoomOutBtn);
      }
      expect(zoomText).toHaveTextContent('20%');
    });

    it('handles zoom slider values across 0.2x to 5.0x boundary values', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const slider = screen.getByTestId('zoom-slider');
      const zoomText = screen.getByTestId('zoom-level-text');

      // Set slider to 5.0x
      fireEvent.change(slider, { target: { value: '5.0' } });
      expect(zoomText).toHaveTextContent('500%');

      // Set slider to 0.2x
      fireEvent.change(slider, { target: { value: '0.2' } });
      expect(zoomText).toHaveTextContent('20%');

      // Set slider to 2.5x
      fireEvent.change(slider, { target: { value: '2.5' } });
      expect(zoomText).toHaveTextContent('250%');
    });

    it('handles wheel zoom with massive delta values', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const canvas = screen.getByTestId('viewer-canvas-wrapper');
      const viewport = canvas.parentElement!;
      const zoomText = screen.getByTestId('zoom-level-text');

      // Wheel zoom in with large negative delta
      fireEvent.wheel(viewport, { deltaY: -50000, ctrlKey: true });
      expect(zoomText).toHaveTextContent('115%');

      // Wheel zoom out with large positive delta
      fireEvent.wheel(viewport, { deltaY: 50000, ctrlKey: true });
      expect(zoomText).toHaveTextContent('98%');
    });

    it('handles touch pinch zoom with extreme distance ratios', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const container = screen.getByTestId('document-viewer-container');
      const zoomText = screen.getByTestId('zoom-level-text');

      // Start pinch with 2 fingers at distance 100
      fireEvent.touchStart(container, {
        touches: [
          { clientX: 100, clientY: 100 },
          { clientX: 200, clientY: 100 },
        ],
      });

      // Move fingers apart to distance 1000 (10x ratio) -> should clamp to 500%
      fireEvent.touchMove(container, {
        touches: [
          { clientX: 0, clientY: 100 },
          { clientX: 1000, clientY: 100 },
        ],
      });
      expect(zoomText).toHaveTextContent('500%');

      // Move fingers together to distance 1 (0.01x ratio) -> should clamp to 20%
      fireEvent.touchMove(container, {
        touches: [
          { clientX: 100, clientY: 100 },
          { clientX: 101, clientY: 100 },
        ],
      });
      expect(zoomText).toHaveTextContent('20%');

      // End touch
      fireEvent.touchEnd(container);
    });

    it('resets view cleanly after extreme zoom, pan, and rotation', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const canvas = screen.getByTestId('viewer-canvas-wrapper');
      const viewport = canvas.parentElement!;
      const zoomInBtn = screen.getByTestId('btn-zoom-in');
      const rotateBtn = screen.getByTestId('btn-rotate');
      const resetBtn = screen.getByTestId('btn-reset-view');
      const zoomText = screen.getByTestId('zoom-level-text');

      // Zoom in, rotate twice, drag pan
      fireEvent.click(zoomInBtn);
      fireEvent.click(zoomInBtn);
      fireEvent.click(rotateBtn);
      fireEvent.click(rotateBtn);
      fireEvent.mouseDown(viewport, { clientX: 200, clientY: 200, button: 0 });
      fireEvent.mouseMove(viewport, { clientX: 500, clientY: 600 });
      fireEvent.mouseUp(viewport);

      expect(canvas.style.transform).toContain('rotate(180deg)');

      // Reset
      fireEvent.click(resetBtn);
      expect(zoomText).toHaveTextContent('100%');
      expect(canvas.style.transform).toContain('translate(0px, 0px) scale(1) rotate(0deg)');
    });
  });

  // =========================================================================
  // 3. MALFORMED DOCUMENT DATA & CORRUPTED BOUNDING BOXES IN SVG LAYER
  // =========================================================================
  describe('3. Malformed Document Data & Corrupted Bounding Boxes Rendering', () => {
    it('renders safely when page has NaN dimensions and corrupt bounding boxes', () => {
      const corruptPage: PageResult = {
        page_number: 1,
        width: 0,
        height: 0,
        mean_confidence: NaN,
        full_text: 'Corrupt Page',
        lines: [
          {
            line_id: 'l_corrupt_1',
            line_number: 1,
            text: 'Corrupted Line with NaN bbox',
            original_text: 'Corrupted Line with NaN bbox',
            confidence: NaN,
            bbox: [NaN, Infinity, -Infinity, NaN] as unknown as BoundingBoxTuple,
            words: [
              {
                word_id: 'w_corrupt_1',
                text: 'NaNWord',
                original_text: 'NaNWord',
                confidence: -0.5,
                bbox: [-2.0, 1.5, 3.0, -1.0] as unknown as BoundingBoxTuple,
                alternatives: [
                  { text: 'Alt1', confidence: 0.8 },
                  'PlainStringAlt',
                  { text: 'AltNaN', confidence: NaN },
                ],
              },
            ],
          },
        ],
      };

      render(<DocumentViewer page={corruptPage} />);

      expect(screen.getByTestId('document-viewer-container')).toBeInTheDocument();
      expect(screen.getByTestId('svg-bbox-layer')).toBeInTheDocument();
      expect(screen.getByTestId('svg-line-rect-l_corrupt_1')).toBeInTheDocument();
      expect(screen.getByTestId('svg-word-rect-w_corrupt_1')).toBeInTheDocument();
    });

    it('renders hover tooltips with candidate alternatives and clears on line mouseLeave', () => {
      const pageWithAlts: PageResult = {
        page_number: 1,
        width: 1000,
        height: 1000,
        mean_confidence: 0.85,
        full_text: 'Amoxicillin',
        lines: [
          {
            line_id: 'l_alt',
            line_number: 1,
            text: 'Amoxicillin',
            confidence: 0.85,
            bbox: [0.1, 0.1, 0.2, 0.8],
            words: [
              {
                word_id: 'w_alt',
                text: 'Amoxicillin',
                confidence: 0.85,
                bbox: [0.1, 0.1, 0.2, 0.5],
                alternatives: [
                  { text: 'Ampicillin', confidence: 0.72 },
                  'Amoxil',
                  { text: 'Augmentin', confidence: 0.65 },
                ],
              },
            ],
          },
        ],
      };

      render(<DocumentViewer page={pageWithAlts} />);

      const wordRect = screen.getByTestId('svg-word-rect-w_alt');
      const lineRect = screen.getByTestId('svg-line-rect-l_alt');

      fireEvent.mouseEnter(wordRect);

      const tooltip = screen.getByTestId('viewer-tooltip');
      expect(tooltip).toBeInTheDocument();
      expect(tooltip).toHaveTextContent('Amoxicillin');
      expect(screen.getByTestId('tooltip-alternatives')).toHaveTextContent('Ampicillin (72%)');
      expect(screen.getByTestId('tooltip-alternatives')).toHaveTextContent('Amoxil');
      expect(screen.getByTestId('tooltip-alternatives')).toHaveTextContent('Augmentin (65%)');

      fireEvent.mouseLeave(lineRect);
      expect(screen.queryByTestId('viewer-tooltip')).not.toBeInTheDocument();
    });

    it('handles multi-page navigation across 5+ pages with callback triggers', () => {
      const onPageChange = vi.fn();
      const multiPages: PageResult[] = Array.from({ length: 5 }, (_, i) => ({
        page_number: i + 1,
        width: 1000,
        height: 1200,
        mean_confidence: 0.8 + i * 0.03,
        full_text: `Page text ${i + 1}`,
        lines: [],
      }));

      render(
        <DocumentViewer
          page={multiPages[0]}
          allPages={multiPages}
          activePageIndex={0}
          onPageChange={onPageChange}
        />
      );

      const sidebar = screen.getByTestId('multipage-sidebar');
      expect(sidebar).toBeInTheDocument();
      expect(screen.getByTestId('page-thumbnail-1')).toBeInTheDocument();
      expect(screen.getByTestId('page-thumbnail-5')).toBeInTheDocument();

      fireEvent.click(screen.getByTestId('page-thumbnail-3'));
      expect(onPageChange).toHaveBeenCalledWith(2);
    });

    it('toggles bounding boxes and confidence heatmaps correctly', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const toggleBboxBtn = screen.getByTestId('btn-toggle-bboxes');
      const toggleHeatmapBtn = screen.getByTestId('btn-toggle-heatmap');

      // SVG layer initially visible
      expect(screen.getByTestId('svg-bbox-layer')).toBeInTheDocument();

      // Toggle bboxes off
      fireEvent.click(toggleBboxBtn);
      expect(screen.queryByTestId('svg-bbox-layer')).not.toBeInTheDocument();

      // Toggle bboxes back on
      fireEvent.click(toggleBboxBtn);
      expect(screen.getByTestId('svg-bbox-layer')).toBeInTheDocument();

      // Toggle heatmap off
      fireEvent.click(toggleHeatmapBtn);
      const lineRect = screen.getByTestId('svg-line-rect-p1_l1');
      expect(lineRect.getAttribute('fill')).toBe('rgba(99, 102, 241, 0.08)');
    });
  });
});

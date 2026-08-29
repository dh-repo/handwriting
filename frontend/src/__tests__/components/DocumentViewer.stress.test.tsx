/**
 * frontend/src/__tests__/components/DocumentViewer.stress.test.tsx
 * Dedicated Empirical Stress Test Suite for DocumentViewer Canvas, Pan/Zoom,
 * Rotation, SVG Heatmap Layer, and Edge-Case Rendering.
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DocumentViewer } from '@/components/DocumentViewer';
import { PageResult, BoundingBoxTuple } from '@/types/ocr';
import { SAMPLE_CLEAN_CURSIVE } from '@/lib/sampleDocuments';

describe('DocumentViewer Component Empirical Stress Test Suite', () => {
  describe('1. Zoom Limits & Boundary Clamping Stress', () => {
    it('strictly restricts zoom factor to [0.2x, 5.0x] regardless of repeated button triggers', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const zoomInBtn = screen.getByTestId('btn-zoom-in');
      const zoomOutBtn = screen.getByTestId('btn-zoom-out');
      const zoomText = screen.getByTestId('zoom-level-text');

      // Click Zoom In 40 times
      for (let i = 0; i < 40; i++) {
        fireEvent.click(zoomInBtn);
      }
      expect(zoomText).toHaveTextContent('500%');

      // Click Zoom Out 50 times
      for (let i = 0; i < 50; i++) {
        fireEvent.click(zoomOutBtn);
      }
      expect(zoomText).toHaveTextContent('20%');
    });

    it('handles direct slider adjustment within limits', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const slider = screen.getByTestId('zoom-slider');
      const zoomText = screen.getByTestId('zoom-level-text');

      fireEvent.change(slider, { target: { value: '3.5' } });
      expect(zoomText).toHaveTextContent('350%');

      fireEvent.change(slider, { target: { value: '0.2' } });
      expect(zoomText).toHaveTextContent('20%');
    });

    it('handles wheel zooming with massive delta values', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const canvas = screen.getByTestId('viewer-canvas-wrapper');
      const viewport = canvas.parentElement!;
      const zoomText = screen.getByTestId('zoom-level-text');

      // Negative delta (zoom in)
      fireEvent.wheel(viewport, { deltaY: -10000, ctrlKey: true });
      expect(zoomText).toHaveTextContent('115%');

      // Positive delta (zoom out)
      fireEvent.wheel(viewport, { deltaY: 10000, ctrlKey: true });
      expect(zoomText).toHaveTextContent('98%');
    });

    it('handles pinch zoom gestures with extreme touch ratios', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const container = screen.getByTestId('document-viewer-container');
      const zoomText = screen.getByTestId('zoom-level-text');

      fireEvent.touchStart(container, {
        touches: [
          { clientX: 100, clientY: 100 },
          { clientX: 200, clientY: 100 },
        ],
      });

      // Expand to massive distance
      fireEvent.touchMove(container, {
        touches: [
          { clientX: 0, clientY: 100 },
          { clientX: 2000, clientY: 100 },
        ],
      });
      expect(zoomText).toHaveTextContent('500%');

      // Shrink to tiny distance
      fireEvent.touchMove(container, {
        touches: [
          { clientX: 100, clientY: 100 },
          { clientX: 101, clientY: 100 },
        ],
      });
      expect(zoomText).toHaveTextContent('20%');

      fireEvent.touchEnd(container);
    });
  });

  describe('2. Rotation, Pan, and Reset View Operations', () => {
    it('cycles through 90-degree rotation increments and resets correctly', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const canvas = screen.getByTestId('viewer-canvas-wrapper');
      const rotateBtn = screen.getByTestId('btn-rotate');
      const resetBtn = screen.getByTestId('btn-reset-view');
      const zoomText = screen.getByTestId('zoom-level-text');

      expect(canvas.style.transform).toContain('rotate(0deg)');

      fireEvent.click(rotateBtn);
      expect(canvas.style.transform).toContain('rotate(90deg)');

      fireEvent.click(rotateBtn);
      expect(canvas.style.transform).toContain('rotate(180deg)');

      fireEvent.click(rotateBtn);
      expect(canvas.style.transform).toContain('rotate(270deg)');

      fireEvent.click(rotateBtn);
      expect(canvas.style.transform).toContain('rotate(0deg)');

      // Perform pan drag
      const viewport = canvas.parentElement!;
      fireEvent.mouseDown(viewport, { clientX: 100, clientY: 100, button: 0 });
      fireEvent.mouseMove(viewport, { clientX: 300, clientY: 400 });
      fireEvent.mouseUp(viewport);

      // Reset
      fireEvent.click(resetBtn);
      expect(zoomText).toHaveTextContent('100%');
      expect(canvas.style.transform).toContain('translate(0px, 0px) scale(1) rotate(0deg)');
    });
  });

  describe('3. Degenerate Coordinate Resilience & SVG Layer', () => {
    it('safely renders page with NaN coordinates, negative bounding boxes, and missing polygons', () => {
      const degeneratePage: PageResult = {
        page_number: 1,
        width: 1200,
        height: 1600,
        mean_confidence: 0.75,
        full_text: 'Degenerate Coordinates Test',
        lines: [
          {
            line_id: 'l_degen_1',
            line_number: 1,
            text: 'Degenerate Line',
            original_text: 'Degenerate Line',
            confidence: 0.75,
            bbox: [NaN, -0.5, 1.5, Infinity] as unknown as BoundingBoxTuple,
            words: [
              {
                word_id: 'w_degen_1',
                text: 'DegenWord',
                confidence: 0.45,
                bbox: [0.8, 0.9, 0.2, 0.1] as unknown as BoundingBoxTuple, // inverted
                alternatives: [
                  { text: 'AltTerm', confidence: 0.65 },
                  'PlainStringCandidate',
                ],
              },
            ],
          },
        ],
      };

      render(<DocumentViewer page={degeneratePage} />);

      expect(screen.getByTestId('document-viewer-container')).toBeInTheDocument();
      expect(screen.getByTestId('svg-bbox-layer')).toBeInTheDocument();
      expect(screen.getByTestId('svg-line-rect-l_degen_1')).toBeInTheDocument();
      expect(screen.getByTestId('svg-word-rect-w_degen_1')).toBeInTheDocument();

      // Test hover tooltip display
      const wordRect = screen.getByTestId('svg-word-rect-w_degen_1');
      fireEvent.mouseEnter(wordRect);

      const tooltip = screen.getByTestId('viewer-tooltip');
      expect(tooltip).toBeInTheDocument();
      expect(tooltip).toHaveTextContent('DegenWord');
      expect(screen.getByTestId('tooltip-alternatives')).toHaveTextContent('AltTerm (65%)');
      expect(screen.getByTestId('tooltip-alternatives')).toHaveTextContent('PlainStringCandidate');
    });

    it('toggles bounding box visibility and heatmap styling', () => {
      const page = SAMPLE_CLEAN_CURSIVE.pages[0];
      render(<DocumentViewer page={page} />);

      const toggleBboxBtn = screen.getByTestId('btn-toggle-bboxes');
      const toggleHeatmapBtn = screen.getByTestId('btn-toggle-heatmap');

      expect(screen.getByTestId('svg-bbox-layer')).toBeInTheDocument();

      // Hide bounding boxes
      fireEvent.click(toggleBboxBtn);
      expect(screen.queryByTestId('svg-bbox-layer')).not.toBeInTheDocument();

      // Show bounding boxes again
      fireEvent.click(toggleBboxBtn);
      expect(screen.getByTestId('svg-bbox-layer')).toBeInTheDocument();

      // Disable heatmap
      fireEvent.click(toggleHeatmapBtn);
      const lineRect = screen.getByTestId('svg-line-rect-p1_l1');
      expect(lineRect.getAttribute('fill')).toBe('rgba(99, 102, 241, 0.08)');
    });
  });

  describe('4. Multi-Page Navigation Sidebar', () => {
    it('renders thumbnails for all pages and fires onPageChange callback', () => {
      const onPageChange = vi.fn();
      const pages: PageResult[] = [
        {
          page_number: 1,
          width: 800,
          height: 1000,
          mean_confidence: 0.9,
          full_text: 'Page 1',
          lines: [],
        },
        {
          page_number: 2,
          width: 800,
          height: 1000,
          mean_confidence: 0.85,
          full_text: 'Page 2',
          lines: [],
        },
        {
          page_number: 3,
          width: 800,
          height: 1000,
          mean_confidence: 0.95,
          full_text: 'Page 3',
          lines: [],
        },
      ];

      render(
        <DocumentViewer
          page={pages[0]}
          allPages={pages}
          activePageIndex={0}
          onPageChange={onPageChange}
        />
      );

      const sidebar = screen.getByTestId('multipage-sidebar');
      expect(sidebar).toBeInTheDocument();
      expect(screen.getByTestId('page-thumbnail-1')).toBeInTheDocument();
      expect(screen.getByTestId('page-thumbnail-2')).toBeInTheDocument();
      expect(screen.getByTestId('page-thumbnail-3')).toBeInTheDocument();

      fireEvent.click(screen.getByTestId('page-thumbnail-2'));
      expect(onPageChange).toHaveBeenCalledWith(1);
    });
  });
});

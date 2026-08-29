import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { DocumentViewer } from '@/components/DocumentViewer';
import { PageResult } from '@/types/ocr';

describe('DocumentViewer Interactive Features & Heatmaps', () => {
  const tieredPage: PageResult = {
    page_number: 1,
    width: 800,
    height: 1000,
    full_text: 'High line\nMedium line\nLow line',
    mean_confidence: 0.80,
    lines: [
      {
        line_id: 'line_high',
        text: 'High line',
        confidence: 0.95, // High tier >= 0.90
        bbox: [0.1, 0.1, 0.2, 0.9],
        words: [
          { word_id: 'w_high', text: 'High', confidence: 0.95, bbox: [0.1, 0.1, 0.2, 0.5] },
        ],
      },
      {
        line_id: 'line_med',
        text: 'Medium line',
        confidence: 0.78, // Medium tier 0.70-0.89
        bbox: [0.3, 0.1, 0.4, 0.9],
        words: [
          { word_id: 'w_med', text: 'Medium', confidence: 0.78, bbox: [0.3, 0.1, 0.4, 0.5] },
        ],
      },
      {
        line_id: 'line_low',
        text: 'Low line',
        confidence: 0.55, // Low tier < 0.70
        bbox: [0.5, 0.1, 0.6, 0.9],
        words: [
          {
            word_id: 'w_low',
            text: 'Low',
            confidence: 0.55,
            bbox: [0.5, 0.1, 0.6, 0.5],
            alternatives: [
              { text: 'Law', confidence: 0.45 },
              { text: 'Lay', confidence: 0.35 },
            ],
          },
        ],
      },
    ],
  };

  describe('Continuous Zoom Slider & Fit to Width', () => {
    it('updates scale when using continuous zoom slider', () => {
      render(<DocumentViewer page={tieredPage} />);

      const slider = screen.getByTestId('zoom-slider');
      expect(slider).toBeInTheDocument();
      expect(slider).toHaveValue('1');

      fireEvent.change(slider, { target: { value: '2.5' } });
      expect(screen.getByTestId('zoom-level-text')).toHaveTextContent('250%');
    });

    it('handles Fit to Width button', () => {
      render(<DocumentViewer page={tieredPage} />);

      const fitWidthBtn = screen.getByTestId('btn-zoom-fit-width');
      fireEvent.click(fitWidthBtn);

      expect(screen.getByTestId('zoom-level-text')).toBeInTheDocument();
    });
  });

  describe('3-Tier Confidence Heatmap & Dashed Bounding Box Styling', () => {
    it('applies emerald stroke for high confidence line (>=90%)', () => {
      render(<DocumentViewer page={tieredPage} />);

      const highRect = screen.getByTestId('svg-line-rect-line_high');
      expect(highRect).toHaveAttribute('stroke', '#10b981'); // emerald-500
      expect(highRect).not.toHaveAttribute('stroke-dasharray');
    });

    it('applies amber stroke for medium confidence line (70%-89%)', () => {
      render(<DocumentViewer page={tieredPage} />);

      const medRect = screen.getByTestId('svg-line-rect-line_med');
      expect(medRect).toHaveAttribute('stroke', '#f59e0b'); // amber-500
      expect(medRect).not.toHaveAttribute('stroke-dasharray');
    });

    it('applies rose stroke and dashed strokeDasharray for low confidence line (<70%)', () => {
      render(<DocumentViewer page={tieredPage} />);

      const lowRect = screen.getByTestId('svg-line-rect-line_low');
      expect(lowRect).toHaveAttribute('stroke', '#f43f5e'); // rose-500
      expect(lowRect).toHaveAttribute('stroke-dasharray', '4 2');
    });
  });

  describe('Hover Tooltip with Alternatives Beam Display', () => {
    it('displays candidate alternatives when hovering over low confidence word', () => {
      render(<DocumentViewer page={tieredPage} />);

      const lowWordRect = screen.getByTestId('svg-word-rect-w_low');
      fireEvent.mouseEnter(lowWordRect);

      expect(screen.getByTestId('viewer-tooltip')).toBeInTheDocument();
      expect(screen.getByTestId('tooltip-alternatives')).toBeInTheDocument();
      expect(screen.getByTestId('tooltip-alternatives')).toHaveTextContent('Alternatives: Law (45%), Lay (35%)');
    });
  });

  describe('Touch Gesture Pinch Zoom', () => {
    it('handles multi-touch pinch to zoom in and out', () => {
      render(<DocumentViewer page={tieredPage} />);

      const container = screen.getByTestId('document-viewer-container');

      // Pinch Start: 2 touch points 100px apart
      fireEvent.touchStart(container, {
        touches: [
          { clientX: 100, clientY: 100 },
          { clientX: 200, clientY: 100 },
        ],
      });

      // Pinch Move: fingers move apart to 200px (2x scale)
      fireEvent.touchMove(container, {
        touches: [
          { clientX: 50, clientY: 100 },
          { clientX: 250, clientY: 100 },
        ],
      });

      expect(screen.getByTestId('zoom-level-text')).toHaveTextContent('200%');

      // Pinch End
      fireEvent.touchEnd(container);
    });
  });
});

import React, { useState } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DocumentViewer } from '@/components/DocumentViewer';
import { InlineEditor } from '@/components/InlineEditor';
import { PageResult, LineItem } from '@/types/ocr';

describe('Side-by-Side Verification & Synchronized Hover / Confidence Highlighting', () => {
  const testPage: PageResult = {
    page_number: 1,
    width: 800,
    height: 600,
    mean_confidence: 0.82,
    full_text: 'Review cursive handwriting specimen',
    lines: [
      {
        line_id: 'line_1',
        text: 'Review cursive handwriting specimen',
        confidence: 0.82,
        bbox: [0.1, 0.1, 0.2, 0.9],
        words: [
          {
            word_id: 'w_high',
            text: 'Review',
            confidence: 0.96, // high confidence
            bbox: [0.1, 0.1, 0.2, 0.3],
          },
          {
            word_id: 'w_low',
            text: 'cursive',
            confidence: 0.68, // low confidence -> amber highlight!
            bbox: [0.1, 0.35, 0.2, 0.55],
          },
          {
            word_id: 'w_specimen',
            text: 'specimen',
            confidence: 0.92,
            bbox: [0.1, 0.6, 0.2, 0.9],
          },
        ],
      },
    ],
  };

  const Harness: React.FC = () => {
    const [selectedLineId, setSelectedLineId] = useState<string | null>(null);
    const [selectedWordId, setSelectedWordId] = useState<string | null>(null);
    const [hoveredLineId, setHoveredLineId] = useState<string | null>(null);
    const [hoveredWordId, setHoveredWordId] = useState<string | null>(null);

    return (
      <div className="grid grid-cols-2">
        <DocumentViewer
          page={testPage}
          selectedLineId={selectedLineId}
          selectedWordId={selectedWordId}
          hoveredLineId={hoveredLineId}
          hoveredWordId={hoveredWordId}
          onSelectLine={(lId) => setSelectedLineId(lId)}
          onSelectWord={(wId, lId) => {
            setSelectedWordId(wId);
            if (lId) setSelectedLineId(lId);
          }}
          onHoverLine={(lId) => setHoveredLineId(lId)}
          onHoverWord={(wId) => setHoveredWordId(wId)}
        />
        <InlineEditor
          page={testPage}
          selectedLineId={selectedLineId}
          selectedWordId={selectedWordId}
          hoveredLineId={hoveredLineId}
          hoveredWordId={hoveredWordId}
          onSelectLine={(lId) => setSelectedLineId(lId)}
          onSelectWord={(wId, lId) => {
            setSelectedWordId(wId);
            if (lId) setSelectedLineId(lId);
          }}
          onHoverLine={(lId) => setHoveredLineId(lId)}
          onHoverWord={(wId) => setHoveredWordId(wId)}
        />
      </div>
    );
  };

  it('flags low-confidence tokens in light amber in both editor and canvas', () => {
    render(<Harness />);

    // In InlineEditor: low confidence word chip has data-confidence-low="true"
    const lowChip = screen.getByTestId('word-chip-w_low');
    expect(lowChip).toHaveAttribute('data-confidence-low', 'true');
    expect(lowChip.className).toContain('text-amber');

    // High confidence word is not flagged
    const highChip = screen.getByTestId('word-chip-w_high');
    expect(highChip).toHaveAttribute('data-confidence-low', 'false');

    // In DocumentViewer: word bounding box rect has data-confidence-low="true" and amber stroke
    const svgLowRect = screen.getByTestId('svg-word-rect-w_low');
    expect(svgLowRect).toHaveAttribute('data-confidence-low', 'true');
    expect(svgLowRect).toHaveAttribute('stroke', '#f59e0b');
  });

  it('synchronizes hover state: hovering word in editor highlights exact crop on scan', () => {
    render(<Harness />);

    const lowChip = screen.getByTestId('word-chip-w_low');
    const svgLowRect = screen.getByTestId('svg-word-rect-w_low');

    // Hover word chip in editor
    fireEvent.mouseEnter(lowChip);

    // Bounding box on scan should have active amber stroke and fill
    expect(svgLowRect).toHaveAttribute('stroke', '#f59e0b');
    expect(svgLowRect).toHaveAttribute('fill', 'rgba(245, 158, 11, 0.3)');

    // Leave hover
    fireEvent.mouseLeave(lowChip);
  });

  it('synchronizes selection: clicking word in editor sets active bounding box on scan', () => {
    render(<Harness />);

    const highChip = screen.getByTestId('word-chip-w_high');
    const svgHighRect = screen.getByTestId('svg-word-rect-w_high');

    fireEvent.click(highChip);

    // Selected word rect has cyan stroke
    expect(svgHighRect).toHaveAttribute('stroke', '#06b6d4');
    expect(svgHighRect).toHaveAttribute('fill', 'rgba(6, 182, 212, 0.35)');

    // Precision crop halo is rendered around the selected word
    expect(screen.getByTestId('svg-crop-halo-w_high')).toBeInTheDocument();

    // Focus crop button is displayed in viewer toolbar
    const focusBtn = screen.getByTestId('btn-focus-crop');
    expect(focusBtn).toBeInTheDocument();
    fireEvent.click(focusBtn);
  });
});

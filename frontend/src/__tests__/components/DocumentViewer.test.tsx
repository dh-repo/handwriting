import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { DocumentViewer } from '@/components/DocumentViewer';
import { SAMPLE_CLEAN_CURSIVE, SAMPLE_MULTIPAGE } from '@/lib/sampleDocuments';

describe('DocumentViewer Component', () => {
  it('renders base canvas wrapper and SVG vector bounding box layer', () => {
    render(<DocumentViewer page={SAMPLE_CLEAN_CURSIVE.pages[0]} />);

    expect(screen.getByTestId('document-viewer-container')).toBeInTheDocument();
    expect(screen.getByTestId('viewer-canvas-wrapper')).toBeInTheDocument();
    expect(screen.getByTestId('svg-bbox-layer')).toBeInTheDocument();
  });

  it('renders all line bounding box rectangles', () => {
    render(<DocumentViewer page={SAMPLE_CLEAN_CURSIVE.pages[0]} />);

    SAMPLE_CLEAN_CURSIVE.pages[0].lines.forEach((line) => {
      expect(screen.getByTestId(`svg-line-rect-${line.line_id}`)).toBeInTheDocument();
    });
  });

  it('handles zoom controls properly', () => {
    render(<DocumentViewer page={SAMPLE_CLEAN_CURSIVE.pages[0]} />);

    expect(screen.getByTestId('zoom-level-text')).toHaveTextContent('100%');

    const zoomInBtn = screen.getByTestId('btn-zoom-in');
    fireEvent.click(zoomInBtn);
    expect(screen.getByTestId('zoom-level-text')).toHaveTextContent('120%');

    const zoomOutBtn = screen.getByTestId('btn-zoom-out');
    fireEvent.click(zoomOutBtn);
    expect(screen.getByTestId('zoom-level-text')).toHaveTextContent('100%');

    const resetBtn = screen.getByTestId('btn-reset-view');
    fireEvent.click(resetBtn);
    expect(screen.getByTestId('zoom-level-text')).toHaveTextContent('100%');
  });

  it('toggles bounding box layer visibility', () => {
    render(<DocumentViewer page={SAMPLE_CLEAN_CURSIVE.pages[0]} />);

    expect(screen.getByTestId('svg-bbox-layer')).toBeInTheDocument();

    const toggleBBoxesBtn = screen.getByTestId('btn-toggle-bboxes');
    fireEvent.click(toggleBBoxesBtn);

    expect(screen.queryByTestId('svg-bbox-layer')).not.toBeInTheDocument();
  });

  it('triggers line selection and hover callbacks', () => {
    const onSelectLine = vi.fn();
    const onHoverLine = vi.fn();

    render(
      <DocumentViewer
        page={SAMPLE_CLEAN_CURSIVE.pages[0]}
        onSelectLine={onSelectLine}
        onHoverLine={onHoverLine}
      />
    );

    const firstLineRect = screen.getByTestId('svg-line-rect-p1_l1');
    fireEvent.click(firstLineRect);
    expect(onSelectLine).toHaveBeenCalledWith('p1_l1');

    fireEvent.mouseEnter(firstLineRect);
    expect(onHoverLine).toHaveBeenCalledWith('p1_l1');
    expect(screen.getByTestId('viewer-tooltip')).toBeInTheDocument();

    fireEvent.mouseLeave(firstLineRect);
    expect(onHoverLine).toHaveBeenCalledWith(null);
  });

  it('renders multi-page sidebar when multiple pages are passed', () => {
    const onPageChange = vi.fn();
    render(
      <DocumentViewer
        page={SAMPLE_MULTIPAGE.pages[0]}
        allPages={SAMPLE_MULTIPAGE.pages}
        activePageIndex={0}
        onPageChange={onPageChange}
      />
    );

    expect(screen.getByTestId('multipage-sidebar')).toBeInTheDocument();
    expect(screen.getByTestId('page-thumbnail-1')).toBeInTheDocument();
    expect(screen.getByTestId('page-thumbnail-2')).toBeInTheDocument();
    expect(screen.getByTestId('page-thumbnail-3')).toBeInTheDocument();

    fireEvent.click(screen.getByTestId('page-thumbnail-2'));
    expect(onPageChange).toHaveBeenCalledWith(1);
  });
});

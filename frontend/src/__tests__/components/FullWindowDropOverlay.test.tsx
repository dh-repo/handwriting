import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { FullWindowDropOverlay } from '@/components/FullWindowDropOverlay';

describe('FullWindowDropOverlay Component', () => {
  it('activates assertive visual overlay on window dragenter with files', () => {
    const onFilesDropped = vi.fn();
    render(<FullWindowDropOverlay onFilesDropped={onFilesDropped} />);

    // Initially not visible
    expect(screen.queryByTestId('full-window-drop-overlay')).not.toBeInTheDocument();

    // Trigger dragenter on window with Files type
    fireEvent.dragEnter(window, {
      dataTransfer: {
        types: ['Files'],
      },
    });

    expect(screen.getByTestId('full-window-drop-overlay')).toBeInTheDocument();
    expect(screen.getByText(/Drop Files or Folders Anywhere/i)).toBeInTheDocument();
    expect(screen.getByText(/PNG, JPEG, TIFF/i)).toBeInTheDocument();
    expect(screen.getByText(/Multi-page PDFs/i)).toBeInTheDocument();
    expect(screen.getByText(/Nested Folders/i)).toBeInTheDocument();
  });

  it('hides overlay when dragleave fires back to 0 counter', () => {
    const onFilesDropped = vi.fn();
    render(<FullWindowDropOverlay onFilesDropped={onFilesDropped} />);

    fireEvent.dragEnter(window, {
      dataTransfer: { types: ['Files'] },
    });
    expect(screen.getByTestId('full-window-drop-overlay')).toBeInTheDocument();

    fireEvent.dragLeave(window);
    expect(screen.queryByTestId('full-window-drop-overlay')).not.toBeInTheDocument();
  });

  it('dispatches dropped files and hides overlay on drop', () => {
    const onFilesDropped = vi.fn();
    render(<FullWindowDropOverlay onFilesDropped={onFilesDropped} />);

    fireEvent.dragEnter(window, {
      dataTransfer: { types: ['Files'] },
    });

    const file1 = new File(['content-1'], 'page1.png', { type: 'image/png' });
    const file2 = new File(['content-2'], 'notes.pdf', { type: 'application/pdf' });

    fireEvent.drop(window, {
      dataTransfer: {
        files: [file1, file2],
      },
    });

    expect(screen.queryByTestId('full-window-drop-overlay')).not.toBeInTheDocument();
    expect(onFilesDropped).toHaveBeenCalledWith([file1, file2]);
  });
});

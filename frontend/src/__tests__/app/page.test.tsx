import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import WorkspacePage from '@/app/page';
import { DocumentProvider } from '@/context/DocumentContext';

describe('Workspace Page Integration', () => {
  it('renders initial dropzone and sample gallery when no document is loaded', () => {
    render(
      <DocumentProvider>
        <WorkspacePage />
      </DocumentProvider>
    );

    expect(screen.getByText(/Handwriting Recognition AI/i)).toBeInTheDocument();
    expect(screen.getByTestId('dropzone-container')).toBeInTheDocument();
    expect(screen.getByTestId('sample-gallery-container')).toBeInTheDocument();
  });

  it('loads sample document when clicked and renders viewer, editor, and export toolbar', () => {
    render(
      <DocumentProvider>
        <WorkspacePage />
      </DocumentProvider>
    );

    // Click Clean Cursive card
    const sampleCard = screen.getByTestId('sample-card-sample_clean_cursive');
    fireEvent.click(sampleCard);

    // Verify workspace transitions to active document state
    expect(screen.getByTestId('metrics-summary-container')).toBeInTheDocument();
    expect(screen.getByTestId('document-viewer-container')).toBeInTheDocument();
    expect(screen.getByTestId('inline-editor-container')).toBeInTheDocument();
    expect(screen.getByTestId('export-toolbar-container')).toBeInTheDocument();
    expect(screen.getByTestId('btn-load-new-document')).toBeInTheDocument();
  });

  it('synchronizes line edits and allows resetting back to dropzone', () => {
    render(
      <DocumentProvider>
        <WorkspacePage />
      </DocumentProvider>
    );

    // Load sample
    const sampleCard = screen.getByTestId('sample-card-sample_clean_cursive');
    fireEvent.click(sampleCard);

    // Edit a line
    const lineInput = screen.getByTestId('line-input-p1_l1');
    fireEvent.change(lineInput, { target: { value: 'Modified cursive line' } });

    expect(screen.getByTestId('edited-badge-p1_l1')).toBeInTheDocument();
    expect(screen.getByTestId('edited-count-badge')).toBeInTheDocument();

    // Reset workspace
    const resetBtn = screen.getByTestId('btn-load-new-document');
    fireEvent.click(resetBtn);

    expect(screen.getByTestId('dropzone-container')).toBeInTheDocument();
    expect(screen.queryByTestId('document-viewer-container')).not.toBeInTheDocument();
  });

  it('revokes object URLs when unmounting or switching documents to prevent memory leaks', async () => {
    // Mock fetch for file upload
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        document_id: 'doc_test_123',
        pages: [
          {
            page_number: 1,
            image_url: 'blob:mock-url',
            width: 800,
            height: 600,
            lines: [],
          },
        ],
        full_text: '',
        processing_time_ms: 50,
      }),
    });
    global.fetch = mockFetch;

    const { unmount } = render(
      <DocumentProvider>
        <WorkspacePage />
      </DocumentProvider>
    );

    const input = screen.getByTestId('dropzone-input');
    const file = new File(['fake-image-bytes'], 'test.png', { type: 'image/png' });
    fireEvent.change(input, { target: { files: [file] } });

    // Wait for async file accepted handling
    await screen.findByTestId('btn-load-new-document');

    expect(window.URL.createObjectURL).toHaveBeenCalled();

    // Reset document
    const resetBtn = screen.getByTestId('btn-load-new-document');
    fireEvent.click(resetBtn);

    expect(window.URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');

    // Unmount component
    unmount();
  });
});


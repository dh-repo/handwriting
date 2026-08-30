import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import WorkspacePage from '@/app/page';
import { DocumentProvider } from '@/context/DocumentContext';

describe('Workspace Page Integration', () => {
  it('renders initial Apple HIG dropzone when no document is loaded', () => {
    render(
      <DocumentProvider>
        <WorkspacePage />
      </DocumentProvider>
    );

    expect(screen.getByText(/Handwriting AI/i)).toBeInTheDocument();
    expect(screen.getByTestId('dropzone-container')).toBeInTheDocument();
  });

  it('renders workspace with document viewer and editor when file is uploaded', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        document_id: 'doc_test_123',
        filename: 'test_cursive.png',
        total_pages: 1,
        pages: [
          {
            page_number: 1,
            image_url: 'blob:mock-url',
            width: 800,
            height: 600,
            mean_confidence: 0.95,
            lines: [
              {
                line_id: 'p1_l1',
                text: 'First line of cursive text',
                confidence: 0.95,
                bbox: [0.1, 0.1, 0.2, 0.9],
                words: [
                  {
                    word_id: 'p1_l1_w1',
                    text: 'First',
                    confidence: 0.95,
                    bbox: [0.1, 0.1, 0.2, 0.3],
                    polygon: [[0.1, 0.1], [0.1, 0.3], [0.2, 0.3], [0.2, 0.1]],
                  },
                ],
              },
            ],
          },
        ],
        full_text: 'First line of cursive text',
        processing_time_ms: 120,
      }),
    });
    global.fetch = mockFetch;

    render(
      <DocumentProvider>
        <WorkspacePage />
      </DocumentProvider>
    );

    const input = screen.getByTestId('dropzone-input');
    const file = new File(['fake-image-bytes'], 'test_cursive.png', { type: 'image/png' });
    fireEvent.change(input, { target: { files: [file] } });

    expect(await screen.findByText(/test_cursive.png/i)).toBeInTheDocument();
    expect(screen.getByText(/New Scan/i)).toBeInTheDocument();
    expect(screen.getByTestId('document-viewer-container')).toBeInTheDocument();
    expect(screen.getByTestId('inline-editor-container')).toBeInTheDocument();
  });

  it('allows resetting back to clean dropzone via New Scan button', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        document_id: 'doc_test_123',
        filename: 'test_cursive.png',
        total_pages: 1,
        pages: [
          {
            page_number: 1,
            image_url: 'blob:mock-url',
            width: 800,
            height: 600,
            mean_confidence: 0.95,
            lines: [],
          },
        ],
        full_text: '',
        processing_time_ms: 50,
      }),
    });
    global.fetch = mockFetch;

    render(
      <DocumentProvider>
        <WorkspacePage />
      </DocumentProvider>
    );

    const input = screen.getByTestId('dropzone-input');
    const file = new File(['fake-image-bytes'], 'test_cursive.png', { type: 'image/png' });
    fireEvent.change(input, { target: { files: [file] } });

    const newScanBtn = await screen.findByText(/New Scan/i);
    fireEvent.click(newScanBtn);

    expect(screen.getByTestId('dropzone-container')).toBeInTheDocument();
    expect(screen.queryByTestId('document-viewer-container')).not.toBeInTheDocument();
  });
});

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import WorkspacePage from '@/app/page';
import { DocumentProvider } from '@/context/DocumentContext';

describe('Batch Processing & High-Throughput Ingestion Integration', () => {
  it('transitions to staging queue when multiple files are dropped', async () => {
    render(
      <DocumentProvider>
        <WorkspacePage />
      </DocumentProvider>
    );

    const input = screen.getByTestId('dropzone-input');
    const file1 = new File(['bytes1'], 'archival_letter.png', { type: 'image/png' });
    const file2 = new File(['bytes2'], 'notes.pdf', { type: 'application/pdf' });

    // Drop multiple files
    fireEvent.change(input, { target: { files: [file1, file2] } });

    // Expect staging queue to appear with both files
    expect(await screen.findByTestId('staging-queue-container')).toBeInTheDocument();
    expect(screen.getByText(/archival_letter.png/i)).toBeInTheDocument();
    expect(screen.getByText(/notes.pdf/i)).toBeInTheDocument();
    expect(screen.getByTestId('queue-count-badge')).toHaveTextContent('2 files');
    expect(screen.getByTestId('batch-configuration-panel')).toBeInTheDocument();
  });

  it('allows 1-click zero-friction sample preview directly from landing', async () => {
    render(
      <DocumentProvider>
        <WorkspacePage />
      </DocumentProvider>
    );

    // Verify 3 zero-friction sample cards are present below drop box
    expect(screen.getByTestId('zero-friction-samples-container')).toBeInTheDocument();
    const card18th = screen.getByTestId('sample-card-sample_18th_century');

    // Click 18th-century cursive sample card
    fireEvent.click(card18th);

    // Enters split-pane workspace immediately
    expect(await screen.findByTestId('document-viewer-container')).toBeInTheDocument();
    expect(screen.getByTestId('inline-editor-container')).toBeInTheDocument();
    expect(screen.getByText(/sample_clean_cursive.png/i)).toBeInTheDocument();
  });

  it('runs batch processing and enables live streaming review and multi-doc switching', async () => {
    const mockFetch = vi.fn().mockImplementation((url: string) => {
      // Return a stream response for recognize-stream
      const encoder = new TextEncoder();
      const mockDoc = {
        document_id: 'doc_batch_1',
        filename: 'archival_letter.png',
        total_pages: 1,
        pages: [
          {
            page_number: 1,
            width: 800,
            height: 600,
            mean_confidence: 0.94,
            lines: [
              {
                line_id: 'l1',
                text: 'Transcribed batch line one',
                confidence: 0.94,
                bbox: [0.1, 0.1, 0.2, 0.8],
                words: [{ word_id: 'w1', text: 'Transcribed', confidence: 0.94, bbox: [0.1, 0.1, 0.2, 0.4] }],
              },
            ],
          },
        ],
        full_text: 'Transcribed batch line one',
        processing_time_ms: 80,
      };

      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(
            encoder.encode(
              `event: line\ndata: ${JSON.stringify({
                line_id: 'l1',
                text: 'Transcribed batch line one',
                confidence: 0.94,
                bbox: [0.1, 0.1, 0.2, 0.8],
                words: [],
                page_number: 1,
              })}\n\n`
            )
          );
          controller.enqueue(encoder.encode(`event: complete\ndata: ${JSON.stringify(mockDoc)}\n\n`));
          controller.close();
        },
      });

      return Promise.resolve(
        new Response(stream, {
          headers: { 'Content-Type': 'text/event-stream' },
        })
      );
    });
    global.fetch = mockFetch;

    render(
      <DocumentProvider>
        <WorkspacePage />
      </DocumentProvider>
    );

    const input = screen.getByTestId('dropzone-input');
    const file1 = new File(['bytes1'], 'archival_letter.png', { type: 'image/png' });
    const file2 = new File(['bytes2'], 'notes.png', { type: 'image/png' });
    fireEvent.change(input, { target: { files: [file1, file2] } });

    expect(await screen.findByTestId('staging-queue-container')).toBeInTheDocument();

    // Start batch processing
    const startBtn = screen.getByTestId('btn-start-batch-execution');
    fireEvent.click(startBtn);

    // Expect workspace to activate with live streaming review
    await waitFor(() => {
      expect(screen.getByTestId('document-viewer-container')).toBeInTheDocument();
      expect(screen.getByTestId('inline-editor-container')).toBeInTheDocument();
      expect(screen.getByTestId('btn-export-searchable-pdf')).toBeInTheDocument();
    });
  });
});

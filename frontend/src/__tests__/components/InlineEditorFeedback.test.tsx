/**
 * frontend/src/__tests__/components/InlineEditorFeedback.test.tsx
 * Comprehensive Integration & Lifecycle Test Suite for Darkroom Operator Feedback:
 * 500ms Debouncing, Immediate Dispatch (Enter, Blur, 1-5 Quick-Picks),
 * 4-State Visual Indicators (debouncing -> syncing -> synced -> error),
 * Optimistic UI Updates, and Speed Review Queue Dispatch.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { InlineEditor } from '@/components/InlineEditor';
import { PageResult, FeedbackSubmissionRequest, FeedbackSubmissionResponse } from '@/types/ocr';
import { ApiClient } from '@/lib/apiClient';
import { POST as feedbackRoute } from '@/app/api/feedback/route';

describe('InlineEditor Feedback Integration Suite', () => {
  const mockPage: PageResult = {
    page_number: 1,
    width: 800,
    height: 1100,
    image_url: 'blob:test-document-page',
    mean_confidence: 0.72,
    full_text: 'Amoxicilln 500mg\nTake one tablet daily',
    lines: [
      {
        line_id: 'line_01',
        text: 'Amoxicilln 500mg',
        original_text: 'Amoxicilln 500mg',
        confidence: 0.65,
        bbox: [0.1, 0.1, 0.2, 0.9],
        words: [
          {
            word_id: 'word_01',
            text: 'Amoxicilln',
            original_text: 'Amoxicilln',
            confidence: 0.55,
            bbox: [0.1, 0.1, 0.2, 0.4],
          },
          {
            word_id: 'word_02',
            text: '500mg',
            original_text: '500mg',
            confidence: 0.95,
            bbox: [0.1, 0.45, 0.2, 0.7],
          },
        ],
      },
      {
        line_id: 'line_02',
        text: 'Take one tablet daily',
        original_text: 'Take one tablet daily',
        confidence: 0.92,
        bbox: [0.25, 0.1, 0.35, 0.9],
        words: [
          {
            word_id: 'word_03',
            text: 'Take',
            original_text: 'Take',
            confidence: 0.94,
            bbox: [0.25, 0.1, 0.35, 0.25],
          },
        ],
      },
    ],
  };

  let mockSubmitFeedback: ReturnType<typeof vi.fn>;
  let mockClient: ApiClient;

  beforeEach(() => {
    mockSubmitFeedback = vi.fn().mockImplementation(async (req: FeedbackSubmissionRequest) => {
      return {
        status: 'persisted',
        feedback_id: `fb_mock_${Date.now()}`,
        document_id: req.document_id,
        line_id: req.line_id,
        manifest_path: 'data/feedback/manifest.jsonl',
        confusion_pairs_count: 1,
        timestamp: req.timestamp || new Date().toISOString(),
      } as FeedbackSubmissionResponse;
    });

    mockClient = {
      submitFeedback: mockSubmitFeedback,
    } as unknown as ApiClient;
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  describe('1. 500ms Debouncing & Continuous Keystroke Typing', () => {
    it('debounces rapid continuous typing and dispatches only once after 500ms trailing-edge', async () => {
      vi.useFakeTimers();

      render(
        <InlineEditor
          page={mockPage}
          documentId="doc_test_flywheel"
          apiClientInstance={mockClient}
        />
      );

      const input = screen.getByTestId('line-input-line_01');

      // Continuous typing: "A" -> "Am" -> "Amox"
      act(() => {
        fireEvent.change(input, { target: { value: 'A' } });
      });

      // Visual indicator immediately transitions to 'debouncing' (Staged...)
      expect(screen.getByTestId('feedback-status-line_01')).toHaveTextContent(/Staged/i);
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Advance 200ms and type more
      act(() => {
        vi.advanceTimersByTime(200);
        fireEvent.change(input, { target: { value: 'Am' } });
      });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Advance 200ms more and finish word
      act(() => {
        vi.advanceTimersByTime(200);
        fireEvent.change(input, { target: { value: 'Amoxicillin 500mg' } });
      });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Advance 300ms (total 300ms since last keystroke, still < 500ms debounce)
      act(() => {
        vi.advanceTimersByTime(300);
      });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Advance remaining 250ms to cross 500ms boundary
      await act(async () => {
        vi.advanceTimersByTime(250);
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      const callArgs = mockSubmitFeedback.mock.calls[0][0];
      expect(callArgs.document_id).toBe('doc_test_flywheel');
      expect(callArgs.line_id).toBe('line_01');
      expect(callArgs.original_prediction).toBe('Amoxicilln 500mg');
      expect(callArgs.operator_correction).toBe('Amoxicillin 500mg');
      expect(callArgs.original_text).toBe('Amoxicilln 500mg');
      expect(callArgs.corrected_text).toBe('Amoxicillin 500mg');

      // Visual badge transitions to 'synced' (Flywheel Saved)
      await act(async () => {
        await Promise.resolve();
      });
      expect(screen.getByTestId('feedback-status-line_01')).toHaveTextContent(/Flywheel Saved/i);
    });
  });

  describe('2. Immediate Dispatch on Discrete Actions (Enter & Blur)', () => {
    it('immediately flushes feedback on Enter key without waiting for 500ms debounce', async () => {
      vi.useFakeTimers();

      render(
        <InlineEditor
          page={mockPage}
          documentId="doc_test_flywheel"
          apiClientInstance={mockClient}
        />
      );

      const input = screen.getByTestId('line-input-line_01');

      act(() => {
        fireEvent.change(input, { target: { value: 'Amoxicillin 500mg' } });
      });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Immediate Enter keypress
      await act(async () => {
        fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      expect(mockSubmitFeedback.mock.calls[0][0].corrected_text).toBe('Amoxicillin 500mg');

      // Ensure advancing time does NOT fire a duplicate submission
      await act(async () => {
        vi.advanceTimersByTime(1000);
        await Promise.resolve();
      });
      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
    });

    it('immediately flushes feedback on onBlur event', async () => {
      vi.useFakeTimers();

      render(
        <InlineEditor
          page={mockPage}
          documentId="doc_test_flywheel"
          apiClientInstance={mockClient}
        />
      );

      const input = screen.getByTestId('line-input-line_01');

      act(() => {
        fireEvent.change(input, { target: { value: 'Amoxicillin 500mg' } });
      });
      expect(mockSubmitFeedback).not.toHaveBeenCalled();

      // Blur input
      await act(async () => {
        fireEvent.blur(input);
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      expect(mockSubmitFeedback.mock.calls[0][0].corrected_text).toBe('Amoxicillin 500mg');
    });
  });

  describe('3. Optimistic UI Updates & Visual Confirmation Statuses', () => {
    it('updates text optimistically without waiting for server response', () => {
      render(
        <InlineEditor
          page={mockPage}
          documentId="doc_test_flywheel"
          apiClientInstance={mockClient}
        />
      );

      const input = screen.getByTestId('line-input-line_01') as HTMLInputElement;
      fireEvent.change(input, { target: { value: 'Instant Text' } });
      expect(input.value).toBe('Instant Text');
    });

    it('transitions through error state and allows retry on network failure', async () => {
      const failingSubmit = vi.fn().mockRejectedValueOnce(new Error('Backend connection refused'));
      const retryClient = {
        submitFeedback: failingSubmit,
      } as unknown as ApiClient;

      render(
        <InlineEditor
          page={mockPage}
          documentId="doc_test_flywheel"
          apiClientInstance={retryClient}
        />
      );

      const input = screen.getByTestId('line-input-line_01');

      // Submit via Enter
      await act(async () => {
        fireEvent.change(input, { target: { value: 'Retry Candidate' } });
        fireEvent.keyDown(input, { key: 'Enter' });
      });

      // Verify Error state
      await waitFor(() => {
        expect(screen.getByTestId('feedback-status-line_01')).toHaveTextContent(/Sync Failed/i);
      });
      const retryBtn = screen.getByTestId('retry-feedback-line_01');
      expect(retryBtn).toBeInTheDocument();

      // Make next submission succeed
      failingSubmit.mockResolvedValueOnce({
        status: 'persisted',
        feedback_id: 'fb_retry_success',
        document_id: 'doc_test_flywheel',
        line_id: 'line_01',
        timestamp: new Date().toISOString(),
      });

      await act(async () => {
        fireEvent.click(retryBtn);
      });

      await waitFor(() => {
        expect(screen.getByTestId('feedback-status-line_01')).toHaveTextContent(/Flywheel Saved/i);
      });
      expect(failingSubmit).toHaveBeenCalledTimes(2);
    });
  });

  describe('4. Speed Review Integration & Immediate Dispatch', () => {
    it('dispatches feedback immediately on speed review correction submit and advances to next item', async () => {
      render(
        <InlineEditor
          page={mockPage}
          documentId="doc_test_flywheel"
          apiClientInstance={mockClient}
        />
      );

      // Switch to speed review tab
      const speedTab = screen.getByTestId('tab-speed-review');
      fireEvent.click(speedTab);

      expect(screen.getByTestId('speed-review-panel')).toBeInTheDocument();
      const speedInput = screen.getByTestId('speed-review-input');
      expect(speedInput).toHaveValue('Amoxicilln');

      // Type correction and accept
      await act(async () => {
        fireEvent.change(speedInput, { target: { value: 'Amoxicillin' } });
        const acceptBtn = screen.getByTestId('btn-speed-accept');
        fireEvent.click(acceptBtn);
      });

      await waitFor(() => {
        expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      });

      const call = mockSubmitFeedback.mock.calls[0][0];
      expect(call.line_id).toBe('line_01');
      expect(call.word_id).toBe('word_01');
      expect(call.original_prediction).toBe('Amoxicilln');
      expect(call.operator_correction).toBe('Amoxicillin');
      expect(call.original_text).toBe('Amoxicilln');
      expect(call.corrected_text).toBe('Amoxicillin');
    });

    it('dispatches feedback when selecting a quick-pick suggestion in speed review', async () => {
      render(
        <InlineEditor
          page={mockPage}
          documentId="doc_test_flywheel"
          apiClientInstance={mockClient}
        />
      );

      // Switch to speed review tab
      const speedTab = screen.getByTestId('tab-speed-review');
      fireEvent.click(speedTab);

      // Wait for suggestion 1
      const suggestionBtn = await screen.findByTestId('speed-suggestion-1');
      await act(async () => {
        fireEvent.click(suggestionBtn);
      });

      await waitFor(() => {
        expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      });

      const call = mockSubmitFeedback.mock.calls[0][0];
      expect(call.line_id).toBe('line_01');
      expect(call.word_id).toBe('word_01');
      expect(call.original_prediction).toBe('Amoxicilln');
      expect(call.operator_correction).toMatch(/amoxicillin/i);
      expect(call.original_text).toBe('Amoxicilln');
      expect(call.corrected_text).toMatch(/amoxicillin/i);
    });
  });

  describe('5. Structured Word Chip Editing & Autocomplete Popover Feedback', () => {
    it('dispatches feedback when editing a word chip and pressing Enter', async () => {
      render(
        <InlineEditor
          page={mockPage}
          documentId="doc_test_flywheel"
          apiClientInstance={mockClient}
        />
      );

      // Click word chip to activate inline word edit input
      const wordChip = screen.getByTestId('word-chip-word_01');
      fireEvent.click(wordChip);

      const wordInput = screen.getByTestId('word-edit-input-word_01');
      expect(wordInput).toBeInTheDocument();

      await act(async () => {
        fireEvent.change(wordInput, { target: { value: 'Amoxicillin' } });
        fireEvent.keyDown(wordInput, { key: 'Enter' });
      });

      await waitFor(() => {
        expect(mockSubmitFeedback).toHaveBeenCalledTimes(1);
      });

      const call = mockSubmitFeedback.mock.calls[0][0];
      expect(call.line_id).toBe('line_01');
      expect(call.word_id).toBe('word_01');
      expect(call.original_prediction).toBe('Amoxicilln');
      expect(call.operator_correction).toBe('Amoxicillin');
      expect(call.original_text).toBe('Amoxicilln');
      expect(call.corrected_text).toBe('Amoxicillin');
    });

    it('clears feedback statuses and pending timers when Revert All is clicked', async () => {
      const onRevertAll = vi.fn();
      const modifiedPage: PageResult = {
        ...mockPage,
        lines: mockPage.lines.map((l, i) =>
          i === 0 ? { ...l, is_edited: true, text: 'Modified text' } : l
        ),
      };

      render(
        <InlineEditor
          page={modifiedPage}
          documentId="doc_test_flywheel"
          apiClientInstance={mockClient}
          onRevertAll={onRevertAll}
        />
      );

      const input = screen.getByTestId('line-input-line_01');
      act(() => {
        fireEvent.change(input, { target: { value: 'Staged change' } });
      });

      expect(screen.getByTestId('feedback-status-line_01')).toHaveTextContent(/Staged/i);

      const revertBtn = screen.getByTestId('btn-revert-all');
      act(() => {
        fireEvent.click(revertBtn);
      });

      expect(onRevertAll).toHaveBeenCalled();
      expect(screen.queryByTestId('feedback-status-line_01')).not.toBeInTheDocument();
    });
  });

  describe('6. ApiClient submitFeedback Lifecycle & Fallback Harness', () => {
    it('calls FastAPI /v1/feedback when baseUrl is configured', async () => {
      const origFetch = global.fetch;
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          status: 'persisted',
          feedback_id: 'fb_fastapi_123',
          document_id: 'doc_backend',
          line_id: 'line_backend',
          manifest_path: 'data/feedback/manifest.jsonl',
          timestamp: '2026-09-03T12:00:00Z',
        }),
      });
      global.fetch = mockFetch;

      try {
        const client = new ApiClient({ baseUrl: 'http://localhost:8000', enableFallback: true });
        const res = await client.submitFeedback({
          document_id: 'doc_backend',
          line_id: 'line_backend',
          original_text: 'pred',
          corrected_text: 'corr',
        });

        expect(res.status).toBe('persisted');
        expect(res.feedback_id).toBe('fb_fastapi_123');
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/v1/feedback',
          expect.objectContaining({
            method: 'POST',
          })
        );
      } finally {
        global.fetch = origFetch;
      }
    });

    it('falls back to Next.js route proxy /api/feedback when baseUrl is not configured', async () => {
      const origFetch = global.fetch;
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          status: 'persisted',
          feedback_id: 'fb_proxy_456',
          document_id: 'doc_proxy',
          line_id: 'line_proxy',
          timestamp: '2026-09-03T12:00:00Z',
        }),
      });
      global.fetch = mockFetch;

      try {
        const client = new ApiClient({ baseUrl: '', enableFallback: true });
        const res = await client.submitFeedback({
          document_id: 'doc_proxy',
          line_id: 'line_proxy',
          original_text: 'pred',
          corrected_text: 'corr',
        });

        expect(res.feedback_id).toBe('fb_proxy_456');
        expect(mockFetch).toHaveBeenCalledWith(
          '/api/feedback',
          expect.objectContaining({
            method: 'POST',
          })
        );
      } finally {
        global.fetch = origFetch;
      }
    });

    it('falls back to simulated persisted mock response when offline or proxy fails and fallback is enabled', async () => {
      const origFetch = global.fetch;
      global.fetch = vi.fn().mockRejectedValue(new Error('Network offline'));

      try {
        const client = new ApiClient({ baseUrl: 'http://localhost:8000', enableFallback: true });
        const res = await client.submitFeedback({
          document_id: 'doc_offline',
          line_id: 'line_offline',
          original_text: 'pred',
          corrected_text: 'corr',
        });

        expect(res.status).toBe('persisted');
        expect(res.document_id).toBe('doc_offline');
        expect(res.line_id).toBe('line_offline');
        expect(res.feedback_id).toMatch(/^fb_mock_/);
      } finally {
        global.fetch = origFetch;
      }
    });
  });

  describe('7. /api/feedback Route Handler', () => {
    const origEnv = process.env.BACKEND_URL;

    afterEach(() => {
      process.env.BACKEND_URL = origEnv;
    });

    it('returns 400 Bad Request when required fields are missing', async () => {
      const badReq = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: 'doc_123' }), // missing line_id, texts
      });

      const res = await feedbackRoute(badReq);
      expect(res.status).toBe(400);
      const json = await res.json();
      expect(json.error).toMatch(/Missing required feedback fields/i);
    });

    it('returns simulated persisted response with mock header when BACKEND_URL is not set', async () => {
      delete process.env.BACKEND_URL;

      const validReq = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'doc_mock_route',
          line_id: 'line_mock_route',
          original_text: 'old text',
          corrected_text: 'new text',
        }),
      });

      const res = await feedbackRoute(validReq);
      expect(res.status).toBe(200);
      expect(res.headers.get('X-Feedback-Provider')).toBe('mock');
      const json = await res.json();
      expect(json.status).toBe('persisted');
      expect(json.document_id).toBe('doc_mock_route');
      expect(json.feedback_id).toMatch(/^fb_mock_/);
    });

    it('proxies to backend when BACKEND_URL is configured and returns 200', async () => {
      process.env.BACKEND_URL = 'http://backend-ai:8000';
      const origFetch = global.fetch;
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({
          status: 'persisted',
          feedback_id: 'fb_remote_999',
          document_id: 'doc_remote',
          line_id: 'line_remote',
          confusion_pairs_updated: [{ source: 'c', target: 'e' }],
          timestamp: '2026-09-03T14:00:00Z',
        }),
      });
      global.fetch = mockFetch;

      try {
        const req = new Request('http://localhost:3000/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            document_id: 'doc_remote',
            line_id: 'line_remote',
            original_text: 'c',
            corrected_text: 'e',
          }),
        });

        const res = await feedbackRoute(req);
        expect(res.status).toBe(200);
        expect(res.headers.get('X-Feedback-Provider')).toBe('backend');
        const json = await res.json();
        expect(json.feedback_id).toBe('fb_remote_999');
        expect(mockFetch).toHaveBeenCalledWith(
          'http://backend-ai:8000/v1/feedback',
          expect.objectContaining({ method: 'POST' })
        );
        const forwardedBody = JSON.parse(mockFetch.mock.calls[0][1].body);
        expect(forwardedBody.original_prediction).toBe('c');
        expect(forwardedBody.operator_correction).toBe('e');
      } finally {
        global.fetch = origFetch;
      }
    });

    it('returns 502 Bad Gateway when backend is unreachable', async () => {
      process.env.BACKEND_URL = 'http://backend-ai:8000';
      const origFetch = global.fetch;
      global.fetch = vi.fn().mockRejectedValue(new Error('ECONNREFUSED'));

      try {
        const req = new Request('http://localhost:3000/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            document_id: 'doc_err',
            line_id: 'line_err',
            original_text: 'a',
            corrected_text: 'b',
          }),
        });

        const res = await feedbackRoute(req);
        expect(res.status).toBe(502);
        const json = await res.json();
        expect(json.error).toMatch(/unreachable/i);
      } finally {
        global.fetch = origFetch;
      }
    });
  });
});



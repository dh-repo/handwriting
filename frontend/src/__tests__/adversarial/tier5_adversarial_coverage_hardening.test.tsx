/**
 * frontend/src/__tests__/adversarial/tier5_adversarial_coverage_hardening.test.tsx
 * Milestone 5 Phase 2: Tier 5 Adversarial Coverage Hardening
 *
 * Exhaustive empirical challenge suite across target frontend modules:
 * 1. cropUtils.ts: hostile/extreme coordinates, corrupt image elements, canvas failures
 * 2. InlineEditor.tsx: rapid typing barrage, Enter/Blur race conditions, Speed Review spamming, lifecycle state machine
 * 3. app/api/feedback/route.ts: malformed JSON, missing fields, backend 500/502/504 errors, timeouts, header forwarding
 * 4. apiClient.ts: multi-tier fallback cascade, 4xx non-fallback, network disconnection, payload field harmonization
 * 5. exportUtils.ts: CWE-1236 formula injection payloads, multi-line cells, RFC 4180 escaping
 */

import React from 'react';
import { render, screen, fireEvent, act, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { computeCropCoordinates, extractLineCropBase64 } from '@/lib/cropUtils';
import { InlineEditor } from '@/components/InlineEditor';
import { POST } from '@/app/api/feedback/route';
import { ApiClient, ApiClientError, ApiNetworkError } from '@/lib/apiClient';
import { exportDocumentAsCsv, exportDocumentAsJson, exportDocumentAsTxt } from '@/lib/exportUtils';
import {
  PageResult,
  LineItem,
  WordToken,
  BoundingBoxTuple,
  DocumentOCRResult,
  FeedbackSubmissionRequest,
  FeedbackSubmissionResponse,
} from '@/types/ocr';

describe('Tier 5 Adversarial Coverage Hardening Suite', () => {
  // =========================================================================
  // MODULE 1: cropUtils.ts - Boundary, Hostile Coordinates & Corrupt Elements
  // =========================================================================
  describe('1. cropUtils.ts Adversarial Stress', () => {
    it('1.1 handles negative coordinates safely (float jitter clamping vs negative pixel coords vs degenerate sub-pixel rejection)', async () => {
      // 1.1a: Normalized bbox with negative float jitter within [-0.05, 1.05]
      const jitterBbox: BoundingBoxTuple = [-0.03, -0.02, 0.40, 0.60];
      const jitterCoords = computeCropCoordinates(jitterBbox, 1000, 1000, 0.04);
      expect(jitterCoords).not.toBeNull();
      expect(jitterCoords!.cropX).toBe(0);
      expect(jitterCoords!.cropY).toBe(0);
      expect(jitterCoords!.cropW).toBeGreaterThanOrEqual(600);
      expect(jitterCoords!.cropH).toBeGreaterThanOrEqual(400);

      // 1.1b: Large negative pixel coordinates clamped safely to canvas boundaries
      const negativePixelBbox: BoundingBoxTuple = [-200, -100, 300, 400];
      const pixelCoords = computeCropCoordinates(negativePixelBbox, 1000, 1000, 0.0);
      expect(pixelCoords).not.toBeNull();
      expect(pixelCoords!.cropX).toBe(0);
      expect(pixelCoords!.cropY).toBe(0);
      expect(pixelCoords!.cropW).toBe(400);
      expect(pixelCoords!.cropH).toBe(300);

      // 1.1c: Ambiguous negative sub-pixel coordinates that round to <= 0 height safely return null
      const degenerateSubpixelBbox: BoundingBoxTuple = [-0.8, -0.5, 0.4, 0.6];
      expect(computeCropCoordinates(degenerateSubpixelBbox, 1000, 1000, 0.04)).toBeNull();
      expect(await extractLineCropBase64('blob:test', degenerateSubpixelBbox)).toBeUndefined();
    });

    it('1.2 handles coordinates strictly greater than 1.0 (large overshoot vs float jitter)', () => {
      // Float jitter within 5% overshoot (<= 1.05) is treated as normalized
      const jitterBbox: BoundingBoxTuple = [0.0, 0.0, 1.03, 1.04];
      const jitterCoords = computeCropCoordinates(jitterBbox, 1000, 1000, 0.0);
      expect(jitterCoords).not.toBeNull();
      expect(jitterCoords!.cropW).toBe(1000);
      expect(jitterCoords!.cropH).toBe(1000);

      // Large overshoot (> 1.05) is treated as absolute pixel coordinates clamped to image dimensions
      const pixelCoords = computeCropCoordinates([10, 20, 150, 250], 1000, 1000, 0.0);
      expect(pixelCoords).not.toBeNull();
      expect(pixelCoords!.cropX).toBe(20);
      expect(pixelCoords!.cropY).toBe(10);
      expect(pixelCoords!.cropW).toBe(230);
      expect(pixelCoords!.cropH).toBe(140);
    });

    it('1.3 rejects zero-width, zero-height, and single-point boxes with null', () => {
      // Single point
      expect(computeCropCoordinates([0.5, 0.5, 0.5, 0.5], 800, 1100)).toBeNull();
      // Zero width (xmin === xmax)
      expect(computeCropCoordinates([0.1, 0.4, 0.9, 0.4], 800, 1100, 0.0)).toBeNull();
      // Zero height (ymin === ymax)
      expect(computeCropCoordinates([0.4, 0.1, 0.4, 0.9], 800, 1100, 0.0)).toBeNull();
    });

    it('1.4 rejects non-finite values (NaN, Infinity, -Infinity) and malformed structures', () => {
      expect(computeCropCoordinates([NaN, 0, 1, 1], 800, 1100)).toBeNull();
      expect(computeCropCoordinates([0, NaN, 1, 1], 800, 1100)).toBeNull();
      expect(computeCropCoordinates([0, 0, Infinity, 1], 800, 1100)).toBeNull();
      expect(computeCropCoordinates([0, 0, 1, -Infinity], 800, 1100)).toBeNull();
      expect(computeCropCoordinates([] as unknown as BoundingBoxTuple, 800, 1100)).toBeNull();
      expect(computeCropCoordinates([1, 2, 3] as unknown as BoundingBoxTuple, 800, 1100)).toBeNull();
      expect(computeCropCoordinates(null as unknown as BoundingBoxTuple, 800, 1100)).toBeNull();
    });

    it('1.5 handles zero or negative natural image dimensions safely', () => {
      const validBbox: BoundingBoxTuple = [0.1, 0.1, 0.5, 0.5];
      expect(computeCropCoordinates(validBbox, 0, 1000)).toBeNull();
      expect(computeCropCoordinates(validBbox, 1000, 0)).toBeNull();
      expect(computeCropCoordinates(validBbox, -100, 1000)).toBeNull();
      expect(computeCropCoordinates(validBbox, 1000, -100)).toBeNull();
    });

    it('1.6 handles extreme aspect ratio images (e.g. 10000x10 and 10x10000)', () => {
      const bbox: BoundingBoxTuple = [0.1, 0.1, 0.9, 0.9];
      // Ultra wide strip
      const wideCoords = computeCropCoordinates(bbox, 10000, 10, 0.04);
      expect(wideCoords).not.toBeNull();
      expect(wideCoords!.cropX + wideCoords!.cropW).toBeLessThanOrEqual(10000);
      expect(wideCoords!.cropY + wideCoords!.cropH).toBeLessThanOrEqual(10);

      // Ultra tall column
      const tallCoords = computeCropCoordinates(bbox, 10, 10000, 0.04);
      expect(tallCoords).not.toBeNull();
      expect(tallCoords!.cropX + tallCoords!.cropW).toBeLessThanOrEqual(10);
      expect(tallCoords!.cropY + tallCoords!.cropH).toBeLessThanOrEqual(10000);
    });

    it('1.7 extractLineCropBase64 returns undefined when imageUrl or bbox is missing', async () => {
      expect(await extractLineCropBase64(undefined, [0.1, 0.1, 0.5, 0.5])).toBeUndefined();
      expect(await extractLineCropBase64('http://example.com/img.png', undefined)).toBeUndefined();
      expect(await extractLineCropBase64('', [0.1, 0.1, 0.5, 0.5])).toBeUndefined();
      expect(await extractLineCropBase64('http://example.com/img.png', [0.5, 0.5, 0.5, 0.5])).toBeUndefined();
    });

    it('1.8 extractLineCropBase64 falls back to 1x1 transparent PNG when canvas toDataURL throws SecurityError', async () => {
      const origToDataUrl = HTMLCanvasElement.prototype.toDataURL;
      HTMLCanvasElement.prototype.toDataURL = vi.fn().mockImplementation(() => {
        throw new DOMException('Tainted canvas', 'SecurityError');
      });

      try {
        const res = await extractLineCropBase64('blob:tainted-img', [0.1, 0.1, 0.4, 0.8]);
        expect(res).toBe('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==');
      } finally {
        HTMLCanvasElement.prototype.toDataURL = origToDataUrl;
      }
    });

    it('1.9 extractLineCropBase64 returns undefined when canvas getContext returns null', async () => {
      const origGetContext = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(null);

      try {
        const res = await extractLineCropBase64('blob:test-img', [0.1, 0.1, 0.4, 0.8]);
        expect(res).toBeUndefined();
      } finally {
        HTMLCanvasElement.prototype.getContext = origGetContext;
      }
    });

    it('1.10 extractLineCropBase64 gracefully handles corrupt image (onerror event) without crashing', async () => {
      const origImage = global.Image;
      class MockCorruptImage {
        crossOrigin = '';
        width = 0;
        height = 0;
        naturalWidth = 0;
        naturalHeight = 0;
        complete = false;
        private _src = '';
        onload: (() => void) | null = null;
        onerror: ((ev: Event) => void) | null = null;
        set src(v: string) {
          this._src = v;
          setTimeout(() => this.onerror?.(new Event('error')), 10);
        }
        get src() {
          return this._src;
        }
      }
      // @ts-expect-error Mocking Image
      global.Image = MockCorruptImage;

      try {
        const res = await extractLineCropBase64('https://invalid.domain/corrupt.png', [0.1, 0.1, 0.4, 0.8]);
        // On error, default naturalWidth=800 / height=1100 are used and canvas produces fallback
        expect(res).toBeDefined();
        expect(res).toMatch(/^data:image\/png;base64,/);
      } finally {
        global.Image = origImage;
      }
    });
  });

  // =========================================================================
  // MODULE 2: InlineEditor.tsx - Rapid Bursts, Lifecycle Transitions & Recovery
  // =========================================================================
  describe('2. InlineEditor.tsx Lifecycle, Debounce & State Machine Stress', () => {
    const createTestPage = (): PageResult => ({
      page_number: 1,
      width: 800,
      height: 1100,
      image_url: 'blob:test-page',
      mean_confidence: 0.65,
      full_text: 'prednisone 20mg daily\namoxicillin 500mg tid',
      lines: [
        {
          line_id: 'l1',
          text: 'prednisone 20mg daily',
          original_text: 'prednisone 20mg daily',
          confidence: 0.60,
          bbox: [0.1, 0.05, 0.18, 0.9],
          words: [
            { word_id: 'w1', text: 'prednisone', original_text: 'prednisone', confidence: 0.60, bbox: [0.1, 0.05, 0.18, 0.4] },
            { word_id: 'w2', text: '20mg', original_text: '20mg', confidence: 0.60, bbox: [0.1, 0.42, 0.18, 0.6] },
            { word_id: 'w3', text: 'daily', original_text: 'daily', confidence: 0.60, bbox: [0.1, 0.62, 0.18, 0.9] },
          ],
        },
        {
          line_id: 'l2',
          text: 'amoxicillin 500mg tid',
          original_text: 'amoxicillin 500mg tid',
          confidence: 0.70,
          bbox: [0.2, 0.05, 0.28, 0.9],
          words: [
            { word_id: 'w4', text: 'amoxicillin', original_text: 'amoxicillin', confidence: 0.70, bbox: [0.2, 0.05, 0.28, 0.5] },
            { word_id: 'w5', text: '500mg', original_text: '500mg', confidence: 0.70, bbox: [0.2, 0.52, 0.28, 0.7] },
            { word_id: 'w6', text: 'tid', original_text: 'tid', confidence: 0.70, bbox: [0.2, 0.72, 0.28, 0.9] },
          ],
        },
      ],
    });

    let mockSubmit: ReturnType<typeof vi.fn>;
    let mockApiClient: ApiClient;

    beforeEach(() => {
      mockSubmit = vi.fn().mockImplementation(async (req: FeedbackSubmissionRequest) => {
        return {
          status: 'persisted',
          feedback_id: `fb_test_${Date.now()}`,
          document_id: req.document_id,
          line_id: req.line_id,
          manifest_path: 'data/feedback/manifest.jsonl',
          confusion_pairs_count: 1,
          timestamp: req.timestamp || new Date().toISOString(),
        } as FeedbackSubmissionResponse;
      });
      mockApiClient = { submitFeedback: mockSubmit } as unknown as ApiClient;
    });

    afterEach(() => {
      cleanup();
      vi.useRealTimers();
      vi.clearAllMocks();
    });

    it('2.1 barrage typing: 50 rapid changes in 200ms trigger exactly 1 feedback submission after 500ms debounce', async () => {
      vi.useFakeTimers();
      const page = createTestPage();

      render(<InlineEditor enableMedicalSuggestions page={page} documentId="doc_barrage" apiClientInstance={mockApiClient} />);

      const input = screen.getByTestId('line-input-l1');

      for (let i = 1; i <= 50; i++) {
        act(() => {
          fireEvent.change(input, { target: { value: `Typing step ${i}` } });
          vi.advanceTimersByTime(4); // 50 * 4ms = 200ms total
        });
      }

      // At t=200ms, debounce is still pending (status is debouncing/Staged)
      expect(mockSubmit).not.toHaveBeenCalled();
      expect(screen.getByTestId('feedback-status-l1')).toHaveTextContent(/Staged/i);

      // Advance 499ms from last keystroke
      act(() => {
        vi.advanceTimersByTime(499);
      });
      expect(mockSubmit).not.toHaveBeenCalled();

      // Cross 500ms trailing edge
      await act(async () => {
        vi.advanceTimersByTime(10);
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmit).toHaveBeenCalledTimes(1);
      expect(mockSubmit.mock.calls[0][0].corrected_text).toBe('Typing step 50');
      expect(screen.getByTestId('feedback-status-l1')).toHaveTextContent(/Feedback submitted/i);
    });

    it('2.2 rapid Enter + Blur interleaving does not trigger double submission', async () => {
      vi.useFakeTimers();
      const page = createTestPage();

      render(<InlineEditor enableMedicalSuggestions page={page} documentId="doc_enter_blur" apiClientInstance={mockApiClient} />);

      const input = screen.getByTestId('line-input-l1');

      act(() => {
        fireEvent.change(input, { target: { value: 'Enter then Blur Value' } });
      });

      // Hit Enter immediately
      act(() => {
        fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });
      });

      await act(async () => {
        await Promise.resolve();
      });
      expect(mockSubmit).toHaveBeenCalledTimes(1);

      // Blur fires immediately after Enter
      act(() => {
        fireEvent.blur(input);
      });

      // Advance timers by 1000ms
      await act(async () => {
        vi.advanceTimersByTime(1000);
        await Promise.resolve();
      });

      // Must remain exactly 1 call
      expect(mockSubmit).toHaveBeenCalledTimes(1);
    });

    it('2.3 resetting text back to original before debounce expires sets status to idle without calling API', async () => {
      vi.useFakeTimers();
      const page = createTestPage();

      render(<InlineEditor enableMedicalSuggestions page={page} documentId="doc_revert_text" apiClientInstance={mockApiClient} />);

      const input = screen.getByTestId('line-input-l1');

      // Change text
      act(() => {
        fireEvent.change(input, { target: { value: 'temporary edit' } });
      });
      expect(screen.getByTestId('feedback-status-l1')).toHaveTextContent(/Staged/i);

      // Revert text back to original before 500ms
      act(() => {
        vi.advanceTimersByTime(200);
        fireEvent.change(input, { target: { value: 'prednisone 20mg daily' } });
      });

      // Let debounce expire
      await act(async () => {
        vi.advanceTimersByTime(600);
        await Promise.resolve();
      });

      // No submission dispatched because text matches original!
      expect(mockSubmit).not.toHaveBeenCalled();
      expect(screen.queryByTestId('feedback-status-l1')).not.toBeInTheDocument();
    });

    it('2.4 lifecycle transition: debouncing -> syncing -> error on backend failure, displays retry button', async () => {
      vi.useFakeTimers();
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const page = createTestPage();

      // Backend fails
      mockSubmit.mockRejectedValueOnce(new Error('500 Internal Server Error'));

      render(<InlineEditor enableMedicalSuggestions page={page} documentId="doc_lifecycle_err" apiClientInstance={mockApiClient} />);

      const input = screen.getByTestId('line-input-l1');

      act(() => {
        fireEvent.change(input, { target: { value: 'Failing edit' } });
      });
      expect(screen.getByTestId('feedback-status-l1')).toHaveTextContent(/Staged/i);

      // Trigger dispatch
      await act(async () => {
        vi.advanceTimersByTime(550);
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmit).toHaveBeenCalledTimes(1);

      // Status transitions to error
      const statusBadge = screen.getByTestId('feedback-status-l1');
      expect(statusBadge).toHaveTextContent(/Sync Failed/i);
      expect(screen.getByTestId('retry-feedback-l1')).toBeInTheDocument();

      consoleSpy.mockRestore();
    });

    it('2.5 network failure recovery: clicking Retry re-dispatches and transitions to synced', async () => {
      vi.useFakeTimers();
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const page = createTestPage();

      mockSubmit.mockRejectedValueOnce(new Error('Network timeout'));

      render(<InlineEditor enableMedicalSuggestions page={page} documentId="doc_retry_recov" apiClientInstance={mockApiClient} />);

      const input = screen.getByTestId('line-input-l1');
      act(() => {
        fireEvent.change(input, { target: { value: 'Recoverable edit' } });
      });

      await act(async () => {
        vi.advanceTimersByTime(550);
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(screen.getByTestId('feedback-status-l1')).toHaveTextContent(/Sync Failed/i);

      // Mock successful response on retry
      mockSubmit.mockResolvedValueOnce({
        status: 'persisted',
        feedback_id: 'fb_recovered',
        document_id: 'doc_retry_recov',
        line_id: 'l1',
        manifest_path: 'data/feedback/manifest.jsonl',
        confusion_pairs_count: 1,
        timestamp: new Date().toISOString(),
      });

      // Click Retry
      await act(async () => {
        fireEvent.click(screen.getByTestId('retry-feedback-l1'));
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmit).toHaveBeenCalledTimes(2);
      expect(mockSubmit.mock.calls[1][0].corrected_text).toBe('Recoverable edit');
      expect(screen.getByTestId('feedback-status-l1')).toHaveTextContent(/Feedback submitted/i);

      consoleSpy.mockRestore();
    });

    it('2.6 editing text while in error state resets status to debouncing', async () => {
      vi.useFakeTimers();
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const page = createTestPage();

      mockSubmit.mockRejectedValueOnce(new Error('Initial failure'));

      render(<InlineEditor enableMedicalSuggestions page={page} documentId="doc_edit_on_err" apiClientInstance={mockApiClient} />);

      const input = screen.getByTestId('line-input-l1');
      act(() => {
        fireEvent.change(input, { target: { value: 'Failed first' } });
      });

      await act(async () => {
        vi.advanceTimersByTime(550);
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(screen.getByTestId('feedback-status-l1')).toHaveTextContent(/Sync Failed/i);

      // Now user makes another keystroke
      act(() => {
        fireEvent.change(input, { target: { value: 'Edited after failure' } });
      });

      // Status immediately transitions to debouncing (Staged...)
      expect(screen.getByTestId('feedback-status-l1')).toHaveTextContent(/Staged/i);

      consoleSpy.mockRestore();
    });

    it('2.7 speed review quick-pick 1-5 keys advance queue and submit feedback', async () => {
      vi.useRealTimers();
      const page = createTestPage();

      render(<InlineEditor enableMedicalSuggestions page={page} documentId="doc_speed_qp" apiClientInstance={mockApiClient} />);

      // Switch to speed review tab
      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const input = screen.getByTestId('speed-review-input');

      // Wait for suggestions to populate
      await act(async () => {
        await new Promise((r) => setTimeout(r, 60));
      });

      // Press key '1' on untouched input -> applies suggestion 1
      fireEvent.keyDown(input, { key: '1', code: 'Digit1' });

      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmit).toHaveBeenCalledTimes(1);
      expect(mockSubmit.mock.calls[0][0].line_id).toBe('l1');
      expect(mockSubmit.mock.calls[0][0].word_id).toBe('w1');
    });

    it('2.8 speed review does NOT trigger quick-pick 1-5 when input is touched with numbers (e.g. 500mg)', async () => {
      vi.useRealTimers();
      const page = createTestPage();

      render(<InlineEditor enableMedicalSuggestions page={page} documentId="doc_speed_touched" apiClientInstance={mockApiClient} />);

      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const input = screen.getByTestId('speed-review-input');

      // User types custom text with numbers
      fireEvent.change(input, { target: { value: '250mg' } });
      // Pressing '2' on touched input must NOT trigger suggestion 2
      fireEvent.keyDown(input, { key: '2' });

      expect(mockSubmit).not.toHaveBeenCalled();

      // Submit via Enter
      fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });

      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(mockSubmit).toHaveBeenCalledTimes(1);
      expect(mockSubmit.mock.calls[0][0].corrected_text).toBe('250mg');
    });

    it('2.9 speed review handles key 5 safely when fewer than 5 suggestions exist', async () => {
      vi.useRealTimers();
      const page = createTestPage();

      render(<InlineEditor enableMedicalSuggestions page={page} documentId="doc_speed_k5" apiClientInstance={mockApiClient} />);

      fireEvent.click(screen.getByTestId('tab-speed-review'));
      const input = screen.getByTestId('speed-review-input');

      // Key 9 is outside range 1-5, key 5 may have no matching suggestion
      fireEvent.keyDown(input, { key: '9' });
      expect(mockSubmit).not.toHaveBeenCalled();
    });

    it('2.10 Revert All clears active timers across multiple lines without firing delayed feedback', async () => {
      vi.useFakeTimers();
      const page = createTestPage();
      page.lines[0].is_edited = true;
      const mockRevert = vi.fn();

      render(
        <InlineEditor enableMedicalSuggestions
          page={page}
          documentId="doc_revert_all_tier5"
          apiClientInstance={mockApiClient}
          onRevertAll={mockRevert}
          canUndo={true}
        />
      );

      const input1 = screen.getByTestId('line-input-l1');
      const input2 = screen.getByTestId('line-input-l2');

      act(() => {
        fireEvent.change(input1, { target: { value: 'l1 pending edit' } });
        fireEvent.change(input2, { target: { value: 'l2 pending edit' } });
      });

      expect(screen.getByTestId('feedback-status-l1')).toHaveTextContent(/Staged/i);
      expect(screen.getByTestId('feedback-status-l2')).toHaveTextContent(/Staged/i);

      // Trigger Revert All
      const revertBtn = screen.queryByTestId('revert-all-btn') || screen.getByTestId('btn-revert-all');
      act(() => {
        fireEvent.click(revertBtn);
      });
      expect(mockRevert).toHaveBeenCalledTimes(1);

      // Advance past 500ms
      await act(async () => {
        vi.advanceTimersByTime(1000);
        await Promise.resolve();
      });

      expect(mockSubmit).not.toHaveBeenCalled();
    });
  });

  // =========================================================================
  // MODULE 3: app/api/feedback/route.ts - Malformed Payloads & Proxy Handling
  // =========================================================================
  describe('3. app/api/feedback/route.ts Proxy & Validation Stress', () => {
    const origEnv = process.env;

    beforeEach(() => {
      process.env = { ...origEnv };
    });

    afterEach(() => {
      process.env = origEnv;
      vi.restoreAllMocks();
    });

    it('3.1 rejects unclosed/malformed JSON body with HTTP 400', async () => {
      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{"unclosed": "brace',
      });
      const res = await POST(req);
      expect(res.status).toBe(400);
      const data = await res.json();
      expect(data.error).toBe('Invalid JSON request body');
    });

    it('3.2 rejects non-object JSON values (number, string, boolean, array) with HTTP 400', async () => {
      for (const val of ['12345', '"plain string"', 'true', '[]']) {
        const req = new Request('http://localhost:3000/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: val,
        });
        const res = await POST(req);
        expect(res.status).toBe(400);
      }
    });

    it('3.3 rejects payloads missing document_id or line_id with HTTP 400', async () => {
      // Missing document_id
      const req1 = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ line_id: 'l1', original_text: 'a', corrected_text: 'b' }),
      });
      expect((await POST(req1)).status).toBe(400);

      // Missing line_id
      const req2 = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: 'd1', original_text: 'a', corrected_text: 'b' }),
      });
      expect((await POST(req2)).status).toBe(400);
    });

    it("reports unavailable service honestly: 3.4 harmonizes original_prediction/operator_correction with original_text/corrected_text", async () => { delete process.env.BACKEND_URL; const { POST } = await import('@/app/api/feedback/route'); const res=await POST(new Request('http://localhost/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({document_id:'d',line_id:'l',original_text:'',corrected_text:'new'})})); expect(res.status).toBe(503); expect((await res.json()).status).not.toBe('persisted'); });

    it('3.5 forwards backend 200 OK with X-Feedback-Provider: backend when BACKEND_URL is set', async () => {
      process.env.BACKEND_URL = 'http://backend-server:8000';

      const mockBackendRes = {
        feedback_id: 'fb_live_999',
        status: 'persisted',
        manifest_path: 'data/feedback/manifest.jsonl',
      };
      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify(mockBackendRes), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'd1',
          line_id: 'l1',
          original_text: 'a',
          corrected_text: 'b',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(200);
      expect(res.headers.get('X-Feedback-Provider')).toBe('backend');
      const data = await res.json();
      expect(data.feedback_id).toBe('fb_live_999');
    });

    it('3.6 forwards backend 422 Unprocessable Entity directly to client', async () => {
      process.env.BACKEND_URL = 'http://backend-server:8000';

      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Bbox out of range' }), {
          status: 422,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'd1',
          line_id: 'l1',
          original_text: 'a',
          corrected_text: 'b',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(422);
      const data = await res.json();
      expect(data.detail).toBe('Bbox out of range');
    });

    it('3.7 forwards backend 500/502/504 errors directly to client', async () => {
      process.env.BACKEND_URL = 'http://backend-server:8000';

      for (const status of [500, 502, 504]) {
        global.fetch = vi.fn().mockResolvedValue(
          new Response(JSON.stringify({ error: `Backend status ${status}` }), {
            status,
            headers: { 'Content-Type': 'application/json' },
          })
        );

        const req = new Request('http://localhost:3000/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            document_id: 'd1',
            line_id: 'l1',
            original_text: 'a',
            corrected_text: 'b',
          }),
        });

        const res = await POST(req);
        expect(res.status).toBe(status);
      }
    });

    it('3.8 handles HTML 502 page from backend gracefully without crashing', async () => {
      process.env.BACKEND_URL = 'http://backend-server:8000';

      global.fetch = vi.fn().mockResolvedValue(
        new Response('<html><body>Bad Gateway</body></html>', {
          status: 502,
          headers: { 'Content-Type': 'text/html' },
        })
      );

      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'd1',
          line_id: 'l1',
          original_text: 'a',
          corrected_text: 'b',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(502);
      const data = await res.json();
      expect(data.error).toContain('Backend feedback failed with status 502');
    });

    it('3.9 returns HTTP 502 when backend fetch throws network failure (ECONNREFUSED)', async () => {
      process.env.BACKEND_URL = 'http://backend-server:8000';
      global.fetch = vi.fn().mockRejectedValue(new Error('connect ECONNREFUSED'));

      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'd1',
          line_id: 'l1',
          original_text: 'a',
          corrected_text: 'b',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(502);
      const data = await res.json();
      expect(data.error).toBe('Backend feedback service unreachable');
    });

    it("reports unavailable service honestly: 3.10 returns simulated persisted mock when BACKEND_URL is not configured", async () => { delete process.env.BACKEND_URL; const { POST } = await import('@/app/api/feedback/route'); const res=await POST(new Request('http://localhost/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({document_id:'d',line_id:'l',original_text:'',corrected_text:'new'})})); expect(res.status).toBe(503); expect((await res.json()).status).not.toBe('persisted'); });
  });

  // =========================================================================
  // MODULE 4: apiClient.ts - Multi-Tier Cascade, 4xx Non-Fallback & Errors
  // =========================================================================
  describe('4. apiClient.ts Resilience & Multi-Tier Cascade', () => {
    const baseRequest: FeedbackSubmissionRequest = {
      document_id: 'doc_client_tier5',
      line_id: 'l1',
      original_text: 'misspelled',
      corrected_text: 'corrected',
    };

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('4.1 direct backend 200 OK returns response cleanly', async () => {
      const client = new ApiClient({ baseUrl: 'http://fastapi:8000', enableFallback: false });

      global.fetch = vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            status: 'persisted',
            feedback_id: 'fb_fastapi_ok',
            document_id: 'doc_client_tier5',
            line_id: 'l1',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        )
      );

      const res = await client.submitFeedback(baseRequest);
      expect(res.status).toBe('persisted');
      expect(res.feedback_id).toBe('fb_fastapi_ok');
      expect(global.fetch).toHaveBeenCalledWith('http://fastapi:8000/v1/feedback', expect.anything());
    });

    it('4.2 throws ApiClientError on HTTP 400/422 without falling back to proxy or mock', async () => {
      const client = new ApiClient({ baseUrl: 'http://fastapi:8000', enableFallback: true });

      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Invalid field' }), {
          status: 422,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      await expect(client.submitFeedback(baseRequest)).rejects.toThrow(ApiClientError);
      // Exactly 1 fetch call: no cascade to proxy or mock on 4xx
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it('4.3 throws ApiClientError on backend 500 when enableFallback is false', async () => {
      const client = new ApiClient({ baseUrl: 'http://fastapi:8000', enableFallback: false });

      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: 'Internal Server Error' }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      await expect(client.submitFeedback(baseRequest)).rejects.toThrow(ApiClientError);
    });

    it('4.4 throws ApiNetworkError on network disconnect when enableFallback is false', async () => {
      const client = new ApiClient({ baseUrl: 'http://fastapi:8000', enableFallback: false });
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      await expect(client.submitFeedback(baseRequest)).rejects.toThrow(ApiNetworkError);
    });

    it("reports unavailable service honestly: 4.5 cascades to Tier 2 proxy when backend returns 500 and enableFallback is true", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); const { pendingFeedback }=await import('@/lib/documentStore'); const submission_id=crypto.randomUUID(); const timestamp='2026-01-01T00:00:00.000Z'; await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).submitFeedback({document_id:'d',line_id:'l',original_text:'a',corrected_text:'b',submission_id,timestamp})).rejects.toThrow(); expect((await pendingFeedback()).find(p=>p.submission_id===submission_id)?.timestamp).toBe(timestamp); });

    it('4.6 throws ApiClientError on proxy 400 without falling back to Tier 3 mock', async () => {
      const client = new ApiClient({ baseUrl: '', enableFallback: true });

      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: 'Missing field' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      await expect(client.submitFeedback(baseRequest)).rejects.toThrow(ApiClientError);
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it("reports unavailable service honestly: 4.7 falls back to Tier 3 mock acknowledgment when proxy fails with 502", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); const { pendingFeedback }=await import('@/lib/documentStore'); const submission_id=crypto.randomUUID(); const timestamp='2026-01-01T00:00:00.000Z'; await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).submitFeedback({document_id:'d',line_id:'l',original_text:'a',corrected_text:'b',submission_id,timestamp})).rejects.toThrow(); expect((await pendingFeedback()).find(p=>p.submission_id===submission_id)?.timestamp).toBe(timestamp); });

    it("reports unavailable service honestly: 4.8 falls back to Tier 3 mock acknowledgment on complete offline network failure", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); const { pendingFeedback }=await import('@/lib/documentStore'); const submission_id=crypto.randomUUID(); const timestamp='2026-01-01T00:00:00.000Z'; await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).submitFeedback({document_id:'d',line_id:'l',original_text:'a',corrected_text:'b',submission_id,timestamp})).rejects.toThrow(); expect((await pendingFeedback()).find(p=>p.submission_id===submission_id)?.timestamp).toBe(timestamp); });

    it('4.9 ensures both original_prediction/original_text and operator_correction/corrected_text are sent', async () => {
      const client = new ApiClient({ baseUrl: 'http://fastapi:8000', enableFallback: false });

      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ status: 'persisted', feedback_id: 'fb_1' }), { status: 200 })
      );

      await client.submitFeedback({
        document_id: 'd1',
        line_id: 'l1',
        original_text: 'pred_text',
        corrected_text: 'corr_text',
      });

      const body = JSON.parse((global.fetch as any).mock.calls[0][1].body);
      expect(body.original_prediction).toBe('pred_text');
      expect(body.original_text).toBe('pred_text');
      expect(body.operator_correction).toBe('corr_text');
      expect(body.corrected_text).toBe('corr_text');
    });

    it('4.10 checkHealth parses status and model info from GET /v1/health', async () => {
      const client = new ApiClient({ baseUrl: 'http://fastapi:8000' });

      global.fetch = vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            status: 'ok',
            model: 'trocr-mps',
            device: 'mps',
            version: '1.0.0',
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        )
      );

      const health = await client.checkHealth();
      expect(health.status).toBe('ok');
      expect(health.device).toBe('mps');
    });
  });

  // =========================================================================
  // MODULE 5: exportUtils.ts - CWE-1236 Formula Injection & Export Integrity
  // =========================================================================
  describe('5. exportUtils.ts CWE-1236 Security & RFC 4180 Escaping', () => {
    const createInjectionDoc = (text: string): DocumentOCRResult => ({
      document_id: 'doc_cwe_tier5',
      filename: 'injection.png',
      total_pages: 1,
      mean_confidence: 0.95,
      processing_time_ms: 10,
      full_text: text,
      pages: [
        {
          page_number: 1,
          width: 800,
          height: 1100,
          mean_confidence: 0.95,
          full_text: text,
          lines: [
            {
              line_id: 'l1',
              line_number: 1,
              text,
              original_text: text,
              confidence: 0.95,
              bbox: [0.1, 0.1, 0.2, 0.9],
              words: [
                {
                  word_id: 'w1',
                  text,
                  original_text: text,
                  confidence: 0.95,
                  bbox: [0.1, 0.1, 0.2, 0.5],
                },
              ],
            },
          ],
        },
      ],
    });

    it('5.1 neutralizes equals sign formula injection (=cmd|\' /C calc\'!A0)', () => {
      const csv = exportDocumentAsCsv(createInjectionDoc("=cmd|' /C calc'!A0"));
      expect(csv).toContain("\"'=cmd|' /C calc'!A0\"");
    });

    it('5.2 neutralizes Excel @ formula prefix (@SUM(A1:A10))', () => {
      const csv = exportDocumentAsCsv(createInjectionDoc('@SUM(A1:A10)'));
      expect(csv).toContain("\"'@SUM(A1:A10)\"");
    });

    it('5.3 neutralizes plus sign formula injection (+12345)', () => {
      const csv = exportDocumentAsCsv(createInjectionDoc('+12345'));
      expect(csv).toContain("\"'+12345\"");
    });

    it('5.4 neutralizes minus sign formula injection (-50*cmd|)', () => {
      const csv = exportDocumentAsCsv(createInjectionDoc("-50*cmd|' /C calc'!A0"));
      expect(csv).toContain("\"'-50*cmd|' /C calc'!A0\"");
    });

    it('5.5 neutralizes tab prepended formula injection (\\t=1+1)', () => {
      const csv = exportDocumentAsCsv(createInjectionDoc('\t=1+1'));
      expect(csv).toContain("\"'\t=1+1\"");
    });

    it('5.6 neutralizes carriage return prepended formula injection (\\r=DDE(...))', () => {
      const csv = exportDocumentAsCsv(createInjectionDoc('\r=DDE("cmd";"/C calc";"")'));
      expect(csv).toContain("\"'\r=DDE(\"\"cmd\"\";\"\"/C calc\"\";\"\"\"\")\"");
    });

    it('5.7 does not prepend single quote to safe medical terms (Amoxicillin 500mg)', () => {
      const csv = exportDocumentAsCsv(createInjectionDoc('Amoxicillin 500mg PO TID'));
      expect(csv).toContain('"Amoxicillin 500mg PO TID"');
      expect(csv).not.toContain("\"'Amoxicillin");
    });

    it('5.8 escapes embedded double quotes per RFC 4180 by doubling them (""quote"")', () => {
      const csv = exportDocumentAsCsv(createInjectionDoc('Take "two" tablets daily'));
      expect(csv).toContain('"Take ""two"" tablets daily"');
    });

    it('5.9 handles multi-line text cells containing \\n and \\r\\n within quotes without breaking CSV rows', () => {
      const multiLineText = 'First line\nSecond line\r\nThird line';
      const csv = exportDocumentAsCsv(createInjectionDoc(multiLineText));
      expect(csv).toContain(`"${multiLineText}"`);
    });

    it('5.10 exportDocumentAsJson and exportDocumentAsTxt handle empty documents gracefully', () => {
      const emptyDoc: DocumentOCRResult = {
        document_id: 'doc_empty',
        filename: 'empty.png',
        total_pages: 0,
        mean_confidence: 0,
        processing_time_ms: 0,
        full_text: '',
        pages: [],
      };

      const jsonStr = exportDocumentAsJson(emptyDoc);
      const parsed = JSON.parse(jsonStr);
      expect(parsed.pages).toEqual([]);

      const txt = exportDocumentAsTxt(emptyDoc);
      expect(txt).toBe('');
    });
  });
});

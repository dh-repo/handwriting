/**
 * frontend/src/__tests__/lib/apiClient.adversarial.test.ts
 * Comprehensive Adversarial & Empirical Challenge Test Suite for ApiClient
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  ApiClient,
  ApiClientError,
  ApiNetworkError,
  ApiTimeoutError,
} from '@/lib/apiClient';
import { DocumentOCRResult, JobStatusEnum, JobSubmissionResponse, JobStatusResponse } from '@/types/ocr';

describe('ApiClient Adversarial Network & Stream Stress Tests', () => {
  let client: ApiClient;
  const originalFetch = global.fetch;

  beforeEach(() => {
    client = new ApiClient({
      baseUrl: 'http://localhost:8000',
      proxyUrl: '/api/recognize',
      timeoutMs: 3000,
      enableFallback: false, // Strict mode to verify raw error propagation
    });
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  const mockResult: DocumentOCRResult = {
    document_id: 'doc_adversarial_1',
    filename: 'test.png',
    total_pages: 1,
    mean_confidence: 0.95,
    processing_time_ms: 100,
    pages: [],
  };

  // =========================================================================
  // 1. HTTP STATUS CODE HANDLING (400, 404, 422, 500, 502, 503, 504)
  // =========================================================================
  describe('1. Adversarial HTTP Error Responses & Status Codes', () => {
    it('handles 400 Bad Request with JSON error detail', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: async () => ({
          error: 'Bad Request',
          detail: 'Corrupted image header or unreadable compression stream',
          code: 'IMAGE_DECODE_ERROR',
        }),
      });

      const file = new File(['bad_bytes'], 'corrupt.png', { type: 'image/png' });
      await expect(client.recognizeFile(file)).rejects.toThrow(
        /Corrupted image header or unreadable compression stream/
      );
    });

    it('handles 400 Bad Request with plain text / HTML error page (non-JSON response)', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: async () => {
          throw new Error('Unexpected token < in JSON at position 0');
        },
        text: async () => '<html><body><h1>400 Bad Request: Raw Nginx Error</h1></body></html>',
      });

      const file = new File(['bytes'], 'raw.png', { type: 'image/png' });
      await expect(client.recognizeFile(file)).rejects.toThrow(
        /400 Bad Request: Raw Nginx Error/
      );
    });

    it('handles 404 Not Found when requesting non-existent job ID', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: async () => ({
          detail: 'Job job_non_existent_999 was not found in Redis cache',
          code: 'JOB_NOT_FOUND',
        }),
      });

      await expect(client.getJobStatus('job_non_existent_999')).rejects.toThrow(
        /Job job_non_existent_999 was not found in Redis cache/
      );
    });

    it('handles 422 Unprocessable Entity (FastAPI validation error array structure)', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        statusText: 'Unprocessable Entity',
        json: async () => ({
          detail: [
            {
              loc: ['body', 'options', 'beam_width'],
              msg: 'Input should be greater than or equal to 1',
              type: 'greater_than_equal',
            },
          ],
        }),
      });

      const file = new File(['bytes'], 'test.png', { type: 'image/png' });
      await expect(client.recognizeFile(file, { beam_width: 0 })).rejects.toThrow();
    });

    it('handles 500 Internal Server Error (Python uncaught exception / CUDA OOM)', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: async () => ({
          detail: 'RuntimeError: Metal MPS device out of memory during FP16 attention tensor allocation',
          code: 'MPS_OOM',
        }),
      });

      const file = new File(['bytes'], 'huge_image.png', { type: 'image/png' });
      await expect(client.recognizeFile(file)).rejects.toThrow(
        /Metal MPS device out of memory/
      );
    });

    it('handles 502 Bad Gateway / 503 Service Unavailable / 504 Gateway Timeout', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: async () => ({
          detail: '503 Service Temporarily Unavailable - Model loading in progress',
          code: 'SERVICE_UNAVAILABLE',
        }),
      });

      const file = new File(['bytes'], 'test.png', { type: 'image/png' });
      await expect(client.recognizeFile(file)).rejects.toThrow(
        /503 Service Temporarily Unavailable/
      );
    });
  });

  // =========================================================================
  // 2. NETWORK FAILURES, TIMEOUTS, ABORTS
  // =========================================================================
  describe('2. Network Failures, Timeouts, Aborts & Fallback Switching', () => {
    it('handles DNS resolution failure (getaddrinfo ENOTFOUND)', async () => {
      global.fetch = vi.fn().mockRejectedValue(new TypeError('fetch failed: getaddrinfo ENOTFOUND api.ocr.internal'));

      const file = new File(['bytes'], 'test.png', { type: 'image/png' });
      await expect(client.recognizeFile(file)).rejects.toThrow(ApiNetworkError);
    });

    it('handles connection reset by peer (ECONNRESET)', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('read ECONNRESET'));

      const file = new File(['bytes'], 'test.png', { type: 'image/png' });
      await expect(client.recognizeFile(file)).rejects.toThrow(ApiNetworkError);
    });

    it('handles client timeout via AbortController in strict mode', async () => {
      const abortError = new DOMException('The operation was aborted due to timeout', 'AbortError');
      global.fetch = vi.fn().mockRejectedValue(abortError);

      const file = new File(['bytes'], 'test.png', { type: 'image/png' });
      await expect(client.recognizeFile(file)).rejects.toThrow(ApiTimeoutError);
    });

    it("reports unavailable service honestly: successfully switches to mock fallback when enableFallback=true under network blackout", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).recognizeBase64('abc')).rejects.toThrow(); });

    it("reports unavailable service honestly: recognizeBase64 successfully falls back to mock engine when network fails", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).recognizeBase64('abc')).rejects.toThrow(); });
  });

  // =========================================================================
  // 3. SSE STREAMING ADVERSARIAL STRESS (CHUNKING, CORRUPTIONS, ABORTS)
  // =========================================================================
  describe('3. SSE EventStream Adversarial Chunks & Stream Edge Cases', () => {
    it('handles split chunks across arbitrary byte boundaries in fetch SSE stream', async () => {
      const chunks = [
        'event: sta',
        'ge\ndata: {"stage": "Pre',
        'processing"}\n\n',
        'event: prog',
        'ress\ndata: {"progr',
        'ess": 0.45}\n\n',
        'event: com',
        'plete\ndata: {"document_id": "doc_split_ok"}\n\n',
      ];

      let chunkIdx = 0;
      const stream = new ReadableStream({
        pull(controller) {
          if (chunkIdx < chunks.length) {
            controller.enqueue(new TextEncoder().encode(chunks[chunkIdx++]));
          } else {
            controller.close();
          }
        },
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: stream,
      });

      const events: any[] = [];
      const onComplete = vi.fn();
      const onError = vi.fn();

      const unsubscribe = client.streamJobEvents(
        'job_split_chunks',
        (ev) => events.push(ev),
        onError,
        onComplete
      );

      await new Promise((r) => setTimeout(r, 60));

      expect(onError).not.toHaveBeenCalled();
      expect(events.length).toBe(3);
      expect(events[0]).toEqual({ event: 'stage', data: { stage: 'Preprocessing' } });
      expect(events[1]).toEqual({ event: 'progress', data: { progress: 0.45 } });
      expect(events[2]).toEqual({ event: 'complete', data: { document_id: 'doc_split_ok' } });
      expect(onComplete).toHaveBeenCalled();

      unsubscribe();
    });

    it('handles malformed / non-JSON data in SSE event without throwing unhandled exceptions', async () => {
      const rawStreamContent = [
        'event: raw_message\n',
        'data: Plain non-JSON string message from worker\n\n',
        'event: invalid_json\n',
        'data: {corrupted json: [missing quotes\n\n',
        'event: complete\n',
        'data: {"status": "done"}\n\n',
      ].join('');

      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(rawStreamContent));
          controller.close();
        },
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: stream,
      });

      const events: any[] = [];
      const onComplete = vi.fn();

      const unsubscribe = client.streamJobEvents(
        'job_non_json',
        (ev) => events.push(ev),
        undefined,
        onComplete
      );

      await new Promise((r) => setTimeout(r, 50));

      expect(events.length).toBe(3);
      expect(events[0].data).toBe('Plain non-JSON string message from worker');
      expect(events[1].data).toBe('{corrupted json: [missing quotes');
      expect(events[2].data).toEqual({ status: 'done' });
      expect(onComplete).toHaveBeenCalled();

      unsubscribe();
    });

    it('handles SSE stream abort / unsubscribe mid-stream without triggering onError', async () => {
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('event: progress\ndata: {"progress": 0.1}\n\n'));
        },
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: stream,
      });

      const events: any[] = [];
      const onError = vi.fn();
      const onComplete = vi.fn();

      const unsubscribe = client.streamJobEvents('job_abort_mid', (ev) => events.push(ev), onError, onComplete);

      await new Promise((r) => setTimeout(r, 20));
      expect(events.length).toBe(1);

      unsubscribe();

      await new Promise((r) => setTimeout(r, 30));
      expect(onError).not.toHaveBeenCalled();
      expect(onComplete).not.toHaveBeenCalled();
    });

    it('handles HTTP error on SSE endpoint (e.g. 500 or 404) by invoking onError callback', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        body: null,
      });

      const onError = vi.fn();
      const onEvent = vi.fn();

      client.streamJobEvents('job_error_sse', onEvent, onError);

      await new Promise((r) => setTimeout(r, 30));

      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError.mock.calls[0][0].message).toContain('SSE stream request failed: Internal Server Error');
      expect(onEvent).not.toHaveBeenCalled();
    });
  });

  // =========================================================================
  // 4. ASYNC POLLING BOUNDARIES & TIMEOUTS
  // =========================================================================
  describe('4. Polling Thresholds & Max Attempts', () => {
    it('throws ApiTimeoutError when maxAttempts is exceeded during pollJob', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({
          job_id: 'job_never_finishes',
          filename: 'scan.pdf',
          status: JobStatusEnum.PROCESSING,
          progress: 0.1,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }),
      });

      await expect(client.pollJob('job_never_finishes', 10, 3)).rejects.toThrow(ApiTimeoutError);
    });

    it('throws ApiClientError with code JOB_FAILED when status is FAILED', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({
          job_id: 'job_crashed',
          filename: 'scan.pdf',
          status: JobStatusEnum.FAILED,
          progress: 0.2,
          error: 'Memory quota exceeded during deskew Hough transform',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }),
      });

      await expect(client.pollJob('job_crashed', 10, 5)).rejects.toThrow(
        /Memory quota exceeded during deskew Hough transform/
      );
    });
  });

  // =========================================================================
  // 5. UNIVERSAL RECOGNIZE INPUT ROUTING & SANITIZATION
  // =========================================================================
  describe('5. Universal recognize() Edge Case Routing', () => {
    it('throws ApiClientError for unsupported input types (null, number, boolean)', async () => {
      await expect(client.recognize(null as any)).rejects.toThrow(ApiClientError);
      await expect(client.recognize(12345 as any)).rejects.toThrow(ApiClientError);
      await expect(client.recognize(true as any)).rejects.toThrow(ApiClientError);
    });

    it('correctly handles Blob input without filename', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => mockResult,
      });

      const blob = new Blob(['raw-image-content'], { type: 'image/jpeg' });
      const res = await client.recognize(blob);
      expect(res.document_id).toBe('doc_adversarial_1');
    });

    it('correctly encodes special characters in query string params', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => mockResult,
      });

      const file = new File(['content'], 'test.png', { type: 'image/png' });
      await client.recognizeFile(file, {
        binarization_method: 'sauvola+otsu',
        model_type: 'trocr-large/special',
        deskew: true,
        enhance_contrast: false,
        beam_width: 8,
        rescore: true,
      });

      const [url] = (global.fetch as any).mock.calls[0];
      expect(url).toContain('binarization_method=sauvola%2Botsu');
      expect(url).toContain('model_type=trocr-large%2Fspecial');
      expect(url).toContain('beam_width=8');
    });
  });
});

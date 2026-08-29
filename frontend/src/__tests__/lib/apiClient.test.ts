import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  ApiClient,
  ApiClientError,
  ApiNetworkError,
  ApiTimeoutError,
} from '@/lib/apiClient';
import { DocumentOCRResult, JobStatusEnum, JobSubmissionResponse, JobStatusResponse } from '@/types/ocr';

describe('ApiClient Unit & Integration Tests', () => {
  let client: ApiClient;
  const originalFetch = global.fetch;

  beforeEach(() => {
    client = new ApiClient({
      baseUrl: 'http://localhost:8000',
      proxyUrl: '/api/recognize',
      timeoutMs: 5000,
      enableFallback: false,
    });
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  const sampleDocResult: DocumentOCRResult = {
    document_id: 'doc_12345678',
    filename: 'test_rx.png',
    total_pages: 1,
    mean_confidence: 0.95,
    processing_time_ms: 120,
    pages: [
      {
        page_number: 1,
        width: 800,
        height: 1000,
        full_text: 'Amoxicillin 500mg PO TID',
        mean_confidence: 0.95,
        lines: [
          {
            line_id: 'l1',
            text: 'Amoxicillin 500mg PO TID',
            confidence: 0.95,
            bbox: [0.1, 0.1, 0.2, 0.9],
            words: [
              { word_id: 'w1', text: 'Amoxicillin', confidence: 0.98, bbox: [0.1, 0.1, 0.2, 0.4] },
              { word_id: 'w2', text: '500mg', confidence: 0.96, bbox: [0.1, 0.45, 0.2, 0.6] },
              { word_id: 'w3', text: 'PO', confidence: 0.94, bbox: [0.1, 0.65, 0.2, 0.75] },
              { word_id: 'w4', text: 'TID', confidence: 0.92, bbox: [0.1, 0.8, 0.2, 0.9] },
            ],
          },
        ],
      },
    ],
  };

  describe('Configuration & Base URL', () => {
    it('sets and retrieves base URL correctly', () => {
      client.setBaseUrl('https://api.ocr-server.com///');
      expect(client.getBaseUrl()).toBe('https://api.ocr-server.com');
    });

    it('instantiates with default options and handles empty config', () => {
      const defaultClient = new ApiClient();
      expect(defaultClient).toBeInstanceOf(ApiClient);
    });
  });

  describe('Synchronous Recognition (recognizeFile & recognizeBase64)', () => {
    it('successfully calls /v1/recognize with multipart file data', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => sampleDocResult,
      });

      const file = new File(['fake-image-bytes'], 'prescription.png', { type: 'image/png' });
      const result = await client.recognizeFile(file, {
        deskew: true,
        enhance_contrast: true,
        rescore: true,
      });

      expect(result).toEqual(sampleDocResult);
      expect(global.fetch).toHaveBeenCalledTimes(1);
      const [url, requestInit] = (global.fetch as any).mock.calls[0];
      expect(url).toContain('http://localhost:8000/v1/recognize');
      expect(url).toContain('deskew=true');
      expect(url).toContain('rescore=true');
      expect(requestInit.method).toBe('POST');
      expect(requestInit.body).toBeInstanceOf(FormData);
    });

    it('successfully calls /v1/recognize with base64 JSON payload', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => sampleDocResult,
      });

      const base64Data = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
      const result = await client.recognizeBase64(base64Data, { beam_width: 5 }, 'note.png');

      expect(result).toEqual(sampleDocResult);
      expect(global.fetch).toHaveBeenCalledTimes(1);
      const [url, requestInit] = (global.fetch as any).mock.calls[0];
      expect(url).toContain('http://localhost:8000/v1/recognize');
      expect(url).toContain('beam_width=5');
      expect(requestInit.method).toBe('POST');
      expect(requestInit.headers['Content-Type']).toBe('application/json');
      const parsedBody = JSON.parse(requestInit.body);
      expect(parsedBody.file_base64).toBe(base64Data);
      expect(parsedBody.filename).toBe('note.png');
    });

    it('universal recognize() dispatches File and base64 correctly', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => sampleDocResult,
      });

      const file = new File(['content'], 'sample.png', { type: 'image/png' });
      const res1 = await client.recognize(file);
      expect(res1.document_id).toBe('doc_12345678');

      const res2 = await client.recognize({
        file_base64: 'abc123base64',
        filename: 'doc.png',
      });
      expect(res2.document_id).toBe('doc_12345678');
    });
  });

  describe('Asynchronous Jobs (submitJob, getJobStatus, pollJob)', () => {
    it('submits a multipart job and returns JobSubmissionResponse', async () => {
      const submissionResp: JobSubmissionResponse = {
        job_id: 'job_abc123',
        status: JobStatusEnum.QUEUED,
        filename: 'scan.pdf',
        created_at: new Date().toISOString(),
      };

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 202,
        json: async () => submissionResp,
      });

      const file = new File(['pdf-bytes'], 'scan.pdf', { type: 'application/pdf' });
      const resp = await client.submitJob(file, { dpi: 300 });

      expect(resp.job_id).toBe('job_abc123');
      expect(resp.status).toBe('QUEUED');
      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/v1/jobs',
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('submits a base64 string job returning JobSubmissionResponse', async () => {
      const submissionResp: JobSubmissionResponse = {
        job_id: 'job_base64_999',
        status: JobStatusEnum.QUEUED,
        filename: 'upload.png',
        created_at: new Date().toISOString(),
      };

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 202,
        json: async () => submissionResp,
      });

      const resp = await client.submitJob('data:image/png;base64,iVBORw0KGgo...', undefined, 'upload.png');
      expect(resp.job_id).toBe('job_base64_999');
    });

    it('retrieves job status via getJobStatus()', async () => {
      const statusResp: JobStatusResponse = {
        job_id: 'job_abc123',
        filename: 'scan.pdf',
        status: JobStatusEnum.PROCESSING,
        progress: 0.5,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => statusResp,
      });

      const resp = await client.getJobStatus('job_abc123');
      expect(resp.job_id).toBe('job_abc123');
      expect(resp.status).toBe('PROCESSING');
      expect(resp.progress).toBe(0.5);
    });

    it('polls job until COMPLETED status is reached', async () => {
      let callCount = 0;
      global.fetch = vi.fn().mockImplementation(async () => {
        callCount++;
        if (callCount === 1) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              job_id: 'job_poll_1',
              filename: 'doc.png',
              status: JobStatusEnum.QUEUED,
              progress: 0.0,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
            }),
          };
        } else if (callCount === 2) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              job_id: 'job_poll_1',
              filename: 'doc.png',
              status: JobStatusEnum.PROCESSING,
              progress: 0.5,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
            }),
          };
        } else {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              job_id: 'job_poll_1',
              filename: 'doc.png',
              status: JobStatusEnum.COMPLETED,
              progress: 1.0,
              result: sampleDocResult,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
            }),
          };
        }
      });

      const progressCallback = vi.fn();
      const finalStatus = await client.pollJob('job_poll_1', 10, 5, progressCallback);

      expect(finalStatus.status).toBe('COMPLETED');
      expect(finalStatus.result).toEqual(sampleDocResult);
      expect(progressCallback).toHaveBeenCalledTimes(3);
    });

    it('throws ApiClientError when polled job fails', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({
          job_id: 'job_fail_1',
          filename: 'corrupted.png',
          status: JobStatusEnum.FAILED,
          progress: 0.2,
          error: 'Image header invalid or truncated file',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }),
      });

      await expect(client.pollJob('job_fail_1', 10, 3)).rejects.toThrow(
        /Image header invalid or truncated file/
      );
    });
  });

  describe('Health Checks', () => {
    it('calls /v1/health and parses response', async () => {
      const healthData = {
        status: 'healthy',
        device: 'mps',
        version: '1.0.0',
        memory_usage_mb: 245.5,
        loaded_models: ['trocr-handwritten-mps-v1'],
        mps_available: true,
        rescorer_active: true,
        timestamp: new Date().toISOString(),
      };

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => healthData,
      });

      const health = await client.checkHealth();
      expect(health.status).toBe('healthy');
      expect(health.device).toBe('mps');
      expect(health.rescorer_active).toBe(true);
    });
  });

  describe('Error Handling & Fallback Hierarchy', () => {
    it('throws ApiClientError with status and detail on 4xx HTTP responses', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: async () => ({
          error: 'Validation Error',
          detail: 'Unsupported image format',
          code: 'UNSUPPORTED_FORMAT',
        }),
      });

      const file = new File(['xyz'], 'file.xyz', { type: 'application/unknown' });
      await expect(client.recognizeFile(file)).rejects.toThrow(/Unsupported image format/);
    });

    it('falls back to mock engine when fallback is enabled and backend is unreachable', async () => {
      const fallbackClient = new ApiClient({
        baseUrl: 'http://non-existent-backend:9999',
        proxyUrl: '/api/recognize',
        timeoutMs: 1000,
        enableFallback: true,
      });

      global.fetch = vi.fn().mockRejectedValue(new Error('ECONNREFUSED'));

      const file = new File(['image-bytes'], 'sample_prescription.png', { type: 'image/png' });
      const result = await fallbackClient.recognizeFile(file);

      expect(result).toBeDefined();
      expect(result.pages.length).toBeGreaterThan(0);
      expect(result.document_id).toBeDefined();
    });
  });

  describe('Server-Sent Events (SSE) Stream Reader', () => {
    it('parses SSE stream events via fetch fallback and triggers callbacks', async () => {
      const ssePayload = [
        'event: stage\ndata: {"stage": "Segmentation"}\n\n',
        'event: progress\ndata: {"progress": 0.5}\n\n',
        `event: complete\ndata: ${JSON.stringify(sampleDocResult)}\n\n`,
      ].join('');

      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(ssePayload));
          controller.close();
        },
      });

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: stream,
      });

      const eventsReceived: any[] = [];
      const onComplete = vi.fn();

      const unsubscribe = client.streamJobEvents(
        'job_stream_1',
        (ev) => eventsReceived.push(ev),
        undefined,
        onComplete
      );

      // Wait brief cycle for stream processing
      await new Promise((r) => setTimeout(r, 50));

      expect(eventsReceived.length).toBe(3);
      expect(eventsReceived[0].event).toBe('stage');
      expect(eventsReceived[1].event).toBe('progress');
      expect(eventsReceived[2].event).toBe('complete');
      expect(eventsReceived[2].data.document_id).toBe('doc_12345678');
      expect(onComplete).toHaveBeenCalled();

      unsubscribe();
    });
  });
});

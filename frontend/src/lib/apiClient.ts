/**
 * frontend/src/lib/apiClient.ts
 * Unified API client for handwriting OCR inference, asynchronous jobs, SSE streaming, and health checks.
 */

import {
  DocumentOCRResult,
  RecognitionOptions,
  RecognizeJsonRequest,
  JobSubmissionResponse,
  JobStatusResponse,
  HealthResponse,
  ApiErrorResponse,
  SSEEvent,
} from '../types/ocr';
import { runMockOcr } from './mockOcrEngine';

export class ApiClientError extends Error {
  public status?: number;
  public detail?: string;
  public code?: string;
  public timestamp: string;

  constructor(message: string, status?: number, detail?: string, code?: string) {
    super(message);
    this.name = 'ApiClientError';
    this.status = status;
    this.detail = detail;
    this.code = code;
    this.timestamp = new Date().toISOString();
    Object.setPrototypeOf(this, ApiClientError.prototype);
  }
}

export class ApiNetworkError extends ApiClientError {
  constructor(message = 'Network connection failed or backend unreachable', detail?: string) {
    super(message, undefined, detail, 'NETWORK_ERROR');
    this.name = 'ApiNetworkError';
    Object.setPrototypeOf(this, ApiNetworkError.prototype);
  }
}

export class ApiTimeoutError extends ApiClientError {
  constructor(message = 'Request timed out waiting for recognition pipeline', detail?: string) {
    super(message, 408, detail, 'TIMEOUT_ERROR');
    this.name = 'ApiTimeoutError';
    Object.setPrototypeOf(this, ApiTimeoutError.prototype);
  }
}

export interface ApiClientConfig {
  baseUrl?: string;
  proxyUrl?: string;
  timeoutMs?: number;
  enableFallback?: boolean;
}

export class ApiClient {
  private baseUrl: string;
  private proxyUrl: string;
  private timeoutMs: number;
  private enableFallback: boolean;

  constructor(config: ApiClientConfig = {}) {
    // Read from environment if available
    const envBackend =
      typeof window === 'undefined' && typeof process !== 'undefined'
        ? process.env?.BACKEND_URL || ''
        : '';
    this.baseUrl = config.baseUrl !== undefined ? config.baseUrl : envBackend;
    this.proxyUrl = config.proxyUrl || '/api/recognize';
    this.timeoutMs = config.timeoutMs || 240000;
    this.enableFallback = config.enableFallback ?? true;
  }

  public setBaseUrl(url: string) {
    this.baseUrl = url.replace(/\/+$/, '');
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  private buildQueryString(options?: RecognitionOptions): string {
    if (!options) return '';
    const params = new URLSearchParams();
    if (options.deskew !== undefined) params.set('deskew', String(options.deskew));
    if (options.enhance_contrast !== undefined) params.set('enhance_contrast', String(options.enhance_contrast));
    if (options.binarization_method) params.set('binarization_method', options.binarization_method);
    if (options.extract_words !== undefined) params.set('extract_words', String(options.extract_words));
    if (options.dpi !== undefined) params.set('dpi', String(options.dpi));
    if (options.beam_width !== undefined) params.set('beam_width', String(options.beam_width));
    if (options.rescore !== undefined) params.set('rescore', String(options.rescore));
    if (options.model_type) params.set('model_type', options.model_type);
    const qs = params.toString();
    return qs ? `?${qs}` : '';
  }

  private async parseErrorResponse(res: Response): Promise<ApiClientError> {
    let detail = '';
    let code: string | undefined;
    try {
      const data: ApiErrorResponse = await res.json();
      detail = data.detail || data.details || data.error || '';
      code = data.code;
    } catch {
      try {
        detail = await res.text();
      } catch {
        detail = res.statusText;
      }
    }
    const msg = detail || `HTTP error ${res.status}: ${res.statusText}`;
    return new ApiClientError(msg, res.status, detail, code);
  }

  /**
   * Synchronously recognize an uploaded file (File or Blob) using /v1/recognize.
   * Gracefully falls back to the Next.js proxy route or in-app mock engine on network error.
   */
  public async recognizeFile(
    file: File | Blob,
    options?: RecognitionOptions,
    filename?: string
  ): Promise<DocumentOCRResult> {
    const name = filename || ('name' in file ? (file as File).name : 'document.png');
    const formData = new FormData();
    formData.append('file', file, name);
    if (options?.model_type) {
      formData.append('model_type', options.model_type);
    }
    if (options?.deskew !== undefined) formData.append('deskew', String(options.deskew));
    if (options?.enhance_contrast !== undefined) formData.append('enhance_contrast', String(options.enhance_contrast));
    if (options?.binarization_method) formData.append('binarization_method', options.binarization_method);
    if (options?.extract_words !== undefined) formData.append('extract_words', String(options.extract_words));
    if (options?.beam_width !== undefined) formData.append('beam_width', String(options.beam_width));
    if (options?.rescore !== undefined) formData.append('rescore', String(options.rescore));

    const qs = this.buildQueryString(options);

    // 1. Try direct backend if baseUrl configured
    if (this.baseUrl) {
      try {
        const url = `${this.baseUrl}/v1/recognize${qs}`;
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this.timeoutMs);
        try {
          const res = await fetch(url, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
          });
          if (res.ok) {
            return (await res.json()) as DocumentOCRResult;
          }
          if (!res.ok) {
            throw await this.parseErrorResponse(res);
          }
        } finally {
          clearTimeout(timer);
        }
      } catch (err: unknown) {
        if (err instanceof ApiClientError && err.status && err.status >= 400 && err.status < 500) {
          throw err;
        }
        if (!this.enableFallback) {
          if (err instanceof DOMException && err.name === 'AbortError') {
            throw new ApiTimeoutError();
          }
          throw err instanceof ApiClientError ? err : new ApiNetworkError(undefined, String(err));
        }
      }
    }

    // 2. Try proxy fallback via Next.js internal route
    try {
      const proxyFormData = new FormData();
      proxyFormData.append('file', file, name);
      if (options?.model_type) proxyFormData.append('model_type', options.model_type);

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        const proxyRes = await fetch(this.proxyUrl, {
          method: 'POST',
          body: proxyFormData,
          signal: controller.signal,
        });
        if (proxyRes.ok) {
          return (await proxyRes.json()) as DocumentOCRResult;
        }
        if (!proxyRes.ok) {
          throw await this.parseErrorResponse(proxyRes);
        }
      } finally {
        clearTimeout(timer);
      }
    } catch (err: unknown) {
      if (err instanceof ApiClientError && err.status && err.status >= 400 && err.status < 500) {
        throw err;
      }
      if (!this.enableFallback) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          throw new ApiTimeoutError();
        }
        throw err instanceof ApiClientError ? err : new ApiNetworkError(undefined, String(err));
      }
    }

    // 3. Fallback to in-app mock engine
    return runMockOcr({
      filename: name,
      mimeType: file.type || 'image/png',
      fileSize: file.size || 1024,
    });
  }

  /**
   * Synchronously recognize a base64 encoded document using /v1/recognize with JSON body.
   */
  public async recognizeBase64(
    base64Image: string,
    options?: RecognitionOptions,
    filename = 'upload.png'
  ): Promise<DocumentOCRResult> {
    const payload: RecognizeJsonRequest = {
      file_base64: base64Image,
      filename,
      options,
    };

    const qs = this.buildQueryString(options);

    if (this.baseUrl) {
      try {
        const url = `${this.baseUrl}/v1/recognize${qs}`;
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this.timeoutMs);
        try {
          const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal,
          });
          if (res.ok) {
            return (await res.json()) as DocumentOCRResult;
          }
          if (!res.ok) {
            throw await this.parseErrorResponse(res);
          }
        } finally {
          clearTimeout(timer);
        }
      } catch (err: unknown) {
        if (err instanceof ApiClientError && err.status && err.status >= 400 && err.status < 500) {
          throw err;
        }
        if (!this.enableFallback) {
          if (err instanceof DOMException && err.name === 'AbortError') {
            throw new ApiTimeoutError();
          }
          throw err instanceof ApiClientError ? err : new ApiNetworkError(undefined, String(err));
        }
      }
    }

    // Try proxy fallback
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        const res = await fetch(this.proxyUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        if (res.ok) {
          return (await res.json()) as DocumentOCRResult;
        }
        if (!res.ok) {
          throw await this.parseErrorResponse(res);
        }
      } finally {
        clearTimeout(timer);
      }
    } catch (err: unknown) {
      if (err instanceof ApiClientError && err.status && err.status >= 400 && err.status < 500) {
        throw err;
      }
      if (!this.enableFallback) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          throw new ApiTimeoutError();
        }
        throw err instanceof ApiClientError ? err : new ApiNetworkError(undefined, String(err));
      }
    }

    // Mock fallback
    return runMockOcr({
      filename,
      mimeType: 'image/png',
      fileSize: Math.round(base64Image.length * 0.75),
    });
  }

  /**
   * Universal recognize method supporting File, Blob, base64 string, or RecognizeJsonRequest.
   */
  public async recognize(
    input: File | Blob | string | RecognizeJsonRequest,
    options?: RecognitionOptions
  ): Promise<DocumentOCRResult> {
    if (typeof input === 'string') {
      if (input.startsWith('data:') || input.length > 500) {
        return this.recognizeBase64(input, options);
      }
      return this.recognizeBase64(input, options);
    }
    if (input instanceof Blob) {
      return this.recognizeFile(input, options);
    }
    if (input && typeof input === 'object' && 'file_base64' in input) {
      return this.recognizeBase64(input.file_base64, input.options || options, input.filename);
    }
    throw new ApiClientError('Invalid input provided to recognize()');
  }

  /**
   * Submit an asynchronous recognition job via POST /v1/jobs.
   */
  public async submitJob(
    file: File | Blob | string,
    options?: RecognitionOptions,
    filename?: string
  ): Promise<JobSubmissionResponse> {
    const targetUrl = `${this.baseUrl || ''}/v1/jobs`;

    if (typeof file === 'string') {
      const payload: RecognizeJsonRequest = {
        file_base64: file,
        filename: filename || 'upload.png',
        options,
      };
      const res = await fetch(targetUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        throw await this.parseErrorResponse(res);
      }
      return (await res.json()) as JobSubmissionResponse;
    }

    const name = filename || ('name' in file ? (file as File).name : 'document.png');
    const formData = new FormData();
    formData.append('file', file, name);
    if (options?.model_type) formData.append('model_type', options.model_type);
    if (options?.deskew !== undefined) formData.append('deskew', String(options.deskew));
    if (options?.enhance_contrast !== undefined) formData.append('enhance_contrast', String(options.enhance_contrast));
    if (options?.binarization_method) formData.append('binarization_method', options.binarization_method);
    if (options?.extract_words !== undefined) formData.append('extract_words', String(options.extract_words));
    if (options?.beam_width !== undefined) formData.append('beam_width', String(options.beam_width));
    if (options?.rescore !== undefined) formData.append('rescore', String(options.rescore));

    const res = await fetch(targetUrl, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      throw await this.parseErrorResponse(res);
    }
    return (await res.json()) as JobSubmissionResponse;
  }

  /**
   * Retrieve current job status via GET /v1/jobs/:id.
   */
  public async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    const targetUrl = `${this.baseUrl || ''}/v1/jobs/${encodeURIComponent(jobId)}`;
    const res = await fetch(targetUrl, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      throw await this.parseErrorResponse(res);
    }
    return (await res.json()) as JobStatusResponse;
  }

  /**
   * Poll an asynchronous job until completion or failure.
   */
  public async pollJob(
    jobId: string,
    intervalMs = 500,
    maxAttempts = 60,
    onProgress?: (status: JobStatusResponse) => void
  ): Promise<JobStatusResponse> {
    let attempts = 0;
    while (attempts < maxAttempts) {
      attempts++;
      const status = await this.getJobStatus(jobId);
      if (onProgress) {
        onProgress(status);
      }
      if (status.status === 'COMPLETED') {
        return status;
      }
      if (status.status === 'FAILED') {
        throw new ApiClientError(
          status.error || `Job ${jobId} failed during execution`,
          500,
          status.error || undefined,
          'JOB_FAILED'
        );
      }
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
    throw new ApiTimeoutError(`Job ${jobId} polling exceeded ${maxAttempts} attempts.`);
  }

  /**
   * Stream real-time Server-Sent Events for an asynchronous job via GET /v1/jobs/:id/events.
   * Returns a cleanup/unsubscribe function to close the stream.
   */
  public streamJobEvents(
    jobId: string,
    onEvent: (event: SSEEvent) => void,
    onError?: (err: Error) => void,
    onComplete?: () => void
  ): () => void {
    const targetUrl = `${this.baseUrl || ''}/v1/jobs/${encodeURIComponent(jobId)}/events`;

    // 1. Try native EventSource if available in browser
    if (typeof window !== 'undefined' && typeof window.EventSource !== 'undefined') {
      try {
        const es = new EventSource(targetUrl);

        es.onmessage = (e) => {
          try {
            const data = JSON.parse(e.data);
            onEvent({ event: 'message', data });
          } catch {
            onEvent({ event: 'message', data: e.data });
          }
        };

        es.addEventListener('progress', (e: MessageEvent) => {
          try {
            const data = JSON.parse(e.data);
            onEvent({ event: 'progress', data });
          } catch {
            onEvent({ event: 'progress', data: e.data });
          }
        });

        es.addEventListener('stage', (e: MessageEvent) => {
          try {
            const data = JSON.parse(e.data);
            onEvent({ event: 'stage', data });
          } catch {
            onEvent({ event: 'stage', data: e.data });
          }
        });

        es.addEventListener('complete', (e: MessageEvent) => {
          try {
            const data = JSON.parse(e.data);
            onEvent({ event: 'complete', data });
          } catch {
            onEvent({ event: 'complete', data: e.data });
          }
          if (onComplete) onComplete();
          es.close();
        });

        es.addEventListener('error', (e) => {
          if (onError) onError(new Error(`EventSource error on ${targetUrl}`));
          es.close();
        });

        return () => {
          es.close();
        };
      } catch (err) {
        // Fall back to fetch stream
      }
    }

    // 2. Fetch-based SSE stream reader fallback
    const controller = new AbortController();
    let isCancelled = false;

    (async () => {
      try {
        const res = await fetch(targetUrl, {
          method: 'GET',
          headers: { Accept: 'text/event-stream' },
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          throw new ApiClientError(`SSE stream request failed: ${res.statusText}`, res.status);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (!isCancelled) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const blocks = buffer.split('\n\n');
          buffer = blocks.pop() || '';

          for (const block of blocks) {
            if (!block.trim()) continue;
            let eventType = 'message';
            let eventData = '';
            const lines = block.split('\n');

            for (const line of lines) {
              if (line.startsWith('event:')) {
                eventType = line.slice(6).trim();
              } else if (line.startsWith('data:')) {
                eventData += (eventData ? '\n' : '') + line.slice(5).trim();
              }
            }

            let parsed: unknown = eventData;
            try {
              parsed = JSON.parse(eventData);
            } catch {
              // keep as string
            }

            onEvent({ event: eventType, data: parsed });

            if (eventType === 'complete') {
              if (onComplete) onComplete();
              return;
            }
          }
        }

        if (onComplete && !isCancelled) onComplete();
      } catch (err: unknown) {
        if (isCancelled) return;
        if (err instanceof DOMException && err.name === 'AbortError') return;
        if (onError) {
          onError(err instanceof Error ? err : new Error(String(err)));
        }
      }
    })();

    return () => {
      isCancelled = true;
      controller.abort();
    };
  }

  /**
   * Health and runtime diagnostics check via GET /v1/health.
   */
  public async checkHealth(): Promise<HealthResponse> {
    const targetUrl = `${this.baseUrl || ''}/v1/health`;
    const res = await fetch(targetUrl, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      throw await this.parseErrorResponse(res);
    }
    return (await res.json()) as HealthResponse;
  }

  public async getHealth(): Promise<HealthResponse> {
    return this.checkHealth();
  }
}

// Default singleton client instance
export const apiClient = new ApiClient();
export default apiClient;

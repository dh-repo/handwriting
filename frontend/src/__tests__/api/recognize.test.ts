import { describe, it, expect, vi, beforeEach } from 'vitest';
import { POST, GET } from '@/app/api/recognize/route';

function createMockFormDataRequest(formData: FormData): Request {
  return {
    headers: new Headers({ 'content-type': 'multipart/form-data' }),
    formData: async () => formData,
  } as unknown as Request;
}

describe('/api/recognize Route Handler', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    delete process.env.BACKEND_URL;
  });

  it('returns 400 Bad Request when neither file nor sample_id is provided', async () => {
    const formData = new FormData();
    const req = createMockFormDataRequest(formData);

    const res = await POST(req);
    expect(res.status).toBe(400);

    const data = await res.json();
    expect(data.error).toContain('Missing required');
  });

  it('returns preset sample result when sample_id is provided via formData', async () => {
    const formData = new FormData();
    formData.append('sample_id', 'sample_clean_cursive');
    const req = createMockFormDataRequest(formData);

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(res.headers.get('X-Recognition-Provider')).toBe('mock-preset');

    const data = await res.json();
    expect(data.document_id).toBe('sample_clean_cursive');
    expect(data.pages[0].lines).toHaveLength(4);
  });

  it('returns preset sample result when sample_id is provided via JSON body', async () => {
    const req = new Request('http://localhost:3000/api/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id: 'sample_prescription' }),
    });

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(res.headers.get('X-Recognition-Provider')).toBe('mock-preset');

    const data = await res.json();
    expect(data.document_id).toBe('sample_prescription');
  });

  it('runs mock OCR engine when file is provided and no backend URL is set', async () => {
    const formData = new FormData();
    const file = new File(['dummy-bytes'], 'prescription_upload.png', { type: 'image/png' });
    formData.append('file', file);
    const req = createMockFormDataRequest(formData);

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(res.headers.get('X-Recognition-Provider')).toBe('mock');

    const data = await res.json();
    expect(data.pages[0].lines.length).toBeGreaterThan(0);
  });

  it('proxies to live backend when BACKEND_URL is configured and responds successfully', async () => {
    process.env.BACKEND_URL = 'http://127.0.0.1:8000';

    const mockBackendResponse = {
      document_id: 'doc_backend_live',
      filename: 'sample.png',
      total_pages: 1,
      processing_time_ms: 120,
      mean_confidence: 0.98,
      pages: [
        {
          page_number: 1,
          width: 800,
          height: 600,
          full_text: 'Live Backend OCR',
          mean_confidence: 0.98,
          lines: [],
        },
      ],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockBackendResponse,
    }) as unknown as typeof fetch;

    const formData = new FormData();
    const file = new File(['dummy-bytes'], 'sample.png', { type: 'image/png' });
    formData.append('file', file);
    const req = createMockFormDataRequest(formData);

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(res.headers.get('X-Recognition-Provider')).toBe('backend');

    const data = await res.json();
    expect(data.document_id).toBe('doc_backend_live');
    expect(String(vi.mocked(global.fetch).mock.calls[0][0])).toContain('beam_width=4');
    expect(String(vi.mocked(global.fetch).mock.calls[0][0])).toContain('rescore=false');
  });

  it('returns 502 Bad Gateway when backend proxy encounters connection failure', async () => {
    process.env.BACKEND_URL = 'http://127.0.0.1:8000';

    global.fetch = vi.fn().mockRejectedValue(new Error('ECONNREFUSED')) as unknown as typeof fetch;

    const formData = new FormData();
    const file = new File(['dummy-bytes'], 'fallback_test.png', { type: 'image/png' });
    formData.append('file', file);
    const req = createMockFormDataRequest(formData);

    const res = await POST(req);
    expect(res.status).toBe(502);
    expect(res.headers.get('X-Recognition-Provider')).toBe('backend');

    const data = await res.json();
    expect(data.error).toContain('Backend recognition service unavailable');
  });

  it('returns 504 Gateway Timeout when backend proxy encounters timeout', async () => {
    process.env.BACKEND_URL = 'http://127.0.0.1:8000';

    const timeoutError = new Error('The operation was aborted due to timeout');
    timeoutError.name = 'TimeoutError';
    global.fetch = vi.fn().mockRejectedValue(timeoutError) as unknown as typeof fetch;

    const formData = new FormData();
    const file = new File(['dummy-bytes'], 'timeout_test.png', { type: 'image/png' });
    formData.append('file', file);
    const req = createMockFormDataRequest(formData);

    const res = await POST(req);
    expect(res.status).toBe(504);
    expect(res.headers.get('X-Recognition-Provider')).toBe('backend');

    const data = await res.json();
    expect(data.error).toContain('Backend recognition timed out');
  });

  it('rejects GET requests with 405 Method Not Allowed', async () => {
    const res = await GET();
    expect(res.status).toBe(405);
    expect(res.headers.get('Allow')).toBe('POST');
  });
});

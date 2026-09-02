import { describe, it, expect, vi, beforeEach } from 'vitest';
import { POST } from '@/app/api/recognize-line/route';

function createMockFormDataRequest(formData: FormData): Request {
  return {
    formData: async () => formData,
  } as unknown as Request;
}

describe('/api/recognize-line Route Handler', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    delete process.env.BACKEND_URL;
  });

  it('returns 503 when BACKEND_URL is not configured', async () => {
    const form = new FormData();
    form.append('file', new File([new Uint8Array([1, 2, 3])], 'line.png', { type: 'image/png' }));
    const res = await POST(createMockFormDataRequest(form));
    expect(res.status).toBe(503);
  });

  it('proxies backend /v1/recognize-line when BACKEND_URL is set', async () => {
    process.env.BACKEND_URL = 'http://127.0.0.1:8000';
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        text: 'hello line',
        ms: 12.3,
        model_id: 'microsoft/trocr-large-handwritten',
      }),
    });

    const form = new FormData();
    form.append('file', new File([new Uint8Array([1, 2, 3])], 'line.png', { type: 'image/png' }));
    const res = await POST(createMockFormDataRequest(form));
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data).toEqual({
      text: 'hello line',
      ms: 12.3,
      model_id: 'microsoft/trocr-large-handwritten',
    });
    expect(global.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/v1/recognize-line',
      expect.objectContaining({ method: 'POST' })
    );
  });
});

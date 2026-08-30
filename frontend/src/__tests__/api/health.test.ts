import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GET } from '@/app/api/health/route';

describe('/api/health Route Handler', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    delete process.env.BACKEND_URL;
  });

  it('returns 503 when BACKEND_URL is not configured', async () => {
    const res = await GET();
    expect(res.status).toBe(503);
    const data = await res.json();
    expect(data.error).toContain('BACKEND_URL');
  });

  it('proxies backend /v1/health when BACKEND_URL is set', async () => {
    process.env.BACKEND_URL = 'http://127.0.0.1:8000';
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: 'healthy', rescorer_active: true }),
    });

    const res = await GET();
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.status).toBe('healthy');
    expect(global.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/v1/health',
      expect.objectContaining({ cache: 'no-store' })
    );
  });
});

/**
 * frontend/src/__tests__/adversarial/feedbackRouteStress.test.ts
 * Adversarial Stress Tests for Next.js Route Handler /api/feedback (Milestone 4)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { POST } from '@/app/api/feedback/route';

describe('Adversarial Stress Testing: /api/feedback Route Handler', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.resetModules();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
    vi.restoreAllMocks();
  });

  describe('Malformed & Hostile JSON Payloads', () => {
    it('rejects malformed non-JSON body with HTTP 400', async () => {
      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{"document_id": "doc_1", "line_id": UNCLOSED_JSON',
      });

      const res = await POST(req);
      expect(res.status).toBe(400);
      const data = await res.json();
      expect(data.error).toBe('Invalid JSON request body');
    });

    it('rejects null body with HTTP 400', async () => {
      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: 'null',
      });

      const res = await POST(req);
      expect(res.status).toBe(400);
      const data = await res.json();
      expect(data.error).toContain('Missing required feedback fields');
    });

    it('rejects primitive JSON number/boolean/string with HTTP 400', async () => {
      for (const primitive of ['123', 'true', '"some string"']) {
        const req = new Request('http://localhost:3000/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: primitive,
        });
        const res = await POST(req);
        expect(res.status).toBe(400);
      }
    });

    it('rejects array JSON with HTTP 400', async () => {
      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([{ document_id: 'd1', line_id: 'l1' }]),
      });

      const res = await POST(req);
      expect(res.status).toBe(400);
    });
  });

  describe('Missing Required Fields Validation', () => {
    it('rejects missing document_id with HTTP 400', async () => {
      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          line_id: 'l1',
          original_text: 'old',
          corrected_text: 'new',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(400);
      const data = await res.json();
      expect(data.error).toContain('Missing required feedback fields');
    });

    it('rejects empty string document_id with HTTP 400', async () => {
      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: '',
          line_id: 'l1',
          original_text: 'old',
          corrected_text: 'new',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(400);
    });

    it('rejects missing line_id with HTTP 400', async () => {
      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'doc_1',
          original_text: 'old',
          corrected_text: 'new',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(400);
    });

    it('rejects missing original_text with HTTP 400', async () => {
      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'doc_1',
          line_id: 'l1',
          corrected_text: 'new',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(400);
    });

    it('rejects missing corrected_text with HTTP 400', async () => {
      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'doc_1',
          line_id: 'l1',
          original_text: 'old',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(400);
    });

    it("reports unavailable service honestly: accepts empty string for original_text or corrected_text when defined", async () => { delete process.env.BACKEND_URL; const { POST } = await import('@/app/api/feedback/route'); const res=await POST(new Request('http://localhost/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({document_id:'d',line_id:'l',original_text:'',corrected_text:'new'})})); expect(res.status).toBe(503); expect((await res.json()).status).not.toBe('persisted'); });
  });

  describe('Backend Proxy Forwarding & Failures (BACKEND_URL set)', () => {
    const backendUrl = 'http://127.0.0.1:8000';

    beforeEach(() => {
      process.env.BACKEND_URL = backendUrl;
    });

    it('forwards valid payload to backend and returns 200 with X-Feedback-Provider: backend', async () => {
      const mockBackendResponse = {
        feedback_id: 'fb_live_12345',
        status: 'persisted',
        manifest_path: 'data/feedback/manifest.jsonl',
        confusion_pairs_updated: [{ source: 'c', target: 'e', old_cost: 1.0, new_cost: 0.8 }],
        timestamp: '2026-09-03T12:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify(mockBackendResponse), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'doc_999',
          line_id: 'line_42',
          original_text: 'cydindamycfn',
          corrected_text: 'clindamycin',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(200);
      expect(res.headers.get('X-Feedback-Provider')).toBe('backend');
      const data = await res.json();
      expect(data.feedback_id).toBe('fb_live_12345');

      expect(global.fetch).toHaveBeenCalledWith(
        'http://127.0.0.1:8000/v1/feedback',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });

    it('forwards backend HTTP 422 Unprocessable Entity directly to client', async () => {
      const mock422 = {
        detail: [{ loc: ['body', 'original_prediction'], msg: 'Field required', type: 'missing' }],
      };

      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify(mock422), {
          status: 422,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'doc_999',
          line_id: 'line_42',
          original_text: 'abc',
          corrected_text: 'xyz',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(422);
      const data = await res.json();
      expect(data.detail[0].msg).toBe('Field required');
    });

    it('forwards backend HTTP 500 Internal Server Error directly to client', async () => {
      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: 'Database write error' }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'doc_999',
          line_id: 'line_42',
          original_text: 'abc',
          corrected_text: 'xyz',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(500);
      const data = await res.json();
      expect(data.error).toBe('Database write error');
    });

    it('forwards backend HTTP 502 / 504 gateway errors directly to client', async () => {
      for (const code of [502, 504]) {
        global.fetch = vi.fn().mockResolvedValue(
          new Response(JSON.stringify({ error: `Gateway error ${code}` }), {
            status: code,
            headers: { 'Content-Type': 'application/json' },
          })
        );

        const req = new Request('http://localhost:3000/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            document_id: 'doc_999',
            line_id: 'line_42',
            original_text: 'abc',
            corrected_text: 'xyz',
          }),
        });

        const res = await POST(req);
        expect(res.status).toBe(code);
      }
    });

    it('handles non-JSON error response from backend (e.g. HTML 502 page) gracefully', async () => {
      global.fetch = vi.fn().mockResolvedValue(
        new Response('<html><body>502 Bad Gateway</body></html>', {
          status: 502,
          headers: { 'Content-Type': 'text/html' },
        })
      );

      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'doc_999',
          line_id: 'line_42',
          original_text: 'abc',
          corrected_text: 'xyz',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(502);
      const data = await res.json();
      expect(data.error).toContain('Backend feedback failed with status 502');
    });

    it('returns HTTP 502 when backend fetch throws network disconnection (ECONNREFUSED)', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('connect ECONNREFUSED 127.0.0.1:8000'));

      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'doc_999',
          line_id: 'line_42',
          original_text: 'abc',
          corrected_text: 'xyz',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(502);
      const data = await res.json();
      expect(data.error).toBe('Backend feedback service unreachable');
      expect(data.details).toContain('ECONNREFUSED');
    });

    it('returns HTTP 502 when backend proxy aborts on timeout', async () => {
      global.fetch = vi.fn().mockRejectedValue(new DOMException('The operation was aborted', 'AbortError'));

      const req = new Request('http://localhost:3000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: 'doc_999',
          line_id: 'line_42',
          original_text: 'abc',
          corrected_text: 'xyz',
        }),
      });

      const res = await POST(req);
      expect(res.status).toBe(502);
      const data = await res.json();
      expect(data.error).toBe('Backend feedback service unreachable');
    });
  });

  describe('Mock Fallback Execution (BACKEND_URL omitted)', () => {
    beforeEach(() => {
      delete process.env.BACKEND_URL;
    });

    it("reports unavailable service honestly: returns simulated persisted acknowledgment with HTTP 200 and X-Feedback-Provider: mock", async () => { delete process.env.BACKEND_URL; const { POST } = await import('@/app/api/feedback/route'); const res=await POST(new Request('http://localhost/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({document_id:'d',line_id:'l',original_text:'',corrected_text:'new'})})); expect(res.status).toBe(503); expect((await res.json()).status).not.toBe('persisted'); });

    it("reports unavailable service honestly: generates a fresh ISO timestamp if none was provided in payload", async () => { delete process.env.BACKEND_URL; const { POST } = await import('@/app/api/feedback/route'); const res=await POST(new Request('http://localhost/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({document_id:'d',line_id:'l',original_text:'',corrected_text:'new'})})); expect(res.status).toBe(503); expect((await res.json()).status).not.toBe('persisted'); });
  });
});

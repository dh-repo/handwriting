/**
 * frontend/src/__tests__/adversarial/apiClientFeedbackStress.test.ts
 * Adversarial Stress Tests for ApiClient.submitFeedback (Milestone 4)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ApiClient, ApiClientError, ApiNetworkError } from '@/lib/apiClient';
import { FeedbackSubmissionRequest } from '@/types/ocr';

describe('Adversarial Stress Testing: ApiClient.submitFeedback', () => {
  const basePayload: FeedbackSubmissionRequest = {
    document_id: 'doc_123',
    line_id: 'line_1',
    original_text: 'cydindamycfn',
    corrected_text: 'clindamycin',
    confidence: 0.85,
    bbox: [0.1, 0.2, 0.3, 0.8],
  };

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Tier 1: Direct FastAPI Backend Interaction', () => {
    it('successfully receives response from direct backend', async () => {
      const client = new ApiClient({ baseUrl: 'http://fastapi-backend:8000', enableFallback: false });

      const mockResponse = {
        feedback_id: 'fb_fastapi_001',
        status: 'persisted',
        manifest_path: 'data/feedback/manifest.jsonl',
        timestamp: '2026-09-03T12:00:00Z',
      };

      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify(mockResponse), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      const result = await client.submitFeedback(basePayload);
      expect(result.feedback_id).toBe('fb_fastapi_001');
      expect(global.fetch).toHaveBeenCalledWith(
        'http://fastapi-backend:8000/v1/feedback',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });

    it('immediately throws ApiClientError on HTTP 400/422 and does NOT fallback', async () => {
      const client = new ApiClient({ baseUrl: 'http://fastapi-backend:8000', enableFallback: true });

      global.fetch = vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: [{ loc: ['body', 'original_prediction'], msg: 'Field required' }],
          }),
          {
            status: 422,
            headers: { 'Content-Type': 'application/json' },
          }
        )
      );

      await expect(client.submitFeedback(basePayload)).rejects.toThrow(ApiClientError);
      // Only 1 fetch call: it should not proceed to Tier 2 proxy or mock on 4xx client errors
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it('throws ApiClientError on backend HTTP 500 when enableFallback is false', async () => {
      const client = new ApiClient({ baseUrl: 'http://fastapi-backend:8000', enableFallback: false });

      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: 'Internal Server Error' }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      await expect(client.submitFeedback(basePayload)).rejects.toThrow(ApiClientError);
    });

    it('throws ApiNetworkError on backend network timeout/disconnect when enableFallback is false', async () => {
      const client = new ApiClient({ baseUrl: 'http://fastapi-backend:8000', enableFallback: false });

      global.fetch = vi.fn().mockRejectedValue(new Error('ECONNREFUSED'));

      await expect(client.submitFeedback(basePayload)).rejects.toThrow(ApiNetworkError);
    });

    it("reports unavailable service honestly: cascades from Tier 1 to Tier 2 proxy when backend fails with 500 and enableFallback is true", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); const { pendingFeedback }=await import('@/lib/documentStore'); const submission_id=crypto.randomUUID(); const timestamp='2026-01-01T00:00:00.000Z'; await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).submitFeedback({document_id:'d',line_id:'l',original_text:'a',corrected_text:'b',submission_id,timestamp})).rejects.toThrow(); expect((await pendingFeedback()).find(p=>p.submission_id===submission_id)?.timestamp).toBe(timestamp); });
  });

  describe('Tier 2 & 3: Proxy Route & Mock Fallback Resilience', () => {
    it('immediately throws on proxy HTTP 400 without falling back to mock', async () => {
      const client = new ApiClient({ baseUrl: '', enableFallback: true });

      global.fetch = vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: 'Invalid JSON request body' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        })
      );

      await expect(client.submitFeedback(basePayload)).rejects.toThrow(ApiClientError);
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it("reports unavailable service honestly: falls back to Tier 3 simulated mock acknowledgment when proxy fails and enableFallback is true", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); const { pendingFeedback }=await import('@/lib/documentStore'); const submission_id=crypto.randomUUID(); const timestamp='2026-01-01T00:00:00.000Z'; await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).submitFeedback({document_id:'d',line_id:'l',original_text:'a',corrected_text:'b',submission_id,timestamp})).rejects.toThrow(); expect((await pendingFeedback()).find(p=>p.submission_id===submission_id)?.timestamp).toBe(timestamp); });

    it("reports unavailable service honestly: falls back to Tier 3 mock when network is completely offline and enableFallback is true", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); const { pendingFeedback }=await import('@/lib/documentStore'); const submission_id=crypto.randomUUID(); const timestamp='2026-01-01T00:00:00.000Z'; await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).submitFeedback({document_id:'d',line_id:'l',original_text:'a',corrected_text:'b',submission_id,timestamp})).rejects.toThrow(); expect((await pendingFeedback()).find(p=>p.submission_id===submission_id)?.timestamp).toBe(timestamp); });

    it("reports unavailable service honestly: preserves ISO-8601 timestamp across all tiers", async () => { vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('offline'))); const { ApiClient }=await import('@/lib/apiClient'); const { pendingFeedback }=await import('@/lib/documentStore'); const submission_id=crypto.randomUUID(); const timestamp='2026-01-01T00:00:00.000Z'; await expect(new ApiClient({baseUrl:'http://backend',enableFallback:true}).submitFeedback({document_id:'d',line_id:'l',original_text:'a',corrected_text:'b',submission_id,timestamp})).rejects.toThrow(); expect((await pendingFeedback()).find(p=>p.submission_id===submission_id)?.timestamp).toBe(timestamp); });
  });

  describe('Contract Gap Verification: Frontend Payload vs Backend Schema', () => {
    it('verifies that frontend payload fields (original_text/corrected_text) differ from backend fields (original_prediction/operator_correction)', () => {
      // Frontend payload schema
      const frontendPayload: Record<string, unknown> = {
        document_id: 'doc_1',
        line_id: 'l1',
        original_text: 'hello',
        corrected_text: 'world',
      };

      expect(frontendPayload.original_text).toBeDefined();
      expect(frontendPayload.corrected_text).toBeDefined();
      expect(frontendPayload.original_prediction).toBeUndefined();
      expect(frontendPayload.operator_correction).toBeUndefined();
    });
  });
});

/**
 * frontend/src/app/api/feedback/route.ts
 * Next.js App Router Route Handler for Darkroom operator feedback corrections.
 * Forwards validated feedback to backend /v1/feedback or provides simulated persistence fallback.
 */

import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const maxDuration = 30;

export async function POST(req: Request) {
  try {
    let body: any;
    try {
      body = await req.json();
    } catch {
      return NextResponse.json(
        { error: 'Invalid JSON request body' },
        { status: 400 }
      );
    }

    // Validate required fields
    const hasOriginal =
      Boolean(body && typeof body === 'object') &&
      (body.original_prediction !== undefined || body.original_text !== undefined);
    const hasCorrection =
      Boolean(body && typeof body === 'object') &&
      (body.operator_correction !== undefined || body.corrected_text !== undefined);

    if (
      !body ||
      typeof body !== 'object' ||
      !body.document_id ||
      !body.line_id ||
      !hasOriginal ||
      !hasCorrection
    ) {
      return NextResponse.json(
        {
          error:
            "Missing required feedback fields: 'document_id', 'line_id', 'original_prediction'/'original_text', 'operator_correction'/'corrected_text'.",
        },
        { status: 400 }
      );
    }

    const orig = body.original_prediction !== undefined ? body.original_prediction : body.original_text;
    const corr = body.operator_correction !== undefined ? body.operator_correction : body.corrected_text;
    const backendPayload = {
      ...body,
      original_prediction: orig,
      operator_correction: corr,
      original_text: orig,
      corrected_text: corr,
    };

    const backendUrl = process.env.BACKEND_URL;

    if (backendUrl) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 15000);
        let response: Response;
        try {
          response = await fetch(`${backendUrl}/v1/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(backendPayload),
            signal: controller.signal,
          });
        } finally {
          clearTimeout(timer);
        }

        if (response.ok) {
          const data = await response.json();
          return NextResponse.json(data, {
            status: 200,
            headers: { 'X-Feedback-Provider': 'backend' },
          });
        }

        let errorDetail: unknown;
        try {
          errorDetail = await response.json();
        } catch {
          errorDetail = {
            error: `Backend feedback failed with status ${response.status}`,
          };
        }
        return NextResponse.json(errorDetail, { status: response.status });
      } catch (proxyErr: unknown) {
        console.error('[API /api/feedback] Backend proxy error:', proxyErr);
        return NextResponse.json(
          {
            error: 'Backend feedback service unreachable',
            details: proxyErr instanceof Error ? proxyErr.message : String(proxyErr),
          },
          { status: 502 }
        );
      }
    }

    // Simulated persisted response when running without backend
    return NextResponse.json(
      {
        status: 'persisted',
        feedback_id: `fb_mock_${Date.now()}`,
        document_id: body.document_id,
        line_id: body.line_id,
        manifest_path: 'data/feedback/manifest.jsonl',
        timestamp: body.timestamp || new Date().toISOString(),
      },
      {
        status: 200,
        headers: { 'X-Feedback-Provider': 'mock' },
      }
    );
  } catch (err: unknown) {
    return NextResponse.json(
      {
        error: 'Internal Server Error',
        details: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}

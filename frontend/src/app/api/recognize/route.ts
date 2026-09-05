import { NextRequest, NextResponse } from 'next/server';
import { SAMPLE_PRESETS } from '@/lib/sampleDocuments';
import { runMockOcr } from '@/lib/mockOcrEngine';

export const dynamic = 'force-dynamic';
export const maxDuration = 300; // 300s timeout limit for cloud inference

export async function POST(req: Request | NextRequest) {
  try {
    let file: File | null = null;
    let sampleId: string | null = null;
    let fileBase64: string | null = null;
    let filename = 'upload.png';
    let modelType = 'trocr-handwritten-mps-v1';
    let options: Record<string, unknown> | null = null;
    let beamWidth = '4';
    let rescore = 'false';
    let turbo: string | undefined;
    let processingMode = 'local';

    const contentType = req.headers?.get('content-type') || '';

    if (contentType.includes('application/json')) {
      try {
        const json = await req.json();
        sampleId = json.sample_id || null;
        fileBase64 = json.file_base64 || null;
        filename = json.filename || filename;
        modelType = json.model_type || modelType;
        options = json.options || null;
        processingMode = options?.processing_mode as string || json.processing_mode || "local";
        if (json.turbo !== undefined) turbo = String(json.turbo);
        if (options && options.turbo !== undefined) turbo = String(options.turbo);
      } catch {
        // invalid JSON body
      }
    } else if (typeof req.formData === 'function') {
      try {
        const formData = await req.formData();
        const fileEntry = formData.get('file');
        if (fileEntry && typeof fileEntry === 'object' && 'name' in fileEntry) {
          file = fileEntry as File;
          filename = file.name;
        }
        const sampleIdEntry = formData.get('sample_id');
        if (typeof sampleIdEntry === 'string') {
          sampleId = sampleIdEntry;
        }
        const modelTypeEntry = formData.get('model_type');
        if (typeof modelTypeEntry === 'string') {
          modelType = modelTypeEntry;
        }
        const beamWidthEntry = formData.get('beam_width');
        if (typeof beamWidthEntry === 'string' && beamWidthEntry.length > 0) {
          beamWidth = beamWidthEntry;
        }
        const rescoreEntry = formData.get('rescore');
        if (typeof rescoreEntry === 'string' && rescoreEntry.length > 0) {
          rescore = rescoreEntry;
        }
        processingMode = String(formData.get('processing_mode') || 'local');
        const turboEntry = formData.get('turbo');
        if (typeof turboEntry === 'string' && turboEntry.length > 0) {
          turbo = turboEntry;
        }
      } catch (formErr: unknown) {
        console.warn('[API /api/recognize] FormData parsing failed:', formErr);
      }
    }

    // 1. If explicit sample ID requested
    if (sampleId && SAMPLE_PRESETS[sampleId]) {
      const presetData = { ...structuredClone(SAMPLE_PRESETS[sampleId]), is_demo: true, engine_used: "demo" };
      return NextResponse.json(presetData, {
        status: 200,
        headers: {
          'X-Recognition-Provider': 'mock-preset',
          'Content-Type': 'application/json',
        },
      });
    }

    // 2. Validate file presence
    if (!file && !fileBase64) {
      return NextResponse.json(
        { error: "Missing required 'file', 'file_base64', or 'sample_id' in request body." },
        { status: 400 }
      );
    }

    const backendUrl = process.env.BACKEND_URL;

    // 3. If backend is configured, proxy to FastAPI /v1/recognize with 240s timeout
    if (backendUrl) {
      try {
        let response: Response;

        if (file) {
          const proxyFormData = new FormData();
          proxyFormData.append('file', file);
          proxyFormData.append('model_type', modelType);


          const qs = new URLSearchParams({
            beam_width: beamWidth,
            rescore,
            processing_mode: processingMode,
            ...(turbo !== undefined ? { turbo } : {}),
          }).toString();
          response = await fetch(`${backendUrl}/v1/recognize?${qs}`, {
            method: 'POST',
            body: proxyFormData,
            signal: AbortSignal.timeout(300000), // 300s timeout
          });
        } else {
          const payload = {
            file_base64: fileBase64,
            filename,
            options: { ...(options || {}), processing_mode: processingMode, ...(turbo !== undefined ? { turbo: turbo === 'true' } : {}) },
          };
          const optRecord = options ?? {};
          const qs = new URLSearchParams({
            beam_width: String(optRecord.beam_width ?? 4),
            rescore: String(optRecord.rescore ?? false),
            processing_mode: processingMode,
            ...(turbo !== undefined ? { turbo } : {}),
          }).toString();
          response = await fetch(`${backendUrl}/v1/recognize?${qs}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(300000), // 300s timeout
          });
        }

        if (response.ok) {
          const backendData = await response.json();
          return NextResponse.json(backendData, {
            status: 200,
            headers: {
              'X-Recognition-Provider': 'backend',
              'Content-Type': 'application/json',
            },
          });
        }

        // Forward upstream HTTP errors directly from backend (4xx, 5xx)
        let errorBody: unknown;
        try {
          errorBody = await response.json();
        } catch {
          const text = await response.text().catch(() => response.statusText);
          errorBody = { error: text || `Backend returned status ${response.status}` };
        }

        return NextResponse.json(errorBody, {
          status: response.status,
          headers: {
            'X-Recognition-Provider': 'backend',
            'Content-Type': 'application/json',
          },
        });
      } catch (proxyError: unknown) {
        console.error('[API /api/recognize] Backend proxy error:', proxyError);

        const isTimeout =
          proxyError instanceof Error &&
          (proxyError.name === 'TimeoutError' ||
            proxyError.name === 'AbortError' ||
            proxyError.message.toLowerCase().includes('timeout') ||
            proxyError.message.toLowerCase().includes('aborted'));

        if (isTimeout) {
          return NextResponse.json(
            {
              error: 'Backend recognition timed out',
              details: 'Inference request exceeded the 240-second timeout limit.',
            },
            {
              status: 504,
              headers: {
                'X-Recognition-Provider': 'backend',
                'Content-Type': 'application/json',
              },
            }
          );
        }

        return NextResponse.json(
          {
            error: 'Backend recognition service unavailable',
            details: proxyError instanceof Error ? proxyError.message : String(proxyError),
          },
          {
            status: 502,
            headers: {
              'X-Recognition-Provider': 'backend',
              'Content-Type': 'application/json',
            },
          }
        );
      }
    }

    return NextResponse.json({ error: 'Recognition service is not configured' }, { status: 503 });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Unknown server error';
    return NextResponse.json(
      { error: 'Internal Server Error', details: message },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json(
    { error: 'Method Not Allowed. Use POST.' },
    {
      status: 405,
      headers: { Allow: 'POST' },
    }
  );
}

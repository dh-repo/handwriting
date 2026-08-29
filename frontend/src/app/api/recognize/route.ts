import { NextRequest, NextResponse } from 'next/server';
import { SAMPLE_PRESETS } from '@/lib/sampleDocuments';
import { runMockOcr } from '@/lib/mockOcrEngine';

export const dynamic = 'force-dynamic';
export const maxDuration = 60; // Vercel function timeout limit

export async function POST(req: Request | NextRequest) {
  try {
    let file: File | null = null;
    let sampleId: string | null = null;
    let fileBase64: string | null = null;
    let filename = 'upload.png';
    let modelType = 'trocr-handwritten-mps-v1';
    let options: Record<string, unknown> | null = null;

    const contentType = req.headers?.get('content-type') || '';

    if (contentType.includes('application/json')) {
      try {
        const json = await req.json();
        sampleId = json.sample_id || null;
        fileBase64 = json.file_base64 || null;
        filename = json.filename || filename;
        modelType = json.model_type || modelType;
        options = json.options || null;
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
      } catch (formErr: unknown) {
        console.warn('[API /api/recognize] FormData parsing failed:', formErr);
      }
    }

    // 1. If explicit sample ID requested
    if (sampleId && SAMPLE_PRESETS[sampleId]) {
      const presetData = structuredClone(SAMPLE_PRESETS[sampleId]);
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

    const backendUrl = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL;

    // 3. If backend is configured, attempt proxying to FastAPI /v1/recognize
    if (backendUrl) {
      try {
        if (file) {
          const proxyFormData = new FormData();
          proxyFormData.append('file', file);
          proxyFormData.append('model_type', modelType);

          const response = await fetch(`${backendUrl}/v1/recognize`, {
            method: 'POST',
            body: proxyFormData,
            signal: AbortSignal.timeout(15000), // 15s timeout
          });

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
        } else if (fileBase64) {
          const payload = {
            file_base64: fileBase64,
            filename,
            options,
          };
          const response = await fetch(`${backendUrl}/v1/recognize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(15000),
          });

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
        }

        console.warn(
          '[API /api/recognize] Backend request failed. Falling back to in-app mock engine.'
        );
      } catch (proxyError) {
        console.warn(
          '[API /api/recognize] Backend proxy failed, falling back to in-app mock engine:',
          proxyError
        );
      }
    }

    // 4. Standalone in-app mock engine execution
    const mockResult = await runMockOcr({
      filename: file ? file.name : filename,
      mimeType: file ? file.type : 'image/png',
      fileSize: file ? file.size : fileBase64 ? Math.round(fileBase64.length * 0.75) : 1024,
    });

    return NextResponse.json(mockResult, {
      status: 200,
      headers: {
        'X-Recognition-Provider': 'mock',
        'Content-Type': 'application/json',
      },
    });
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

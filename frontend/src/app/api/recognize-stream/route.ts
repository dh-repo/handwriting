import { NextRequest, NextResponse } from 'next/server';
import { SAMPLE_PRESETS } from '@/lib/sampleDocuments';
import { runMockOcr } from '@/lib/mockOcrEngine';

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

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
    let adaptive = 'true';
    let turbo: string | undefined;
    let processingMode = new URL(req.url).searchParams.get('processing_mode') || 'local';

    const contentType = req.headers?.get('content-type') || '';

    if (contentType.includes('application/json')) {
      try {
        const json = await req.json();
        sampleId = json.sample_id || null;
        fileBase64 = json.file_base64 || null;
        filename = json.filename || filename;
        modelType = json.model_type || modelType;
        options = json.options || null;
        processingMode = options?.processing_mode as string || processingMode;
        if (json.turbo !== undefined) turbo = String(json.turbo);
        if (options && options.turbo !== undefined) turbo = String(options.turbo);
      } catch {
        // invalid JSON
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
        if (typeof sampleIdEntry === 'string') sampleId = sampleIdEntry;
        const modelTypeEntry = formData.get('model_type');
        if (typeof modelTypeEntry === 'string') modelType = modelTypeEntry;
        const beamWidthEntry = formData.get('beam_width');
        if (typeof beamWidthEntry === 'string' && beamWidthEntry.length > 0) beamWidth = beamWidthEntry;
        const rescoreEntry = formData.get('rescore');
        if (typeof rescoreEntry === 'string' && rescoreEntry.length > 0) rescore = rescoreEntry;
        const adaptiveEntry = formData.get('adaptive');
        if (typeof adaptiveEntry === 'string' && adaptiveEntry.length > 0) adaptive = adaptiveEntry;
        processingMode = String(formData.get('processing_mode') || processingMode);
        const turboEntry = formData.get('turbo');
        if (typeof turboEntry === 'string' && turboEntry.length > 0) turbo = turboEntry;
      } catch (formErr: unknown) {
        console.warn('[API /api/recognize-stream] FormData parsing failed:', formErr);
      }
    }

    // 1. If explicit sample ID requested, stream mock preset
    if (sampleId && SAMPLE_PRESETS[sampleId]) {
      const presetData = { ...structuredClone(SAMPLE_PRESETS[sampleId]), is_demo: true, engine_used: "demo" };
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(
            encoder.encode(
              `event: metadata\ndata: ${JSON.stringify({
                document_id: presetData.document_id,
                filename: presetData.filename,
                total_pages: presetData.total_pages,
                pages: presetData.pages.map((p) => ({
                  page_number: p.page_number,
                  width: p.width,
                  height: p.height,
                  total_lines: p.lines.length,
                })),
              })}\n\n`
            )
          );
          for (const page of presetData.pages) {
            for (const line of page.lines) {
              controller.enqueue(
                encoder.encode(
                  `event: line\ndata: ${JSON.stringify({ ...line, page_number: page.page_number })}\n\n`
                )
              );
            }
          }
          controller.enqueue(encoder.encode(`event: complete\ndata: ${JSON.stringify(presetData)}\n\n`));
          controller.close();
        },
      });
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
          'X-Recognition-Provider': 'mock-preset',
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

    // 3. Proxy to FastAPI /v1/recognize/stream
    if (backendUrl) {
      try {
        let backendRes: Response;
        const qs = new URLSearchParams({
          beam_width: beamWidth,
          rescore,
          adaptive,
          processing_mode: processingMode,
          ...(turbo !== undefined ? { turbo } : {}),
        }).toString();

        if (file) {
          const proxyFormData = new FormData();
          proxyFormData.append('file', file);
          proxyFormData.append('model_type', modelType);


          backendRes = await fetch(`${backendUrl}/v1/recognize/stream?${qs}`, {
            method: 'POST',
            body: proxyFormData,
            signal: AbortSignal.timeout(300000),
          });
        } else {
          const payload = {
            file_base64: fileBase64,
            filename,
            options: { ...(options || {}), processing_mode: processingMode, ...(turbo !== undefined ? { turbo: turbo === 'true' } : {}) },
          };
          backendRes = await fetch(`${backendUrl}/v1/recognize/stream?${qs}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(300000),
          });
        }

        if (backendRes.ok && backendRes.body) {
          return new Response(backendRes.body, {
            headers: {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
              Connection: 'keep-alive',
              'X-Recognition-Provider': 'fastapi-backend-stream',
            },
          });
        }
      } catch (backendErr: unknown) {
        console.warn('[API /api/recognize-stream] Backend stream failed, falling back to mock:', backendErr);
      }
    }

    return NextResponse.json({ error: 'Recognition service unavailable' }, { status: 503 });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[API /api/recognize-stream] Uncaught error:', msg);
    return NextResponse.json({ error: 'Internal recognition stream error', details: msg }, { status: 500 });
  }
}

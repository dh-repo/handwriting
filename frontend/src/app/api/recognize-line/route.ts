import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const maxDuration = 120;

export async function POST(req: Request) {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    return NextResponse.json({ error: 'BACKEND_URL is not configured' }, { status: 503 });
  }

  let formData: FormData;
  try {
    formData = await req.formData();
  } catch {
    return NextResponse.json({ error: 'Expected multipart file upload' }, { status: 400 });
  }

  const file = formData.get('file');
  if (file == null) {
    return NextResponse.json({ error: "Missing required 'file'" }, { status: 400 });
  }

  const proxy = new FormData();
  proxy.append('file', file);

  try {
    const response = await fetch(`${backendUrl}/v1/recognize-line`, {
      method: 'POST',
      body: proxy,
      signal: AbortSignal.timeout(120000),
    });
    const body = await response.json();
    return NextResponse.json(body, { status: response.status });
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : 'Backend line recognition proxy failed';
    return NextResponse.json({ error: detail }, { status: 502 });
  }
}

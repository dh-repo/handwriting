import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET() {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    return NextResponse.json(
      { error: 'BACKEND_URL is not configured' },
      { status: 503 }
    );
  }

  try {
    const response = await fetch(`${backendUrl}/v1/health`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(15000),
    });
    const body = await response.json();
    return NextResponse.json(body, { status: response.status });
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : 'Backend health proxy failed';
    return NextResponse.json({ error: detail }, { status: 502 });
  }
}

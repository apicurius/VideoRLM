import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const LOGS_DIR = path.resolve(process.cwd(), '../logs/');

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ fileName: string }> },
) {
  try {
    const { fileName } = await params;
    // Prevent directory traversal
    const safe = path.basename(fileName);
    const filePath = path.join(LOGS_DIR, safe);

    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ error: 'Not found' }, { status: 404 });
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    return new NextResponse(content, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-store, max-age=0',
      },
    });
  } catch (error) {
    if (process.env.NODE_ENV === 'development') console.error('Failed to read log file:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

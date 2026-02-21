import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const LOGS_DIR = path.resolve(process.cwd(), '../logs/');

/**
 * Serve a single JSONL log file by name.
 * URL: /api/logs/<fileName>
 */
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ fileName: string }> }
) {
  const { fileName } = await params;

  // Validate: must be a .jsonl file, no path traversal
  if (!fileName.endsWith('.jsonl') || fileName.includes('..') || fileName.includes('/')) {
    return NextResponse.json({ error: 'Invalid file name' }, { status: 400 });
  }

  const filePath = path.join(LOGS_DIR, fileName);
  const resolved = path.resolve(filePath);

  // Ensure resolved path is within LOGS_DIR
  if (!resolved.startsWith(path.resolve(LOGS_DIR))) {
    return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
  }

  try {
    if (!fs.existsSync(resolved)) {
      return NextResponse.json({ error: 'File not found' }, { status: 404 });
    }

    const content = fs.readFileSync(resolved, 'utf-8');
    return new NextResponse(content, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache',
      },
    });
  } catch {
    return NextResponse.json({ error: 'Failed to read file' }, { status: 500 });
  }
}

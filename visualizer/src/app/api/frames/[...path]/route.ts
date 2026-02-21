import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const LOGS_DIR = path.resolve(process.cwd(), '../logs/');

/**
 * Serve frame images from sidecar .frames/ directories.
 * URL: /api/frames/<log_stem>.frames/<frame_file>
 */
export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const segments = (await params).path;

  if (segments.length !== 2) {
    return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
  }

  const [framesDir, frameFile] = segments;

  // Validate: frames dir must end with .frames, frame file must be an image
  if (
    !framesDir.endsWith('.frames') ||
    framesDir.includes('..') ||
    frameFile.includes('..') ||
    frameFile.includes('/')
  ) {
    return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
  }

  const filePath = path.join(LOGS_DIR, framesDir, frameFile);
  const resolved = path.resolve(filePath);

  if (!resolved.startsWith(path.resolve(LOGS_DIR))) {
    return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
  }

  try {
    if (!fs.existsSync(resolved)) {
      return NextResponse.json({ error: 'Frame not found' }, { status: 404 });
    }

    const data = fs.readFileSync(resolved);
    const ext = path.extname(frameFile).toLowerCase();
    const contentType = ext === '.png' ? 'image/png' : 'image/jpeg';

    return new NextResponse(data, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=86400',
      },
    });
  } catch {
    return NextResponse.json({ error: 'Failed to read frame' }, { status: 500 });
  }
}

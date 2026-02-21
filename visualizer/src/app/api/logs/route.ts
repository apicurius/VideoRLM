import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const LOGS_DIR = path.resolve(process.cwd(), '../logs/');

interface FilePreview {
  name: string;
  size: number;
  mtime: string;
  traceType: 'rlm' | 'kuavi';
  lineCount: number;
  toolCallCount?: number;
  model: string | null;
  videoPath: string | null;
}

/**
 * Detect trace type and extract lightweight metadata from the first few lines
 * of a JSONL file without parsing the entire file.
 */
function getFilePreview(filePath: string, stat: fs.Stats): FilePreview | null {
  const name = path.basename(filePath);
  if (!name.endsWith('.jsonl')) return null;

  try {
    // Read first 8KB for quick classification
    const fd = fs.openSync(filePath, 'r');
    const buf = Buffer.alloc(8192);
    const bytesRead = fs.readSync(fd, buf, 0, 8192, 0);
    fs.closeSync(fd);

    const head = buf.toString('utf-8', 0, bytesRead);
    const lines = head.split('\n').filter((l) => l.trim());

    let traceType: 'rlm' | 'kuavi' = 'rlm';
    let model: string | null = null;
    let videoPath: string | null = null;
    let toolCallCount = 0;

    // Count total lines (approximate from file size for large files)
    const content = fs.readFileSync(filePath, 'utf-8');
    const allLines = content.split('\n').filter((l) => l.trim());
    const lineCount = allLines.length;

    // Classify from the first line
    if (lines.length > 0) {
      try {
        const first = JSON.parse(lines[0]);
        if (
          first.type === 'tool_call' ||
          first.type === 'session_start' ||
          first.type === 'agent_start' ||
          first.type === 'session_end' ||
          first.type === 'agent_stop' ||
          first.type === 'final_answer' ||
          first.type === 'turn_start' ||
          first.type === 'reasoning' ||
          first.type === 'system_prompt' ||
          first.type === 'question' ||
          (first.type === 'metadata' && first.video_path != null)
        ) {
          traceType = 'kuavi';
        }
      } catch {
        // Not valid JSON — treat as RLM
      }
    }

    // Extract model and video path from early lines
    for (const line of allLines.slice(0, 50)) {
      try {
        const obj = JSON.parse(line);
        if (obj.type === 'session_start' && obj.model) {
          model = obj.model;
        }
        if (obj.type === 'metadata' && obj.video_path) {
          videoPath = obj.video_path;
        }
        if (obj.type === 'tool_call') {
          toolCallCount++;
        }
        // RLM metadata
        if (obj.type === 'metadata' && obj.root_model) {
          model = obj.root_model;
        }
        if (obj.type === 'metadata' && obj.video_path) {
          videoPath = obj.video_path;
        }
      } catch {
        // Skip malformed lines
      }
    }

    // For KUAVi traces, count all tool calls
    if (traceType === 'kuavi') {
      toolCallCount = 0;
      for (const line of allLines) {
        try {
          const obj = JSON.parse(line);
          if (obj.type === 'tool_call') toolCallCount++;
        } catch {
          // Skip
        }
      }
    }

    return {
      name,
      size: stat.size,
      mtime: stat.mtime.toISOString(),
      traceType,
      lineCount,
      toolCallCount: traceType === 'kuavi' ? toolCallCount : undefined,
      model,
      videoPath,
    };
  } catch {
    return null;
  }
}

export async function GET() {
  try {
    if (!fs.existsSync(LOGS_DIR)) {
      return NextResponse.json({ files: [] });
    }

    const entries = fs.readdirSync(LOGS_DIR).filter((f) => f.endsWith('.jsonl'));

    // Get stats and sort by mtime descending (newest first)
    const withStats = entries
      .map((name) => {
        const filePath = path.join(LOGS_DIR, name);
        try {
          const stat = fs.statSync(filePath);
          return { filePath, stat };
        } catch {
          return null;
        }
      })
      .filter((e): e is { filePath: string; stat: fs.Stats } => e !== null)
      .sort((a, b) => b.stat.mtime.getTime() - a.stat.mtime.getTime())
      .slice(0, 20); // Latest 20

    const files: FilePreview[] = [];
    for (const { filePath, stat } of withStats) {
      const preview = getFilePreview(filePath, stat);
      if (preview) files.push(preview);
    }

    return NextResponse.json({ files });
  } catch {
    return NextResponse.json({ files: [] });
  }
}

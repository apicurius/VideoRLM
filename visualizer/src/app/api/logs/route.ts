import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const LOGS_DIR = path.resolve(process.cwd(), '../logs/');

export async function GET() {
  try {
    if (!fs.existsSync(LOGS_DIR)) {
      return NextResponse.json({ files: [] });
    }

    const allFiles = fs.readdirSync(LOGS_DIR);
    const jsonlFiles = allFiles
      .filter((f) => f.endsWith('.jsonl'))
      .sort((a, b) => {
        // Sort by modification time (newest first)
        try {
          const aStat = fs.statSync(path.join(LOGS_DIR, a));
          const bStat = fs.statSync(path.join(LOGS_DIR, b));
          return bStat.mtimeMs - aStat.mtimeMs;
        } catch {
          return a < b ? 1 : -1;
        }
      })
      .slice(0, 40);

    // Build metadata for each file, filtering out empty/session-only traces
    const filesWithMeta = jsonlFiles.map(name => {
      const filePath = path.join(LOGS_DIR, name);
      const stat = fs.statSync(filePath);
      const traceType = name.startsWith('kuavi_') ? 'kuavi' : 'rlm';

      // For files under 500KB, read full content for accurate line count.
      // For larger files, estimate from file size.
      let lineCount: number;
      let content: string | null = null;
      if (stat.size <= 500 * 1024) {
        content = fs.readFileSync(filePath, 'utf-8');
        lineCount = (content.match(/\n/g) || []).length;
      } else {
        // Estimate: average ~500 bytes per JSONL line for KUAVi traces
        lineCount = Math.round(stat.size / 500);
      }

      // Extract metadata from first few lines
      const preview = content ?? (() => {
        const fd = fs.openSync(filePath, 'r');
        const buf = Buffer.alloc(8192);
        const bytesRead = fs.readSync(fd, buf, 0, 8192, 0);
        fs.closeSync(fd);
        return buf.toString('utf-8', 0, bytesRead);
      })();
      const previewLines = preview.split('\n').slice(0, 10);

      let model = null, videoPath = null;
      for (const line of previewLines) {
        if (!line.trim()) continue;
        try {
          const obj = JSON.parse(line);
          if (!model) {
            // KUAVi: session_start has "model"; RLM metadata has "root_model"
            model = obj.model ?? obj.root_model ?? obj.metadata?.model ?? null;
          }
          if (!videoPath) {
            // KUAVi: tool_input.video_path on index_video calls
            // RLM: top-level video_path on metadata lines
            videoPath = obj.tool_input?.video_path ?? obj.video_path ?? null;
          }
        } catch {}
        if (model && videoPath) break;
      }

      // Count tool calls, llm_calls, eval_executions for KUAVi traces
      let toolCallCount = 0;
      let llmCallCount = 0;
      let evalCount = 0;
      let hasQuestion = false;
      if (traceType === 'kuavi') {
        // Always read full file for accurate event counts (JSONL files are line-oriented, fast to scan)
        const text = content ?? fs.readFileSync(filePath, 'utf-8');
        for (const line of text.split('\n')) {
          if (line.includes('"tool_call"')) toolCallCount++;
          else if (line.includes('"llm_call"')) llmCallCount++;
          else if (line.includes('"eval_execution"')) evalCount++;
          if (!hasQuestion && line.includes('"question"')) hasQuestion = true;
        }
      }

      return {
        name,
        size: stat.size,
        mtime: stat.mtime.toISOString(),
        traceType,
        lineCount: Math.max(lineCount, 1),
        toolCallCount: toolCallCount + llmCallCount + evalCount,
        llmCallCount,
        evalCount,
        hasQuestion,
        model,
        videoPath,
      };
    }).filter(f => {
      // Hide KUAVi traces with zero tool calls (session-only lifecycle events)
      if (f.traceType === 'kuavi' && f.toolCallCount === 0) return false;
      // Hide old hook-only traces when a unified _mcp trace exists for the same timestamp.
      // Hook files: kuavi_YYYY-MM-DD_HH-MM-SS_SESSION.jsonl
      // MCP files:  kuavi_YYYY-MM-DD_HH-MM-SS[_rN]_mcp.jsonl
      if (f.traceType === 'kuavi' && !f.name.includes('_mcp.jsonl')) {
        const tsMatch = f.name.match(/^kuavi_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})/);
        if (tsMatch) {
          const ts = tsMatch[1];
          const hasMcp = jsonlFiles.some(
            other => other !== f.name && other.includes(ts) && other.includes('_mcp.jsonl')
          );
          if (hasMcp) return false;
        }
      }
      return true;
    }).slice(0, 20);

    return NextResponse.json({ files: filesWithMeta }, {
      headers: { 'Cache-Control': 'no-store, max-age=0' },
    });
  } catch (error) {
    if (process.env.NODE_ENV === 'development') console.error('Failed to list log files:', error);
    return NextResponse.json({ files: [] }, { status: 500 });
  }
}

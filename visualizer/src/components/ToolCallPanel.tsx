'use client';

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { VideoFrameViewer, extractImageFrames } from '@/components/VideoFrameViewer';
import { cn } from '@/lib/utils';
import { KUAViToolCall } from '@/lib/types';
import { shortToolName } from '@/lib/parse-logs';
import { SyntaxHighlight } from '@/components/SyntaxHighlight';

interface ToolCallPanelProps {
  toolCall: KUAViToolCall | null;
  logStem?: string;
}

function formatTimestamp(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString();
  } catch {
    return ts;
  }
}

function JsonDisplay({ value }: { value: unknown }) {
  const text = JSON.stringify(value, null, 2);
  return (
    <pre className="text-xs font-mono bg-muted/50 rounded-lg p-4 overflow-auto whitespace-pre-wrap break-words">
      <code className="text-foreground/90">{text}</code>
    </pre>
  );
}

function SearchResultDisplay({ response }: { response: unknown }) {
  if (!Array.isArray(response)) return <JsonDisplay value={response} />;

  return (
    <div className="space-y-2">
      {(response as unknown[]).map((item, idx) => {
        if (typeof item !== 'object' || item === null) {
          return (
            <div key={idx} className="text-xs font-mono bg-muted/50 rounded p-2">
              {String(item)}
            </div>
          );
        }
        const obj = item as Record<string, unknown>;
        // Extract caption text from annotation if available
        const annotation = obj.annotation as Record<string, unknown> | undefined;
        const summaryText = obj.caption
          ? String(obj.caption)
          : annotation?.summary
            ? typeof annotation.summary === 'object' && annotation.summary !== null
              ? String((annotation.summary as Record<string, unknown>).brief ?? '')
              : String(annotation.summary)
            : '';
        const actionText = annotation?.action
          ? typeof annotation.action === 'object' && annotation.action !== null
            ? String((annotation.action as Record<string, unknown>).brief ?? '')
            : String(annotation.action)
          : '';
        return (
          <div key={idx} className="rounded-lg border border-border bg-card p-3 space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              {obj.scene_index !== undefined && (
                <Badge className="bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30 text-xs">
                  Scene {String(obj.scene_index)}
                </Badge>
              )}
              {obj.score !== undefined && (
                <Badge className="bg-sky-500/15 text-sky-600 dark:text-sky-400 border-sky-500/30 text-xs">
                  score: {typeof obj.score === 'number' ? obj.score.toFixed(3) : String(obj.score)}
                </Badge>
              )}
              {obj.confidence !== undefined && (
                <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-xs">
                  confidence: {typeof obj.confidence === 'number' ? obj.confidence.toFixed(3) : String(obj.confidence)}
                </Badge>
              )}
              {obj.start_time !== undefined && obj.end_time !== undefined && (
                <Badge variant="outline" className="text-xs font-mono">
                  {String(obj.start_time)}s – {String(obj.end_time)}s
                </Badge>
              )}
              {!!obj.field && (
                <Badge variant="outline" className="text-xs">{String(obj.field)}</Badge>
              )}
            </div>
            {!!obj.answer && (
              <p className="text-xs text-foreground/80 font-medium">{String(obj.answer)}</p>
            )}
            {summaryText && summaryText !== 'N/A' && (
              <p className="text-xs text-foreground/80">{summaryText}</p>
            )}
            {actionText && actionText !== 'N/A' && (
              <p className="text-xs text-muted-foreground">{actionText}</p>
            )}
            {!!obj.text && !summaryText && (
              <p className="text-xs text-foreground/80">{String(obj.text)}</p>
            )}
            {!!obj.context && (
              <p className="text-xs text-muted-foreground italic">{String(obj.context)}</p>
            )}
          </div>
        );
      })}
    </div>
  );
}

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function KeyValueDisplay({ data, title }: { data: Record<string, unknown>; title?: string }) {
  const entries = Object.entries(data).filter(([, v]) => v !== null && v !== undefined);
  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      {title && (
        <div className="px-3 py-2 border-b border-border bg-muted/30">
          <span className="text-xs font-semibold text-foreground">{title}</span>
        </div>
      )}
      <div className="divide-y divide-border">
        {entries.map(([key, value]) => (
          <div key={key} className="flex items-start gap-3 px-3 py-1.5">
            <span className="text-xs font-mono text-muted-foreground w-40 shrink-0 pt-0.5">
              {key.replace(/_/g, ' ')}
            </span>
            <span className="text-xs text-foreground break-all">
              {typeof value === 'boolean'
                ? value ? 'Yes' : 'No'
                : typeof value === 'number'
                  ? key.includes('duration') || key.includes('elapsed') || key.includes('time')
                    ? formatDuration(value)
                    : String(value)
                  : typeof value === 'object'
                    ? JSON.stringify(value)
                    : String(value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EvalInputDisplay({ toolCall }: { toolCall: KUAViToolCall }) {
  const code = typeof toolCall.tool_input?.code === 'string' ? toolCall.tool_input.code : '';
  const resp = toolCall.tool_response;
  const unwrapped = typeof resp === 'object' && resp !== null && 'result' in resp
    ? (resp as Record<string, unknown>).result
    : resp;
  const stdout = typeof unwrapped === 'object' && unwrapped !== null
    ? String((unwrapped as Record<string, unknown>).stdout ?? '')
    : '';

  return (
    <div className="space-y-3">
      <pre className="text-xs font-mono bg-slate-950 dark:bg-slate-950/80 rounded-lg p-4 overflow-auto whitespace-pre-wrap break-words leading-relaxed">
        <SyntaxHighlight code={code} language="python" />
      </pre>
      {stdout && (
        <div className="space-y-1">
          <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">stdout</span>
          <pre className="text-xs font-mono bg-emerald-950/30 dark:bg-emerald-950/40 text-emerald-300 border border-emerald-500/20 rounded-lg p-3 overflow-auto whitespace-pre-wrap break-words">
            {stdout}
          </pre>
        </div>
      )}
    </div>
  );
}

function TranscriptDisplay({ text }: { text: string }) {
  // Parse lines like "[123.4s] Some text" into structured entries
  const lines = text.split('\n').filter((l) => l.trim());
  const parsed = lines.map((line) => {
    const match = line.match(/^\[(\d+(?:\.\d+)?)s\]\s*(.*)/);
    if (match) return { time: parseFloat(match[1]), text: match[2] };
    return { time: null, text: line };
  });

  return (
    <div className="space-y-1">
      {parsed.map((entry, idx) => (
        <div key={idx} className="flex gap-2 text-xs">
          {entry.time !== null && (
            <span className="text-xs font-mono text-muted-foreground w-16 shrink-0 text-right tabular-nums">
              {formatDuration(entry.time)}
            </span>
          )}
          <span className="text-foreground/80">{entry.text}</span>
        </div>
      ))}
    </div>
  );
}

function IndexStatusDisplay({ data }: { data: Record<string, unknown> }) {
  const status = data.status as string | undefined;
  return (
    <div className="space-y-3">
      {status && (
        <div className="flex items-center gap-2">
          <Badge className={cn(
            'text-xs',
            status === 'indexed'
              ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30'
              : 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30'
          )}>
            {status}
          </Badge>
        </div>
      )}
      <KeyValueDisplay data={data} />
    </div>
  );
}

/** Unwrap {result: ...} wrapper that MCP tools return */
function unwrapResult(response: unknown): unknown {
  if (typeof response === 'object' && response !== null && 'result' in response) {
    return (response as Record<string, unknown>).result;
  }
  return response;
}

function ResponseDisplay({ toolCall, logStem }: { toolCall: KUAViToolCall; logStem?: string }) {
  const resp = unwrapResult(toolCall.tool_response);
  const frames = extractImageFrames(toolCall.tool_response);
  const short = shortToolName(toolCall.tool_name);
  const isSearchLike = ['search_video', 'search_transcript', 'discriminative_vqa', 'get_scene_list'].some(
    (s) => short.includes(s)
  );

  if (frames.length > 0) {
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Badge className="bg-violet-500/15 text-violet-600 dark:text-violet-400 border-violet-500/30 text-xs">
            {frames.length} frame{frames.length !== 1 ? 's' : ''} extracted
          </Badge>
        </div>
        <VideoFrameViewer frames={frames} thumbSize={140} logStem={logStem} />
      </div>
    );
  }

  // Array-based results: search, scene list, VQA
  if (isSearchLike) {
    const data = Array.isArray(resp) ? resp : resp;
    return <SearchResultDisplay response={Array.isArray(data) ? data : toolCall.tool_response} />;
  }

  // Transcript text display
  if (short.includes('get_transcript') && typeof resp === 'string') {
    return <TranscriptDisplay text={resp} />;
  }

  // Index status card
  if (short.includes('index_video') && typeof resp === 'object' && resp !== null) {
    return <IndexStatusDisplay data={resp as Record<string, unknown>} />;
  }

  // Key-value displays for info/stats tools
  if (
    (short.includes('get_index_info') || short.includes('get_session_stats')) &&
    typeof resp === 'object' &&
    resp !== null &&
    !Array.isArray(resp)
  ) {
    const title = short.includes('get_index_info') ? 'Index Info' : 'Session Stats';
    return <KeyValueDisplay data={resp as Record<string, unknown>} title={title} />;
  }

  return <JsonDisplay value={toolCall.tool_response} />;
}

export function ToolCallPanel({ toolCall, logStem }: ToolCallPanelProps) {
  if (!toolCall) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center space-y-2">
          <p className="text-4xl">◈</p>
          <p className="text-sm text-muted-foreground">Select a tool call to inspect</p>
        </div>
      </div>
    );
  }

  const short = shortToolName(toolCall.tool_name);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-card/60 flex items-start gap-3 flex-shrink-0">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-sm font-mono text-foreground">{short}</span>
            <Badge variant="outline" className="text-xs font-mono truncate max-w-[200px]" title={toolCall.tool_name}>
              {toolCall.tool_name}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground font-mono mt-0.5">
            {formatTimestamp(toolCall.timestamp)}
          </p>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="input" className="flex-1 min-h-0 flex flex-col">
        <TabsList className="mx-4 mt-2 mb-0 w-fit flex-shrink-0">
          <TabsTrigger value="input" className="text-xs">Input</TabsTrigger>
          <TabsTrigger value="response" className="text-xs">Response</TabsTrigger>
        </TabsList>

        <TabsContent value="input" className={cn('flex-1 min-h-0 mt-2 px-4 pb-4')}>
          <ScrollArea className="h-full">
            {short === 'eval' ? (
              <EvalInputDisplay toolCall={toolCall} />
            ) : (
              <JsonDisplay value={toolCall.tool_input} />
            )}
          </ScrollArea>
        </TabsContent>

        <TabsContent value="response" className={cn('flex-1 min-h-0 mt-2 px-4 pb-4')}>
          <ScrollArea className="h-full">
            <ResponseDisplay toolCall={toolCall} logStem={logStem} />
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  );
}

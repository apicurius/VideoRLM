'use client';

import { useRef, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { KUAViEvent, KUAViToolCall, KUAViAgentEvent, KUAViTurnStartEvent } from '@/lib/types';
import { shortToolName, KUAViTurn } from '@/lib/parse-logs';

interface ToolCallTimelineProps {
  events: KUAViEvent[];
  selectedIndex: number;
  onSelectIndex: (index: number) => void;
}

type ToolCategory = 'search' | 'frame' | 'pixel' | 'eval' | 'meta' | 'analyze';

function getToolCategory(name: string): ToolCategory {
  const short = shortToolName(name);
  if (['search_video', 'search_transcript', 'get_transcript', 'discriminative_vqa'].includes(short)) return 'search';
  if (['extract_frames', 'zoom_frames'].includes(short)) return 'frame';
  if (['crop_frame', 'diff_frames', 'blend_frames', 'threshold_frame', 'frame_info'].includes(short)) return 'pixel';
  if (short === 'eval') return 'eval';
  if (short === 'analyze_shards') return 'analyze';
  return 'meta';
}

const categoryStyles: Record<ToolCategory, { border: string; bg: string; selectedBg: string; text: string; dot: string }> = {
  search: {
    border: 'border-sky-500/40 dark:border-sky-400/40',
    bg: 'bg-sky-500/5 dark:bg-sky-400/5 hover:bg-sky-500/10',
    selectedBg: 'border-sky-500 bg-sky-500/15 dark:border-sky-400 dark:bg-sky-400/15',
    text: 'text-sky-600 dark:text-sky-400',
    dot: 'bg-sky-500 dark:bg-sky-400',
  },
  frame: {
    border: 'border-violet-500/40 dark:border-violet-400/40',
    bg: 'bg-violet-500/5 dark:bg-violet-400/5 hover:bg-violet-500/10',
    selectedBg: 'border-violet-500 bg-violet-500/15 dark:border-violet-400 dark:bg-violet-400/15',
    text: 'text-violet-600 dark:text-violet-400',
    dot: 'bg-violet-500 dark:bg-violet-400',
  },
  pixel: {
    border: 'border-orange-500/40 dark:border-orange-400/40',
    bg: 'bg-orange-500/5 dark:bg-orange-400/5 hover:bg-orange-500/10',
    selectedBg: 'border-orange-500 bg-orange-500/15 dark:border-orange-400 dark:bg-orange-400/15',
    text: 'text-orange-600 dark:text-orange-400',
    dot: 'bg-orange-500 dark:bg-orange-400',
  },
  eval: {
    border: 'border-emerald-500/40 dark:border-emerald-400/40',
    bg: 'bg-emerald-500/5 dark:bg-emerald-400/5 hover:bg-emerald-500/10',
    selectedBg: 'border-emerald-500 bg-emerald-500/15 dark:border-emerald-400 dark:bg-emerald-400/15',
    text: 'text-emerald-600 dark:text-emerald-400',
    dot: 'bg-emerald-500 dark:bg-emerald-400',
  },
  meta: {
    border: 'border-slate-500/40 dark:border-slate-400/40',
    bg: 'bg-slate-500/5 dark:bg-slate-400/5 hover:bg-slate-500/10',
    selectedBg: 'border-slate-500 bg-slate-500/15 dark:border-slate-400 dark:bg-slate-400/15',
    text: 'text-slate-600 dark:text-slate-400',
    dot: 'bg-slate-500 dark:bg-slate-400',
  },
  analyze: {
    border: 'border-fuchsia-500/40 dark:border-fuchsia-400/40',
    bg: 'bg-fuchsia-500/5 dark:bg-fuchsia-400/5 hover:bg-fuchsia-500/10',
    selectedBg: 'border-fuchsia-500 bg-fuchsia-500/15 dark:border-fuchsia-400 dark:bg-fuchsia-400/15',
    text: 'text-fuchsia-600 dark:text-fuchsia-400',
    dot: 'bg-fuchsia-500 dark:bg-fuchsia-400',
  },
};

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toTimeString().slice(0, 8);
  } catch {
    return ts.slice(0, 8);
  }
}

/** Unwrap {result: ...} wrapper that MCP tools return */
function unwrapResult(response: unknown): unknown {
  if (typeof response === 'object' && response !== null && 'result' in response) {
    return (response as Record<string, unknown>).result;
  }
  return response;
}

function hasToolError(toolCall: KUAViToolCall): boolean {
  if (toolCall.has_error === true) return true;
  if (toolCall.has_error === false) return false;
  // Fallback for old traces
  const resp = toolCall.tool_response;
  const respStr = typeof resp === 'string' ? resp : JSON.stringify(resp ?? '');
  return respStr.includes('Error:') || respStr.includes('BUDGET EXCEEDED');
}

function formatDurationMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function getResponsePreview(toolCall: KUAViToolCall): { text: string; isError: boolean } {
  if (hasToolError(toolCall)) {
    // Use server-side summary if available (more accurate)
    if (toolCall.response_summary) return { text: toolCall.response_summary.slice(0, 40), isError: true };
    const resp = toolCall.tool_response;
    const respStr = typeof resp === 'string' ? resp : JSON.stringify(resp ?? '');
    const errorText = respStr.slice(0, 40) + (respStr.length > 40 ? '…' : '');
    return { text: errorText, isError: true };
  }

  // Use server-side response_summary if present (pre-computed, accurate)
  if (toolCall.response_summary) {
    return { text: toolCall.response_summary.slice(0, 44), isError: false };
  }

  const short = shortToolName(toolCall.tool_name);
  const resp = unwrapResult(toolCall.tool_response);

  // Search results
  if (['search_video', 'search_transcript', 'discriminative_vqa', 'get_scene_list'].some(s => short.includes(s))) {
    if (Array.isArray(resp)) return { text: `${resp.length} result${resp.length !== 1 ? 's' : ''}`, isError: false };
  }

  // Frame extraction
  if (['extract_frames', 'zoom_frames'].includes(short)) {
    if (Array.isArray(resp)) return { text: `${resp.length} frame${resp.length !== 1 ? 's' : ''}`, isError: false };
  }

  // Index
  if (short.includes('index_video')) {
    if (typeof resp === 'object' && resp !== null) {
      const status = (resp as Record<string, unknown>).status;
      return { text: status === 'indexed' ? 'indexed' : String(status ?? 'done'), isError: false };
    }
  }

  // Transcript
  if (short.includes('get_transcript') && typeof resp === 'string') {
    return { text: resp.slice(0, 40) + (resp.length > 40 ? '…' : ''), isError: false };
  }

  // Eval
  if (short === 'eval') {
    const stdout = typeof resp === 'object' && resp !== null
      ? String((resp as Record<string, unknown>).stdout ?? '')
      : '';
    if (stdout) return { text: stdout.split('\n')[0].slice(0, 40), isError: false };
  }

  // Default
  const str = typeof resp === 'string' ? resp : JSON.stringify(resp ?? '');
  return { text: str.slice(0, 40) + (str.length > 40 ? '…' : ''), isError: false };
}

function getInputPreview(toolCall: KUAViToolCall): string {
  const input = toolCall.tool_input;
  if (!input) return '';
  if (typeof input.query === 'string') return `"${input.query.slice(0, 40)}${input.query.length > 40 ? '…' : ''}"`;
  if (typeof input.question === 'string') return `"${input.question.slice(0, 40)}${input.question.length > 40 ? '…' : ''}"`;
  if (typeof input.start_time === 'number' && typeof input.end_time === 'number') {
    return `${input.start_time}s–${input.end_time}s`;
  }
  if (typeof input.video_path === 'string') return input.video_path.split('/').pop() ?? '';
  const keys = Object.keys(input);
  if (keys.length > 0) {
    const v = input[keys[0]];
    if (typeof v === 'string') return `${keys[0]}: ${v.slice(0, 30)}`;
    if (typeof v === 'number') return `${keys[0]}: ${v}`;
  }
  return '';
}

export function ToolCallTimeline({ events, selectedIndex, onSelectIndex }: ToolCallTimelineProps) {
  const selectedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }, [selectedIndex]);

  // Build display items: tool_calls get an index, agent events are markers
  let toolCallIdx = -1;
  const items = events.map((event) => {
    if (event.type === 'tool_call') {
      toolCallIdx++;
      return { event, toolCallIdx };
    }
    return { event, toolCallIdx: -1 };
  });

  const toolCallCount = items.filter((i) => i.toolCallIdx >= 0).length;

  return (
    <div className="border-b border-border bg-muted/30 flex-shrink-0">
      <div className="px-4 pt-3 pb-2 flex items-center gap-2">
        <div className="w-5 h-5 rounded bg-primary/10 flex items-center justify-center">
          <svg className="w-3 h-3 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
        </div>
        <span className="text-xs font-semibold text-foreground">Tool Call Timeline</span>
        <span className="text-xs text-muted-foreground">({toolCallCount} calls)</span>
        <div className="flex-1" />
        <span className="text-xs text-muted-foreground">← scroll →</span>
      </div>

      <ScrollArea className="w-full">
        <div className="flex items-center gap-1.5 px-3 pb-3">
          {items.map(({ event, toolCallIdx: tcIdx }, itemIdx) => {
            if (event.type === 'agent_start' || event.type === 'agent_stop') {
              const agentEvent = event as KUAViAgentEvent;
              const isStart = event.type === 'agent_start';
              return (
                <div
                  key={`agent-${itemIdx}`}
                  className="flex-shrink-0 flex flex-col items-center gap-0.5 px-1"
                  title={`${isStart ? 'Agent start' : 'Agent stop'}: ${agentEvent.agent_type}`}
                >
                  <div className={cn(
                    'w-1.5 h-8 rounded-full',
                    isStart ? 'bg-amber-400/60' : 'bg-amber-400/30'
                  )} />
                  <span className="text-[9px] text-amber-600 dark:text-amber-400 font-mono rotate-90 mt-1">
                    {isStart ? '▶' : '■'}
                  </span>
                </div>
              );
            }

            if (event.type === 'session_start' || event.type === 'session_end') {
              return (
                <div
                  key={`session-${itemIdx}`}
                  className="flex-shrink-0 w-1 h-10 bg-border/50 rounded-full"
                  title={event.type}
                />
              );
            }

            if (event.type === 'turn_start') {
              const turnEvent = event as KUAViTurnStartEvent;
              return (
                <div
                  key={`turn-${itemIdx}`}
                  className="flex-shrink-0 flex flex-col items-center gap-0.5 px-1"
                  title={`Agent turn ${turnEvent.turn} (${turnEvent.gap_seconds}s gap)`}
                >
                  <div className="w-px h-8 bg-primary/40" />
                  <span className="text-[9px] text-primary/70 font-mono whitespace-nowrap font-semibold">
                    T{turnEvent.turn}
                  </span>
                </div>
              );
            }

            // Skip non-renderable events (e.g., final_answer, metadata)
            if (event.type !== 'tool_call') return null;

            // tool_call
            const toolCall = event as KUAViToolCall;
            const isSelected = tcIdx === selectedIndex;
            const category = getToolCategory(toolCall.tool_name);
            const styles = categoryStyles[category];
            const short = shortToolName(toolCall.tool_name);
            const preview = getInputPreview(toolCall);
            const isError = hasToolError(toolCall);
            const responsePreview = getResponsePreview(toolCall);

            return (
              <div
                key={`tool-${itemIdx}`}
                ref={isSelected ? selectedRef : null}
                onClick={() => onSelectIndex(tcIdx)}
                className={cn(
                  'flex-shrink-0 w-48 cursor-pointer rounded-lg border transition-all duration-150',
                  isSelected ? styles.selectedBg : cn(styles.border, styles.bg),
                  isError && 'border-red-500/50'
                )}
              >
                <div className="p-2">
                  <div className="flex items-center gap-1.5 mb-1">
                    <div className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', isError ? 'bg-red-500' : styles.dot)} />
                    <span className={cn('text-xs font-semibold truncate', styles.text)}>
                      {short}
                    </span>
                    {isError && (
                      <span className="text-[9px] font-bold text-red-500 bg-red-500/10 px-1 rounded flex-shrink-0">
                        ERR
                      </span>
                    )}
                    {!isError && toolCall.duration_ms != null && (
                      <span className="text-[9px] text-muted-foreground bg-muted/60 px-1 rounded font-mono flex-shrink-0">
                        {formatDurationMs(toolCall.duration_ms)}
                      </span>
                    )}
                    <span className="text-[10px] text-muted-foreground ml-auto font-mono flex-shrink-0">
                      #{tcIdx + 1}
                    </span>
                  </div>
                  {preview && (
                    <p className="text-xs text-muted-foreground truncate leading-snug mb-0.5">
                      {preview}
                    </p>
                  )}
                  {responsePreview.text && (
                    <p className={cn(
                      'text-[10px] truncate leading-snug mb-0.5',
                      responsePreview.isError ? 'text-red-500' : 'text-muted-foreground/70'
                    )}>
                      → {responsePreview.text}
                    </p>
                  )}
                  <p className="text-[10px] text-muted-foreground/60 font-mono">
                    {formatTimestamp(toolCall.timestamp)}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}

// ─── TurnTimeline ───────────────────────────────────────────────────────────

interface TurnTimelineProps {
  turns: KUAViTurn[];
  selectedTurnIndex: number;
  onSelectTurnIndex: (index: number) => void;
  finalAnswer?: string | null;
}

function getTurnDuration(turn: KUAViTurn): number | null {
  const calls = turn.toolCalls;
  if (calls.length === 0) return null;
  try {
    const start = turn.reasoning
      ? new Date(turn.reasoning.timestamp).getTime()
      : new Date(calls[0].timestamp).getTime();
    const end = new Date(calls[calls.length - 1].timestamp).getTime();
    const durMs = end - start;
    return durMs >= 0 ? durMs / 1000 : null;
  } catch {
    return null;
  }
}

function formatTurnDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
}

export function TurnTimeline({ turns, selectedTurnIndex, onSelectTurnIndex, finalAnswer }: TurnTimelineProps) {
  const selectedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }, [selectedTurnIndex]);

  const isFinalTurn = (idx: number) => {
    // Only the very last turn can be "final" — when the model gave a final answer instead of calling tools
    return idx === turns.length - 1 && turns[idx].reasoning !== null && turns[idx].toolCalls.length === 0;
  };

  return (
    <div className="border-b border-border bg-muted/30 flex-shrink-0">
      {/* Section header */}
      <div className="px-4 pt-3 pb-2 flex items-center gap-2">
        <div className="w-5 h-5 rounded bg-primary/10 flex items-center justify-center">
          <svg className="w-3 h-3 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <span className="text-xs font-semibold text-foreground">Agent Tool Trajectory</span>
        <span className="text-[10px] text-muted-foreground">({turns.length} turn{turns.length !== 1 ? 's' : ''})</span>
        <div className="flex-1" />
        <span className="text-[10px] text-muted-foreground">← scroll →</span>
      </div>

      <ScrollArea className="w-full">
        <div className="flex gap-2 px-3 pb-3">
          {turns.map((turn, idx) => {
            const isSelected = idx === selectedTurnIndex;
            const isFinal = isFinalTurn(idx);
            const hasError = turn.toolCalls.some((tc) => tc.has_error);
            const reasoningSnippet = turn.reasoning
              ? turn.reasoning.text.slice(0, 60).replace(/\n/g, ' ')
              : null;
            const toolCount = turn.toolCalls.length;
            const totalTokens = turn.tokenUsage?.total_tokens ?? null;
            const duration = getTurnDuration(turn);

            // Collect unique tool categories for a mini pill display
            const toolCategories = [...new Set(turn.toolCalls.map((tc) => getToolCategory(tc.tool_name)))];

            return (
              <div
                key={`turn-${idx}`}
                ref={isSelected ? selectedRef : null}
                onClick={() => onSelectTurnIndex(idx)}
                className={cn(
                  'flex-shrink-0 w-72 cursor-pointer transition-all duration-150 rounded-lg border',
                  isSelected
                    ? 'border-primary bg-primary/10 shadow-md shadow-primary/15'
                    : isFinal
                      ? 'border-emerald-500/40 bg-emerald-500/5 hover:border-emerald-500/60 dark:border-emerald-400/40 dark:bg-emerald-400/5'
                      : hasError
                        ? 'border-red-500/40 bg-red-500/5 hover:border-red-500/60 dark:border-red-400/40 dark:bg-red-400/5'
                        : 'border-border hover:border-primary/40 hover:bg-muted/50'
                )}
              >
                <div className="p-2.5 flex items-start gap-3">
                  {/* Turn number circle */}
                  <div className={cn(
                    'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0',
                    isSelected
                      ? 'bg-primary text-primary-foreground'
                      : isFinal
                        ? 'bg-emerald-500 text-white dark:bg-emerald-400'
                        : hasError
                          ? 'bg-red-500 text-white dark:bg-red-400'
                          : 'bg-muted text-muted-foreground'
                  )}>
                    {idx + 1}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    {/* Top row: badges */}
                    <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                      {isFinal && (
                        <Badge className="bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-[10px] px-1 py-0 h-4">
                          FINAL
                        </Badge>
                      )}
                      {hasError && (
                        <Badge variant="destructive" className="text-[10px] px-1 py-0 h-4">ERR</Badge>
                      )}
                      {toolCount > 0 && (
                        <span className="text-xs text-cyan-600 dark:text-cyan-400">
                          {toolCount} tool{toolCount !== 1 ? 's' : ''}
                        </span>
                      )}
                      {turn.llmCalls.length > 0 && (
                        <span className="text-xs text-fuchsia-600 dark:text-fuchsia-400">
                          {turn.llmCalls.length} llm
                        </span>
                      )}
                      {turn.evalExecutions.length > 0 && (
                        <span className="text-xs text-emerald-600 dark:text-emerald-400">
                          {turn.evalExecutions.length} eval
                        </span>
                      )}
                      {/* Category dots */}
                      {toolCategories.slice(0, 3).map((cat) => (
                        <span
                          key={cat}
                          className={cn('w-1.5 h-1.5 rounded-full flex-shrink-0', categoryStyles[cat].dot)}
                          title={cat}
                        />
                      ))}
                      {duration !== null && (
                        <span className="text-xs text-muted-foreground ml-auto">
                          {formatTurnDuration(duration)}
                        </span>
                      )}
                    </div>

                    {/* Reasoning preview */}
                    <p className="text-[10px] text-muted-foreground truncate leading-relaxed">
                      {reasoningSnippet
                        ? `${reasoningSnippet}${(turn.reasoning?.text.length ?? 0) > 60 ? '...' : ''}`
                        : toolCount > 0
                          ? `${toolCount} tool call${toolCount !== 1 ? 's' : ''}`
                          : 'No reasoning'}
                    </p>

                    {/* Token usage */}
                    {totalTokens !== null && (
                      <div className="flex items-center gap-2 mt-1 text-[9px] font-mono text-muted-foreground/70">
                        {turn.tokenUsage && (
                          <span>
                            <span className="text-sky-600 dark:text-sky-400">
                              {(turn.tokenUsage.prompt_tokens / 1000).toFixed(1)}k
                            </span>
                            <span className="mx-0.5">→</span>
                            <span className="text-emerald-600 dark:text-emerald-400">
                              {(turn.tokenUsage.completion_tokens / 1000).toFixed(1)}k
                            </span>
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}

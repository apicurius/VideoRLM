'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { VideoFrameViewer, extractImageFrames } from '@/components/VideoFrameViewer';
import { SyntaxHighlight } from '@/components/SyntaxHighlight';
import { cn } from '@/lib/utils';
import { KUAViTurn } from '@/lib/parse-logs';
import { shortToolName } from '@/lib/parse-logs';
import { KUAViLLMCallEvent, KUAViEvalExecutionEvent } from '@/lib/types';

interface ConversationPanelProps {
  turns: KUAViTurn[];
  selectedTurn: number;
  systemPrompt: string | null;
  finalAnswer: string | null;
  question?: string | null;
  onToolCallClick?: (toolCallIndex: number) => void;
  logStem?: string;
}

function formatInputPreview(input: Record<string, unknown>): string {
  try {
    const str = JSON.stringify(input);
    if (str.length <= 120) return str;
    return str.slice(0, 120) + '…';
  } catch {
    return String(input);
  }
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

interface ShardResult {
  shard_index: number;
  start_time: number;
  end_time: number;
  answer: string;
  has_frames?: boolean;
  error?: string | null;
}

interface AnalyzeShardsData {
  question: string;
  shard_count: number;
  multimodal: boolean;
  results: ShardResult[];
}

function parseAnalyzeShardsResponse(response: unknown): AnalyzeShardsData | null {
  try {
    // Response format: [[textContent_string], parsed_json_dict]
    if (Array.isArray(response) && response.length >= 2) {
      const parsed = response[response.length - 1];
      if (typeof parsed === 'object' && parsed !== null && 'results' in parsed) {
        return parsed as AnalyzeShardsData;
      }
    }
    // Direct object
    if (typeof response === 'object' && response !== null && 'results' in response) {
      return response as AnalyzeShardsData;
    }
    // Unwrap {result: ...}
    if (typeof response === 'object' && response !== null && 'result' in response) {
      const inner = (response as Record<string, unknown>).result;
      if (typeof inner === 'object' && inner !== null && 'results' in inner) {
        return inner as AnalyzeShardsData;
      }
    }
  } catch { /* ignore parse errors */ }
  return null;
}

function isInformativeShard(answer: string): boolean {
  const lower = answer.toLowerCase();
  return !(
    lower.includes('no tables') ||
    lower.includes('not shown') ||
    lower.includes('no mention') ||
    lower.includes('not visible') ||
    lower.includes('no relevant') ||
    (lower.includes('not') && lower.includes('found'))
  );
}

function AnalyzeShardsCard({ data }: { data: AnalyzeShardsData }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-lg border border-fuchsia-500/20 bg-fuchsia-500/5 dark:bg-fuchsia-500/10 p-3 transition-all">
      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <div className="w-6 h-6 rounded-md bg-gradient-to-br from-fuchsia-500 to-purple-600 flex items-center justify-center flex-shrink-0 shadow-sm shadow-fuchsia-500/20">
          <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
          </svg>
        </div>
        <span className="font-semibold text-sm text-fuchsia-600 dark:text-fuchsia-400">
          Shard Analysis
        </span>
        <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 border-fuchsia-500/30 text-fuchsia-600 dark:text-fuchsia-400">
          {data.shard_count} shards
        </Badge>
        {data.multimodal && (
          <Badge className="bg-cyan-500/15 text-cyan-600 dark:text-cyan-400 border-cyan-500/30 text-[10px] px-1.5 py-0 h-4">
            Multimodal
          </Badge>
        )}
        <button
          onClick={() => setExpanded(v => !v)}
          className="text-[10px] text-fuchsia-600 dark:text-fuchsia-400 hover:underline ml-auto flex-shrink-0"
        >
          {expanded ? 'collapse' : 'expand'}
        </button>
      </div>
      {/* Question */}
      <div className="bg-background/60 rounded px-2 py-1 border border-fuchsia-500/10 dark:border-fuchsia-500/20 mb-2">
        <pre className="text-xs font-mono text-muted-foreground truncate overflow-hidden">
          {data.question}
        </pre>
      </div>
      {/* Per-shard results */}
      {expanded && (
        <div className="space-y-1.5 mt-2">
          {data.results.map((shard) => {
            const informative = isInformativeShard(shard.answer);
            const hasError = !!shard.error;
            return (
              <div
                key={shard.shard_index}
                className={cn(
                  'rounded-md border px-2.5 py-2 text-xs',
                  hasError
                    ? 'border-red-500/30 bg-red-500/5'
                    : informative
                      ? 'border-emerald-500/30 bg-emerald-500/5'
                      : 'border-border/50 bg-muted/20'
                )}
              >
                <div className="flex items-center gap-2 mb-1">
                  <Badge variant="outline" className="text-[10px] px-1 py-0 h-4 font-mono">
                    {formatTime(shard.start_time)} &ndash; {formatTime(shard.end_time)}
                  </Badge>
                  <span className="text-[10px] text-muted-foreground">
                    Shard {shard.shard_index + 1}
                  </span>
                  {shard.has_frames && (
                    <Badge className="bg-violet-500/15 text-violet-600 dark:text-violet-400 border-violet-500/30 text-[9px] px-1 py-0 h-3.5">
                      frames
                    </Badge>
                  )}
                  {hasError && (
                    <Badge className="bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30 text-[9px] px-1 py-0 h-3.5">
                      error
                    </Badge>
                  )}
                </div>
                <p className={cn(
                  'leading-relaxed',
                  hasError
                    ? 'text-red-600 dark:text-red-400'
                    : informative
                      ? 'text-foreground/90'
                      : 'text-muted-foreground/70'
                )}>
                  {hasError ? shard.error : shard.answer}
                </p>
              </div>
            );
          })}
        </div>
      )}
      {/* Collapsed summary */}
      {!expanded && (
        <p className="text-[10px] text-muted-foreground/60 mt-1">
          {data.results.filter(s => isInformativeShard(s.answer)).length} informative / {data.results.length} total shards
        </p>
      )}
    </div>
  );
}

function SystemPromptCard({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const preview = text.length > 300 ? text.slice(0, 300) + '…' : text;

  return (
    <div className="rounded-xl border border-violet-500/20 bg-violet-500/5 dark:bg-violet-500/10 p-4">
      {/* Header */}
      <div className="flex items-center gap-3 mb-3 pb-3 border-b border-violet-500/20">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg shadow-violet-500/20 flex-shrink-0">
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <span className="font-semibold text-sm text-violet-600 dark:text-violet-400">System Prompt</span>
          <p className="text-xs text-muted-foreground mt-0.5">Instructions &amp; context setup</p>
        </div>
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-xs text-violet-600 dark:text-violet-400 hover:underline flex-shrink-0"
        >
          {expanded ? 'collapse' : 'expand'}
        </button>
      </div>
      {/* Content */}
      <div className="bg-background/60 rounded-lg p-3 border border-violet-500/20">
        <pre className="whitespace-pre-wrap font-mono text-foreground/90 text-[12px] leading-relaxed overflow-x-auto">
          {expanded ? text : preview}
        </pre>
      </div>
    </div>
  );
}

function ToolCallCard({
  toolCall,
  globalIndex,
  onClick,
  logStem,
}: {
  toolCall: import('@/lib/types').KUAViToolCall;
  globalIndex: number;
  onClick?: () => void;
  logStem?: string;
}) {
  const name = shortToolName(toolCall.tool_name);
  const durationMs = toolCall.duration_ms;
  const hasError = toolCall.has_error;
  const frames = extractImageFrames(toolCall.tool_response);

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onClick?.(); }}
      className={cn(
        'w-full text-left rounded-lg border p-3 transition-all hover:shadow-md cursor-pointer',
        hasError
          ? 'border-red-500/30 bg-red-500/5 dark:bg-red-500/10 hover:bg-red-500/10'
          : 'border-green-500/20 bg-green-500/5 dark:bg-green-500/10 hover:bg-green-500/10'
      )}
    >
      <div className="flex items-center gap-2 mb-1.5">
        {/* Tool icon */}
        <div
          className={cn(
            'w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0',
            hasError
              ? 'bg-gradient-to-br from-red-500 to-rose-600 shadow-sm shadow-red-500/20'
              : 'bg-gradient-to-br from-green-500 to-emerald-600 shadow-sm shadow-green-500/20'
          )}
        >
          <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <span
          className={cn(
            'font-semibold text-sm',
            hasError ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'
          )}
        >
          {name}
        </span>
        <span className="text-xs text-muted-foreground ml-auto flex-shrink-0">
          #{globalIndex + 1}
        </span>
        {durationMs != null && (
          <Badge
            variant="outline"
            className={cn(
              'text-[10px] px-1.5 py-0 h-4 flex-shrink-0',
              hasError
                ? 'border-red-500/30 text-red-600 dark:text-red-400'
                : 'border-green-500/30 text-green-600 dark:text-green-400'
            )}
          >
            {durationMs < 1000 ? `${durationMs}ms` : `${(durationMs / 1000).toFixed(1)}s`}
          </Badge>
        )}
        {hasError && (
          <Badge className="bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30 text-[10px] px-1.5 py-0 h-4 flex-shrink-0">
            error
          </Badge>
        )}
      </div>
      {/* Input preview */}
      <div className="bg-background/60 rounded px-2 py-1 border border-green-500/10 dark:border-green-500/20">
        <pre className="text-xs font-mono text-muted-foreground truncate overflow-hidden">
          {formatInputPreview(toolCall.tool_input)}
        </pre>
      </div>
      {/* Inline frame thumbnails for frame extraction tools */}
      {frames.length > 0 && (
        <div className="mt-2">
          <VideoFrameViewer frames={frames} thumbSize={80} logStem={logStem} />
        </div>
      )}
      {/* Response summary if available */}
      {toolCall.response_summary && frames.length === 0 && (
        <p className="mt-1.5 text-xs text-muted-foreground/80 truncate">
          ↳ {toolCall.response_summary}
        </p>
      )}
    </div>
  );
}

function QuestionCard({ text }: { text: string }) {
  return (
    <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 dark:bg-emerald-500/10 p-4">
      <div className="flex items-center gap-3 mb-3 pb-3 border-b border-emerald-500/20">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center shadow-lg shadow-emerald-500/20 flex-shrink-0">
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <span className="font-semibold text-sm text-emerald-600 dark:text-emerald-400">User Question</span>
          <p className="text-xs text-muted-foreground mt-0.5">Input query for this analysis</p>
        </div>
      </div>
      <div className="bg-background/60 rounded-lg p-3 border border-emerald-500/20">
        <p className="text-sm font-medium text-foreground/90 leading-relaxed">{text}</p>
      </div>
    </div>
  );
}

function LLMCallCard({ llmCall, index, nested }: { llmCall: KUAViLLMCallEvent; index: number; nested?: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const hasError = llmCall.has_error;
  const hasFullResponse = !!llmCall.response_full && llmCall.response_full.length > (llmCall.response_summary?.length ?? 0);
  const displayResponse = expanded && hasFullResponse ? llmCall.response_full! : llmCall.response_summary;

  return (
    <div className={cn(
      'rounded-lg border p-3 transition-all',
      nested && 'ml-6 border-l-2 border-l-emerald-500/30',
      hasError
        ? 'border-red-500/30 bg-red-500/5 dark:bg-red-500/10'
        : 'border-fuchsia-500/20 bg-fuchsia-500/5 dark:bg-fuchsia-500/10'
    )}>
      <div className="flex items-center gap-2 mb-1.5">
        <div className={cn(
          'w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0',
          hasError
            ? 'bg-gradient-to-br from-red-500 to-rose-600 shadow-sm shadow-red-500/20'
            : 'bg-gradient-to-br from-fuchsia-500 to-purple-600 shadow-sm shadow-fuchsia-500/20'
        )}>
          <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </div>
        <span className={cn(
          'font-semibold text-sm',
          hasError ? 'text-red-600 dark:text-red-400' : 'text-fuchsia-600 dark:text-fuchsia-400'
        )}>
          LLM Call
        </span>
        <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 border-fuchsia-500/30 text-fuchsia-600 dark:text-fuchsia-400">
          {llmCall.model}
        </Badge>
        {(llmCall.num_frames != null && llmCall.num_frames > 0) ? (
          <Badge className="bg-violet-500/15 text-violet-600 dark:text-violet-400 border-violet-500/30 text-[10px] px-1.5 py-0 h-4">
            {llmCall.num_frames} frame{llmCall.num_frames !== 1 ? 's' : ''}
          </Badge>
        ) : llmCall.prompt_summary?.includes('[multimodal') ? (
          <Badge className="bg-cyan-500/15 text-cyan-600 dark:text-cyan-400 border-cyan-500/30 text-[10px] px-1.5 py-0 h-4">
            Multimodal
          </Badge>
        ) : null}
        <span className="text-xs text-muted-foreground ml-auto flex-shrink-0">#{index + 1}</span>
        {llmCall.duration_ms != null && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 flex-shrink-0">
            {llmCall.duration_ms < 1000 ? `${llmCall.duration_ms}ms` : `${(llmCall.duration_ms / 1000).toFixed(1)}s`}
          </Badge>
        )}
        {hasError && (
          <Badge className="bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30 text-[10px] px-1.5 py-0 h-4 flex-shrink-0">
            error
          </Badge>
        )}
      </div>
      {/* Token counts */}
      <div className="flex items-center gap-2 mb-1.5 text-[10px] font-mono text-muted-foreground/70">
        <span className="text-sky-600 dark:text-sky-400">{llmCall.prompt_tokens_approx.toLocaleString()} in</span>
        <span>→</span>
        <span className="text-emerald-600 dark:text-emerald-400">{llmCall.response_tokens_approx.toLocaleString()} out</span>
      </div>
      {/* Prompt summary */}
      <div className="bg-background/60 rounded px-2 py-1 border border-fuchsia-500/10 dark:border-fuchsia-500/20 mb-1.5">
        <pre className="text-xs font-mono text-muted-foreground truncate overflow-hidden">
          {llmCall.prompt_summary}
        </pre>
      </div>
      {/* Response — shows summary by default, full response when expanded */}
      {displayResponse && (
        <div className={cn(
          'rounded px-2 py-1 border',
          hasError ? 'bg-red-500/5 border-red-500/20' : 'bg-fuchsia-500/5 border-fuchsia-500/10'
        )}>
          <pre className={cn(
            'text-xs font-mono overflow-hidden',
            expanded ? 'whitespace-pre-wrap max-h-96 overflow-y-auto' : 'truncate',
            hasError ? 'text-red-600 dark:text-red-400' : 'text-fuchsia-600/80 dark:text-fuchsia-400/80'
          )}>
            {displayResponse}
          </pre>
          {(hasFullResponse || (llmCall.response_summary?.length ?? 0) > 120) && (
            <button
              onClick={() => setExpanded(v => !v)}
              className="text-[10px] text-fuchsia-600 dark:text-fuchsia-400 hover:underline mt-1"
            >
              {expanded ? 'collapse' : hasFullResponse ? 'expand full response' : 'expand'}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function EvalExecutionCard({ evalExec, index }: { evalExec: KUAViEvalExecutionEvent; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const hasError = evalExec.has_error;
  const codePreview = evalExec.code.length > 200 && !expanded ? evalExec.code.slice(0, 200) + '…' : evalExec.code;

  return (
    <div className={cn(
      'rounded-lg border p-3 transition-all',
      hasError
        ? 'border-red-500/30 bg-red-500/5 dark:bg-red-500/10'
        : 'border-emerald-500/20 bg-emerald-500/5 dark:bg-emerald-500/10'
    )}>
      <div className="flex items-center gap-2 mb-1.5">
        <div className={cn(
          'w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0',
          hasError
            ? 'bg-gradient-to-br from-red-500 to-rose-600 shadow-sm shadow-red-500/20'
            : 'bg-gradient-to-br from-emerald-500 to-teal-600 shadow-sm shadow-emerald-500/20'
        )}>
          <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
        </div>
        <span className={cn(
          'font-semibold text-sm',
          hasError ? 'text-red-600 dark:text-red-400' : 'text-emerald-600 dark:text-emerald-400'
        )}>
          Eval Execution
        </span>
        <span className="text-xs text-muted-foreground ml-auto flex-shrink-0">#{index + 1}</span>
        {evalExec.execution_time_ms != null && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 flex-shrink-0">
            {evalExec.execution_time_ms < 1000 ? `${evalExec.execution_time_ms}ms` : `${(evalExec.execution_time_ms / 1000).toFixed(1)}s`}
          </Badge>
        )}
        {hasError && (
          <Badge className="bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30 text-[10px] px-1.5 py-0 h-4 flex-shrink-0">
            error
          </Badge>
        )}
      </div>
      {/* Python code */}
      <div className="rounded overflow-hidden border border-emerald-500/10 dark:border-emerald-500/20 mb-1.5">
        <SyntaxHighlight code={codePreview} language="python" />
        {evalExec.code.length > 200 && (
          <button
            onClick={() => setExpanded(v => !v)}
            className="w-full text-center text-[10px] text-emerald-600 dark:text-emerald-400 hover:underline py-1 bg-muted/30"
          >
            {expanded ? 'collapse' : 'expand full code'}
          </button>
        )}
      </div>
      {/* Stdout */}
      {evalExec.stdout && (
        <div className={cn(
          'rounded px-2 py-1 border',
          hasError ? 'bg-red-500/5 border-red-500/20' : 'bg-emerald-500/5 border-emerald-500/10'
        )}>
          <pre className={cn(
            'text-xs font-mono whitespace-pre-wrap overflow-x-auto max-h-40 overflow-y-auto',
            hasError ? 'text-red-600 dark:text-red-400' : 'text-emerald-600/80 dark:text-emerald-400/80'
          )}>
            {evalExec.stdout}
          </pre>
        </div>
      )}
    </div>
  );
}

const REASONING_TRUNCATE_CHARS = 1500;

function ReasoningCard({ text, turnIndex }: { text: string; turnIndex: number }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = text.length > REASONING_TRUNCATE_CHARS;
  const displayText = isLong && !expanded ? text.slice(0, REASONING_TRUNCATE_CHARS) : text;

  return (
    <div className="rounded-xl border-2 border-sky-500/40 bg-gradient-to-br from-sky-500/10 to-indigo-500/10 p-4 shadow-lg shadow-sky-500/5">
      <div className="flex items-center gap-3 mb-3 pb-3 border-b border-sky-500/20">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20 flex-shrink-0">
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <span className="font-semibold text-sm text-sky-600 dark:text-sky-400">
            Model Response
          </span>
          <p className="text-[10px] text-muted-foreground mt-0.5">
            Turn {turnIndex + 1} reasoning
          </p>
        </div>
        <Badge variant="outline" className="text-[10px] border-sky-500/30 text-sky-600 dark:text-sky-400 flex-shrink-0">
          {text.length.toLocaleString()} chars
        </Badge>
      </div>
      <div className="bg-background/80 rounded-lg p-3 border border-sky-500/20">
        <pre className="whitespace-pre-wrap font-mono text-foreground text-[12px] leading-relaxed overflow-x-auto">
          {displayText}
        </pre>
        {isLong && (
          <button
            onClick={() => setExpanded(v => !v)}
            className="mt-2 text-xs text-sky-600 dark:text-sky-400 hover:underline"
          >
            {expanded ? 'collapse' : `expand (${(text.length - REASONING_TRUNCATE_CHARS).toLocaleString()} more chars)`}
          </button>
        )}
      </div>
    </div>
  );
}

export function ConversationPanel({
  turns,
  selectedTurn,
  systemPrompt,
  finalAnswer,
  question,
  onToolCallClick,
  logStem,
}: ConversationPanelProps) {
  const turn = turns[selectedTurn] ?? null;
  const isLastTurn = selectedTurn === turns.length - 1;

  // Compute global tool call offset for turns before selectedTurn
  const globalOffset = turns.slice(0, selectedTurn).reduce((sum, t) => sum + t.toolCalls.length, 0);

  return (
    <div className="h-full flex flex-col bg-background overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border flex items-center justify-between bg-muted/30 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center">
            <span className="text-white text-sm font-bold">◈</span>
          </div>
          <div>
            <h2 className="font-semibold text-sm">Conversation</h2>
            <p className="text-[11px] text-muted-foreground">
              Turn {selectedTurn + 1} of {turns.length}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          {turn && turn.toolCalls.length > 0 && (
            <Badge variant="secondary" className="text-[10px]">
              {turn.toolCalls.length} tool{turn.toolCalls.length !== 1 ? 's' : ''}
            </Badge>
          )}
          {turn && turn.llmCalls.length > 0 && (
            <Badge variant="outline" className="text-[10px] border-fuchsia-500/30 text-fuchsia-600 dark:text-fuchsia-400">
              {turn.llmCalls.length} llm
            </Badge>
          )}
          {turn && turn.evalExecutions.length > 0 && (
            <Badge variant="outline" className="text-[10px] border-emerald-500/30 text-emerald-600 dark:text-emerald-400">
              {turn.evalExecutions.length} eval
            </Badge>
          )}
          {turn?.tokenUsage && (
            <Badge variant="outline" className="text-[10px] border-sky-500/30 text-sky-600 dark:text-sky-400">
              {turn.tokenUsage.total_tokens.toLocaleString()} tok
            </Badge>
          )}
          {isLastTurn && finalAnswer && (
            <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-[10px]">
              ✓ Answer
            </Badge>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-4 space-y-4">
            {/* System Prompt — only on turn 0 */}
            {selectedTurn === 0 && systemPrompt && (
              <SystemPromptCard text={systemPrompt} />
            )}

            {/* Question — only on turn 0 */}
            {selectedTurn === 0 && question && (
              <QuestionCard text={question} />
            )}

            {/* Reasoning */}
            {turn?.reasoning ? (
              <ReasoningCard text={turn.reasoning.text} turnIndex={selectedTurn} />
            ) : turn && !turn.reasoning && turn.toolCalls.length === 0 ? (
              <div className="rounded-xl border border-dashed border-border p-6 flex items-center justify-center">
                <p className="text-sm text-muted-foreground/60 italic">No reasoning captured for this turn</p>
              </div>
            ) : null}

            {/* Tool call summaries */}
            {turn && turn.toolCalls.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs uppercase tracking-wider text-muted-foreground font-medium px-1">
                  Tool Calls ({turn.toolCalls.length})
                </p>
                {turn.toolCalls.map((tc, i) => {
                  const shardsData = shortToolName(tc.tool_name).includes('analyze_shards')
                    ? parseAnalyzeShardsResponse(tc.tool_response)
                    : null;
                  if (shardsData) {
                    return (
                      <div
                        key={`${tc.timestamp}-${i}`}
                        role="button"
                        tabIndex={0}
                        onClick={onToolCallClick ? () => onToolCallClick(globalOffset + i) : undefined}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onToolCallClick?.(globalOffset + i); }}
                        className="cursor-pointer"
                      >
                        <AnalyzeShardsCard data={shardsData} />
                      </div>
                    );
                  }
                  return (
                    <ToolCallCard
                      key={`${tc.timestamp}-${i}`}
                      toolCall={tc}
                      globalIndex={globalOffset + i}
                      onClick={onToolCallClick ? () => onToolCallClick(globalOffset + i) : undefined}
                      logStem={logStem}
                    />
                  );
                })}
              </div>
            )}

            {/* Eval Executions with linked LLM Calls */}
            {turn && turn.evalExecutions.length > 0 && (() => {
              // Build a set of eval_ids that have linked LLM calls
              const linkedLLMCallIds = new Set(
                turn.llmCalls.filter(lc => lc.eval_id).map(lc => lc.eval_id)
              );
              return (
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-wider text-muted-foreground font-medium px-1">
                    Eval Executions ({turn.evalExecutions.length})
                  </p>
                  {turn.evalExecutions.map((ee, i) => {
                    // Find LLM calls linked to this eval execution via eval_id
                    const linkedCalls = ee.eval_id
                      ? turn.llmCalls.filter(lc => lc.eval_id === ee.eval_id)
                      : [];
                    return (
                      <div key={`${ee.timestamp}-${i}`}>
                        <EvalExecutionCard evalExec={ee} index={i} />
                        {linkedCalls.length > 0 && (
                          <div className="space-y-1.5 mt-1.5">
                            {linkedCalls.map((lc, li) => (
                              <LLMCallCard
                                key={`${lc.timestamp}-${li}`}
                                llmCall={lc}
                                index={turn.llmCalls.indexOf(lc)}
                                nested
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              );
            })()}

            {/* Unlinked LLM Calls (not associated with any eval execution) */}
            {turn && (() => {
              const linkedEvalIds = new Set(
                turn.evalExecutions.filter(ee => ee.eval_id).map(ee => ee.eval_id)
              );
              const unlinkedCalls = turn.llmCalls.filter(
                lc => !lc.eval_id || !linkedEvalIds.has(lc.eval_id)
              );
              if (unlinkedCalls.length === 0) return null;
              return (
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-wider text-muted-foreground font-medium px-1">
                    LLM Calls ({unlinkedCalls.length})
                  </p>
                  {unlinkedCalls.map((lc, i) => (
                    <LLMCallCard key={`${lc.timestamp}-${i}`} llmCall={lc} index={turn.llmCalls.indexOf(lc)} />
                  ))}
                </div>
              );
            })()}

            {/* Final answer — only on last turn */}
            {isLastTurn && finalAnswer && (
              <div className="rounded-xl border-2 border-emerald-500/50 bg-gradient-to-br from-emerald-500/15 to-green-500/15 p-4 shadow-lg shadow-emerald-500/10">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center shadow-lg shadow-emerald-500/30 flex-shrink-0">
                    <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div>
                    <span className="font-bold text-emerald-600 dark:text-emerald-400 text-base">
                      Final Answer
                    </span>
                    <p className="text-[10px] text-muted-foreground">
                      Task completed successfully
                    </p>
                  </div>
                </div>
                <div className="bg-background/80 rounded-lg p-4 border border-emerald-500/30">
                  <div className="text-[15px] font-medium text-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown
                      components={{
                        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                        strong: ({ children }) => <strong className="font-bold">{children}</strong>,
                        ul: ({ children }) => <ul className="list-disc pl-5 mb-2">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal pl-5 mb-2">{children}</ol>,
                        li: ({ children }) => <li className="mb-0.5">{children}</li>,
                      }}
                    >
                      {finalAnswer}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            )}

            {/* Bottom padding */}
            <div className="h-4" />
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}

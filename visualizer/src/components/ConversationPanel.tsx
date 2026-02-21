'use client';

import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { VideoFrameViewer, extractImageFrames } from '@/components/VideoFrameViewer';
import { cn } from '@/lib/utils';
import { KUAViTurn } from '@/lib/parse-logs';
import { shortToolName } from '@/lib/parse-logs';

interface ConversationPanelProps {
  turns: KUAViTurn[];
  selectedTurn: number;
  systemPrompt: string | null;
  finalAnswer: string | null;
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

export function ConversationPanel({
  turns,
  selectedTurn,
  systemPrompt,
  finalAnswer,
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

            {/* Reasoning */}
            {turn?.reasoning ? (
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
                      Turn {selectedTurn + 1} reasoning
                    </p>
                  </div>
                  <Badge variant="outline" className="text-[10px] border-sky-500/30 text-sky-600 dark:text-sky-400 flex-shrink-0">
                    {turn.reasoning.text.length.toLocaleString()} chars
                  </Badge>
                </div>
                <div className="bg-background/80 rounded-lg p-3 border border-sky-500/20">
                  <pre className="whitespace-pre-wrap font-mono text-foreground text-[12px] leading-relaxed overflow-x-auto">
                    {turn.reasoning.text}
                  </pre>
                </div>
              </div>
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
                {turn.toolCalls.map((tc, i) => (
                  <ToolCallCard
                    key={`${tc.timestamp}-${i}`}
                    toolCall={tc}
                    globalIndex={globalOffset + i}
                    onClick={onToolCallClick ? () => onToolCallClick(globalOffset + i) : undefined}
                    logStem={logStem}
                  />
                ))}
              </div>
            )}

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
                  <p className="text-[15px] font-medium text-foreground leading-relaxed">
                    {finalAnswer}
                  </p>
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

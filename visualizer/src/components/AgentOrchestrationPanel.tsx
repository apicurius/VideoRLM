'use client';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import { KUAViEvent, KUAViAgentEvent, KUAViToolCall, KUAViReasoningEvent, KUAViSystemPromptEvent, KUAViFinalAnswerEvent } from '@/lib/types';
import { shortToolName } from '@/lib/parse-logs';

interface AgentOrchestrationPanelProps {
  events: KUAViEvent[];
  toolBreakdown?: Record<string, number>;
}

interface AgentInfo {
  agent_id: string;
  agent_type: string;
  startTime: string;
  stopTime: string | null;
  durationMs: number | null;
}

function formatTimestamp(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString();
  } catch {
    return ts;
  }
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}

function getToolCategoryColor(name: string): string {
  const short = shortToolName(name);
  if (['search_video', 'search_transcript', 'get_transcript', 'discriminative_vqa'].includes(short)) {
    return 'bg-sky-500/15 text-sky-600 dark:text-sky-400 border-sky-500/30';
  }
  if (['extract_frames', 'zoom_frames'].includes(short)) {
    return 'bg-violet-500/15 text-violet-600 dark:text-violet-400 border-violet-500/30';
  }
  if (['crop_frame', 'diff_frames', 'blend_frames', 'threshold_frame', 'frame_info'].includes(short)) {
    return 'bg-orange-500/15 text-orange-600 dark:text-orange-400 border-orange-500/30';
  }
  if (short === 'eval') {
    return 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30';
  }
  if (short === 'analyze_shards') {
    return 'bg-fuchsia-500/15 text-fuchsia-600 dark:text-fuchsia-400 border-fuchsia-500/30';
  }
  return 'bg-slate-500/15 text-slate-600 dark:text-slate-400 border-slate-500/30';
}

/** Summarize a tool response into a brief string for the event log */
function summarizeToolResult(tc: KUAViToolCall): string {
  if (tc.has_error) return 'Error';
  const resp = tc.tool_response;
  const short = shortToolName(tc.tool_name);

  // index_video — "indexed"
  if (short === 'index_video') return 'indexed';

  // Frame extraction tools — count frames
  if (short === 'extract_frames' || short === 'zoom_frames') {
    const unwrapped = typeof resp === 'object' && resp !== null && 'result' in resp
      ? (resp as Record<string, unknown>).result
      : resp;
    if (Array.isArray(unwrapped)) return `${unwrapped.length} frames`;
    return '1 frame';
  }

  // Search tools — count results
  if (short === 'search_video' || short === 'search_transcript') {
    const unwrapped = typeof resp === 'object' && resp !== null && 'result' in resp
      ? (resp as Record<string, unknown>).result
      : resp;
    if (typeof unwrapped === 'object' && unwrapped !== null && 'results' in (unwrapped as Record<string, unknown>)) {
      const results = (unwrapped as Record<string, unknown>).results;
      if (Array.isArray(results)) return `${results.length} results`;
    }
    if (Array.isArray(unwrapped)) return `${unwrapped.length} results`;
    return 'searched';
  }

  // Pixel tools
  if (['crop_frame', 'diff_frames', 'blend_frames', 'threshold_frame', 'frame_info'].includes(short)) {
    return 'ok';
  }

  // Scene list
  if (short === 'get_scene_list') {
    const unwrapped = typeof resp === 'object' && resp !== null && 'result' in resp
      ? (resp as Record<string, unknown>).result
      : resp;
    if (typeof unwrapped === 'object' && unwrapped !== null && 'scenes' in (unwrapped as Record<string, unknown>)) {
      const scenes = (unwrapped as Record<string, unknown>).scenes;
      if (Array.isArray(scenes)) return `${scenes.length} scenes`;
    }
    return 'ok';
  }

  // eval
  if (short === 'eval') return 'executed';

  // analyze_shards
  if (short === 'analyze_shards') return 'analyzed';

  // Fallback: check for error in response string
  const respStr = typeof resp === 'string' ? resp : JSON.stringify(resp ?? '');
  if (respStr.includes('Error:')) return 'error';

  return 'ok';
}

export function AgentOrchestrationPanel({ events, toolBreakdown }: AgentOrchestrationPanelProps) {
  // Group agent events by agent_id
  const agentMap = new Map<string, AgentInfo>();

  for (const event of events) {
    if (event.type === 'agent_start') {
      const e = event as KUAViAgentEvent;
      agentMap.set(e.agent_id, {
        agent_id: e.agent_id,
        agent_type: e.agent_type,
        startTime: e.timestamp,
        stopTime: null,
        durationMs: null,
      });
    } else if (event.type === 'agent_stop') {
      const e = event as KUAViAgentEvent;
      const existing = agentMap.get(e.agent_id);
      if (existing) {
        existing.stopTime = e.timestamp;
        try {
          existing.durationMs = new Date(e.timestamp).getTime() - new Date(existing.startTime).getTime();
        } catch {
          existing.durationMs = null;
        }
      } else {
        agentMap.set(e.agent_id, {
          agent_id: e.agent_id,
          agent_type: e.agent_type,
          startTime: e.timestamp,
          stopTime: e.timestamp,
          durationMs: 0,
        });
      }
    }
  }

  const agents = Array.from(agentMap.values());

  // Sort tool breakdown by count descending
  const sortedTools = toolBreakdown
    ? Object.entries(toolBreakdown).sort(([, a], [, b]) => b - a)
    : [];

  const hasAgents = agents.length > 0;
  const hasTools = sortedTools.length > 0;

  // Build chronological tool call summary for the Summary tab
  const toolCalls = events.filter((e): e is KUAViToolCall => e.type === 'tool_call');

  // Build conversation events for the Conversation tab (reasoning + tool calls + final answer interleaved)
  const conversationEvents = events.filter(
    (e): e is KUAViReasoningEvent | KUAViToolCall | KUAViFinalAnswerEvent | KUAViSystemPromptEvent =>
      e.type === 'reasoning' || e.type === 'tool_call' || e.type === 'final_answer' || e.type === 'system_prompt'
  );
  const hasConversation = conversationEvents.some((e) => e.type === 'reasoning' || e.type === 'system_prompt');

  return (
    <div className="h-full flex flex-col">
      <Tabs defaultValue={hasConversation ? 'conversation' : 'summary'} className="h-full flex flex-col gap-0">
        {/* Tab header */}
        <div className="px-4 py-2 border-b border-border bg-card/60 flex-shrink-0">
          <TabsList className="h-7">
            <TabsTrigger value="summary" className="text-xs px-3 py-1">Summary</TabsTrigger>
            {hasConversation && (
              <TabsTrigger value="conversation" className="text-xs px-3 py-1">Conversation</TabsTrigger>
            )}
            <TabsTrigger value="orchestration" className="text-xs px-3 py-1">Orchestration</TabsTrigger>
          </TabsList>
        </div>

        {/* Summary tab */}
        <TabsContent value="summary" className="flex-1 min-h-0">
          <ScrollArea className="h-full">
            <div className="p-4 space-y-1">
              <p className="text-xs uppercase tracking-wider text-muted-foreground font-medium mb-2">Event Log</p>
              {toolCalls.length === 0 ? (
                <div className="flex items-center justify-center py-8 text-center">
                  <p className="text-sm text-muted-foreground">No tool calls recorded</p>
                </div>
              ) : (
                toolCalls.map((tc, idx) => {
                  const short = shortToolName(tc.tool_name);
                  const summary = summarizeToolResult(tc);
                  const isError = tc.has_error || summary === 'Error' || summary === 'error';
                  const timeStr = formatTimestamp(tc.timestamp);
                  const durationStr = tc.duration_ms != null ? formatDuration(tc.duration_ms) : null;

                  return (
                    <div
                      key={idx}
                      className="flex items-center gap-2 py-1 px-2 rounded hover:bg-muted/50 group"
                    >
                      <span className="text-[10px] text-muted-foreground font-mono w-16 flex-shrink-0">{timeStr}</span>
                      <span className={cn(
                        'text-xs font-mono font-medium truncate min-w-0',
                        getToolCategoryColor(tc.tool_name).replace(/bg-\S+/, '').replace(/border-\S+/, '').trim()
                      )}>
                        {short}
                      </span>
                      <span className={cn(
                        'text-xs flex-shrink-0',
                        isError ? 'text-red-500' : 'text-muted-foreground'
                      )}>
                        {summary}
                      </span>
                      {durationStr && (
                        <span className="text-[10px] text-muted-foreground/60 font-mono ml-auto flex-shrink-0">
                          {durationStr}
                        </span>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </ScrollArea>
        </TabsContent>

        {/* Conversation tab — reasoning + tool calls interleaved */}
        {hasConversation && (
          <TabsContent value="conversation" className="flex-1 min-h-0">
            <ScrollArea className="h-full">
              <div className="p-4 space-y-3">
                {conversationEvents.map((event, idx) => {
                  if (event.type === 'system_prompt') {
                    return (
                      <details key={idx} className="group">
                        <summary className="cursor-pointer text-xs uppercase tracking-wider text-muted-foreground font-medium hover:text-foreground transition-colors">
                          System Prompt
                        </summary>
                        <div className="mt-2 p-3 rounded-md bg-muted/40 border border-border">
                          <pre className="text-xs text-foreground/80 whitespace-pre-wrap font-mono leading-relaxed max-h-60 overflow-y-auto">
                            {event.text}
                          </pre>
                        </div>
                      </details>
                    );
                  }

                  if (event.type === 'reasoning') {
                    return (
                      <div key={idx} className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-muted-foreground font-mono">{formatTimestamp(event.timestamp)}</span>
                          <Badge className="text-[10px] px-1.5 py-0 h-4 bg-blue-500/15 text-blue-600 dark:text-blue-400 border-blue-500/30">
                            reasoning
                          </Badge>
                          {event.token_usage?.total_tokens && (
                            <span className="text-[10px] text-muted-foreground/60 font-mono ml-auto">
                              {event.token_usage.total_tokens.toLocaleString()} tokens
                            </span>
                          )}
                        </div>
                        <div className="pl-2 border-l-2 border-blue-500/30">
                          <p className="text-xs text-foreground/90 whitespace-pre-wrap leading-relaxed">
                            {event.text}
                          </p>
                        </div>
                      </div>
                    );
                  }

                  if (event.type === 'tool_call') {
                    const short = shortToolName(event.tool_name);
                    const summary = summarizeToolResult(event);
                    const isError = event.has_error || summary === 'Error' || summary === 'error';
                    const durationStr = event.duration_ms != null ? formatDuration(event.duration_ms) : null;

                    return (
                      <div key={idx} className="flex items-center gap-2 py-0.5 px-2 rounded bg-muted/30">
                        <span className="text-[10px] text-muted-foreground font-mono w-16 flex-shrink-0">{formatTimestamp(event.timestamp)}</span>
                        <Badge className={cn('text-[10px] px-1.5 py-0 h-4', getToolCategoryColor(event.tool_name))}>
                          {short}
                        </Badge>
                        <span className={cn('text-xs flex-shrink-0', isError ? 'text-red-500' : 'text-muted-foreground')}>
                          {summary}
                        </span>
                        {durationStr && (
                          <span className="text-[10px] text-muted-foreground/60 font-mono ml-auto flex-shrink-0">
                            {durationStr}
                          </span>
                        )}
                      </div>
                    );
                  }

                  if (event.type === 'final_answer') {
                    return (
                      <div key={idx} className="space-y-1 mt-2">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-muted-foreground font-mono">{formatTimestamp(event.timestamp)}</span>
                          <Badge className="text-[10px] px-1.5 py-0 h-4 bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30">
                            final answer
                          </Badge>
                        </div>
                        <div className="pl-2 border-l-2 border-emerald-500/30">
                          <p className="text-xs text-foreground whitespace-pre-wrap leading-relaxed">
                            {event.text.length > 500 ? event.text.slice(0, 500) + '...' : event.text}
                          </p>
                        </div>
                      </div>
                    );
                  }

                  return null;
                })}
              </div>
            </ScrollArea>
          </TabsContent>
        )}

        {/* Orchestration tab (existing content) */}
        <TabsContent value="orchestration" className="flex-1 min-h-0">
          <ScrollArea className="h-full">
            <div className="p-4 space-y-4">
              {/* Agent cards */}
              {!hasAgents && (
                <div className="flex items-center justify-center py-8 text-center">
                  <div className="space-y-1">
                    <p className="text-2xl">◇</p>
                    <p className="text-sm text-muted-foreground">No agent orchestration events</p>
                  </div>
                </div>
              )}

              {hasAgents && (
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-wider text-muted-foreground font-medium">Agents</p>
                  {agents.map((agent) => {
                    const isRunning = agent.stopTime === null;
                    return (
                      <Card
                        key={agent.agent_id}
                        className={cn(
                          'border',
                          isRunning
                            ? 'border-amber-500/40 bg-amber-500/5'
                            : 'border-border bg-card'
                        )}
                      >
                        <CardContent className="p-3 space-y-2">
                          <div className="flex items-start justify-between gap-2">
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-1.5 flex-wrap">
                                <span className="text-xs font-semibold font-mono text-foreground truncate">
                                  {agent.agent_type || 'unknown'}
                                </span>
                                <Badge
                                  className={cn(
                                    'text-[10px] px-1.5 py-0 h-4',
                                    isRunning
                                      ? 'bg-amber-500/20 text-amber-600 dark:text-amber-400 border-amber-500/30'
                                      : 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border-emerald-500/30'
                                  )}
                                >
                                  {isRunning ? 'running' : 'completed'}
                                </Badge>
                              </div>
                              <p className="text-[10px] text-muted-foreground font-mono mt-0.5 truncate" title={agent.agent_id}>
                                {agent.agent_id.slice(0, 16)}…
                              </p>
                            </div>
                            {agent.durationMs !== null && (
                              <span className="text-xs text-muted-foreground font-mono flex-shrink-0">
                                {formatDuration(agent.durationMs)}
                              </span>
                            )}
                          </div>

                          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                            <div>
                              <p className="text-[10px] text-muted-foreground/70 uppercase tracking-wide">Start</p>
                              <p className="text-xs font-mono text-foreground/80">{formatTimestamp(agent.startTime)}</p>
                            </div>
                            {agent.stopTime && (
                              <div>
                                <p className="text-[10px] text-muted-foreground/70 uppercase tracking-wide">Stop</p>
                                <p className="text-xs font-mono text-foreground/80">{formatTimestamp(agent.stopTime)}</p>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              )}

              {/* Tool breakdown */}
              {hasTools && (
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-wider text-muted-foreground font-medium">Tool Breakdown</p>
                  <div className="flex flex-wrap gap-1.5">
                    {sortedTools.map(([tool, count]) => (
                      <div
                        key={tool}
                        className={cn(
                          'flex items-center gap-1 rounded-md border px-2 py-1',
                          getToolCategoryColor(tool)
                        )}
                      >
                        <span className="text-xs font-mono font-medium">{shortToolName(tool)}</span>
                        <span className="text-[10px] opacity-70">×{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  );
}

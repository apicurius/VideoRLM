'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { StatsCard } from '@/components/StatsCard';
import { ThemeToggle } from '@/components/ThemeToggle';
import { TurnTimeline } from '@/components/ToolCallTimeline';
import { ToolCallPanel } from '@/components/ToolCallPanel';
import { AgentOrchestrationPanel } from '@/components/AgentOrchestrationPanel';
import { ConversationPanel } from '@/components/ConversationPanel';
import { KUAViLogFile, KUAViToolCall, KUAViSystemPromptEvent } from '@/lib/types';
import { groupEventsIntoTurns, shortToolName } from '@/lib/parse-logs';

interface KUAViLogViewerProps {
  logFile: KUAViLogFile;
  onBack: () => void;
}

export function KUAViLogViewer({ logFile, onBack }: KUAViLogViewerProps) {
  const { events, metadata, fileName, logStem } = logFile;

  // Group events into turns (memoize to avoid unstable references)
  const turns = useMemo(() => groupEventsIntoTurns(events), [events]);

  const [selectedTurnIndex, setSelectedTurnIndex] = useState(0);
  const [selectedToolCallIndex, setSelectedToolCallIndex] = useState(0);
  const [rightTab, setRightTab] = useState<'tool' | 'orchestration'>('tool');

  // All tool calls flattened (for ToolCallPanel lookup)
  const allToolCalls = events.filter((e): e is KUAViToolCall => e.type === 'tool_call');
  const selectedToolCall = allToolCalls[selectedToolCallIndex] ?? null;

  // System prompt
  const systemPromptEvent = events.find((e): e is KUAViSystemPromptEvent => e.type === 'system_prompt');
  const systemPrompt = systemPromptEvent?.text ?? null;

  const goToPrevious = useCallback(() => {
    setSelectedTurnIndex((prev) => Math.max(0, prev - 1));
  }, []);

  const goToNext = useCallback(() => {
    setSelectedTurnIndex((prev) => Math.min(turns.length - 1, prev + 1));
  }, [turns.length]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        goToPrevious();
      } else if (e.key === 'ArrowRight') {
        goToNext();
      } else if (e.key === 'Escape') {
        onBack();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goToPrevious, goToNext, onBack]);

  // Auto-select first tool call of current turn when turn changes
  useEffect(() => {
    const turn = turns[selectedTurnIndex];
    if (turn && turn.toolCalls.length > 0) {
      const globalOffset = turns.slice(0, selectedTurnIndex).reduce((sum, t) => sum + t.toolCalls.length, 0);
      setSelectedToolCallIndex(globalOffset);
      setRightTab('tool');
    }
  }, [selectedTurnIndex, turns]);

  // When a tool call is clicked in ConversationPanel, switch to Tool Detail tab
  const handleToolCallClick = useCallback((toolCallIndex: number) => {
    setSelectedToolCallIndex(toolCallIndex);
    setRightTab('tool');
  }, []);

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  };

  // Extract video config: prefer metadata event, fall back to index_video response
  const metadataEvent = events.find((e) => e.type === 'metadata') as (Record<string, unknown> | undefined);
  const videoConfig = (() => {
    const parts: string[] = [];

    if (metadataEvent) {
      if (metadataEvent.fps != null) parts.push(`${metadataEvent.fps} fps`);
      if (metadataEvent.num_segments != null) parts.push(`${metadataEvent.num_segments} segments`);
      if (metadataEvent.duration != null) {
        const dur = metadataEvent.duration as number;
        parts.push(`${dur.toFixed(1)}s`);
      }
    } else {
      const indexCall = allToolCalls.find((tc) => shortToolName(tc.tool_name) === 'index_video');
      if (indexCall) {
        const resp = indexCall.tool_response;
        const unwrapped = typeof resp === 'object' && resp !== null && 'result' in resp
          ? (resp as Record<string, unknown>).result
          : resp;
        if (typeof unwrapped === 'object' && unwrapped !== null) {
          const r = unwrapped as Record<string, unknown>;
          if (r.fps != null) parts.push(`${r.fps} fps`);
          if (r.total_segments != null || r.segments != null) parts.push(`${r.total_segments ?? r.segments} segments`);
          if (r.duration != null) parts.push(`${typeof r.duration === 'number' ? r.duration.toFixed(1) : r.duration}s`);
          if (r.resolution) parts.push(String(r.resolution));
          if (r.width && r.height) parts.push(`${r.width}×${r.height}`);
        }
      }
    }
    return parts.length > 0 ? parts.join(' · ') : null;
  })();

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm flex-shrink-0">
        <div className="px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={onBack}
                className="text-muted-foreground hover:text-foreground"
              >
                ← Back
              </Button>
              <div className="h-5 w-px bg-border" />
              <div>
                <h1 className="font-semibold flex items-center gap-2 text-base">
                  <span className="text-primary">◈</span>
                  {fileName}
                  <Badge className="bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 border-cyan-500/30 text-xs px-1.5 py-0 h-4 ml-1">
                    KUAVi
                  </Badge>
                </h1>
                <p className="text-xs text-muted-foreground font-mono mt-0.5">
                  {metadata.model ?? 'Unknown model'}
                  {metadata.videoPath && (
                    <>
                      {' • '}
                      <span className="text-violet-600 dark:text-violet-400">
                        {metadata.videoPath.split('/').pop()}
                      </span>
                    </>
                  )}
                </p>
                {videoConfig && (
                  <p className="text-xs text-muted-foreground/70 font-mono mt-0.5">
                    {videoConfig}
                  </p>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              {metadata.hasErrors && (
                <Badge variant="destructive" className="text-xs">Has Errors</Badge>
              )}
              {metadata.finalAnswer && (
                <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white text-xs">
                  Completed
                </Badge>
              )}
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      {/* Question & Answer + Stats Row (combined, matching RLM layout) */}
      <div className="border-b border-border bg-muted/30 px-6 py-4 flex-shrink-0">
        <div className="flex gap-6">
          {/* Question & Answer Summary */}
          <Card className="flex-1 bg-gradient-to-r from-primary/5 to-accent/5 border-primary/20">
            <CardContent className="p-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-xs uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Context / Question
                  </p>
                  <p className="text-sm font-medium line-clamp-2">
                    {metadata.question || 'No question captured'}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Final Answer
                  </p>
                  <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400 line-clamp-2">
                    {metadata.finalAnswer || 'Not yet completed'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <div className="flex gap-2">
            <StatsCard
              label="Turns"
              value={turns.length}
              icon="↻"
              variant="cyan"
            />
            <StatsCard
              label="Tool Calls"
              value={metadata.totalToolCalls}
              icon="◎"
              variant="green"
            />
            <StatsCard
              label="Searches"
              value={metadata.totalSearches}
              icon="⌕"
              variant="green"
            />
            <StatsCard
              label="Duration"
              value={formatDuration(metadata.sessionDuration)}
              icon="⏱"
              variant="yellow"
            />
            {metadata.hasFrames && (
              <StatsCard
                label="Frames"
                value={metadata.totalFramesExtracted}
                icon="▦"
                variant="magenta"
              />
            )}
          </div>
        </div>
      </div>

      {/* Turn Timeline */}
      {turns.length > 0 ? (
        <TurnTimeline
          turns={turns}
          selectedTurnIndex={selectedTurnIndex}
          onSelectTurnIndex={setSelectedTurnIndex}
        />
      ) : null}

      {/* Main resizable content */}
      <div className="flex-1 min-h-0">
        <ResizablePanelGroup orientation="horizontal">
          {/* Left panel: ConversationPanel */}
          <ResizablePanel defaultSize={50} minSize={25} maxSize={75}>
            <div className="h-full border-r border-border">
              <ConversationPanel
                turns={turns}
                selectedTurn={selectedTurnIndex}
                systemPrompt={systemPrompt}
                finalAnswer={metadata.finalAnswer ?? null}
                onToolCallClick={handleToolCallClick}
                logStem={logStem}
              />
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle className="bg-border hover:bg-primary/30 transition-colors" />

          {/* Right panel: Tool Detail + Orchestration tabs */}
          <ResizablePanel defaultSize={50} minSize={25} maxSize={75}>
            <div className="h-full flex flex-col">
              <Tabs
                value={rightTab}
                onValueChange={(v) => setRightTab(v as 'tool' | 'orchestration')}
                className="h-full flex flex-col"
              >
                <div className="px-4 py-2 border-b border-border bg-card/60 flex-shrink-0">
                  <TabsList className="h-7">
                    <TabsTrigger value="tool" className="text-xs px-3 py-1">Tool Detail</TabsTrigger>
                    <TabsTrigger value="orchestration" className="text-xs px-3 py-1">Orchestration</TabsTrigger>
                  </TabsList>
                </div>
                <TabsContent value="tool" className="flex-1 min-h-0 mt-0">
                  <div className="h-full">
                    <ToolCallPanel toolCall={selectedToolCall} logStem={logStem} />
                  </div>
                </TabsContent>
                <TabsContent value="orchestration" className="flex-1 min-h-0 mt-0">
                  <div className="h-full">
                    <AgentOrchestrationPanel
                      events={events}
                      toolBreakdown={metadata.toolBreakdown}
                    />
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Keyboard hint footer */}
      <div className="border-t border-border bg-muted/30 px-6 py-1.5 flex-shrink-0">
        <div className="flex items-center justify-center gap-6 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">←</kbd>
            <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">→</kbd>
            Navigate turns
          </span>
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">Esc</kbd>
            Back
          </span>
        </div>
      </div>
    </div>
  );
}

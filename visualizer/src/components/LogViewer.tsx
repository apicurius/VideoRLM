'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { StatsCard } from './StatsCard';
import { TrajectoryPanel } from './TrajectoryPanel';
import { ExecutionPanel } from './ExecutionPanel';
import { IterationTimeline } from './IterationTimeline';
import { ThemeToggle } from './ThemeToggle';
import { RLMLogFile } from '@/lib/types';

interface LogViewerProps {
  logFile: RLMLogFile;
  onBack: () => void;
}

export function LogViewer({ logFile, onBack }: LogViewerProps) {
  const [selectedIteration, setSelectedIteration] = useState(0);
  const { iterations, metadata, config } = logFile;

  const goToPrevious = useCallback(() => {
    setSelectedIteration(prev => Math.max(0, prev - 1));
  }, []);

  const goToNext = useCallback(() => {
    setSelectedIteration(prev => Math.min(iterations.length - 1, prev + 1));
  }, [iterations.length]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' || e.key === 'j') {
        goToPrevious();
      } else if (e.key === 'ArrowRight' || e.key === 'k') {
        goToNext();
      } else if (e.key === 'Escape') {
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goToPrevious, goToNext, onBack]);

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-background">
      {/* Top Bar - Compact header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm">
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
                <h1 className="font-semibold flex items-center gap-2 text-sm">
                  <span className="text-primary">◈</span>
                  {logFile.fileName}
                  {metadata.isVideoRun && (
                    <Badge className="bg-violet-500/20 text-violet-600 dark:text-violet-400 border-violet-500/30 text-[10px] px-1.5 py-0 h-4 ml-1">
                      <svg className="w-2.5 h-2.5 mr-0.5 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      Video
                    </Badge>
                  )}
                </h1>
                <p className="text-[10px] text-muted-foreground font-mono mt-0.5">
                  {config.root_model ?? 'Unknown model'} • {config.backend ?? 'Unknown backend'} • {config.environment_type ?? 'Unknown env'}
                  {metadata.isVideoRun && (
                    <>
                      {config.fps != null && ` • ${config.fps} fps`}
                      {config.num_segments != null && ` • ${config.num_segments} segments`}
                      {config.max_frames_per_segment != null && ` • max ${config.max_frames_per_segment} frames/seg`}
                      {config.resize && ` • ${config.resize[0]}×${config.resize[1]}`}
                    </>
                  )}
                </p>
                {metadata.isVideoRun && metadata.videoPath && (
                  <p className="text-[10px] text-violet-600 dark:text-violet-400 font-mono mt-0.5 truncate max-w-md" title={metadata.videoPath}>
                    {metadata.videoPath.split('/').pop()}
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

      {/* Question & Answer + Stats Row */}
      <div className="border-b border-border bg-muted/30 px-6 py-4">
        <div className="flex gap-6">
          {/* Question & Answer Summary */}
          <Card className="flex-1 bg-gradient-to-r from-primary/5 to-accent/5 border-primary/20">
            <CardContent className="p-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Context / Question
                  </p>
                  <p className="text-sm font-medium line-clamp-2">
                    {metadata.contextQuestion}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
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
              label="Iterations"
              value={metadata.totalIterations}
              icon="◎"
              variant="cyan"
            />
            <StatsCard
              label="Code"
              value={metadata.totalCodeBlocks}
              icon="⟨⟩"
              variant="green"
            />
            <StatsCard
              label="Sub-LM"
              value={metadata.totalSubLMCalls}
              icon="◇"
              variant="magenta"
            />
            <StatsCard
              label="Exec"
              value={`${metadata.totalExecutionTime.toFixed(2)}s`}
              icon="⏱"
              variant="yellow"
            />
            {metadata.isVideoRun && (
              <StatsCard
                label="Frames"
                value={metadata.totalFramesSent}
                icon="▦"
                variant="magenta"
                subtext={config.num_segments != null ? `${config.num_segments} segments` : undefined}
              />
            )}
          </div>
        </div>
      </div>

      {/* Iteration Timeline - Full width scrollable row */}
      <IterationTimeline
        iterations={iterations}
        selectedIteration={selectedIteration}
        onSelectIteration={setSelectedIteration}
        isVideoRun={metadata.isVideoRun}
      />

      {/* Main Content - Resizable Split View */}
      <div className="flex-1 min-h-0">
        <ResizablePanelGroup orientation="horizontal">
          {/* Left Panel - Prompt & Response */}
          <ResizablePanel defaultSize={50} minSize={20} maxSize={80}>
            <div className="h-full border-r border-border">
              <TrajectoryPanel
                iterations={iterations}
                selectedIteration={selectedIteration}
                onSelectIteration={setSelectedIteration}
              />
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle className="bg-border hover:bg-primary/30 transition-colors" />

          {/* Right Panel - Code Execution & Sub-LM Calls */}
          <ResizablePanel defaultSize={50} minSize={20} maxSize={80}>
            <div className="h-full bg-background">
              <ExecutionPanel
                iteration={iterations[selectedIteration] || null}
              />
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Keyboard hint footer */}
      <div className="border-t border-border bg-muted/30 px-6 py-1.5">
        <div className="flex items-center justify-center gap-6 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-muted rounded text-[9px]">←</kbd>
            <kbd className="px-1 py-0.5 bg-muted rounded text-[9px]">→</kbd>
            Navigate
          </span>
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-muted rounded text-[9px]">Esc</kbd>
            Back
          </span>
        </div>
      </div>
    </div>
  );
}

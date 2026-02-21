'use client';

import { useState, useCallback, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { FileUploader } from './FileUploader';
import { LogViewer } from './LogViewer';
import { KUAViLogViewer } from './KUAViLogViewer';
import { AsciiRLM, AsciiKUAVi } from './AsciiGlobe';
import { ThemeToggle } from './ThemeToggle';
import { parseLogFile } from '@/lib/parse-logs';
import { LogFile, isKUAViLog, isRLMLog } from '@/lib/types';
import { cn } from '@/lib/utils';

interface DemoLogInfo {
  fileName: string;
  contextPreview: string | null;
  traceType: 'rlm' | 'kuavi';
  toolCount: number;
  llmCallCount: number;
  evalCount: number;
  size: number;
  mtime: string;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function Dashboard() {
  const [logFiles, setLogFiles] = useState<LogFile[]>([]);
  const [selectedLog, setSelectedLog] = useState<LogFile | null>(null);
  const [demoLogs, setDemoLogs] = useState<DemoLogInfo[]>([]);
  const [loadingDemos, setLoadingDemos] = useState(true);

  // Load demo log previews on mount - single API call, no file content fetching
  useEffect(() => {
    async function loadDemoPreviews() {
      try {
        const listResponse = await fetch('/api/logs', { cache: 'no-store' });
        if (!listResponse.ok) {
          throw new Error('Failed to fetch log list');
        }
        const { files } = await listResponse.json();

        const previews: DemoLogInfo[] = files.map((f: { name: string; size: number; mtime: string; traceType: string; lineCount: number; toolCallCount?: number; llmCallCount?: number; evalCount?: number; model: string | null; videoPath: string | null }) => ({
          fileName: f.name,
          contextPreview: f.videoPath
            ? f.videoPath.split('/').pop() ?? f.videoPath
            : f.model ?? null,
          traceType: f.traceType as 'rlm' | 'kuavi',
          toolCount: f.traceType === 'kuavi' ? (f.toolCallCount ?? f.lineCount) : f.lineCount,
          llmCallCount: f.llmCallCount ?? 0,
          evalCount: f.evalCount ?? 0,
          size: f.size,
          mtime: f.mtime,
        }));

        setDemoLogs(previews);
      } catch (e) {
      } finally {
        setLoadingDemos(false);
      }
    }

    loadDemoPreviews();
  }, []);

  const handleFileLoaded = useCallback((fileName: string, content: string) => {
    const parsed = parseLogFile(fileName, content);
    setLogFiles(prev => {
      if (prev.some(f => f.fileName === fileName)) {
        return prev.map(f => f.fileName === fileName ? parsed : f);
      }
      return [...prev, parsed];
    });
    setSelectedLog(parsed);
  }, []);

  const loadDemoLog = useCallback(async (fileName: string) => {
    try {
      const response = await fetch(`/api/logs/${fileName}`);
      if (!response.ok) throw new Error('Failed to load demo log');
      const content = await response.text();
      handleFileLoaded(fileName, content);
    } catch (error) {
      alert('Failed to load demo log. Make sure the log files are in the public/logs folder.');
    }
  }, [handleFileLoaded]);

  if (selectedLog) {
    if (isKUAViLog(selectedLog)) {
      return (
        <KUAViLogViewer
          logFile={selectedLog}
          onBack={() => setSelectedLog(null)}
        />
      );
    }
    return (
      <LogViewer
        logFile={selectedLog}
        onBack={() => setSelectedLog(null)}
      />
    );
  }

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 grid-pattern opacity-30 dark:opacity-15" />
      <div className="absolute top-0 left-1/3 w-[500px] h-[500px] bg-primary/5 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary/3 rounded-full blur-3xl" />

      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-border">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold tracking-tight">
                  <span className="text-primary">KUAVi</span>
                  <span className="text-muted-foreground ml-2 font-normal">Trace Visualizer</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                  RLM &amp; Agent Orchestration Traces
                </p>
              </div>
              <div className="flex items-center gap-4">
                <ThemeToggle />
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground font-mono">
                  <span className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                    READY
                  </span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid lg:grid-cols-2 gap-10">
            {/* Left Column - Upload & ASCII Art */}
            <div className="space-y-8">
              {/* Upload Section */}
              <div>
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">01</span>
                  Upload Log File
                </h2>
                <FileUploader onFileLoaded={handleFileLoaded} />
              </div>

              {/* ASCII Architecture Diagram */}
              <div className="hidden lg:block">
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">◈</span>
                  Architecture
                </h2>
                <div className="bg-muted/50 border border-border rounded-lg p-4 overflow-x-auto">
                  <AsciiRLM />
                </div>
                <h2 className="text-sm font-medium mb-3 mt-6 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">◈</span>
                  KUAVi Pipeline
                </h2>
                <div className="bg-muted/50 border border-border rounded-lg p-4 overflow-x-auto">
                  <AsciiKUAVi />
                </div>
              </div>
            </div>

            {/* Right Column - Demo Logs & Loaded Files */}
            <div className="space-y-8">
              {/* Demo Logs Section */}
              <div>
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">02</span>
                  Recent Traces
                  <span className="text-[10px] text-muted-foreground/60 ml-1">(latest 20)</span>
                </h2>

                {loadingDemos ? (
                  <Card>
                    <CardContent className="p-6 text-center">
                      <div className="animate-pulse flex items-center justify-center gap-2 text-muted-foreground text-sm">
                        Loading traces...
                      </div>
                    </CardContent>
                  </Card>
                ) : demoLogs.length === 0 ? (
                  <Card className="border-dashed">
                    <CardContent className="p-6 text-center text-muted-foreground text-sm">
                      No log files found in /public/logs/
                    </CardContent>
                  </Card>
                ) : (
                  <ScrollArea className="h-[480px]">
                    <div className="space-y-2 pr-4">
                      {demoLogs.map((demo) => (
                        <Card
                          key={demo.fileName}
                          onClick={() => loadDemoLog(demo.fileName)}
                          className={cn(
                            'cursor-pointer transition-all hover:scale-[1.01]',
                            'hover:border-primary/50 hover:bg-primary/5'
                          )}
                        >
                          <CardContent className="p-3">
                            <div className="flex items-center gap-3">
                              {/* Trace type indicator */}
                              <div className="relative flex-shrink-0">
                                <div className={cn(
                                  'w-2.5 h-2.5 rounded-full',
                                  demo.traceType === 'kuavi'
                                    ? 'bg-violet-500'
                                    : 'bg-sky-500'
                                )} />
                              </div>

                              {/* Content */}
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1 flex-wrap">
                                  <span className="font-mono text-xs text-foreground/80 truncate">
                                    {demo.fileName}
                                  </span>
                                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4">
                                    ~{demo.toolCount} {demo.traceType === 'kuavi' ? 'tools' : 'iter'}
                                  </Badge>
                                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4 text-muted-foreground">
                                    {formatFileSize(demo.size)}
                                  </Badge>
                                  <Badge variant="outline" className={cn(
                                    "text-[9px] px-1.5 py-0 h-4",
                                    demo.traceType === 'kuavi'
                                      ? "bg-violet-500/10 text-violet-600 dark:text-violet-400 border-violet-500/30"
                                      : "bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/30"
                                  )}>
                                    {demo.traceType === 'kuavi' ? 'KUAVi' : 'RLM'}
                                  </Badge>
                                  {demo.llmCallCount > 0 && (
                                    <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4 bg-fuchsia-500/10 text-fuchsia-600 dark:text-fuchsia-400 border-fuchsia-500/30">
                                      {demo.llmCallCount} llm
                                    </Badge>
                                  )}
                                  {demo.evalCount > 0 && (
                                    <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30">
                                      {demo.evalCount} eval
                                    </Badge>
                                  )}
                                </div>
                                {demo.contextPreview && (
                                  <p className="text-[11px] font-mono text-muted-foreground truncate">
                                    {demo.contextPreview.length > 80
                                      ? demo.contextPreview.slice(0, 80) + '...'
                                      : demo.contextPreview}
                                  </p>
                                )}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </div>

              {/* Loaded Files Section */}
              {logFiles.length > 0 && (
                <div>
                  <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                    <span className="text-primary font-mono">03</span>
                    Loaded Files
                  </h2>
                  <ScrollArea className="h-[200px]">
                    <div className="space-y-2 pr-4">
                      {logFiles.map((log) => (
                        <Card
                          key={log.fileName}
                          className={cn(
                            'cursor-pointer transition-all hover:scale-[1.01]',
                            'hover:border-primary/50 hover:bg-primary/5'
                          )}
                          onClick={() => setSelectedLog(log)}
                        >
                          <CardContent className="p-3">
                            <div className="flex items-center gap-3">
                              <div className="relative flex-shrink-0">
                                <div className={cn(
                                  'w-2.5 h-2.5 rounded-full',
                                  isKUAViLog(log)
                                    ? log.metadata.hasErrors
                                      ? 'bg-red-500'
                                      : log.metadata.isComplete
                                        ? 'bg-primary'
                                        : 'bg-muted-foreground/30'
                                    : log.metadata.finalAnswer
                                      ? 'bg-primary'
                                      : 'bg-muted-foreground/30'
                                )} />
                                {(isKUAViLog(log) ? (log.metadata.isComplete && !log.metadata.hasErrors) : (isRLMLog(log) && log.metadata.finalAnswer)) && (
                                  <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-primary animate-ping opacity-50" />
                                )}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="font-mono text-xs truncate text-foreground/80">
                                    {log.fileName}
                                  </span>
                                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4">
                                    {isKUAViLog(log) ? `${log.metadata.totalToolCalls} tools` : `${log.metadata.totalIterations} iter`}
                                  </Badge>
                                  {isKUAViLog(log) && (log.metadata.totalInputTokens + log.metadata.totalOutputTokens) > 0 && (
                                    <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4 bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/30">
                                      {`${(((log.metadata.totalInputTokens + log.metadata.totalOutputTokens) / 1000)).toFixed(1)}K tok`}
                                    </Badge>
                                  )}
                                  <Badge variant="outline" className={cn(
                                    "text-[9px] px-1.5 py-0 h-4",
                                    isKUAViLog(log)
                                      ? "bg-violet-500/10 text-violet-600 dark:text-violet-400 border-violet-500/30"
                                      : "bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/30"
                                  )}>
                                    {isKUAViLog(log) ? 'KUAVi' : 'RLM'}
                                  </Badge>
                                </div>
                                <p className="text-[11px] text-muted-foreground truncate">
                                  {isKUAViLog(log)
                                    ? (log.metadata.question || log.metadata.videoPath || 'KUAVi trace')
                                    : log.metadata.contextQuestion}
                                </p>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              )}
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-border mt-8">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <p className="text-[10px] text-muted-foreground font-mono">
              KUAVi Trace Visualizer • RLM &amp; Agent Orchestration
            </p>
            <p className="text-[10px] text-muted-foreground font-mono">
              Prompt → [LM ↔ Tools] → Answer
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}

"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, ChevronDown, ChevronRight, Clock, Wrench, AlertTriangle, MessageSquare, UploadCloud } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { parseLogFile, shortToolName, groupEventsIntoTurns } from "@/lib/parse-logs";
import type { LogFile, KUAViLogFile, RLMLogFile, KUAViTurn, KUAViToolCall } from "@/lib/trace-types";
import { isKUAViLog, isRLMLog } from "@/lib/trace-types";

interface LogListItem {
  name: string;
  size: number;
  mtime: string;
  traceType: "rlm" | "kuavi";
  toolCallCount: number;
  model: string | null;
  videoPath: string | null;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

// Tool call detail row
function ToolCallRow({ tc }: { tc: KUAViToolCall }) {
  const [expanded, setExpanded] = useState(false);
  const name = shortToolName(tc.tool_name);
  const hasError = tc.has_error || (typeof tc.tool_response === "string" && tc.tool_response.includes("Error:"));

  return (
    <div className="border border-white/5 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className={`w-full flex items-center gap-2 px-3 py-2 text-left text-xs font-mono hover:bg-white/5 transition-colors ${hasError ? "bg-red-500/5" : ""}`}
      >
        {expanded ? <ChevronDown className="w-3 h-3 text-zinc-500 shrink-0" /> : <ChevronRight className="w-3 h-3 text-zinc-500 shrink-0" />}
        <Wrench className={`w-3 h-3 shrink-0 ${hasError ? "text-red-400" : "text-amber-500"}`} />
        <span className={`font-semibold ${hasError ? "text-red-400" : "text-zinc-200"}`}>{name}</span>
        {tc.duration_ms != null && (
          <span className="text-zinc-600 ml-auto">{tc.duration_ms}ms</span>
        )}
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-3 py-2 border-t border-white/5 space-y-2">
              {tc.tool_input && Object.keys(tc.tool_input).length > 0 && (
                <div>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase">Input</span>
                  <pre className="text-[10px] text-zinc-400 mt-1 overflow-x-auto max-h-32 whitespace-pre-wrap">
                    {JSON.stringify(tc.tool_input, null, 2)}
                  </pre>
                </div>
              )}
              {tc.response_summary && (
                <div>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase">Response</span>
                  <p className="text-[10px] text-zinc-400 mt-1">{tc.response_summary}</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// KUAVi turn view
function KUAViTurnView({ turn }: { turn: KUAViTurn }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Badge variant="outline" className="text-[10px] font-mono text-amber-400 border-amber-500/30 bg-amber-500/10 px-2 py-0">
          Turn {turn.index + 1}
        </Badge>
        {turn.toolCalls.length > 0 && (
          <span className="text-[10px] text-zinc-500">{turn.toolCalls.length} tool calls</span>
        )}
      </div>
      {turn.reasoning && (
        <div className="pl-3 border-l-2 border-white/5">
          <p className="text-[11px] text-zinc-400 leading-relaxed line-clamp-3">
            {turn.reasoning.text}
          </p>
        </div>
      )}
      <div className="space-y-1">
        {turn.toolCalls.map((tc, i) => (
          <ToolCallRow key={i} tc={tc} />
        ))}
      </div>
    </div>
  );
}

// RLM iteration view
function RLMIterationView({ iter }: { iter: RLMLogFile["iterations"][0] }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="space-y-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left"
      >
        {expanded ? <ChevronDown className="w-3 h-3 text-zinc-500" /> : <ChevronRight className="w-3 h-3 text-zinc-500" />}
        <Badge variant="outline" className="text-[10px] font-mono text-blue-400 border-blue-500/30 bg-blue-500/10 px-2 py-0">
          Iter {iter.iteration}
        </Badge>
        <span className="text-[10px] text-zinc-500">{iter.code_blocks.length} code blocks</span>
        {iter.iteration_time != null && (
          <span className="text-[10px] text-zinc-600 ml-auto">{iter.iteration_time.toFixed(1)}s</span>
        )}
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden pl-5"
          >
            {iter.code_blocks.map((block, i) => (
              <div key={i} className="border border-white/5 rounded-lg p-2 mt-1">
                <pre className="text-[10px] text-zinc-300 overflow-x-auto max-h-24 whitespace-pre-wrap font-mono">
                  {block.code}
                </pre>
                {block.result?.stdout && (
                  <pre className="text-[10px] text-green-400/70 mt-1 max-h-16 overflow-hidden">
                    {block.result.stdout.slice(0, 500)}
                  </pre>
                )}
                {block.result?.stderr && (
                  <pre className="text-[10px] text-red-400/70 mt-1 max-h-16 overflow-hidden">
                    {block.result.stderr.slice(0, 300)}
                  </pre>
                )}
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Main TraceViewer
export function TraceViewer() {
  const [logs, setLogs] = useState<LogListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedLog, setSelectedLog] = useState<LogFile | null>(null);
  const [loadingLog, setLoadingLog] = useState(false);

  useEffect(() => {
    fetch("/api/logs", { cache: "no-store" })
      .then((r) => r.json())
      .then((data) => setLogs(data.files || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const loadLog = useCallback(async (name: string) => {
    setLoadingLog(true);
    try {
      const resp = await fetch(`/api/logs/${encodeURIComponent(name)}`);
      const content = await resp.text();
      const parsed = parseLogFile(name, content);
      setSelectedLog(parsed);
    } catch { /* ignore */ }
    setLoadingLog(false);
  }, []);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const content = reader.result as string;
      const parsed = parseLogFile(file.name, content);
      setSelectedLog(parsed);
    };
    reader.readAsText(file);
  }, []);

  if (selectedLog) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5 bg-white/[0.02]">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelectedLog(null)}
            className="h-6 px-2 text-xs text-zinc-400 hover:text-white"
          >
            ← Back
          </Button>
          <span className="text-xs font-mono text-zinc-500 truncate">{selectedLog.fileName}</span>
          <Badge variant="outline" className={`ml-auto text-[10px] px-1.5 py-0 ${isKUAViLog(selectedLog) ? "text-amber-400 border-amber-500/30" : "text-blue-400 border-blue-500/30"}`}>
            {isKUAViLog(selectedLog) ? "KUAVi" : "RLM"}
          </Badge>
        </div>

        <ScrollArea className="flex-1 p-4">
          {isKUAViLog(selectedLog) && (
            <KUAViTraceDetail log={selectedLog} />
          )}
          {isRLMLog(selectedLog) && (
            <RLMTraceDetail log={selectedLog} />
          )}
        </ScrollArea>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5 bg-white/[0.02]">
        <FileText className="w-4 h-4 text-zinc-400" />
        <span className="text-sm font-semibold text-zinc-200">Execution Traces</span>
        <label className="ml-auto cursor-pointer">
          <input type="file" accept=".jsonl" className="hidden" onChange={handleFileUpload} />
          <div className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium text-zinc-400 hover:text-white hover:bg-white/5 transition-colors border border-white/10">
            <UploadCloud className="w-3 h-3" />
            Upload
          </div>
        </label>
      </div>

      <ScrollArea className="flex-1 p-3">
        {loading ? (
          <div className="text-center text-xs text-zinc-500 py-8">Loading traces...</div>
        ) : logs.length === 0 ? (
          <div className="text-center text-xs text-zinc-500 py-8">
            <FileText className="w-8 h-8 mx-auto mb-2 text-zinc-600" />
            <p>No traces found in ./logs/</p>
            <p className="text-zinc-600 mt-1">Run an analysis to generate traces</p>
          </div>
        ) : (
          <div className="space-y-1.5">
            {logs.map((log) => (
              <button
                key={log.name}
                onClick={() => loadLog(log.name)}
                disabled={loadingLog}
                className="w-full text-left p-3 rounded-xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/10 transition-all group"
              >
                <div className="flex items-center gap-2 mb-1">
                  <Badge variant="outline" className={`text-[9px] px-1.5 py-0 ${log.traceType === "kuavi" ? "text-amber-400 border-amber-500/30 bg-amber-500/5" : "text-blue-400 border-blue-500/30 bg-blue-500/5"}`}>
                    {log.traceType === "kuavi" ? "KUAVi" : "RLM"}
                  </Badge>
                  <span className="text-[10px] text-zinc-500 font-mono truncate">{log.name}</span>
                  <span className="text-[10px] text-zinc-600 ml-auto">{formatSize(log.size)}</span>
                </div>
                {(log.videoPath || log.model) && (
                  <p className="text-[10px] text-zinc-400 truncate">
                    {log.videoPath ? log.videoPath.split("/").pop() : log.model}
                  </p>
                )}
                {log.toolCallCount > 0 && (
                  <span className="text-[10px] text-zinc-600">{log.toolCallCount} tool calls</span>
                )}
              </button>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}

function KUAViTraceDetail({ log }: { log: KUAViLogFile }) {
  const turns = groupEventsIntoTurns(log.events);
  const { metadata } = log;

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: "Tool Calls", value: metadata.totalToolCalls },
          { label: "Turns", value: metadata.totalTurns },
          { label: "Searches", value: metadata.totalSearches },
        ].map((s) => (
          <div key={s.label} className="p-2 rounded-lg bg-white/[0.03] border border-white/5 text-center">
            <p className="text-lg font-bold text-zinc-200">{s.value}</p>
            <p className="text-[9px] text-zinc-500 uppercase tracking-wider">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Tool breakdown */}
      {Object.keys(metadata.toolBreakdown).length > 0 && (
        <div className="space-y-2">
          <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">Tool Breakdown</h4>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(metadata.toolBreakdown)
              .sort(([, a], [, b]) => b - a)
              .map(([name, count]) => (
                <Badge key={name} variant="outline" className="text-[10px] text-zinc-300 border-white/10 bg-white/[0.03] px-2 py-0.5">
                  {name} <span className="text-amber-500 ml-1">×{count}</span>
                </Badge>
              ))}
          </div>
        </div>
      )}

      {/* Turns */}
      <div className="space-y-4">
        <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">Timeline</h4>
        {turns.map((turn) => (
          <KUAViTurnView key={turn.index} turn={turn} />
        ))}
      </div>

      {/* Final answer */}
      {metadata.finalAnswer && (
        <div className="p-3 rounded-xl border border-green-500/20 bg-green-500/5">
          <h4 className="text-[10px] font-bold text-green-400 uppercase tracking-wider mb-2">Final Answer</h4>
          <p className="text-xs text-zinc-300 leading-relaxed line-clamp-10">{metadata.finalAnswer}</p>
        </div>
      )}
    </div>
  );
}

function RLMTraceDetail({ log }: { log: RLMLogFile }) {
  const { metadata, iterations } = log;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: "Iterations", value: metadata.totalIterations },
          { label: "Code Blocks", value: metadata.totalCodeBlocks },
          { label: "Sub-LM Calls", value: metadata.totalSubLMCalls },
        ].map((s) => (
          <div key={s.label} className="p-2 rounded-lg bg-white/[0.03] border border-white/5 text-center">
            <p className="text-lg font-bold text-zinc-200">{s.value}</p>
            <p className="text-[9px] text-zinc-500 uppercase tracking-wider">{s.label}</p>
          </div>
        ))}
      </div>

      {metadata.totalExecutionTime > 0 && (
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <Clock className="w-3 h-3" />
          Total: {formatDuration(metadata.totalExecutionTime)}
        </div>
      )}

      <div className="space-y-3">
        <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">Iterations</h4>
        {iterations.map((iter) => (
          <RLMIterationView key={iter.iteration} iter={iter} />
        ))}
      </div>

      {metadata.finalAnswer && (
        <div className="p-3 rounded-xl border border-green-500/20 bg-green-500/5">
          <h4 className="text-[10px] font-bold text-green-400 uppercase tracking-wider mb-2">Final Answer</h4>
          <p className="text-xs text-zinc-300 leading-relaxed line-clamp-10">{metadata.finalAnswer}</p>
        </div>
      )}
    </div>
  );
}

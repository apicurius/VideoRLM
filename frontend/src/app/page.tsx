"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Settings, PanelRight, UploadCloud, Video, ChevronRight, CheckCircle2, RotateCw, AlertTriangle, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";

// Types
type PipelineStep = {
  id: string;
  label: string;
  status: "pending" | "running" | "done" | "cached" | "error" | "skip";
  detail?: string;
};

type Timestamp = {
  seconds: number;
  label: string;
};

// Main Page Component
export default function VideoRLMInterface() {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pipeline state
  const [pipeline, setPipeline] = useState("rlm");
  const [backend, setBackend] = useState("openrouter");
  const [model, setModel] = useState("openai/gpt-4o");
  const [apiKey, setApiKey] = useState("");

  // Progress/Results State
  const [steps, setSteps] = useState<PipelineStep[]>([]);
  const [agentIterations, setAgentIterations] = useState<any[]>([]);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [answerHtml, setAnswerHtml] = useState<{ __html: string } | null>(null);
  const [timestamps, setTimestamps] = useState<Timestamp[]>([]);
  const [elapsed, setElapsed] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<NodeJS.Timeout>(null);

  // Clean up ObjectURL
  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      setFile(f);
      setVideoUrl(URL.createObjectURL(f));
      setError(null);
    }
  };

  const startTimer = () => {
    setElapsed(0);
    timerRef.current = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);
  };

  const stopTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const formatElapsed = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  const handleSeek = (seconds: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds;
      videoRef.current.play().catch(() => { });
    }
  };

  const handleAnalyze = async () => {
    if (!file || !question.trim()) {
      setError("Please upload a video and enter a question.");
      return;
    }

    setError(null);
    setIsLoading(true);
    setIsPanelOpen(true);
    setAnswerHtml(null);
    setSteps([]);
    setAgentIterations([]);
    startTimer();

    const formData = new FormData();
    formData.append("video", file);
    formData.append("question", question);
    formData.append("backend", backend);
    formData.append("model", model);
    formData.append("pipeline", pipeline);
    formData.append("custom_api_key", apiKey);

    try {
      // NOTE: Pointing to local FastAPI backend on port 8000
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Fast API sends data as standard SSE format: "data: {json}\n\n"
        const chunks = buffer.split('\n\n');

        // The last chunk might be incomplete, so put it back in the buffer
        buffer = chunks.pop() || "";

        for (const chunk of chunks) {
          const line = chunk.trim();
          if (line.startsWith("data: ")) {
            try {
              const eventStr = line.slice(6).trim();
              if (!eventStr) continue;

              const event = JSON.parse(eventStr);

              if (event.type === "init") {
                setSteps(event.steps);
              } else if (event.type === "step") {
                setSteps(prev => prev.map(s =>
                  s.id === event.id
                    ? { ...s, status: event.status, detail: event.detail || s.detail }
                    : s
                ));
              } else if (event.type === "iteration") {
                setCurrentIteration(event.n);
                setAgentIterations(prev => {
                  const newIters = [...prev];
                  newIters[event.n] = { tools: event.tools, errors: event.errors };
                  return newIters;
                });
              } else if (event.type === "result") {
                setAnswerHtml({ __html: event.answer_html });
                setTimestamps(event.timestamps || []);
                stopTimer();
                setIsLoading(false);
              } else if (event.type === "error") {
                throw new Error(event.message);
              }
            } catch (err) {
              console.error("Error parsing event stream payload:", line, err);
            }
          }
        }
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An unknown error occurred");
      stopTimer();
      setIsLoading(false);
    }
  };

  const getStepIcon = (status: PipelineStep["status"]) => {
    switch (status) {
      case "done": return <CheckCircle2 className="w-5 h-5 text-green-500" />;
      case "running": return <RotateCw className="w-5 h-5 text-amber-500 animate-spin" />;
      case "error": return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case "cached": return <CheckCircle2 className="w-5 h-5 text-blue-500" />;
      case "skip": return <ChevronRight className="w-5 h-5 text-muted-foreground" />;
      default: return <div className="w-3 h-3 rounded-full bg-muted-foreground/30 m-1" />;
    }
  };

  return (
    <div className="flex h-screen w-full flex-col bg-[#050505] text-foreground overflow-hidden font-sans">
      {/* Navbar */}
      <header className="flex h-14 items-center gap-4 border-b border-border/40 bg-card/10 px-6 backdrop-blur-xl z-50 shadow-sm">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-amber-600 to-amber-400 text-white shadow-[0_0_15px_rgba(240,168,64,0.3)]">
            <Play className="h-4 w-4 fill-current ml-[2px]" />
          </div>
          <span className="font-semibold tracking-tight text-lg flex items-baseline ml-1 text-zinc-100">
            Video<em className="text-amber-500 not-italic font-black">RLM</em>
          </span>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => setIsSettingsOpen(!isSettingsOpen)} className={`rounded-full px-4 text-xs font-semibold ${isSettingsOpen ? "bg-white/10 text-white" : "text-zinc-400 hover:text-white hover:bg-white/5"}`}>
            <Settings className="mr-2 h-3.5 w-3.5" />
            Settings
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setIsPanelOpen(!isPanelOpen)} className={`rounded-full px-4 text-xs font-semibold ${isPanelOpen ? "bg-white/10 text-white" : "text-zinc-400 hover:text-white hover:bg-white/5"}`}>
            <PanelRight className="mr-2 h-3.5 w-3.5" />
            Panel
          </Button>
        </div>
      </header>

      {/* Main Area */}
      <main className="flex flex-1 overflow-hidden relative">

        {/* Left Side: Video & Input */}
        <div className="flex flex-1 flex-col relative z-20 min-w-0 bg-[#050505] shadow-[20px_0_40px_-20px_rgba(0,0,0,0.5)]">
          {/* Settings Drawer (Animated Dropdown) */}
          <AnimatePresence>
            {isSettingsOpen && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95, y: -10, filter: "blur(10px)" }}
                animate={{ opacity: 1, scale: 1, y: 0, filter: "blur(0px)" }}
                exit={{ opacity: 0, scale: 0.95, y: -10, filter: "blur(10px)" }}
                transition={{ type: "spring", stiffness: 300, damping: 25 }}
                className="absolute top-4 right-4 z-50 w-80 p-5 rounded-3xl bg-zinc-950/95 backdrop-blur-3xl border border-white/10 shadow-2xl"
              >
                <div className="flex flex-col gap-5">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="text-sm font-semibold text-white">Pipeline Settings</h3>
                    <Button variant="ghost" size="sm" onClick={() => setIsSettingsOpen(false)} className="h-6 w-6 p-0 rounded-full text-zinc-400 hover:text-white hover:bg-white/10">
                      <X className="h-4 w-4" />
                    </Button>
                  </div>

                  <div className="space-y-3">
                    <Label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Pipeline</Label>
                    <Select value={pipeline} onValueChange={setPipeline}>
                      <SelectTrigger className="bg-white/5 border-white/10 h-11 text-sm rounded-xl focus:ring-amber-500/30 text-white"><SelectValue /></SelectTrigger>
                      <SelectContent className="bg-zinc-900 border-white/10 rounded-xl text-white">
                        <SelectItem value="rlm">RLM (Recursive Language Models)</SelectItem>
                        <SelectItem value="kuavi">KUAVi (Koc Univesity Agentic Visual intelligence)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-3">
                    <Label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Backend Provider</Label>
                    <Select value={backend} onValueChange={(v) => {
                      setBackend(v);
                      setModel(v === "openrouter" ? "openai/gpt-4o" : v === "openai" ? "gpt-4o" : v === "anthropic" ? "claude-3-5-sonnet-latest" : "gemini-3.1-pro-preview");
                    }}>
                      <SelectTrigger className="bg-white/5 border-white/10 h-11 text-sm rounded-xl focus:ring-amber-500/30 text-white"><SelectValue /></SelectTrigger>
                      <SelectContent className="bg-zinc-900 border-white/10 rounded-xl text-white">
                        <SelectItem value="openrouter">OpenRouter</SelectItem>
                        <SelectItem value="openai">OpenAI</SelectItem>
                        <SelectItem value="anthropic">Anthropic</SelectItem>
                        <SelectItem value="gemini">Gemini</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-3">
                    <Label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Model ID</Label>
                    <Input value={model} onChange={(e) => setModel(e.target.value)} className="bg-white/5 border-white/10 h-11 text-sm rounded-xl text-white focus-visible:ring-amber-500/30" />
                  </div>
                  <div className="space-y-3">
                    <Label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">API Key</Label>
                    <Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="Optional (uses .env)" className="bg-white/5 border-white/10 h-11 text-sm rounded-xl text-white focus-visible:ring-amber-500/30 placeholder:text-zinc-600" />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Video Stage */}
          <div className="flex-1 p-4 sm:p-8 flex flex-col items-center justify-center overflow-hidden bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-zinc-900/30 via-[#050505] to-[#050505]">
            <input type="file" ref={fileInputRef} accept="video/*" className="hidden" onChange={handleFileChange} />

            <AnimatePresence mode="wait">
              {!videoUrl ? (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, scale: 0.95, filter: "blur(10px)" }}
                  animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                  exit={{ opacity: 0, scale: 0.95, filter: "blur(10px)" }}
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                  onClick={() => fileInputRef.current?.click()}
                  className="group flex flex-col items-center justify-center gap-6 rounded-[2.5rem] border border-dashed border-white/10 bg-white/[0.02] px-16 py-24 cursor-pointer transition-all hover:bg-white/[0.04] hover:border-amber-500/40 hover:shadow-[0_0_80px_-20px_rgba(240,168,64,0.15)] shadow-2xl backdrop-blur-sm"
                >
                  <div className="rounded-[1.5rem] bg-white/5 p-6 shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)] border border-white/5 group-hover:bg-amber-500/10 group-hover:text-amber-500 group-hover:border-amber-500/20 transition-all group-hover:scale-110 duration-300">
                    <UploadCloud className="w-12 h-12 text-zinc-500 group-hover:text-amber-500 transition-colors" />
                  </div>
                  <div className="text-center space-y-2">
                    <h3 className="font-medium text-xl text-zinc-200">Drop a video here</h3>
                    <p className="text-sm text-zinc-500 font-medium">or click to browse &mdash; MP4, MOV, AVI, WebM</p>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="video"
                  initial={{ opacity: 0, scale: 0.95, filter: "blur(10px)" }}
                  animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                  className="relative w-full h-full max-h-[80vh] rounded-[2rem] overflow-hidden bg-black shadow-[0_20px_50px_-12px_rgba(0,0,0,0.5),0_0_0_1px_rgba(255,255,255,0.1)] flex items-center justify-center group"
                >
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    controls
                    controlsList="nodownload"
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute top-5 left-5 bg-black/40 backdrop-blur-xl border border-white/10 text-white px-4 py-2 rounded-xl text-xs font-mono flex items-center gap-2.5 opacity-0 group-hover:opacity-100 transition-opacity shadow-lg">
                    <Video className="w-3.5 h-3.5 text-zinc-400" />
                    <span className="truncate max-w-[200px]">{file?.name}</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Timestamps Bar */}
          {timestamps.length > 0 && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              className="px-6 py-4 border-t border-white/5 bg-zinc-950/50 flex items-center gap-4 overflow-x-auto no-scrollbar shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]"
            >
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest shrink-0">Timestamps</span>
              {timestamps.map((ts, i) => (
                <Badge
                  key={i}
                  variant="secondary"
                  className="cursor-pointer hover:bg-amber-500 hover:text-white shrink-0 font-mono transition-colors border border-amber-500/20 text-amber-500 bg-amber-500/10 px-3 py-1.5 rounded-lg shadow-sm"
                  onClick={() => handleSeek(ts.seconds)}
                >
                  <Play className="w-3 h-3 mr-1.5 inline fill-current opacity-70" />
                  {ts.label}
                </Badge>
              ))}
            </motion.div>
          )}

          {/* Input Area */}
          <div className="p-4 sm:p-6 border-t border-white/5 bg-zinc-950/90 backdrop-blur-2xl z-30 shadow-[0_-10px_40px_-10px_rgba(0,0,0,0.5)]">
            {error && (
              <Alert variant="destructive" className="mb-4 bg-red-500/10 border-red-500/20 text-red-400 rounded-xl px-5 py-4">
                <AlertTriangle className="h-5 w-5" />
                <AlertTitle className="font-bold">Error Processing Request</AlertTitle>
                <AlertDescription className="text-sm mt-1">{error}</AlertDescription>
              </Alert>
            )}
            <div className="relative flex items-end gap-3 max-w-5xl mx-auto">
              <div className="relative flex-1 bg-white/5 rounded-[1.5rem] border border-white/5 shadow-inner focus-within:ring-2 focus-within:ring-amber-500/30 focus-within:border-amber-500/50 focus-within:bg-white/[0.07] transition-all duration-300">
                <Textarea
                  placeholder="Ask anything about the video..."
                  className="min-h-[64px] max-h-[250px] w-full resize-none border-0 focus-visible:ring-0 rounded-[1.5rem] py-5 px-6 text-[15px] bg-transparent placeholder:text-zinc-500 text-zinc-100 placeholder:font-medium leading-relaxed"
                  rows={1}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleAnalyze();
                    }
                  }}
                />
              </div>
              <Button
                onClick={handleAnalyze}
                disabled={isLoading || !file || !question.trim()}
                className="h-[64px] w-[64px] rounded-[1.25rem] bg-gradient-to-tr from-amber-600 to-amber-400 hover:opacity-90 text-white shrink-0 self-end shadow-[0_4px_20px_-4px_rgba(240,168,64,0.4)] transition-all transform active:scale-[0.96] border-0 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? <RotateCw className="h-6 w-6 animate-[spin_1.5s_linear_infinite]" /> : <Play className="h-6 w-6 fill-current ml-1" />}
              </Button>
            </div>
          </div>
        </div>

        {/* Right Side: Analysis Panel */}
        <AnimatePresence initial={false}>
          {isPanelOpen && (
            <motion.div
              initial={{ width: 0, opacity: 0, x: 20, filter: "blur(10px)" }}
              animate={{ width: 440, opacity: 1, x: 0, filter: "blur(0px)" }}
              exit={{ width: 0, opacity: 0, x: 20, filter: "blur(10px)" }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
              className="h-full border-l border-white/5 bg-[#0a0a0c] flex flex-col shrink-0 overflow-hidden relative z-10 shadow-[-20px_0_40px_-10px_rgba(0,0,0,0.5)]"
            >
              <div className="flex items-center justify-between px-6 py-5 border-b border-white/5 bg-white/[0.02] shadow-sm">
                <h3 className="font-semibold tracking-tight text-zinc-100 flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-amber-500 shadow-[0_0_10px_rgba(240,168,64,0.8)]"></div>
                  Analysis Pipeline
                </h3>
                <div className="flex items-center gap-3">
                  {elapsed > 0 && <span className="text-xs font-mono font-medium text-zinc-500 bg-white/5 px-2 py-0.5 rounded-md border border-white/5">{formatElapsed(elapsed)}</span>}
                  {isLoading && <Badge variant="outline" className="text-amber-500 border-amber-500/30 bg-amber-500/5 animate-[pulse_2s_ease-in-out_infinite] px-2.5 shadow-[0_0_15px_rgba(240,168,64,0.1)]">Running...</Badge>}
                  {!isLoading && steps.length > 0 && !error && <Badge variant="outline" className="text-green-500 border-green-500/30 bg-green-500/5 px-2.5 shadow-[0_0_20px_-5px_rgba(74,222,128,0.2)]">Complete</Badge>}
                </div>
              </div>

              <ScrollArea className="flex-1 p-6">
                {steps.length === 0 && !answerHtml ? (
                  <div className="flex flex-col items-center justify-center text-center h-[50vh] text-zinc-600 space-y-4">
                    <div className="w-20 h-20 rounded-full bg-white/5 border border-white/5 flex items-center justify-center mb-2 shadow-inner">
                      <PanelRight className="w-8 h-8 stroke-[1.5] text-zinc-500" />
                    </div>
                    <div className="space-y-1">
                      <p className="font-medium text-zinc-400 text-lg">Panel Empty</p>
                      <p className="text-sm text-zinc-600 max-w-[220px] leading-relaxed">Submit a query to see the pipeline progress here.</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-10 pb-12">

                    {/* Pipeline Steps View */}
                    {/* Academic Pipeline Execution Trace */}
                    <div className="space-y-0 w-full relative outline-none pl-2 pr-2">
                      {steps.map((step, idx) => {
                        const isLast = idx === steps.length - 1;
                        const currentStatus = step.status || 'pending';
                        const isRunning = currentStatus === 'running';
                        const isDone = currentStatus === 'done' || currentStatus === 'cached';
                        const isError = currentStatus === 'error';
                        const isPending = currentStatus === 'pending' || currentStatus === 'skip';

                        return (
                          <div key={step.id} className="relative flex flex-col w-full group">
                            {/* Data Flow Pipe (Connecting to next node) */}
                            {!isLast && (
                              <div className="absolute left-[23px] top-[48px] bottom-[-16px] w-[2px] z-0 flex flex-col">
                                <div className={`w-full h-full transition-all duration-700
                                  ${isDone && steps[idx + 1]?.status !== 'pending' ? 'bg-green-500/30' :
                                    (isRunning || (isDone && steps[idx + 1]?.status === 'running')) ? 'bg-gradient-to-b from-amber-500/50 to-transparent' :
                                      'bg-white/5 border-l border-dashed border-white/10'}
                                `} />
                              </div>
                            )}

                            {/* Node Card */}
                            <motion.div
                              initial={{ opacity: 0, y: 15, filter: "blur(5px)" }}
                              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                              transition={{ delay: idx * 0.05, type: "spring", stiffness: 300 }}
                              className="relative z-10 flex gap-5"
                            >
                              {/* Anchor Icon */}
                              <div className="shrink-0 pt-3 relative z-10">
                                <div className={`bg-[#0a0a0c] border-[2px] rounded-full w-12 h-12 flex items-center justify-center transition-all duration-300 shadow-xl
                                    ${isRunning ? 'border-amber-500 shadow-[0_0_30px_-5px_rgba(240,168,64,0.5)] scale-110' :
                                    isDone ? 'border-green-500/40 bg-green-500/10' :
                                      isError ? 'border-red-500' : 'border-white/10 bg-[#0f0f12]'}
                                  `}>
                                  {getStepIcon(currentStatus)}
                                </div>
                              </div>

                              {/* Content Block */}
                              <div className={`flex-1 my-2 rounded-2xl border transition-all duration-300 overflow-hidden
                                ${isRunning ? 'bg-white/[0.03] border-amber-500/30 shadow-[0_0_40px_-15px_rgba(240,168,64,0.1)]' :
                                  'bg-[#0f0f12]/50 border-white/5 hover:bg-white/[0.02] hover:border-white/10'}
                              `}>
                                {/* Header */}
                                <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between bg-white/[0.01]">
                                  <div className="flex flex-col">
                                    <span className="text-[9px] font-mono font-bold text-zinc-500 tracking-[0.2em] uppercase mb-0.5">Stage {(idx + 1).toString().padStart(2, '0')} // Module</span>
                                    <h4 className={`text-sm font-semibold tracking-wide ${isRunning ? 'text-amber-500' : 'text-zinc-200 group-hover:text-zinc-100 transition-colors'}`}>
                                      {step.label}
                                    </h4>
                                  </div>
                                  <div>
                                    <Badge variant="outline" className={`text-[10px] font-mono px-2 py-0 h-5 border ${isRunning ? 'text-amber-400 border-amber-500/30 bg-amber-500/10' :
                                      isDone ? 'text-green-400 border-green-500/30 bg-green-500/10' :
                                        isError ? 'text-red-400 border-red-500/30 bg-red-500/10' :
                                          'text-zinc-500 border-white/10 bg-white/5'
                                      }`}>
                                      {currentStatus.toUpperCase()}
                                    </Badge>
                                  </div>
                                </div>

                                {/* Body / Detail Terminal */}
                                {(step.detail || step.id === "agent") && (
                                  <div className="p-4 bg-black/20">
                                    {step.detail && (
                                      <div className="font-mono text-[11px] leading-relaxed text-zinc-400 flex items-start gap-2">
                                        <span className="text-zinc-600 select-none">&gt;</span>
                                        <span className={isRunning ? 'text-amber-200/70' : ''}>{step.detail}</span>
                                        {isRunning && <span className="w-1.5 h-3 bg-amber-500/50 inline-block animate-pulse ml-1 align-middle" />}
                                      </div>
                                    )}

                                    {/* Agent Iterations Subflow */}
                                    {step.id === "agent" && agentIterations.length > 0 && (
                                      <div className="mt-4 border-l-2 border-amber-500/20 pl-4 py-1 space-y-4">
                                        <AnimatePresence>
                                          <motion.div
                                            initial={{ height: 0, opacity: 0 }}
                                            animate={{ height: "auto", opacity: 1 }}
                                            className="space-y-3"
                                          >
                                            <div className="flex items-center gap-2">
                                              <RotateCw className="w-3.5 h-3.5 text-amber-500/70 animate-spin" />
                                              <p className="font-mono text-[10px] tracking-widest uppercase text-amber-500/70 py-0.5 px-2 bg-amber-500/10 rounded border border-amber-500/20">
                                                Agent Loop // Iteration {currentIteration}
                                              </p>
                                            </div>

                                            <div className="flex flex-col gap-2">
                                              {/* Tools Used */}
                                              {agentIterations[currentIteration]?.tools?.length > 0 && (
                                                <div className="flex flex-wrap gap-2">
                                                  {agentIterations[currentIteration].tools.map((t: string, i: number) => (
                                                    <div key={i} className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-white/[0.04] border border-white/10">
                                                      <Settings className="w-3 h-3 text-zinc-500" />
                                                      <span className="font-mono text-[11px] text-zinc-300">{t}()</span>
                                                    </div>
                                                  ))}
                                                </div>
                                              )}

                                              {/* Errors */}
                                              {agentIterations[currentIteration]?.errors?.map((e: string, i: number) => (
                                                <div key={`err-${i}`} className="flex items-start gap-2 px-3 py-2 rounded-md bg-red-500/10 border border-red-500/20">
                                                  <AlertTriangle className="w-3.5 h-3.5 text-red-500 shrink-0 mt-0.5" />
                                                  <span className="font-mono text-[10px] text-red-400 break-words">{e}</span>
                                                </div>
                                              ))}
                                            </div>
                                          </motion.div>
                                        </AnimatePresence>
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            </motion.div>

                            {/* Spacing between nodes */}
                            <div className="h-4 w-full" />
                          </div>
                        );
                      })}
                    </div>

                    {/* Final Answer Rendering */}
                    {answerHtml && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 15, filter: "blur(5px)" }}
                        animate={{ opacity: 1, scale: 1, y: 0, filter: "blur(0px)" }}
                        transition={{ type: "spring", stiffness: 200, damping: 20 }}
                        className="pt-8 border-t border-white/5"
                      >
                        <h3 className="font-bold text-base mb-5 flex items-center gap-3 text-zinc-100">
                          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-green-600 to-green-400 border border-green-500/20 text-white flex items-center justify-center shadow-[0_0_20px_-5px_rgba(74,222,128,0.4)]">
                            <CheckCircle2 className="w-4 h-4" />
                          </div>
                          Analysis Result
                        </h3>
                        {/* We use prose to automatically style markdown/html output generally coming from backend */}
                        <div
                          className="prose prose-sm dark:prose-invert prose-amber max-w-none text-zinc-400 leading-relaxed
                            prose-headings:text-zinc-100 prose-headings:font-bold prose-a:text-amber-500 hover:prose-a:text-amber-400
                            prose-ul:my-3 prose-li:my-1 prose-p:my-3 marker:text-amber-500"
                          dangerouslySetInnerHTML={answerHtml}
                        />
                      </motion.div>
                    )}
                  </div>
                )}
              </ScrollArea>

              {/* Progress Bar (simulated continuous line while loading) */}
              {isLoading && (
                <div className="absolute top-0 left-0 right-0 h-0.5 overflow-hidden z-20">
                  <div className="w-[300%] h-full bg-gradient-to-r from-transparent via-amber-500 to-transparent animate-[shimmer_2s_infinite_linear]" />
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

      </main>

      {/* Global CSS for some micro-animations */}
      <style dangerouslySetInnerHTML={{
        __html: `
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        /* Make sure scrollbars are invisible for specific areas */
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .no-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}} />
    </div>
  );
}

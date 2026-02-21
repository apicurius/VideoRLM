// Types matching the RLM log format

export interface RLMChatCompletion {
  prompt: string | Record<string, unknown>;
  response: string;
  prompt_tokens: number;
  completion_tokens: number;
  execution_time: number;
}

export interface REPLResult {
  stdout: string;
  stderr: string;
  locals: Record<string, unknown>;
  execution_time: number;
  rlm_calls?: RLMChatCompletion[];
}

export interface CodeBlock {
  code: string;
  result: REPLResult | null;
}

export interface RLMIteration {
  type?: string;
  iteration: number;
  timestamp: string;
  prompt: Array<{ role: string; content: string | unknown[] }>;
  response: string;
  code_blocks: CodeBlock[];
  final_answer: string | [string, string] | null;
  iteration_time: number | null;
}

// Metadata saved at the start of a log file about RLM configuration
export interface RLMConfigMetadata {
  timestamp: string | null;
  root_model: string | null;
  max_depth: number | null;
  max_iterations: number | null;
  backend: string | null;
  backend_kwargs: Record<string, unknown> | null;
  environment_type: string | null;
  environment_kwargs: Record<string, unknown> | null;
  other_backends: string[] | null;
  video_path?: string | null;
  fps?: number | null;
  num_segments?: number | null;
  max_frames_per_segment?: number | null;
  resize?: [number, number] | null;
}

export interface RLMLogFile {
  fileName: string;
  filePath: string;
  iterations: RLMIteration[];
  metadata: LogMetadata;
  config: RLMConfigMetadata;
}

export interface LogMetadata {
  totalIterations: number;
  totalCodeBlocks: number;
  totalSubLMCalls: number;
  contextQuestion: string;
  finalAnswer: string | null;
  totalExecutionTime: number;
  hasErrors: boolean;
  isVideoRun: boolean;
  videoPath: string | null;
  totalFramesSent: number;
}

export function extractFinalAnswer(answer: string | [string, string] | null): string | null {
  if (!answer) return null;
  if (Array.isArray(answer)) {
    return answer[1];
  }
  return answer;
}

// ─── KUAVi Trace Types ───────────────────────────────────────────────

export interface KUAViToolCall {
  type: "tool_call";
  timestamp: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_response: unknown;
  duration_ms?: number | null;
  has_error?: boolean;
}

export interface KUAViAgentEvent {
  type: "agent_start" | "agent_stop";
  timestamp: string;
  agent_id: string;
  agent_type: string;
}

export interface KUAViSessionEvent {
  type: "session_start" | "session_end";
  timestamp: string;
  session_id?: string;
  model?: string;
  source?: string;
  reason?: string;
}

export interface KUAViFinalAnswerEvent {
  type: "final_answer";
  timestamp: string;
  text: string;
}

export type KUAViEvent = KUAViToolCall | KUAViAgentEvent | KUAViSessionEvent | KUAViFinalAnswerEvent;

export interface KUAViLogMetadata {
  totalToolCalls: number;
  totalAgentSpawns: number;
  totalFramesExtracted: number;
  totalSearches: number;
  sessionDuration: number;
  model: string | null;
  videoPath: string | null;
  question: string | null;
  toolBreakdown: Record<string, number>;
  hasFrames: boolean;
  isComplete: boolean;
  hasErrors: boolean;
  finalAnswer: string | null;
}

export interface KUAViLogFile {
  fileName: string;
  filePath: string;
  events: KUAViEvent[];
  metadata: KUAViLogMetadata;
}

// ─── Unified Log File Type ───────────────────────────────────────────

export type LogFile = RLMLogFile | KUAViLogFile;

export function isKUAViLog(log: LogFile): log is KUAViLogFile {
  return 'events' in log;
}

export function isRLMLog(log: LogFile): log is RLMLogFile {
  return 'iterations' in log;
}

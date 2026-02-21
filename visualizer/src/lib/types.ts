// Types matching the RLM log format

export interface ModelUsageSummary {
  total_calls: number;
  total_input_tokens: number;
  total_output_tokens: number;
}

export interface UsageSummary {
  model_usage_summaries: Record<string, ModelUsageSummary>;
}

export interface RLMChatCompletion {
  root_model?: string;
  prompt: string | Record<string, unknown>;
  response: string;
  /** Newer log format: nested usage per model */
  usage_summary?: UsageSummary;
  /** Legacy fields (older logs) */
  prompt_tokens?: number;
  completion_tokens?: number;
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

export interface KUAViTokenUsage {
  input_tokens_approx: number;
  output_tokens_approx: number;
}

export interface KUAViToolCall {
  type: "tool_call";
  timestamp: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_response: unknown;
  response_summary?: string | null;
  duration_ms?: number | null;
  has_error?: boolean;
  token_usage?: KUAViTokenUsage | null;
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

export interface KUAViLLMCallEvent {
  type: "llm_call";
  timestamp: string;
  model: string;
  backend: string;
  prompt_summary: string;
  prompt_tokens_approx: number;
  response_summary: string;
  response_tokens_approx: number;
  duration_ms: number;
  has_error: boolean;
  context?: string | null;
  eval_id?: string;
  num_frames?: number;
  response_full?: string;
}

export interface KUAViEvalExecutionEvent {
  type: "eval_execution";
  timestamp: string;
  code: string;
  stdout: string;
  execution_time_ms: number;
  has_error: boolean;
  result_type?: string | null;
  eval_id?: string;
}

/** Emitted by _TraceLogger after kuavi_index_video succeeds. */
export interface KUAViMetadataEvent {
  type: "metadata";
  timestamp: string;
  video_path?: string | null;
  fps?: number | null;
  duration?: number | null;
  num_segments?: number | null;
  num_scenes?: number | null;
  has_embeddings?: boolean;
  has_transcript?: boolean;
}

/** Emitted when there is a >3s gap between consecutive tool calls (new agent turn). */
export interface KUAViTurnStartEvent {
  type: "turn_start";
  timestamp: string;
  turn: number;
  gap_seconds: number;
}

/** Emitted when the model produces reasoning text between tool calls. */
export interface KUAViReasoningEvent {
  type: "reasoning";
  timestamp: string;
  iteration: number;
  text: string;
  token_usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

/** Emitted once at the start of the session with the full system prompt. */
export interface KUAViSystemPromptEvent {
  type: "system_prompt";
  timestamp: string;
  text: string;
}

/** Emitted right after session_start with the user's question for this run. */
export interface KUAViQuestionEvent {
  type: "question";
  timestamp: string;
  text: string;
}

export type KUAViEvent =
  | KUAViToolCall
  | KUAViAgentEvent
  | KUAViSessionEvent
  | KUAViFinalAnswerEvent
  | KUAViLLMCallEvent
  | KUAViEvalExecutionEvent
  | KUAViMetadataEvent
  | KUAViTurnStartEvent
  | KUAViReasoningEvent
  | KUAViSystemPromptEvent
  | KUAViQuestionEvent;

export interface KUAViLogMetadata {
  totalToolCalls: number;
  totalAgentSpawns: number;
  totalAgentTurns: number;
  totalTurns: number;
  totalFramesExtracted: number;
  totalSearches: number;
  sessionDuration: number;
  model: string | null;
  videoPath: string | null;
  videoDuration: number | null;
  question: string | null;
  toolBreakdown: Record<string, number>;
  hasFrames: boolean;
  isComplete: boolean;
  hasErrors: boolean;
  finalAnswer: string | null;
  finalAnswerBrief: string | null;
}

export interface KUAViLogFile {
  fileName: string;
  filePath: string;
  events: KUAViEvent[];
  metadata: KUAViLogMetadata;
  /** Stem of the log file (without .jsonl), used to resolve frame sidecar files */
  logStem?: string;
}

// ─── Unified Log File Type ───────────────────────────────────────────

export type LogFile = RLMLogFile | KUAViLogFile;

export function isKUAViLog(log: LogFile): log is KUAViLogFile {
  return 'events' in log;
}

export function isRLMLog(log: LogFile): log is RLMLogFile {
  return 'iterations' in log;
}

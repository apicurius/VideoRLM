// KUAVi Trace Types (adapted from visualizer/src/lib/types.ts)

export interface KUAViToolCall {
  type: "tool_call";
  timestamp: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_response: unknown;
  response_summary?: string | null;
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

export interface KUAViLLMCallEvent {
  type: "llm_call";
  timestamp: string;
  model: string;
  backend: string;
  prompt_summary: string;
  response_summary: string;
  duration_ms: number;
  has_error: boolean;
}

export interface KUAViMetadataEvent {
  type: "metadata";
  timestamp: string;
  video_path?: string | null;
  duration?: number | null;
  num_segments?: number | null;
  num_scenes?: number | null;
}

export interface KUAViTurnStartEvent {
  type: "turn_start";
  timestamp: string;
  turn: number;
  gap_seconds: number;
}

export interface KUAViReasoningEvent {
  type: "reasoning";
  timestamp: string;
  iteration: number;
  text: string;
}

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
  | KUAViMetadataEvent
  | KUAViTurnStartEvent
  | KUAViReasoningEvent
  | KUAViQuestionEvent;

// RLM Types
export interface CodeBlock {
  code: string;
  result: {
    stdout: string;
    stderr: string;
    locals: Record<string, unknown>;
    execution_time: number;
    rlm_calls?: Array<{ prompt: string; response: string; execution_time: number }>;
  } | null;
}

export interface RLMIteration {
  type?: string;
  iteration: number;
  timestamp: string;
  response: string;
  code_blocks: CodeBlock[];
  final_answer: string | [string, string] | null;
  iteration_time: number | null;
}

// Unified turn model
export interface KUAViTurn {
  index: number;
  reasoning: KUAViReasoningEvent | null;
  toolCalls: KUAViToolCall[];
  llmCalls: KUAViLLMCallEvent[];
}

export interface KUAViLogMetadata {
  totalToolCalls: number;
  totalTurns: number;
  totalFramesExtracted: number;
  totalSearches: number;
  sessionDuration: number;
  model: string | null;
  videoPath: string | null;
  question: string | null;
  toolBreakdown: Record<string, number>;
  hasErrors: boolean;
  finalAnswer: string | null;
}

export interface KUAViLogFile {
  fileName: string;
  events: KUAViEvent[];
  metadata: KUAViLogMetadata;
}

export interface RLMLogFile {
  fileName: string;
  iterations: RLMIteration[];
  metadata: {
    totalIterations: number;
    totalCodeBlocks: number;
    totalSubLMCalls: number;
    contextQuestion: string;
    finalAnswer: string | null;
    totalExecutionTime: number;
    hasErrors: boolean;
  };
}

export type LogFile = RLMLogFile | KUAViLogFile;

export function isKUAViLog(log: LogFile): log is KUAViLogFile {
  return "events" in log;
}

export function isRLMLog(log: LogFile): log is RLMLogFile {
  return "iterations" in log;
}

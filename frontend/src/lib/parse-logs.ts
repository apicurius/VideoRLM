// Log parsing (adapted from visualizer/src/lib/parse-logs.ts)

import type {
  KUAViEvent,
  KUAViToolCall,
  KUAViFinalAnswerEvent,
  KUAViMetadataEvent,
  KUAViQuestionEvent,
  KUAViLLMCallEvent,
  KUAViTurn,
  KUAViReasoningEvent,
  KUAViLogFile,
  KUAViLogMetadata,
  RLMIteration,
  RLMLogFile,
  LogFile,
} from "./trace-types";

export function shortToolName(fullName: string): string {
  return (fullName ?? "")
    .replace(/^mcp__kuavi__kuavi_/, "")
    .replace(/^mcp__kuavi__/, "")
    .replace(/^kuavi_/, "");
}

export function groupEventsIntoTurns(events: KUAViEvent[]): KUAViTurn[] {
  const turns: KUAViTurn[] = [];
  let turnIndex = 0;
  let current: KUAViTurn = { index: 0, reasoning: null, toolCalls: [], llmCalls: [] };

  const hasTurnContent = (t: KUAViTurn) =>
    t.toolCalls.length > 0 || t.llmCalls.length > 0 || t.reasoning !== null;

  for (const event of events) {
    if (event.type === "reasoning") {
      if (hasTurnContent(current)) {
        turns.push(current);
        turnIndex++;
      }
      current = {
        index: turnIndex,
        reasoning: event as KUAViReasoningEvent,
        toolCalls: [],
        llmCalls: [],
      };
    } else if (event.type === "turn_start") {
      if (hasTurnContent(current)) {
        turns.push(current);
        turnIndex++;
        current = { index: turnIndex, reasoning: null, toolCalls: [], llmCalls: [] };
      }
    } else if (event.type === "tool_call") {
      current.toolCalls.push(event as KUAViToolCall);
    } else if (event.type === "llm_call") {
      current.llmCalls.push(event as KUAViLLMCallEvent);
    }
  }

  if (hasTurnContent(current)) turns.push(current);
  return turns;
}

function parseKUAViJSONL(content: string): KUAViEvent[] {
  return content
    .trim()
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => {
      try {
        const event = JSON.parse(line) as KUAViEvent;
        if (event.type === "tool_call") {
          const tc = event as KUAViToolCall;
          if (typeof tc.tool_response === "string") {
            try { tc.tool_response = JSON.parse(tc.tool_response as string); } catch { /* keep */ }
          }
          if (typeof tc.tool_input === "string") {
            try { tc.tool_input = JSON.parse(tc.tool_input as unknown as string); } catch { /* keep */ }
          }
        }
        return event;
      } catch {
        return null;
      }
    })
    .filter((e): e is KUAViEvent => e !== null);
}

function computeKUAViMetadata(events: KUAViEvent[]): KUAViLogMetadata {
  const toolCalls = events.filter((e): e is KUAViToolCall => e.type === "tool_call");
  const metadataEvent = events.find((e): e is KUAViMetadataEvent => e.type === "metadata");

  const toolBreakdown: Record<string, number> = {};
  for (const tc of toolCalls) {
    const name = shortToolName(tc.tool_name);
    toolBreakdown[name] = (toolBreakdown[name] || 0) + 1;
  }

  const totalSearches = toolCalls.filter(
    (tc) => tc.tool_name.includes("search") || tc.tool_name.includes("discriminative_vqa")
  ).length;

  const frameCalls = toolCalls.filter(
    (tc) => tc.tool_name.includes("extract_frames") || tc.tool_name.includes("zoom_frames")
  );
  let totalFramesExtracted = 0;
  for (const fc of frameCalls) {
    const resp = fc.tool_response;
    const unwrapped =
      typeof resp === "object" && resp !== null && "result" in resp
        ? (resp as Record<string, unknown>).result
        : resp;
    if (Array.isArray(unwrapped)) totalFramesExtracted += unwrapped.length;
  }

  let videoPath: string | null = metadataEvent?.video_path ?? null;
  if (!videoPath) {
    const indexCall = toolCalls.find((tc) => tc.tool_name.includes("index_video"));
    if (indexCall?.tool_input?.video_path) videoPath = String(indexCall.tool_input.video_path);
  }

  let question: string | null = null;
  const qEvent = events.find((e): e is KUAViQuestionEvent => e.type === "question");
  if (qEvent?.text) question = qEvent.text;
  if (!question) {
    const searchCall = toolCalls.find((tc) => tc.tool_name.includes("search_video"));
    if (searchCall?.tool_input?.query) question = String(searchCall.tool_input.query);
  }

  let sessionDuration = 0;
  if (toolCalls.length >= 2) {
    const first = new Date(toolCalls[0].timestamp).getTime();
    const last = new Date(toolCalls[toolCalls.length - 1].timestamp).getTime();
    sessionDuration = Math.max(0, (last - first) / 1000);
  }

  const sessionStart = events.find((e) => e.type === "session_start");
  const model = sessionStart && "model" in sessionStart ? (sessionStart.model ?? null) : null;

  const finalAnswerEvents = events.filter(
    (e): e is KUAViFinalAnswerEvent => e.type === "final_answer"
  );
  const finalAnswer = finalAnswerEvents.length > 0 ? finalAnswerEvents[0].text : null;

  const hasErrors = toolCalls.some((tc) => {
    if (tc.has_error) return true;
    const respStr = typeof tc.tool_response === "string" ? tc.tool_response : JSON.stringify(tc.tool_response ?? "");
    return respStr.includes("Error:");
  });

  return {
    totalToolCalls: toolCalls.length,
    totalTurns: groupEventsIntoTurns(events).length,
    totalFramesExtracted,
    totalSearches,
    sessionDuration,
    model,
    videoPath,
    question,
    toolBreakdown,
    hasErrors,
    finalAnswer,
  };
}

export function parseLogFile(fileName: string, content: string): LogFile {
  const firstLine = content.trim().split("\n")[0];
  if (!firstLine) {
    return {
      fileName,
      iterations: [],
      metadata: {
        totalIterations: 0,
        totalCodeBlocks: 0,
        totalSubLMCalls: 0,
        contextQuestion: "",
        finalAnswer: null,
        totalExecutionTime: 0,
        hasErrors: false,
      },
    } as RLMLogFile;
  }

  try {
    const first = JSON.parse(firstLine);
    const kuaviTypes = [
      "tool_call", "session_start", "agent_start", "session_end",
      "agent_stop", "final_answer", "turn_start", "reasoning",
      "system_prompt", "question",
    ];
    if (kuaviTypes.includes(first.type) || (first.type === "metadata" && first.video_path != null)) {
      const events = parseKUAViJSONL(content);
      return { fileName, events, metadata: computeKUAViMetadata(events) } as KUAViLogFile;
    }
  } catch { /* fall through */ }

  // RLM format
  const lines = content.trim().split("\n").filter((l) => l.trim());
  const iterations: RLMIteration[] = [];
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      if (parsed.type === "iteration" || parsed.iteration != null) {
        iterations.push(parsed);
      }
    } catch { /* skip */ }
  }

  let totalCodeBlocks = 0;
  let totalSubLMCalls = 0;
  let totalExecutionTime = 0;
  let hasErrors = false;
  let finalAnswer: string | null = null;

  for (const iter of iterations) {
    totalCodeBlocks += iter.code_blocks.length;
    if (iter.iteration_time != null) totalExecutionTime += iter.iteration_time;
    for (const block of iter.code_blocks) {
      if (block.result?.stderr) hasErrors = true;
      if (block.result?.rlm_calls) totalSubLMCalls += block.result.rlm_calls.length;
    }
    if (iter.final_answer) {
      finalAnswer = Array.isArray(iter.final_answer) ? iter.final_answer[1] : iter.final_answer;
    }
  }

  return {
    fileName,
    iterations,
    metadata: {
      totalIterations: iterations.length,
      totalCodeBlocks,
      totalSubLMCalls,
      contextQuestion: "",
      finalAnswer,
      totalExecutionTime,
      hasErrors,
    },
  } as RLMLogFile;
}

import { RLMIteration, RLMLogFile, LogMetadata, RLMConfigMetadata, extractFinalAnswer, KUAViEvent, KUAViLogFile, KUAViLogMetadata, KUAViToolCall, KUAViFinalAnswerEvent, KUAViMetadataEvent, KUAViTurnStartEvent, KUAViReasoningEvent, KUAViSystemPromptEvent, KUAViQuestionEvent, LogFile } from './types';

// Extract the context variable from code block locals
export function extractContextVariable(iterations: RLMIteration[]): string | null {
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      if (block.result?.locals?.context) {
        const ctx = block.result.locals.context;
        if (typeof ctx === 'string') {
          return ctx;
        }
      }
    }
  }
  return null;
}

// Default config when metadata is not present (backwards compatibility)
function getDefaultConfig(): RLMConfigMetadata {
  return {
    timestamp: null,
    root_model: null,
    max_depth: null,
    max_iterations: null,
    backend: null,
    backend_kwargs: null,
    environment_type: null,
    environment_kwargs: null,
    other_backends: null,
    video_path: null,
    fps: null,
    num_segments: null,
    max_frames_per_segment: null,
    resize: null,
  };
}

export interface ParsedJSONL {
  iterations: RLMIteration[];
  config: RLMConfigMetadata;
}

export function parseJSONL(content: string): ParsedJSONL {
  const lines = content.trim().split('\n').filter(line => line.trim());
  const iterations: RLMIteration[] = [];
  let config: RLMConfigMetadata = getDefaultConfig();
  
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      
      // Check if this is a metadata entry
      if (parsed.type === 'metadata') {
        config = {
          timestamp: parsed.timestamp ?? null,
          root_model: parsed.root_model ?? null,
          max_depth: parsed.max_depth ?? null,
          max_iterations: parsed.max_iterations ?? null,
          backend: parsed.backend ?? null,
          backend_kwargs: parsed.backend_kwargs ?? null,
          environment_type: parsed.environment_type ?? null,
          environment_kwargs: parsed.environment_kwargs ?? null,
          other_backends: parsed.other_backends ?? null,
          video_path: parsed.video_path ?? null,
          fps: parsed.fps ?? null,
          num_segments: parsed.num_segments ?? null,
          max_frames_per_segment: parsed.max_frames_per_segment ?? null,
          resize: parsed.resize ?? null,
        };
      } else if (parsed.type === 'iteration' || parsed.iteration != null) {
        // This is an iteration entry
        iterations.push(parsed as RLMIteration);
      }
    } catch (e) {
    }
  }
  
  return { iterations, config };
}

/** Safely get string representation of message content (handles multimodal arrays) */
export function getContentText(content: string | unknown[]): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    // Extract text parts from multimodal content arrays
    return content
      .filter((part): part is { type: string; text: string } =>
        typeof part === 'object' && part !== null && 'text' in part && typeof (part as Record<string, unknown>).text === 'string'
      )
      .map(part => part.text)
      .join('\n');
  }
  return String(content ?? '');
}

export function extractContextQuestion(iterations: RLMIteration[]): string {
  if (iterations.length === 0) return 'No context found';

  const firstIteration = iterations[0];
  const prompt = firstIteration.prompt;

  // Look for user message that contains the actual question
  for (const msg of prompt) {
    if (msg.role === 'user' && msg.content) {
      const text = getContentText(msg.content);
      // Try to extract quoted query
      const queryMatch = text.match(/original query: "([^"]+)"/);
      if (queryMatch) {
        return queryMatch[1];
      }

      // Check if it contains the actual query pattern
      if (text.includes('answer the prompt')) {
        continue;
      }

      // Take first substantial user message
      if (text.length > 50 && text.length < 500) {
        return text.slice(0, 200) + (text.length > 200 ? '...' : '');
      }
    }
  }

  // Fallback: look in system prompt for context info
  const systemMsg = prompt.find(m => m.role === 'system');
  if (systemMsg?.content) {
    const text = getContentText(systemMsg.content);
    const contextMatch = text.match(/context variable.*?:(.*?)(?:\n|$)/i);
    if (contextMatch) {
      return contextMatch[1].trim().slice(0, 200);
    }
  }
  
  // Check code block output for actual context
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      if (block.result?.locals?.context) {
        const ctx = block.result.locals.context;
        if (typeof ctx === 'string' && ctx.length < 500) {
          return ctx;
        }
      }
    }
  }
  
  return 'Context available in REPL environment';
}

/** Check if stderr content represents a real error (not model loading progress) */
export function isRealStderrError(stderr: string): boolean {
  if (!stderr) return false;
  const hasErrorPattern = /Traceback|Error:|Exception:|error:|FAILED/i.test(stderr);
  const isModelLoading = /Loading weights|Materializing param|tokenizer_config|model\.safetensors|Downloading|from_pretrained/i.test(stderr);
  return hasErrorPattern && !isModelLoading;
}

// Count __image__ occurrences in a string (serialized prompt/call data)
export function countImageTags(text: string): number {
  let count = 0;
  let idx = 0;
  while ((idx = text.indexOf('"__image__"', idx)) !== -1) {
    count++;
    idx += 11;
  }
  return count;
}

export function computeMetadata(iterations: RLMIteration[], config: RLMConfigMetadata): LogMetadata {
  let totalCodeBlocks = 0;
  let totalSubLMCalls = 0;
  let totalExecutionTime = 0;
  let hasErrors = false;
  let finalAnswer: string | null = null;
  let totalFramesSent = 0;

  // Detect video run: check config for video_path or video-related backend_kwargs,
  // or check if system prompt mentions video analysis
  let isVideoRun = false;
  if (config.video_path) {
    isVideoRun = true;
  } else if (config.backend_kwargs) {
    const bkStr = JSON.stringify(config.backend_kwargs);
    if (bkStr.includes('video') || bkStr.includes('fps') || bkStr.includes('segment')) {
      isVideoRun = true;
    }
  }
  // Also check the system prompt of the first iteration for video-related content
  if (!isVideoRun && iterations.length > 0) {
    const systemMsg = iterations[0].prompt.find(m => m.role === 'system');
    if (systemMsg?.content) {
      const contentStr = getContentText(systemMsg.content);
      if (/analyzing a video|video.*frames|extracted frames/i.test(contentStr)) {
        isVideoRun = true;
      }
    }
  }

  for (const iter of iterations) {
    totalCodeBlocks += iter.code_blocks.length;

    // Use iteration_time if available, otherwise sum code block times
    if (iter.iteration_time != null) {
      totalExecutionTime += iter.iteration_time;
    } else {
      for (const block of iter.code_blocks) {
        if (block.result) {
          totalExecutionTime += block.result.execution_time || 0;
        }
      }
    }

    for (const block of iter.code_blocks) {
      if (block.result) {
        if (block.result.stderr && isRealStderrError(block.result.stderr)) {
          hasErrors = true;
        }
        if (block.result.rlm_calls) {
          totalSubLMCalls += block.result.rlm_calls.length;
          // Count frames sent in sub-LM calls
          if (isVideoRun) {
            for (const call of block.result.rlm_calls) {
              const promptStr = typeof call.prompt === 'string'
                ? call.prompt
                : JSON.stringify(call.prompt);
              totalFramesSent += countImageTags(promptStr);
            }
          }
        }
      }
    }

    // Count frames in iteration prompts
    if (isVideoRun) {
      for (const msg of iter.prompt) {
        const contentStr = typeof msg.content === 'string'
          ? msg.content
          : JSON.stringify(msg.content);
        totalFramesSent += countImageTags(contentStr);
      }
    }

    if (iter.final_answer) {
      finalAnswer = extractFinalAnswer(iter.final_answer);
    }
  }

  return {
    totalIterations: iterations.length,
    totalCodeBlocks,
    totalSubLMCalls,
    contextQuestion: extractContextQuestion(iterations),
    finalAnswer,
    totalExecutionTime,
    hasErrors,
    isVideoRun,
    videoPath: config.video_path ?? null,
    totalFramesSent,
  };
}

// ─── KUAVi Turn Model ────────────────────────────────────────────────

export interface KUAViTurn {
  index: number;
  reasoning: KUAViReasoningEvent | null;
  toolCalls: KUAViToolCall[];
  tokenUsage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
}

/**
 * Groups a flat list of KUAVi events into turns.
 *
 * A turn is one LLM response cycle: optional reasoning text followed by
 * zero or more tool calls. Turn boundaries are defined by:
 *   - A `reasoning` event — starts a new turn carrying that reasoning
 *   - A `turn_start` event — starts a new turn with null reasoning
 *
 * Turn 0 collects any tool calls that appear before the first reasoning
 * or turn_start event (null reasoning).
 */
export function groupEventsIntoTurns(events: KUAViEvent[]): KUAViTurn[] {
  const turns: KUAViTurn[] = [];
  let turnIndex = 0;
  let currentTurn: KUAViTurn = { index: 0, reasoning: null, toolCalls: [] };

  for (const event of events) {
    if (event.type === 'reasoning') {
      // Push current turn if it has content
      if (currentTurn.toolCalls.length > 0 || currentTurn.reasoning !== null) {
        turns.push(currentTurn);
        turnIndex++;
      }
      const usage = (event as KUAViReasoningEvent).token_usage;
      currentTurn = {
        index: turnIndex,
        reasoning: event as KUAViReasoningEvent,
        toolCalls: [],
        tokenUsage: usage
          ? {
              prompt_tokens: usage.prompt_tokens ?? 0,
              completion_tokens: usage.completion_tokens ?? 0,
              total_tokens: usage.total_tokens ?? 0,
            }
          : null,
      };
    } else if (event.type === 'turn_start') {
      // turn_start marks a boundary if current turn has content
      if (currentTurn.toolCalls.length > 0 || currentTurn.reasoning !== null) {
        turns.push(currentTurn);
        turnIndex++;
        currentTurn = { index: turnIndex, reasoning: null, toolCalls: [] };
      }
    } else if (event.type === 'tool_call') {
      currentTurn.toolCalls.push(event as KUAViToolCall);
    }
  }

  // Flush the last turn if it has content
  if (currentTurn.toolCalls.length > 0 || currentTurn.reasoning !== null) {
    turns.push(currentTurn);
  }

  return turns;
}

// ─── KUAVi JSONL Parsing ─────────────────────────────────────────────

export function parseKUAViJSONL(content: string): KUAViEvent[] {
  return content
    .trim()
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => {
      try {
        const event = JSON.parse(line) as KUAViEvent;
        // Hook script double-serializes tool_response via jq — parse it if it's a string
        if (event.type === 'tool_call' && typeof (event as KUAViToolCall).tool_response === 'string') {
          try {
            (event as KUAViToolCall).tool_response = JSON.parse((event as KUAViToolCall).tool_response as string);
          } catch {
            // Keep as string if not valid JSON
          }
        }
        // Same for tool_input if double-serialized
        if (event.type === 'tool_call' && typeof (event as KUAViToolCall).tool_input === 'string') {
          try {
            (event as KUAViToolCall).tool_input = JSON.parse((event as KUAViToolCall).tool_input as unknown as string);
          } catch {
            // Keep as string
          }
        }
        return event;
      } catch {
        return null;
      }
    })
    .filter((e): e is KUAViEvent => e !== null);
}

/** Strip the mcp__kuavi__kuavi_ or mcp__kuavi__ prefix to get a short tool name */
export function shortToolName(fullName: string): string {
  return (fullName ?? '')
    .replace(/^mcp__kuavi__kuavi_/, '')
    .replace(/^mcp__kuavi__/, '');
}

export function computeKUAViMetadata(events: KUAViEvent[]): KUAViLogMetadata {
  const toolCalls = events.filter((e): e is KUAViToolCall => e.type === 'tool_call');
  const agentStarts = events.filter((e) => e.type === 'agent_start');
  const sessionStart = events.find((e) => e.type === 'session_start');
  const metadataEvent = events.find((e): e is KUAViMetadataEvent => e.type === 'metadata');
  const turnStarts = events.filter((e): e is KUAViTurnStartEvent => e.type === 'turn_start');

  // Tool breakdown
  const toolBreakdown: Record<string, number> = {};
  for (const tc of toolCalls) {
    const name = shortToolName(tc.tool_name);
    toolBreakdown[name] = (toolBreakdown[name] || 0) + 1;
  }

  // Count searches and frames
  const totalSearches = toolCalls.filter(
    (tc) => tc.tool_name.includes('search') || tc.tool_name.includes('discriminative_vqa')
  ).length;

  const frameCalls = toolCalls.filter((tc) => tc.tool_name.includes('extract_frames') || tc.tool_name.includes('zoom_frames'));
  let totalFramesExtracted = 0;
  for (const fc of frameCalls) {
    const resp = fc.tool_response;
    // MCP tools wrap results in {result: [...]}
    const unwrapped = typeof resp === 'object' && resp !== null && 'result' in resp
      ? (resp as Record<string, unknown>).result
      : resp;
    if (Array.isArray(unwrapped)) {
      totalFramesExtracted += unwrapped.length;
    } else if (typeof resp !== 'string' || !resp.startsWith('Error')) {
      totalFramesExtracted += 1;
    }
  }

  // Detect video path: prefer metadata event, fall back to index_video input
  let videoPath: string | null = metadataEvent?.video_path ?? null;
  if (!videoPath) {
    const indexCall = toolCalls.find((tc) => tc.tool_name.includes('index_video'));
    if (indexCall?.tool_input?.video_path) {
      videoPath = String(indexCall.tool_input.video_path);
    }
  }

  // Video duration from metadata event (if available)
  const videoDuration: number | null = metadataEvent?.duration ?? null;

  // Detect question: prefer explicit question event, then analyze_shards param, then search query
  let question: string | null = null;
  const questionEvent = events.find((e): e is KUAViQuestionEvent => e.type === 'question');
  if (questionEvent?.text) {
    question = questionEvent.text;
  }
  if (!question) {
    const shardCall = toolCalls.find((tc) => tc.tool_name.includes('analyze_shards'));
    if (shardCall?.tool_input?.question) {
      question = String(shardCall.tool_input.question);
    }
  }
  if (!question) {
    const searchCall = toolCalls.find((tc) => tc.tool_name.includes('search_video'));
    if (searchCall?.tool_input?.query) {
      question = String(searchCall.tool_input.query);
    }
  }

  // Session duration — use tool call span, not total event span
  // (agent_stop / final_answer events from teammates can trail hours later)
  let sessionDuration = 0;
  if (toolCalls.length >= 2) {
    const first = new Date(toolCalls[0].timestamp).getTime();
    const last = new Date(toolCalls[toolCalls.length - 1].timestamp).getTime();
    sessionDuration = Math.max(0, (last - first) / 1000);
  }

  // Model from session_start
  const model = sessionStart && 'model' in sessionStart ? (sessionStart.model ?? null) : null;

  // Agent turns: count turn_start events + 1 for the initial implicit turn (if any tool calls)
  const totalAgentTurns = turnStarts.length + (toolCalls.length > 0 ? 1 : 0);

  // isComplete: has a final_answer event, OR has session_end with reason "complete",
  // OR has index_video + follow-up tool calls
  const hasFinalAnswer = events.some((e) => e.type === 'final_answer');
  const hasSessionEnd = events.some(
    (e) => e.type === 'session_end' && 'reason' in e && e.reason === 'complete'
  );
  const hasIndex = toolCalls.some((tc) => tc.tool_name.includes('index_video'));
  const hasFollowUp = toolCalls.some(
    (tc) =>
      tc.tool_name.includes('search') ||
      tc.tool_name.includes('discriminative_vqa') ||
      tc.tool_name.includes('scene_list')
  );
  const isComplete = hasFinalAnswer || hasSessionEnd || (hasIndex && hasFollowUp);

  // hasErrors: check response content first for known non-errors, then check has_error flag
  const hasErrors = toolCalls.some((tc) => {
    const resp = tc.tool_response;
    const respStr = typeof resp === 'string' ? resp : JSON.stringify(resp ?? '');
    // Truncation warnings from Claude Code are not real errors
    if (respStr.includes('exceeds maximum allowed tokens')) return false;
    if (tc.has_error) return true;
    return respStr.includes('Error:') || respStr.includes('BUDGET EXCEEDED');
  });

  // Extract final answer: prefer the first final_answer that appears after the last tool call.
  // In interactive sessions the Stop hook fires multiple times (every conversation pause),
  // so taking the last final_answer captures unrelated conversational messages.
  const finalAnswerEvents = events.filter(
    (e): e is KUAViFinalAnswerEvent => e.type === 'final_answer'
  );
  let finalAnswer: string | null = null;
  if (finalAnswerEvents.length > 0) {
    if (toolCalls.length > 0) {
      const lastToolTime = new Date(toolCalls[toolCalls.length - 1].timestamp).getTime();
      // Find the first final_answer after the last tool call
      const afterTool = finalAnswerEvents.find(
        (e) => new Date(e.timestamp).getTime() >= lastToolTime
      );
      finalAnswer = (afterTool ?? finalAnswerEvents[0]).text;
    } else {
      finalAnswer = finalAnswerEvents[0].text;
    }
  }

  // Compute turn count via groupEventsIntoTurns
  const totalTurns = groupEventsIntoTurns(events).length;

  return {
    totalToolCalls: toolCalls.length,
    totalAgentSpawns: agentStarts.length,
    totalAgentTurns,
    totalTurns,
    totalFramesExtracted,
    totalSearches,
    sessionDuration,
    model,
    videoPath,
    videoDuration,
    question,
    toolBreakdown,
    hasFrames: totalFramesExtracted > 0,
    isComplete,
    hasErrors,
    finalAnswer,
  };
}

export function parseLogFile(fileName: string, content: string): LogFile {
  const firstLine = content.trim().split('\n')[0];
  if (!firstLine) {
    // Empty file, return empty RLM log
    return {
      fileName,
      filePath: fileName,
      iterations: [],
      metadata: computeMetadata([], getDefaultConfig()),
      config: getDefaultConfig(),
    };
  }

  try {
    const first = JSON.parse(firstLine);

    // KUAVi format detection
    if (
      first.type === 'tool_call' ||
      first.type === 'session_start' ||
      first.type === 'agent_start' ||
      first.type === 'session_end' ||
      first.type === 'agent_stop' ||
      first.type === 'final_answer' ||
      first.type === 'turn_start' ||
      first.type === 'reasoning' ||
      first.type === 'system_prompt' ||
      first.type === 'question' ||
      // metadata-only files from MCP server (session_start comes first normally,
      // but guard against edge cases where metadata could appear early)
      (first.type === 'metadata' && first.video_path != null)
    ) {
      const events = parseKUAViJSONL(content);
      const logStem = fileName.replace(/\.jsonl$/, '');
      return {
        fileName,
        filePath: fileName,
        events,
        metadata: computeKUAViMetadata(events),
        logStem,
      };
    }
  } catch {
    // Fall through to RLM parsing
  }

  // RLM format (default)
  const { iterations, config } = parseJSONL(content);
  const metadata = computeMetadata(iterations, config);

  return {
    fileName,
    filePath: fileName,
    iterations,
    metadata,
    config,
  };
}


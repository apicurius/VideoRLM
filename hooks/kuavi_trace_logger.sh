#!/usr/bin/env bash
# Unified KUAVi trace hook — appends conversation events to the MCP server's trace file.
#
# The MCP server (_TraceLogger in mcp_server.py) writes rich tool_call, llm_call,
# eval_execution, metadata, turn_start, and session events. This hook supplements
# with conversation-level events the MCP server cannot see:
#   - question        (from UserPromptSubmit)
#   - agent_start/stop (from SubagentStart/SubagentStop)
#   - final_answer     (from Stop)
#
# Coordination: MCP server publishes its trace path to logs/.kuavi_mcp_trace.
# This hook reads that path and appends to the SAME file.
#
# NOTE: Do NOT use set -e — jq operations on missing fields must not abort the script.

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null || echo "unknown")
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // "unknown"' 2>/dev/null || echo "unknown")
LOG_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")

# State files
MCP_TRACE_FILE="$LOG_DIR/.kuavi_mcp_trace"   # Published by MCP server _TraceLogger
FINAL_GUARD="$LOG_DIR/.kuavi_final_$SESSION_ID"
QUESTION_STASH="$LOG_DIR/.kuavi_question_$SESSION_ID"

# Get the MCP server's current trace file. Returns empty string if not available.
mcp_trace() {
  if [ -f "$MCP_TRACE_FILE" ]; then
    cat "$MCP_TRACE_FILE" 2>/dev/null
  fi
}

case "$EVENT" in
  PostToolUse)
    TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""' 2>/dev/null || echo "")

    if [[ "$TOOL_NAME" == *index_video* ]]; then
      # MCP server creates a new trace file on index_video. Wait briefly for it.
      sleep 0.3
      LOG_FILE=$(mcp_trace)

      if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        # Carry forward any stashed question into the new trace
        if [ -f "$QUESTION_STASH" ]; then
          cat "$QUESTION_STASH" >> "$LOG_FILE" 2>/dev/null
        fi
      fi

      # Reset final_answer guard for new run
      rm -f "$FINAL_GUARD"
    fi
    ;;

  SubagentStart)
    LOG_FILE=$(mcp_trace)
    if [ -n "$LOG_FILE" ]; then
      echo "$INPUT" | jq -c '{type:"agent_start",timestamp:"'"$TIMESTAMP"'",agent_id:.agent_id,agent_type:.agent_type}' >> "$LOG_FILE" 2>/dev/null
    fi
    ;;

  SubagentStop)
    LOG_FILE=$(mcp_trace)
    if [ -n "$LOG_FILE" ]; then
      echo "$INPUT" | jq -c '{type:"agent_stop",timestamp:"'"$TIMESTAMP"'",agent_id:.agent_id,agent_type:.agent_type}' >> "$LOG_FILE" 2>/dev/null
    fi
    ;;

  UserPromptSubmit)
    PROMPT=$(echo "$INPUT" | jq -r '.prompt // ""' 2>/dev/null || echo "")
    if [ -n "$PROMPT" ] && [ "$PROMPT" != "null" ]; then
      # Build the question event JSON
      QUESTION_EVENT=$(printf '{"type":"question","timestamp":"%s","text":%s}' \
        "$TIMESTAMP" \
        "$(echo "$PROMPT" | jq -Rs '.')")

      # Stash for carry-forward (in case index_video creates a new trace later)
      echo "$QUESTION_EVENT" > "$QUESTION_STASH"

      # Append to current MCP trace if one exists
      LOG_FILE=$(mcp_trace)
      if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        echo "$QUESTION_EVENT" >> "$LOG_FILE" 2>/dev/null
      fi
    fi
    ;;

  Stop)
    # Emit final_answer ONCE per trace on the first Stop after tool calls.
    # Extract only the actual analysis answer, stripping meta-commentary.
    LOG_FILE=$(mcp_trace)
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ] && [ ! -f "$FINAL_GUARD" ]; then
      RAW_ANSWER=$(echo "$INPUT" | jq -r '.last_assistant_message // ""' 2>/dev/null || echo "")
      if [ -n "$RAW_ANSWER" ] && [ "$RAW_ANSWER" != "null" ]; then
        touch "$FINAL_GUARD"
        # Extract just the analysis answer, drop meta-sections
        ANSWER=$(echo "$RAW_ANSWER" | python3 -c "
import sys, re

text = sys.stdin.read().strip()
if not text:
    sys.exit(0)

# Split by markdown headings (## or ###)
sections = re.split(r'^(#{2,3}\s+.+)$', text, flags=re.MULTILINE)

# If no headings, check if the text is meta-commentary or actual analysis
if len(sections) <= 1:
    # Reject pure meta-text about tooling/infrastructure (not video analysis)
    meta_re = re.compile(
        r'\b(trace file|jsonl|tool.?calls?|turn_start|session_start|'
        r'hook|mcp|visualizer|localhost|dev server|npm run|'
        r'what the visualizer shows|three fixes|hot.?reload)\b', re.I)
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip()]
    meta_sentences = sum(1 for s in sentences if meta_re.search(s))
    if sentences and meta_sentences / len(sentences) > 0.3:
        sys.exit(0)
    print(text)
    sys.exit(0)

# Skip meta-sections: verification, logs, visualizer, logging, trace, session
skip_re = re.compile(
    r'\b(verification|verif|visualizer|logging|how the logs|trace|session|new logging|'
    r'what the visualizer|current session|tool call timeline|frame sidecar|'
    r'full eval|eval.*linkage|frame count)\b', re.I)

# Collect answer sections (heading + body pairs)
answer_parts = []
i = 0
while i < len(sections):
    part = sections[i]
    if re.match(r'^#{2,3}\s+', part):
        heading = part
        body = sections[i + 1] if i + 1 < len(sections) else ''
        if not skip_re.search(heading) and not skip_re.search(body[:200]):
            answer_parts.append((heading + '\n' + body).strip())
        i += 2
    else:
        stripped = part.strip()
        if stripped and not skip_re.search(stripped) and len(stripped) > 30:
            if not re.match(r'^(the visualizer|let me|here|i\'ve|now let)', stripped, re.I):
                answer_parts.append(stripped)
        i += 1

if answer_parts:
    print('\n\n'.join(answer_parts))
" 2>/dev/null || echo "$RAW_ANSWER")
        if [ -n "$ANSWER" ]; then
          printf '{"type":"final_answer","timestamp":"%s","text":%s}\n' \
            "$TIMESTAMP" \
            "$(echo "$ANSWER" | jq -Rs '.')" >> "$LOG_FILE" 2>/dev/null
        fi
      fi
    fi
    # Clean up stash
    rm -f "$QUESTION_STASH"
    ;;
esac
exit 0

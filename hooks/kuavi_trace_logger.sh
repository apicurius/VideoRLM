#!/usr/bin/env bash
# Async hook — logs KUAVi MCP tool calls and lifecycle events to JSONL
# Each index_video call starts a NEW trace file for per-run isolation.
# NOTE: Do NOT use set -e — jq operations on missing fields must not abort the script.

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null || echo "unknown")
SESSION_SUFFIX="${SESSION_ID:0:8}"
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // "unknown"' 2>/dev/null || echo "unknown")
LOG_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
EPOCH_MS=$(python3 -c "import time; print(int(time.time()*1000))" 2>/dev/null || date +%s000)
STATE_FILE="$LOG_DIR/.kuavi_run_$SESSION_ID"
TIMING_FILE="$LOG_DIR/.kuavi_timing_$SESSION_ID"

# Create a new run file with a fresh timestamp and update the state file
new_run_file() {
  local file_ts
  file_ts=$(date +"%Y-%m-%d_%H-%M-%S")
  local run_file="$LOG_DIR/kuavi_${file_ts}_${SESSION_SUFFIX}.jsonl"
  echo "$run_file" > "$STATE_FILE"
  echo "$run_file"
}

# Get the current run file from state, or create one if missing
current_run_file() {
  if [ -f "$STATE_FILE" ]; then
    cat "$STATE_FILE"
  else
    new_run_file
  fi
}

# Compute duration_ms from the last recorded timestamp
compute_duration() {
  if [ -f "$TIMING_FILE" ]; then
    local last_ms
    last_ms=$(cat "$TIMING_FILE" 2>/dev/null || echo "0")
    if [ "$last_ms" -gt 0 ] 2>/dev/null; then
      echo $(( EPOCH_MS - last_ms ))
    else
      echo "null"
    fi
  else
    echo "null"
  fi
}

case "$EVENT" in
  PostToolUse)
    TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""' 2>/dev/null || echo "")
    TURN_COUNT_FILE="$LOG_DIR/.kuavi_turn_$SESSION_ID"

    if [[ "$TOOL_NAME" == *index_video* ]]; then
      LOG_FILE=$(new_run_file)
      # Emit session_start event for the new run
      printf '{"type":"session_start","timestamp":"%s","source":"claude-code"}\n' \
        "$TIMESTAMP" >> "$LOG_FILE" 2>/dev/null
      # Reset timing and turn counter for new run
      rm -f "$TIMING_FILE"
      echo "1" > "$TURN_COUNT_FILE"
    else
      LOG_FILE=$(current_run_file)
    fi

    # Detect agent turn boundary: gap > 3000ms since last tool call
    if [ -f "$TIMING_FILE" ]; then
      LAST_MS=$(cat "$TIMING_FILE" 2>/dev/null || echo "0")
      if [ "$LAST_MS" -gt 0 ] 2>/dev/null && [[ "$TOOL_NAME" != *index_video* ]]; then
        GAP=$(( EPOCH_MS - LAST_MS ))
        if [ "$GAP" -gt 3000 ]; then
          PREV_TURN=$(cat "$TURN_COUNT_FILE" 2>/dev/null || echo "0")
          NEW_TURN=$(( PREV_TURN + 1 ))
          echo "$NEW_TURN" > "$TURN_COUNT_FILE"
          GAP_SEC=$(python3 -c "print(round($GAP/1000, 1))" 2>/dev/null || echo "0")
          printf '{"type":"turn_start","timestamp":"%s","turn":%d,"gap_seconds":%s}\n' \
            "$TIMESTAMP" "$NEW_TURN" "$GAP_SEC" >> "$LOG_FILE" 2>/dev/null
        fi
      fi
    fi

    # Compute duration since last PostToolUse
    DURATION=$(compute_duration)
    # Update timing state for next call
    echo "$EPOCH_MS" > "$TIMING_FILE"

    # Detect errors in tool response (exclude Claude Code truncation warnings)
    HAS_ERROR=$(echo "$INPUT" | jq -r '
      (.tool_response // "") |
      if test("exceeds maximum allowed tokens") then false
      elif test("(?i)(error:|exception:|traceback|failed to|cannot |could not )") then true
      else false end
    ' 2>/dev/null || echo "false")

    # Build the tool_call event with duration and error status
    echo "$INPUT" | jq -c \
      --argjson duration "$DURATION" \
      --argjson has_error "$HAS_ERROR" \
      '{type:"tool_call",timestamp:"'"$TIMESTAMP"'",tool_name:.tool_name,tool_input:.tool_input,tool_response:.tool_response,duration_ms:$duration,has_error:$has_error}' \
      >> "$LOG_FILE" 2>/dev/null

    # After index_video: emit a metadata event with structured video info
    if [[ "$TOOL_NAME" == *index_video* ]]; then
      RESP_STR=$(echo "$INPUT" | jq -r '.tool_response // ""' 2>/dev/null || echo "")
      if [ -n "$RESP_STR" ]; then
        echo "$RESP_STR" | python3 -c "
import sys, json
try:
    raw = sys.stdin.read().strip()
    resp = json.loads(raw) if raw.startswith('{') else json.loads(json.loads(raw))
    meta = {
        'type': 'metadata',
        'timestamp': '$TIMESTAMP',
        'video_path': resp.get('video_path'),
        'fps': resp.get('fps'),
        'duration': resp.get('duration'),
        'num_segments': resp.get('segments'),
        'num_scenes': resp.get('scenes'),
        'has_embeddings': resp.get('has_embeddings', False),
        'has_transcript': resp.get('transcript_entries', 0) > 0,
    }
    print(json.dumps(meta))
except Exception:
    pass
" >> "$LOG_FILE" 2>/dev/null
      fi
    fi
    ;;

  SubagentStart)
    LOG_FILE=$(current_run_file)
    echo "$INPUT" | jq -c '{type:"agent_start",timestamp:"'"$TIMESTAMP"'",agent_id:.agent_id,agent_type:.agent_type}' >> "$LOG_FILE" 2>/dev/null
    # Reset timing when subagent starts
    echo "$EPOCH_MS" > "$TIMING_FILE"
    ;;

  SubagentStop)
    LOG_FILE=$(current_run_file)
    echo "$INPUT" | jq -c '{type:"agent_stop",timestamp:"'"$TIMESTAMP"'",agent_id:.agent_id,agent_type:.agent_type}' >> "$LOG_FILE" 2>/dev/null
    ;;

  Stop)
    LOG_FILE=$(current_run_file)
    # Capture the final assistant message as a final_answer event
    LAST_MSG=$(echo "$INPUT" | jq -r '.last_assistant_message // ""' 2>/dev/null || echo "")
    if [ -n "$LAST_MSG" ] && [ "$LAST_MSG" != "null" ]; then
      echo "$INPUT" | jq -c '{type:"final_answer",timestamp:"'"$TIMESTAMP"'",text:.last_assistant_message}' >> "$LOG_FILE" 2>/dev/null
    fi
    # Clean up timing state
    rm -f "$TIMING_FILE"
    ;;
esac
exit 0

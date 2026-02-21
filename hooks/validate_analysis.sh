#!/usr/bin/env bash
# Non-blocking validation hook for video-analyst agent output.
# Checks for timestamps, evidence markers, and confidence indicators.
# Warnings only — always exits 0.
#
# NOTE: Stop hooks do NOT support matchers. We check the last_assistant_message
# content ourselves and skip validation for non-analysis responses.

INPUT=$(cat)

# Extract last_assistant_message from the hook JSON
MESSAGE=$(echo "$INPUT" | jq -r '.last_assistant_message // ""')

# Skip if no message or if message is too short to be an analysis
if [ ${#MESSAGE} -lt 200 ]; then
    exit 0
fi

# Skip if this doesn't look like a video analysis (no video/frame/scene keywords)
if ! echo "$MESSAGE" | grep -qiE '(video|frame|scene|segment|transcript|timestamp)'; then
    exit 0
fi

# Check for timestamp references (HH:MM:SS, MM:SS, or Ns/seconds patterns)
if ! echo "$MESSAGE" | grep -qE '([0-9]+:[0-9]{2}(:[0-9]{2})?|[0-9]+(\.[0-9]+)?s\b|seconds?)'; then
    echo "WARNING: Analysis output contains no timestamp references. Good analyses cite specific times." >&2
fi

# Check for evidence/findings markers
if ! echo "$MESSAGE" | grep -qiE '(evidence|finding|observation|confirmed|verified|visual)'; then
    echo "WARNING: Analysis output lacks evidence markers. Consider citing specific visual or transcript evidence." >&2
fi

# Check for confidence indicators
if ! echo "$MESSAGE" | grep -qiE '(confidence|confident|certain|uncertain|unclear|ambiguous|cannot confirm)'; then
    echo "WARNING: Analysis output lacks confidence indicators. State how confident you are in your findings." >&2
fi

# Always succeed — these are advisory warnings only
exit 0

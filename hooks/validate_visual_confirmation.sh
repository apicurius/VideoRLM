#!/usr/bin/env bash
# Stop hook: Validates that the final analysis cites frame evidence for numbers/names.
# If the output contains specific numbers but lacks frame/visual evidence markers,
# it warns the agent to verify before submitting.
#
# Exit code 0 = always allow (advisory warnings via stderr).
# Exit code 2 = block and ask agent to fix (use sparingly).

INPUT=$(cat)

MESSAGE=$(echo "$INPUT" | jq -r '.last_assistant_message // ""' 2>/dev/null || echo "")

# Skip short or non-analysis messages
if [ ${#MESSAGE} -lt 300 ]; then
    exit 0
fi

# Skip if not video analysis
if ! echo "$MESSAGE" | grep -qiE '(video|frame|scene|segment|transcript|timestamp)'; then
    exit 0
fi

# Count specific numbers in the output (likely claimed values)
NUM_COUNT=$(echo "$MESSAGE" | grep -oE '\b[0-9]+(\.[0-9]+)?(s|%|ms|fps|x|px|k|m|K|M)?\b' | wc -l)

# Count visual evidence markers
VISUAL_EVIDENCE=$(echo "$MESSAGE" | grep -ciE '(frame|screenshot|visually|visible|screen|displayed|shown|extracted|zoom|crop)')

# Check for unverified claims: many numbers but few visual references
if [ "$NUM_COUNT" -gt 5 ] && [ "$VISUAL_EVIDENCE" -lt 2 ]; then
    echo "WARNING: Output contains $NUM_COUNT numeric values but only $VISUAL_EVIDENCE visual evidence references. Ensure numbers are visually confirmed from frames, not just from transcript." >&2
fi

# Check for the word "transcript says" without corresponding "frame shows" or "visually confirmed"
TRANSCRIPT_CLAIMS=$(echo "$MESSAGE" | grep -ciE '(transcript (says|mentions|indicates|states|shows)|according to (the )?transcript|ASR|spoken)')
VISUAL_CONFIRMS=$(echo "$MESSAGE" | grep -ciE '(visually confirmed|frame shows|visible in|extracted frame|seen in|displayed on)')

if [ "$TRANSCRIPT_CLAIMS" -gt 2 ] && [ "$VISUAL_CONFIRMS" -lt 1 ]; then
    echo "WARNING: Analysis relies heavily on transcript ($TRANSCRIPT_CLAIMS references) without visual confirmation ($VISUAL_CONFIRMS references). Cross-reference with extracted frames." >&2
fi

exit 0

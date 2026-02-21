#!/usr/bin/env bash
# PostToolUse hook for kuavi_search_transcript results.
# When transcript search returns names or numbers, injects a reminder
# that these MUST be visually confirmed before reporting to the user.
#
# This makes anti-hallucination enforcement structural, not advisory.
# Exit code 0 = allow (with optional stderr feedback to the agent).

INPUT=$(cat)

TOOL_RESPONSE=$(echo "$INPUT" | jq -r '.tool_response // ""' 2>/dev/null || echo "")

# Skip if empty response or error
if [ -z "$TOOL_RESPONSE" ] || [ "$TOOL_RESPONSE" = "null" ]; then
    exit 0
fi

# Check if the transcript response contains numbers (potential misrecognitions)
HAS_NUMBERS=$(echo "$TOOL_RESPONSE" | grep -oE '\b[0-9]{2,}\b' | head -5)

# Check if it contains potential proper names (capitalized words that aren't common)
HAS_NAMES=$(echo "$TOOL_RESPONSE" | grep -oE '\b[A-Z][a-z]{2,}[A-Z][a-z]*\b|\b[A-Z][a-z]{3,}\b' | \
    grep -vE '^(The|This|That|When|Where|What|Which|There|Here|After|Before|About|These|Those|Their|Today|Could|Would|Should|Every|Never|Always|Also|Just|Still|Into|From|With|Over|Under|Between)$' | \
    head -5)

WARNINGS=""

if [ -n "$HAS_NUMBERS" ]; then
    NUMS=$(echo "$HAS_NUMBERS" | tr '\n' ', ' | sed 's/, $//')
    WARNINGS="ANTI-HALLUCINATION: Transcript contains numbers ($NUMS). ASR frequently misrecognizes numbers. You MUST visually confirm these from extracted frames before reporting them."
fi

if [ -n "$HAS_NAMES" ]; then
    NAMES=$(echo "$HAS_NAMES" | tr '\n' ', ' | sed 's/, $//')
    if [ -n "$WARNINGS" ]; then
        WARNINGS="$WARNINGS Also contains potential names ($NAMES) — ASR misrecognizes names. Extract title/name slides to visually confirm."
    else
        WARNINGS="ANTI-HALLUCINATION: Transcript contains potential names ($NAMES). ASR frequently misrecognizes proper names. Extract frames showing titles or name tags to visually confirm before reporting."
    fi
fi

if [ -n "$WARNINGS" ]; then
    echo "$WARNINGS" >&2
fi

exit 0

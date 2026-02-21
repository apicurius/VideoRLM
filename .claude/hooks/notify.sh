#!/bin/bash
# Notification hook: macOS desktop notification via osascript

title=$(jq -r '.title // "Claude Code"' 2>/dev/null)
message=$(jq -r '.message // "Notification"' 2>/dev/null)

osascript -e "display notification \"$message\" with title \"$title\"" 2>/dev/null || true

exit 0

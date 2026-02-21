#!/bin/bash
# PostToolUse hook: Run py_compile on edited Python files
# Reads tool result from stdin, extracts file_path, compiles if .py

file_path=$(jq -r '.tool_input.file_path // empty' 2>/dev/null)

if [[ -n "$file_path" && "$file_path" == *.py ]]; then
    uv run python -m py_compile "$file_path" 2>&1 || true
fi

exit 0

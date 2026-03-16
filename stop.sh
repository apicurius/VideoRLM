#!/usr/bin/env bash
# stop.sh — stop KUAVi backend (:7860) and frontend (:4000)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

kill_by_port() {
    local port="$1"
    local label="$2"
    local pids

    pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        echo "→ Stopping $label on :$port (PID: $pids)"
        kill $pids 2>/dev/null || true
        sleep 1

        pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
        if [ -n "$pids" ]; then
            echo "→ Force stopping $label on :$port (PID: $pids)"
            kill -9 $pids 2>/dev/null || true
        fi
    else
        echo "→ No $label process listening on :$port"
    fi
}

kill_by_port 7860 "backend"
kill_by_port 4000 "frontend"

# Fallback cleanup for common dev process names.
pkill -f "uvicorn web_app:app" 2>/dev/null || true
pkill -f "next dev -p 4000" 2>/dev/null || true

echo "✓ Stop command completed"
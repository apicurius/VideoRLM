#!/usr/bin/env bash
# stop.sh — stop KUAVi backend (:8000) and frontend (:4001)
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

kill_by_port 8000 "backend"
kill_by_port 4001 "frontend"

# Fallback cleanup for common dev process names.
pkill -f "uvicorn web_app:app" 2>/dev/null || true
pkill -f "next dev -p 4001" 2>/dev/null || true

# Remove stale Next.js locks
rm -f "$ROOT/frontend/.next/dev/lock" 2>/dev/null || true

echo "✓ Stop command completed"
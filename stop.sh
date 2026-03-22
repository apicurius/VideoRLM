#!/usr/bin/env bash
# stop.sh — stop KUAVi backend and frontend started by run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-4001}"
RUN_DIR="$ROOT/.run"
BACKEND_PID_FILE="$RUN_DIR/backend.pid"

kill_pid_file() {
    local pid_file="$1"
    local label="$2"

    if [ ! -f "$pid_file" ]; then
        return
    fi

    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "→ Stopping $label via pid file ($pid)"
        kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
}

kill_by_port() {
    local port="$1"
    local label="$2"
    local pids=""

    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    elif command -v ss >/dev/null 2>&1; then
        pids="$(ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p {print $NF}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)"
    fi

    if [ -n "$pids" ]; then
        echo "→ Stopping $label on :$port (PID: $pids)"
        kill $pids 2>/dev/null || true
        sleep 1

        if command -v lsof >/dev/null 2>&1; then
            pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
        elif command -v ss >/dev/null 2>&1; then
            pids="$(ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p {print $NF}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)"
        fi

        if [ -n "$pids" ]; then
            echo "→ Force stopping $label on :$port (PID: $pids)"
            kill -9 $pids 2>/dev/null || true
        fi
    else
        echo "→ No $label process listening on :$port"
    fi
}

kill_pid_file "$BACKEND_PID_FILE" "backend"

kill_by_port "$BACKEND_PORT" "backend"
kill_by_port "$FRONTEND_PORT" "frontend"

# Fallback cleanup for common dev process names.
pkill -f "uvicorn web_app:app" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true

# Remove stale Next.js locks
rm -f "$ROOT/frontend/.next/dev/lock" 2>/dev/null || true

echo "✓ Stop command completed"
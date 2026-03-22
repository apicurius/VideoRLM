#!/usr/bin/env bash
# run.sh — start BOTH the backend API and the Next.js frontend
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-4001}"
RUN_DIR="$ROOT/.run"
BACKEND_PID_FILE="$RUN_DIR/backend.pid"

mkdir -p "$RUN_DIR"

# ── Load .env ────────────────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "✗ Missing virtual environment at .venv. Run: uv sync"
    exit 1
fi

# ── GPU memory tuning + vendored LanguageBind path ──────────────────────────
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [ -d "$ROOT/vendor/LanguageBind" ]; then
    export PYTHONPATH="$ROOT/vendor/LanguageBind${PYTHONPATH:+:$PYTHONPATH}"
fi

# ── Ensure frontend deps are installed ───────────────────────────────────────
if [ ! -d "$ROOT/frontend/node_modules" ]; then
    echo "→ Installing frontend dependencies..."
    (cd "$ROOT/frontend" && npm install --silent)
fi

# ── Stop stale listeners on target ports ─────────────────────────────────────
if [ -x "$ROOT/stop.sh" ]; then
    BACKEND_PORT="$BACKEND_PORT" FRONTEND_PORT="$FRONTEND_PORT" "$ROOT/stop.sh" >/dev/null 2>&1 || true
fi

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║        VideoRLM  ·  KUAVi  Demo          ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  API server : http://localhost:$BACKEND_PORT"
echo "  Frontend   : http://localhost:$FRONTEND_PORT"
echo ""

# ── Start backend (background) ───────────────────────────────────────────────
echo "→ Starting API server on :$BACKEND_PORT ..."
uv run python -m uvicorn web_app:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    --reload \
    --log-level info &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

cleanup() {
    if [ -f "$BACKEND_PID_FILE" ]; then
        PID="$(cat "$BACKEND_PID_FILE" 2>/dev/null || true)"
        if [ -n "${PID:-}" ]; then
            kill "$PID" 2>/dev/null || true
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
}

trap cleanup INT TERM EXIT

# ── Start frontend (foreground, exits on Ctrl-C) ─────────────────────────────
echo "→ Starting frontend on :$FRONTEND_PORT ..."
(
    cd "$ROOT/frontend"
    BACKEND_URL="http://localhost:$BACKEND_PORT" \
    NEXT_PUBLIC_BACKEND_URL="http://localhost:$BACKEND_PORT" \
    npm run dev -- --port "$FRONTEND_PORT"
)

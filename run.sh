#!/usr/bin/env bash
# run.sh — start BOTH the backend API and the Next.js frontend
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Load .env ────────────────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠  WARNING: OPENROUTER_API_KEY is not set."
    echo "   Add it to .env or export it before running."
fi

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║        VideoRLM  ·  KUAVi  Demo          ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  API server : http://localhost:7860"
echo "  Frontend   : http://localhost:4000"
echo "  Backend    : OpenRouter  (${OPENROUTER_API_KEY:0:20}...)"
echo ""

# ── GPU memory tuning ────────────────────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── Ensure Python deps are installed ─────────────────────────────────────────
VENV_PYTHON="$ROOT/.venv/bin/python"
uv pip install openai markdown --python "$VENV_PYTHON" --quiet

# ── Ensure frontend deps are installed ───────────────────────────────────────
if [ ! -d "$ROOT/frontend/node_modules" ]; then
    echo "→ Installing frontend dependencies..."
    (cd "$ROOT/frontend" && npm install --silent)
fi

# ── Start backend (background) ───────────────────────────────────────────────
echo "→ Starting API server on :7860 ..."
"$ROOT/.venv/bin/python" -m uvicorn web_app:app \
    --host 0.0.0.0 \
    --port 7860 \
    --reload \
    --log-level info &
BACKEND_PID=$!

# ── Start frontend (foreground, exits on Ctrl-C) ─────────────────────────────
echo "→ Starting frontend on :4000 ..."
trap "kill $BACKEND_PID 2>/dev/null; exit" INT TERM
(cd "$ROOT/frontend" && npm run dev)

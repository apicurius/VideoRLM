#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# ── Load .env so keys are available in this shell session too ──────────────
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

# ── Pre-flight checks ──────────────────────────────────────────────────────
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠  WARNING: OPENROUTER_API_KEY is not set."
    echo "   Add it to .env or export it before running."
fi

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║        VideoRLM  ·  KUAVi  Demo          ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  Backend   : OpenRouter  (${OPENROUTER_API_KEY:0:20}...)"
echo "  Server    : http://localhost:7860"
echo "  Hot-reload: enabled"
echo ""

# ── GPU memory tuning ─────────────────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── Ensure required packages are installed in the venv ────────────────────
VENV_PYTHON="$(dirname "$0")/.venv/bin/python"
uv pip install openai markdown --python "$VENV_PYTHON" --quiet

# ── Start server ───────────────────────────────────────────────────────────
exec .venv/bin/python -m uvicorn web_app:app \
    --host 0.0.0.0 \
    --port 7860 \
    --reload \
    --log-level info

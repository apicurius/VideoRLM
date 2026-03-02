#!/usr/bin/env bash
# run_backend.sh — start only the FastAPI / uvicorn backend
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠  WARNING: OPENROUTER_API_KEY is not set."
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True

VENV_PYTHON="$ROOT/.venv/bin/python"
uv pip install openai markdown --python "$VENV_PYTHON" --quiet

echo "→ API server starting on http://localhost:7860"
exec "$ROOT/.venv/bin/python" -m uvicorn web_app:app \
    --host 0.0.0.0 \
    --port 7860 \
    --reload \
    --log-level info

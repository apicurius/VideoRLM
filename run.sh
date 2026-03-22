#!/usr/bin/env bash
# run.sh — start BOTH the backend API and the Next.js frontend
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Ensure Virtual Environment exists ──────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "→ Creating virtual environment..."
    uv venv --python 3.12 .venv
fi
VENV_PYTHON="$ROOT/.venv/bin/python"

# ── Load .env ────────────────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

# ── Vendor LanguageBind if missing ───────────────────────────────────────────
mkdir -p vendor
if [ ! -d "vendor/LanguageBind" ]; then
    echo "→ Local multimodal backend missing. Vendoring LanguageBind..."
    git clone https://github.com/PKU-YuanGroup/LanguageBind.git vendor/LanguageBind
    
    # RELAX version pins: old pins like numpy==1.23.0 fail on Python 3.12+ 
    # as they lack wheels and require building from source (missing Python.h).
    # Using >= allows uv to find modern, compatible wheels.
    sed -i 's/==/>=/g' vendor/LanguageBind/requirements.txt
fi

# ── Pre-flight checks ────────────────────────────────────────────────────────
echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║        VideoRLM  ·  KUAVi  Demo          ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  API server : http://localhost:8000"
echo "  Frontend   : http://localhost:4001"
echo "  Backend    : ${EMBEDDING_BACKEND:-languagebind} (Local multimodal)"
echo ""

# ── GPU memory tuning ────────────────────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Ensure LanguageBind is discoverable
export PYTHONPATH="$ROOT/vendor/LanguageBind:$PYTHONPATH"

# ── Ensure Python deps are installed ─────────────────────────────────────────
# Use a sentinel to check if the full install happened
if ! "$VENV_PYTHON" -c "import languagebind, fastapi" >/dev/null 2>&1; then
    echo "→ Installing backend dependencies..."
    # Always install main project deps
    uv pip install \
        fastapi uvicorn python-multipart numpy opencv-python scikit-learn \
        torch torchvision torchaudio torchcodec transformers pillow openai markdown \
        --python "$VENV_PYTHON" --quiet
        
    echo "→ Installing LanguageBind sub-dependencies (relaxing strict pins)..."
    # Relax pins to avoid build failures with newer versions of torch/transformers
    sed -i 's/==/>=/g' vendor/LanguageBind/requirements.txt
    # Install from the vendored requirements, ensure we relaxed the pins
    uv pip install -r vendor/LanguageBind/requirements.txt --python .venv/bin/python --quiet

    # Patch pytorchvideo for compatibility with newer torchvision
    find .venv/lib* -name "augmentations.py" -path "*/pytorchvideo/transforms/*" -exec sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms.functional/g' {} +
fi

# ── Ensure frontend deps are installed ───────────────────────────────────────
if [ ! -d "$ROOT/frontend/node_modules" ]; then
    echo "→ Installing frontend dependencies..."
    (cd "$ROOT/frontend" && npm install --silent)
fi

# ── Start backend (background) ───────────────────────────────────────────────
echo "→ Starting API server on :8000 ..."
"$VENV_PYTHON" -m uvicorn web_app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info &
BACKEND_PID=$!

# ── Start frontend (foreground, exits on Ctrl-C) ─────────────────────────────
echo "→ Starting frontend on :4001 ..."
trap "kill $BACKEND_PID 2>/dev/null; exit" INT TERM
(cd "$ROOT/frontend" && npm run dev)

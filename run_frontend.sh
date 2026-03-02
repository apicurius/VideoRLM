#!/usr/bin/env bash
# run_frontend.sh — start only the Next.js frontend
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$ROOT/frontend/node_modules" ]; then
    echo "→ Installing frontend dependencies..."
    (cd "$ROOT/frontend" && npm install)
fi

echo "→ Frontend starting on http://localhost:4000"
exec sh -c "cd '$ROOT/frontend' && npm run dev"

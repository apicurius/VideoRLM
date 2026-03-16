#!/usr/bin/env bash
# validate_cli.sh — lightweight KUAVi CLI compatibility smoke checks
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

uv run python -m kuavi.cli --help >/dev/null
uv run python -m kuavi.cli query --help >/dev/null
uv run python -m kuavi.cli agent --help >/dev/null
uv run python -m kuavi.cli index --help | grep -E "force-reindex|stages" >/dev/null

echo "CLI validation: OK"

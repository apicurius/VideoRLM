#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

.venv/bin/python -m uvicorn web_app:app --host 0.0.0.0 --port 8000 --reload

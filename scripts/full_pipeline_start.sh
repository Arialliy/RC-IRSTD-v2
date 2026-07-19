#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${1:-$ROOT/configs/pipeline.yaml}"
cd "$ROOT"
"$PYTHON_BIN" run_pipeline.py --config "$CONFIG"

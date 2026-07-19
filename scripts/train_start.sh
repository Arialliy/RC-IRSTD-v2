#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${1:-$ROOT/configs/detector.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
cd "$ROOT"
"$PYTHON_BIN" train_detector.py --config "$CONFIG" "$@"

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" -m risk_curve.select_zero_label_threshold "$@"

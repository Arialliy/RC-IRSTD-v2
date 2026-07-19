#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="${1:-datasets/NUDT-SIRST}"
SAVE_DIR="${2:-repro_runs/smoke}"

"${PYTHON_BIN}" train.py \
  --dataset-dir "${DATASET_DIR}" \
  --batch-size 2 \
  --epochs 1 \
  --num-workers 0 \
  --save-dir "${SAVE_DIR}"

LATEST_WEIGHT="$(find "${SAVE_DIR}" -type f -name weight.pkl -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "${LATEST_WEIGHT}" ]]; then
  echo "No smoke-test weight was produced under ${SAVE_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" test.py \
  --dataset-dir "${DATASET_DIR}" \
  --weight-path "${LATEST_WEIGHT}" \
  --num-workers 0

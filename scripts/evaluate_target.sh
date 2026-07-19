#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  cat <<'USAGE'
Usage:
  scripts/evaluate_target.sh \
    DETECTOR_PT CALIBRATOR_PT DATASET_DIR OUTPUT_DIR SUPPORT_SIZE \
    "BUDGETS..." [SPLIT] [DEVICE]
USAGE
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DETECTOR="$1"
CALIBRATOR="$2"
DATASET_DIR="$3"
OUTPUT_DIR="$4"
SUPPORT_SIZE="$5"
read -r -a BUDGETS <<< "$6"
SPLIT="${7:-test}"
DEVICE="${8:-auto}"
SCORES_DIR="$OUTPUT_DIR/scores"
METRICS_DIR="$OUTPUT_DIR/causal_metrics"

cd "$ROOT"
"$PYTHON_BIN" export_scores.py \
  --checkpoint "$DETECTOR" \
  --dataset-dir "$DATASET_DIR" \
  --split "$SPLIT" \
  --output-dir "$SCORES_DIR" \
  --device "$DEVICE"

"$PYTHON_BIN" evaluate_causal.py \
  --score-dir "$SCORES_DIR" \
  --calibrator "$CALIBRATOR" \
  --output-dir "$METRICS_DIR" \
  --support-size "$SUPPORT_SIZE" \
  --budgets "${BUDGETS[@]}" \
  --all-windows \
  --device "$DEVICE"

printf '\nEvaluation complete:\n  %s\n' "$METRICS_DIR/causal_metrics.json"

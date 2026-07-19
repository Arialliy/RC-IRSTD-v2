#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  cat <<'USAGE'
Usage:
  scripts/deploy_target.sh \
    DETECTOR_PT CALIBRATOR_PT DATASET_DIR OUTPUT_DIR BUDGET SUPPORT_SIZE \
    [SPLIT] [DEVICE]

The target dataset may omit masks. It must provide images/ and a frozen split
manifest such as img_idx/test.txt. Every complete trained support/query window
uses only its unlabeled support prefix; predictions are written for its future
query segment. Incomplete tails are reported but not silently predicted.
USAGE
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DETECTOR="$1"
CALIBRATOR="$2"
DATASET_DIR="$3"
OUTPUT_DIR="$4"
BUDGET="$5"
SUPPORT_SIZE="$6"
SPLIT="${7:-test}"
DEVICE="${8:-auto}"
SCORES_DIR="$OUTPUT_DIR/scores"
PREDICTIONS_DIR="$OUTPUT_DIR/predictions"

cd "$ROOT"
"$PYTHON_BIN" export_scores.py \
  --checkpoint "$DETECTOR" \
  --dataset-dir "$DATASET_DIR" \
  --split "$SPLIT" \
  --output-dir "$SCORES_DIR" \
  --device "$DEVICE" \
  --allow-missing-masks

"$PYTHON_BIN" infer_online.py \
  --score-dir "$SCORES_DIR" \
  --calibrator "$CALIBRATOR" \
  --output-dir "$PREDICTIONS_DIR" \
  --budget "$BUDGET" \
  --support-size "$SUPPORT_SIZE" \
  --all-windows \
  --device "$DEVICE"

printf '\nDeployment complete:\n  %s\n' "$PREDICTIONS_DIR/inference.json"

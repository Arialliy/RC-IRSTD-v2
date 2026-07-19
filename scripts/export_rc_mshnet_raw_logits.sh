#!/usr/bin/env bash
# Integrity-checked FP32 raw-logit export for an RC-MSHNet trainer checkpoint.
set -Eeuo pipefail
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 CHECKPOINT DATASET_DIR OUTPUT_DIR DATASET_NAME [train|test]" >&2
  exit 2
fi
CHECKPOINT="$1"; DATASET_DIR="$2"; OUTPUT_DIR="$3"; DATASET_NAME="$4"; SPLIT="${5:-train}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LABELS_LOADED="${LABELS_LOADED:-true}"
SOURCE_DATASETS="${SOURCE_DATASETS:?Set comma-separated detector training domains}"
SPLIT_FILE="${SPLIT_FILE:-}"
IFS=',' read -r -a SOURCES <<< "$SOURCE_DATASETS"
args=(
  "$PYTHON_BIN" -m rc_irstd.cli.export_scores
  --checkpoint "$CHECKPOINT"
  --dataset-dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --dataset-name "$DATASET_NAME"
  --split "$SPLIT"
  --device "$DEVICE"
  --num-workers "$NUM_WORKERS"
  --batch-size 1
  --export-raw-logits
  --overwrite
)
for source in "${SOURCES[@]}"; do args+=(--source-dataset "$source"); done
if [[ -n "$SPLIT_FILE" ]]; then args+=(--split-file "$SPLIT_FILE"); fi
case "$LABELS_LOADED" in
  true|1|yes) args+=(--labels-loaded) ;;
  false|0|no) args+=(--no-labels-loaded) ;;
  *) echo "LABELS_LOADED must be true or false" >&2; exit 2 ;;
esac
"${args[@]}"

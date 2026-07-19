#!/usr/bin/env bash
# Build the formal source-only dense raw-logit grid.
set -Eeuo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_SCORE_DIRS="${SOURCE_SCORE_DIRS:?Comma-separated source self-score directories are required}"
EXPECTED_SOURCE_DOMAINS="${EXPECTED_SOURCE_DOMAINS:?Comma-separated outer source domains are required}"
OUTER_TARGET="${OUTER_TARGET:?Set the held-out outer target name}"
OUTPUT_DIR="${OUTPUT_DIR:?Set the grid output directory}"
MAX_GRID_POINTS="${MAX_GRID_POINTS:-1024}"
FORCE="${FORCE:-false}"
IFS=',' read -r -a DIRS <<< "$SOURCE_SCORE_DIRS"
IFS=',' read -r -a DOMAINS <<< "$EXPECTED_SOURCE_DOMAINS"
args=(
  "$PYTHON_BIN" -m risk_curve.build_logit_threshold_grid
  --outer-target "$OUTER_TARGET"
  --output-dir "$OUTPUT_DIR"
  --max-grid-points "$MAX_GRID_POINTS"
)
for directory in "${DIRS[@]}"; do args+=(--source-score-dir "$directory"); done
for domain in "${DOMAINS[@]}"; do args+=(--expected-source-domain "$domain"); done
if [[ "$FORCE" == "true" ]]; then args+=(--force); fi
"${args[@]}"

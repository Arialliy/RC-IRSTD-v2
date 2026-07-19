#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$#" -ne 8 ]]; then
  echo "Usage: $0 ZERO_JSON CAL_SCORE_DIR TEST_SCORE_DIR GRID_NPY PIXEL_BUDGET COMPONENT_BUDGET ALPHA OUTPUT_DIR" >&2
  exit 2
fi

ZERO_JSON="$1"
CAL_SCORE_DIR="$2"
TEST_SCORE_DIR="$3"
GRID_NPY="$4"
PIXEL_BUDGET="$5"
COMPONENT_BUDGET="$6"
ALPHA="$7"
OUTPUT_DIR="$8"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m certification.build_calibration_losses \
  --score-dir "${CAL_SCORE_DIR}" \
  --threshold-grid "${GRID_NPY}" \
  --pixel-budget "${PIXEL_BUDGET}" \
  --component-budget "${COMPONENT_BUDGET}" \
  --loss-mode budget_violation \
  --output "${OUTPUT_DIR}/calibration_curves.npz"

"${PYTHON_BIN}" -m certification.build_calibration_losses \
  --score-dir "${TEST_SCORE_DIR}" \
  --threshold-grid "${GRID_NPY}" \
  --pixel-budget "${PIXEL_BUDGET}" \
  --component-budget "${COMPONENT_BUDGET}" \
  --loss-mode budget_violation \
  --output "${OUTPUT_DIR}/test_curves.npz"

"${PYTHON_BIN}" -m certification.calibrate_target_offset \
  --calibration-curves "${OUTPUT_DIR}/calibration_curves.npz" \
  --test-curves "${OUTPUT_DIR}/test_curves.npz" \
  --zero-result "${ZERO_JSON}" \
  --alpha "${ALPHA}" \
  --loss-mode budget_violation \
  --output "${OUTPUT_DIR}/selection.json"

"${PYTHON_BIN}" -m certification.evaluate_certified_mode \
  --selection-result "${OUTPUT_DIR}/selection.json" \
  --test-curves "${OUTPUT_DIR}/test_curves.npz" \
  --output "${OUTPUT_DIR}/test_audit.json"

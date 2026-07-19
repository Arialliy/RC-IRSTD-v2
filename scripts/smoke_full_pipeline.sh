#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$ROOT"
rm -rf artifacts/smoke_data artifacts/smoke_pipeline

"$PYTHON_BIN" make_synthetic.py \
  --output-root artifacts/smoke_data \
  --num-train 2 \
  --num-test 3 \
  --image-size 32

"$PYTHON_BIN" run_pipeline.py --config configs/smoke_pipeline.yaml

RESULT="$ROOT/artifacts/smoke_pipeline/targets/domain_d/causal_evaluation/causal_metrics.json"
test -f "$RESULT"
printf '\nSmoke pipeline passed. Result: %s\n' "$RESULT"

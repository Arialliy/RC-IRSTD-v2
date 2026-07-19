#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$ROOT"

configs=(
  configs/pipeline_outer_nuaa.yaml
  configs/pipeline_outer_nudt.yaml
  configs/pipeline_outer_irstd.yaml
)
pids=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for config in "${configs[@]}"; do
  "$PYTHON_BIN" run_pipeline.py --config "$config" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
trap - INT TERM
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

printf '\nAll three outer-LODO folds completed under outputs/outer_lodo/.\n'

#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$PYTHON_BIN" scripts/smoke_rc_mshnet.py --size 32 --batch-size 2 --device cpu

# Dataset/trainer smoke. Generate the repository's standard synthetic data first
# when artifacts/smoke_data does not already exist.
if [[ "${RUN_DATA_SMOKE:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m rc_irstd.cli.train_detector \
    --config configs/smoke_rc_mshnet.yaml
fi

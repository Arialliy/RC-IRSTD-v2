#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TRAIN_EPISODES="${TRAIN_EPISODES:-${ROOT}/outputs/curve_episodes/train.npz}"
VAL_EPISODES="${VAL_EPISODES:-${ROOT}/outputs/curve_episodes/val.npz}"
OUT="${OUT:-${ROOT}/outputs/aaai27/risk_curve_main/smoke.pt}"

QUANTILE="${QUANTILE:-0.90}"
LAMBDA_COMPONENT="${LAMBDA_COMPONENT:-1.0}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.10}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
PATIENCE="${PATIENCE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

if [[ "${OUT}" != *.pt ]]; then
  printf 'OUT must name a .pt checkpoint, got: %s\n' "${OUT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT}")"
cd "${ROOT}"

exec "${PYTHON_BIN}" -m risk_curve.train_curve_predictor \
  --train-file "${TRAIN_EPISODES}" \
  --val-file "${VAL_EPISODES}" \
  --output "${OUT}" \
  --quantile "${QUANTILE}" \
  --lambda-component "${LAMBDA_COMPONENT}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --dropout "${DROPOUT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --patience "${PATIENCE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

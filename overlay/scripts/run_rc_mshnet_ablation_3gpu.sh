#!/usr/bin/env bash
# Run three matched RC-MSHNet ablations for ONE outer fold.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:?Set CONFIG to detector_rc_mshnet_outer_<fold>_fast.yaml}"
MSHNET_INIT="${MSHNET_INIT:?Set MSHNET_INIT to the matching tensor-only MSHNet checkpoint}"
RUN_ROOT="${RUN_ROOT:-outputs/aaai27_rc_mshnet_gate}"
EPOCHS="${EPOCHS:-80}"
SEED="${SEED:-42}"
ROUND="${ROUND:-core}"
CUDA_IDS="${CUDA_IDS:-0,1,2}"
IFS=',' read -r -a GPUS <<< "$CUDA_IDS"
if [[ "${#GPUS[@]}" -ne 3 ]]; then
  echo "CUDA_IDS must contain exactly three comma-separated GPU ids" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" || ! -f "$MSHNET_INIT" ]]; then
  echo "CONFIG or MSHNET_INIT does not exist" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT/logs"

case "$ROUND" in
  core)
    NAMES=(full no_contrast no_component)
    OVERRIDES=(
      "model.use_contrast=true model.use_component_context=true model.use_risk_gate=true"
      "model.use_contrast=false model.use_component_context=true model.use_risk_gate=true"
      "model.use_contrast=true model.use_component_context=false model.use_risk_gate=true"
    )
    ;;
  fusion)
    NAMES=(no_gate no_branch_aux contrast_only)
    OVERRIDES=(
      "model.use_contrast=true model.use_component_context=true model.use_risk_gate=false"
      "model.use_contrast=true model.use_component_context=true model.use_risk_gate=true model.expose_branch_auxiliary=false"
      "model.use_contrast=true model.use_component_context=false model.use_risk_gate=false"
    )
    ;;
  *)
    echo "ROUND must be core or fusion" >&2
    exit 2
    ;;
esac

pids=()
cleanup() {
  local status=$?
  if [[ $status -ne 0 ]]; then
    for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

for index in 0 1 2; do
  name="${NAMES[$index]}"
  gpu="${GPUS[$index]}"
  output="$RUN_ROOT/${ROUND}_${name}_seed${SEED}"
  log="$RUN_ROOT/logs/${ROUND}_${name}_seed${SEED}.log"
  read -r -a variant <<< "${OVERRIDES[$index]}"
  command=(
    "$PYTHON_BIN" -m rc_irstd.cli.train_detector
    --config "$CONFIG"
    --set "device=cuda:0"
    --set "seed=$SEED"
    --set "output_dir=$output"
    --set "training.epochs=$EPOCHS"
    --set "training.initialize_from=$MSHNET_INIT"
  )
  for item in "${variant[@]}"; do command+=(--set "$item"); done
  echo "[$name] CUDA_VISIBLE_DEVICES=$gpu -> $output"
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in 0 1 2; do
  if ! wait "${pids[$index]}"; then
    echo "${NAMES[$index]} failed; inspect $RUN_ROOT/logs" >&2
    failed=1
  fi
done
trap - EXIT INT TERM
if [[ $failed -ne 0 ]]; then exit 1; fi
printf 'All ablations completed under %s\n' "$RUN_ROOT"

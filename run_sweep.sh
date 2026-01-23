#!/usr/bin/env bash
set -euo pipefail

############################################
# User config
############################################
BASE_EPOCHS=5
LARGE_EPOCHS=20

LR=0.4
MOM=0.9
CLIP=1.0

OUT_DIR="runs_sweeps"

# FTRL-specific knobs
RESTART=1
TREE_COMPLETION=True
EFFI_NOISE=False

# Epsilons: 10 values from 0.1 → 32, uniform in 1/ε
EPSILONS=($(python - <<'PY'
import numpy as np
inv = np.linspace(1/0.1, 1/32, 10)
eps = 1.0 / inv
print(" ".join(f"{e:.6g}" for e in eps))
PY
))

############################################
# Helpers
############################################
mkdir -p "${OUT_DIR}"
stamp() { date -u +"%Y%m%dT%H%M%SZ"; }

run_one () {
  local dataset="$1"
  local batch="$2"
  local epochs="$3"
  local algo="$4"
  local eps="$5"
  local dpdl="$6"        # true / false
  local tag="$7"         # sweep | single

  local ts
  ts="$(stamp)"

  local log="${OUT_DIR}/${dataset}_bs${batch}_ep${epochs}_${algo}_${tag}_dpdl${dpdl}_eps${eps}_${ts}.log"

  echo "==== RUN data=${dataset} batch=${batch} epochs=${epochs} algo=${algo} eps=${eps} dp_dataloader=${dpdl}"
  echo "==== LOG ${log}"

  export ML_DATA=/userhome/cs3/zmsxsl/data

  python -u main.py \
    --data="${dataset}" \
    --algo="${algo}" \
    --epochs="${epochs}" \
    --batch_size="${batch}" \
    --learning_rate="${LR}" \
    --momentum="${MOM}" \
    --l2_norm_clip="${CLIP}" \
    --noise_multiplier="${eps}" \
    --restart="${RESTART}" \
    --tree_completion="${TREE_COMPLETION}" \
    --effi_noise="${EFFI_NOISE}" \
    --dp_dataloader="${dpdl}" \
    --dir="${OUT_DIR}" \
    2>&1 | tee "${log}"
}

run_dataset () {
  local dataset="$1"
  local base_batch="$2"

  for batch in "${base_batch}" "$((4 * base_batch))"; do

    if [[ "${batch}" -eq "$((4 * base_batch))" ]]; then
      epochs="${LARGE_EPOCHS}"
    else
      epochs="${BASE_EPOCHS}"
    fi

    echo "=============================="
    echo "Dataset=${dataset}, batch=${batch}, epochs=${epochs}"
    echo "=============================="

    for dpdl in false true; do

      ########################################
      # 0) Non-private FTRL (once)
      ########################################
      run_one "${dataset}" "${batch}" "${epochs}" "ftrl_nodp" 0.0 "${dpdl}" "single"

      ########################################
      # 1) DP-FTRL sweep
      ########################################
      for eps in "${EPSILONS[@]}"; do
        run_one "${dataset}" "${batch}" "${epochs}" "ftrl_dp" "${eps}" "${dpdl}" "sweep"
      done

      ########################################
      # 2) DP-SGD (with amplification)
      ########################################
      for eps in "${EPSILONS[@]}"; do
        run_one "${dataset}" "${batch}" "${epochs}" "sgd_amp" "${eps}" "${dpdl}" "sweep"
      done

      ########################################
      # 3) DP-SGD (no amplification reporting)
      ########################################
      for eps in "${EPSILONS[@]}"; do
        run_one "${dataset}" "${batch}" "${epochs}" "sgd_noamp" "${eps}" "${dpdl}" "sweep"
      done

    done
  done
}

############################################
# Run all datasets
############################################

run_dataset "mnist"        250
run_dataset "cifar10"      500
run_dataset "emnist_merge" 500

echo "======================================"
echo "All sweeps finished."
echo "Results:"
echo "  - ${OUT_DIR}/results.jsonl"
echo "  - ${OUT_DIR}/<dataset>/results.jsonl"
echo "Logs:"
echo "  - ${OUT_DIR}/*.log"
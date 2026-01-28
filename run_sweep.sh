#!/usr/bin/env bash
set -euo pipefail

# ====== User config ======
EPOCHS=5
LR=4.0
MOM=0
CLIP=1.0

# Where logs/results are written (must match your code's --dir flag if it exists)
OUT_DIR="runs_sweeps"

# Simulate epsilons
# EPSILONS=(0 0.1 0.2 0.3 0.4 0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 12.0 15.0)
EPSILONS=(0.6 0.7 0.8 0.9 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5)
DP_DATALOADER_VALUES=(True False)

# FTRL-specific knobs (paper-style)
RESTART=1
TREE_COMPLETION=True
EFFI_NOISE=False

mkdir -p "${OUT_DIR}"

stamp() { date -u +"%Y%m%dT%H%M%SZ"; }

run_one () {
  local algo="$1"
  local eps="$2"
  local dp_dl="$3"  # New argument for dp_dataloader
  local tag="$4"
  
  local ts
  ts="$(stamp)"
  # Added dp_dl to the log filename to distinguish runs
  local log="${OUT_DIR}/${DATA}_${algo}_${tag}_eps${eps}_dpdl${dp_dl}_${ts}.log"

  echo "==== RUN algo=${algo} eps=${eps} dp_dataloader=${dp_dl} tag=${tag} -> ${log}"

  export ML_DATA=/userhome/cs3/zmsxsl/data
  
  python -u main.py \
    --algo="${algo}" \
    --data="${DATA}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH}" \
    --learning_rate="${LR}" \
    --momentum="${MOM}" \
    --l2_norm_clip="${CLIP}" \
    --noise_multiplier="${eps}" \
    --dp_dataloader="${dp_dl}" \
    --restart="${RESTART}" \
    --tree_completion="${TREE_COMPLETION}" \
    --effi_noise="${EFFI_NOISE}" \
    --dir="${OUT_DIR}" \
    2>&1 | tee "${log}"
}

run_dataset () {
  local dataset="$1"
  local batch_size="$2"
  DATA="${dataset}"
  BATCH="${batch_size}"

  # We nest the loops: For every Algo -> For every DP_Dataloader setting -> For every Epsilon
  
  # ====== 1) DP-FTRL sweep ======
  for dp_val in "${DP_DATALOADER_VALUES[@]}"; do
    for e in "${EPSILONS[@]}"; do
      run_one "ftrl_dp" "${e}" "${dp_val}" "sweep"
    done
  done

  # ====== 2) DP-FTRL Matrix sweep ======
  for dp_val in "${DP_DATALOADER_VALUES[@]}"; do
    for e in "${EPSILONS[@]}"; do
      run_one "ftrl_dp_matrix" "${e}" "${dp_val}" "sweep"
    done
  done

  # ====== 3) DP-SGD no-amplification reporting sweep ======
  for dp_val in "${DP_DATALOADER_VALUES[@]}"; do
    for e in "${EPSILONS[@]}"; do
      run_one "sgd_noamp" "${e}" "${dp_val}" "sweep"
    done
  done
}

# ====== MNIST ======
run_dataset "mnist" 250

# ====== CIFAR-10 ======
run_dataset "cifar10" 500

# ====== EMNIST ======
run_dataset "emnist_merge" 500

echo "All sweeps finished. Logs in: ${OUT_DIR}"
echo "Results JSONL should be under: ${OUT_DIR}/results.jsonl"
#!/usr/bin/env bash
set -euo pipefail

# ====== User config ======
EPOCHS=5
LR=0.4
MOM=0.9
CLIP=1.0

# Where logs/results are written (must match your code's --dir flag if it exists)
OUT_DIR="runs_sweeps"

# Simulate 10 epsilons from 0.1 to 32 with uniform increments of 1/epsilon.
EPSILONS=($(python - <<'PY'
import numpy as np
inv_lo = 1.0 / 0.1
inv_hi = 1.0 / 32.0
inv = np.linspace(inv_lo, inv_hi, 10)
eps = 1.0 / inv
print(" ".join(f"{e:.6g}" for e in eps))
PY
))

# FTRL-specific knobs (paper-style)
RESTART=1
TREE_COMPLETION=True
EFFI_NOISE=False

mkdir -p "${OUT_DIR}"

stamp() { date -u +"%Y%m%dT%H%M%SZ"; }

run_one () {
  local algo="$1"
  local eps="$2"
  local tag="$3"
  local ts
  ts="$(stamp)"
  local log="${OUT_DIR}/${DATA}_${algo}_${tag}_eps${eps}_${ts}.log"

  echo "==== RUN algo=${algo} eps=${eps} tag=${tag} -> ${log}"

  export ML_DATA=/userhome/cs3/zmsxsl/data
  # python -u forces unbuffered output so you see it immediately
  # 2>&1 | tee "${log}" sends output to both screen and file
  python -u main.py \
    --algo="${algo}" \
    --data="${DATA}" \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH}" \
    --learning_rate="${LR}" \
    --momentum="${MOM}" \
    --l2_norm_clip="${CLIP}" \
    --noise_multiplier="${eps}" \
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

  # ====== 0) Non-private FTRL (run once; eps irrelevant) ======
  run_one "ftrl_nodp" 0.0 "single"

  # ====== 1) DP-FTRL sweep ======
  for e in "${EPSILONS[@]}"; do
    run_one "ftrl_dp" "${e}" "sweep"
  done

  # ====== 2) DP-SGD with amplification sweep ======
  for e in "${EPSILONS[@]}"; do
    run_one "sgd_amp" "${e}" "sweep"
  done

  # ====== 3) DP-SGD no-amplification reporting sweep ======
  for e in "${EPSILONS[@]}"; do
    run_one "sgd_noamp" "${e}" "sweep"
  done
}

# ====== MNIST ======
run_dataset "mnist" 250

# ====== CIFAR-10 ======
run_dataset "cifar10" 500

# ====== EMNIST ======
run_dataset "emnist_merge" 500

echo "All sweeps finished. Logs in: ${OUT_DIR}"
echo "Results JSONL should be under: ${OUT_DIR}/results.jsonl and ${OUT_DIR}/${DATA}/results.jsonl"

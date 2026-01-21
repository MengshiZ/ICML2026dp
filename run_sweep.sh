#!/usr/bin/env bash
set -euo pipefail

# ====== User config ======
DATA="emnist_merge"          # mnist | cifar10 | emnist_merge
EPOCHS=5
BATCH=250
LR=0.4
MOM=0.9
CLIP=1.0


# Where logs/results are written (must match your code's --dir flag if it exists)
OUT_DIR="runs_sweeps"

# Noise multipliers to sweep
SIGMAS=(0.5 0.8 1.0 1.2 1.5 2.0 3.0 4.0)

# FTRL-specific knobs (paper-style)
RESTART=1
TREE_COMPLETION=True
EFFI_NOISE=False

mkdir -p "${OUT_DIR}"

stamp() { date -u +"%Y%m%dT%H%M%SZ"; }

run_one () {
  local algo="$1"
  local sigma="$2"
  local tag="$3"
  local ts
  ts="$(stamp)"
  local log="${OUT_DIR}/${DATA}_${algo}_${tag}_sigma${sigma}_${ts}.log"

  echo "==== RUN algo=${algo} sigma=${sigma} tag=${tag} -> ${log}"

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
    --noise_multiplier="${sigma}" \
    --restart="${RESTART}" \
    --tree_completion="${TREE_COMPLETION}" \
    --effi_noise="${EFFI_NOISE}" \
    --dir="${OUT_DIR}" \
    2>&1 | tee "${log}"
}

# ====== 0) Non-private FTRL (run once; sigma irrelevant) ======
run_one "ftrl_nodp" 0.0 "single"

# ====== 1) DP-FTRL sweep ======
for s in "${SIGMAS[@]}"; do
  run_one "ftrl_dp" "${s}" "sweep"
done

# ====== 2) DP-SGD with amplification sweep ======
for s in "${SIGMAS[@]}"; do
  run_one "sgd_amp" "${s}" "sweep"
done

# ====== 3) DP-SGD no-amplification reporting sweep ======
for s in "${SIGMAS[@]}"; do
  run_one "sgd_noamp" "${s}" "sweep"
done

echo "All sweeps finished. Logs in: ${OUT_DIR}"
echo "Results JSONL should be under: ${OUT_DIR}/results.jsonl and ${OUT_DIR}/${DATA}/results.jsonl"

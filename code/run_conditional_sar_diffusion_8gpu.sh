#!/usr/bin/env bash
# Start the short DDPM pilot only after all A100s have been idle long enough.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../A02}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/runs/conditional_ddpm64_pilot_20260904}"
IDLE_POLLS="${IDLE_POLLS:-2}"
POLL_SECONDS="${POLL_SECONDS:-30}"

all_gpus_idle() {
  local rows
  rows="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  [[ "$(wc -l <<<"${rows}")" -ge 8 ]] || return 1
  awk -F, '$1 + 0 > 1000 || $2 + 0 > 10 { busy = 1 } END { exit busy }' <<<"${rows}"
}

idle=0
while (( idle < IDLE_POLLS )); do
  if all_gpus_idle; then
    ((idle += 1))
    echo "[$(date -Is)] all GPUs idle (${idle}/${IDLE_POLLS})"
  else
    idle=0
    echo "[$(date -Is)] waiting for existing GPU work to finish"
  fi
  (( idle < IDLE_POLLS )) && sleep "${POLL_SECONDS}"
done

cd "${SCRIPT_DIR}"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec torchrun --standalone --nproc-per-node=8 train_conditional_sar_diffusion.py \
  --rgb-root "${DATA_ROOT}/RGB" \
  --sar-train-root "${DATA_ROOT}/SOC_40classes_cut/train" \
  --output "${OUTPUT}" \
  --band all --polarization all --depression all \
  --epochs 20 --epoch-size 24000 --batch-size 8 --workers 4 \
  --base 64 --rgb-base 32 --token-dim 256 \
  --diffusion-steps 1000 --condition-drop-prob .10 \
  --lr 0.0002 --weight-decay 0.0001 --ema-decay .9999 \
  --sample-steps 24 --guidance-scale 1.0 --preview-every 1 --save-every 5 \
  --seed 20260904

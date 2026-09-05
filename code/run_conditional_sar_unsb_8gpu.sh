#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${CODE_DIR}"

RGB_ROOT="${RGB_ROOT:-../A02/RGB}"
SAR_ROOT="${SAR_ROOT:-../A02/SOC_40classes_cut/train}"
OUTPUT="${OUTPUT:-runs/unsb_sar_bridge_all_conditions}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCH_SIZE="${EPOCH_SIZE:-24000}"
WORKERS="${WORKERS:-4}"
BRIDGE_STEPS="${BRIDGE_STEPS:-5}"
SAMPLE_STEPS="${SAMPLE_STEPS:-5}"
PREVIEW_EVERY="${PREVIEW_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-1}"
RESUME="${RESUME:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-0}"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi
if [[ "${LIMIT_TRAIN_BATCHES}" != "0" ]]; then
  EXTRA_ARGS+=(--limit-train-batches "${LIMIT_TRAIN_BATCHES}")
fi

exec torchrun --standalone --nproc-per-node=8 train_conditional_sar_unsb.py \
  --rgb-root "${RGB_ROOT}" \
  --sar-train-root "${SAR_ROOT}" \
  --output "${OUTPUT}" \
  --band all --polarization all --depression all \
  --epochs "${EPOCHS}" --epoch-size "${EPOCH_SIZE}" \
  --batch-size "${BATCH_SIZE}" --workers "${WORKERS}" \
  --base 64 --token-dim 256 --control-base 32 \
  --discriminator-base 32 --energy-base 16 --patch-base 16 \
  --bridge-steps "${BRIDGE_STEPS}" --sample-steps "${SAMPLE_STEPS}" \
  --preview-every "${PREVIEW_EVERY}" --save-every "${SAVE_EVERY}" \
  "${EXTRA_ARGS[@]}"

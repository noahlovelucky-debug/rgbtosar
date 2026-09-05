#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../A02}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/runs/hifc_domain_uniform_full_20260905}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# Publication candidate: train from a fresh initialization.  The only model
# variable relative to the record-frequency control is the condition sampler.
export CUDA_VISIBLE_DEVICES="${GPUS}"
exec torchrun --standalone --nproc-per-node=8 \
  "${SCRIPT_DIR}/train_hifc_unpaired_sar_gan.py" \
  --rgb-root "${DATA_ROOT}/RGB" \
  --sar-train-root "${DATA_ROOT}/SOC_40classes_cut/train" \
  --native-classifier-checkpoint "${SCRIPT_DIR}/server_results/sar_native64_multitask_v1/best.pt" \
  --output "${OUTPUT}" \
  --band all \
  --polarization all \
  --depression all \
  --epochs 120 \
  --epoch-size 24000 \
  --batch-size 8 \
  --workers 2 \
  --generator-lr 0.00015 \
  --identity-lr 0.0001 \
  --discriminator-lr 0.0001 \
  --condition-sampler domain_uniform \
  --condition-sampler-seed 20260830

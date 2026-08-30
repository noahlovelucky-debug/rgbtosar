#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../A02}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/runs/hifc_unpaired_all_conditions}"

# HiFC-inspired unpaired training.  All SAR conditions are kept in the
# training set so band/polarisation are learnable; evaluate X/HH separately.
exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_hifc_unpaired_sar_gan.py" \
  --rgb-root "${DATA_ROOT}/RGB" \
  --sar-train-root "${DATA_ROOT}/SOC_40classes_cut/train" \
  --native-classifier-checkpoint "${SCRIPT_DIR}/server_results/sar_native64_multitask_v1/best.pt" \
  --output "${OUTPUT}" \
  --band all \
  --polarization all \
  --depression all \
  --epochs 120 \
  --epoch-size 24000 \
  --batch-size 16 \
  --workers 4 \
  --generator-lr 0.00015 \
  --identity-lr 0.0001 \
  --discriminator-lr 0.0001 \
  --device "${DEVICE}"


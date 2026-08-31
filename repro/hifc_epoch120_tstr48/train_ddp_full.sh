#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory containing RGB and SOC_40classes_cut}"
OUTPUT="${OUTPUT:?Set OUTPUT to the run directory}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to split_manifest__all_all_all.json}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29680}"

if [[ ! -f "$SPLIT_MANIFEST" ]]; then
  printf 'Missing split manifest: %s\\n' "$SPLIT_MANIFEST" >&2
  exit 2
fi
mkdir -p "$OUTPUT"
cp "$SPLIT_MANIFEST" "$OUTPUT/split_manifest__all_all_all.json"

exec torchrun --standalone --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
  "$REPO_ROOT/code/train_hifc_unpaired_sar_gan.py" \
  --rgb-root "$DATA_ROOT/RGB" \
  --sar-train-root "$DATA_ROOT/SOC_40classes_cut/train" \
  --native-classifier-checkpoint "$REPO_ROOT/checkpoints/native_classifier_best.pt" \
  --output "$OUTPUT" \
  --band all --polarization all --depression all \
  --epochs 120 --epoch-size 24000 --batch-size 8 --workers 2 \
  --generator-lr 0.00015 --identity-lr 0.0001 --discriminator-lr 0.0001 \
  --seed 20260830 --native-gradient-mode full --device cuda:0

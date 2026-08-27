#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/noah/workspace/rgb2sar_direction_gan/.conda/bin/python
DATASET_ROOT="../amplitude 8-bit data_地距幅度8位数据.7z"
VALIDATOR_OUTPUT="server_results/sar_geometry_validator_xhh_v2"
GAN_OUTPUT="runs/dual_component_v2_xhh"
DEVICE="cuda:2"
mkdir -p "$VALIDATOR_OUTPUT" "$GAN_OUTPUT"

if [[ ! -f "$VALIDATOR_OUTPUT/best.pt" ]]; then
  "$PYTHON" -u train_sar_geometry_validator.py \
    --train-root "$DATASET_ROOT/SOC_40classes_cut/train" \
    --test-root "$DATASET_ROOT/SOC_40classes_cut/test" \
    --classifier-initialization server_results/sar_native64_multitask_v1/best.pt \
    --output "$VALIDATOR_OUTPUT" \
    --epochs 60 \
    --batch-size 128 \
    --workers 8 \
    --device "$DEVICE" 2>&1 | tee -a "$VALIDATOR_OUTPUT/train.log"
fi

resume_args=()
if [[ -f "$GAN_OUTPUT/latest.pt" ]]; then
  resume_args=(--resume "$GAN_OUTPUT/latest.pt")
fi

"$PYTHON" -u train_dual_component_sar_gan_v2.py \
  --rgb-root "$DATASET_ROOT/RGB" \
  --sar-train-root "$DATASET_ROOT/SOC_40classes_cut/train" \
  --geometry-validator-checkpoint "$VALIDATOR_OUTPUT/best.pt" \
  --output "$GAN_OUTPUT" \
  --clean-epochs 70 \
  --noise-epochs 40 \
  --joint-epochs 40 \
  --epoch-size 16000 \
  --batch-size 16 \
  --workers 4 \
  "${resume_args[@]}" \
  --device "$DEVICE" 2>&1 | tee -a "$GAN_OUTPUT/train.log"

"$PYTHON" audit_dual_component_sar_v2.py \
  --gan-checkpoint "$GAN_OUTPUT/latest.pt" \
  --geometry-validator-checkpoint "$VALIDATOR_OUTPUT/best.pt" \
  --rgb-root "$DATASET_ROOT/RGB" \
  --sar-root "$DATASET_ROOT/SOC_40classes_cut/test" \
  --output "$GAN_OUTPUT/official_test_audit.json" \
  --device "$DEVICE"

for component in full clean noise; do
  "$PYTHON" render_dual_component_sar_v2.py \
    --gan-checkpoint "$GAN_OUTPUT/latest.pt" \
    --rgb-root "$DATASET_ROOT/RGB" \
    --class-name Buick_GL8 \
    --component "$component" \
    --noise-mode fixed \
    --output "$GAN_OUTPUT/Buick_GL8_all_depressions_${component}.png" \
    --device "$DEVICE"
done

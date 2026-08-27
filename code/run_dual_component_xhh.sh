#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/noah/workspace/rgb2sar_direction_gan/.conda/bin/python
OUTPUT=runs/dual_component_xhh

"$PYTHON" -u train_dual_component_sar_gan.py \
  --rgb-root "../amplitude 8-bit data_地距幅度8位数据.7z/RGB" \
  --sar-train-root "../amplitude 8-bit data_地距幅度8位数据.7z/SOC_40classes_cut/train" \
  --native-classifier-checkpoint server_results/sar_native64_multitask_v1/best.pt \
  --output "$OUTPUT" \
  --epochs 120 \
  --epoch-size 24000 \
  --batch-size 16 \
  --workers 4 \
  --generator-lr 0.00015 \
  --identity-lr 0.0001 \
  --discriminator-lr 0.0001 \
  --device cuda:1

"$PYTHON" render_dual_component_sar.py \
  --gan-checkpoint "$OUTPUT/latest.pt" \
  --rgb-root "../amplitude 8-bit data_地距幅度8位数据.7z/RGB" \
  --class-name Buick_GL8 \
  --source-angle 0 \
  --output "$OUTPUT/Buick_GL8_all_depressions.png" \
  --device cuda:1

#!/usr/bin/env bash
set -euo pipefail
project=/home/noah/workspace/rgb2sar_direction_gan
base='/home/noah/workspace/DS_datasets/amplitude 8-bit data_地距幅度8位数据.7z'
cut='/home/noah/workspace/DS_datasets/SOC_40classes_cut'
cd "$project"
.conda/bin/python train_bbox.py --rgb-root "$base/RGB" --sar-root "$cut/train" --pre-cropped \
  --output runs/identity_conditioned_gan --epochs 15 --epoch-size 5000 --batch-size 32 --workers 0 --device cuda:0 \
  --aligned-checkpoint runs/rgb_sar_alignment_cut/best.pt --freeze-rgb-encoder --identity-weight 10 --l1-weight 15
.conda/bin/python visualize_bbox_swaps.py --checkpoint runs/identity_conditioned_gan/latest.pt --rgb-root "$base/RGB" --sar-root "$base/SOC_40classes/test" --output runs/identity_conditioned_gan/swaps.png --samples 8 --device cuda:0

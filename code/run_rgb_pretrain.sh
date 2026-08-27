#!/usr/bin/env bash
set -euo pipefail
project=/home/noah/workspace/rgb2sar_direction_gan
base='/home/noah/workspace/DS_datasets/amplitude 8-bit data_地距幅度8位数据.7z'
cd "$project"
.conda/bin/python train_rgb_encoder.py --rgb-root "$base/RGB" --output runs/rgb_multiview_encoder --epochs 200 --episodes-per-class 32 --batch-size 64 --device cuda:0
.conda/bin/python train_bbox.py --rgb-root "$base/RGB" --sar-root "$base/SOC_40classes/train" --output runs/bbox_frozen_rgb_demo --epochs 20 --epoch-size 2000 --batch-size 16 --workers 0 --device cuda:0 --rgb-encoder-checkpoint runs/rgb_multiview_encoder/best.pt --freeze-rgb-encoder
.conda/bin/python visualize_bbox_swaps.py --checkpoint runs/bbox_frozen_rgb_demo/latest.pt --rgb-root "$base/RGB" --sar-root "$base/SOC_40classes/test" --output runs/bbox_frozen_rgb_demo/swaps.png --samples 6 --device cuda:0

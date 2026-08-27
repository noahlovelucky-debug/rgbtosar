#!/usr/bin/env bash
set -euo pipefail
project=/home/noah/workspace/rgb2sar_direction_gan
base='/home/noah/workspace/DS_datasets/amplitude 8-bit data_地距幅度8位数据.7z'
cd "$project"
.conda/bin/python train_bbox.py --rgb-root "$base/RGB" --sar-root "$base/SOC_40classes/train" --output runs/bbox_swap_demo_v3 --epochs 30 --epoch-size 1000 --batch-size 16 --workers 0 --device cuda:0
.conda/bin/python visualize_bbox_swaps.py --checkpoint runs/bbox_swap_demo_v3/latest.pt --rgb-root "$base/RGB" --sar-root "$base/SOC_40classes/test" --output runs/bbox_swap_demo_v3/swaps.png --samples 6 --device cuda:0

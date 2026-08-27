#!/usr/bin/env bash
set -euo pipefail
project=/home/noah/workspace/rgb2sar_direction_gan
base='/home/noah/workspace/DS_datasets/amplitude 8-bit data_地距幅度8位数据.7z'
cd "$project"
.conda/bin/python train.py --rgb-root "$base/RGB" --sar-root "$base/SOC_40classes/train" --rgb-index 1 \
  --angle-tolerance 15 --band X --polarization HH --depression 30 \
  --output runs/angle_a_x_hh_30 --image-size 128 --batch-size 4 --epoch-size 400 --epochs 10 --workers 4 --device cuda:0
.conda/bin/python visualize.py --checkpoint runs/angle_a_x_hh_30/latest.pt --output-dir runs/angle_a_x_hh_30/visualization --samples 8 --device cuda:0

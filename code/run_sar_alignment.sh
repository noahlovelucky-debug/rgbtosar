#!/usr/bin/env bash
set -euo pipefail
project=/home/noah/workspace/rgb2sar_direction_gan
base='/home/noah/workspace/DS_datasets/amplitude 8-bit data_地距幅度8位数据.7z'
cut='/home/noah/workspace/DS_datasets/SOC_40classes_cut'
cd "$project"
.conda/bin/python train_sar_encoder.py --train-root "$cut/train" --test-root "$cut/test" --output runs/sar_identity_encoder_cut --epochs 15 --batch-size 256 --workers 8 --pre-cropped --device cuda:0
.conda/bin/python align_rgb_sar.py --rgb-root "$base/RGB" --sar-train-root "$cut/train" --sar-test-root "$cut/test" --rgb-checkpoint runs/rgb_multiview_encoder/best.pt --sar-checkpoint runs/sar_identity_encoder_cut/best.pt --output runs/rgb_sar_alignment_cut --epochs 5 --batch-size 64 --pre-cropped --device cuda:0

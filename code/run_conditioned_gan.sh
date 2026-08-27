#!/usr/bin/env bash
set -euo pipefail
project=/home/noah/workspace/rgb2sar_direction_gan
base='/home/noah/workspace/DS_datasets/amplitude 8-bit data_地距幅度8位数据.7z'
cut='/home/noah/workspace/DS_datasets/SOC_40classes_cut'
cd "$project"
.conda/bin/python train_sar_condition.py --train-root "$cut/train" --test-root "$cut/test" --output runs/sar_condition_encoder --epochs 10 --device cuda:0
.conda/bin/python train_bbox.py --rgb-root "$base/RGB" --sar-root "$cut/train" --pre-cropped --output runs/identity_angle_conditioned_gan --epochs 15 --epoch-size 5000 --batch-size 32 --device cuda:0 --aligned-checkpoint runs/rgb_sar_alignment_cut/best.pt --condition-checkpoint runs/sar_condition_encoder/best.pt --identity-weight 10 --condition-weight 5 --l1-weight 15
.conda/bin/python visualize_conditions.py --checkpoint runs/identity_angle_conditioned_gan/latest.pt --rgb-root "$base/RGB" --sar-root "$base/SOC_40classes/test" --source-class Buick_GL8 --output runs/identity_angle_conditioned_gan/conditions.png --device cuda:0

#!/usr/bin/env bash
set -euo pipefail

code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
dataset_root=$(cd -- "$code_dir/.." && pwd)
base="$dataset_root/amplitude 8-bit data_地距幅度8位数据.7z"
python_bin=${PYTHON_BIN:-python}
classifier_dir="$code_dir/server_results/saratrx_64_cut_stage2"
native_classifier_dir="$code_dir/server_results/sar_native64_multitask_v1"
run_dir="$code_dir/runs/joint_native_judge_gan"
legacy_prototype_cache="$code_dir/runs/joint_identity_roi_gan/saratrx_prototypes.pt"
previous_gan="$code_dir/runs/joint_identity_roi_gan/selected.pt"

if [[ ! -f "$classifier_dir/best.pt" ]]; then
  "$python_bin" "$code_dir/finetune_saratrx_64.py" \
    --checkpoint "$dataset_root/分类器/SARatrX/model/SOC_40classes.pth" \
    --train-root "$base/SOC_40classes_cut/train" \
    --test-root "$base/SOC_40classes_cut/test" \
    --output "$classifier_dir" --epochs 20 --batch-size 128 --workers 8 --device cuda:0
fi

if [[ ! -f "$native_classifier_dir/best.pt" ]]; then
  "$python_bin" "$code_dir/train_sar_classifier_64.py" \
    --train-root "$base/SOC_40classes_cut/train" \
    --test-root "$base/SOC_40classes_cut/test" \
    --output "$native_classifier_dir" --epochs 30 --batch-size 256 --workers 8 --device cuda:0
fi

initialise_args=()
if [[ -f "$previous_gan" ]]; then
  # Reuse the already learned RGB/SAR structure, then immediately optimise it
  # against the stronger independent classifier under full speckle.
  initialise_args=(--initialise-from "$previous_gan" --speckle-warmup-epochs 0 --speckle-ramp-epochs 1)
fi

"$python_bin" "$code_dir/train_joint_roi_gan.py" \
  --rgb-root "$base/RGB" \
  --sar-root "$base/SOC_40classes_cut/train" \
  --saratrx-checkpoint "$classifier_dir/best.pt" \
  --native-classifier-checkpoint "$native_classifier_dir/best.pt" \
  --output "$run_dir" \
  --epochs 30 --epoch-size 5000 --batch-size 32 --prototype-batch-size 128 \
  --band X --polarization HH --depression 30 --workers 4 --device cuda:0 "${initialise_args[@]}"

"$python_bin" "$code_dir/select_joint_roi_checkpoint.py" \
  --run-dir "$run_dir" \
  --output "$run_dir/selected.pt"

if [[ -f "$legacy_prototype_cache" ]]; then
  "$python_bin" "$code_dir/evaluate_joint_roi_gan.py" \
    --checkpoint "$run_dir/selected.pt" \
    --prototype-cache "$legacy_prototype_cache" \
    --rgb-root "$base/RGB" \
    --sar-root "$base/SOC_40classes_cut/test" \
    --saratrx-checkpoint "$classifier_dir/best.pt" \
    --output "$run_dir/legacy_classifier_audit.json" \
    --samples 5000 --batch-size 64 --band X --polarization HH --depression 30 \
    --workers 4 --device cuda:0
fi

"$python_bin" "$code_dir/visualize_joint_roi_gan.py" \
  --checkpoint "$run_dir/selected.pt" \
  --rgb-root "$base/RGB" \
  --sar-root "$base/SOC_40classes_cut/test" \
  --saratrx-checkpoint "$classifier_dir/best.pt" \
  --output "$run_dir/rgb_to_sar_visualization.png" \
  --samples 24 --batch-size 8 --band X --polarization HH --depression 30 --device cuda:0

"$python_bin" "$code_dir/evaluate_joint_native_classifier.py" \
  --gan-checkpoint "$run_dir/selected.pt" \
  --classifier-checkpoint "$native_classifier_dir/best.pt" \
  --rgb-root "$base/RGB" --sar-root "$base/SOC_40classes_cut/test" \
  --output "$run_dir/native_classifier_audit.json" \
  --batch-size 64 --workers 4 --band X --polarization HH --depression 30 --device cuda:0

#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/noah/workspace/rgb2sar_direction_gan/.conda/bin/python
DATASET_ROOT="../amplitude 8-bit data_地距幅度8位数据.7z"
OUTPUT="runs/continuous_spatial_one_stage_v3_xhh"
V1="runs/continuous_spatial_x_hh/best.pt"
SPLIT="runs/dual_component_v2_xhh/split_manifest.json"
GEOMETRY_VALIDATOR="server_results/sar_geometry_validator_xhh_v2/best.pt"
NATIVE_CLASSIFIER="server_results/sar_native64_multitask_v1/best.pt"
DEVICE="cuda:2"

mkdir -p "$OUTPUT"
resume=()
if [[ -f "$OUTPUT/latest.pt" ]]; then
  resume=(--resume "$OUTPUT/latest.pt")
fi

"$PYTHON" -u train_continuous_spatial_one_stage_v3.py \
  --rgb-root "$DATASET_ROOT/RGB" \
  --sar-train-root "$DATASET_ROOT/SOC_40classes_cut/train" \
  --v1-checkpoint "$V1" \
  --split-manifest-source "$SPLIT" \
  --output "$OUTPUT" \
  --epochs 80 \
  --epoch-size 16000 \
  --batch-size 8 \
  --gradient-accumulation 2 \
  --workers 4 \
  --device "$DEVICE" \
  "${resume[@]}" 2>&1 | tee -a "$OUTPUT/train.log"

CHECKPOINT="$OUTPUT/best_visual.pt"
"$PYTHON" audit_continuous_spatial_one_stage_v3.py \
  --gan-checkpoint "$CHECKPOINT" \
  --geometry-validator-checkpoint "$GEOMETRY_VALIDATOR" \
  --native-classifier-checkpoint "$NATIVE_CLASSIFIER" \
  --rgb-root "$DATASET_ROOT/RGB" \
  --sar-root "$DATASET_ROOT/SOC_40classes_cut/test" \
  --output "$OUTPUT/official_test_audit.json" \
  --device "$DEVICE"

for mode in fixed independent; do
  "$PYTHON" render_continuous_spatial_one_stage_v3.py \
    --gan-checkpoint "$CHECKPOINT" \
    --rgb-root "$DATASET_ROOT/RGB" \
    --class-name Buick_GL8 \
    --noise-mode "$mode" \
    --output "$OUTPUT/Buick_GL8_all_depressions_${mode}.png" \
    --device "$DEVICE"
done

"$PYTHON" compare_continuous_sar_models_v3.py \
  --v1-checkpoint "$V1" \
  --wavelet-checkpoint runs/one_stage_wavelet_xhh/latest.pt \
  --dual-v2-checkpoint runs/dual_component_v2_xhh/latest.pt \
  --v3-checkpoint "$CHECKPOINT" \
  --rgb-root "$DATASET_ROOT/RGB" \
  --sar-root "$DATASET_ROOT/SOC_40classes_cut/test" \
  --output "$OUTPUT/model_comparison.png" \
  --device "$DEVICE"

"$PYTHON" benchmark_continuous_spatial_one_stage_v3.py \
  --v3-checkpoint "$CHECKPOINT" \
  --v1-checkpoint "$V1" \
  --rgb-root "$DATASET_ROOT/RGB" \
  --output "$OUTPUT/inference_benchmark.json" \
  --device "$DEVICE"

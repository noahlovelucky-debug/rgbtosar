#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_root="${script_dir}/runs/v1_ablation"

exec "${PYTHON_BIN:-python}" "${script_dir}/train_continuous_spatial_v1_ablation.py" \
  --rgb-root /data/newdata/A25_T37_down_大图/A02/RGB \
  --sar-root /data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/train \
  --native-classifier-checkpoint "${script_dir}/server_results/sar_native64_multitask_v1/best.pt" \
  --initialise-checkpoint "${script_dir}/runs/continuous_spatial_x_hh/milestone_0070.pt" \
  --geometry-validator-checkpoint "${script_dir}/server_results/sar_geometry_validator_xhh_v2/best.pt" \
  --prototype-cache "${run_root}/native_conditional_prototypes.pt" \
  --split-manifest "${run_root}/split.json" \
  --validation-proxy-manifest "${run_root}/validation_proxy_640.json" \
  --output "${run_root}/recommended_sar_class_1" \
  --epochs 30 --epoch-size 4000 --batch-size 32 --workers 4 \
  --validation-batches 20 --save-every 1 \
  --sar-class-weight 1 --device "${V1_ABLATION_DEVICE:-cuda:0}" "$@"

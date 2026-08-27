#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
run_root="${script_dir}/runs/v1_ablation"

run_one() {
  local seed="$1"
  local tag="$2"
  local parent="${run_root}/P1_seed${seed}_control/epoch_0082.pt"
  local output="${run_root}/${tag}_seed${seed}"
  shift 2
  "${python_bin}" "${script_dir}/train_continuous_spatial_v1_ablation.py" \
    --rgb-root /data/newdata/A25_T37_down_大图/A02/RGB \
    --sar-root /data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/train \
    --native-classifier-checkpoint "${script_dir}/server_results/sar_native64_multitask_v1/best.pt" \
    --initialise-checkpoint "${parent}" --parent-epoch 82 \
    --geometry-validator-checkpoint "${script_dir}/server_results/sar_geometry_validator_xhh_v2/best.pt" \
    --prototype-cache "${run_root}/native_conditional_prototypes.pt" \
    --split-manifest "${run_root}/split.json" \
    --validation-proxy-manifest "${run_root}/validation_proxy_640.json" \
    --output "${output}" \
    --epochs 4 --epoch-size 4000 --batch-size 32 --workers 4 \
    --validation-batches 20 --save-every 1 --device "${V1_ABLATION_DEVICE:-cuda:0}" \
    --seed "${seed}" "$@"
}

for seed in 2718 451 9201; do
  run_one "${seed}" A0_no_angle --angle-smooth-weight 0
done

for seed in 2718 451 9201; do
  run_one "${seed}" A1_curvature --angle-loss-mode curvature --angle-smooth-weight .2
done

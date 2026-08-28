#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
run_root="${script_dir}/runs/v1_ablation"
validator="${script_dir}/server_results/sar_geometry_validator_xhh_v2/best.pt"
rgb_root=/data/newdata/A25_T37_down_大图/A02/RGB
sar_root=/data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/train

run_one() {
  local tag="$1"
  local seed="$2"
  "${python_bin}" "${script_dir}/probe_v1_cross_domain_transfer.py" \
    --gan-checkpoint "${run_root}/${tag}_long_seed${seed}/epoch_0116.pt" \
    --geometry-validator-checkpoint "${validator}" \
    --rgb-root "${rgb_root}" --sar-root "${sar_root}" \
    --split-manifest "${run_root}/split.json" \
    --output "${run_root}/${tag}_long_seed${seed}/transfer.json" \
    --seed "${seed}" --device "${V1_ABLATION_DEVICE:-cuda:0}"
}

for tag in D0_class_head_disabled D1_class_real_only; do
  for seed in 2718 451 9201; do
    run_one "${tag}" "${seed}"
  done
done

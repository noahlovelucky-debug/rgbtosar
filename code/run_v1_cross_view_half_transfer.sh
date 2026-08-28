#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
run_root="${script_dir}/runs/v1_ablation"
validator="${script_dir}/server_results/sar_geometry_validator_xhh_v2/best.pt"

for seed in 2718 451 9201; do
  "${python_bin}" "${script_dir}/probe_v1_cross_domain_transfer.py" \
    --gan-checkpoint "${run_root}/G1_cross_view_half_seed${seed}/epoch_0104.pt" \
    --geometry-validator-checkpoint "${validator}" \
    --rgb-root /data/newdata/A25_T37_down_大图/A02/RGB \
    --sar-root /data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/train \
    --split-manifest "${run_root}/split.json" \
    --output "${run_root}/G1_cross_view_half_seed${seed}/transfer.json" \
    --seed "${seed}" --device "${V1_ABLATION_DEVICE:-cuda:0}"
done

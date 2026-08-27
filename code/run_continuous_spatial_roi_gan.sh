#!/usr/bin/env bash
# X/HH only.  Train all 15/30/45/60 degree depressions with continuous azimuth.
set -euo pipefail
code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
base_dir=$(cd "$code_dir/.." && pwd)
python_bin=${PYTHON_BIN:-$(command -v python)}
rgb_root="$base_dir/A02/RGB"
train_root="$base_dir/A02/SOC_40classes_cut/train"
test_root="$base_dir/A02/SOC_40classes_cut/test"
classifier="$code_dir/server_results/sar_native64_multitask_v1/best.pt"
run_dir="$code_dir/runs/continuous_spatial_fused_v2_x_hh"

if [[ ! -f "$classifier" ]]; then
  echo "Missing 64x64 native classifier: $classifier" >&2
  echo "Run train_sar_classifier_64.py first; GAN training deliberately refuses the old 224x224 judge." >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter is not executable: $python_bin" >&2
  exit 2
fi

"$python_bin" "$code_dir/train_continuous_spatial_roi_gan.py" \
  --rgb-root "$rgb_root" --sar-root "$train_root" \
  --native-classifier-checkpoint "$classifier" --output "$run_dir" \
  --epochs 100 --epoch-size 8000 --batch-size 32 --workers 4 --device cuda:0

checkpoint="$run_dir/best.pt"
if [[ ! -f "$checkpoint" ]]; then checkpoint="$run_dir/latest.pt"; fi
"$python_bin" "$code_dir/evaluate_continuous_spatial_roi_gan.py" \
  --gan-checkpoint "$checkpoint" --classifier-checkpoint "$classifier" \
  --rgb-root "$rgb_root" --sar-root "$test_root" --output "$run_dir/test_audit.json" \
  --batch-size 128 --workers 4 --device cuda:0

"$python_bin" "$code_dir/render_continuous_spatial_sar.py" \
  --gan-checkpoint "$checkpoint" --rgb-root "$rgb_root" --class-name Buick_GL8 --source-angle 0 \
  --output "$run_dir/Buick_GL8_all_depressions.png" --device cuda:0

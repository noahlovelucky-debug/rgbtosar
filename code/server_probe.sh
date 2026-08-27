#!/usr/bin/env bash
set -euo pipefail
base='/home/noah/workspace/DS_datasets/amplitude 8-bit data_地距幅度8位数据.7z'
echo '[data dirs]'
find "$base" -maxdepth 1 -mindepth 1 -printf '%f\n' | sort | head -30
echo '[counts: RGB, cut train SAR, full train SAR]'
find "$base/RGB" -type f -iname '*.png' | wc -l
find "$base/SOC_40classes_cut/train" -type f -iname '*.tif' 2>/dev/null | wc -l || true
find "$base/SOC_40classes/train" -type f -iname '*.tif' 2>/dev/null | wc -l || true
echo '[conda candidates]'
find "$HOME" -maxdepth 3 -type f -path '*/bin/conda' -print 2>/dev/null || true
echo '[system torch]'
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' 2>/dev/null || true

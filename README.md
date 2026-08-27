# RGB to SAR

Research code for conditional RGB-to-SAR vehicle ROI generation. The current
training entry point is `code/train_continuous_spatial_roi_gan.py`.

## Fused V2

`continuous_spatial_fused_v2` uses a single K+1 SAR classifier-discriminator:
40 logits represent real vehicle identities and one logit represents generated
SAR. It is initialised from the native 64px SAR classifier. The generator is
trained with RGB CosFace identity, class-adversarial, non-registered structure,
physics, and angular-curvature objectives.

The non-registered structure objective compares low-frequency distributions,
residual/Sobel statistics, and discriminator feature moments. It does not
compare aligned SAR pixels.

## Layout

- `code/`: training, evaluation, rendering, models, and data loading.
- `visualizations/continuous_spatial_v1/`: representative historical V1
  previews, all-depression rendering, training history, and audit metrics.
- `visualizations/continuous_spatial_fused_v2/`: in-progress Fused V2 training
  history and the first-epoch preview. This is an early optimisation snapshot,
  not a final quality claim.

Datasets, model checkpoints, and generated run directories are intentionally
excluded from this repository.

## Training

Set the dataset roots in `code/run_continuous_spatial_roi_gan.sh`, then run:

```bash
cd code
bash run_continuous_spatial_roi_gan.sh
```

The script expects an RGB root, a `SOC_40classes_cut/train` SAR root, and a
native 64px SAR classifier checkpoint.

## Fused V2 Run Status

The tracked preview and history were exported after epoch 1 of the active
100-epoch run. The following later checks completed through epoch 3 without
NaN, CUDA OOM, or training-process errors: RGB identity accuracy reached
0.976, real SAR class accuracy 0.983, and generated-SAR rejection accuracy
0.977. Checkpoints, source data, and raw logs remain local and are excluded.

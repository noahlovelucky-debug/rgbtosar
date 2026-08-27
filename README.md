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
- `visualizations/continuous_spatial_fused_v2/`: final Fused V2 training history,
  previews, a full azimuth/depression scan, and test audit metrics.

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

The 100-epoch Fused V2 run completed without NaN, CUDA OOM, or process errors.
The tracked audit covers 5,260 test samples: RGB identity top-1 is 1.000,
generated SAR native-classifier top-1 is 0.9998, real SAR native-classifier
top-1 is 0.9234, and feature cosine to real SAR is 0.4087. These values are
diagnostic measurements, not an independent perceptual-quality claim.
Checkpoints, source data, and raw logs remain local and are excluded.

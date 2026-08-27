# RGB to SAR

Research code for conditional RGB-to-SAR vehicle ROI generation.

## V1 Ablation (Active)

The active experiment path is the preserved V1 PatchGAN, not Fused V2. It
changes one loss or discriminator property at a time from the same V1
milestone, uses a fixed balanced holdout proxy, and selects with a frozen
real-SAR geometry validator plus generated-to-real transfer probes. See
[`code/V1_ABLATION_PROTOCOL.md`](code/V1_ABLATION_PROTOCOL.md).

Three matched short screens found that reducing V1 `sar_class_weight` from 12
to 1 improves generated-to-real identity transfer while retaining all frozen
geometry gates; a 2,000-step confirmation also improved all three primary
transfer axes in all three matched seeds (identity +0.78 pp, depression +3.59
pp, azimuth MAE -2.31 degrees on average). The recommended next run therefore
passes `--sar-class-weight 1` while keeping all other V1 coefficients unchanged.
In contrast, removing only the weak,
translation-aligned 64px pixel term has no consistent transfer benefit and is
not the default change. The prototype-cluster objective cannot be removed or
lightly reduced without a depression-transfer regression in the current
screens, so it remains at its V1 value. Artifacts are under
[`visualizations/v1_ablation/`](visualizations/v1_ablation/).

Follow-up one-variable screens on SSIM, edge, physics scattering, angle
regularisation, and discriminator conditioning are recorded in the same
directory. They reject the large Fused V2/K+1 discriminator change for now:
the existing V1 PatchGAN remains the reproducible baseline, and no change is
promoted without paired generated-to-real transfer evidence across seeds.

The recommended epoch-100 run completed without NaN/OOM. Its independent
generated-to-real transfer is identity `98.59%`, depression `85.16%`, and
azimuth MAE `23.07°`; preview, metrics, history, and JSON are in
[`visualizations/v1_ablation/`](visualizations/v1_ablation/).

## Fused V2 (Historical)

`continuous_spatial_fused_v2` uses a single K+1 SAR classifier-discriminator:
40 logits represent real vehicle identities and one logit represents generated
SAR. It is initialised from the native 64px SAR classifier. The generator is
trained with RGB CosFace identity, class-adversarial, non-registered structure,
physics, and angular-curvature objectives.

The non-registered structure objective compares low-frequency distributions,
residual/Sobel statistics, and discriminator feature moments. It does not
compare aligned SAR pixels. This run changed too many variables at once and is
kept only as a diagnostic reference, not as the current baseline.

## Layout

- `code/`: training, evaluation, rendering, models, and data loading.
- `visualizations/continuous_spatial_v1/`: representative historical V1
  previews, all-depression rendering, training history, and audit metrics.
- `visualizations/continuous_spatial_fused_v2/`: final Fused V2 training history,
  previews, a full azimuth/depression scan, and test audit metrics.
- `visualizations/v1_ablation/`: V1 loss-gradient probe, fixed-proxy milestone
  audits, paired L1/S1 screen data, previews, and multi-seed comparison plots.

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

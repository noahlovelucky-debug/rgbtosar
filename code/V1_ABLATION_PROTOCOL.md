# V1 One-Variable Ablation Protocol

## Why V1 is the baseline

The Fused V2 run changed the discriminator, objective, loss weights, learning
rates, selection rule, and initialization at once. It cannot identify which
change caused its loss of geometry. `train_continuous_spatial_v1_ablation.py`
therefore preserves V1's RGB encoder, spatial generator, conditional PatchGAN,
optimizers, and default loss coefficients in a separate entry point.

All screens branch from `milestone_0070.pt`. It has full-strength speckle and
already contains usable conditional geometry, while leaving room to observe
whether a change helps or hurts. Every branch uses the same fixed split and
the same balanced 640-condition validation proxy.

## Evidence Before Changing Losses

The native SAR teacher was already nearly perfect at epoch 10, but independent
geometry was not. Its score must not select an ablation winner.

| V1 epoch | native fake class accuracy | frozen geometry azimuth MAE | frozen geometry depression top-1 | generated-to-real azimuth MAE |
| --- | ---: | ---: | ---: | ---: |
| 10 | 99.99% | 83.3 deg | 23.8% | 74.9 deg |
| 70 | 99.95% | 46.2 deg | 45.6% | 34.8 deg |
| 100 | 100.00% | 43.5 deg | 59.7% | 29.5 deg |

The final column trains only a small readout on generated frozen features and
tests it on held-out real SAR. It is the useful answer to whether information
from the generated domain transfers to the real domain. The `100%` native
score alone cannot answer that question because V1 directly optimizes it.

## Screen Procedure

`run_v1_ablation_screen.sh` defaults to four epochs of 4,000 sampled
conditions with batch 32: 500 generator updates, with a frozen evaluation
after each 125 updates. A screen is a rejector, not final evidence.

```bash
cd /data/newdata/A25_T37_down_大图/code
PYTHON_BIN=/home/star/anaconda3/envs/tessera/bin/python ./run_v1_ablation_screen.sh C0
PYTHON_BIN=/home/star/anaconda3/envs/tessera/bin/python ./run_v1_ablation_screen.sh L1a_sar_class_1 --sar-class-weight 1
PYTHON_BIN=/home/star/anaconda3/envs/tessera/bin/python ./run_v1_ablation_screen.sh L1b_sar_class_0 --sar-class-weight 0
python compare_v1_ablation_screens.py --control runs/v1_ablation/C0 --candidates runs/v1_ablation/L1a_sar_class_1 runs/v1_ablation/L1b_sar_class_0 --output runs/v1_ablation/L1_screen_report.json
```

A candidate must not regress by more than the fixed gates on identity,
depression, azimuth, +30 degree response, frozen feature cosine, and aligned
lowpass error. A pass is not an automatic winner; retain Pareto improvements.
`native_fake_accuracy` is logged but never gates or ranks candidates.

For a passing finalist, run the cheap cross-domain confirmation:

```bash
python probe_v1_cross_domain_transfer.py \
  --gan-checkpoint runs/v1_ablation/L1a_sar_class_1/epoch_0074.pt \
  --geometry-validator-checkpoint server_results/sar_geometry_validator_xhh_v2/best.pt \
  --rgb-root /data/newdata/A25_T37_down_大图/A02/RGB \
  --sar-root /data/newdata/A25_T37_down_大图/A02/SOC_40classes_cut/train \
  --split-manifest runs/v1_ablation/split.json \
  --output runs/v1_ablation/L1a_transfer.json
```

Only finalists should receive the 2,000-step screen and then a confirmation
from the same parent with at least three seeds. Official test data remains
sealed until that point.

## Ordered One-Variable Matrix

| ID | Parent | Only change | Reason |
| --- | --- | --- | --- |
| C0 | V1 epoch 70 | none | Reproduced control |
| NC-S | C0 | `structure_weight=0` | Negative control: validates that the gates catch loss of shape |
| NC-A | C0 | `adversarial_weight=0` | Negative control for realism/texture |
| L1a | C0 | `sar_class_weight: 12 -> 1` | First likely teacher-shortcut reduction |
| L1b | C0 | `sar_class_weight: 12 -> 0` | Tests whether CE is needed at all |
| L2a/L2b | best L1 | `cluster_weight: 5 -> 1/0` | Separates class CE from prototype matching |
| S1 | best L2 | `structure_pixel_64_weight: 1 -> 0` | Tests strict 64x64 pixel matching alone |
| S2 | S1 | `structure_ssim_weight: 1 -> 0` | Tests positional global-SSIM alone |
| S3 | S2 | `structure_edge_weight: .5 -> 0` | Tests exact edge placement alone |
| P1 | best S | `physics_scatter_weight: 1 -> 0` | Tests exact bright-scatter map matching alone |
| A1 | P1 | `angle_loss_mode: first_order -> curvature` | Removes constant-angle incentive without removing all smoothness |
| D1 | A1 | `discriminator_condition: full -> target` | Removes source RGB angle and nonphysical condition inputs from D |
| D2 | D1 | `wrong_azimuth_discriminator_weight: 0 -> .25` | Adds one wrong-target angle negative to the same PatchGAN |
| R1 | best D | `cross_view_weight: 2 -> 0` | Low-priority RGB identity stability test |
| F1 | P1 control | `feature_match_weight: 5 -> 0` | Tests whether D feature moments are redundant with structure |
| T1 | P1 control | `statistics_weight: 5 -> 0` | Tests whether global SAR statistics are redundant with structure |

Do not remove all pixel or structure supervision as one experiment. V1's
translation-tolerant structure objective is its largest generator signal. The
correct question is whether the 64px term, SSIM, edge placement, and
physics-scatter map are individually helpful once coarse alignment remains.

## Current Paired Results

All values below are generated-to-real transfer deltas, candidate minus a
matched V1 control. They use the frozen real-SAR representation and a readout
trained only on generated features; native classifier accuracy is excluded.

| Change | Seeds | Identity top-1 | Depression top-1 | Azimuth MAE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `sar_class_weight: 12 -> 1` | 3 | +0.89 pp | +2.14 pp | -2.21 deg | Short-screen winner; see 2,000-step confirmation below |
| `sar_class_weight: 12 -> 0` | 3 | +0.31 pp | +2.19 pp | -1.89 deg | Reject: identity is not consistent and one azimuth seed regresses |
| `cluster_weight: 5 -> 4` | 3 | +0.21 pp | -0.94 pp | -0.29 deg | Reject: one depression seed loses 5.0 pp |
| `structure_pixel_64_weight: 1 -> 0` | 3 | -0.36 pp | +0.00 pp | +0.68 deg | Reject as default replacement; retain weak aligned 64px term |
| `structure_ssim_weight: 1 -> 0` | 3 | +0.00 pp | +1.41 pp | +2.65 deg | Reject: azimuth worsens in every seed |
| `structure_edge_weight: .5 -> 0` | 3 | -0.16 pp | -0.47 pp | +0.39 deg | Reject: no consistent primary improvement |
| `physics_scatter_weight: 1 -> 0` | 3 | -0.21 pp | -0.68 pp | -0.69 deg | Reject: azimuth gain does not offset identity/depression loss |
| `angle_smooth_weight: .2 -> 0` | 3 | -0.47 pp | +1.04 pp | +0.85 deg | Keep V1 first-order term provisionally; no robust gain from removal |
| `angle_loss_mode: first_order -> curvature` | 3 | -0.31 pp | -0.62 pp | +0.59 deg | Reject: depression regresses and one screen gate fails |
| `discriminator_condition: full -> target` | 3 | -0.52 pp | -0.73 pp | -1.24 deg | Reject: no consistent primary improvement |
| `wrong-azimuth negative: 0 -> .25` | 3 | -0.47 pp | -1.88 pp | -0.75 deg | Reject: depression regression; keep as diagnostic only |
| `cross_view_weight: 2 -> 0` | 3 | -0.52 pp | +0.42 pp | +0.00 deg | Reject: identity falls in every seed |
| `feature_match_weight: 5 -> 0` | 3 | -0.52 pp | -0.57 pp | +0.01 deg | Reject: identity falls in every seed |
| `statistics_weight: 5 -> 0` | 3 | -0.52 pp | +0.16 pp | +0.81 deg | Reject: one azimuth seed fails |

For `sar_class=1`, all three short screens passed the frozen geometry gates;
identity transfer improved in every seed. For `pixel64=0`, all geometry gates
passed, but identity transfer fell in every seed, one seed exceeded the
depression non-regression tolerance, and no primary transfer metric improved
consistently. This is evidence for keeping the weakly translation-aligned
pixel term in V1; it does not justify reintroducing a rigid unaligned pixel
loss.

The `sar_class=0` screen improved depression on average but failed the
non-regression policy because identity was worse in two seeds and azimuth
regressed by 2.9 degrees in one. The `cluster=4` screen likewise failed: its
mean is close to neutral only because one seed lost 5.0 percentage points of
depression transfer. Both weights therefore remain conservative (`1` and `5`)
while longer confirmations continue.

The 2,000-step `sar_class=1` confirmation now covers three matched seeds. Every
seed passes the screen and transfer non-regression policy; identity improves
in all three (+0.47 to +1.25 percentage points), depression improves in all
three (+3.13 to +4.06 points), and mean azimuth MAE improves by 2.31 degrees
(two seeds improve, one changes by +1.08 degrees). This is sufficient evidence
to promote `sar_class_weight=1` for the next full V1 run. The ablation trainer
keeps its default at 12 so that an explicit control remains reproducible; pass
`--sar-class-weight 1` in the recommended run.

The structure ablations confirm that V1's pixel supervision must be weakened
selectively, not removed wholesale. Removing the translation-tolerant 64px
term reduced identity transfer in all three seeds. Removing global SSIM made
azimuth MAE worse in all three seeds (mean +2.65 degrees), while removing the
edge term had no consistent primary benefit. The retained structure objective
therefore remains the aligned 64/32/16 terms plus edge and SSIM; this is not a
rigid unaligned pixel comparison.

The physics scattering-map experiment also does not justify deletion. Its
removal improved azimuth MAE by 0.69 degrees on average, but identity and
depression transfer both fell, and one seed exceeded the depression tolerance.
Keep the full physics prior until a later experiment replaces only its exact
pixel-like map term with a distributional statistic.

For angle regularisation, setting the weight to zero passed all short hard
gates and was therefore safe enough to audit, but it lowered identity by 0.47
percentage points and worsened azimuth MAE by 0.85 degrees on average versus
the matched P1 controls. The curvature replacement was worse: depression
transfer fell in two seeds and one short screen failed. The default remains
the original low-weight first-order term; neither angle variant is promoted
without a longer, same-parent confirmation.

The discriminator ablations do not support the proposed large fusion change.
Restricting the existing PatchGAN to target azimuth/depression improved
azimuth transfer by 1.24 degrees on average, but reduced identity and
depression transfer and had no consistent primary improvement. Adding a
wrong-azimuth real negative reduced the apparent gain to 0.75 degrees while
depression fell 1.88 percentage points. Relative to target-only D, the extra
negative made azimuth MAE worse by 0.49 degrees on average. Keep the V1
full-condition PatchGAN for the controlled baseline; do not merge the K+1
classifier/discriminator until a separately validated architecture is shown
to preserve these transfer metrics.

The low-contribution auxiliary terms are not automatically redundant. Removing
`cross_view` or `feature_match` passed the short hard gates, but identity
transfer fell in every seed (about 0.52 percentage points on average), with no
consistent primary improvement. Removing `statistics` failed one azimuth gate
and worsened azimuth MAE by 0.81 degrees on average. Keep all three for the
recommended run; any future simplification should test a smaller coefficient,
not jump directly from the V1 value to zero.

## Recommended Next Run

Use the preserved V1 architecture with only the validated class-weight change:

```bash
cd /data/newdata/A25_T37_down_大图/code
PYTHON_BIN=/home/star/anaconda3/envs/tessera/bin/python ./run_v1_recommended.sh
```

This continues V1 milestone 70 for 30 epochs (to epoch 100), sets
`sar_class_weight=1`, and leaves cluster, aligned structure, physics, angle,
feature-match, and full-condition PatchGAN at their V1 values. It is separate
from the ablation control entry point, whose default `sar_class_weight=12`
must remain unchanged for future paired experiments.

The completed epoch-100 run has no NaN/OOM. Its independent generated-to-real
probe reports identity `98.59%`, depression `85.16%`, azimuth MAE `23.07`
degrees, and frozen feature cosine `0.532`. The preview and metric figure are
published with the corresponding history and transfer JSON; this is an audit
of the recommended run, not a claim that the native classifier is an
independent quality oracle.

The paired artifacts are kept under `runs/v1_ablation/`:
`S2_ssim_1_to_0_three_seed_report.json`,
`S3_edge_05_to_0_three_seed_report.json`,
`P1_scatter_1_to_0_three_seed_report.json`,
`A0_no_angle_vs_P1_three_seed_report.json`, and
`A1_curvature_vs_P1_three_seed_report.json`,
`D1_target_vs_P1_three_seed_report.json`,
`D2_target_wrong_az_vs_P1_three_seed_report.json`, and
`D2_wrong_az_increment_three_seed_report.json`,
`R1_cross_view_2_to_0_three_seed_report.json`,
`F1_feature_match_5_to_0_three_seed_report.json`,
`T1_statistics_5_to_0_three_seed_report.json`, and
`L1_sar_class_12_to_1_2000_three_seed_report.json`. Their PNG counterparts are
intended for visual inspection, but the JSON values are the selection record.

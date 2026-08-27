# V1 Ablation Artifacts

`L1_sar_class_12_to_1_three_seed.png` and
`S1_pixel64_three_seed.png` show paired generated-to-real transfer deltas for
three matched random seeds. The first supports reducing V1 `sar_class_weight`
from 12 to 1; the second rejects removing the weak aligned 64px pixel term as
the default change. `L1_sar_class_12_to_0_three_seed.png` and
`L2_cluster_5_to_4_three_seed.png` document the corresponding rejected
classification and cluster reductions. `*_transfer.json` reports a
generated-to-real (TSTR)
probe: small readouts train only on generated frozen features and are evaluated
on held-out real SAR features. This prevents the native SAR classifier used in
V1's loss from serving as its own scorekeeper.

`S2_ssim_1_to_0_three_seed.png`, `S3_edge_05_to_0_three_seed.png`, and
`P1_scatter_1_to_0_three_seed.png` test structure and physics terms one at a
time; they reject deleting SSIM, edge, or the scattering-map prior. The angle
reports show that removing the low-weight first-order term is safe but not an
improvement, while curvature regresses depression transfer. `D1_target_vs_P1`
and `D2_target_wrong_az_vs_P1` show that the discriminator input/negative
experiments do not yet justify replacing V1's PatchGAN or merging it with the
classifier.

`R1_cross_view_2_to_0_three_seed.png`, `F1_feature_match_5_to_0_three_seed.png`,
and `T1_statistics_5_to_0_three_seed.png` cover the remaining small auxiliary
terms. Zeroing any of them has no consistent primary transfer gain; identity
falls in every seed for the first two, and the statistics removal also fails an
azimuth gate. The validated long class-loss confirmation is recorded in
`L1_sar_class_12_to_1_2000_three_seed.png`.

The milestone audit files show why native fake-class accuracy is not a valid
selection metric: it is already near 100 percent at V1 epoch 10 while frozen
geometry and generated-to-real azimuth transfer are still poor. The protocol
and exact commands are in `code/V1_ABLATION_PROTOCOL.md`.

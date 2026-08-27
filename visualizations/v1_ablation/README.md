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

The milestone audit files show why native fake-class accuracy is not a valid
selection metric: it is already near 100 percent at V1 epoch 10 while frozen
geometry and generated-to-real azimuth transfer are still poor. The protocol
and exact commands are in `code/V1_ABLATION_PROTOCOL.md`.

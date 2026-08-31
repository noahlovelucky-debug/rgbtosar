# HiFC 120 Epoch Reproduction

This directory freezes the training recipe for the all-condition HiFC checkpoint
that reached 48.32% X/HH real-SAR TSTR Top-1. It is a reproduction record for
the checkpoint, not the later `all_off` native-gradient ablation.

## Immutable identities

| Item | Value |
| --- | --- |
| Git commit | `0208dc002148c46e1d754cae94b07c44a52d616d` |
| Architecture | `hifc_unpaired_conditioned_v1` |
| Checkpoint | `checkpoints/hifc_unpaired_all_conditions_epoch120_tstr48.pt` |
| Checkpoint SHA256 | `0578a68dafded01c2c348492a3520411f05e9b90c5556eb2c3026dd30c4e7b4c` |
| Native teacher SHA256 | `e7e6e433fbe480b25072669586700f6f7fe2d599e7bb681d34778f0e1cdd2707` |
| Data split SHA256 | `95bd7d7b41b9ba2791354255eae6c995a2a3f6b898f49d7c7ea886f838f6a964` |

The bucket release is `volcantos:dataspacety/rgbtosar/hifc_20260830/`. It
contains the RGB data, SOC split, native teacher and this checkpoint. The
release supplement under `releases/hifc_epoch120_tstr48/` contains this
directory, the exact split manifest and a source archive.

## Environment

```bash
conda create -n rgbtosar python=3.12 -y
conda activate rgbtosar
pip install -r code/requirements.txt
```

The recorded run used PyTorch with CUDA and 8 local GPUs. The training recipe
uses 8 processes with a per-rank batch size of 8, for a global batch size of
64.

## Required layout

```text
workspace/
  code/                              # this Git repository's code directory
  data/RGB/                           # RGB data, 40 class directories
  data/SOC_40classes_cut/train/       # SOC SAR training split
  checkpoints/native_classifier_best.pt
```

Download `split_manifest__all_all_all.json` from the release supplement and
set `SPLIT_MANIFEST` to its local path. The script copies it into the output
directory before training, preserving the recorded train/validation split.

## Exact 120-epoch command

```bash
GPUS=8 DATA_ROOT="$PWD/data" OUTPUT="$PWD/runs/hifc_unpaired_all_conditions_ddp" \\
SPLIT_MANIFEST="$PWD/releases/hifc_epoch120_tstr48/split_manifest__all_all_all.json" \\
bash repro/hifc_epoch120_tstr48/train_ddp_full.sh
```

The original checkpoint predates the `--native-gradient-mode` CLI field. Its
effective behavior is explicitly reproduced here with
`--native-gradient-mode full`: the frozen native teacher's embedding SFM term
and its geometry auxiliary heads both return gradients to the RGB encoder and
generator. The native teacher's 40-class hard CE is not a generator loss.

## Resume and inference

To resume, append `--resume "$OUTPUT/latest.pt"` to the torchrun command. For
generation, load the `ema_identity_encoder` and `ema_generator` states from
the checkpoint. The discriminator and native teacher are needed for training
or evaluation only, not for ordinary inference.

## Integrity check

```bash
sha256sum checkpoints/hifc_unpaired_all_conditions_epoch120_tstr48.pt
sha256sum checkpoints/native_classifier_best.pt
```

Expected values are recorded in `SHA256SUMS`.

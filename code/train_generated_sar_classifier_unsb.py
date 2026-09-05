"""Train an independent SAR classifier on frozen UNSB-SAR samples.

The generator only receives RGB/mask plus acquisition metadata.  The
classifier sees generated X/HH images during training and is evaluated once
on held-out real X/HH images, matching the repository's TSTR protocol.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bbox_data import metadata_vector
from hifc_unpaired_sar_gan import condition_from_batch
from joint_data import JointROIDataset
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES
from train_generated_sar_classifier_64 import (
    DEPRESSION_TO_ID, BAND_TO_ID, POLARIZATION_TO_ID, RealConditionTestDataset,
    augment, evaluate,
)
from unsb_sar_bridge import (
    UNSB_SAR_UNPAIRED_ARCHITECTURE, SilhouetteBridge, bridge_sample,
)


class UNSBGeneratedDataset(Dataset):
    """Condition/RGB records; real SAR pixels are never read."""

    def __init__(self, rgb_root: Path, sar_root: Path, band: str = "X",
                 polarization: str = "HH", depression: str = "all") -> None:
        self.base = JointROIDataset(
            rgb_root, sar_root, rgb_size=128, epoch_size=0,
            band=band, polarization=polarization, depression=depression,
            augment_rgb=False, source_view_mode="random", return_rgb_mask=True,
        )

    def __len__(self) -> int:
        return len(self.base.records)

    def __getitem__(self, index: int):
        _, _, class_name, bbox, meta, _ = self.base.records[index]
        angles = self.base.class_rgb_angles[class_name]
        source_angle = random.choice(angles)
        alternate = random.choice([angle for angle in angles if angle != source_angle] or angles)
        source_path = self.base.rgb_paths[class_name, source_angle]
        alternate_path = self.base.rgb_paths[class_name, alternate]
        targets = torch.tensor((
            self.base.class_to_id[class_name],
            BAND_TO_ID[str(meta["band"]).upper()],
            POLARIZATION_TO_ID[str(meta["pol"]).upper()],
            DEPRESSION_TO_ID[int(meta["depression"])],
            ((int(meta["azimuth"]) + 15) % 360) // 30,
        ), dtype=torch.long)
        return (
            self.base._rgb(source_path), self.base._rgb_mask(source_path),
            self.base._rgb(alternate_path), self.base._rgb_mask(alternate_path),
            metadata_vector(meta, bbox), torch.tensor(source_angle, dtype=torch.float32), targets,
        )


def angle_features(angle: torch.Tensor) -> torch.Tensor:
    radians = angle * (math.pi / 180.0)
    return torch.stack((radians.sin(), radians.cos()), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="UNSB-SAR TSTR classifier")
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--real-test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--sample-steps", type=int, default=5)
    parser.add_argument("--sample-temperature", type=float, default=.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--aux-weight", type=float, default=.12)
    parser.add_argument("--seed", type=int, default=415)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.sample_steps < 1:
        raise ValueError("epochs, batch-size, and sample-steps must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") != UNSB_SAR_UNPAIRED_ARCHITECTURE:
        raise RuntimeError("checkpoint is not an UNSB-SAR G/D/E checkpoint")
    saved_args = state.get("args", {})
    generator = SilhouetteBridge(
        base=int(saved_args.get("base", 64)),
        token_dim=int(saved_args.get("token_dim", 256)),
        control_base=int(saved_args.get("control_base", 32)),
    ).to(device).eval()
    generator.load_state_dict(state["ema_generator"])
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    generated = UNSBGeneratedDataset(args.rgb_root, args.condition_root)
    real_test = RealConditionTestDataset(args.real_test_root, band="X", polarization="HH")
    loader = DataLoader(
        generated, args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        real_test, args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    classifier = SARClassifier64(len(SOC40_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup = max(1, min(3, args.epochs // 8))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: ((epoch + 1) / warmup if epoch < warmup else
                                  .5 * (1 + np.cos(np.pi * (epoch - warmup + 1) /
                                                   max(1, args.epochs - warmup + 1)))))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and not args.no_amp)
    class_loss, aux_loss = nn.CrossEntropyLoss(label_smoothing=.03), nn.CrossEntropyLoss()
    history_path = args.output / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(("epoch", "train_loss", "train_top1", "real_test_top1",
                                     "real_test_top5", "real_test_azimuth_top1",
                                     "real_test_azimuth_mae"))
    for epoch in range(1, args.epochs + 1):
        classifier.train(); total = correct = 0; loss_sum = 0.0
        for batch_index, batch in enumerate(tqdm(loader, desc=f"UNSB TSTR {epoch}/{args.epochs}")):
            rgb, mask, rgb_alt, mask_alt, meta, source_angle, targets = batch
            rgb = rgb.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
            rgb_alt = rgb_alt.to(device, non_blocking=True); mask_alt = mask_alt.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True); source_angle = source_angle.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.inference_mode():
                depression = (targets[:, 3] + 1).mul(15)
                acquisition = condition_from_batch(meta, depression)
                synthetic = bridge_sample(
                    generator, rgb, mask, acquisition,
                    steps=args.sample_steps, temperature=args.sample_temperature,
                    source_angle=angle_features(source_angle),
                    rgb_alt=rgb_alt, mask_alt=mask_alt,
                )
                synthetic = augment((synthetic + 1.0) * .5)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits, features = classifier(synthetic, return_features=True)
                auxiliary = classifier.auxiliary_logits(features)
                loss = class_loss(logits, targets[:, 0])
                loss += args.aux_weight * sum(
                    aux_loss(logit, target) for logit, target in
                    zip(auxiliary, targets[:, 1:].unbind(1))) / len(auxiliary)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            prediction = logits.argmax(1)
            loss_sum += float(loss.detach()) * len(targets)
            correct += int((prediction == targets[:, 0]).sum())
            total += len(targets)
            if args.steps_per_epoch and batch_index + 1 >= args.steps_per_epoch:
                break
        metrics, by_depression = evaluate(classifier, test_loader, device)
        row = (epoch, loss_sum / max(1, total), correct / max(1, total),
               metrics["top1"], metrics["top5"], metrics["azimuth_top1"],
               metrics["azimuth_circular_mae"])
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        torch.save({"model": classifier.state_dict(), "epoch": epoch,
                    "metrics": metrics, "by_depression": by_depression,
                    "gan_checkpoint": str(args.gan_checkpoint),
                    "protocol": "UNSB synthetic X/HH train -> real X/HH test",
                    "args": vars(args)}, args.output / "latest.pt")
        print(dict(zip(("epoch", "train_loss", "train_top1", "real_test_top1",
                        "real_test_top5", "real_test_azimuth_top1", "real_test_azimuth_mae"), row)),
              flush=True)
    torch.save({"model": classifier.state_dict(), "epoch": args.epochs,
                "metrics": metrics, "by_depression": by_depression,
                "gan_checkpoint": str(args.gan_checkpoint),
                "protocol": "UNSB synthetic X/HH train -> real X/HH test",
                "args": vars(args)}, args.output / "best.pt")


if __name__ == "__main__":
    main()

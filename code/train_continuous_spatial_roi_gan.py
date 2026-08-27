"""Continuous-azimuth, four-depression RGB-spatial-feature to SAR ROI GAN.

The input is one RGB view of a vehicle and its source azimuth.  The target
condition is a continuous SAR azimuth (sin/cos) and one of 15/30/45/60 degree
depression angles, fixed to X-band HH in this experiment.  Real SAR ROIs are
weakly paired by class and target observation condition; they are never assumed
pixel-registered to the RGB cut-out.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from joint_data import JointROIDataset
from joint_models import (RGBIdentityEncoder, SpatialROIGenerator, angle_curvature_loss,
                          distributional_structure_loss, initialise, sar_physics_prior_loss)
from sar_classifier_64 import SARClassDiscriminator64
from saratrx import SOC40_CLASSES


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous azimuth and depression spatial RGB-to-SAR GAN")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--epoch-size", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rgb-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--identity-lr", type=float, default=5e-5)
    parser.add_argument("--discriminator-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--discriminator-warmup-epochs", type=int, default=1)
    parser.add_argument("--rgb-id-weight", type=float, default=2.0)
    parser.add_argument("--class-adversarial-weight", type=float, default=1.0)
    parser.add_argument("--structure-weight", type=float, default=6.0)
    parser.add_argument("--physics-weight", type=float, default=3.0)
    parser.add_argument("--angle-smooth-weight", type=float, default=0.5)
    parser.add_argument("--angle-every", type=int, default=4)
    parser.add_argument("--fake-class-weight", type=float, default=0.5)
    parser.add_argument("--wrong-condition-weight", type=float, default=0.25)
    parser.add_argument("--wrong-condition-margin", type=float, default=0.2)
    parser.add_argument("--speckle-warmup-epochs", type=int, default=8)
    parser.add_argument("--speckle-ramp-epochs", type=int, default=5)
    parser.add_argument("--source-view-mode", choices=("nearest", "random", "mixed"), default="mixed")
    parser.add_argument("--initialise-checkpoint", type=Path,
                        help="warm-start encoder and generator from a continuous_spatial_v1 checkpoint")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def target_condition(meta: torch.Tensor, source_angle: torch.Tensor) -> torch.Tensor:
    # ROI extent comes from the real-SAR annotation, so feeding it to G would
    # leak target-side information and encourage a vehicle-template shortcut.
    # The spatial RGB pyramid must supply body scale and silhouette instead.
    meta = meta.clone()
    meta[:, -2:] = 0.0
    source_radians = source_angle.float() * (math.pi / 180.0)
    return torch.cat((meta, source_radians.sin()[:, None], source_radians.cos()[:, None]), dim=1)


def rotate_target_azimuth(condition: torch.Tensor, degrees: float = 5.0) -> torch.Tensor:
    angle = torch.atan2(condition[:, 0], condition[:, 1]) + math.radians(degrees)
    result = condition.clone()
    result[:, 0], result[:, 1] = angle.sin(), angle.cos()
    return result


def set_grad(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def discriminator_condition(condition: torch.Tensor) -> torch.Tensor:
    """Only target SAR azimuth and depression condition the shared SAR judge."""
    return condition[:, :3]


def set_discriminator_grad(discriminator: SARClassDiscriminator64, enabled: bool, warmup: bool) -> None:
    for name, parameter in discriminator.named_parameters():
        is_new_head = name.startswith(("condition", "fake_classifier", "classifier"))
        parameter.requires_grad_(enabled and (is_new_head or not warmup))


def save_preview(rgb: torch.Tensor, real: torch.Tensor, fake: torch.Tensor, path: Path) -> None:
    rows = []
    for index in range(min(8, len(fake))):
        rgb_panel = F.interpolate(rgb[index:index + 1], (64, 64), mode="bilinear", align_corners=False)[0]
        rgb_panel = ((rgb_panel.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        panels = [rgb_panel]
        for panel in (real[index, 0], fake[index, 0]):
            panel = ((panel.detach().cpu().clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
            panels.append(np.repeat(panel[..., None], 3, axis=2))
        rows.append(np.concatenate(panels, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0), "RGB").save(path)


def main() -> None:
    args = arguments()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    train_data = JointROIDataset(args.rgb_root, args.sar_root, rgb_size=args.rgb_size,
                                 epoch_size=args.epoch_size, band="X", polarization="HH", depression="all",
                                 augment_rgb=True, source_view_mode=args.source_view_mode)
    loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    saved_classifier = torch.load(args.native_classifier_checkpoint, map_location=device, weights_only=False)
    if saved_classifier.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("native classifier class order does not match RGB/SAR data")
    encoder = RGBIdentityEncoder(len(SOC40_CLASSES)).to(device)
    generator = SpatialROIGenerator(meta_dim=12).to(device)
    discriminator = SARClassDiscriminator64(len(SOC40_CLASSES)).to(device)
    encoder.apply(initialise); generator.apply(initialise)
    missing, unexpected = discriminator.load_state_dict(saved_classifier["model"], strict=False)
    expected_missing = {name for name in missing if name.startswith(("condition", "fake_classifier"))}
    if unexpected or len(expected_missing) != len(missing):
        raise RuntimeError(f"native classifier transfer failed: missing={missing}, unexpected={unexpected}")
    if args.initialise_checkpoint:
        initial = torch.load(args.initialise_checkpoint, map_location=device, weights_only=False)
        if initial.get("architecture") != "continuous_spatial_v1":
            raise RuntimeError("initialise checkpoint is not continuous_spatial_v1")
        encoder.load_state_dict(initial["identity_encoder"])
        generator.load_state_dict(initial["generator"])
        print(f"warm-started encoder/generator from {args.initialise_checkpoint} epoch {initial.get('epoch')}; "
              "old PatchGAN ignored", flush=True)
    generator_optimizer = torch.optim.Adam((
        {"params": encoder.parameters(), "lr": args.identity_lr},
        {"params": generator.parameters(), "lr": args.lr},
    ), betas=(0.0, .99), foreach=False)
    head_prefixes = ("condition", "fake_classifier", "classifier")
    head_parameters = [parameter for name, parameter in discriminator.named_parameters()
                       if name.startswith(head_prefixes)]
    backbone_parameters = [parameter for name, parameter in discriminator.named_parameters()
                           if not name.startswith(head_prefixes)]
    discriminator_optimizer = torch.optim.Adam((
        {"params": backbone_parameters, "lr": args.discriminator_backbone_lr},
        {"params": head_parameters, "lr": args.discriminator_lr},
    ), betas=(0.0, .99), foreach=False)
    generator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    discriminator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    rgb_cross_entropy = nn.CrossEntropyLoss(label_smoothing=.02)
    discriminator_cross_entropy = nn.CrossEntropyLoss()
    history = args.output / "history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow((
            "epoch", "loss_total", "loss_rgb_identity", "loss_class_adversarial", "loss_structure",
            "loss_physics", "loss_angle", "loss_discriminator", "loss_discriminator_real_class",
            "loss_discriminator_fake_class", "loss_discriminator_wrong_condition", "rgb_identity_accuracy",
            "fake_target_class_accuracy", "real_class_accuracy", "fake_rejection_accuracy",
            "angle_near_delta", "angle_far_delta", "angle_far_to_near", "speckle_strength"))
    best_quality = float("inf")
    print("dataset:", train_data.summary(),
          "condition: X/HH, continuous azimuth, depressions 15/30/45/60; fused K+1 classifier-discriminator",
          flush=True)
    for epoch in range(1, args.epochs + 1):
        encoder.train(); generator.train(); discriminator.train()
        totals = torch.zeros(15, dtype=torch.float64)
        angle_near_total = angle_far_total = 0.0
        angle_measurements = 0
        speckle = generator.speckle_strength * min(1., max(0., (epoch - args.speckle_warmup_epochs) /
                                                              max(1, args.speckle_ramp_epochs)))
        for batch_index, batch in enumerate(tqdm(loader, desc=f"continuous spatial GAN {epoch}/{args.epochs}")):
            rgb, rgb_alt = batch["rgb"].to(device), batch["rgb_alt"].to(device)
            real, meta, labels = batch["roi"].to(device), batch["meta"].to(device), batch["class_id"].to(device)
            source_angle = batch["rgb_angle"].to(device)
            condition = target_condition(meta, source_angle)
            sar_condition = discriminator_condition(condition)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                identity, _, pyramid = encoder(rgb, return_pyramid=True)
                fake_clean = generator(identity, condition, pyramid, apply_speckle=False)
                fake = generator.apply_speckle(fake_clean, speckle)

            warmup = epoch <= args.discriminator_warmup_epochs
            set_discriminator_grad(discriminator, True, warmup)
            discriminator.train()
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                real_logits, _ = discriminator((real + 1) * .5, sar_condition)
                fake_logits_for_d, _ = discriminator((fake.detach() + 1) * .5, sar_condition)
                wrong_condition = rotate_target_azimuth(sar_condition, random.choice((-90., -60., -30., 30., 60., 90.)))
                wrong_logits, _ = discriminator((real + 1) * .5, wrong_condition)
                fake_labels = torch.full_like(labels, len(SOC40_CLASSES))
                discriminator_real_loss = discriminator_cross_entropy(real_logits, labels)
                discriminator_fake_loss = discriminator_cross_entropy(fake_logits_for_d, fake_labels)
                target_index = torch.arange(len(labels), device=device)
                discriminator_condition_loss = F.softplus(
                    args.wrong_condition_margin + wrong_logits[target_index, labels] - real_logits[target_index, labels]
                ).mean()
                discriminator_loss = (discriminator_real_loss + args.fake_class_weight * discriminator_fake_loss +
                                      args.wrong_condition_weight * discriminator_condition_loss)
            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.unscale_(discriminator_optimizer)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            discriminator_scaler.step(discriminator_optimizer); discriminator_scaler.update()

            set_grad(discriminator, False); generator_optimizer.zero_grad(set_to_none=True)
            discriminator.eval()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                alternate_identity, _ = encoder(rgb_alt)
                rgb_logits = encoder.class_logits(identity, labels)
                alternate_logits = encoder.class_logits(alternate_identity, labels)
                fake_logits, _, fake_disc_pyramid = discriminator((fake + 1) * .5, sar_condition, return_pyramid=True)
                with torch.no_grad():
                    _, _, real_disc_pyramid = discriminator((real + 1) * .5, sar_condition, return_pyramid=True)
                rgb_loss = .5 * (rgb_cross_entropy(rgb_logits, labels) + rgb_cross_entropy(alternate_logits, labels))
                class_adversarial_loss = discriminator_cross_entropy(fake_logits, labels)
                structure_loss, _, _, _ = distributional_structure_loss(
                    fake_clean, fake, real, fake_disc_pyramid, real_disc_pyramid)
                physics_loss = sar_physics_prior_loss(fake_clean, real)
                angle_loss = fake.new_zeros(())
                if batch_index % args.angle_every == 0:
                    left = generator(identity, rotate_target_azimuth(condition, -5.), pyramid, apply_speckle=False)
                    right = generator(identity, rotate_target_azimuth(condition, 5.), pyramid, apply_speckle=False)
                    far = generator(identity, rotate_target_azimuth(condition, 30.), pyramid, apply_speckle=False)
                    angle_raw = angle_curvature_loss(left, fake_clean, right)
                    angle_loss = angle_raw * args.angle_every
                    near_delta = .5 * (F.l1_loss(F.avg_pool2d(left, 4), F.avg_pool2d(fake_clean, 4)) +
                                        F.l1_loss(F.avg_pool2d(right, 4), F.avg_pool2d(fake_clean, 4)))
                    far_delta = F.l1_loss(F.avg_pool2d(far, 4), F.avg_pool2d(fake_clean, 4))
                    angle_near_total += near_delta.detach().item()
                    angle_far_total += far_delta.detach().item()
                    angle_measurements += 1
                total_loss = (args.rgb_id_weight * rgb_loss + args.class_adversarial_weight * class_adversarial_loss +
                              args.structure_weight * structure_loss + args.physics_weight * physics_loss +
                              args.angle_smooth_weight * angle_loss)
            generator_scaler.scale(total_loss).backward()
            generator_scaler.unscale_(generator_optimizer)
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(generator.parameters()), 5.0)
            generator_scaler.step(generator_optimizer); generator_scaler.update()
            values = (
                total_loss, rgb_loss, class_adversarial_loss, structure_loss, physics_loss, angle_loss,
                discriminator_loss, discriminator_real_loss, discriminator_fake_loss, discriminator_condition_loss,
                .5 * ((rgb_logits.argmax(1) == labels).float().mean() +
                      (alternate_logits.argmax(1) == labels).float().mean()),
                (fake_logits.argmax(1) == labels).float().mean(),
                (real_logits.argmax(1) == labels).float().mean(),
                (fake_logits_for_d.argmax(1) == fake_labels).float().mean(), fake.new_tensor(speckle))
            totals += torch.tensor([value.detach().item() for value in values], dtype=torch.float64)
        averages = (totals / len(loader)).tolist()
        angle_near = angle_near_total / max(1, angle_measurements)
        angle_far = angle_far_total / max(1, angle_measurements)
        angle_ratio = angle_far / max(angle_near, 1e-8)
        row = (epoch, *averages[:-1], angle_near, angle_far, angle_ratio, averages[-1])
        with history.open("a", newline="", encoding="utf-8") as handle: csv.writer(handle).writerow(row)
        quality = averages[3] + .25 * averages[4] + .1 * averages[2] + .1 * averages[5]
        checkpoint = {"epoch": epoch, "identity_encoder": encoder.state_dict(), "generator": generator.state_dict(),
                      "classifier_discriminator": discriminator.state_dict(), "classes": list(SOC40_CLASSES),
                      "args": {**vars(args), "rgb_root": str(args.rgb_root), "sar_root": str(args.sar_root)},
                      "quality": quality, "speckle_strength": speckle,
                      "native_classifier_source": str(args.native_classifier_checkpoint),
                      "architecture": "continuous_spatial_fused_v2"}
        torch.save(checkpoint, args.output / "latest.pt")
        if speckle >= generator.speckle_strength - 1e-6 and averages[12] >= .85 and quality < best_quality:
            best_quality = quality; torch.save(checkpoint, args.output / "best.pt")
        if epoch == 1 or epoch % 5 == 0:
            torch.save(checkpoint, args.output / f"milestone_{epoch:04d}.pt")
            save_preview(rgb, real, fake, args.output / f"preview_{epoch:04d}.png")
        print(dict(zip(("epoch", "total", "rgb", "class_adv", "structure", "physics", "angle", "disc",
                        "disc_real", "disc_fake", "disc_wrong_condition", "rgb_acc", "fake_target_acc",
                        "real_class_acc", "fake_rejection", "angle_near", "angle_far", "angle_ratio", "speckle"), row)),
              flush=True)


if __name__ == "__main__":
    main()

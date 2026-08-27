"""Learn a sampleable real-SAR style latent on top of the spatial RGB GAN."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from joint_data import JointROIDataset
from joint_models import (ContinuousROIDiscriminator, RGBIdentityEncoder, SARStyleEncoder,
                          StyleSpatialROIGenerator, _align_translation, initialise,
                          multiscale_structure_loss, sar_perceptual_pyramid_loss,
                          sar_physics_prior_loss, sar_statistics_loss)
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES
from train_continuous_spatial_roi_gan import (DEPRESSION_TO_ID, conditional_prototypes,
                                              save_preview, set_grad, target_condition)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Variational real-SAR style spatial GAN")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--base-gan-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--epoch-size", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prototype-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--style-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=8e-5)
    parser.add_argument("--class-weight", type=float, default=4.)
    parser.add_argument("--cluster-weight", type=float, default=3.)
    parser.add_argument("--structure-weight", type=float, default=12.)
    parser.add_argument("--statistics-weight", type=float, default=6.)
    parser.add_argument("--physics-weight", type=float, default=5.)
    parser.add_argument("--perceptual-weight", type=float, default=8.)
    parser.add_argument("--adversarial-weight", type=float, default=5.)
    parser.add_argument("--feature-weight", type=float, default=8.)
    parser.add_argument("--kl-weight", type=float, default=.02)
    parser.add_argument("--style-reconstruction-weight", type=float, default=3.)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); use_amp = device.type == "cuda" and not args.no_amp
    train_data = JointROIDataset(args.rgb_root, args.sar_root, epoch_size=args.epoch_size,
                                 band="X", polarization="HH", depression="all",
                                 augment_rgb=True, source_view_mode="mixed")
    prototype_data = JointROIDataset(args.rgb_root, args.sar_root, epoch_size=0,
                                     band="X", polarization="HH", depression="all",
                                     augment_rgb=False, preload_rgb=False)
    loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)

    judge_state = torch.load(args.native_classifier_checkpoint, map_location=device, weights_only=False)
    judge = SARClassifier64(len(SOC40_CLASSES)).to(device)
    judge.load_state_dict(judge_state["model"]); judge.eval()
    for parameter in judge.parameters(): parameter.requires_grad_(False)
    prototypes = conditional_prototypes(judge, prototype_data, args, device)

    base = torch.load(args.base_gan_checkpoint, map_location=device, weights_only=False)
    if base.get("architecture") != "continuous_spatial_v1":
        raise RuntimeError("base checkpoint must be continuous_spatial_v1")
    encoder = RGBIdentityEncoder(len(SOC40_CLASSES)).to(device)
    encoder.load_state_dict(base["identity_encoder"]); encoder.eval()
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    generator = StyleSpatialROIGenerator(meta_dim=12, style_dim=args.style_dim).to(device)
    generator.apply(initialise)
    missing, unexpected = generator.load_state_dict(base["generator"], strict=False)
    if unexpected or any(not key.startswith(("identity_style.", "style_affine.")) for key in missing):
        raise RuntimeError(f"incompatible base generator: missing={missing}, unexpected={unexpected}")
    generator.reset_style_injection()
    style_encoder = SARStyleEncoder(args.style_dim).to(device); style_encoder.apply(initialise)
    discriminator = ContinuousROIDiscriminator(meta_dim=12).to(device)
    discriminator.load_state_dict(base["discriminator"])
    generator_optimizer = torch.optim.Adam((*generator.parameters(), *style_encoder.parameters()),
                                            lr=args.lr, betas=(.5, .999), foreach=False)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr,
                                                betas=(.5, .999), foreach=False)
    generator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    discriminator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=.02)
    header = ("epoch", "total", "class", "cluster", "structure", "statistics", "physics",
              "perceptual", "adversarial", "feature", "kl", "style_reconstruction",
              "discriminator", "fake_accuracy", "cluster_cosine")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(header)
    best_quality = float("inf")
    for epoch in range(1, args.epochs + 1):
        generator.train(); style_encoder.train(); discriminator.train()
        totals = torch.zeros(len(header), dtype=torch.float64)
        for batch in tqdm(loader, desc=f"style spatial GAN {epoch}/{args.epochs}"):
            rgb = batch["rgb"].to(device); real = batch["roi"].to(device)
            meta, labels = batch["meta"].to(device), batch["class_id"].to(device)
            source_angle = batch["rgb_angle"].to(device)
            depression_id = torch.tensor([DEPRESSION_TO_ID[int(value)]
                                          for value in batch["depression"].tolist()], device=device)
            condition = target_condition(meta, source_angle)
            with torch.no_grad():
                identity, _, pyramid = encoder(rgb, return_pyramid=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                posterior_style, mu, logvar = style_encoder(real)
                prior_style = torch.randn_like(posterior_style)
                paired_clean = generator(identity, condition, pyramid, posterior_style, apply_speckle=False)
                prior_clean = generator(identity, condition, pyramid, prior_style, apply_speckle=False)
                fake = generator.apply_speckle(prior_clean)

            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                real_score, _ = discriminator(real, condition)
                fake_score, _ = discriminator(fake.detach(), condition)
                discriminator_loss = F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.step(discriminator_optimizer); discriminator_scaler.update()

            set_grad(discriminator, False); generator_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fake_score, fake_disc_features = discriminator(fake, condition)
                with torch.no_grad():
                    _, real_disc_features = discriminator(real, condition)
                logits, features, fake_pyramid = judge((fake + 1) * .5, return_pyramid=True)
                class_loss = cross_entropy(logits, labels)
                cosine = (F.normalize(features, dim=1) * prototypes[labels, depression_id]).sum(1).mean()
                cluster_loss = 1 - cosine
                structure_loss = multiscale_structure_loss(paired_clean, real)
                statistics_loss = sar_statistics_loss(fake, real)
                physics_loss = sar_physics_prior_loss(paired_clean, real)
                aligned_real = _align_translation(paired_clean, real)
                with torch.no_grad():
                    _, _, real_pyramid = judge((aligned_real + 1) * .5, return_pyramid=True)
                perceptual_loss = sar_perceptual_pyramid_loss(fake_pyramid, real_pyramid)
                adversarial_loss = -fake_score.mean()
                feature_loss = (F.l1_loss(fake_disc_features.mean((2, 3)), real_disc_features.mean((2, 3)))
                                + F.l1_loss(fake_disc_features.std((2, 3)), real_disc_features.std((2, 3))))
                kl_loss = -.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
                _, recovered_style, _ = style_encoder(prior_clean, sample=False)
                style_loss = F.l1_loss(recovered_style, prior_style)
                total_loss = (args.class_weight * class_loss + args.cluster_weight * cluster_loss
                              + args.structure_weight * structure_loss
                              + args.statistics_weight * statistics_loss
                              + args.physics_weight * physics_loss
                              + args.perceptual_weight * perceptual_loss
                              + args.adversarial_weight * adversarial_loss
                              + args.feature_weight * feature_loss + args.kl_weight * kl_loss
                              + args.style_reconstruction_weight * style_loss)
            generator_scaler.scale(total_loss).backward()
            generator_scaler.unscale_(generator_optimizer)
            torch.nn.utils.clip_grad_norm_((*generator.parameters(), *style_encoder.parameters()), 5.)
            generator_scaler.step(generator_optimizer); generator_scaler.update()
            set_grad(discriminator, True)
            values = (total_loss, class_loss, cluster_loss, structure_loss, statistics_loss,
                      physics_loss, perceptual_loss, adversarial_loss, feature_loss, kl_loss,
                      style_loss, discriminator_loss, (logits.argmax(1) == labels).float().mean(), cosine)
            totals[1:] += torch.tensor([value.detach().item() for value in values], dtype=torch.float64)
            totals[0] += 1
        averages = (totals[1:] / totals[0]).tolist(); row = (epoch, *averages)
        with (args.output / "history.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        quality = averages[2] + .5 * averages[3] + .25 * averages[4] + .5 * averages[5] + .1 * averages[8]
        checkpoint = {"epoch": epoch, "identity_encoder": encoder.state_dict(),
                      "generator": generator.state_dict(), "style_encoder": style_encoder.state_dict(),
                      "discriminator": discriminator.state_dict(), "classes": list(SOC40_CLASSES),
                      "architecture": "continuous_spatial_style_v2", "style_dim": args.style_dim,
                      "quality": quality, "args": {key: str(value) if isinstance(value, Path) else value
                                                  for key, value in vars(args).items()}}
        torch.save(checkpoint, args.output / "latest.pt")
        if averages[12] >= .85 and quality < best_quality:
            best_quality = quality; torch.save(checkpoint, args.output / "best.pt")
        if epoch == 1 or epoch % 5 == 0:
            torch.save(checkpoint, args.output / f"milestone_{epoch:04d}.pt")
            save_preview(rgb, real, fake, args.output / f"preview_{epoch:04d}.png")
        print(dict(zip(header, row)), flush=True)
    (args.output / "config.json").write_text(json.dumps(
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

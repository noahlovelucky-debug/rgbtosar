"""Train v5 RGB-driven hybrid GAN on real X/HH 64px SAR ROIs."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from joint_models import (RGBIdentityEncoder, _align_translation,
                          sar_physics_prior_loss, sar_statistics_loss)
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES
from v3_latent_sar import V3PairDataset, build_manifest
from v5_hybrid_sar_gan import (MultiDomainDiscriminator, RGBReflectivityGenerator,
                               highpass_view, sar_observation, spectrum_view)


class V5PairDataset(V3PairDataset):
    """v3 fixed split plus a second same-vehicle RGB view."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        class_id = int(sample["class_id"])
        name = SOC40_CLASSES[class_id]
        alternatives = self.class_angles[name]
        angle = random.choice(alternatives)
        rgb_alt = self.rgb_cache[self.rgb_paths[name, angle]].float().div(127.5).sub(1.)
        if self.augment_rgb:
            rgb_alt = (rgb_alt * random.uniform(.92, 1.08) + random.uniform(-.03, .03)
                       + torch.randn_like(rgb_alt) * random.uniform(0, .01)).clamp(-1, 1)
        sample["rgb_alt"] = rgb_alt
        return sample


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v5 hybrid RGB-to-SAR GAN")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--generator-lr", type=float, default=1.5e-4)
    parser.add_argument("--discriminator-lr", type=float, default=1.5e-4)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--r1-weight", type=float, default=1.0)
    parser.add_argument("--r1-every", type=int, default=16)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-validation-batches", type=int, default=0)
    return parser.parse_args()


def loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle)


def set_grad(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def update_ema(ema: nn.Module, source: nn.Module, decay: float) -> None:
    for target, current in zip(ema.parameters(), source.parameters()):
        target.lerp_(current, 1 - decay)
    for target, current in zip(ema.buffers(), source.buffers()):
        target.copy_(current)


def azimuth_bin(azimuth: torch.Tensor) -> torch.Tensor:
    return ((azimuth + 15) % 360) // 30


def prepare_teacher_prototypes(judge: SARClassifier64, dataset: V5PairDataset,
                               device: torch.device, workers: int) -> torch.Tensor:
    sums = torch.zeros(40, 4, 12, judge.feature_dim, device=device)
    counts = torch.zeros(40, 4, 12, device=device)
    with torch.inference_mode():
        for batch in tqdm(loader(dataset, 256, workers, False), desc="v5 real teacher prototypes"):
            real = batch["sar"].to(device)
            _, features = judge((real + 1) * .5, return_features=True)
            features = F.normalize(features, dim=1)
            labels = batch["class_id"].to(device)
            depression = batch["depression"].to(device)
            az_bin = azimuth_bin(batch["azimuth"].to(device))
            sums.index_put_((labels, depression, az_bin), features, accumulate=True)
            counts.index_put_((labels, depression, az_bin),
                              torch.ones_like(labels, dtype=torch.float), accumulate=True)
    fallback_sum = sums.sum(2)
    fallback_count = counts.sum(2).clamp_min(1)
    missing = counts == 0
    fallback = fallback_sum / fallback_count[..., None]
    centres = sums / counts[..., None].clamp_min(1)
    centres[missing] = fallback[:, :, None, :].expand_as(centres)[missing]
    return F.normalize(centres, dim=3)


def centre_and_extent(image: torch.Tensor) -> torch.Tensor:
    amplitude = ((image + 1) * .5).clamp_min(0)
    # Remove most of the background floor before computing spatial moments.
    weight = F.relu(amplitude - amplitude.mean((2, 3), keepdim=True) * .55)
    weight = weight / weight.sum((2, 3), keepdim=True).clamp_min(1e-5)
    height, width = image.shape[-2:]
    yy = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)[None, None, :, None]
    xx = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)[None, None, None, :]
    cx = (weight * xx).sum((2, 3)); cy = (weight * yy).sum((2, 3))
    sx = (weight * (xx - cx[:, :, None, None]).square()).sum((2, 3)).sqrt()
    sy = (weight * (yy - cy[:, :, None, None]).square()).sum((2, 3)).sqrt()
    return torch.cat((cx, cy, sx, sy), 1)


def low_frequency_structure_loss(clean: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    aligned = _align_translation(clean, real)
    low4 = F.l1_loss(F.avg_pool2d(clean, 4), F.avg_pool2d(aligned, 4))
    low8 = F.l1_loss(F.avg_pool2d(clean, 8), F.avg_pool2d(aligned, 8))
    moments = F.l1_loss(centre_and_extent(clean), centre_and_extent(aligned))
    return low4 + .5 * low8 + .5 * moments


def spectral_statistics_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    fake_spectrum, real_spectrum = spectrum_view(fake), spectrum_view(real)
    return (F.l1_loss(F.avg_pool2d(fake_spectrum, 4), F.avg_pool2d(real_spectrum, 4))
            + .5 * F.l1_loss(fake_spectrum.std((2, 3)), real_spectrum.std((2, 3))))


def feature_matching(fake_features: tuple[torch.Tensor, ...],
                     real_features: tuple[torch.Tensor, ...]) -> torch.Tensor:
    total = fake_features[0].new_zeros(())
    for fake, real in zip(fake_features, real_features):
        real = real.detach()
        total = total + F.l1_loss(fake.mean((2, 3)), real.mean((2, 3)))
        total = total + F.l1_loss(fake.std((2, 3)), real.std((2, 3)))
    return total / len(fake_features)


def save_preview(path: Path, rgb: torch.Tensor, real: torch.Tensor,
                 clean: torch.Tensor, fake: torch.Tensor) -> None:
    rows = []
    for index in range(min(8, len(fake))):
        rgb_panel = F.interpolate(rgb[index:index + 1], (64, 64), mode="bilinear",
                                  align_corners=False)[0]
        rgb_panel = ((rgb_panel.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy()
                      + 1) * 127.5).astype(np.uint8)
        panels = [rgb_panel]
        for source in (real, clean, fake):
            panel = ((source[index, 0].detach().cpu().clamp(-1, 1).numpy()
                      + 1) * 127.5).astype(np.uint8)
            panels.append(np.repeat(panel[..., None], 3, axis=2))
        rows.append(np.concatenate(panels, 1))
    Image.fromarray(np.concatenate(rows, 0), "RGB").save(path)


def main() -> None:
    args = arguments()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    manifest = build_manifest(args.sar_train_root, args.output / "split_manifest.json",
                              args.validation_fraction, args.seed)
    train = V5PairDataset(args.rgb_root, args.sar_train_root, manifest, "train", augment_rgb=True)
    validation = V5PairDataset(args.rgb_root, args.sar_train_root, manifest, "validation")
    train_loader = loader(train, args.batch_size, args.workers, True)
    validation_loader = loader(validation, args.batch_size, args.workers, False)

    judge_state = torch.load(args.native_classifier_checkpoint, map_location=device, weights_only=False)
    if judge_state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("teacher class order mismatch")
    judge = SARClassifier64(40).to(device)
    judge.load_state_dict(judge_state["model"]); judge.eval(); set_grad(judge, False)
    prototypes = prepare_teacher_prototypes(judge, train, device, args.workers)

    encoder = RGBIdentityEncoder(40).to(device)
    generator = RGBReflectivityGenerator().to(device)
    discriminator = MultiDomainDiscriminator().to(device)
    ema_encoder, ema_generator = copy.deepcopy(encoder).eval(), copy.deepcopy(generator).eval()
    set_grad(ema_encoder, False); set_grad(ema_generator, False)
    generator_opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(generator.parameters()), lr=args.generator_lr,
        betas=(0., .99), weight_decay=1e-4)
    discriminator_opt = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0., .99))
    generator_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    discriminator_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ce = nn.CrossEntropyLoss(label_smoothing=.03)
    start_epoch, best_quality = 1, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(state["rgb_encoder"]); generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        ema_encoder.load_state_dict(state["ema_rgb_encoder"]); ema_generator.load_state_dict(state["ema_generator"])
        generator_opt.load_state_dict(state["generator_optimizer"])
        discriminator_opt.load_state_dict(state["discriminator_optimizer"])
        generator_scaler.load_state_dict(state.get("generator_scaler", {}))
        discriminator_scaler.load_state_dict(state.get("discriminator_scaler", {}))
        start_epoch = int(state["epoch"]) + 1
        best_quality = float(state["best_quality"])

    columns = ("epoch", "generator", "adversarial", "rgb_identity", "cross_view",
               "structure", "statistics", "physics", "spectrum", "teacher_class",
               "teacher_contrast", "feature_match", "diversity", "discriminator",
               "r1", "rgb_accuracy", "fake_teacher_accuracy", "validation_quality",
               "validation_structure", "validation_statistics", "validation_spectrum",
               "validation_teacher_accuracy")
    history = args.output / "history.csv"
    if start_epoch == 1:
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(columns)
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config.update({"train_samples": len(train), "validation_samples": len(validation),
                       "condition": "X/HH; continuous azimuth; depressions 15/30/45/60",
                       "test_policy": "SOC_40classes_cut/test is untouched during training"})
        (args.output / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train(); generator.train(); discriminator.train()
        totals = torch.zeros(17, dtype=torch.float64); steps = 0
        progress = tqdm(train_loader, desc=f"v5 hybrid GAN {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            rgb = batch["rgb"].to(device, non_blocking=True)
            rgb_alt = batch["rgb_alt"].to(device, non_blocking=True)
            real = batch["sar"].to(device, non_blocking=True)
            labels = batch["class_id"].to(device, non_blocking=True)
            geometry = batch["condition"].to(device, non_blocking=True)
            depression = batch["depression"].to(device, non_blocking=True)
            az_bin = azimuth_bin(batch["azimuth"].to(device, non_blocking=True))
            style = torch.randn(len(real), generator.noise_dim, device=device)
            observation_noise = torch.randn_like(real)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                identity, rgb_logits, pyramid = encoder(rgb, return_pyramid=True)
                alt_identity, alt_logits = encoder(rgb_alt)
                clean, sigma = generator(identity, geometry, pyramid, style)
                fake = sar_observation(clean, sigma, observation_noise)

            discriminator_opt.zero_grad(set_to_none=True)
            do_r1 = args.r1_weight > 0 and batch_index % args.r1_every == 0
            real_for_d = real.detach().requires_grad_(do_r1)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                real_scores, _ = discriminator(real_for_d, labels, geometry)
                fake_scores, _ = discriminator(fake.detach(), labels, geometry)
                wrong_label_score, _ = discriminator.spatial(real, labels.roll(1), geometry)
                wrong_geometry_score, _ = discriminator.spatial(real, labels, geometry.roll(1, 0))
                discriminator_loss = sum(
                    F.relu(1 - score).mean() + F.relu(1 + fake_score).mean()
                    for score, fake_score in zip(real_scores, fake_scores)) / len(real_scores)
                discriminator_loss = (discriminator_loss
                                      + .25 * F.relu(1 + wrong_label_score).mean()
                                      + .25 * F.relu(1 + wrong_geometry_score).mean())
                r1 = real.new_zeros(())
                if do_r1:
                    gradient = torch.autograd.grad(
                        sum(score.sum() for score in real_scores), real_for_d,
                        create_graph=True)[0]
                    r1 = gradient.flatten(1).square().sum(1).mean()
                    discriminator_loss = discriminator_loss + (
                        .5 * args.r1_weight * args.r1_every) * r1
            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.unscale_(discriminator_opt)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.)
            discriminator_scaler.step(discriminator_opt); discriminator_scaler.update()

            set_grad(discriminator, False)
            generator_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                fake_scores, fake_disc_features = discriminator(fake, labels, geometry)
                with torch.no_grad():
                    _, real_disc_features = discriminator(real, labels, geometry)
                adversarial = -sum(score.mean() for score in fake_scores) / len(fake_scores)
                match_loss = feature_matching(fake_disc_features, real_disc_features)
                rgb_identity_loss = .5 * (ce(rgb_logits, labels) + ce(alt_logits, labels))
                cross_view = 1 - (
                    F.normalize(identity, dim=1) * F.normalize(alt_identity, dim=1)).sum(1).mean()
                structure = low_frequency_structure_loss(clean, real)
                statistics = sar_statistics_loss(fake, real)
                physics = sar_physics_prior_loss(fake, real)
                spectrum = spectral_statistics_loss(fake, real)
                teacher_logits, teacher_features = judge((fake + 1) * .5, return_features=True)
                teacher_features = F.normalize(teacher_features, dim=1)
                positive = (teacher_features * prototypes[labels, depression, az_bin]).sum(1)
                negative = (teacher_features * prototypes[labels.roll(1), depression, az_bin]).sum(1)
                teacher_class = ce(teacher_logits, labels)
                teacher_contrast = (1 - positive).mean() + F.relu(.15 - positive + negative).mean()
                style2 = torch.randn_like(style)
                clean2, sigma2 = generator(identity, geometry, pyramid, style2)
                fake2 = sar_observation(clean2, sigma2, observation_noise)
                high_difference = F.l1_loss(highpass_view(fake), highpass_view(fake2))
                low_consistency = F.l1_loss(F.avg_pool2d(clean, 8), F.avg_pool2d(clean2, 8))
                diversity = F.relu(fake.new_tensor(.025) - high_difference) + .2 * low_consistency
                generator_loss = (
                    2.0 * adversarial
                    + 1.0 * rgb_identity_loss + .5 * cross_view
                    + 5.0 * structure + 1.5 * statistics + .5 * physics
                    + 1.0 * spectrum + .05 * teacher_class
                    + .3 * teacher_contrast + 2.0 * match_loss + .1 * diversity)
            generator_scaler.scale(generator_loss).backward()
            generator_scaler.unscale_(generator_opt)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(generator.parameters()), 5.)
            generator_scaler.step(generator_opt); generator_scaler.update()
            set_grad(discriminator, True)
            update_ema(ema_encoder, encoder, args.ema_decay)
            update_ema(ema_generator, generator, args.ema_decay)

            values = (generator_loss, adversarial, rgb_identity_loss, cross_view,
                      structure, statistics, physics, spectrum, teacher_class,
                      teacher_contrast, match_loss, diversity, discriminator_loss, r1,
                      .5 * ((rgb_logits.argmax(1) == labels).float().mean()
                            + (alt_logits.argmax(1) == labels).float().mean()),
                      (teacher_logits.argmax(1) == labels).float().mean())
            totals[:16] += torch.tensor(
                [value.detach().item() for value in values], dtype=torch.float64)
            totals[16] += 1; steps += 1
            if args.limit_train_batches and batch_index + 1 >= args.limit_train_batches:
                break

        ema_encoder.eval(); ema_generator.eval()
        val_total = val_structure = val_statistics = val_spectrum = val_correct = 0.
        preview = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(validation_loader):
                rgb = batch["rgb"].to(device); real = batch["sar"].to(device)
                labels = batch["class_id"].to(device); geometry = batch["condition"].to(device)
                identity, _, pyramid = ema_encoder(rgb, return_pyramid=True)
                style = torch.zeros(len(real), ema_generator.noise_dim, device=device)
                noise_generator = torch.Generator(device=device).manual_seed(args.seed + batch_index)
                noise = torch.randn(real.shape, generator=noise_generator, device=device)
                clean, sigma = ema_generator(identity, geometry, pyramid, style)
                fake = sar_observation(clean, sigma, noise)
                size = len(real)
                val_structure += low_frequency_structure_loss(clean, real).item() * size
                val_statistics += sar_statistics_loss(fake, real).item() * size
                val_spectrum += spectral_statistics_loss(fake, real).item() * size
                val_correct += (judge((fake + 1) * .5).argmax(1) == labels).sum().item()
                val_total += size
                if preview is None:
                    preview = (rgb, real, clean, fake)
                if args.limit_validation_batches and batch_index + 1 >= args.limit_validation_batches:
                    break
        val_structure /= val_total; val_statistics /= val_total
        val_spectrum /= val_total; val_accuracy = val_correct / val_total
        quality = val_structure + val_statistics + val_spectrum + .05 * (1 - val_accuracy)
        averages = (totals[:16] / totals[16]).tolist()
        row = (epoch, *averages, quality, val_structure, val_statistics,
               val_spectrum, val_accuracy)
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        state = {
            "architecture": "v5_rgb_reflectivity_multidomain_gan",
            "epoch": epoch, "classes": list(SOC40_CLASSES),
            "rgb_encoder": encoder.state_dict(), "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "ema_rgb_encoder": ema_encoder.state_dict(), "ema_generator": ema_generator.state_dict(),
            "generator_optimizer": generator_opt.state_dict(),
            "discriminator_optimizer": discriminator_opt.state_dict(),
            "generator_scaler": generator_scaler.state_dict(),
            "discriminator_scaler": discriminator_scaler.state_dict(),
            "best_quality": min(best_quality, quality), "validation_quality": quality,
            "validation_structure": val_structure, "validation_statistics": val_statistics,
            "validation_spectrum": val_spectrum, "validation_teacher_accuracy": val_accuracy,
            "split_manifest": str(args.output / "split_manifest.json"),
        }
        torch.save(state, args.output / "latest.pt")
        if quality < best_quality:
            best_quality = quality
            state["best_quality"] = quality
            torch.save(state, args.output / "best.pt")
        if epoch == 1 or epoch % 5 == 0:
            assert preview is not None
            save_preview(args.output / f"validation_{epoch:03d}.png", *preview)
        print(dict(zip(columns, row)), flush=True)


if __name__ == "__main__":
    main()

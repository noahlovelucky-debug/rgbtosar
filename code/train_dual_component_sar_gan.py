"""Train the large dual-generator, three-discriminator continuous SAR GAN.

This experiment is deliberately trained from random initialisation.  It never
loads continuous-spatial-v1 ``best.pt``; ``--resume`` only resumes a checkpoint
created by this exact architecture.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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

from dual_component_sar_gan import (
    DenoisedSARGenerator, DualComponentDiscriminators, LargeRGBIdentityEncoder,
    SARNoiseGenerator, compose_sar, decompose_real_sar, initialise, noise_view)
from joint_data import JointROIDataset
from joint_models import _align_translation, sar_physics_prior_loss, sar_statistics_loss
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES
from train_continuous_spatial_roi_gan import target_condition


ARCHITECTURE = "dual_component_continuous_sar_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="from-scratch clean-SAR + learned-noise conditional GAN")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--epoch-size", type=int, default=24000,
                        help="augmented samples drawn per epoch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--generator-lr", type=float, default=1.5e-4)
    parser.add_argument("--identity-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=1e-4)
    parser.add_argument("--adversarial-warmup-epochs", type=int, default=2)
    parser.add_argument("--r1-weight", type=float, default=.25)
    parser.add_argument("--r1-every", type=int, default=16)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-validation-batches", type=int, default=0)
    return parser.parse_args()


def make_loader(dataset: JointROIDataset, batch: int, workers: int,
                shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        drop_last=shuffle)


def split_records(records: list[tuple], root: Path, manifest: Path,
                  fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("source_root") == str(root.resolve()):
            return set(saved["train"]), set(saved["validation"])
    groups: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for record in records:
        groups[record[2], int(record[4]["depression"])].append(record)
    train, validation = [], []
    for group, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda record: hashlib.sha256(
            f"{seed}:{group}:{record[0].relative_to(root)}".encode()).hexdigest())
        count = max(1, round(len(ordered) * fraction))
        validation.extend(str(item[0].relative_to(root)) for item in ordered[:count])
        train.extend(str(item[0].relative_to(root)) for item in ordered[count:])
    payload = {
        "version": ARCHITECTURE, "source_root": str(root.resolve()),
        "seed": seed, "validation_fraction": fraction,
        "train": sorted(train), "validation": sorted(validation)}
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return set(train), set(validation)


def configure_records(dataset: JointROIDataset, selected: set[str],
                      root: Path, epoch_size: int = 0) -> None:
    dataset.records = [
        record for record in dataset.records
        if str(record[0].relative_to(root)) in selected]
    if not dataset.records:
        raise RuntimeError("empty train/validation split")
    dataset.epoch_size = epoch_size or len(dataset.records)
    # Re-sampling with fresh RGB and SAR augmentation expands the effective set.
    dataset.random_epoch = bool(epoch_size)


def set_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    for ema, current in zip(target.parameters(), source.parameters()):
        ema.lerp_(current, 1 - decay)
    for ema, current in zip(target.buffers(), source.buffers()):
        ema.copy_(current)


def augment_real_sar(image: torch.Tensor) -> torch.Tensor:
    """Label-preserving radiometric and small-translation SAR augmentation."""
    amplitude = ((image + 1) * .5).clamp(0, 1)
    batch = len(amplitude)
    gain = amplitude.new_empty(batch, 1, 1, 1).uniform_(.85, 1.15)
    gamma = amplitude.new_empty(batch, 1, 1, 1).uniform_(.90, 1.10)
    amplitude = amplitude.clamp_min(1e-5).pow(gamma) * gain
    amplitude = amplitude + amplitude.new_empty(batch, 1, 1, 1).uniform_(0, .008) * torch.randn_like(amplitude)
    padded = F.pad(amplitude, (3, 3, 3, 3), mode="replicate")
    shifted = []
    for index in range(batch):
        dy = int(torch.randint(0, 7, (), device=image.device))
        dx = int(torch.randint(0, 7, (), device=image.device))
        shifted.append(padded[index:index + 1, :, dy:dy + 64, dx:dx + 64])
    return torch.cat(shifted, 0).clamp(0, 1) * 2 - 1


def low_structure_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    real = _align_translation(fake, real, max_shift=4)
    return (F.l1_loss(F.avg_pool2d(fake, 4), F.avg_pool2d(real, 4))
            + .5 * F.l1_loss(F.avg_pool2d(fake, 8), F.avg_pool2d(real, 8)))


def _correlation(image: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    source = image[..., :image.shape[-2] - dy, :image.shape[-1] - dx]
    shifted = image[..., dy:, dx:]
    dims = (2, 3)
    source = source - source.mean(dims, keepdim=True)
    shifted = shifted - shifted.mean(dims, keepdim=True)
    return ((source * shifted).mean(dims)
            / (source.std(dims) * shifted.std(dims) + 1e-5))


def noise_statistics_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    dims = (2, 3)
    moments = (F.l1_loss(fake.mean(dims), real.mean(dims))
               + F.l1_loss(fake.std(dims), real.std(dims))
               + .5 * F.l1_loss(fake.abs().mean(dims), real.abs().mean(dims)))
    correlation = sum(
        F.l1_loss(_correlation(fake, dy, dx), _correlation(real, dy, dx))
        for dy, dx in ((0, 1), (1, 0), (1, 1))) / 3
    return moments + correlation


def spectrum_statistics_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Compare log Fourier radial bands without requiring registration."""
    def bands(image: torch.Tensor) -> torch.Tensor:
        amplitude = (image + 1) * .5
        spectrum = torch.log1p(torch.fft.fftshift(
            torch.fft.fft2(amplitude.float(), norm="ortho"), dim=(-2, -1)).abs())
        side = spectrum.shape[-1]
        axis = torch.arange(side, device=image.device) - side // 2
        radius = torch.sqrt(axis[:, None].square() + axis[None, :].square())
        outputs = []
        for low, high in ((0, 4), (4, 8), (8, 16), (16, 32)):
            mask = ((radius >= low) & (radius < high)).to(spectrum.dtype)
            outputs.append((spectrum * mask).sum((2, 3))
                           / mask.sum().clamp_min(1))
        return torch.stack(outputs, 1)
    return F.l1_loss(bands(fake), bands(real))


def feature_match(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return (F.l1_loss(fake.mean((2, 3)), real.detach().mean((2, 3)))
            + F.l1_loss(fake.std((2, 3)), real.detach().std((2, 3))))


def discriminator_hinge(real_score: torch.Tensor,
                        fake_score: torch.Tensor) -> torch.Tensor:
    return F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()


def noise_visual(log_noise: torch.Tensor) -> torch.Tensor:
    return noise_view(log_noise)


def save_preview(path: Path, rgb: torch.Tensor, real: torch.Tensor,
                 real_clean: torch.Tensor, real_noise: torch.Tensor,
                 fake_clean: torch.Tensor, fake_noise: torch.Tensor,
                 fake: torch.Tensor) -> None:
    """Columns: RGB, real, real-clean, real-noise, fake-clean, fake-noise, fake."""
    rows = []
    for index in range(min(8, len(fake))):
        rgb_panel = F.interpolate(
            rgb[index:index + 1], (64, 64), mode="bilinear",
            align_corners=False)[0]
        rgb_panel = (((rgb_panel.detach().cpu().clamp(-1, 1)
                       .permute(1, 2, 0).numpy()) + 1) * 127.5).astype(np.uint8)
        panels = [rgb_panel]
        for tensor in (real, real_clean, noise_visual(real_noise),
                       fake_clean, noise_visual(fake_noise), fake):
            panel = (((tensor[index, 0].detach().cpu().clamp(-1, 1).numpy())
                      + 1) * 127.5).astype(np.uint8)
            panels.append(np.repeat(panel[..., None], 3, 2))
        rows.append(np.concatenate(panels, 1))
    Image.fromarray(np.concatenate(rows, 0), "RGB").save(path)


def parameter_count(*modules: nn.Module) -> int:
    return sum(parameter.numel() for module in modules
               for parameter in module.parameters())


def checkpoint_state(
        epoch: int, encoder: nn.Module, clean_generator: nn.Module,
        noise_generator: nn.Module, discriminators: nn.Module,
        ema_encoder: nn.Module, ema_clean: nn.Module, ema_noise: nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        validation: dict[str, float], args: argparse.Namespace) -> dict:
    return {
        "architecture": ARCHITECTURE, "epoch": epoch,
        "classes": list(SOC40_CLASSES),
        "identity_encoder": encoder.state_dict(),
        "clean_generator": clean_generator.state_dict(),
        "noise_generator": noise_generator.state_dict(),
        "discriminators": discriminators.state_dict(),
        "ema_identity_encoder": ema_encoder.state_dict(),
        "ema_clean_generator": ema_clean.state_dict(),
        "ema_noise_generator": ema_noise.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "validation": validation,
        "split_manifest": str(args.output / "split_manifest.json"),
        "training_policy": "from scratch; no continuous-spatial best checkpoint",
    }


def main() -> None:
    args = arguments()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    train_data = JointROIDataset(
        args.rgb_root, args.sar_train_root, epoch_size=0, band="X",
        polarization="HH", depression="all", augment_rgb=True,
        source_view_mode="mixed")
    train_keys, validation_keys = split_records(
        train_data.records, args.sar_train_root,
        args.output / "split_manifest.json",
        args.validation_fraction, args.seed)
    configure_records(
        train_data, train_keys, args.sar_train_root, args.epoch_size)
    validation_data = JointROIDataset(
        args.rgb_root, args.sar_train_root, epoch_size=0, band="X",
        polarization="HH", depression="all", augment_rgb=False,
        source_view_mode="nearest")
    configure_records(
        validation_data, validation_keys, args.sar_train_root)
    train_loader = make_loader(
        train_data, args.batch_size, args.workers, True)
    validation_loader = make_loader(
        validation_data, args.batch_size, args.workers, False)

    teacher_state = torch.load(
        args.native_classifier_checkpoint, map_location=device,
        weights_only=False)
    if teacher_state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("native classifier class order mismatch")
    teacher = SARClassifier64(40).to(device)
    teacher.load_state_dict(teacher_state["model"])
    teacher.eval(); set_grad(teacher, False)

    encoder = LargeRGBIdentityEncoder(40).to(device)
    clean_generator = DenoisedSARGenerator().to(device)
    noise_generator = SARNoiseGenerator().to(device)
    discriminators = DualComponentDiscriminators().to(device)
    encoder.apply(initialise); clean_generator.apply(initialise)
    noise_generator.apply(initialise); discriminators.apply(initialise)
    ema_encoder = copy.deepcopy(encoder).eval()
    ema_clean = copy.deepcopy(clean_generator).eval()
    ema_noise = copy.deepcopy(noise_generator).eval()
    set_grad(ema_encoder, False); set_grad(ema_clean, False)
    set_grad(ema_noise, False)

    generator_optimizer = torch.optim.AdamW((
        {"params": encoder.parameters(), "lr": args.identity_lr},
        {"params": clean_generator.parameters(), "lr": args.generator_lr},
        {"params": noise_generator.parameters(), "lr": args.generator_lr},
    ), betas=(0., .99), weight_decay=1e-4)
    discriminator_optimizer = torch.optim.Adam(
        discriminators.parameters(), lr=args.discriminator_lr,
        betas=(0., .99))
    generator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    discriminator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ce = nn.CrossEntropyLoss(label_smoothing=.03)
    start_epoch = 1
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("architecture") != ARCHITECTURE:
            raise RuntimeError("--resume is not a dual-component checkpoint")
        encoder.load_state_dict(saved["identity_encoder"])
        clean_generator.load_state_dict(saved["clean_generator"])
        noise_generator.load_state_dict(saved["noise_generator"])
        discriminators.load_state_dict(saved["discriminators"])
        ema_encoder.load_state_dict(saved["ema_identity_encoder"])
        ema_clean.load_state_dict(saved["ema_clean_generator"])
        ema_noise.load_state_dict(saved["ema_noise_generator"])
        generator_optimizer.load_state_dict(saved["generator_optimizer"])
        discriminator_optimizer.load_state_dict(
            saved["discriminator_optimizer"])
        start_epoch = int(saved["epoch"]) + 1

    counts = {
        "encoder": parameter_count(encoder),
        "clean_generator": parameter_count(clean_generator),
        "noise_generator": parameter_count(noise_generator),
        "three_discriminators": parameter_count(discriminators),
    }
    counts["total"] = sum(counts.values())
    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "architecture": ARCHITECTURE,
        "parameters": counts,
        "train_records": len(train_data.records),
        "validation_records": len(validation_data.records),
        "effective_augmented_epoch": len(train_data),
        "real_decomposition": "Lee-style amplitude + bounded log multiplicative residual",
        "checkpoint_selection": "latest/milestones only; no best-model selection",
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print({"parameters": counts, "train": train_data.summary(),
           "validation": validation_data.summary()}, flush=True)

    columns = (
        "epoch", "generator", "adv_clean", "adv_noise", "adv_full",
        "rgb_identity", "cross_view", "clean_structure", "clean_statistics",
        "noise_statistics", "full_statistics", "physics", "spectrum",
        "feature_match", "teacher_class", "noise_diversity",
        "noise_zero_mean", "discriminator", "disc_clean", "disc_noise",
        "disc_full", "r1", "rgb_accuracy", "fake_teacher_accuracy",
        "validation_quality", "validation_clean_structure",
        "validation_clean_statistics", "validation_noise_statistics",
        "validation_full_statistics", "validation_physics",
        "validation_spectrum", "validation_teacher_accuracy")
    history = args.output / "history.csv"
    if start_epoch == 1:
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(columns)

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train(); clean_generator.train(); noise_generator.train()
        discriminators.train()
        totals = defaultdict(float)
        steps = 0
        progress = tqdm(
            train_loader, desc=f"dual component {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            rgb = batch["rgb"].to(device)
            rgb_alt = batch["rgb_alt"].to(device)
            labels = batch["class_id"].to(device)
            geometry = target_condition(
                batch["meta"].to(device), batch["rgb_angle"].to(device))
            real = augment_real_sar(batch["roi"].to(device))
            real_clean, real_noise = decompose_real_sar(real)
            latent_noise = torch.randn(
                len(real), noise_generator.noise_dim, device=device)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                identity, rgb_logits, pyramid = encoder(
                    rgb, return_pyramid=True)
                alt_identity, alt_logits = encoder(rgb_alt)
                fake_clean = clean_generator(identity, geometry, pyramid)
                fake_noise = noise_generator(
                    fake_clean, geometry, pyramid, latent_noise)
                fake = compose_sar(fake_clean, fake_noise)

            discriminator_optimizer.zero_grad(set_to_none=True)
            do_r1 = (args.r1_weight > 0
                     and batch_index % args.r1_every == 0)
            real_for_full = real.detach().requires_grad_(do_r1)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                real_clean_score, _ = discriminators.clean(
                    real_clean, labels, geometry)
                fake_clean_score, _ = discriminators.clean(
                    fake_clean.detach(), labels, geometry)
                real_noise_score, _ = discriminators.noise(
                    noise_view(real_noise), labels, geometry)
                fake_noise_score, _ = discriminators.noise(
                    noise_view(fake_noise.detach()), labels, geometry)
                real_full_score, _ = discriminators.full(
                    real_for_full, labels, geometry)
                fake_full_score, _ = discriminators.full(
                    fake.detach(), labels, geometry)
                disc_clean = discriminator_hinge(
                    real_clean_score, fake_clean_score)
                disc_noise = discriminator_hinge(
                    real_noise_score, fake_noise_score)
                disc_full = discriminator_hinge(
                    real_full_score, fake_full_score)
                wrong_label, _ = discriminators.full(
                    real, labels.roll(1), geometry)
                wrong_geometry, _ = discriminators.full(
                    real, labels, geometry.roll(1, 0))
                discriminator_loss = (
                    disc_clean + disc_noise + 1.5 * disc_full
                    + .25 * F.relu(1 + wrong_label).mean()
                    + .25 * F.relu(1 + wrong_geometry).mean())
                r1 = real.new_zeros(())
                if do_r1:
                    gradient = torch.autograd.grad(
                        real_full_score.sum(), real_for_full,
                        create_graph=True)[0]
                    r1 = gradient.flatten(1).square().sum(1).mean()
                    discriminator_loss = (
                        discriminator_loss + .5 * args.r1_weight
                        * args.r1_every * r1)
            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.unscale_(discriminator_optimizer)
            torch.nn.utils.clip_grad_norm_(
                discriminators.parameters(), 5.)
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()

            set_grad(discriminators, False)
            generator_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                fake_clean_score, fake_clean_feature = discriminators.clean(
                    fake_clean, labels, geometry)
                fake_noise_score, fake_noise_feature = discriminators.noise(
                    noise_view(fake_noise), labels, geometry)
                fake_full_score, fake_full_feature = discriminators.full(
                    fake, labels, geometry)
                with torch.no_grad():
                    _, real_clean_feature = discriminators.clean(
                        real_clean, labels, geometry)
                    _, real_noise_feature = discriminators.noise(
                        noise_view(real_noise), labels, geometry)
                    _, real_full_feature = discriminators.full(
                        real, labels, geometry)
                adv_scale = 0. if epoch <= args.adversarial_warmup_epochs else 1.
                adv_clean = -fake_clean_score.mean()
                adv_noise = -fake_noise_score.mean()
                adv_full = -fake_full_score.mean()
                rgb_identity = .5 * (
                    ce(rgb_logits, labels) + ce(alt_logits, labels))
                cross_view = 1 - (
                    F.normalize(identity, dim=1)
                    * F.normalize(alt_identity, dim=1)).sum(1).mean()
                clean_structure = low_structure_loss(
                    fake_clean, real_clean)
                clean_statistics = sar_statistics_loss(
                    fake_clean, real_clean)
                noise_statistics = noise_statistics_loss(
                    fake_noise, real_noise)
                full_statistics = sar_statistics_loss(fake, real)
                physics = sar_physics_prior_loss(fake, real)
                spectrum = spectrum_statistics_loss(fake, real)
                matching = (
                    feature_match(fake_clean_feature, real_clean_feature)
                    + feature_match(fake_noise_feature, real_noise_feature)
                    + feature_match(fake_full_feature, real_full_feature)) / 3
                teacher_logits = teacher((fake + 1) * .5)
                teacher_class = ce(teacher_logits, labels)
                second_noise = noise_generator(
                    fake_clean, geometry, pyramid, torch.randn_like(latent_noise))
                diversity_distance = (
                    fake_noise - second_noise).abs().mean((1, 2, 3))
                noise_diversity = F.relu(.04 - diversity_distance).mean()
                noise_zero_mean = fake_noise.mean((2, 3)).abs().mean()
                generator_loss = (
                    adv_scale * (
                        1.5 * adv_clean + adv_noise + 2 * adv_full)
                    + 2 * rgb_identity + .75 * cross_view
                    + 4 * clean_structure + 2 * clean_statistics
                    + 2 * noise_statistics + 3 * full_statistics
                    + .75 * physics + 1.5 * spectrum + 3 * matching
                    + .05 * teacher_class + 2 * noise_diversity
                    + .5 * noise_zero_mean)
            generator_scaler.scale(generator_loss).backward()
            generator_scaler.unscale_(generator_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                + list(clean_generator.parameters())
                + list(noise_generator.parameters()), 5.)
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()
            set_grad(discriminators, True)
            update_ema(ema_encoder, encoder, args.ema_decay)
            update_ema(ema_clean, clean_generator, args.ema_decay)
            update_ema(ema_noise, noise_generator, args.ema_decay)

            rgb_accuracy = .5 * (
                (rgb_logits.argmax(1) == labels).float().mean()
                + (alt_logits.argmax(1) == labels).float().mean())
            fake_accuracy = (
                teacher_logits.argmax(1) == labels).float().mean()
            values = {
                "generator": generator_loss, "adv_clean": adv_clean,
                "adv_noise": adv_noise, "adv_full": adv_full,
                "rgb_identity": rgb_identity, "cross_view": cross_view,
                "clean_structure": clean_structure,
                "clean_statistics": clean_statistics,
                "noise_statistics": noise_statistics,
                "full_statistics": full_statistics, "physics": physics,
                "spectrum": spectrum, "feature_match": matching,
                "teacher_class": teacher_class,
                "noise_diversity": noise_diversity,
                "noise_zero_mean": noise_zero_mean,
                "discriminator": discriminator_loss,
                "disc_clean": disc_clean, "disc_noise": disc_noise,
                "disc_full": disc_full, "r1": r1,
                "rgb_accuracy": rgb_accuracy,
                "fake_teacher_accuracy": fake_accuracy,
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            steps += 1
            progress.set_postfix(
                g=f"{float(generator_loss.detach()):.3f}",
                d=f"{float(discriminator_loss.detach()):.3f}",
                rgb=f"{float(rgb_accuracy.detach()):.3f}",
                fake=f"{float(fake_accuracy.detach()):.3f}")
            if (args.limit_train_batches
                    and batch_index + 1 >= args.limit_train_batches):
                break

        ema_encoder.eval(); ema_clean.eval(); ema_noise.eval()
        validation_totals = defaultdict(float)
        validation_count = 0
        preview = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(validation_loader):
                rgb = batch["rgb"].to(device)
                labels = batch["class_id"].to(device)
                geometry = target_condition(
                    batch["meta"].to(device),
                    batch["rgb_angle"].to(device))
                real = batch["roi"].to(device)
                real_clean, real_noise = decompose_real_sar(real)
                identity, _, pyramid = ema_encoder(
                    rgb, return_pyramid=True)
                fake_clean = ema_clean(identity, geometry, pyramid)
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + batch_index)
                latent = torch.randn(
                    len(real), ema_noise.noise_dim, device=device,
                    generator=generator)
                fake_noise = ema_noise(
                    fake_clean, geometry, pyramid, latent)
                fake = compose_sar(fake_clean, fake_noise)
                size = len(real)
                metrics = {
                    "clean_structure": low_structure_loss(
                        fake_clean, real_clean),
                    "clean_statistics": sar_statistics_loss(
                        fake_clean, real_clean),
                    "noise_statistics": noise_statistics_loss(
                        fake_noise, real_noise),
                    "full_statistics": sar_statistics_loss(fake, real),
                    "physics": sar_physics_prior_loss(fake, real),
                    "spectrum": spectrum_statistics_loss(fake, real),
                }
                for name, value in metrics.items():
                    validation_totals[name] += float(value) * size
                validation_totals["teacher_correct"] += int(
                    (teacher((fake + 1) * .5).argmax(1)
                     == labels).sum())
                validation_count += size
                if preview is None:
                    preview = (
                        rgb, real, real_clean, real_noise,
                        fake_clean, fake_noise, fake)
                if (args.limit_validation_batches
                        and batch_index + 1
                        >= args.limit_validation_batches):
                    break
        validation = {
            name: value / validation_count
            for name, value in validation_totals.items()}
        validation["quality"] = (
            validation["clean_structure"]
            + validation["clean_statistics"]
            + validation["noise_statistics"]
            + 1.5 * validation["full_statistics"]
            + .5 * validation["physics"]
            + validation["spectrum"])
        validation["teacher_accuracy"] = validation.pop(
            "teacher_correct")
        averages = {
            name: value / max(steps, 1)
            for name, value in totals.items()}
        row = (
            epoch,
            *[averages[name] for name in columns[1:24]],
            validation["quality"], validation["clean_structure"],
            validation["clean_statistics"],
            validation["noise_statistics"],
            validation["full_statistics"], validation["physics"],
            validation["spectrum"], validation["teacher_accuracy"])
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        state = checkpoint_state(
            epoch, encoder, clean_generator, noise_generator,
            discriminators, ema_encoder, ema_clean, ema_noise,
            generator_optimizer, discriminator_optimizer,
            validation, args)
        torch.save(state, args.output / "latest.pt")
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(state, args.output / f"epoch_{epoch:03d}.pt")
        if epoch == 1 or epoch % 5 == 0:
            assert preview is not None
            save_preview(
                args.output / f"validation_{epoch:03d}.png", *preview)
        print(dict(zip(columns, row)), flush=True)


if __name__ == "__main__":
    main()

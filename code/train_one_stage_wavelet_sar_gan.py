"""Train the one-stage alias-free wavelet RGB-to-SAR comparison model."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dual_component_sar_gan import (
    LargeRGBIdentityEncoder, decompose_real_sar)
from joint_data import JointROIDataset
from joint_models import sar_physics_prior_loss, sar_statistics_loss
from one_stage_wavelet_sar_gan import (
    OneStageWaveletDiscriminators, OneStageWaveletSARGenerator,
    haar_texture, initialise)
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES
from train_continuous_spatial_roi_gan import target_condition
from train_dual_component_sar_gan import (
    augment_real_sar, configure_records, discriminator_hinge, feature_match,
    low_structure_loss, make_loader, noise_statistics_loss, parameter_count,
    save_preview, set_grad, spectrum_statistics_loss, split_records, update_ema)


ARCHITECTURE = "one_stage_aliasfree_wavelet_sar_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="one-stage alias-free wavelet RGB-to-SAR GAN")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--epoch-size", type=int, default=24000)
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
    parser.add_argument("--equivariance-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-validation-batches", type=int, default=0)
    return parser.parse_args()


def differentiable_augment(image: torch.Tensor) -> torch.Tensor:
    """DiffAugment-style radiometry and translation for both GAN domains."""
    batch = len(image)
    brightness = image.new_empty(batch, 1, 1, 1).uniform_(-.06, .06)
    contrast = image.new_empty(batch, 1, 1, 1).uniform_(.90, 1.10)
    mean = image.mean((2, 3), keepdim=True)
    image = (image - mean) * contrast + mean + brightness
    padded = F.pad(image, (2, 2, 2, 2), mode="reflect")
    outputs = []
    for index in range(batch):
        dy = int(torch.randint(0, 5, (), device=image.device))
        dx = int(torch.randint(0, 5, (), device=image.device))
        height, width = image.shape[-2:]
        outputs.append(
            padded[index:index + 1, :, dy:dy + height, dx:dx + width])
    return torch.cat(outputs, 0).clamp(-1, 1)


def wavelet_statistics_loss(fake: torch.Tensor,
                            real: torch.Tensor) -> torch.Tensor:
    fake_texture, real_texture = haar_texture(fake), haar_texture(real)
    dims = (2, 3)
    return (F.l1_loss(fake_texture.mean(dims), real_texture.mean(dims))
            + F.l1_loss(fake_texture.std(dims), real_texture.std(dims))
            + .5 * F.l1_loss(
                fake_texture.abs().mean(dims),
                real_texture.abs().mean(dims)))


def shift_tensor(image: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    return torch.roll(image, shifts=(dy, dx), dims=(-2, -1))


def checkpoint_state(
        epoch: int, encoder: nn.Module, generator: nn.Module,
        discriminators: nn.Module, ema_encoder: nn.Module,
        ema_generator: nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        validation: dict[str, float], args: argparse.Namespace) -> dict:
    return {
        "architecture": ARCHITECTURE, "epoch": epoch,
        "classes": list(SOC40_CLASSES),
        "identity_encoder": encoder.state_dict(),
        "generator": generator.state_dict(),
        "discriminators": discriminators.state_dict(),
        "ema_identity_encoder": ema_encoder.state_dict(),
        "ema_generator": ema_generator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "validation": validation,
        "split_manifest": str(args.output / "split_manifest.json"),
        "training_policy": "from scratch; one-stage alias-free wavelet comparison",
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
    configure_records(validation_data, validation_keys, args.sar_train_root)
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
    generator = OneStageWaveletSARGenerator().to(device)
    discriminators = OneStageWaveletDiscriminators().to(device)
    encoder.apply(initialise); generator.apply(initialise)
    discriminators.apply(initialise)
    ema_encoder = copy.deepcopy(encoder).eval()
    ema_generator = copy.deepcopy(generator).eval()
    set_grad(ema_encoder, False); set_grad(ema_generator, False)
    generator_optimizer = torch.optim.AdamW((
        {"params": encoder.parameters(), "lr": args.identity_lr},
        {"params": generator.parameters(), "lr": args.generator_lr},
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
            raise RuntimeError("--resume architecture mismatch")
        encoder.load_state_dict(saved["identity_encoder"])
        generator.load_state_dict(saved["generator"])
        discriminators.load_state_dict(saved["discriminators"])
        ema_encoder.load_state_dict(saved["ema_identity_encoder"])
        ema_generator.load_state_dict(saved["ema_generator"])
        generator_optimizer.load_state_dict(saved["generator_optimizer"])
        discriminator_optimizer.load_state_dict(
            saved["discriminator_optimizer"])
        start_epoch = int(saved["epoch"]) + 1

    counts = {
        "encoder": parameter_count(encoder),
        "one_stage_generator": parameter_count(generator),
        "three_discriminators": parameter_count(discriminators)}
    counts["total"] = sum(counts.values())
    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "architecture": ARCHITECTURE, "parameters": counts,
        "train_records": len(train_data.records),
        "validation_records": len(validation_data.records),
        "effective_augmented_epoch": len(train_data),
        "comparison_baseline": "runs/dual_component_xhh",
        "innovations": [
            "one-pass shared clean/noise decoder",
            "SPADE-like RGB modulation at every scale",
            "anti-aliased bilinear/FIR upsampling",
            "heteroscedastic spatial-random-field log speckle",
            "observable Haar wavelet texture discriminator",
            "real/fake differentiable discriminator augmentation",
            "translation equivariance and real-SAR teacher feature matching",
        ],
        "checkpoint_selection": "latest/milestones only",
    }
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"parameters": counts, "train": train_data.summary(),
           "validation": validation_data.summary()}, flush=True)

    columns = (
        "epoch", "generator", "adv_clean", "adv_full", "adv_texture",
        "rgb_identity", "cross_view", "clean_structure", "clean_statistics",
        "noise_statistics", "full_statistics", "physics", "spectrum",
        "wavelet", "feature_match", "teacher_class", "teacher_feature",
        "equivariance", "discriminator", "disc_clean", "disc_full",
        "disc_texture", "r1", "rgb_accuracy", "fake_teacher_accuracy",
        "validation_quality", "validation_clean_structure",
        "validation_clean_statistics", "validation_noise_statistics",
        "validation_full_statistics", "validation_physics",
        "validation_spectrum", "validation_wavelet",
        "validation_teacher_accuracy")
    history = args.output / "history.csv"
    if start_epoch == 1:
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(columns)

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train(); generator.train(); discriminators.train()
        totals = defaultdict(float); steps = 0
        progress = tqdm(
            train_loader, desc=f"one-stage wavelet {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            rgb = batch["rgb"].to(device)
            rgb_alt = batch["rgb_alt"].to(device)
            labels = batch["class_id"].to(device)
            geometry = target_condition(
                batch["meta"].to(device), batch["rgb_angle"].to(device))
            real = augment_real_sar(batch["roi"].to(device))
            real_clean, real_noise = decompose_real_sar(real)
            spatial_noise = torch.randn(
                len(real), 1, 64, 64, device=device)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                identity, rgb_logits, pyramid = encoder(
                    rgb, return_pyramid=True)
                alt_identity, alt_logits = encoder(rgb_alt)
                fake_clean, fake_noise, fake, _ = generator(
                    identity, geometry, pyramid, spatial_noise)

            discriminator_optimizer.zero_grad(set_to_none=True)
            do_r1 = args.r1_weight > 0 and batch_index % args.r1_every == 0
            real_for_full = differentiable_augment(
                real).detach().requires_grad_(do_r1)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                real_clean_score, _ = discriminators.clean(
                    differentiable_augment(real_clean), labels, geometry)
                fake_clean_score, _ = discriminators.clean(
                    differentiable_augment(fake_clean.detach()),
                    labels, geometry)
                real_full_score, _ = discriminators.full(
                    real_for_full, labels, geometry)
                fake_full_score, _ = discriminators.full(
                    differentiable_augment(fake.detach()), labels, geometry)
                real_texture_score, _ = discriminators.texture(
                    haar_texture(real_for_full), labels, geometry)
                fake_texture_score, _ = discriminators.texture(
                    haar_texture(differentiable_augment(fake.detach())),
                    labels, geometry)
                disc_clean = discriminator_hinge(
                    real_clean_score, fake_clean_score)
                disc_full = discriminator_hinge(
                    real_full_score, fake_full_score)
                disc_texture = discriminator_hinge(
                    real_texture_score, fake_texture_score)
                wrong_label, _ = discriminators.full(
                    real_for_full, labels.roll(1), geometry)
                wrong_geometry, _ = discriminators.full(
                    real_for_full, labels, geometry.roll(1, 0))
                discriminator_loss = (
                    disc_clean + 1.5 * disc_full + disc_texture
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
                clean_aug = differentiable_augment(fake_clean)
                full_aug = differentiable_augment(fake)
                fake_clean_score, fake_clean_feature = discriminators.clean(
                    clean_aug, labels, geometry)
                fake_full_score, fake_full_feature = discriminators.full(
                    full_aug, labels, geometry)
                fake_texture_score, fake_texture_feature = discriminators.texture(
                    haar_texture(full_aug), labels, geometry)
                with torch.no_grad():
                    _, real_clean_feature = discriminators.clean(
                        differentiable_augment(real_clean), labels, geometry)
                    _, real_full_feature = discriminators.full(
                        differentiable_augment(real), labels, geometry)
                    _, real_texture_feature = discriminators.texture(
                        haar_texture(differentiable_augment(real)),
                        labels, geometry)
                adv_scale = (
                    0. if epoch <= args.adversarial_warmup_epochs else 1.)
                adv_clean = -fake_clean_score.mean()
                adv_full = -fake_full_score.mean()
                adv_texture = -fake_texture_score.mean()
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
                wavelet = wavelet_statistics_loss(fake, real)
                matching = (
                    feature_match(fake_clean_feature, real_clean_feature)
                    + feature_match(fake_full_feature, real_full_feature)
                    + feature_match(
                        fake_texture_feature, real_texture_feature)) / 3
                teacher_logits, teacher_fake_feature = teacher(
                    (fake + 1) * .5, return_features=True)
                with torch.no_grad():
                    _, teacher_real_feature = teacher(
                        (real + 1) * .5, return_features=True)
                teacher_class = ce(teacher_logits, labels)
                teacher_feature = 1 - (
                    F.normalize(teacher_fake_feature, dim=1)
                    * F.normalize(
                        teacher_real_feature, dim=1)).sum(1).mean()
                equivariance = real.new_zeros(())
                if batch_index % args.equivariance_every == 0:
                    shifted_rgb = shift_tensor(rgb, 4, 4)
                    shifted_identity, _, shifted_pyramid = encoder(
                        shifted_rgb, return_pyramid=True)
                    shifted_clean, _, _, _ = generator(
                        shifted_identity, geometry, shifted_pyramid,
                        shift_tensor(spatial_noise, 2, 2))
                    expected = shift_tensor(fake_clean, 2, 2)
                    equivariance = F.l1_loss(
                        shifted_clean[..., 4:-4, 4:-4],
                        expected[..., 4:-4, 4:-4])
                generator_loss = (
                    adv_scale * (
                        1.5 * adv_clean + 2 * adv_full
                        + 1.5 * adv_texture)
                    + 2 * rgb_identity + .75 * cross_view
                    + 3 * clean_structure + 2 * clean_statistics
                    + 2 * noise_statistics + 3 * full_statistics
                    + .25 * physics + 1.5 * spectrum + 2 * wavelet
                    + 3 * matching + .03 * teacher_class
                    + .15 * teacher_feature + .5 * equivariance)
            generator_scaler.scale(generator_loss).backward()
            generator_scaler.unscale_(generator_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(generator.parameters()), 5.)
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()
            set_grad(discriminators, True)
            update_ema(ema_encoder, encoder, args.ema_decay)
            update_ema(ema_generator, generator, args.ema_decay)

            rgb_accuracy = .5 * (
                (rgb_logits.argmax(1) == labels).float().mean()
                + (alt_logits.argmax(1) == labels).float().mean())
            fake_accuracy = (
                teacher_logits.argmax(1) == labels).float().mean()
            values = {
                "generator": generator_loss, "adv_clean": adv_clean,
                "adv_full": adv_full, "adv_texture": adv_texture,
                "rgb_identity": rgb_identity, "cross_view": cross_view,
                "clean_structure": clean_structure,
                "clean_statistics": clean_statistics,
                "noise_statistics": noise_statistics,
                "full_statistics": full_statistics, "physics": physics,
                "spectrum": spectrum, "wavelet": wavelet,
                "feature_match": matching, "teacher_class": teacher_class,
                "teacher_feature": teacher_feature,
                "equivariance": equivariance,
                "discriminator": discriminator_loss,
                "disc_clean": disc_clean, "disc_full": disc_full,
                "disc_texture": disc_texture, "r1": r1,
                "rgb_accuracy": rgb_accuracy,
                "fake_teacher_accuracy": fake_accuracy}
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

        ema_encoder.eval(); ema_generator.eval()
        val_totals = defaultdict(float); val_count = 0; preview = None
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
                random_generator = torch.Generator(device=device)
                random_generator.manual_seed(args.seed + batch_index)
                spatial_noise = torch.randn(
                    len(real), 1, 64, 64, device=device,
                    generator=random_generator)
                fake_clean, fake_noise, fake, _ = ema_generator(
                    identity, geometry, pyramid, spatial_noise)
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
                    "wavelet": wavelet_statistics_loss(fake, real)}
                for name, value in metrics.items():
                    val_totals[name] += float(value) * size
                val_totals["teacher_correct"] += int(
                    (teacher((fake + 1) * .5).argmax(1)
                     == labels).sum())
                val_count += size
                if preview is None:
                    preview = (
                        rgb, real, real_clean, real_noise,
                        fake_clean, fake_noise, fake)
                if (args.limit_validation_batches
                        and batch_index + 1
                        >= args.limit_validation_batches):
                    break
        validation = {
            name: value / val_count for name, value in val_totals.items()}
        validation["quality"] = (
            .5 * validation["clean_structure"]
            + validation["clean_statistics"]
            + validation["noise_statistics"]
            + validation["full_statistics"]
            + .1 * validation["physics"]
            + validation["spectrum"] + validation["wavelet"])
        validation["teacher_accuracy"] = validation.pop("teacher_correct")
        averages = {
            name: value / max(steps, 1)
            for name, value in totals.items()}
        row = (
            epoch,
            *[averages[name] for name in columns[1:25]],
            validation["quality"], validation["clean_structure"],
            validation["clean_statistics"],
            validation["noise_statistics"],
            validation["full_statistics"], validation["physics"],
            validation["spectrum"], validation["wavelet"],
            validation["teacher_accuracy"])
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        state = checkpoint_state(
            epoch, encoder, generator, discriminators,
            ema_encoder, ema_generator, generator_optimizer,
            discriminator_optimizer, validation, args)
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

"""Train Continuous Spatial V3 on final observed SAR images only."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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

from continuous_spatial_one_stage_v3 import (
    ARCHITECTURE, ContinuousSpatialOneStageV3,
    OneStageConditionalDiscriminator, target_geometry)
from joint_data import JointROIDataset
from joint_models import _align_translation, sar_statistics_loss
from saratrx import SOC40_CLASSES
from train_dual_component_sar_gan_v2 import (
    augment_real_sar, configure_records, make_loader,
    spectrum_statistics_loss)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="train the one-stage multi-view stochastic SAR GAN V3")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--epoch-size", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--generator-lr", type=float, default=5e-5)
    parser.add_argument("--discriminator-lr", type=float, default=2e-5)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--r1-weight", type=float, default=.25)
    parser.add_argument("--r1-every", type=int, default=16)
    parser.add_argument("--angle-every", type=int, default=4)
    parser.add_argument("--ada-target", type=float, default=.60)
    parser.add_argument("--ada-kimg", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--device",
        default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-validation-batches", type=int, default=0)
    return parser.parse_args()


def set_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module,
               decay: float) -> None:
    for target_value, source_value in zip(
            target.parameters(), source.parameters()):
        target_value.lerp_(source_value, 1.0 - decay)
    for target_value, source_value in zip(
            target.buffers(), source.buffers()):
        target_value.copy_(source_value)


def load_split(
        source: Path, sar_root: Path,
        output: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    train = set(payload.get("train", ()))
    validation = set(payload.get("validation", ()))
    if not train or not validation:
        raise RuntimeError("split manifest has an empty partition")
    known = {
        str(path.relative_to(sar_root))
        for path in sar_root.rglob("*.tif")}
    if not train.issubset(known) or not validation.issubset(known):
        raise RuntimeError("split manifest does not match SAR training root")
    if train & validation:
        raise RuntimeError("train and validation split overlap")
    saved = {
        "version": ARCHITECTURE,
        "source_manifest": str(source.resolve()),
        "source_root": str(sar_root.resolve()),
        "train": sorted(train), "validation": sorted(validation)}
    output.write_text(
        json.dumps(saved, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return train, validation


def initialize_from_v1(
        model: ContinuousSpatialOneStageV3,
        checkpoint: Path, device: torch.device) -> dict:
    state = torch.load(
        checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") != "continuous_spatial_v1":
        raise RuntimeError("--v1-checkpoint is not continuous_spatial_v1")
    model.encoder.load_state_dict(state["identity_encoder"], strict=True)
    incompatible = model.generator.load_state_dict(
        state["generator"], strict=False)
    expected_prefixes = (
        "view_attention.", "geometry_affine.",
        "antialias.", "raw_sigma", "raw_receiver")
    unexpected = list(incompatible.unexpected_keys)
    bad_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(expected_prefixes)]
    if unexpected or bad_missing:
        raise RuntimeError(
            f"V1 initialization mismatch: missing={bad_missing}, "
            f"unexpected={unexpected}")
    return {
        "path": str(checkpoint.resolve()),
        "epoch": int(state.get("epoch", -1)),
        "quality": state.get("quality")}


def legacy_generator_parameters(model: ContinuousSpatialOneStageV3):
    generator = model.generator
    for module in (
            generator.meta, generator.fc, generator.net,
            generator.spatial_projection):
        yield from module.parameters()


def configure_warm_start(
        model: ContinuousSpatialOneStageV3, epoch: int) -> None:
    frozen = epoch <= 5
    set_grad(model.encoder, not frozen)
    for parameter in legacy_generator_parameters(model):
        parameter.requires_grad_(not frozen)
    # New attention, geometry and acquisition parameters always train.
    for module in (
            model.generator.view_attention,
            model.generator.geometry_affine):
        set_grad(module, True)
    model.generator.raw_sigma.requires_grad_(True)
    model.generator.raw_receiver.requires_grad_(True)
    model.generator.set_antialias_strength(
        .15 * min(1.0, epoch / 5.0))


def learning_rate(
        initial: float, epoch: int, epochs: int,
        minimum: float = 1e-6) -> float:
    if epoch <= 25:
        return initial
    progress = (epoch - 25) / max(1, epochs - 25)
    cosine = .5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (initial - minimum) * cosine


def set_optimizer_lrs(
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        args: argparse.Namespace, epoch: int) -> None:
    generator_optimizer.param_groups[0]["lr"] = learning_rate(
        args.encoder_lr, epoch, args.epochs)
    generator_optimizer.param_groups[1]["lr"] = learning_rate(
        args.generator_lr, epoch, args.epochs)
    discriminator_optimizer.param_groups[0]["lr"] = learning_rate(
        args.discriminator_lr, epoch, args.epochs)


def sar_safe_augment(image: torch.Tensor, probability: float) -> torch.Tensor:
    """ADA-compatible translation/radiometry; never changes target geometry."""
    if probability <= 0:
        return image
    batch = len(image)
    active = (
        torch.rand(batch, 1, 1, 1, device=image.device)
        < probability).to(image.dtype)
    amplitude = ((image + 1.0) * .5).clamp(1e-5, 1.0)
    gain = torch.exp(
        torch.randn(batch, 1, 1, 1, device=image.device) * .08)
    gamma = torch.exp(
        torch.randn(batch, 1, 1, 1, device=image.device) * .04)
    radiometric = amplitude.pow(gamma) * gain
    amplitude = amplitude.lerp(radiometric, active).clamp(0.0, 1.0)
    padded = F.pad(amplitude, (2, 2, 2, 2), mode="reflect")
    translated = []
    for index in range(batch):
        if float(active[index]) == 0:
            translated.append(amplitude[index:index + 1])
            continue
        dy = int(torch.randint(0, 5, (), device=image.device))
        dx = int(torch.randint(0, 5, (), device=image.device))
        translated.append(
            padded[index:index + 1, :, dy:dy + 64, dx:dx + 64])
    return torch.cat(translated, 0) * 2.0 - 1.0


def discriminator_hinge(real: torch.Tensor,
                        fake: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()


def low_structure_loss(fake: torch.Tensor,
                       real: torch.Tensor) -> torch.Tensor:
    aligned = _align_translation(fake, real, max_shift=4)
    return (
        F.l1_loss(F.avg_pool2d(fake, 4), F.avg_pool2d(aligned, 4))
        + .5 * F.l1_loss(
            F.avg_pool2d(fake, 8), F.avg_pool2d(aligned, 8)))


def quantile_statistics_loss(
        fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    quantiles = fake.new_tensor((.01, .05, .10, .25, .50, .75, .90, .95, .99))
    fake_amplitude = ((fake.float() + 1.0) * .5).flatten(1)
    real_amplitude = ((real.float() + 1.0) * .5).flatten(1)
    fake_values = torch.quantile(fake_amplitude, quantiles, dim=1)
    real_values = torch.quantile(real_amplitude, quantiles, dim=1)
    return F.l1_loss(fake_values, real_values)


def scattering_moment_loss(
        fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    real = _align_translation(fake, real, max_shift=4)

    def moments(image: torch.Tensor) -> torch.Tensor:
        amplitude = ((image + 1.0) * .5).clamp(0.0, 1.0)
        local = F.relu(
            amplitude - F.avg_pool2d(amplitude, 9, 1, 4))
        side = amplitude.shape[-1]
        axis = torch.linspace(
            -1.0, 1.0, side, device=image.device,
            dtype=image.dtype)
        mass = local.sum((2, 3)).clamp_min(1e-5)
        x = (local * axis[None, None, None, :]).sum((2, 3)) / mass
        y = (local * axis[None, None, :, None]).sum((2, 3)) / mass
        density = (local > .05).to(local.dtype).mean((2, 3))
        strength = local.mean((2, 3))
        return torch.cat((x, y, density, strength), 1)

    return F.l1_loss(moments(fake), moments(real))


def feature_matching(
        fake: tuple[torch.Tensor, ...],
        real: tuple[torch.Tensor, ...]) -> torch.Tensor:
    losses = []
    for fake_feature, real_feature in zip(fake, real):
        real_feature = real_feature.detach()
        losses.append(
            F.l1_loss(
                fake_feature.mean((2, 3)),
                real_feature.mean((2, 3)))
            + F.l1_loss(
                fake_feature.std((2, 3)),
                real_feature.std((2, 3))))
    return sum(losses) / len(losses)


def cross_view_loss(encoding, view_mask: torch.Tensor) -> torch.Tensor:
    target = F.normalize(encoding.identity, dim=1)[:, None]
    per_view = F.normalize(encoding.per_view_identity, dim=2)
    loss = 1.0 - (target * per_view).sum(2)
    return (
        (loss * view_mask).sum()
        / view_mask.sum().clamp_min(1.0))


def random_field(
        batch: int, device: torch.device, dtype: torch.dtype,
        seed: int | None = None) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    return torch.randn(
        batch, 3, 64, 64, device=device,
        dtype=dtype, generator=generator)


def angle_smoothness(
        model: ContinuousSpatialOneStageV3, encoding,
        view_angles: torch.Tensor, view_mask: torch.Tensor,
        azimuth: torch.Tensor, depression: torch.Tensor,
        metadata: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    neighbours = []
    for offset in (-5.0, 5.0):
        target = (azimuth + offset).remainder(360.0)
        geometry = target_geometry(metadata, target, depression)
        base, _ = model.generator.base_amplitude(
            encoding, view_angles, view_mask,
            target, depression, geometry)
        neighbours.append(F.avg_pool2d(base, 4))
    centre = F.avg_pool2d(centre, 4)
    return F.smooth_l1_loss(
        neighbours[0] + neighbours[1], 2.0 * centre)


def preview_image(
        path: Path, rgb: torch.Tensor, real: torch.Tensor,
        first: torch.Tensor, second: torch.Tensor) -> None:
    rows = []
    low_real = F.interpolate(
        F.avg_pool2d(real, 4), (64, 64),
        mode="bilinear", align_corners=False)
    low_fake = F.interpolate(
        F.avg_pool2d(first, 4), (64, 64),
        mode="bilinear", align_corners=False)
    for index in range(min(8, len(real))):
        rgb_panel = F.interpolate(
            rgb[index:index + 1], (64, 64),
            mode="bilinear", align_corners=False)[0]
        rgb_array = (
            (rgb_panel.detach().cpu().clamp(-1, 1)
             .permute(1, 2, 0).numpy() + 1.0) * 127.5
        ).astype(np.uint8)
        panels = [rgb_array]
        for image in (
                real, first, second, low_real, low_fake):
            array = (
                (image[index, 0].detach().cpu()
                 .clamp(-1, 1).numpy() + 1.0) * 127.5
            ).astype(np.uint8)
            panels.append(np.repeat(array[..., None], 3, 2))
        rows.append(np.concatenate(panels, 1))
    Image.fromarray(np.concatenate(rows, 0), "RGB").save(path)


def checkpoint_state(
        epoch: int, model: ContinuousSpatialOneStageV3,
        discriminator: nn.Module,
        ema_model: ContinuousSpatialOneStageV3,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        validation: dict[str, float], ada_probability: float,
        best_quality: float, args: argparse.Namespace) -> dict:
    return {
        "architecture": ARCHITECTURE,
        "epoch": epoch, "classes": list(SOC40_CLASSES),
        "encoder": model.encoder.state_dict(),
        "generator": model.generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "ema_encoder": ema_model.encoder.state_dict(),
        "ema_generator": ema_model.generator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "validation": validation,
        "ada_probability": ada_probability,
        "best_visual_quality": best_quality,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        "training_policy": (
            "V1 warm start; one generator; final observed SAR supervision; "
            "teacher-free generator optimization"),
    }


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    train_data = JointROIDataset(
        args.rgb_root, args.sar_train_root, epoch_size=0,
        band="X", polarization="HH", depression="all",
        augment_rgb=True, source_view_mode="nearest",
        return_all_views=True)
    validation_data = JointROIDataset(
        args.rgb_root, args.sar_train_root, epoch_size=0,
        band="X", polarization="HH", depression="all",
        augment_rgb=False, source_view_mode="nearest",
        return_all_views=True)
    train_keys, validation_keys = load_split(
        args.split_manifest_source, args.sar_train_root,
        args.output / "split_manifest.json")
    configure_records(
        train_data, train_keys, args.sar_train_root,
        args.epoch_size)
    configure_records(
        validation_data, validation_keys,
        args.sar_train_root)
    train_loader = make_loader(
        train_data, args.batch_size, args.workers, True)
    validation_loader = make_loader(
        validation_data, args.batch_size, args.workers, False)

    model = ContinuousSpatialOneStageV3(len(SOC40_CLASSES)).to(device)
    discriminator = OneStageConditionalDiscriminator(
        len(SOC40_CLASSES)).to(device)
    initialization = initialize_from_v1(
        model, args.v1_checkpoint, device)
    ema_model = copy.deepcopy(model).eval()
    set_grad(ema_model, False)
    generator_optimizer = torch.optim.AdamW((
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": model.generator.parameters(), "lr": args.generator_lr},
    ), betas=(0.0, .99), weight_decay=1e-4)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr,
        betas=(0.0, .99))
    generator_scaler = torch.amp.GradScaler(
        device.type, enabled=use_amp)
    discriminator_scaler = torch.amp.GradScaler(
        device.type, enabled=use_amp)
    identity_ce = nn.CrossEntropyLoss(label_smoothing=.03)
    start_epoch, ada_probability = 1, 0.0
    best_quality = float("inf")
    if args.resume:
        saved = torch.load(
            args.resume, map_location=device, weights_only=False)
        if saved.get("architecture") != ARCHITECTURE:
            raise RuntimeError("--resume architecture mismatch")
        model.encoder.load_state_dict(saved["encoder"])
        model.generator.load_state_dict(saved["generator"])
        discriminator.load_state_dict(saved["discriminator"])
        ema_model.encoder.load_state_dict(saved["ema_encoder"])
        ema_model.generator.load_state_dict(saved["ema_generator"])
        generator_optimizer.load_state_dict(
            saved["generator_optimizer"])
        discriminator_optimizer.load_state_dict(
            saved["discriminator_optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        ada_probability = float(saved.get("ada_probability", 0.0))
        best_quality = float(
            saved.get("best_visual_quality", float("inf")))

    parameter_counts = {
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "generator": sum(p.numel() for p in model.generator.parameters()),
        "discriminator": sum(
            p.numel() for p in discriminator.parameters())}
    parameter_counts["total"] = sum(parameter_counts.values())
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        "architecture": ARCHITECTURE,
        "parameters": parameter_counts,
        "initialization": initialization,
        "train_records": len(train_data.records),
        "validation_records": len(validation_data.records),
        "loss_weights": {
            "adversarial": 1.0, "low_structure": 8.0,
            "radiometry": 2.0, "radial_spectrum": 1.5,
            "scattering_moments": 1.0, "feature_matching": 2.0,
            "angle_smoothness": .5, "rgb_identity": 2.0,
            "cross_view": .5},
        "removed_supervision": [
            "Lee clean/noise decomposition", "noise image discriminator",
            "teacher classifier generator gradients"]}
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8")

    columns = (
        "epoch", "generator", "adversarial", "structure",
        "radiometry", "spectrum", "scattering", "feature_match",
        "angle_smoothness", "rgb_identity", "cross_view",
        "discriminator", "r1", "rgb_accuracy", "ada_probability",
        "sigma_mean", "receiver_mean", "validation_quality",
        "validation_structure", "validation_radiometry",
        "validation_spectrum", "validation_scattering",
        "validation_seed_correlation", "validation_seed_l1",
        "validation_lowpass_seed_l1", "validation_rgb_accuracy")
    history = args.output / "history.csv"
    if start_epoch == 1:
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(columns)

    optimizer_step = (start_epoch - 1) * max(
        1, math.ceil(len(train_loader) / args.gradient_accumulation))
    for epoch in range(start_epoch, args.epochs + 1):
        configure_warm_start(model, epoch)
        set_optimizer_lrs(
            generator_optimizer, discriminator_optimizer,
            args, epoch)
        model.train()
        discriminator.train()
        stochastic_scale = min(
            1.0, .25 + .75 * max(0, epoch - 1) / 14.0)
        totals: defaultdict[str, float] = defaultdict(float)
        steps = 0
        generator_optimizer.zero_grad(set_to_none=True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            train_loader, desc=f"one-stage V3 {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            views = batch["rgb_views"].to(device)
            view_angles = batch["rgb_view_angles"].to(device)
            view_mask = batch["rgb_view_mask"].to(device)
            labels = batch["class_id"].to(device)
            real = augment_real_sar(batch["roi"].to(device))
            azimuth = batch["azimuth"].to(device).float()
            depression = batch["depression"].to(device).float()
            metadata = batch["meta"].to(device)
            geometry = target_geometry(
                metadata, azimuth, depression)
            field = random_field(
                len(real), device, real.dtype)

            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                encoding = model.encode(views, view_mask)
                generated = model.generator(
                    encoding, view_angles, view_mask, azimuth,
                    depression, geometry, field, stochastic_scale)

            set_grad(discriminator, True)
            real_for_r1 = sar_safe_augment(
                real, ada_probability)
            compute_r1 = optimizer_step % args.r1_every == 0
            if compute_r1:
                real_for_r1.requires_grad_(True)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                real_output = discriminator(
                    real_for_r1, labels, geometry)
                fake_output = discriminator(
                    sar_safe_augment(
                        generated.sar.detach(), ada_probability),
                    labels, geometry)
                discriminator_loss = discriminator_hinge(
                    real_output.score, fake_output.score)
                r1 = real.new_zeros(())
                if compute_r1:
                    gradient = torch.autograd.grad(
                        real_output.score.sum(), real_for_r1,
                        create_graph=True)[0]
                    r1 = gradient.square().flatten(1).sum(1).mean()
                    discriminator_loss = (
                        discriminator_loss
                        + args.r1_weight * args.r1_every * .5 * r1)
            discriminator_scaler.scale(
                discriminator_loss
                / args.gradient_accumulation).backward()

            set_grad(discriminator, False)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                fake_for_g = discriminator(
                    sar_safe_augment(
                        generated.sar, ada_probability),
                    labels, geometry)
                with torch.no_grad():
                    real_features = discriminator(
                        sar_safe_augment(real, ada_probability),
                        labels, geometry).features
                adversarial = -fake_for_g.score.mean()
                structure = low_structure_loss(
                    generated.sar, real)
                radiometry = (
                    sar_statistics_loss(generated.sar, real)
                    + quantile_statistics_loss(
                        generated.sar, real))
                spectrum = spectrum_statistics_loss(
                    generated.sar, real)
                scattering = scattering_moment_loss(
                    generated.sar, real)
                matching = feature_matching(
                    fake_for_g.features, real_features)
                rgb_identity = identity_ce(
                    encoding.logits, labels)
                view_consistency = cross_view_loss(
                    encoding, view_mask)
                smoothness = real.new_zeros(())
                if batch_index % args.angle_every == 0:
                    smoothness = angle_smoothness(
                        model, encoding, view_angles, view_mask,
                        azimuth, depression, metadata,
                        generated.base)
                adversarial_scale = 0.0 if epoch <= 2 else 1.0
                generator_loss = (
                    adversarial_scale * adversarial
                    + 8.0 * structure
                    + 2.0 * radiometry
                    + 1.5 * spectrum
                    + scattering
                    + 2.0 * matching
                    + .5 * smoothness
                    + 2.0 * rgb_identity
                    + .5 * view_consistency)
            generator_scaler.scale(
                generator_loss
                / args.gradient_accumulation).backward()

            boundary = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(train_loader))
            if boundary:
                discriminator_scaler.unscale_(
                    discriminator_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(), 5.0)
                discriminator_scaler.step(
                    discriminator_optimizer)
                discriminator_scaler.update()
                discriminator_optimizer.zero_grad(set_to_none=True)

                generator_scaler.unscale_(generator_optimizer)
                trainable = [
                    parameter for parameter in model.parameters()
                    if parameter.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                generator_scaler.step(generator_optimizer)
                generator_scaler.update()
                generator_optimizer.zero_grad(set_to_none=True)
                update_ema(ema_model, model, args.ema_decay)
                optimizer_step += 1
                real_sign = (
                    real_output.score.detach() > 0).float().mean()
                direction = float(real_sign - args.ada_target)
                adjustment = (
                    len(real) * args.gradient_accumulation
                    / max(1.0, args.ada_kimg * 1000.0))
                ada_probability = max(
                    0.0, min(1.0, ada_probability
                             + math.copysign(adjustment, direction)
                             if direction != 0 else ada_probability))

            metrics = {
                "generator": generator_loss,
                "adversarial": adversarial,
                "structure": structure, "radiometry": radiometry,
                "spectrum": spectrum, "scattering": scattering,
                "feature_match": matching,
                "angle_smoothness": smoothness,
                "rgb_identity": rgb_identity,
                "cross_view": view_consistency,
                "discriminator": discriminator_loss, "r1": r1,
                "rgb_accuracy":
                    (encoding.logits.argmax(1) == labels).float().mean(),
                "sigma_mean": generated.sigma.mean(),
                "receiver_mean": generated.receiver_scale.mean()}
            for name, value in metrics.items():
                totals[name] += float(value.detach())
            steps += 1
            progress.set_postfix(
                g=f"{float(generator_loss):.3f}",
                d=f"{float(discriminator_loss):.3f}",
                rgb=f"{float(metrics['rgb_accuracy']):.3f}",
                ada=f"{ada_probability:.3f}")
            if (args.limit_train_batches
                    and batch_index + 1
                    >= args.limit_train_batches):
                break

        ema_model.eval()
        ema_model.generator.set_antialias_strength(
            .15 * min(1.0, epoch / 5.0))
        validation_totals: defaultdict[str, float] = defaultdict(float)
        validation_samples = 0
        preview = None
        with torch.inference_mode():
            for validation_index, batch in enumerate(validation_loader):
                views = batch["rgb_views"].to(device)
                view_angles = batch["rgb_view_angles"].to(device)
                view_mask = batch["rgb_view_mask"].to(device)
                labels = batch["class_id"].to(device)
                real = batch["roi"].to(device)
                azimuth = batch["azimuth"].to(device).float()
                depression = batch["depression"].to(device).float()
                metadata = batch["meta"].to(device)
                geometry = target_geometry(
                    metadata, azimuth, depression)
                encoding = ema_model.encode(views, view_mask)
                first_field = random_field(
                    len(real), device, real.dtype,
                    args.seed + validation_index)
                second_field = random_field(
                    len(real), device, real.dtype,
                    args.seed + 100000 + validation_index)
                first = ema_model.generator(
                    encoding, view_angles, view_mask,
                    azimuth, depression, geometry, first_field)
                second = ema_model.generator(
                    encoding, view_angles, view_mask,
                    azimuth, depression, geometry, second_field)
                first_flat = (
                    first.whitened_noise.flatten(1)
                    - first.whitened_noise.flatten(1).mean(
                        1, keepdim=True))
                second_flat = (
                    second.whitened_noise.flatten(1)
                    - second.whitened_noise.flatten(1).mean(
                        1, keepdim=True))
                correlation = (
                    (first_flat * second_flat).mean(1)
                    / (first_flat.std(1) * second_flat.std(1)
                       + 1e-6)).abs().mean()
                low_seed = F.l1_loss(
                    F.avg_pool2d(first.sar, 4),
                    F.avg_pool2d(second.sar, 4))
                metrics = {
                    "structure": low_structure_loss(first.sar, real),
                    "radiometry": (
                        sar_statistics_loss(first.sar, real)
                        + quantile_statistics_loss(first.sar, real)),
                    "spectrum": spectrum_statistics_loss(first.sar, real),
                    "scattering": scattering_moment_loss(first.sar, real),
                    "seed_correlation": correlation,
                    "seed_l1": F.l1_loss(first.sar, second.sar),
                    "lowpass_seed_l1": low_seed,
                    "rgb_accuracy":
                        (encoding.logits.argmax(1)
                         == labels).float().mean()}
                size = len(real)
                for name, value in metrics.items():
                    validation_totals[name] += float(value) * size
                validation_samples += size
                if preview is None:
                    preview = (
                        batch["rgb"].to(device), real,
                        first.sar, second.sar)
                if (args.limit_validation_batches
                        and validation_index + 1
                        >= args.limit_validation_batches):
                    break
        validation = {
            name: value / max(1, validation_samples)
            for name, value in validation_totals.items()}
        quality = (
            8.0 * validation["structure"]
            + 2.0 * validation["radiometry"]
            + 1.5 * validation["spectrum"]
            + validation["scattering"])
        validation["quality"] = quality
        averages = {
            name: value / max(1, steps)
            for name, value in totals.items()}
        row = (
            epoch, *[
                averages[name] for name in (
                    "generator", "adversarial", "structure",
                    "radiometry", "spectrum", "scattering",
                    "feature_match", "angle_smoothness",
                    "rgb_identity", "cross_view", "discriminator",
                    "r1", "rgb_accuracy")],
            ada_probability, averages["sigma_mean"],
            averages["receiver_mean"], quality,
            validation["structure"], validation["radiometry"],
            validation["spectrum"], validation["scattering"],
            validation["seed_correlation"], validation["seed_l1"],
            validation["lowpass_seed_l1"],
            validation["rgb_accuracy"])
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        state = checkpoint_state(
            epoch, model, discriminator, ema_model,
            generator_optimizer, discriminator_optimizer,
            validation, ada_probability,
            min(best_quality, quality), args)
        torch.save(state, args.output / "latest.pt")
        if quality < best_quality:
            best_quality = quality
            state["best_visual_quality"] = best_quality
            torch.save(state, args.output / "best_visual.pt")
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            torch.save(
                state, args.output / f"epoch_{epoch:03d}.pt")
            assert preview is not None
            preview_image(
                args.output / f"validation_{epoch:03d}.png",
                *preview)
        print(dict(zip(columns, row)), flush=True)


if __name__ == "__main__":
    main()

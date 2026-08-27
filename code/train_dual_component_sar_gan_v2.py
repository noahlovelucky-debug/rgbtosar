"""Train the multi-view, non-collapsible dual-stage SAR GAN v2.

Curriculum:
  1. clean RGB->SAR reflectivity and angle conditioning;
  2. stochastic observation model with the clean stage frozen;
  3. low-learning-rate joint fine-tuning.

The geometry validator is trained separately on real SAR and remains frozen.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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

from dual_component_sar_gan import decompose_real_sar, noise_view
from dual_component_sar_gan_v2 import (
    DualComponentDiscriminatorsV2, GEOMETRY_DIM,
    MultiViewDenoisedSARGenerator, MultiViewRGBEncoder,
    NoiseLeakageClassifier, StochasticSARObservation, angle_fourier,
    initialise, residual_view, target_geometry)
from joint_data import JointROIDataset
from joint_models import (
    _align_translation, sar_physics_prior_loss, sar_statistics_loss)
from sar_geometry_validator import (
    DEPRESSION_VALUES, SARGeometryValidator, circular_soft_cross_entropy)
from saratrx import SOC40_CLASSES


ARCHITECTURE = "dual_component_multiview_stochastic_v2"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--geometry-validator-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-epochs", type=int, default=70)
    parser.add_argument("--noise-epochs", type=int, default=40)
    parser.add_argument("--joint-epochs", type=int, default=40)
    parser.add_argument("--epoch-size", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--clean-lr", type=float, default=1.5e-4)
    parser.add_argument("--noise-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=1e-4)
    parser.add_argument("--joint-clean-lr-scale", type=float, default=.1)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--r1-weight", type=float, default=.25)
    parser.add_argument("--r1-every", type=int, default=16)
    parser.add_argument("--angle-triplet-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
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
        if (saved.get("source_root") == str(root.resolve())
                and saved.get("version") == ARCHITECTURE):
            return set(saved["train"]), set(saved["validation"])
    groups: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for record in records:
        groups[record[2], int(record[4]["depression"])].append(record)
    train, validation = [], []
    for group, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda record: hashlib.sha256(
            f"{seed}:{group}:{record[0].relative_to(root)}".encode()).hexdigest())
        count = max(1, round(len(ordered) * fraction))
        validation.extend(
            str(item[0].relative_to(root)) for item in ordered[:count])
        train.extend(
            str(item[0].relative_to(root)) for item in ordered[count:])
    payload = {
        "version": ARCHITECTURE, "source_root": str(root.resolve()),
        "seed": seed, "validation_fraction": fraction,
        "train": sorted(train), "validation": sorted(validation)}
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return set(train), set(validation)


def configure_records(dataset: JointROIDataset, selected: set[str],
                      root: Path, epoch_size: int = 0) -> None:
    dataset.records = [
        record for record in dataset.records
        if str(record[0].relative_to(root)) in selected]
    if not dataset.records:
        raise RuntimeError("empty train/validation split")
    dataset.epoch_size = epoch_size or len(dataset.records)
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
    amplitude = ((image + 1) * .5).clamp(0, 1)
    batch = len(amplitude)
    gain = amplitude.new_empty(batch, 1, 1, 1).uniform_(.88, 1.12)
    gamma = amplitude.new_empty(batch, 1, 1, 1).uniform_(.92, 1.08)
    amplitude = amplitude.clamp_min(1e-5).pow(gamma) * gain
    amplitude = (
        amplitude
        + amplitude.new_empty(batch, 1, 1, 1).uniform_(0, .006)
        * torch.randn_like(amplitude))
    return amplitude.clamp(0, 1) * 2 - 1


def batch_geometry(batch: dict[str, object],
                   device: torch.device,
                   azimuth: torch.Tensor | None = None,
                   depression: torch.Tensor | None = None) -> tuple[
                       torch.Tensor, torch.Tensor, torch.Tensor]:
    meta = batch["meta"].to(device)
    if azimuth is None:
        azimuth = batch["azimuth"].to(device).float()
    if depression is None:
        depression = batch["depression"].to(device).float()
    geometry = target_geometry(azimuth, depression, meta[:, 3:8])
    if geometry.shape[1] != GEOMETRY_DIM:
        raise RuntimeError("unexpected target geometry dimension")
    return azimuth, depression, geometry


def low_structure_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    aligned = _align_translation(fake, real, max_shift=4)
    return (
        F.l1_loss(F.avg_pool2d(fake, 4), F.avg_pool2d(aligned, 4))
        + .5 * F.l1_loss(
            F.avg_pool2d(fake, 8), F.avg_pool2d(aligned, 8)))


def _correlation(image: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    source = image[..., :image.shape[-2] - dy, :image.shape[-1] - dx]
    shifted = image[..., dy:, dx:]
    source = source - source.mean((2, 3), keepdim=True)
    shifted = shifted - shifted.mean((2, 3), keepdim=True)
    return ((source * shifted).mean((2, 3))
            / (source.std((2, 3)) * shifted.std((2, 3)) + 1e-5))


def noise_statistics_loss(fake: torch.Tensor,
                          real: torch.Tensor) -> torch.Tensor:
    dimensions = (2, 3)
    moments = (
        F.l1_loss(fake.mean(dimensions), real.mean(dimensions))
        + F.l1_loss(fake.std(dimensions), real.std(dimensions))
        + .5 * F.l1_loss(
            fake.abs().mean(dimensions), real.abs().mean(dimensions)))
    spatial = sum(
        F.l1_loss(
            _correlation(fake, dy, dx), _correlation(real, dy, dx))
        for dy, dx in ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0))) / 5
    return moments + spatial


def spectrum_statistics_loss(fake: torch.Tensor,
                             real: torch.Tensor) -> torch.Tensor:
    def bands(image: torch.Tensor) -> torch.Tensor:
        spectrum = torch.log1p(torch.fft.fftshift(
            torch.fft.fft2(image.float(), norm="ortho"),
            dim=(-2, -1)).abs())
        side = spectrum.shape[-1]
        axis = torch.arange(side, device=image.device) - side // 2
        radius = torch.sqrt(axis[:, None].square() + axis[None].square())
        values = []
        for low, high in ((0, 4), (4, 8), (8, 16), (16, 32)):
            mask = ((radius >= low) & (radius < high)).to(spectrum.dtype)
            values.append(
                (spectrum * mask).sum((2, 3)) / mask.sum().clamp_min(1))
        return torch.stack(values, 1)
    return F.l1_loss(bands(fake), bands(real))


def feature_match(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return (
        F.l1_loss(fake.mean((2, 3)), real.detach().mean((2, 3)))
        + F.l1_loss(fake.std((2, 3)), real.detach().std((2, 3))))


def discriminator_hinge(real_score: torch.Tensor,
                        fake_score: torch.Tensor) -> torch.Tensor:
    return (
        F.relu(1 - real_score).mean()
        + F.relu(1 + fake_score).mean())


def pair_correlation(first: torch.Tensor,
                     second: torch.Tensor) -> torch.Tensor:
    first = first.flatten(1)
    second = second.flatten(1)
    first = first - first.mean(1, keepdim=True)
    second = second - second.mean(1, keepdim=True)
    return ((first * second).mean(1)
            / (first.std(1) * second.std(1) + 1e-6)).abs().mean()


def noise_edge_independence(log_noise: torch.Tensor,
                            clean: torch.Tensor) -> torch.Tensor:
    dx = F.pad(clean[..., 1:] - clean[..., :-1], (0, 1, 0, 0))
    dy = F.pad(clean[..., 1:, :] - clean[..., :-1, :], (0, 0, 0, 1))
    edge = (dx.square() + dy.square() + 1e-6).sqrt()
    return pair_correlation(log_noise.abs(), edge.detach())


def uniform_prediction_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(1)
    return (
        probabilities
        * (probabilities.clamp_min(1e-8).log() + math.log(logits.shape[1]))
    ).sum(1).mean()


def attention_locality_loss(weights: torch.Tensor,
                            source_angles: torch.Tensor,
                            target_azimuth: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
    relative = (
        target_azimuth[:, None] - source_angles + 180
    ).remainder(360) - 180
    distance = 1 - torch.cos(relative * (math.pi / 180))
    valid_weights = weights * mask
    valid_weights = valid_weights / valid_weights.sum(1, keepdim=True).clamp_min(1e-6)
    return (valid_weights * distance).sum(1).mean()


def geometry_validator_losses(
        validator: SARGeometryValidator, fake: torch.Tensor,
        real: torch.Tensor, labels: torch.Tensor, depression: torch.Tensor,
        azimuth: torch.Tensor) -> dict[str, torch.Tensor]:
    fake_output = validator((fake + 1) * .5)
    with torch.no_grad():
        real_output = validator((real + 1) * .5)
    depression_id = (
        depression.div(15).round().long() - 1
    ).clamp(0, len(DEPRESSION_VALUES) - 1)
    azimuth_bin = (azimuth / 5).round().long().remainder(72)
    radians = azimuth * (math.pi / 180)
    target_vector = torch.stack((radians.sin(), radians.cos()), 1)
    return {
        "validator_identity": F.cross_entropy(
            fake_output.identity_logits, labels, label_smoothing=.03),
        "validator_depression": F.cross_entropy(
            fake_output.depression_logits, depression_id),
        "validator_azimuth_bin": circular_soft_cross_entropy(
            fake_output.azimuth_logits, azimuth_bin),
        "validator_azimuth_vector": (
            1 - (fake_output.azimuth_vector * target_vector).sum(1)).mean(),
        "validator_feature": (
            1 - F.cosine_similarity(
                fake_output.features, real_output.features.detach(), dim=1)
        ).mean(),
        "validator_identity_accuracy": (
            fake_output.identity_logits.argmax(1) == labels).float().mean(),
        "validator_depression_accuracy": (
            fake_output.depression_logits.argmax(1)
            == depression_id).float().mean(),
        "validator_azimuth_cosine": (
            fake_output.azimuth_vector * target_vector).sum(1).mean(),
    }


def angle_triplet_loss(
        encoder_output, clean_generator: MultiViewDenoisedSARGenerator,
        observation: StochasticSARObservation,
        validator: SARGeometryValidator, source_angles: torch.Tensor,
        view_mask: torch.Tensor, azimuth: torch.Tensor,
        depression: torch.Tensor, acquisition: torch.Tensor,
        centre_clean: torch.Tensor, random_field: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
    neighbour_images = []
    angle_loss = centre_clean.new_zeros(())
    for offset in (-5.0, 5.0):
        neighbour_angle = (azimuth + offset).remainder(360)
        geometry = target_geometry(
            neighbour_angle, depression, acquisition)
        neighbour_clean, _ = clean_generator(
            encoder_output, source_angles, view_mask,
            neighbour_angle, depression, geometry)
        neighbour_full = observation(
            neighbour_clean, depression, random_field).observed
        output = validator((neighbour_full + 1) * .5)
        radians = neighbour_angle * (math.pi / 180)
        target = torch.stack((radians.sin(), radians.cos()), 1)
        angle_loss = angle_loss + (
            1 - (output.azimuth_vector * target).sum(1)).mean()
        neighbour_images.append(neighbour_clean)
    centre_low = F.avg_pool2d(centre_clean, 4)
    smoothness = F.smooth_l1_loss(
        F.avg_pool2d(neighbour_images[0], 4)
        + F.avg_pool2d(neighbour_images[1], 4),
        2 * centre_low)
    return .5 * angle_loss, smoothness


def phase_for_epoch(epoch: int, args: argparse.Namespace) -> str:
    if epoch <= args.clean_epochs:
        return "clean"
    if epoch <= args.clean_epochs + args.noise_epochs:
        return "noise"
    return "joint"


def save_preview(path: Path, rgb: torch.Tensor, real: torch.Tensor,
                 real_clean: torch.Tensor, real_noise: torch.Tensor,
                 fake_clean: torch.Tensor, fake_noise: torch.Tensor,
                 fake: torch.Tensor, second_fake: torch.Tensor) -> None:
    rows = []
    for index in range(min(6, len(fake))):
        rgb_panel = F.interpolate(
            rgb[index:index + 1], (64, 64), mode="bilinear",
            align_corners=False)[0]
        rgb_panel = (((rgb_panel.detach().cpu().clamp(-1, 1)
                       .permute(1, 2, 0).numpy()) + 1) * 127.5).astype(np.uint8)
        panels = [rgb_panel]
        for tensor in (
                real, real_clean, noise_view(real_noise), fake_clean,
                fake_noise, fake, second_fake):
            panel = (((tensor[index, 0].detach().cpu().clamp(-1, 1).numpy())
                      + 1) * 127.5).astype(np.uint8)
            panels.append(np.repeat(panel[..., None], 3, 2))
        rows.append(np.concatenate(panels, 1))
    Image.fromarray(np.concatenate(rows, 0), "RGB").save(path)


def checkpoint_state(
        epoch: int, encoder: nn.Module, clean: nn.Module,
        observation: nn.Module, discriminators: nn.Module, leakage: nn.Module,
        ema_encoder: nn.Module, ema_clean: nn.Module, ema_observation: nn.Module,
        clean_optimizer: torch.optim.Optimizer,
        noise_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        validation: dict[str, float], args: argparse.Namespace) -> dict:
    return {
        "architecture": ARCHITECTURE, "epoch": epoch,
        "phase": phase_for_epoch(epoch, args),
        "classes": list(SOC40_CLASSES),
        "encoder": encoder.state_dict(),
        "clean_generator": clean.state_dict(),
        "observation": observation.state_dict(),
        "discriminators": discriminators.state_dict(),
        "noise_leakage_classifier": leakage.state_dict(),
        "ema_encoder": ema_encoder.state_dict(),
        "ema_clean_generator": ema_clean.state_dict(),
        "ema_observation": ema_observation.state_dict(),
        "clean_optimizer": clean_optimizer.state_dict(),
        "noise_optimizer": noise_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "validation": validation,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
    }


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    total_epochs = (
        args.clean_epochs + args.noise_epochs + args.joint_epochs)

    train_data = JointROIDataset(
        args.rgb_root, args.sar_train_root, epoch_size=0, band="X",
        polarization="HH", depression="all", augment_rgb=True,
        source_view_mode="nearest", return_all_views=True)
    train_keys, validation_keys = split_records(
        train_data.records, args.sar_train_root,
        args.output / "split_manifest.json",
        args.validation_fraction, args.seed)
    configure_records(
        train_data, train_keys, args.sar_train_root, args.epoch_size)
    validation_data = JointROIDataset(
        args.rgb_root, args.sar_train_root, epoch_size=0, band="X",
        polarization="HH", depression="all", augment_rgb=False,
        source_view_mode="nearest", return_all_views=True)
    configure_records(
        validation_data, validation_keys, args.sar_train_root)
    train_loader = make_loader(
        train_data, args.batch_size, args.workers, True)
    validation_loader = make_loader(
        validation_data, args.batch_size, args.workers, False)

    validator_state = torch.load(
        args.geometry_validator_checkpoint, map_location=device,
        weights_only=False)
    if validator_state.get("architecture") != "sar_geometry_validator_v2":
        raise RuntimeError("incompatible geometry validator checkpoint")
    if validator_state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("geometry validator class order mismatch")
    validator = SARGeometryValidator(len(SOC40_CLASSES)).to(device)
    validator.load_state_dict(validator_state["model"])
    validator.eval()
    set_grad(validator, False)

    encoder = MultiViewRGBEncoder(len(SOC40_CLASSES)).to(device)
    clean_generator = MultiViewDenoisedSARGenerator().to(device)
    observation = StochasticSARObservation().to(device)
    discriminators = DualComponentDiscriminatorsV2(
        len(SOC40_CLASSES)).to(device)
    leakage = NoiseLeakageClassifier(len(SOC40_CLASSES)).to(device)
    for module in (
            encoder, clean_generator, observation, discriminators, leakage):
        module.apply(initialise)
    # Start from a neutral observation model: equal correlation bases, zero
    # skew, and mid-range bounded noise scales.  Random fields remain active.
    nn.init.zeros_(observation.parameter_net[-1].weight)
    nn.init.zeros_(observation.parameter_net[-1].bias)
    ema_encoder = copy.deepcopy(encoder).eval()
    ema_clean = copy.deepcopy(clean_generator).eval()
    ema_observation = copy.deepcopy(observation).eval()
    for module in (ema_encoder, ema_clean, ema_observation):
        set_grad(module, False)

    clean_optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(clean_generator.parameters()),
        lr=args.clean_lr, betas=(0, .99), weight_decay=1e-4)
    noise_optimizer = torch.optim.AdamW(
        observation.parameters(), lr=args.noise_lr,
        betas=(0, .99), weight_decay=1e-4)
    discriminator_optimizer = torch.optim.Adam(
        list(discriminators.parameters()) + list(leakage.parameters()),
        lr=args.discriminator_lr, betas=(0, .99))
    generator_scaler = torch.amp.GradScaler(
        device.type, enabled=use_amp)
    discriminator_scaler = torch.amp.GradScaler(
        device.type, enabled=use_amp)
    ce = nn.CrossEntropyLoss(label_smoothing=.03)
    start_epoch = 1

    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("architecture") != ARCHITECTURE:
            raise RuntimeError("incompatible --resume checkpoint")
        encoder.load_state_dict(saved["encoder"])
        clean_generator.load_state_dict(saved["clean_generator"])
        observation.load_state_dict(saved["observation"])
        discriminators.load_state_dict(saved["discriminators"])
        leakage.load_state_dict(saved["noise_leakage_classifier"])
        ema_encoder.load_state_dict(saved["ema_encoder"])
        ema_clean.load_state_dict(saved["ema_clean_generator"])
        ema_observation.load_state_dict(saved["ema_observation"])
        clean_optimizer.load_state_dict(saved["clean_optimizer"])
        noise_optimizer.load_state_dict(saved["noise_optimizer"])
        discriminator_optimizer.load_state_dict(
            saved["discriminator_optimizer"])
        start_epoch = int(saved["epoch"]) + 1

    parameter_counts = {
        "encoder": sum(parameter.numel() for parameter in encoder.parameters()),
        "clean_generator": sum(
            parameter.numel() for parameter in clean_generator.parameters()),
        "observation": sum(
            parameter.numel() for parameter in observation.parameters()),
        "discriminators": sum(
            parameter.numel() for parameter in discriminators.parameters()),
        "noise_leakage_classifier": sum(
            parameter.numel() for parameter in leakage.parameters()),
    }
    parameter_counts["total"] = sum(parameter_counts.values())
    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "architecture": ARCHITECTURE,
        "parameters": parameter_counts,
        "train": train_data.summary(),
        "validation": validation_data.summary(),
        "noise_policy": (
            "full-resolution random field; acquisition-only bounded parameters;"
            " no RGB/class input"),
        "checkpoint_policy": (
            "latest plus phase-complete/milestone checkpoints; final selection "
            "uses geometry, realism, and randomness audit"),
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(config, flush=True)

    columns = (
        "epoch", "phase", "generator", "discriminator",
        "rgb_identity", "clean_structure", "clean_statistics",
        "noise_statistics", "noise_seed_correlation",
        "noise_edge_independence", "noise_leakage_uniform",
        "full_statistics", "spectrum", "physics", "feature_match",
        "validator_identity", "validator_depression",
        "validator_azimuth_bin", "validator_azimuth_vector",
        "validator_feature", "angle_triplet", "angle_smoothness",
        "attention_locality", "fake_identity_accuracy",
        "fake_depression_accuracy", "fake_azimuth_cosine",
        "validation_clean_structure", "validation_noise_statistics",
        "validation_full_statistics", "validation_spectrum",
        "validation_noise_seed_correlation",
        "validation_noise_seed_l1", "validation_identity_accuracy",
        "validation_depression_accuracy", "validation_azimuth_cosine")
    history = args.output / "history.csv"
    if start_epoch == 1:
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(columns)

    for epoch in range(start_epoch, total_epochs + 1):
        phase = phase_for_epoch(epoch, args)
        train_clean = phase in {"clean", "joint"}
        train_noise = phase in {"noise", "joint"}
        set_grad(encoder, train_clean)
        set_grad(clean_generator, train_clean)
        set_grad(observation, train_noise)
        encoder.train(train_clean)
        clean_generator.train(train_clean)
        observation.train(train_noise)
        discriminators.train()
        leakage.train()
        if phase == "joint":
            for group in clean_optimizer.param_groups:
                group["lr"] = args.clean_lr * args.joint_clean_lr_scale

        totals: defaultdict[str, float] = defaultdict(float)
        steps = 0
        progress = tqdm(
            train_loader, desc=f"dual v2 {phase} {epoch}/{total_epochs}")
        for batch_index, batch in enumerate(progress):
            views = batch["rgb_views"].to(device)
            source_angles = batch["rgb_view_angles"].to(device)
            view_mask = batch["rgb_view_mask"].to(device)
            labels = batch["class_id"].to(device)
            real = augment_real_sar(batch["roi"].to(device))
            azimuth, depression, geometry = batch_geometry(batch, device)
            acquisition = batch["meta"].to(device)[:, 3:8]
            real_clean, real_log_noise = decompose_real_sar(real)
            real_residual = noise_view(real_log_noise)

            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                encoding = encoder(views, view_mask)
                fake_clean, attention = clean_generator(
                    encoding, source_angles, view_mask,
                    azimuth, depression, geometry)
                random_field = torch.randn(
                    len(real), observation.random_channels, 64, 64,
                    device=device, dtype=fake_clean.dtype)
                observed = observation(
                    fake_clean, depression, random_field)
                fake = observed.observed
                fake_residual = residual_view(fake_clean, fake)

            # Conditional critics and the separate residual leakage auditor.
            discriminator_optimizer.zero_grad(set_to_none=True)
            do_r1 = (
                phase != "clean" and args.r1_weight > 0
                and batch_index % args.r1_every == 0)
            real_for_full = real.detach().requires_grad_(do_r1)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                discriminator_loss = real.new_zeros(())
                wrong_azimuth = (azimuth + 90).remainder(360)
                wrong_depression = (
                    depression.div(15).round().long().remainder(4) + 1
                ).to(depression.dtype) * 15
                wrong_azimuth_geometry = target_geometry(
                    wrong_azimuth, depression, acquisition)
                wrong_depression_geometry = target_geometry(
                    azimuth, wrong_depression, acquisition)
                if phase in {"clean", "joint"}:
                    real_score, _ = discriminators.clean(
                        real_clean, labels, geometry)
                    fake_score, _ = discriminators.clean(
                        fake_clean.detach(), labels, geometry)
                    wrong_clean_azimuth, _ = discriminators.clean(
                        real_clean.detach(), labels,
                        wrong_azimuth_geometry)
                    wrong_clean_depression, _ = discriminators.clean(
                        real_clean.detach(), labels,
                        wrong_depression_geometry)
                    discriminator_loss = (
                        discriminator_loss
                        + discriminator_hinge(real_score, fake_score)
                        + .125 * F.relu(
                            1 + wrong_clean_azimuth).mean()
                        + .125 * F.relu(
                            1 + wrong_clean_depression).mean())
                if phase in {"noise", "joint"}:
                    real_noise_score, _ = discriminators.noise(
                        real_residual, depression)
                    fake_noise_score, _ = discriminators.noise(
                        fake_residual.detach(), depression)
                    wrong_noise_depression, _ = discriminators.noise(
                        real_residual.detach(), wrong_depression)
                    real_full_score, _ = discriminators.full(
                        real_for_full, labels, geometry)
                    fake_full_score, _ = discriminators.full(
                        fake.detach(), labels, geometry)
                    wrong_full_azimuth, _ = discriminators.full(
                        real.detach(), labels,
                        wrong_azimuth_geometry)
                    wrong_full_depression, _ = discriminators.full(
                        real.detach(), labels,
                        wrong_depression_geometry)
                    discriminator_loss = (
                        discriminator_loss
                        + discriminator_hinge(
                            real_noise_score, fake_noise_score)
                        + 1.5 * discriminator_hinge(
                            real_full_score, fake_full_score)
                        + .20 * F.relu(
                            1 + wrong_noise_depression).mean()
                        + .20 * F.relu(
                            1 + wrong_full_azimuth).mean()
                        + .20 * F.relu(
                            1 + wrong_full_depression).mean())
                    leakage_loss = .5 * (
                        F.cross_entropy(
                            leakage(real_residual.detach()), labels)
                        + F.cross_entropy(
                            leakage(fake_residual.detach()), labels))
                    discriminator_loss = (
                        discriminator_loss + .25 * leakage_loss)
                    if do_r1:
                        gradient = torch.autograd.grad(
                            real_full_score.sum(), real_for_full,
                            create_graph=True)[0]
                        r1 = gradient.flatten(1).square().sum(1).mean()
                        discriminator_loss = (
                            discriminator_loss
                            + .5 * args.r1_weight * args.r1_every * r1)
            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.unscale_(discriminator_optimizer)
            nn.utils.clip_grad_norm_(
                list(discriminators.parameters())
                + list(leakage.parameters()), 5)
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()

            set_grad(discriminators, False)
            set_grad(leakage, False)
            clean_optimizer.zero_grad(set_to_none=True)
            noise_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                zero = fake.new_zeros(())
                losses = {name: zero for name in (
                    "rgb_identity", "clean_structure", "clean_statistics",
                    "noise_statistics", "noise_seed_correlation",
                    "noise_edge_independence", "noise_leakage_uniform",
                    "full_statistics", "spectrum", "physics",
                    "feature_match", "validator_identity",
                    "validator_depression", "validator_azimuth_bin",
                    "validator_azimuth_vector", "validator_feature",
                    "angle_triplet", "angle_smoothness",
                    "attention_locality")}
                generator_loss = zero

                if phase in {"clean", "joint"}:
                    clean_score, clean_feature = discriminators.clean(
                        fake_clean, labels, geometry)
                    with torch.no_grad():
                        _, real_clean_feature = discriminators.clean(
                            real_clean, labels, geometry)
                    losses["rgb_identity"] = ce(
                        encoding.logits, labels)
                    losses["clean_structure"] = low_structure_loss(
                        fake_clean, real_clean)
                    losses["clean_statistics"] = sar_statistics_loss(
                        fake_clean, real_clean)
                    losses["attention_locality"] = attention_locality_loss(
                        attention, source_angles, azimuth, view_mask)
                    clean_matching = feature_match(
                        clean_feature, real_clean_feature)
                    losses["feature_match"] = clean_matching
                    generator_loss = (
                        -1.5 * clean_score.mean()
                        + 2 * losses["rgb_identity"]
                        + 4 * losses["clean_structure"]
                        + 2 * losses["clean_statistics"]
                        + 2 * clean_matching
                        + .25 * losses["attention_locality"])

                if phase in {"noise", "joint"}:
                    noise_score, noise_feature = discriminators.noise(
                        fake_residual, depression)
                    full_score, full_feature = discriminators.full(
                        fake, labels, geometry)
                    with torch.no_grad():
                        _, real_noise_feature = discriminators.noise(
                            real_residual, depression)
                        _, real_full_feature = discriminators.full(
                            real, labels, geometry)
                    second_field = torch.randn_like(random_field)
                    second_observed = observation(
                        fake_clean, depression, second_field)
                    second_residual = residual_view(
                        fake_clean, second_observed.observed)
                    losses["noise_statistics"] = noise_statistics_loss(
                        fake_residual, real_residual)
                    losses["noise_seed_correlation"] = pair_correlation(
                        observed.log_multiplicative,
                        second_observed.log_multiplicative)
                    losses["noise_edge_independence"] = noise_edge_independence(
                        observed.log_multiplicative, fake_clean)
                    losses["noise_leakage_uniform"] = uniform_prediction_loss(
                        leakage(fake_residual))
                    losses["full_statistics"] = sar_statistics_loss(
                        fake, real)
                    losses["spectrum"] = spectrum_statistics_loss(
                        fake, real)
                    losses["physics"] = sar_physics_prior_loss(
                        fake, real)
                    noise_matching = (
                        feature_match(noise_feature, real_noise_feature)
                        + feature_match(full_feature, real_full_feature)) / 2
                    losses["feature_match"] = (
                        losses["feature_match"] + noise_matching)
                    generator_loss = generator_loss + (
                        -noise_score.mean() - 2 * full_score.mean()
                        + 2 * losses["noise_statistics"]
                        + 3 * losses["full_statistics"]
                        + 1.5 * losses["spectrum"]
                        + .75 * losses["physics"]
                        + 2 * noise_matching
                        + 2 * losses["noise_seed_correlation"]
                        + .5 * losses["noise_edge_independence"]
                        + .5 * losses["noise_leakage_uniform"])
                else:
                    second_observed = observation(
                        fake_clean, depression, torch.randn_like(random_field))

                validation_losses = geometry_validator_losses(
                    validator, fake, real, labels, depression, azimuth)
                for name in (
                        "validator_identity", "validator_depression",
                        "validator_azimuth_bin", "validator_azimuth_vector",
                        "validator_feature"):
                    losses[name] = validation_losses[name]
                generator_loss = generator_loss + (
                    .10 * losses["validator_identity"]
                    + .50 * losses["validator_depression"]
                    + .25 * losses["validator_azimuth_bin"]
                    + 1.0 * losses["validator_azimuth_vector"]
                    + .75 * losses["validator_feature"])

                if (phase in {"clean", "joint"}
                        and batch_index % args.angle_triplet_every == 0):
                    triplet, smoothness = angle_triplet_loss(
                        encoding, clean_generator, observation, validator,
                        source_angles, view_mask, azimuth, depression,
                        acquisition, fake_clean, random_field)
                    losses["angle_triplet"] = triplet
                    losses["angle_smoothness"] = smoothness
                    generator_loss = (
                        generator_loss + .5 * triplet + .1 * smoothness)

            generator_scaler.scale(generator_loss).backward()
            if train_clean:
                generator_scaler.unscale_(clean_optimizer)
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters())
                    + list(clean_generator.parameters()), 5)
                generator_scaler.step(clean_optimizer)
            if train_noise:
                generator_scaler.unscale_(noise_optimizer)
                nn.utils.clip_grad_norm_(observation.parameters(), 5)
                generator_scaler.step(noise_optimizer)
            generator_scaler.update()
            set_grad(discriminators, True)
            set_grad(leakage, True)
            update_ema(ema_encoder, encoder, args.ema_decay)
            update_ema(ema_clean, clean_generator, args.ema_decay)
            update_ema(ema_observation, observation, args.ema_decay)

            values = {
                "generator": generator_loss,
                "discriminator": discriminator_loss,
                **losses,
                "fake_identity_accuracy":
                    validation_losses["validator_identity_accuracy"],
                "fake_depression_accuracy":
                    validation_losses["validator_depression_accuracy"],
                "fake_azimuth_cosine":
                    validation_losses["validator_azimuth_cosine"],
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            steps += 1
            progress.set_postfix(
                g=f"{float(generator_loss.detach()):.3f}",
                d=f"{float(discriminator_loss.detach()):.3f}",
                az=f"{float(validation_losses['validator_azimuth_cosine']):.3f}",
                corr=f"{float(losses['noise_seed_correlation']):.3f}")
            if (args.limit_train_batches
                    and batch_index + 1 >= args.limit_train_batches):
                break

        # Fixed-seed validation plus a second seed for stochastic audit.
        for module in (ema_encoder, ema_clean, ema_observation):
            module.eval()
        validator.eval()
        validation_totals: defaultdict[str, float] = defaultdict(float)
        validation_samples = 0
        preview = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(validation_loader):
                views = batch["rgb_views"].to(device)
                source_angles = batch["rgb_view_angles"].to(device)
                view_mask = batch["rgb_view_mask"].to(device)
                labels = batch["class_id"].to(device)
                real = batch["roi"].to(device)
                azimuth, depression, geometry = batch_geometry(batch, device)
                real_clean, real_log_noise = decompose_real_sar(real)
                real_residual = noise_view(real_log_noise)
                encoding = ema_encoder(views, view_mask)
                fake_clean, _ = ema_clean(
                    encoding, source_angles, view_mask,
                    azimuth, depression, geometry)
                first_generator = torch.Generator(device=device)
                first_generator.manual_seed(args.seed + batch_index)
                first_field = torch.randn(
                    len(real), ema_observation.random_channels, 64, 64,
                    device=device, generator=first_generator,
                    dtype=fake_clean.dtype)
                second_generator = torch.Generator(device=device)
                second_generator.manual_seed(
                    args.seed + 100000 + batch_index)
                second_field = torch.randn(
                    len(real), ema_observation.random_channels, 64, 64,
                    device=device, generator=second_generator,
                    dtype=fake_clean.dtype)
                first = ema_observation(
                    fake_clean, depression, first_field)
                second = ema_observation(
                    fake_clean, depression, second_field)
                fake_residual = residual_view(
                    fake_clean, first.observed)
                validator_losses = geometry_validator_losses(
                    validator, first.observed, real, labels,
                    depression, azimuth)
                size = len(real)
                metrics = {
                    "clean_structure": low_structure_loss(
                        fake_clean, real_clean),
                    "noise_statistics": noise_statistics_loss(
                        fake_residual, real_residual),
                    "full_statistics": sar_statistics_loss(
                        first.observed, real),
                    "spectrum": spectrum_statistics_loss(
                        first.observed, real),
                    "noise_seed_correlation": pair_correlation(
                        first.log_multiplicative,
                        second.log_multiplicative),
                    "noise_seed_l1": (
                        first.log_multiplicative
                        - second.log_multiplicative).abs().mean(),
                    "identity_accuracy":
                        validator_losses["validator_identity_accuracy"],
                    "depression_accuracy":
                        validator_losses["validator_depression_accuracy"],
                    "azimuth_cosine":
                        validator_losses["validator_azimuth_cosine"],
                }
                for name, value in metrics.items():
                    validation_totals[name] += float(value) * size
                validation_samples += size
                if preview is None:
                    preview = (
                        batch["rgb"].to(device), real, real_clean,
                        real_log_noise, fake_clean, fake_residual,
                        first.observed, second.observed)
                if (args.limit_validation_batches
                        and batch_index + 1
                        >= args.limit_validation_batches):
                    break
        validation = {
            name: value / max(validation_samples, 1)
            for name, value in validation_totals.items()}
        averages = {
            name: value / max(steps, 1)
            for name, value in totals.items()}
        row = (
            epoch, phase,
            *[averages.get(name, 0.0) for name in columns[2:26]],
            *[validation[name] for name in (
                "clean_structure", "noise_statistics", "full_statistics",
                "spectrum", "noise_seed_correlation", "noise_seed_l1",
                "identity_accuracy", "depression_accuracy",
                "azimuth_cosine")])
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        state = checkpoint_state(
            epoch, encoder, clean_generator, observation,
            discriminators, leakage, ema_encoder, ema_clean,
            ema_observation, clean_optimizer, noise_optimizer,
            discriminator_optimizer, validation, args)
        torch.save(state, args.output / "latest.pt")
        if (epoch % 10 == 0 or epoch in {
                args.clean_epochs,
                args.clean_epochs + args.noise_epochs,
                total_epochs}):
            torch.save(state, args.output / f"epoch_{epoch:03d}.pt")
        if epoch == 1 or epoch % 5 == 0:
            assert preview is not None
            save_preview(
                args.output / f"validation_{epoch:03d}.png", *preview)
        print(dict(zip(columns, row)), flush=True)


if __name__ == "__main__":
    main()

"""Real-domain fine-tuning of the visually strong continuous spatial v1 GAN.

v1.1 deliberately preserves the warm-started RGB encoder, spatial generator,
and stochastic SAR observation renderer.  It replaces the teacher-dominated
objective with native/high-pass/Fourier adversaries, weak teacher supervision,
conditional real-SAR contrast, and validation-only checkpoint selection.
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

from joint_data import JointROIDataset
from joint_models import (RGBIdentityEncoder, SpatialROIGenerator, _align_translation,
                          sar_physics_prior_loss, sar_statistics_loss)
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES
from train_continuous_spatial_roi_gan import (DEPRESSION_TO_ID, rotate_target_azimuth,
                                              target_condition)
from v5_hybrid_sar_gan import (CalibratedMultiDomainDiscriminator,
                               raw_highpass_view, raw_spectrum_view)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="continuous spatial v1.2 conservative real-domain fine-tuning")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--epoch-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--generator-lr", type=float, default=1e-5)
    parser.add_argument("--identity-lr", type=float, default=5e-6)
    parser.add_argument("--discriminator-lr", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--r1-weight", type=float, default=1.0)
    parser.add_argument("--r1-every", type=int, default=16)
    parser.add_argument("--discriminator-warmup-epochs", type=int, default=3)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-validation-batches", type=int, default=0)
    return parser.parse_args()


def make_loader(dataset: JointROIDataset, batch_size: int, workers: int,
                shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      drop_last=shuffle)


def split_records(records: list[tuple], root: Path, output: Path,
                  fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if output.is_file():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved.get("source_root") == str(root.resolve()):
            return set(saved["train"]), set(saved["validation"])
    groups: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    for record in records:
        path, _, class_name, _, meta, _ = record
        groups[class_name, int(meta["depression"])].append(record)
    train, validation = [], []
    for group, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda record: hashlib.sha256(
            f"{seed}:{group}:{record[0].relative_to(root)}".encode()).hexdigest())
        count = max(1, round(len(ordered) * fraction))
        validation.extend(str(record[0].relative_to(root)) for record in ordered[:count])
        train.extend(str(record[0].relative_to(root)) for record in ordered[count:])
    payload = {"version": "continuous-v1.1", "source_root": str(root.resolve()),
               "seed": seed, "validation_fraction": fraction,
               "train": sorted(train), "validation": sorted(validation)}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return set(train), set(validation)


def configure_records(dataset: JointROIDataset, requested: set[str],
                      root: Path, epoch_size: int = 0) -> None:
    dataset.records = [
        record for record in dataset.records
        if str(record[0].relative_to(root)) in requested]
    dataset.epoch_size = epoch_size or len(dataset.records)
    dataset.random_epoch = 0 < epoch_size < len(dataset.records)
    if not dataset.records:
        raise RuntimeError("empty continuous v1.1 split")


def set_grad(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def update_ema(ema: nn.Module, source: nn.Module, decay: float) -> None:
    for target, current in zip(ema.parameters(), source.parameters()):
        target.lerp_(current, 1 - decay)
    for target, current in zip(ema.buffers(), source.buffers()):
        target.copy_(current)


def teacher_view(fake: torch.Tensor) -> torch.Tensor:
    image = (fake + 1) * .5
    gain = image.new_empty(len(image), 1, 1, 1).uniform_(.94, 1.06)
    bias = image.new_empty(len(image), 1, 1, 1).uniform_(-.02, .02)
    return (image * gain + bias).clamp(0, 1)


def azimuth_bin(value: torch.Tensor) -> torch.Tensor:
    return ((value + 15) % 360) // 30


def prepare_teacher_prototypes(judge: SARClassifier64, dataset: JointROIDataset,
                               device: torch.device, workers: int) -> torch.Tensor:
    sums = torch.zeros(40, 4, 12, judge.feature_dim, device=device)
    counts = torch.zeros(40, 4, 12, device=device)
    with torch.inference_mode():
        for batch in tqdm(make_loader(dataset, 256, workers, False),
                          desc="continuous v1.1 real teacher prototypes"):
            real = batch["roi"].to(device)
            _, features = judge((real + 1) * .5, return_features=True)
            features = F.normalize(features, dim=1)
            labels = batch["class_id"].to(device)
            depression = torch.tensor(
                [DEPRESSION_TO_ID[int(value)] for value in batch["depression"].tolist()],
                device=device)
            az_bin = azimuth_bin(batch["azimuth"].to(device))
            sums.index_put_((labels, depression, az_bin), features, accumulate=True)
            counts.index_put_((labels, depression, az_bin),
                              torch.ones_like(labels, dtype=torch.float), accumulate=True)
    fallback = sums.sum(2) / counts.sum(2).clamp_min(1)[..., None]
    centres = sums / counts[..., None].clamp_min(1)
    missing = counts == 0
    centres[missing] = fallback[:, :, None, :].expand_as(centres)[missing]
    return F.normalize(centres, dim=3)


def centre_extent(image: torch.Tensor) -> torch.Tensor:
    amplitude = ((image + 1) * .5).clamp_min(0)
    weight = F.relu(amplitude - .55 * amplitude.mean((2, 3), keepdim=True))
    weight = weight / weight.sum((2, 3), keepdim=True).clamp_min(1e-5)
    side = image.shape[-1]
    axis = torch.linspace(-1, 1, side, device=image.device, dtype=image.dtype)
    yy, xx = axis[None, None, :, None], axis[None, None, None, :]
    cx, cy = (weight * xx).sum((2, 3)), (weight * yy).sum((2, 3))
    sx = (weight * (xx - cx[:, :, None, None]).square()).sum((2, 3)).sqrt()
    sy = (weight * (yy - cy[:, :, None, None]).square()).sum((2, 3)).sqrt()
    return torch.cat((cx, cy, sx, sy), 1)


def low_structure_loss(clean: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    aligned = _align_translation(clean, real)
    return (F.l1_loss(F.avg_pool2d(clean, 4), F.avg_pool2d(aligned, 4))
            + .5 * F.l1_loss(F.avg_pool2d(clean, 8), F.avg_pool2d(aligned, 8))
            + .5 * F.l1_loss(centre_extent(clean), centre_extent(aligned)))


def calibrate_domain_views(dataset: JointROIDataset, device: torch.device,
                           workers: int) -> dict[str, torch.Tensor]:
    """Fixed normalisation moments from real SAR only."""
    sums = {"highpass": torch.zeros((), device=device), "spectrum": torch.zeros((), device=device)}
    squares = {key: torch.zeros((), device=device) for key in sums}
    count = 0
    with torch.inference_mode():
        for batch in tqdm(make_loader(dataset, 256, workers, False), desc="v1.2 real-domain calibration"):
            real = batch["roi"].to(device)
            for key, value in (("highpass", raw_highpass_view(real)),
                               ("spectrum", raw_spectrum_view(real))):
                value = value.float()
                sums[key] += value.sum(); squares[key] += value.square().sum()
            count += real.numel()
    answer = {}
    for key in sums:
        mean = sums[key] / count
        answer[f"{key}_mean"] = mean
        answer[f"{key}_std"] = (squares[key] / count - mean.square()).clamp_min(1e-8).sqrt()
    return answer


def spectral_loss(fake: torch.Tensor, real: torch.Tensor,
                  spectrum_mean: torch.Tensor, spectrum_std: torch.Tensor) -> torch.Tensor:
    fake_spectrum = (raw_spectrum_view(fake) - spectrum_mean) / spectrum_std
    real_spectrum = (raw_spectrum_view(real) - spectrum_mean) / spectrum_std
    return F.l1_loss(F.avg_pool2d(fake_spectrum, 4), F.avg_pool2d(real_spectrum, 4))


def _sobel_magnitude(amplitude: torch.Tensor) -> torch.Tensor:
    kernel = amplitude.new_tensor(((-1., 0., 1.), (-2., 0., 2.), (-1., 0., 1.))).reshape(1, 1, 3, 3) / 8
    gx = F.conv2d(amplitude, kernel, padding=1)
    gy = F.conv2d(amplitude, kernel.transpose(2, 3), padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-6)


def dark_artifact_terms(fake: torch.Tensor, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match dark-tail distribution and reject sharp boundaries in dark regions."""
    fake_a, real_a = (fake + 1) * .5, (real + 1) * .5
    thresholds = fake.new_tensor((.01, .02, .04, .08, .16))[None, :, None, None, None]
    temperature = .012
    fake_cdf = torch.sigmoid((thresholds - fake_a[:, None]) / temperature).mean((2, 3, 4))
    real_cdf = torch.sigmoid((thresholds - real_a[:, None]) / temperature).mean((2, 3, 4))
    cdf_loss = F.l1_loss(fake_cdf, real_cdf)
    dark_fake = torch.sigmoid((.10 - fake_a) / .02)
    dark_real = torch.sigmoid((.10 - real_a) / .02)
    contour_fake = (_sobel_magnitude(fake_a) * dark_fake).mean((1, 2, 3))
    contour_real = (_sobel_magnitude(real_a) * dark_real).mean((1, 2, 3))
    contour_loss = F.l1_loss(contour_fake, contour_real)
    saturation = F.l1_loss(fake_cdf[:, 0], real_cdf[:, 0])
    return cdf_loss, contour_loss, saturation


def dark_contour_score(image: torch.Tensor) -> torch.Tensor:
    amplitude = (image + 1) * .5
    return (_sobel_magnitude(amplitude) * torch.sigmoid((.10 - amplitude) / .02)).mean()


def saturation_fraction(image: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid((.01 - (image + 1) * .5) / .004).mean()


def reference_anchor(clean: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve v1 low-frequency appearance and its natural dark background."""
    low = F.l1_loss(F.avg_pool2d(clean, 4), F.avg_pool2d(reference, 4))
    reference_amplitude = (reference + 1) * .5
    background = torch.sigmoid((.15 - reference_amplitude) / .025)
    background_loss = ((clean - reference).abs() * background).sum() / background.sum().clamp_min(1)
    return low, background_loss


def feature_match(fake_features: tuple[torch.Tensor, ...],
                  real_features: tuple[torch.Tensor, ...]) -> torch.Tensor:
    total = fake_features[0].new_zeros(())
    for fake, real in zip(fake_features, real_features):
        total += F.l1_loss(fake.mean((2, 3)), real.detach().mean((2, 3)))
        total += F.l1_loss(fake.std((2, 3)), real.detach().std((2, 3)))
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
        for image in (real, clean, fake):
            panel = ((image[index, 0].detach().cpu().clamp(-1, 1).numpy()
                      + 1) * 127.5).astype(np.uint8)
            panels.append(np.repeat(panel[..., None], 3, 2))
        rows.append(np.concatenate(panels, 1))
    Image.fromarray(np.concatenate(rows, 0), "RGB").save(path)


def main() -> None:
    args = arguments()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); amp = device.type == "cuda"
    full = JointROIDataset(args.rgb_root, args.sar_train_root, epoch_size=0,
                           band="X", polarization="HH", depression="all",
                           augment_rgb=True, source_view_mode="mixed")
    train_keys, validation_keys = split_records(
        full.records, args.sar_train_root, args.output / "split_manifest.json",
        args.validation_fraction, args.seed)
    configure_records(full, train_keys, args.sar_train_root, args.epoch_size)
    validation = JointROIDataset(
        args.rgb_root, args.sar_train_root, epoch_size=0, band="X",
        polarization="HH", depression="all", augment_rgb=False,
        source_view_mode="nearest")
    configure_records(validation, validation_keys, args.sar_train_root)
    train_loader = make_loader(full, args.batch_size, args.workers, True)
    validation_loader = make_loader(validation, args.batch_size, args.workers, False)

    teacher_state = torch.load(
        args.native_classifier_checkpoint, map_location=device, weights_only=False)
    judge = SARClassifier64(40).to(device)
    judge.load_state_dict(teacher_state["model"]); judge.eval(); set_grad(judge, False)
    prototype_dataset = copy.copy(full)
    prototype_dataset.epoch_size = len(prototype_dataset.records)
    prototype_dataset.random_epoch = False
    prototypes = prepare_teacher_prototypes(judge, prototype_dataset, device, args.workers)
    calibration = calibrate_domain_views(prototype_dataset, device, args.workers)
    torch.save({key: value.cpu() for key, value in calibration.items()},
               args.output / "real_domain_calibration.pt")

    initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
    if initial.get("architecture") != "continuous_spatial_v1":
        raise RuntimeError("initial checkpoint must be continuous_spatial_v1")
    encoder = RGBIdentityEncoder(40).to(device)
    generator = SpatialROIGenerator(meta_dim=12).to(device)
    encoder.load_state_dict(initial["identity_encoder"])
    generator.load_state_dict(initial["generator"])
    # Frozen original-v1 reference: it anchors the visually successful
    # low-frequency SAR appearance while new losses correct domain artefacts.
    reference_encoder = copy.deepcopy(encoder).eval()
    reference_generator = copy.deepcopy(generator).eval()
    set_grad(reference_encoder, False); set_grad(reference_generator, False)
    discriminator = CalibratedMultiDomainDiscriminator(
        **calibration, classes=40, geometry_dim=12).to(device)
    ema_encoder, ema_generator = copy.deepcopy(encoder).eval(), copy.deepcopy(generator).eval()
    set_grad(ema_encoder, False); set_grad(ema_generator, False)

    generator_opt = torch.optim.AdamW((
        {"params": encoder.parameters(), "lr": args.identity_lr},
        {"params": generator.parameters(), "lr": args.generator_lr},
    ), betas=(0., .99), weight_decay=1e-4)
    discriminator_opt = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0., .99))
    generator_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    discriminator_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ce = nn.CrossEntropyLoss(label_smoothing=.03)
    start_epoch, best_quality = 1, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(saved["identity_encoder"]); generator.load_state_dict(saved["generator"])
        discriminator.load_state_dict(saved["discriminator"])
        ema_encoder.load_state_dict(saved["ema_identity_encoder"])
        ema_generator.load_state_dict(saved["ema_generator"])
        generator_opt.load_state_dict(saved["generator_optimizer"])
        discriminator_opt.load_state_dict(saved["discriminator_optimizer"])
        start_epoch, best_quality = int(saved["epoch"]) + 1, float(saved["best_quality"])

    columns = ("epoch", "generator", "adversarial", "rgb_identity", "cross_view",
               "structure", "statistics", "physics", "spectrum", "teacher_class",
               "teacher_contrast", "feature_match", "angle", "reference_low",
               "reference_background", "dark_cdf", "dark_contour", "saturation",
               "discriminator", "r1",
               "rgb_accuracy", "fake_teacher_accuracy", "validation_quality",
               "validation_structure", "validation_statistics", "validation_physics",
               "validation_spectrum", "validation_dark_cdf", "validation_dark_contour",
               "validation_saturation_gap", "validation_reference_dark_contour",
               "validation_fake_saturation", "validation_real_saturation",
               "validation_teacher_accuracy")
    history = args.output / "history.csv"
    if start_epoch == 1:
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(columns)
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config.update({"train_records": len(full.records),
                       "validation_records": len(validation.records),
                       "policy": "conservative v1.2: original-v1 visual anchor; calibrated multi-domain D; dark-contour validation gates; test untouched"})
        (args.output / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train(); generator.train(); discriminator.train()
        totals = torch.zeros(22, dtype=torch.float64)
        progress = tqdm(train_loader, desc=f"continuous v1.1 {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            rgb, rgb_alt = batch["rgb"].to(device), batch["rgb_alt"].to(device)
            real = batch["roi"].to(device); labels = batch["class_id"].to(device)
            geometry = target_condition(
                batch["meta"].to(device), batch["rgb_angle"].to(device))
            depression = torch.tensor(
                [DEPRESSION_TO_ID[int(value)] for value in batch["depression"].tolist()],
                device=device)
            az_bin = azimuth_bin(batch["azimuth"].to(device))
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                identity, rgb_logits, pyramid = encoder(rgb, return_pyramid=True)
                alt_identity, alt_logits = encoder(rgb_alt)
                clean = generator(identity, geometry, pyramid, apply_speckle=False)
                fake = generator.apply_speckle(clean)

            discriminator_opt.zero_grad(set_to_none=True)
            do_r1 = args.r1_weight > 0 and batch_index % args.r1_every == 0
            real_for_d = real.detach().requires_grad_(do_r1)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                real_scores, _ = discriminator(real_for_d, labels, geometry)
                fake_scores, _ = discriminator(fake.detach(), labels, geometry)
                wrong_label, _ = discriminator.spatial(real, labels.roll(1), geometry)
                wrong_geometry, _ = discriminator.spatial(real, labels, geometry.roll(1, 0))
                discriminator_loss = sum(
                    F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
                    for real_score, fake_score in zip(real_scores, fake_scores)) / 3
                discriminator_loss += .25 * (
                    F.relu(1 + wrong_label).mean() + F.relu(1 + wrong_geometry).mean())
                r1 = real.new_zeros(())
                if do_r1:
                    gradient = torch.autograd.grad(
                        sum(score.sum() for score in real_scores),
                        real_for_d, create_graph=True)[0]
                    r1 = gradient.flatten(1).square().sum(1).mean()
                    discriminator_loss += .5 * args.r1_weight * args.r1_every * r1
            discriminator_scaler.scale(discriminator_loss).backward()
            discriminator_scaler.unscale_(discriminator_opt)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.)
            discriminator_scaler.step(discriminator_opt); discriminator_scaler.update()

            zero = real.new_zeros(())
            rgb_accuracy = .5 * ((rgb_logits.argmax(1) == labels).float().mean()
                                  + (alt_logits.argmax(1) == labels).float().mean())
            if epoch <= args.discriminator_warmup_epochs:
                # New D branches first learn the original-v1/real distinction;
                # G cannot immediately exploit a randomly initialised D.
                values = (zero, zero, zero, zero, zero, zero, zero, zero, zero,
                          zero, zero, zero, zero, zero, zero, zero, zero,
                          discriminator_loss, r1, rgb_accuracy, zero)
            else:
                set_grad(discriminator, False); generator_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    fake_scores, fake_features = discriminator(fake, labels, geometry)
                    with torch.no_grad():
                        _, real_features = discriminator(real, labels, geometry)
                        reference_identity, _, reference_pyramid = reference_encoder(rgb, return_pyramid=True)
                        reference_clean = reference_generator(
                            reference_identity, geometry, reference_pyramid, apply_speckle=False)
                    adversarial = -sum(score.mean() for score in fake_scores) / 3
                    rgb_identity = .5 * (ce(rgb_logits, labels) + ce(alt_logits, labels))
                    cross_view = 1 - (
                        F.normalize(identity, dim=1) * F.normalize(alt_identity, dim=1)).sum(1).mean()
                    structure = low_structure_loss(clean, real)
                    statistics = sar_statistics_loss(fake, real)
                    physics = sar_physics_prior_loss(fake, real)
                    spectrum = spectral_loss(fake, real, calibration["spectrum_mean"], calibration["spectrum_std"])
                    teacher_logits, teacher_features = judge(teacher_view(fake), return_features=True)
                    teacher_features = F.normalize(teacher_features, dim=1)
                    positive = (teacher_features * prototypes[labels, depression, az_bin]).sum(1)
                    negative = (
                        teacher_features * prototypes[labels.roll(1), depression, az_bin]).sum(1)
                    teacher_class = ce(teacher_logits, labels)
                    teacher_contrast = (1 - positive).mean() + F.relu(
                        .15 - positive + negative).mean()
                    matching = feature_match(fake_features, real_features)
                    neighbour = generator(
                        identity, rotate_target_azimuth(geometry), pyramid, apply_speckle=False)
                    angle = F.l1_loss(F.avg_pool2d(clean, 4), F.avg_pool2d(neighbour, 4))
                    reference_low, reference_background = reference_anchor(clean, reference_clean)
                    dark_cdf, dark_contour, saturation = dark_artifact_terms(fake, real)
                    generator_loss = (
                        2 * adversarial + 1.5 * rgb_identity + .75 * cross_view
                        + 2 * structure + 3 * statistics + .25 * physics + 2 * spectrum
                        + .02 * teacher_class + .2 * teacher_contrast + 4 * matching + .2 * angle
                        + 5 * reference_low + 3 * reference_background
                        + 3 * dark_cdf + 2 * dark_contour + 2 * saturation)
                generator_scaler.scale(generator_loss).backward()
                generator_scaler.unscale_(generator_opt)
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(generator.parameters()), 5.)
                generator_scaler.step(generator_opt); generator_scaler.update()
                set_grad(discriminator, True)
                update_ema(ema_encoder, encoder, args.ema_decay)
                update_ema(ema_generator, generator, args.ema_decay)
                values = (generator_loss, adversarial, rgb_identity, cross_view, structure,
                          statistics, physics, spectrum, teacher_class, teacher_contrast,
                          matching, angle, reference_low, reference_background, dark_cdf,
                          dark_contour, saturation, discriminator_loss, r1, rgb_accuracy,
                          (teacher_logits.argmax(1) == labels).float().mean())
            totals[:21] += torch.tensor(
                [value.detach().item() for value in values], dtype=torch.float64)
            totals[21] += 1
            if args.limit_train_batches and batch_index + 1 >= args.limit_train_batches:
                break

        ema_encoder.eval(); ema_generator.eval()
        val_total = val_structure = val_statistics = val_physics = val_spectrum = 0.
        val_dark_cdf = val_dark_contour = val_saturation_gap = 0.
        val_reference_dark_contour = val_fake_saturation = val_real_saturation = 0.
        val_correct = 0; preview = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(validation_loader):
                rgb, real = batch["rgb"].to(device), batch["roi"].to(device)
                labels = batch["class_id"].to(device)
                geometry = target_condition(
                    batch["meta"].to(device), batch["rgb_angle"].to(device))
                identity, _, pyramid = ema_encoder(rgb, return_pyramid=True)
                clean = ema_generator(identity, geometry, pyramid, apply_speckle=False)
                with torch.no_grad():
                    reference_identity, _, reference_pyramid = reference_encoder(rgb, return_pyramid=True)
                    reference_clean = reference_generator(
                        reference_identity, geometry, reference_pyramid, apply_speckle=False)
                with torch.random.fork_rng(devices=[device]):
                    torch.manual_seed(args.seed + batch_index)
                    fake = ema_generator.apply_speckle(clean)
                size = len(real)
                val_structure += low_structure_loss(clean, real).item() * size
                val_statistics += sar_statistics_loss(fake, real).item() * size
                val_physics += sar_physics_prior_loss(fake, real).item() * size
                val_spectrum += spectral_loss(
                    fake, real, calibration["spectrum_mean"], calibration["spectrum_std"]).item() * size
                dark_cdf, dark_contour, saturation_gap = dark_artifact_terms(fake, real)
                val_dark_cdf += dark_cdf.item() * size
                val_dark_contour += dark_contour.item() * size
                val_saturation_gap += saturation_gap.item() * size
                val_reference_dark_contour += dark_contour_score(reference_clean).item() * size
                val_fake_saturation += saturation_fraction(clean).item() * size
                val_real_saturation += saturation_fraction(real).item() * size
                val_correct += (judge((fake + 1) * .5).argmax(1) == labels).sum().item()
                val_total += size
                if preview is None:
                    preview = (rgb, real, clean, fake)
                if args.limit_validation_batches and batch_index + 1 >= args.limit_validation_batches:
                    break
        val_structure /= val_total; val_statistics /= val_total
        val_physics /= val_total; val_spectrum /= val_total
        val_dark_cdf /= val_total; val_dark_contour /= val_total
        val_saturation_gap /= val_total; val_reference_dark_contour /= val_total
        val_fake_saturation /= val_total; val_real_saturation /= val_total
        val_accuracy = val_correct / val_total
        quality = (val_structure + val_statistics + .25 * val_physics + val_spectrum
                   + 2 * val_dark_cdf + 2 * val_dark_contour + val_saturation_gap)
        averages = (totals[:21] / totals[21]).tolist()
        row = (epoch, *averages, quality, val_structure, val_statistics,
               val_physics, val_spectrum, val_dark_cdf, val_dark_contour,
               val_saturation_gap, val_reference_dark_contour, val_fake_saturation,
               val_real_saturation, val_accuracy)
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        state = {
            "architecture": "continuous_spatial_v1_2", "epoch": epoch,
            "classes": list(SOC40_CLASSES), "identity_encoder": encoder.state_dict(),
            "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
            "ema_identity_encoder": ema_encoder.state_dict(),
            "ema_generator": ema_generator.state_dict(),
            "generator_optimizer": generator_opt.state_dict(),
            "discriminator_optimizer": discriminator_opt.state_dict(),
            "best_quality": min(best_quality, quality), "validation_quality": quality,
            "validation_teacher_accuracy": val_accuracy,
            "validation_dark_cdf": val_dark_cdf,
            "validation_dark_contour": val_dark_contour,
            "validation_saturation_gap": val_saturation_gap,
            "validation_reference_dark_contour": val_reference_dark_contour,
            "validation_fake_saturation": val_fake_saturation,
            "validation_real_saturation": val_real_saturation,
            "initial_checkpoint": str(args.initial_checkpoint),
            "split_manifest": str(args.output / "split_manifest.json"),
        }
        torch.save(state, args.output / "latest.pt")
        visual_gate = (
            val_dark_contour <= 1.05 * val_reference_dark_contour
            and val_fake_saturation <= val_real_saturation + .02)
        if ((quality < best_quality and visual_gate)
                or (epoch == start_epoch and not (args.output / "best.pt").is_file())):
            best_quality = quality; state["best_quality"] = quality
            torch.save(state, args.output / "best.pt")
        if epoch == 1 or epoch % 5 == 0:
            assert preview is not None
            save_preview(args.output / f"validation_{epoch:03d}.png", *preview)
        print(dict(zip(columns, row)), flush=True)


if __name__ == "__main__":
    main()

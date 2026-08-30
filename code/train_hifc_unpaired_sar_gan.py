"""Train the HiFC-inspired unpaired RGB-to-SAR model.

This is a separate experiment from all V1 trainers.  It keeps the two ideas
from HiFC-GAN (local texture contrast and deep semantic feature mapping), but
removes the optical-to-SAR pixel-registration assumption.  RGB and SAR are
matched by vehicle class only; acquisition metadata is supplied as a target
condition.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dual_component_sar_gan import LargeRGBIdentityEncoder
from hifc_unpaired_sar_gan import (
    HIFC_ARCHITECTURE, HIFCConditionedDiscriminator, HIFCUnpairedGenerator,
    condition_from_batch, discriminator_hinge, geometry_auxiliary_loss,
    initialise_hifc, local_texture_contrast_loss,
    parameter_count, rgb_identity_loss, semantic_feature_mapping_loss,
    set_grad, update_ema)
from joint_data import JointROIDataset
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HiFC-inspired unpaired RGB-to-SAR GAN (no pixel alignment)")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band", choices=("all", "X", "KU"), default="all",
                        help="train conditions; use all to learn band conditioning")
    parser.add_argument("--polarization", choices=("all", "HH", "HV", "VH", "VV"),
                        default="all", help="train conditions; use all to learn polarization")
    parser.add_argument("--depression", choices=("all", "15", "30", "45", "60"),
                        default="all")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--epoch-size", type=int, default=24000,
                        help="random weakly-unpaired samples per training epoch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--generator-lr", type=float, default=1.5e-4)
    parser.add_argument("--identity-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=1e-4)
    parser.add_argument("--adversarial-warmup-epochs", type=int, default=1)
    parser.add_argument("--r1-weight", type=float, default=.25)
    parser.add_argument("--r1-every", type=int, default=16)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--rgb-identity-weight", type=float, default=1.0)
    parser.add_argument("--ltc-weight", type=float, default=2.0)
    parser.add_argument("--sfm-weight", type=float, default=2.0)
    parser.add_argument("--geometry-weight", type=float, default=.30)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-validation-batches", type=int, default=0)
    return parser.parse_args()


def autocast_context(device: torch.device, enabled: bool):
    return (torch.amp.autocast(device_type=device.type, enabled=True)
            if enabled else nullcontext())


def make_loader(dataset: JointROIDataset, batch_size: int, workers: int,
                shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        drop_last=shuffle)


def split_records(records: list[tuple], root: Path, manifest: Path,
                  fraction: float, seed: int,
                  filters: dict[str, str]) -> tuple[set[str], set[str]]:
    """Make a deterministic class/condition split without pairing RGB pixels."""
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if (saved.get("source_root") == str(root.resolve())
                and saved.get("filters") == filters):
            return set(saved["train"]), set(saved["validation"])
        # A manifest made by an earlier invocation may not contain the filter;
        # retaining it would silently mix all-condition and X/HH experiments.
    groups: dict[tuple[str, str, str, int], list[tuple]] = defaultdict(list)
    for record in records:
        meta = record[4]
        groups[(record[2], str(meta["band"]), str(meta["pol"]),
                int(meta["depression"]))].append(record)
    train, validation = [], []
    for group, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda record: hashlib.sha256(
            f"{seed}:{group}:{record[0].relative_to(root)}".encode()).hexdigest())
        count = max(1, round(len(ordered) * fraction)) if fraction else 0
        validation.extend(str(item[0].relative_to(root)) for item in ordered[:count])
        train.extend(str(item[0].relative_to(root)) for item in ordered[count:])
    payload = {
        "version": HIFC_ARCHITECTURE, "source_root": str(root.resolve()),
        "seed": seed, "validation_fraction": fraction,
        "filters": filters,
        "train": sorted(train), "validation": sorted(validation)}
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return set(train), set(validation)


def configure_records(dataset: JointROIDataset, selected: set[str], root: Path,
                      epoch_size: int = 0) -> None:
    dataset.records = [
        record for record in dataset.records
        if str(record[0].relative_to(root)) in selected]
    if not dataset.records:
        raise RuntimeError("empty HiFC train/validation split")
    dataset.epoch_size = epoch_size or len(dataset.records)
    dataset.random_epoch = bool(epoch_size)


def differentiable_augment(image: torch.Tensor) -> torch.Tensor:
    """Shared radiometric augmentation; no domain-specific spatial warp."""
    batch = len(image)
    amplitude = ((image + 1.0) * .5).clamp(0, 1)
    gain = amplitude.new_empty(batch, 1, 1, 1).uniform_(.90, 1.10)
    bias = amplitude.new_empty(batch, 1, 1, 1).uniform_(-.03, .03)
    amplitude = amplitude * gain + bias
    # A tiny multiplicative perturbation models acquisition variation without
    # creating an RGB/SAR coordinate correspondence.
    amplitude = amplitude * torch.exp(torch.randn_like(amplitude) * .015)
    return amplitude.clamp(0, 1) * 2.0 - 1.0


def save_preview(path: Path, rgb: torch.Tensor, real: torch.Tensor,
                 fake: torch.Tensor, clean: torch.Tensor) -> None:
    rows = []
    for index in range(min(8, len(fake))):
        rgb_panel = F.interpolate(rgb[index:index + 1], (64, 64),
                                  mode="bilinear", align_corners=False)[0]
        rgb_panel = (((rgb_panel.detach().cpu().clamp(-1, 1)
                       .permute(1, 2, 0).numpy()) + 1) * 127.5).astype(np.uint8)
        panels = [rgb_panel]
        for tensor in (real, clean, fake):
            panel = (((tensor[index, 0].detach().cpu().clamp(-1, 1).numpy())
                      + 1) * 127.5).astype(np.uint8)
            panels.append(np.repeat(panel[..., None], 3, axis=2))
        rows.append(np.concatenate(panels, axis=1))
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0), "RGB").save(path)


def load_teacher(checkpoint: Path, device: torch.device) -> SARClassifier64:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("native classifier class order mismatch")
    teacher = SARClassifier64(len(SOC40_CLASSES)).to(device)
    teacher.load_state_dict(state["model"])
    teacher.eval()
    set_grad(teacher, False)
    return teacher


def checkpoint_state(epoch: int, encoder: nn.Module, generator: nn.Module,
                     discriminator: nn.Module, ema_encoder: nn.Module,
                     ema_generator: nn.Module, generator_optimizer: torch.optim.Optimizer,
                     discriminator_optimizer: torch.optim.Optimizer,
                     validation: dict[str, float], args: argparse.Namespace) -> dict:
    return {
        "architecture": HIFC_ARCHITECTURE,
        "epoch": epoch,
        "classes": list(SOC40_CLASSES),
        "condition_dim": 12,
        "identity_encoder": encoder.state_dict(),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "ema_identity_encoder": ema_encoder.state_dict(),
        "ema_generator": ema_generator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "validation": validation,
        "training_policy": "class-matched unpaired RGB/SAR; no pixel alignment",
        "filters": {"band": args.band, "polarization": args.polarization,
                    "depression": args.depression},
    }


def main() -> None:
    args = arguments()
    if args.epochs <= 0 or args.batch_size <= 0 or args.epoch_size <= 0:
        raise ValueError("epochs, batch-size and epoch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    train_data = JointROIDataset(
        args.rgb_root, args.sar_train_root, rgb_size=128, roi_size=64,
        epoch_size=0, band=args.band, polarization=args.polarization,
        depression=args.depression, augment_rgb=True, source_view_mode="random")
    manifest = args.output / f"split_manifest__{args.band}_{args.polarization}_{args.depression}.json"
    train_keys, validation_keys = split_records(
        train_data.records, args.sar_train_root, manifest,
        args.validation_fraction, args.seed,
        {"band": args.band, "polarization": args.polarization,
         "depression": args.depression})
    configure_records(train_data, train_keys, args.sar_train_root, args.epoch_size)
    validation_data = JointROIDataset(
        args.rgb_root, args.sar_train_root, rgb_size=128, roi_size=64,
        epoch_size=0, band=args.band, polarization=args.polarization,
        depression=args.depression, augment_rgb=False, source_view_mode="random")
    configure_records(validation_data, validation_keys, args.sar_train_root)
    train_loader = make_loader(train_data, args.batch_size, args.workers, True, device)
    validation_loader = make_loader(validation_data, args.batch_size, args.workers, False, device)

    teacher = load_teacher(args.native_classifier_checkpoint, device)
    encoder = LargeRGBIdentityEncoder(len(SOC40_CLASSES)).to(device)
    generator = HIFCUnpairedGenerator().to(device)
    discriminator = HIFCConditionedDiscriminator().to(device)
    initialise_hifc(encoder, generator, discriminator)
    ema_encoder = copy.deepcopy(encoder).eval()
    ema_generator = copy.deepcopy(generator).eval()
    set_grad(ema_encoder, False)
    set_grad(ema_generator, False)

    generator_optimizer = torch.optim.AdamW((
        {"params": encoder.parameters(), "lr": args.identity_lr},
        {"params": generator.parameters(), "lr": args.generator_lr},
    ), betas=(0.0, .99), weight_decay=1e-4)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0.0, .99))
    generator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    discriminator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    start_epoch = 1
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("architecture") != HIFC_ARCHITECTURE:
            raise RuntimeError("--resume architecture mismatch")
        encoder.load_state_dict(saved["identity_encoder"])
        generator.load_state_dict(saved["generator"])
        discriminator.load_state_dict(saved["discriminator"])
        ema_encoder.load_state_dict(saved["ema_identity_encoder"])
        ema_generator.load_state_dict(saved["ema_generator"])
        generator_optimizer.load_state_dict(saved["generator_optimizer"])
        discriminator_optimizer.load_state_dict(saved["discriminator_optimizer"])
        start_epoch = int(saved["epoch"]) + 1

    counts = {
        "rgb_identity_encoder": parameter_count(encoder),
        "hifc_generator": parameter_count(generator),
        "shared_conditioned_discriminator": parameter_count(discriminator),
    }
    counts["total_trainable"] = sum(counts.values())
    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "architecture": HIFC_ARCHITECTURE,
        "parameters": counts,
        "train_records": len(train_data.records),
        "validation_records": len(validation_data.records),
        "condition_layout": ["azimuth_sin", "azimuth_cos", "dep_15", "dep_30",
                              "dep_45", "dep_60", "band_X", "band_KU",
                              "pol_HH", "pol_HV", "pol_VH", "pol_VV"],
        "losses": {
            "rgb_identity": "two-view class CE + merged cosine invariance",
            "adversarial": "single conditional projection PatchGAN",
            "ltc": "local residual/contrast/Haar batch moments; no pixel matching",
            "sfm": "native pre-classifier embedding cosine/moments + D feature moments",
            "geometry": "frozen native band/polarization/depression/azimuth heads",
        },
        "unpaired_policy": "same vehicle class only; source RGB view is random and target SAR condition is metadata",
        "pixel_alignment_used": False,
        "comparison_to_v1": "new standalone HiFC-inspired path; V1 untouched",
    }
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"parameters": counts, "train": train_data.summary(),
           "validation": validation_data.summary(),
           "condition": config["condition_layout"]}, flush=True)

    columns = (
        "epoch", "generator", "adversarial", "rgb_identity", "ltc", "sfm",
        "geometry", "discriminator", "disc_wrong_class", "disc_wrong_condition",
        "r1", "rgb_accuracy", "native_class_accuracy", "native_band_accuracy",
        "native_polarization_accuracy", "native_depression_accuracy",
        "native_azimuth_accuracy", "validation_ltc", "validation_sfm",
        "validation_geometry", "validation_native_class_accuracy")
    history = args.output / "history.csv"
    if start_epoch == 1:
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(columns)

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train(); generator.train(); discriminator.train()
        set_grad(discriminator, True)
        totals = defaultdict(float)
        steps = 0
        progress = tqdm(train_loader, desc=f"HiFC unpaired {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            rgb = batch["rgb"].to(device, non_blocking=True)
            rgb_alt = batch["rgb_alt"].to(device, non_blocking=True)
            labels = batch["class_id"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            depression = batch["depression"].to(device, non_blocking=True)
            azimuth = batch["azimuth"].to(device, non_blocking=True)
            condition = condition_from_batch(meta, depression)
            real = batch["roi"].to(device, non_blocking=True)
            real_d = differentiable_augment(real)
            spatial_noise = torch.randn(len(real), 1, 64, 64, device=device)

            # The D step sees a detached sample.  A second G forward below is
            # intentional: it keeps the E/G gradient route explicit.
            with torch.no_grad(), autocast_context(device, use_amp):
                d_identity, _, d_pyramid = encoder(rgb, return_pyramid=True)
                _, _, d_fake, _ = generator(d_identity, condition, d_pyramid,
                                             spatial_noise)
            discriminator_optimizer.zero_grad(set_to_none=True)
            do_r1 = args.r1_weight > 0 and batch_index % args.r1_every == 0
            real_for_d = real_d.detach().requires_grad_(do_r1)
            with autocast_context(device, use_amp):
                real_score, _ = discriminator(real_for_d, labels, condition)
                fake_score, _ = discriminator(d_fake.detach(), labels, condition)
                wrong_class_score, _ = discriminator(
                    real_for_d, labels.roll(1), condition)
                wrong_condition_score, _ = discriminator(
                    real_for_d, labels, condition.roll(1, 0))
                d_main = discriminator_hinge(real_score, fake_score)
                wrong_class = F.relu(1.0 + wrong_class_score).mean()
                wrong_condition = F.relu(1.0 + wrong_condition_score).mean()
                d_loss = d_main + .25 * wrong_class + .25 * wrong_condition
                r1 = real.new_zeros(())
                if do_r1:
                    gradient = torch.autograd.grad(
                        real_score.sum(), real_for_d, create_graph=True)[0]
                    r1 = gradient.flatten(1).square().sum(1).mean()
                    d_loss = d_loss + .5 * args.r1_weight * args.r1_every * r1
            discriminator_scaler.scale(d_loss).backward()
            discriminator_scaler.unscale_(discriminator_optimizer)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()

            set_grad(discriminator, False)
            generator_optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                identity, rgb_logits, pyramid = encoder(rgb, return_pyramid=True)
                alt_identity, alt_logits = encoder(rgb_alt)
                fake_clean, _, fake, _ = generator(
                    identity, condition, pyramid, spatial_noise)
                fake_score, fake_d_feature = discriminator(fake, labels, condition)
                with torch.no_grad():
                    _, real_d_feature = discriminator(real_d, labels, condition)
                adversarial = -fake_score.mean()
                rgb_loss = rgb_identity_loss(
                    rgb_logits, alt_logits, identity, alt_identity, labels)
                ltc = local_texture_contrast_loss(
                    differentiable_augment(fake), real_d)

            # Keep the frozen native teacher in FP32.  requires_grad=False on
            # its parameters still permits a gradient through its input fake.
            with torch.autocast(device_type=device.type, enabled=False):
                fake_logits, fake_teacher_feature = teacher(
                    ((fake.float() + 1.0) * .5).clamp(0, 1), return_features=True)
                with torch.no_grad():
                    _, real_teacher_feature = teacher(
                        ((real.float() + 1.0) * .5).clamp(0, 1), return_features=True)
                sfm = semantic_feature_mapping_loss(
                    fake_teacher_feature, real_teacher_feature,
                    fake_d_feature.float(), real_d_feature.float())
                geometry, geometry_terms = geometry_auxiliary_loss(
                    teacher, fake.float(), meta.float(), depression, azimuth)
                adv_scale = 0.0 if epoch <= args.adversarial_warmup_epochs else 1.0
                g_loss = (adv_scale * adversarial
                          + args.rgb_identity_weight * rgb_loss
                          + args.ltc_weight * ltc
                          + args.sfm_weight * sfm
                          + args.geometry_weight * geometry)
            generator_scaler.scale(g_loss).backward()
            generator_scaler.unscale_(generator_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(generator.parameters()), 5.0)
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()
            set_grad(discriminator, True)
            update_ema(ema_encoder, encoder, args.ema_decay)
            update_ema(ema_generator, generator, args.ema_decay)

            with torch.no_grad():
                rgb_accuracy = .5 * (
                    (rgb_logits.argmax(1) == labels).float().mean()
                    + (alt_logits.argmax(1) == labels).float().mean())
                native_class = fake_logits.argmax(1)
                native_aux = teacher.auxiliary_logits(fake_teacher_feature)
                target_band = (1 - meta[:, 3].round().long()).clamp(0, 1)
                target_pol = meta[:, 4:8].argmax(1)
                target_dep = torch.round(depression.float() / 15).long().sub(1).clamp(0, 3)
                target_az = ((azimuth.long() + 15) % 360) // 30
                aux_targets = (target_band, target_pol, target_dep, target_az)
                aux_acc = [
                    (logit.argmax(1) == target).float().mean()
                    for logit, target in zip(native_aux, aux_targets)
                ]
            values = {
                "generator": g_loss, "adversarial": adversarial,
                "rgb_identity": rgb_loss, "ltc": ltc, "sfm": sfm,
                "geometry": geometry, "discriminator": d_loss,
                "disc_wrong_class": wrong_class,
                "disc_wrong_condition": wrong_condition, "r1": r1,
                "rgb_accuracy": rgb_accuracy,
                "native_class_accuracy": (native_class == labels).float().mean(),
                "native_band_accuracy": aux_acc[0],
                "native_polarization_accuracy": aux_acc[1],
                "native_depression_accuracy": aux_acc[2],
                "native_azimuth_accuracy": aux_acc[3],
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            steps += 1
            progress.set_postfix(
                g=f"{float(g_loss.detach()):.3f}",
                d=f"{float(d_loss.detach()):.3f}",
                ltc=f"{float(ltc.detach()):.3f}",
                cls=f"{float(values['native_class_accuracy']):.3f}")
            if args.limit_train_batches and batch_index + 1 >= args.limit_train_batches:
                break

        ema_encoder.eval(); ema_generator.eval(); discriminator.eval()
        val_totals = defaultdict(float)
        val_count = 0
        preview = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(validation_loader):
                rgb = batch["rgb"].to(device)
                labels = batch["class_id"].to(device)
                meta = batch["meta"].to(device)
                depression = batch["depression"].to(device)
                condition = condition_from_batch(meta, depression)
                real = batch["roi"].to(device)
                identity, _, pyramid = ema_encoder(rgb, return_pyramid=True)
                noise = torch.randn(len(real), 1, 64, 64, device=device)
                clean, _, fake, _ = ema_generator(identity, condition, pyramid, noise)
                real_d = differentiable_augment(real)
                _, fake_d_feature = discriminator(fake, labels, condition)
                _, real_d_feature = discriminator(real_d, labels, condition)
                _, fake_teacher_feature = teacher(
                    ((fake + 1.0) * .5).clamp(0, 1), return_features=True)
                _, real_teacher_feature = teacher(
                    ((real + 1.0) * .5).clamp(0, 1), return_features=True)
                sfm = semantic_feature_mapping_loss(
                    fake_teacher_feature, real_teacher_feature,
                    fake_d_feature, real_d_feature)
                geometry, _ = geometry_auxiliary_loss(
                    teacher, fake, meta, depression, batch["azimuth"].to(device))
                ltc = local_texture_contrast_loss(fake, real_d)
                size = len(real)
                val_totals["ltc"] += float(ltc) * size
                val_totals["sfm"] += float(sfm) * size
                val_totals["geometry"] += float(geometry) * size
                val_totals["native_class_accuracy"] += float(
                    (teacher(((fake + 1.0) * .5).clamp(0, 1)).argmax(1)
                     == labels).float().sum())
                val_count += size
                if preview is None:
                    preview = (rgb, real, fake, clean)
                if (args.limit_validation_batches
                        and batch_index + 1 >= args.limit_validation_batches):
                    break
        validation = {name: value / max(val_count, 1)
                      for name, value in val_totals.items()}
        averages = {name: value / max(steps, 1) for name, value in totals.items()}
        row = (epoch, *[averages.get(name, float("nan")) for name in columns[1:17]],
               validation.get("ltc", float("nan")),
               validation.get("sfm", float("nan")),
               validation.get("geometry", float("nan")),
               validation.get("native_class_accuracy", float("nan")))
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        state = checkpoint_state(
            epoch, encoder, generator, discriminator, ema_encoder, ema_generator,
            generator_optimizer, discriminator_optimizer, validation, args)
        torch.save(state, args.output / "latest.pt")
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(state, args.output / f"epoch_{epoch:03d}.pt")
        if preview is not None and (epoch == 1 or epoch % 5 == 0 or epoch == args.epochs):
            save_preview(args.output / f"validation_{epoch:03d}.png", *preview)
        print(dict(zip(columns, row)), flush=True)


if __name__ == "__main__":
    main()

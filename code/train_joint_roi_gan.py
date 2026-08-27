"""Jointly train RGB identity recognition and identity-conditioned SAR ROI GAN.

Loss priority:
  1. RGB vehicle identity CE and frozen SARATR-X fake-ROI identity CE.
  2. Generated-to-real class prototype similarity in SARATR-X feature space.
  3. Same-class/same-azimuth structural, adversarial and discriminator-feature
     matching losses.

The RGB encoder is never frozen: its identity embedding is the generator's
primary input and receives gradients from both recognition and generation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bbox_data import image_tensor
from joint_data import JointROIDataset
from joint_models import (RGBIdentityEncoder, ROIDiscriminator, ROIGenerator, initialise,
                          multiscale_structure_loss, sar_statistics_loss)
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES, load_saratrx, saratrx_input


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint identity-first RGB-to-SAR ROI GAN")
    parser.add_argument("--rgb-root", type=Path, required=True,
                        help="original RGB root (1.png=0 degrees ... 12.png=330 degrees)")
    parser.add_argument("--sar-root", type=Path, required=True, help="SOC_40classes_cut/train or full train")
    parser.add_argument("--saratrx-checkpoint", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path,
                        help="optional native 64px image-only classifier; takes priority over SARATR-X")
    parser.add_argument("--saratrx-input-size", type=int, default=64, choices=(64,))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype-cache", type=Path)
    parser.add_argument("--prototype-samples", type=int, default=0,
                        help="real ROIs per class for centres; 0 means every ROI")
    parser.add_argument("--prototype-batch-size", type=int, default=128,
                        help="SAR-only batch size used when computing class centres")
    parser.add_argument("--pre-cropped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epoch-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--rgb-size", type=int, default=128)
    parser.add_argument("--roi-size", type=int, default=64, choices=(64,))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=5e-5)
    parser.add_argument("--discriminator-every", type=int, default=2,
                        help="update D once per this many generator steps")
    parser.add_argument("--identity-lr", type=float, default=1e-4)
    parser.add_argument("--rgb-id-weight", type=float, default=10.0)
    parser.add_argument("--cross-view-weight", type=float, default=2.0,
                        help="same-vehicle RGB embedding consistency across two views")
    parser.add_argument("--sar-class-weight", type=float, default=10.0)
    parser.add_argument("--cluster-weight", type=float, default=5.0)
    parser.add_argument("--structure-weight", type=float, default=20.0)
    parser.add_argument("--statistics-weight", type=float, default=5.0)
    parser.add_argument("--adversarial-weight", type=float, default=2.0)
    parser.add_argument("--feature-match-weight", type=float, default=5.0)
    parser.add_argument("--speckle-warmup-epochs", type=int, default=10)
    parser.add_argument("--speckle-ramp-epochs", type=int, default=5)
    parser.add_argument("--selection-min-rgb-accuracy", type=float, default=0.98)
    parser.add_argument("--selection-min-fake-accuracy", type=float, default=0.95)
    parser.add_argument("--band", default="all", choices=("all", "X", "KU"))
    parser.add_argument("--polarization", default="all", choices=("all", "HH", "HV", "VH", "VV"))
    parser.add_argument("--depression", default="all", choices=("all", "15", "30", "45", "60"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialise-from", type=Path,
                        help="copy G/D/RGB encoder weights but begin a fresh optimisation run")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="small encoder/G/D for an integration smoke run")
    return parser.parse_args()


def prototype_indices(dataset: JointROIDataset, samples_per_class: int, seed: int) -> list[int]:
    by_class: dict[str, list[int]] = {name: [] for name in SOC40_CLASSES}
    for index, record in enumerate(dataset.records):
        by_class[record[2]].append(index)
    generator = random.Random(seed)
    selected: list[int] = []
    for class_name in SOC40_CLASSES:
        indices = by_class[class_name]
        if not indices:
            raise RuntimeError(f"no SAR prototype samples for {class_name}")
        if samples_per_class > 0 and len(indices) > samples_per_class:
            indices = generator.sample(indices, samples_per_class)
        selected.extend(indices)
    return selected


def prototype_signature(args: argparse.Namespace, dataset: JointROIDataset) -> dict[str, object]:
    return {
        "sar_root": str(args.sar_root.resolve()),
        "checkpoint": str(args.saratrx_checkpoint.resolve()),
        "native_classifier_checkpoint": (str(args.native_classifier_checkpoint.resolve())
                                         if args.native_classifier_checkpoint else None),
        "saratrx_input_size": args.saratrx_input_size,
        "pre_cropped": args.pre_cropped,
        "roi_size": args.roi_size,
        "prototype_samples": args.prototype_samples,
        "records": len(dataset.records),
        "classes": list(SOC40_CLASSES),
    }


class PrototypeROIDataset(Dataset):
    """SAR-only view of JointROIDataset, avoiding unused RGB decode/collation."""

    def __init__(self, dataset: JointROIDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tif, _, class_name, bbox, _, _ = self.dataset.records[self.indices[index]]
        with Image.open(tif) as image:
            source = image if self.dataset.pre_cropped else image.crop(bbox)
            roi = image_tensor(source, self.dataset.roi_size, False)
        return {"roi": roi,
                "class_id": torch.tensor(self.dataset.class_to_id[class_name], dtype=torch.long)}


def build_prototypes(model: nn.Module, input_transform, dataset: JointROIDataset,
                     args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, float]:
    cache = args.prototype_cache or args.output / "saratrx_prototypes.pt"
    signature = prototype_signature(args, dataset)
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved.get("signature") == signature:
            print(f"loaded SARATR-X class centres from {cache}")
            return saved["prototypes"].to(device), float(saved.get("real_top1", float("nan")))
    indices = prototype_indices(dataset, args.prototype_samples, args.seed)
    loader = DataLoader(PrototypeROIDataset(dataset, indices),
                        batch_size=args.prototype_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    sums = torch.zeros(len(SOC40_CLASSES), model.feature_dim, device=device)
    counts = torch.zeros(len(SOC40_CLASSES), device=device)
    correct = total = 0
    model.eval()
    use_amp = device.type == "cuda" and not args.no_amp
    with torch.inference_mode():
        for batch in tqdm(loader, desc="SARATR-X real ROI centres"):
            roi = batch["roi"].to(device, non_blocking=True)
            labels = batch["class_id"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, features = model(input_transform(roi), return_features=True)
            features = F.normalize(features, dim=1)
            sums.index_add_(0, labels, features)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
    if (counts == 0).any():
        missing = [SOC40_CLASSES[i] for i in torch.where(counts == 0)[0].tolist()]
        raise RuntimeError(f"missing prototype classes: {missing}")
    prototypes = F.normalize(sums / counts[:, None], dim=1)
    real_top1 = correct / total
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": prototypes.cpu(), "counts": counts.cpu(),
                "real_top1": real_top1, "signature": signature}, cache)
    print(f"saved centres to {cache}; fixed SARATR-X top-1 on centre data: {real_top1:.4f}")
    return prototypes, real_top1


def set_grad(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def load_frozen_sar_judge(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, object, str]:
    """Load the image-only native judge when supplied, otherwise SARATR-X."""
    if args.native_classifier_checkpoint:
        saved = torch.load(args.native_classifier_checkpoint, map_location=device, weights_only=False)
        if saved.get("classes") != list(SOC40_CLASSES):
            raise RuntimeError("native classifier class order does not match SOC_40classes")
        model = SARClassifier64(len(SOC40_CLASSES)).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model, lambda roi: (roi + 1.0) * .5, "native64"
    model = load_saratrx(args.saratrx_checkpoint, device=device, freeze=True,
                         input_size=args.saratrx_input_size)
    return model, lambda roi: saratrx_input(roi, args.saratrx_input_size), "saratrx64"


def save_preview(rgb: torch.Tensor, fake: torch.Tensor, real: torch.Tensor,
                 path: Path, maximum: int = 8) -> None:
    """Save rows of input RGB | matched real SAR | generated SAR."""
    count = min(maximum, fake.shape[0])
    rows = []
    for index in range(count):
        rgb_panel = F.interpolate(rgb[index:index + 1], fake.shape[-2:], mode="bilinear",
                                  align_corners=False)[0].detach().cpu().permute(1, 2, 0)
        rgb_panel = ((rgb_panel.clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
        real_panel = ((real[index, 0].detach().cpu().clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
        fake_panel = ((fake[index, 0].detach().cpu().clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
        rows.append(np.concatenate((rgb_panel, np.repeat(real_panel[..., None], 3, axis=2),
                                    np.repeat(fake_panel[..., None], 3, axis=2)), axis=1))
    Image.fromarray(np.concatenate(rows, axis=0), "RGB").save(path)


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)

    dataset = JointROIDataset(
        args.rgb_root, args.sar_root, args.rgb_size, args.roi_size, args.epoch_size,
        args.pre_cropped, args.band, args.polarization, args.depression, augment_rgb=True,
    )
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    prototype_dataset = JointROIDataset(
        args.rgb_root, args.sar_root, args.rgb_size, args.roi_size, 0,
        args.pre_cropped, args.band, args.polarization, args.depression, augment_rgb=False,
        preload_rgb=False,
    )
    sar_judge, sar_input, judge_kind = load_frozen_sar_judge(args, device)
    prototypes, real_saratrx_top1 = build_prototypes(sar_judge, sar_input, prototype_dataset, args, device)

    base = 16 if args.tiny else 32
    encoder = RGBIdentityEncoder(len(SOC40_CLASSES), base=base).to(device)
    generator = ROIGenerator(base=base).to(device)
    discriminator = ROIDiscriminator(base=base).to(device)
    encoder.apply(initialise)
    generator.apply(initialise)
    discriminator.apply(initialise)
    generator_optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": args.identity_lr},
            {"params": generator.parameters(), "lr": args.lr},
        ], betas=(0.5, 0.999), foreach=False,
    )
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr,
                                                betas=(0.5, 0.999), foreach=False)
    use_amp = device.type == "cuda" and not args.no_amp
    generator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    discriminator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    start_epoch = 1
    best_rgb_accuracy = -1.0
    best_fake_at_best_rgb = -1.0
    best_quality = float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(saved["identity_encoder"])
        generator.load_state_dict(saved["generator"])
        discriminator.load_state_dict(saved["discriminator"])
        generator_optimizer.load_state_dict(saved["generator_optimizer"])
        discriminator_optimizer.load_state_dict(saved["discriminator_optimizer"])
        if "generator_scaler" in saved:
            generator_scaler.load_state_dict(saved["generator_scaler"])
        if "discriminator_scaler" in saved:
            discriminator_scaler.load_state_dict(saved["discriminator_scaler"])
        start_epoch = int(saved["epoch"]) + 1
        best_rgb_accuracy = float(saved.get("best_rgb_accuracy", -1.0))
        best_fake_at_best_rgb = float(saved.get("best_fake_at_best_rgb",
                                                saved.get("best_fake_accuracy", -1.0)))
        best_quality = float(saved.get("best_quality", float("inf")))
    elif args.initialise_from:
        saved = torch.load(args.initialise_from, map_location=device, weights_only=False)
        encoder.load_state_dict(saved["identity_encoder"])
        generator.load_state_dict(saved["generator"])
        discriminator.load_state_dict(saved["discriminator"])

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(dataset.summary())
    config["class_order"] = list(SOC40_CLASSES)
    config["fixed_saratrx_real_roi_top1"] = real_saratrx_top1
    config["sar_judge_kind"] = judge_kind
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    history = args.output / "history.csv"
    if start_epoch == 1 or not history.exists():
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow((
                "epoch", "loss_identity_total", "loss_rgb_identity", "loss_cross_view",
                "loss_sar_class",
                "loss_cluster", "loss_structure", "loss_statistics", "loss_adversarial",
                "loss_feature_match",
                "loss_discriminator", "rgb_identity_accuracy", "fake_saratrx_accuracy",
                "cluster_cosine", "identity_to_generator_gate", "speckle_strength",
            ))
    print("dataset:", dataset.summary(), "device:", device,
          "fixed SARATR-X real ROI top1:", f"{real_saratrx_top1:.4f}")
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.02)

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        generator.train()
        discriminator.train()
        totals = torch.zeros(16, dtype=torch.float64)
        ramp_position = ((epoch - args.speckle_warmup_epochs)
                         / max(1, args.speckle_ramp_epochs))
        current_speckle = generator.speckle_strength * min(1.0, max(0.0, ramp_position))
        progress = tqdm(loader, desc=f"joint epoch {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            rgb = batch["rgb"].to(device, non_blocking=True)
            rgb_alt = batch["rgb_alt"].to(device, non_blocking=True)
            real = batch["roi"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            labels = batch["class_id"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                identity, rgb_logits = encoder(rgb)
                alternate_identity, alternate_logits = encoder(rgb_alt)
                # G always receives the live identity embedding. Its generation
                # gradient into the recogniser is confidence-gated.
                correct_probability = rgb_logits.softmax(1).gather(1, labels[:, None]).mean().detach()
                identity_gate = correct_probability.clamp(0.05, 1.0)
                identity_for_generator = (identity.detach()
                                          + identity_gate * (identity - identity.detach()))
                fake_clean = generator(identity_for_generator, meta, apply_speckle=False)
                fake = generator.apply_speckle(fake_clean, current_speckle)

            set_grad(discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                real_score, _ = discriminator(real)
                fake_score, _ = discriminator(fake.detach())
                discriminator_loss = (F.relu(1.0 - real_score).mean()
                                      + F.relu(1.0 + fake_score).mean())
            if batch_index % args.discriminator_every == 0:
                discriminator_scaler.scale(discriminator_loss).backward()
                discriminator_scaler.step(discriminator_optimizer)
                discriminator_scaler.update()

            set_grad(discriminator, False)
            generator_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fake_score, fake_disc_features = discriminator(fake)
                with torch.no_grad():
                    _, real_disc_features = discriminator(real)
                sar_logits, sar_features = sar_judge(sar_input(fake), return_features=True)
                rgb_identity_loss = 0.5 * (cross_entropy(rgb_logits, labels)
                                           + cross_entropy(alternate_logits, labels))
                cross_view_loss = 1.0 - (F.normalize(identity, dim=1)
                                         * F.normalize(alternate_identity, dim=1)).sum(1).mean()
                sar_class_loss = cross_entropy(sar_logits, labels)
                cluster_cosine = (F.normalize(sar_features, dim=1)
                                  * prototypes[labels]).sum(1).mean()
                cluster_loss = 1.0 - cluster_cosine
                structure_loss = multiscale_structure_loss(fake_clean, real)
                statistics_loss = sar_statistics_loss(fake, real)
                adversarial_loss = -fake_score.mean()
                feature_match_loss = (F.l1_loss(
                    fake_disc_features.mean((2, 3)), real_disc_features.mean((2, 3)))
                    + F.l1_loss(fake_disc_features.std((2, 3)),
                                real_disc_features.std((2, 3))))
                identity_total = (args.rgb_id_weight * rgb_identity_loss
                                  + args.cross_view_weight * cross_view_loss
                                  + args.sar_class_weight * sar_class_loss)
                generator_loss = (
                    identity_total
                    + args.cluster_weight * cluster_loss
                    + args.structure_weight * structure_loss
                    + args.statistics_weight * statistics_loss
                    + args.adversarial_weight * adversarial_loss
                    + args.feature_match_weight * feature_match_loss
                )
            generator_scaler.scale(generator_loss).backward()
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()
            set_grad(discriminator, True)

            rgb_accuracy = 0.5 * ((rgb_logits.argmax(1) == labels).float().mean()
                                  + (alternate_logits.argmax(1) == labels).float().mean())
            fake_accuracy = (sar_logits.argmax(1) == labels).float().mean()
            values = (
                identity_total, rgb_identity_loss, cross_view_loss, sar_class_loss,
                cluster_loss, structure_loss, statistics_loss,
                adversarial_loss, feature_match_loss, discriminator_loss, rgb_accuracy,
                fake_accuracy, cluster_cosine, identity_gate,
                identity_gate.new_tensor(current_speckle),
            )
            totals[:15] += torch.tensor([value.detach().item() for value in values], dtype=torch.float64)
            totals[15] += 1
            progress.set_postfix(rgb=f"{rgb_accuracy.item():.2f}", sar=f"{fake_accuracy.item():.2f}",
                                 cluster=f"{cluster_cosine.item():.2f}")

        averages = (totals[:15] / totals[15]).tolist()
        row = [epoch, *averages]
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        rgb_accuracy = averages[10]
        fake_accuracy = averages[11]
        # Do not let a clean warm-up image win merely because it is easy for
        # SARATR-X to classify. Once identity is reliable, generation quality
        # decides among checkpoints from the full-speckle phase.
        quality = averages[5] + 0.5 * averages[6] + 0.1 * averages[8]
        full_speckle = current_speckle >= generator.speckle_strength - 1e-6
        identity_ready = (rgb_accuracy >= args.selection_min_rgb_accuracy
                          and fake_accuracy >= args.selection_min_fake_accuracy)
        is_best = full_speckle and identity_ready and quality < best_quality
        # Short smoke runs may end before the curriculum; still leave a usable
        # checkpoint, but never prefer it over an eligible full-speckle model.
        if epoch == args.epochs and best_quality == float("inf"):
            is_best = True
        if is_best:
            best_rgb_accuracy = rgb_accuracy
            best_fake_at_best_rgb = fake_accuracy
            best_quality = quality
        checkpoint = {
            "epoch": epoch,
            "identity_encoder": encoder.state_dict(),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "generator_scaler": generator_scaler.state_dict(),
            "discriminator_scaler": discriminator_scaler.state_dict(),
            "best_rgb_accuracy": best_rgb_accuracy,
            "best_fake_at_best_rgb": best_fake_at_best_rgb,
            "best_fake_accuracy": fake_accuracy,
            "best_quality": best_quality,
            "classes": list(SOC40_CLASSES),
            "args": config,
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if is_best:
            torch.save(checkpoint, args.output / "best.pt")
        if epoch == 1 or epoch % 5 == 0:
            save_preview(rgb, fake, real, args.output / f"preview_{epoch:04d}.png")
            torch.save({"epoch": epoch,
                        "identity_encoder": encoder.state_dict(),
                        "generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "classes": list(SOC40_CLASSES), "args": config},
                       args.output / f"milestone_{epoch:04d}.pt")
        print(dict(zip(("epoch", "identity", "rgb_ce", "cross_view", "sar_ce", "cluster",
                        "structure", "statistics", "adv", "feature", "disc", "rgb_acc", "fake_acc",
                        "cluster_cos", "id_to_g_gate", "speckle_strength"), row)), flush=True)


if __name__ == "__main__":
    main()

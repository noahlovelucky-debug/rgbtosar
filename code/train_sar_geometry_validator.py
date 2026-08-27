"""Train the independent geometry validator on real X/HH 64x64 SAR only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bbox_data import SAR_RE, image_tensor
from sar_geometry_validator import (
    AZIMUTH_BINS, DEPRESSION_VALUES, SARGeometryValidator,
    circular_bin_distance, circular_degree_error,
    circular_soft_cross_entropy)
from saratrx import SOC40_CLASSES


class RealSARGeometryDataset(Dataset):
    def __init__(self, paths: list[Path], augment: bool) -> None:
        self.paths = paths
        self.augment = augment
        self.class_to_id = {
            name: index for index, name in enumerate(SOC40_CLASSES)}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index]
        match = SAR_RE.match(path.stem)
        if match is None:
            raise ValueError(path)
        _, _, depression_text, azimuth_text = match.groups()
        depression = int(depression_text)
        azimuth = int(azimuth_text) % 360
        with Image.open(path) as image:
            sar = image_tensor(image, 64, False)
        if self.augment:
            # Geometry-preserving radiometric augmentation only: flips and
            # rotations would invalidate the azimuth label.
            gain = random.uniform(.88, 1.12)
            bias = random.uniform(-.04, .04)
            granular = torch.randn_like(sar) * random.uniform(0, .025)
            sar = (sar * gain + bias + granular).clamp(-1, 1)
        radians = math.radians(azimuth)
        return {
            "image": sar,
            "identity": torch.tensor(
                self.class_to_id[path.parent.name], dtype=torch.long),
            "depression": torch.tensor(
                DEPRESSION_VALUES.index(depression), dtype=torch.long),
            "azimuth_bin": torch.tensor(
                round(azimuth / (360 / AZIMUTH_BINS)) % AZIMUTH_BINS,
                dtype=torch.long),
            "azimuth_vector": torch.tensor(
                (math.sin(radians), math.cos(radians)), dtype=torch.float32),
        }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classifier-initialization", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=.35)
    parser.add_argument("--head-warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-validation-batches", type=int, default=0)
    return parser.parse_args()


def xhh_paths(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.glob("*/*.tif")):
        match = SAR_RE.match(path.stem)
        if match is None:
            continue
        band, polarization, depression, _ = match.groups()
        if (band.upper() == "X" and polarization.upper() == "HH"
                and int(depression) in DEPRESSION_VALUES
                and path.parent.name in SOC40_CLASSES):
            paths.append(path)
    if not paths:
        raise RuntimeError(f"no X/HH SAR found under {root}")
    return paths


def split_paths(paths: list[Path], root: Path, fraction: float,
                seed: int) -> tuple[list[Path], list[Path]]:
    groups: dict[tuple[str, str], list[Path]] = {}
    for path in paths:
        match = SAR_RE.match(path.stem)
        assert match is not None
        key = (path.parent.name, match.group(3))
        groups.setdefault(key, []).append(path)
    train, validation = [], []
    for key, values in sorted(groups.items()):
        values = sorted(values, key=lambda path: hashlib.sha256(
            f"{seed}:{path.relative_to(root)}".encode()).hexdigest())
        count = max(1, round(len(values) * fraction))
        validation.extend(values[:count])
        train.extend(values[count:])
    return sorted(train), sorted(validation)


def make_loader(dataset: Dataset, batch_size: int, workers: int,
                shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(), drop_last=shuffle)


def evaluate(model: SARGeometryValidator, loader: DataLoader,
             device: torch.device, use_amp: bool,
             limit_batches: int = 0) -> dict[str, float]:
    model.eval()
    totals = {
        "samples": 0, "identity_correct": 0, "depression_correct": 0,
        "azimuth_bin_correct": 0, "azimuth_bin_distance": 0.0,
        "azimuth_degree_error": 0.0, "azimuth_cosine": 0.0}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device)
            identity = batch["identity"].to(device)
            depression = batch["depression"].to(device)
            azimuth_bin = batch["azimuth_bin"].to(device)
            azimuth_vector = batch["azimuth_vector"].to(device)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                output = model((image + 1) * .5)
            size = len(image)
            totals["samples"] += size
            totals["identity_correct"] += int(
                (output.identity_logits.argmax(1) == identity).sum())
            totals["depression_correct"] += int(
                (output.depression_logits.argmax(1) == depression).sum())
            totals["azimuth_bin_correct"] += int(
                (output.azimuth_logits.argmax(1) == azimuth_bin).sum())
            totals["azimuth_bin_distance"] += float(
                circular_bin_distance(
                    output.azimuth_logits, azimuth_bin).sum())
            totals["azimuth_degree_error"] += float(
                circular_degree_error(
                    output.azimuth_vector, azimuth_vector).sum())
            totals["azimuth_cosine"] += float(
                (output.azimuth_vector * azimuth_vector).sum())
            if limit_batches and batch_index + 1 >= limit_batches:
                break
    samples = max(1, int(totals["samples"]))
    return {
        "samples": int(totals["samples"]),
        "identity_top1": totals["identity_correct"] / samples,
        "depression_top1": totals["depression_correct"] / samples,
        "azimuth_bin_top1": totals["azimuth_bin_correct"] / samples,
        "azimuth_bin_mae": totals["azimuth_bin_distance"] / samples,
        "azimuth_degree_mae": totals["azimuth_degree_error"] / samples,
        "azimuth_cosine": totals["azimuth_cosine"] / samples,
    }


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    all_train = xhh_paths(args.train_root)
    train_paths, validation_paths = split_paths(
        all_train, args.train_root, args.validation_fraction, args.seed)
    test_paths = xhh_paths(args.test_root)
    train_loader = make_loader(
        RealSARGeometryDataset(train_paths, True),
        args.batch_size, args.workers, True)
    validation_loader = make_loader(
        RealSARGeometryDataset(validation_paths, False),
        args.batch_size, args.workers, False)
    test_loader = make_loader(
        RealSARGeometryDataset(test_paths, False),
        args.batch_size, args.workers, False)

    model = SARGeometryValidator(len(SOC40_CLASSES)).to(device)
    if args.classifier_initialization:
        saved = torch.load(
            args.classifier_initialization, map_location=device,
            weights_only=False)
        state = saved.get("model", saved)
        missing, unexpected = model.backbone.load_state_dict(
            state, strict=False)
        # Reuse the real-SAR auxiliary geometry knowledge already present in
        # the native classifier, then refine it to 5-degree resolution.
        model.depression_head.load_state_dict(
            model.backbone.depression_head.state_dict())
        old_weight = model.backbone.azimuth_head.weight.detach()
        old_bias = model.backbone.azimuth_head.bias.detach()
        with torch.no_grad():
            for target_bin in range(AZIMUTH_BINS):
                position = target_bin / (AZIMUTH_BINS / 12)
                lower = int(math.floor(position)) % 12
                upper = (lower + 1) % 12
                fraction = position - math.floor(position)
                model.azimuth_head.weight[target_bin].copy_(
                    old_weight[lower].lerp(old_weight[upper], fraction))
                model.azimuth_head.bias[target_bin].copy_(
                    old_bias[lower].lerp(old_bias[upper], fraction))
            centres = torch.arange(
                12, device=device, dtype=old_weight.dtype
            ) * (2 * math.pi / 12)
            model.azimuth_vector_head.weight[0].copy_(
                (centres.sin()[:, None] * old_weight).mean(0))
            model.azimuth_vector_head.weight[1].copy_(
                (centres.cos()[:, None] * old_weight).mean(0))
            model.azimuth_vector_head.bias[0].copy_(
                (centres.sin() * old_bias).mean())
            model.azimuth_vector_head.bias[1].copy_(
                (centres.cos() * old_bias).mean())
        print({
            "classifier_initialization": str(args.classifier_initialization),
            "missing": missing, "unexpected": unexpected}, flush=True)
    head_parameters = (
        list(model.depression_head.parameters())
        + list(model.azimuth_head.parameters())
        + list(model.azimuth_vector_head.parameters()))
    optimizer = torch.optim.AdamW((
        {"params": model.backbone.parameters(),
         "lr": args.learning_rate * args.backbone_lr_scale},
        {"params": head_parameters, "lr": args.learning_rate},
    ), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate * .05)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ce = nn.CrossEntropyLoss(label_smoothing=.03)

    manifest = {
        "train": [str(path.relative_to(args.train_root)) for path in train_paths],
        "validation": [
            str(path.relative_to(args.train_root)) for path in validation_paths],
        "test_root": str(args.test_root.resolve()),
        "test_policy": "official test evaluated only after checkpoint selection",
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "architecture": "sar_geometry_validator_v2",
        "train_samples": len(train_paths),
        "validation_samples": len(validation_paths),
        "test_samples": len(test_paths),
        "azimuth_bins": AZIMUTH_BINS,
        "depressions": DEPRESSION_VALUES,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = (
        "epoch", "train_loss", "train_identity_top1",
        "validation_identity_top1", "validation_depression_top1",
        "validation_azimuth_bin_top1", "validation_azimuth_bin_mae",
        "validation_azimuth_degree_mae", "validation_azimuth_cosine")
    with (args.output / "history.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(columns)

    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        backbone_trainable = epoch > args.head_warmup_epochs
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(backbone_trainable)
        model.train()
        loss_sum = correct = samples = 0.0
        optimizer_steps = 0
        progress = tqdm(train_loader, desc=f"geometry validator {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            image = batch["image"].to(device)
            identity = batch["identity"].to(device)
            depression = batch["depression"].to(device)
            azimuth_bin = batch["azimuth_bin"].to(device)
            azimuth_vector = batch["azimuth_vector"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp):
                output = model((image + 1) * .5)
                identity_loss = ce(output.identity_logits, identity)
                depression_loss = ce(output.depression_logits, depression)
                azimuth_bin_loss = circular_soft_cross_entropy(
                    output.azimuth_logits, azimuth_bin)
                azimuth_vector_loss = (
                    1 - (output.azimuth_vector * azimuth_vector).sum(1)).mean()
                loss = (
                    identity_loss + depression_loss
                    + 1.5 * azimuth_bin_loss + 2 * azimuth_vector_loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                optimizer_steps += 1
            size = len(image)
            loss_sum += float(loss.detach()) * size
            correct += int(
                (output.identity_logits.argmax(1) == identity).sum())
            samples += size
            progress.set_postfix(
                loss=f"{loss_sum / samples:.3f}",
                identity=f"{correct / samples:.3f}")
            if args.limit_train_batches and batch_index + 1 >= args.limit_train_batches:
                break
        if optimizer_steps:
            scheduler.step()
        metrics = evaluate(
            model, validation_loader, device, use_amp,
            args.limit_validation_batches)
        row = (
            epoch, loss_sum / max(samples, 1), correct / max(samples, 1),
            metrics["identity_top1"], metrics["depression_top1"],
            metrics["azimuth_bin_top1"], metrics["azimuth_bin_mae"],
            metrics["azimuth_degree_mae"], metrics["azimuth_cosine"])
        with (args.output / "history.csv").open(
                "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        print(dict(zip(columns, row)), flush=True)
        score = (
            metrics["azimuth_degree_mae"] / 30
            + (1 - metrics["depression_top1"])
            + .5 * (1 - metrics["identity_top1"]))
        state = {
            "architecture": "sar_geometry_validator_v2",
            "epoch": epoch, "model": model.state_dict(),
            "classes": list(SOC40_CLASSES),
            "metrics": metrics, "config": config}
        torch.save(state, args.output / "latest.pt")
        if score < best_score:
            best_score = score
            torch.save(state, args.output / "best.pt")

    best = torch.load(
        args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, device, use_amp)
    report = {
        "selected_epoch": best["epoch"],
        "validation": best["metrics"],
        "official_test": test_metrics,
        "checkpoint": str((args.output / "best.pt").resolve())}
    (args.output / "test_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()

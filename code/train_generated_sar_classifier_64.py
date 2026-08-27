"""Train SARClassifier64 solely on ROIs produced by a frozen continuous GAN.

Real train TIFF pixel values are never passed to the classifier.  They provide
only the legal observation condition (class, X/HH, depression, azimuth) used
to ask the frozen GAN for a synthetic training image.  Evaluation uses real,
held-out X/HH TIFFs.
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bbox_data import image_tensor, metadata_vector, read_annotation
from joint_data import JointROIDataset
from joint_models import (CodebookSpatialROIGenerator, RGBIdentityEncoder, SARStyleEncoder,
                          SpatialROIGenerator, StyleSpatialROIGenerator)
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES


DEPRESSION_TO_ID = {15: 0, 30: 1, 45: 2, 60: 3}


def gan_condition(meta: torch.Tensor, source_angle: torch.Tensor) -> torch.Tensor:
    """Match the GAN's condition exactly, excluding real annotation-box extent."""
    meta = meta.clone()
    meta[:, -2:] = 0.0
    radians = source_angle.float() * (math.pi / 180.0)
    return torch.cat((meta, radians.sin()[:, None], radians.cos()[:, None]), dim=1)


class GeneratedConditionDataset(Dataset):
    """Conditions plus RGB source images; does not load real SAR image pixels."""

    def __init__(self, rgb_root: Path, sar_root: Path, rgb_size: int = 128,
                 include_style_roi: bool = False) -> None:
        self.include_style_roi = include_style_roi
        self.base = JointROIDataset(rgb_root, sar_root, rgb_size=rgb_size, epoch_size=0,
                                   band="X", polarization="HH", depression="all",
                                   augment_rgb=False, source_view_mode="random")

    def __len__(self) -> int:
        return len(self.base.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tif, _, class_name, bbox, meta, _ = self.base.records[index]
        source_angle = random.choice(self.base.class_rgb_angles[class_name])
        rgb = self.base._rgb(self.base.rgb_paths[class_name, source_angle])
        targets = torch.tensor((self.base.class_to_id[class_name], 0, 0,
                                DEPRESSION_TO_ID[int(meta["depression"])],
                                ((int(meta["azimuth"]) + 15) % 360) // 30), dtype=torch.long)
        if self.include_style_roi:
            with Image.open(tif) as image:
                style_roi = image_tensor(image, 64, False)
        else:
            style_roi = torch.zeros(1, 64, 64)
        return rgb, metadata_vector(meta, bbox), torch.tensor(source_angle, dtype=torch.long), targets, style_roi


class RealXHHTestDataset(Dataset):
    """Real 64x64 X/HH ROI test images and metadata labels."""

    def __init__(self, root: Path) -> None:
        self.records: list[tuple[Path, int, int, int]] = []
        for class_id, class_name in enumerate(SOC40_CLASSES):
            for path in sorted((Path(root) / class_name).glob("X_HH_*.tif")):
                try:
                    _, meta = read_annotation(path.with_suffix(".xml"))
                except Exception:
                    continue
                self.records.append((path, class_id, DEPRESSION_TO_ID[int(meta["depression"])],
                                     ((int(meta["azimuth"]) + 15) % 360) // 30))
        if not self.records:
            raise RuntimeError(f"no X/HH TIFFs under {root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, class_id, depression, azimuth = self.records[index]
        with Image.open(path) as image:
            roi = image_tensor(image, 64, False).add(1).mul(.5)
        return roi, torch.tensor((class_id, 0, 0, depression, azimuth), dtype=torch.long)


def augment(image: torch.Tensor) -> torch.Tensor:
    """Same image-only perturbations used for the real-SAR classifier method."""
    batch = len(image)
    output = image.clone()
    for index in range(batch):
        limit = 3
        dy, dx = random.randint(-limit, limit), random.randint(-limit, limit)
        if dy or dx:
            padded = F.pad(output[index:index + 1], (limit,) * 4, mode="replicate")
            output[index:index + 1] = padded[..., limit + dy:limit + dy + 64, limit + dx:limit + dx + 64]
    gains = output.new_empty(batch, 1, 1, 1).uniform_(.90, 1.10)
    bias = output.new_empty(batch, 1, 1, 1).uniform_(-.025, .025)
    speckle = torch.exp(torch.randn_like(output) * output.new_empty(batch, 1, 1, 1).uniform_(0, .07))
    output = output * gains * speckle + bias
    for index in range(batch):
        if random.random() < .12:
            side = random.randint(3, 7); y, x = random.randint(0, 64 - side), random.randint(0, 64 - side)
            output[index, :, y:y + side, x:x + side] = output[index].mean()
    return output.clamp(0, 1)


def evaluate(model: SARClassifier64, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.eval(); total = correct = top5 = 0; loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(); by_depression: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    with torch.inference_mode():
        for image, targets in tqdm(loader, desc="real X/HH classifier test", leave=False):
            image, targets = image.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            logits = model(image); labels = targets[:, 0]; prediction = logits.argmax(1)
            loss_sum += criterion(logits, labels).item() * len(labels)
            correct += (prediction == labels).sum().item()
            top5 += (logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item(); total += len(labels)
            for depression in (15, 30, 45, 60):
                mask = targets[:, 3] == DEPRESSION_TO_ID[depression]
                by_depression[depression][0] += int(mask.sum().item())
                by_depression[depression][1] += int((prediction[mask] == labels[mask]).sum().item())
    return ({"loss": loss_sum / total, "top1": correct / total, "top5": top5 / total, "samples": total},
            {str(key): {"samples": n, "top1": right / n} for key, (n, right) in by_depression.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SARClassifier64 on frozen-GAN samples, test on real X/HH")
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True, help="real SAR train root; metadata only")
    parser.add_argument("--real-test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--aux-weight", type=float, default=.12)
    parser.add_argument("--style-source", choices=("prior", "posterior"), default="prior",
                        help="posterior is a diagnostic upper bound that encodes a real train ROI style")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=415)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device); args.output.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
    architecture = state.get("architecture")
    if architecture not in {"continuous_spatial_v1", "continuous_spatial_style_v2",
                            "continuous_spatial_codebook_v3"} or state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("expected a continuous spatial SOC40 GAN checkpoint")
    encoder = RGBIdentityEncoder(len(SOC40_CLASSES)).to(device)
    if architecture == "continuous_spatial_style_v2":
        generator = StyleSpatialROIGenerator(meta_dim=12, style_dim=int(state["style_dim"])).to(device)
        style_encoder = SARStyleEncoder(int(state["style_dim"])).to(device)
        style_encoder.load_state_dict(state["style_encoder"]); style_encoder.eval()
        for parameter in style_encoder.parameters(): parameter.requires_grad_(False)
        latent_codes = None
        code_lookup = None
    elif architecture == "continuous_spatial_codebook_v3":
        generator = CodebookSpatialROIGenerator(
            meta_dim=12, code_channels=int(state["code_channels"])).to(device)
        style_encoder = None
        required = ("latent_codes", "latent_class", "latent_depression", "latent_azimuth_bin")
        if any(key not in state for key in required):
            raise RuntimeError("codebook checkpoint is missing exported latent codes")
        latent_codes = state["latent_codes"].to(device)
        code_lookup = defaultdict(list)
        for index, key in enumerate(zip(state["latent_class"].tolist(),
                                        state["latent_depression"].tolist(),
                                        state["latent_azimuth_bin"].tolist())):
            code_lookup[tuple(map(int, key))].append(index)
    else:
        generator = SpatialROIGenerator(meta_dim=12).to(device)
        style_encoder = None
        latent_codes = None
        code_lookup = None
    encoder.load_state_dict(state["identity_encoder"]); generator.load_state_dict(state["generator"])
    encoder.eval(); generator.eval()
    for parameter in (*encoder.parameters(), *generator.parameters()): parameter.requires_grad_(False)

    if args.style_source == "posterior" and architecture != "continuous_spatial_style_v2":
        raise ValueError("--style-source posterior requires a continuous_spatial_style_v2 checkpoint")
    generated_train = GeneratedConditionDataset(args.rgb_root, args.condition_root,
                                                include_style_roi=args.style_source == "posterior")
    real_test = RealXHHTestDataset(args.real_test_root)
    train_loader = DataLoader(generated_train, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    test_loader = DataLoader(real_test, args.batch_size * 2, shuffle=False, num_workers=args.workers,
                             pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    classifier = SARClassifier64(len(SOC40_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, min(3, args.epochs // 8))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: ((epoch + 1) / warmup if epoch < warmup else
        .5 * (1 + np.cos(np.pi * (epoch - warmup + 1) / max(1, args.epochs - warmup + 1)))))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and not args.no_amp)
    class_loss, aux_loss = nn.CrossEntropyLoss(label_smoothing=.03), nn.CrossEntropyLoss()
    history = args.output / "history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(("epoch", "synthetic_train_loss", "synthetic_train_top1", "real_test_loss", "real_test_top1", "real_test_top5", "lr"))
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        classifier.train(); loss_sum = correct = total = 0
        for rgb, meta, source_angle, targets, style_roi in tqdm(
                train_loader, desc=f"synthetic SAR classifier {epoch}/{args.epochs}"):
            rgb, meta = rgb.to(device, non_blocking=True), meta.to(device, non_blocking=True)
            source_angle, targets = source_angle.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.inference_mode():
                identity, _, pyramid = encoder(rgb, return_pyramid=True)
                condition = gan_condition(meta, source_angle)
                if architecture == "continuous_spatial_codebook_v3":
                    assert latent_codes is not None and code_lookup is not None
                    keys = zip(targets[:, 0].tolist(), targets[:, 3].tolist(),
                               targets[:, 4].tolist())
                    selected = []
                    for key in keys:
                        candidates = code_lookup[tuple(map(int, key))]
                        if not candidates:
                            raise RuntimeError(f"no spatial SAR code for condition {key}")
                        selected.append(random.choice(candidates))
                    code_index = torch.tensor(selected, device=device)
                    code = latent_codes[code_index].float()
                    synthetic = generator(identity, condition, pyramid, code, apply_speckle=True)
                elif architecture == "continuous_spatial_style_v2":
                    if args.style_source == "posterior":
                        assert style_encoder is not None
                        _, style, _ = style_encoder(style_roi.to(device, non_blocking=True), sample=False)
                    else:
                        noise = torch.randn(len(rgb), int(state["style_dim"]), device=device)
                        style = noise
                    if args.style_source == "prior" and "style_prior_mean" in state and "style_prior_cholesky" in state:
                        depression = targets[:, 3]
                        prior_mean = state["style_prior_mean"].to(device)
                        prior_factor = state["style_prior_cholesky"].to(device)
                        if prior_mean.ndim == 3:
                            mean = prior_mean[targets[:, 0], depression]
                            factor = prior_factor[targets[:, 0], depression]
                        else:
                            mean = prior_mean[depression]
                            factor = prior_factor[depression]
                        style = mean + torch.bmm(factor, noise[:, :, None]).squeeze(2)
                    synthetic = generator(identity, condition, pyramid, style, apply_speckle=True)
                else:
                    synthetic = generator(identity, condition, pyramid, apply_speckle=True)
                synthetic = augment((synthetic + 1) * .5)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits, features = classifier(synthetic, return_features=True)
                auxiliary = classifier.auxiliary_logits(features)
                loss = class_loss(logits, targets[:, 0])
                loss += args.aux_weight * sum(aux_loss(logit, target) for logit, target in zip(auxiliary, targets[:, 1:].unbind(1))) / len(auxiliary)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.)
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.detach().item() * len(targets); correct += (logits.argmax(1) == targets[:, 0]).sum().item(); total += len(targets)
        metrics, by_depression = evaluate(classifier, test_loader, device)
        row = (epoch, loss_sum / total, correct / total, metrics["loss"], metrics["top1"], metrics["top5"], optimizer.param_groups[0]["lr"])
        with history.open("a", newline="", encoding="utf-8") as handle: csv.writer(handle).writerow(row)
        scheduler.step()
        saved = {"model": classifier.state_dict(), "epoch": epoch, "classes": list(SOC40_CLASSES), "input_size": 64,
                 "metrics": metrics, "gan_checkpoint": str(args.gan_checkpoint), "training_source": "frozen GAN samples only"}
        torch.save(saved, args.output / "latest.pt")
        if metrics["top1"] >= best:
            best = metrics["top1"]; torch.save(saved, args.output / "best.pt")
            (args.output / "best_real_test_metrics.json").write_text(json.dumps({**metrics, "by_depression": by_depression}, indent=2), encoding="utf-8")
        print(dict(zip(("epoch", "synthetic_loss", "synthetic_top1", "real_loss", "real_top1", "real_top5", "lr"), row)), flush=True)
    (args.output / "config.json").write_text(json.dumps({**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "synthetic_train_samples_per_epoch": len(generated_train), "real_test_samples": len(real_test),
        "training_policy": (
            "classifier sees generated pixels only; generator samples a frozen empirical spatial-code prior "
            "learned from real train SAR" if architecture == "continuous_spatial_codebook_v3" else
            "classifier sees generated pixels only; posterior diagnostic uses real train ROI only "
            "inside the frozen style encoder" if args.style_source == "posterior" else
            "no real SAR pixels in classifier training; real train root supplies condition labels only")},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

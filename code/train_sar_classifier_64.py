"""Train and audit an image-only, native 64x64 SAR vehicle classifier."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES


NAME = re.compile(r"^(X|KU)_(HH|HV|VH|VV)_(15|30|45|60)_(\d{1,3})_\d+$", re.I)
BANDS = ("X", "KU")
POLS = ("HH", "HV", "VH", "VV")
DEPRESSIONS = (15, 30, 45, 60)


class SARImageDataset(Dataset):
    def __init__(self, root: Path, train: bool, band: str = "all",
                 polarization: str = "all", depression: str = "all") -> None:
        self.root, self.train = Path(root), train
        if band not in {"all", *BANDS}:
            raise ValueError(f"unsupported band: {band}")
        if polarization not in {"all", *POLS}:
            raise ValueError(f"unsupported polarization: {polarization}")
        if depression not in {"all", *(str(value) for value in DEPRESSIONS)}:
            raise ValueError(f"unsupported depression: {depression}")
        self.band = band
        self.polarization = polarization
        self.depression = depression
        self.records: list[tuple[Path, int, int, int, int, int]] = []
        for class_id, class_name in enumerate(SOC40_CLASSES):
            paths = sorted((self.root / class_name).glob("*.tif"))
            if not paths:
                raise RuntimeError(f"missing TIFFs for {class_name} under {root}")
            for path in paths:
                match = NAME.match(path.stem)
                if match is None:
                    raise RuntimeError(f"unrecognised SAR filename: {path}")
                band, pol, depression, azimuth = match.groups()
                if self.band != "all" and band.upper() != self.band:
                    continue
                if self.polarization != "all" and pol.upper() != self.polarization:
                    continue
                if self.depression != "all" and int(depression) != int(self.depression):
                    continue
                azimuth_bin = ((int(azimuth) + 15) % 360) // 30
                self.records.append((path, class_id, BANDS.index(band.upper()), POLS.index(pol.upper()),
                                     DEPRESSIONS.index(int(depression)), azimuth_bin))

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _translate(image: torch.Tensor, limit: int = 3) -> torch.Tensor:
        dy, dx = random.randint(-limit, limit), random.randint(-limit, limit)
        if not (dy or dx):
            return image
        padded = F.pad(image, (limit, limit, limit, limit), mode="replicate")
        y0, x0 = limit + dy, limit + dx
        return padded[:, y0:y0 + 64, x0:x0 + 64]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, class_id, band, pol, depression, azimuth = self.records[index]
        with Image.open(path) as source:
            source = source.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
            image = np.asarray(source, dtype=np.float32).copy() / 255.0
        image = torch.from_numpy(image)[None]
        if self.train:
            image = self._translate(image)
            image = image * random.uniform(0.90, 1.10) + random.uniform(-0.025, 0.025)
            # Mild multiplicative noise makes the classifier robust to genuine
            # SAR speckle, without changing image geometry or metadata labels.
            image = image * torch.exp(torch.randn_like(image) * random.uniform(0.0, 0.07))
            if random.random() < 0.12:
                side = random.randint(3, 7)
                y, x = random.randint(0, 64 - side), random.randint(0, 64 - side)
                image[:, y:y + side, x:x + side] = image.mean()
        targets = torch.tensor((class_id, band, pol, depression, azimuth), dtype=torch.long)
        return image.clamp(0, 1), targets


def evaluate(model: SARClassifier64, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.eval()
    total = correct = top5 = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    by_condition: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with torch.inference_mode():
        for image, targets in tqdm(loader, desc="SAR classifier validation", leave=False):
            image, targets = image.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            logits = model(image)
            labels = targets[:, 0]
            prediction = logits.argmax(1)
            loss_sum += criterion(logits, labels).item() * len(labels)
            correct += (prediction == labels).sum().item()
            top5 += (logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
            total += len(labels)
            for index in range(len(labels)):
                key = (f"{BANDS[int(targets[index, 1])]}/{POLS[int(targets[index, 2])]}"
                       f"/{DEPRESSIONS[int(targets[index, 3])]}")
                by_condition[key][0] += 1
                by_condition[key][1] += int((prediction[index] == labels[index]).item())
    metrics = {"loss": loss_sum / total, "top1": correct / total, "top5": top5 / total, "samples": total}
    detail = {key: {"samples": values[0], "top1": values[1] / values[0]} for key, values in by_condition.items()}
    return metrics, detail


def fgsm_adversarial_image(model: SARClassifier64, image: torch.Tensor,
                           labels: torch.Tensor, criterion: nn.Module,
                           epsilon: float) -> torch.Tensor:
    """Create a detached single-step FGSM image without accumulating model grads."""
    if epsilon < 0:
        raise ValueError("FGSM epsilon must be non-negative")
    if epsilon == 0:
        return image.detach()
    was_training = model.training
    model.eval()
    attack_image = image.detach().float().requires_grad_(True)
    try:
        attack_logits = model(attack_image)
        attack_loss = criterion(attack_logits, labels)
        attack_gradient = torch.autograd.grad(
            attack_loss, attack_image, retain_graph=False, only_inputs=True)[0]
    finally:
        model.train(was_training)
    return (image.detach().float() + epsilon * attack_gradient.sign()).clamp(0.0, 1.0).detach()


def pgd_adversarial_image(model: SARClassifier64, image: torch.Tensor,
                          labels: torch.Tensor, criterion: nn.Module,
                          epsilon: float, step_size: float, steps: int) -> torch.Tensor:
    """Run deterministic zero-start projected sign-gradient ascent in image space."""
    if epsilon < 0 or step_size <= 0 or steps <= 0:
        raise ValueError("PGD epsilon must be non-negative, with positive step size and steps")
    clean = image.detach().float()
    adversarial = clean.clone()
    for _ in range(steps):
        adversarial.requires_grad_(True)
        logits = model(adversarial)
        loss = criterion(logits, labels)
        gradient = torch.autograd.grad(loss, adversarial, retain_graph=False)[0]
        with torch.no_grad():
            adversarial = adversarial + step_size * gradient.sign()
            adversarial = torch.maximum(torch.minimum(adversarial, clean + epsilon), clean - epsilon)
            adversarial = adversarial.clamp(0.0, 1.0)
        adversarial = adversarial.detach()
    return adversarial


def evaluate_adversarial(model: SARClassifier64, loader: DataLoader, device: torch.device,
                          epsilon: float, step_size: float, steps: int) -> dict[str, float]:
    """Evaluate clean-label classification under a fixed zero-start PGD protocol."""
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    total = correct = top5 = 0
    loss_sum = 0.0
    with torch.enable_grad():
        for image, targets in tqdm(loader, desc="SAR classifier adversarial validation", leave=False):
            image, targets = image.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            labels = targets[:, 0]
            adversarial = pgd_adversarial_image(
                model, image, labels, criterion, epsilon, step_size, steps)
            with torch.no_grad():
                logits = model(adversarial)
                prediction = logits.argmax(1)
                loss_sum += criterion(logits, labels).item() * len(labels)
                correct += (prediction == labels).sum().item()
                top5 += (logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
            total += len(labels)
    return {"loss": loss_sum / max(total, 1), "top1": correct / max(total, 1),
            "top5": top5 / max(total, 1), "samples": total,
            "epsilon": epsilon, "step_size": step_size, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an image-only native 64px SAR classifier")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--band", choices=("all", "X", "KU"), default="all")
    parser.add_argument("--polarization", choices=("all", "HH", "HV", "VH", "VV"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--aux-weight", type=float, default=0.12)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fgsm-epsilon", "--adversarial-epsilon", dest="fgsm_epsilon",
                        type=float, default=0.0,
                        help="single-step FGSM radius used for every training image; zero preserves V1")
    args = parser.parse_args()
    if args.fgsm_epsilon < 0 or args.fgsm_epsilon > 1:
        raise ValueError("--fgsm-epsilon must be in [0, 1]")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train = SARImageDataset(args.train_root, True, args.band, args.polarization)
    test = SARImageDataset(args.test_root, False, args.band, args.polarization)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    test_loader = DataLoader(test, args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    model = SARClassifier64(len(SOC40_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, min(3, args.epochs // 8))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: ((epoch + 1) / warmup if epoch < warmup else
                                   0.5 * (1 + np.cos(np.pi * (epoch - warmup + 1) / max(1, args.epochs - warmup + 1)))))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and not args.no_amp)
    class_loss = nn.CrossEntropyLoss(label_smoothing=0.03)
    aux_loss = nn.CrossEntropyLoss()
    start_epoch, best_top1 = 1, -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"]); scaler.load_state_dict(saved.get("scaler", {}))
        start_epoch, best_top1 = int(saved["epoch"]) + 1, float(saved["best_top1"])
    history = args.output / "history.csv"
    if start_epoch == 1:
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(("epoch", "train_loss", "train_top1", "test_loss", "test_top1", "test_top5", "lr"))
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); loss_sum = correct = total = 0
        for image, targets in tqdm(train_loader, desc=f"SAR native classifier {epoch}/{args.epochs}"):
            image, targets = image.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.fgsm_epsilon:
                image_for_loss = fgsm_adversarial_image(
                    model, image, targets[:, 0], class_loss, args.fgsm_epsilon)
            else:
                image_for_loss = image
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits, features = model(image_for_loss, return_features=True)
                auxiliary = model.auxiliary_logits(features)
                loss = class_loss(logits, targets[:, 0])
                loss = loss + args.aux_weight * sum(
                    aux_loss(logit, target) for logit, target in zip(auxiliary, targets[:, 1:].unbind(1))
                ) / len(auxiliary)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.detach().item() * len(targets)
            correct += (logits.argmax(1) == targets[:, 0]).sum().item(); total += len(targets)
        metrics, conditions = evaluate(model, test_loader, device)
        row = (epoch, loss_sum / total, correct / total, metrics["loss"], metrics["top1"], metrics["top5"], optimizer.param_groups[0]["lr"])
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        scheduler.step()
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch, "best_top1": max(best_top1, metrics["top1"]),
                 "classes": list(SOC40_CLASSES), "input_size": 64, "metrics": metrics}
        if args.fgsm_epsilon:
            state["adversarial_training"] = {
                "method": "fgsm", "epsilon": args.fgsm_epsilon,
                "random_start": False, "attack_label_smoothing": 0.03,
                "final_loss": "original_class_plus_aux_on_adversarial",
            }
        torch.save(state, args.output / "latest.pt")
        if metrics["top1"] >= best_top1:
            best_top1 = metrics["top1"]; state["best_top1"] = best_top1
            torch.save(state, args.output / "best.pt")
            (args.output / "best_test_metrics.json").write_text(json.dumps({**metrics, "by_condition": conditions}, indent=2), encoding="utf-8")
        print(dict(zip(("epoch", "train_loss", "train_top1", "test_loss", "test_top1", "test_top5", "lr"), row)), flush=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"train_samples": len(train), "test_samples": len(test), "classes": list(SOC40_CLASSES), "best_top1": best_top1,
                   "input_policy": "SAR intensity image only; filename metadata is supervision only",
                   "adversarial_training": ({
                       "method": "fgsm", "epsilon": args.fgsm_epsilon,
                       "random_start": False, "attack_label_smoothing": 0.03,
                       "final_loss": "original_class_plus_aux_on_adversarial",
                   } if args.fgsm_epsilon else None)})
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

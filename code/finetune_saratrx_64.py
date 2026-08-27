"""Adapt the supplied legacy SOC classifier to direct 64x64 cut-ROI classification."""
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from saratrx import SOC40_CLASSES, load_saratrx


class CutROIDataset(Dataset):
    def __init__(self, root: Path, train: bool, input_size: int = 64) -> None:
        self.root, self.train, self.input_size = Path(root), train, input_size
        records = []
        for label, class_name in enumerate(SOC40_CLASSES):
            folder = self.root / class_name
            if not folder.is_dir():
                raise RuntimeError(f"missing classifier class folder: {folder}")
            records.extend((path, label) for path in folder.glob("*.tif"))
        if not records:
            raise RuntimeError(f"no TIFF cut ROIs under {root}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.records[index]
        with Image.open(path) as image:
            image = image.convert("L").resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array)[None]
        if self.train:
            if random.random() < 0.5:
                tensor = tensor.flip(2)
            tensor = tensor * random.uniform(0.88, 1.12) + random.uniform(-0.04, 0.04)
            tensor = tensor + torch.randn_like(tensor) * random.uniform(0.0, 0.02)
            if random.random() < 0.25:
                shift_y, shift_x = random.randint(-3, 3), random.randint(-3, 3)
                shifted = torch.zeros_like(tensor)
                src_y0, src_y1 = max(0, -shift_y), min(self.input_size, self.input_size - shift_y)
                src_x0, src_x1 = max(0, -shift_x), min(self.input_size, self.input_size - shift_x)
                dst_y0, dst_y1 = max(0, shift_y), min(self.input_size, self.input_size + shift_y)
                dst_x0, dst_x1 = max(0, shift_x), min(self.input_size, self.input_size + shift_x)
                shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = tensor[:, src_y0:src_y1, src_x0:src_x1]
                tensor = shifted
        return tensor.clamp(0, 1).repeat(3, 1, 1), label


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    loss_sum = correct = top5 = total = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="SARATR-X-64 validation", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            top5 += (logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
            total += labels.numel()
    return loss_sum / total, correct / total, top5 / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SOC_40classes.pth on direct 64x64 cut ROIs")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="legacy SOC weights or a resumed native-64 checkpoint")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=64, choices=(64,))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train = CutROIDataset(args.train_root, True, args.input_size)
    test = CutROIDataset(args.test_root, False, args.input_size)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers,
                              persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test, args.batch_size, shuffle=False, num_workers=args.workers,
                             persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    model = load_saratrx(args.checkpoint, device=device, freeze=False, input_size=args.input_size)
    head_names = ("absolute_pos_embed", "fc_norm", "head")
    backbone = [parameter for name, parameter in model.named_parameters()
                if not name.startswith(head_names)]
    head = [parameter for name, parameter in model.named_parameters()
            if name.startswith(head_names)]
    optimizer = torch.optim.AdamW((
        {"params": backbone, "lr": args.backbone_lr},
        {"params": head, "lr": args.head_lr},
    ), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    start_epoch, best_top1 = 1, -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, best_top1 = int(saved["epoch"]) + 1, float(saved["best_top1"])
    history = args.output / "history.csv"
    if start_epoch == 1:
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(("epoch", "train_loss", "train_top1", "test_loss",
                                         "test_top1", "test_top5", "backbone_lr", "head_lr"))
        baseline = evaluate(model, test_loader, device)
        print(f"original/interpolated 64px baseline: loss={baseline[0]:.4f}, "
              f"top1={baseline[1]:.4f}, top5={baseline[2]:.4f}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); loss_sum = correct = total = 0.0
        progress = tqdm(train_loader, desc=f"SARATR-X-64 epoch {epoch}/{args.epochs}")
        for images, labels in progress:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.detach().item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item(); total += labels.numel()
            progress.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / total:.3f}")
        test_loss, test_top1, test_top5 = evaluate(model, test_loader, device)
        row = (epoch, loss_sum / total, correct / total, test_loss, test_top1, test_top5,
               optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        scheduler.step()
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch,
                 "best_top1": max(best_top1, test_top1), "input_size": args.input_size,
                 "classes": list(SOC40_CLASSES), "metrics": {"top1": test_top1, "top5": test_top5}}
        torch.save(state, args.output / "latest.pt")
        if test_top1 >= best_top1:
            best_top1 = test_top1; state["best_top1"] = best_top1
            torch.save(state, args.output / "best.pt")
        print(dict(zip(("epoch", "train_loss", "train_top1", "test_loss", "test_top1",
                        "test_top5", "backbone_lr", "head_lr"), row)), flush=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"train_samples": len(train), "test_samples": len(test),
                   "classes": list(SOC40_CLASSES), "best_top1": best_top1})
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

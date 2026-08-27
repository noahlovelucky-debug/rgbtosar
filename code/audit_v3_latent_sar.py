"""Strict v3.0 synthetic-to-real classifier audit and final visualisation.

Validation is used for classifier checkpoint selection.  The supplied test
directory is loaded only after selection, once, for the final report.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES
from v3_latent_sar import (LatentDenoiser, LatentDiffusion, RGBSpatialConditioner, SARAutoencoder,
                           V3PairDataset, build_manifest, save_visual_grid)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v3.0 generated-SAR classifier audit")
    parser.add_argument("--mode", choices=("reconstruction", "diffusion"), required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--sar-test-root", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_test_manifest(root: Path) -> dict:
    samples = []
    for name in SOC40_CLASSES:
        samples.extend(str(path.relative_to(root)) for path in sorted((root / name).glob("X_HH_*.tif")))
    return {"train": samples, "validation": []}


def loader(dataset, batch_size, workers, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


def mild_augment(image: torch.Tensor) -> torch.Tensor:
    gains = image.new_empty(len(image), 1, 1, 1).uniform_(.94, 1.06)
    bias = image.new_empty(len(image), 1, 1, 1).uniform_(-.02, .02)
    return (image * gains + bias).clamp(0, 1)


def instantiate(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    ae = SARAutoencoder().to(device); ae.load_state_dict(state["autoencoder"]); ae.eval()
    conditioner = RGBSpatialConditioner().to(device); conditioner.load_state_dict(state["conditioner"]); conditioner.eval()
    denoiser = LatentDenoiser().to(device); denoiser.load_state_dict(state["denoiser"]); denoiser.eval()
    for model in (ae, conditioner, denoiser):
        for parameter in model.parameters(): parameter.requires_grad_(False)
    diffusion = LatentDiffusion(timesteps=int(state["timesteps"])).to(device)
    return ae, conditioner, denoiser, diffusion, state["latent_mean"].to(device), state["latent_std"].to(device), state


@torch.inference_mode()
def generated(batch, mode, ae, conditioner, denoiser, diffusion, mean, std, sample_steps, seed):
    device = mean.device
    sar, rgb = batch["sar"].to(device, non_blocking=True), batch["rgb"].to(device, non_blocking=True)
    labels, vector = batch["class_id"].to(device, non_blocking=True), batch["condition"].to(device, non_blocking=True)
    if mode == "reconstruction":
        return ae.decode(ae.encode(sar)), sar, rgb
    condition = conditioner(rgb, labels, vector)
    latent = diffusion.sample(denoiser, condition, sample_steps, seed=seed)
    return ae.decode(latent * std + mean), sar, rgb


@torch.inference_mode()
def evaluate_generated(classifier, dataset, mode, components, batch_size, workers, sample_steps, device, seed):
    classifier.eval(); total = correct = 0; loss_sum = 0.; criterion = nn.CrossEntropyLoss()
    by_dep = {index: [0, 0] for index in range(4)}
    for index, batch in enumerate(tqdm(loader(dataset, batch_size, workers, False), desc="v3 generated validation", leave=False)):
        fake, _, _ = generated(batch, mode, *components[:-1], sample_steps, seed + index)
        labels = batch["class_id"].to(device); logits = classifier((fake + 1) * .5)
        prediction = logits.argmax(1); loss_sum += criterion(logits, labels).item() * len(labels)
        correct += (prediction == labels).sum().item(); total += len(labels)
        for dep in range(4):
            mask = batch["depression"] == dep; by_dep[dep][0] += int(mask.sum()); by_dep[dep][1] += int((prediction.cpu()[mask] == labels.cpu()[mask]).sum())
    return {"loss": loss_sum / total, "top1": correct / total, "samples": total,
            "by_depression": {str((15, 30, 45, 60)[key]): {"samples": n, "top1": right / n}
                              for key, (n, right) in by_dep.items()}}


@torch.inference_mode()
def final_test(classifier, test_data, device):
    classifier.eval(); total = correct = top5 = 0; loss_sum = 0.; criterion = nn.CrossEntropyLoss()
    by_dep = {index: [0, 0] for index in range(4)}
    for batch in tqdm(loader(test_data, 256, 4, False), desc="v3 final real test"):
        image, labels = (batch["sar"].to(device) + 1) * .5, batch["class_id"].to(device)
        logits = classifier(image); prediction = logits.argmax(1); loss_sum += criterion(logits, labels).item() * len(labels)
        correct += (prediction == labels).sum().item(); top5 += (logits.topk(5, 1).indices == labels[:, None]).any(1).sum().item(); total += len(labels)
        for dep in range(4):
            mask = batch["depression"] == dep; by_dep[dep][0] += int(mask.sum()); by_dep[dep][1] += int((prediction.cpu()[mask] == labels.cpu()[mask]).sum())
    return {"loss": loss_sum / total, "top1": correct / total, "top5": top5 / total, "samples": total,
            "by_depression": {str((15, 30, 45, 60)[key]): {"samples": n, "top1": right / n}
                              for key, (n, right) in by_dep.items()}}


def main() -> None:
    args = arguments(); random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train = V3PairDataset(args.rgb_root, args.sar_train_root, manifest, "train", augment_rgb=False)
    validation = V3PairDataset(args.rgb_root, args.sar_train_root, manifest, "validation")
    test = V3PairDataset(args.rgb_root, args.sar_test_root, make_test_manifest(args.sar_test_root), "train")
    components = instantiate(args.diffusion_checkpoint, device)
    classifier = SARClassifier64(len(SOC40_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=.03)
    history = args.output / "history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(("epoch", "generated_train_loss", "generated_train_top1", "generated_validation_top1", "lr"))
    best = -1.
    for epoch in range(1, args.epochs + 1):
        classifier.train(); loss_sum = 0.; total = correct = 0
        for index, batch in enumerate(tqdm(loader(train, args.batch_size, args.workers, True), desc=f"v3 {args.mode} classifier {epoch}/{args.epochs}")):
            fake, _, _ = generated(batch, args.mode, *components[:-1], args.sample_steps, args.seed + epoch * 10000 + index)
            labels = batch["class_id"].to(device); image = mild_augment((fake + 1) * .5)
            optimizer.zero_grad(set_to_none=True); logits = classifier(image); loss = criterion(logits, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.); optimizer.step()
            loss_sum += loss.detach().item() * len(labels); correct += (logits.argmax(1) == labels).sum().item(); total += len(labels)
        validation_metrics = evaluate_generated(classifier, validation, args.mode, components, args.batch_size,
                                                args.workers, args.sample_steps, device, args.seed)
        row = (epoch, loss_sum / total, correct / total, validation_metrics["top1"], optimizer.param_groups[0]["lr"])
        with history.open("a", newline="", encoding="utf-8") as handle: csv.writer(handle).writerow(row)
        state = {"architecture": "SARClassifier64", "mode": args.mode, "epoch": epoch,
                 "model": classifier.state_dict(), "validation": validation_metrics,
                 "diffusion_checkpoint": str(args.diffusion_checkpoint), "test_policy": "not used for selection"}
        torch.save(state, args.output / "latest.pt")
        if validation_metrics["top1"] > best:
            best = validation_metrics["top1"]; torch.save(state, args.output / "best.pt")
        scheduler.step(); print(dict(zip(("epoch", "train_loss", "train_top1", "val_top1", "lr"), row)), flush=True)
    best_state = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    classifier.load_state_dict(best_state["model"])
    metrics = final_test(classifier, test, device)
    (args.output / "final_real_test.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # The final test set is visualised only after model selection is complete.
    batch = next(iter(loader(test, 8, 0, False)))
    fake, real, rgb = generated(batch, args.mode, *components[:-1], args.sample_steps, args.seed)
    reconstruction = components[0].decode(components[0].encode(real))
    save_visual_grid(args.output / "final_test_visualisation.png", rgb, real, reconstruction, fake)
    print(json.dumps({"best_validation_top1": best, "final_real_test": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()

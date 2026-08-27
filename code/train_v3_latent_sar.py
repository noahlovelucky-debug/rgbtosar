"""Train v3.0 in two stages: SAR autoencoder, then RGB-conditioned latent diffusion."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from v3_latent_sar import (LatentDenoiser, LatentDiffusion, RGBSpatialConditioner, SARAutoencoder,
                           V3PairDataset, build_manifest, sar_reconstruction_loss, save_visual_grid)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v3.0 real-SAR latent RGB-to-SAR training")
    parser.add_argument("--stage", choices=("autoencoder", "diffusion"), required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--limit-train-batches", type=int, default=0,
                        help="non-zero only for a short smoke test")
    parser.add_argument("--limit-validation-batches", type=int, default=0,
                        help="non-zero only for a short smoke test")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def loader(dataset, batch_size, workers, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


def main() -> None:
    args = arguments()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); amp = device.type == "cuda" and not args.no_amp
    manifest = build_manifest(args.sar_train_root, args.output / "split_manifest.json",
                              args.validation_fraction, args.seed)
    train = V3PairDataset(args.rgb_root, args.sar_train_root, manifest, "train", augment_rgb=args.stage == "diffusion",
                          load_rgb=args.stage == "diffusion")
    validation = V3PairDataset(args.rgb_root, args.sar_train_root, manifest, "validation",
                               load_rgb=args.stage == "diffusion")
    train_loader = loader(train, args.batch_size, args.workers, True)
    val_loader = loader(validation, args.batch_size, args.workers, False)
    (args.output / "data_summary.json").write_text(json.dumps({"train": len(train), "validation": len(validation),
        "test_policy": "original SOC_40classes_cut/test is untouched until final audit"}, indent=2), encoding="utf-8")
    if args.stage == "autoencoder":
        train_autoencoder(args, train_loader, val_loader, device, amp)
    else:
        if args.ae_checkpoint is None:
            raise ValueError("--ae-checkpoint is required for diffusion stage")
        train_diffusion(args, train_loader, val_loader, device, amp)


def train_autoencoder(args, train_loader, val_loader, device, amp) -> None:
    model = SARAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    history = args.output / "autoencoder_history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(("epoch", "train_loss", "validation_loss", "validation_l1", "validation_gradient", "validation_moments"))
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.; total = 0
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"v3 autoencoder {epoch}/{args.epochs}")):
            image = batch["sar"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                reconstruction, _ = model(image); loss, _ = sar_reconstruction_loss(reconstruction, image)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer); scaler.update(); train_loss += loss.detach().item() * len(image); total += len(image)
            if args.limit_train_batches and batch_index + 1 >= args.limit_train_batches:
                break
        train_total = total
        model.eval(); validation_loss = 0.; values = {"l1": 0., "gradient": 0., "moments": 0.}; total = 0; preview = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(val_loader):
                image = batch["sar"].to(device, non_blocking=True); reconstruction, _ = model(image); loss, parts = sar_reconstruction_loss(reconstruction, image)
                validation_loss += loss.item() * len(image); total += len(image)
                for key, value in parts.items(): values[key] += value.item() * len(image)
                if preview is None: preview = (batch["rgb"].to(device), image, reconstruction)
                if args.limit_validation_batches and batch_index + 1 >= args.limit_validation_batches:
                    break
        row = (epoch, train_loss / train_total, validation_loss / total,
               values["l1"] / total, values["gradient"] / total, values["moments"] / total)
        with history.open("a", newline="", encoding="utf-8") as handle: csv.writer(handle).writerow(row)
        state = {"architecture": "v3_sar_autoencoder", "epoch": epoch, "model": model.state_dict(),
                 "latent_channels": 16, "validation_loss": row[2], "split_manifest": str(args.output / "split_manifest.json")}
        torch.save(state, args.output / "autoencoder_latest.pt")
        if row[2] < best:
            best = row[2]; torch.save(state, args.output / "autoencoder_best.pt")
        if epoch == 1 or epoch % 5 == 0:
            assert preview is not None
            save_visual_grid(args.output / f"autoencoder_validation_{epoch:03d}.png", *preview)
        print(dict(zip(("epoch", "train", "val", "l1", "gradient", "moments"), row)), flush=True)


def train_diffusion(args, train_loader, val_loader, device, amp) -> None:
    ae_state = torch.load(args.ae_checkpoint, map_location=device, weights_only=False)
    autoencoder = SARAutoencoder().to(device); autoencoder.load_state_dict(ae_state["model"]); autoencoder.eval()
    for parameter in autoencoder.parameters(): parameter.requires_grad_(False)
    stats_path = args.ae_checkpoint.parent / "latent_stats.pt"
    if not stats_path.is_file():
        raise RuntimeError(f"missing latent normalisation statistics: {stats_path}")
    latent_stats = torch.load(stats_path, map_location=device, weights_only=True)
    latent_mean, latent_std = latent_stats["mean"].to(device), latent_stats["std"].to(device)
    conditioner = RGBSpatialConditioner().to(device)
    denoiser = LatentDenoiser().to(device)
    diffusion = LatentDiffusion(timesteps=args.timesteps).to(device)
    optimizer = torch.optim.AdamW((*conditioner.parameters(), *denoiser.parameters()), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    history = args.output / "diffusion_history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(("epoch", "train_noise_mse", "validation_noise_mse", "validation_latent_cosine"))
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        conditioner.train(); denoiser.train(); loss_sum = 0.; total = 0
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"v3 latent diffusion {epoch}/{args.epochs}")):
            sar = batch["sar"].to(device, non_blocking=True); rgb = batch["rgb"].to(device, non_blocking=True)
            classes = batch["class_id"].to(device, non_blocking=True); condition_vector = batch["condition"].to(device, non_blocking=True)
            with torch.inference_mode(): latent = (autoencoder.encode(sar) - latent_mean) / latent_std
            timestep = torch.randint(args.timesteps, (len(sar),), device=device); noise = torch.randn_like(latent)
            noisy = diffusion.noisy(latent, timestep, noise)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                condition = conditioner(rgb, classes, condition_vector)
                prediction = denoiser(noisy, timestep, condition)
                loss = F.mse_loss(prediction, noise)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_((*conditioner.parameters(), *denoiser.parameters()), 5.)
            scaler.step(optimizer); scaler.update(); loss_sum += loss.detach().item() * len(sar); total += len(sar)
            if args.limit_train_batches and batch_index + 1 >= args.limit_train_batches:
                break
        train_total = total
        conditioner.eval(); denoiser.eval(); validation = 0.; cosine = 0.; total = 0; preview = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(val_loader):
                sar = batch["sar"].to(device); rgb = batch["rgb"].to(device); classes = batch["class_id"].to(device); vector = batch["condition"].to(device)
                raw_latent = autoencoder.encode(sar); latent = (raw_latent - latent_mean) / latent_std
                timestep = torch.randint(args.timesteps, (len(sar),), device=device); noise = torch.randn_like(latent)
                condition = conditioner(rgb, classes, vector); prediction = denoiser(diffusion.noisy(latent, timestep, noise), timestep, condition)
                validation += F.mse_loss(prediction, noise).item() * len(sar); total += len(sar)
                if preview is None:
                    generated_latent = diffusion.sample(denoiser, condition[:8], args.sample_steps, seed=args.seed)
                    preview = (rgb[:8], sar[:8], autoencoder.decode(raw_latent[:8]),
                               autoencoder.decode(generated_latent * latent_std + latent_mean))
                    cosine = F.cosine_similarity(generated_latent.flatten(1), latent[:8].flatten(1), dim=1).sum().item()
                if args.limit_validation_batches and batch_index + 1 >= args.limit_validation_batches:
                    break
        row = (epoch, loss_sum / train_total, validation / total, cosine / min(8, total))
        with history.open("a", newline="", encoding="utf-8") as handle: csv.writer(handle).writerow(row)
        state = {"architecture": "v3_rgb_conditioned_latent_diffusion", "epoch": epoch,
                 "autoencoder_checkpoint": str(args.ae_checkpoint), "autoencoder": autoencoder.state_dict(),
                 "conditioner": conditioner.state_dict(), "denoiser": denoiser.state_dict(), "timesteps": args.timesteps,
                 "latent_mean": latent_mean.cpu(), "latent_std": latent_std.cpu(),
                 "validation_noise_mse": row[2], "split_manifest": str(args.output / "split_manifest.json")}
        torch.save(state, args.output / "diffusion_latest.pt")
        if row[2] < best:
            best = row[2]; torch.save(state, args.output / "diffusion_best.pt")
        if epoch == 1 or epoch % 5 == 0:
            assert preview is not None
            save_visual_grid(args.output / f"diffusion_validation_{epoch:03d}.png", *preview)
        print(dict(zip(("epoch", "train", "val", "latent_cosine"), row)), flush=True)


if __name__ == "__main__":
    main()

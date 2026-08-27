"""Train v4 fast SPADE RGB-to-SAR GAN with strict train/validation selection."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES
from v3_latent_sar import SARAutoencoder, V3PairDataset, build_manifest, save_visual_grid
from v4_spade_gan import MultiScaleProjectionDiscriminator, SARSPADEGenerator


def arguments():
    parser = argparse.ArgumentParser(description="v4 fast SPADE RGB-to-SAR GAN")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--generator-lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--class-weight", type=float, default=.08,
                        help="low-weight frozen-teacher class constraint; it must not dominate realism")
    parser.add_argument("--teacher-feature-weight", type=float, default=.35,
                        help="real-domain classifier feature prototype constraint")
    parser.add_argument("--r1-weight", type=float, default=2.0)
    parser.add_argument("--r1-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--limit-train-batches", type=int, default=0, help="smoke-test only")
    parser.add_argument("--limit-validation-batches", type=int, default=0, help="smoke-test only")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def make_loader(dataset, batch_size, workers, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


def set_grad(model, enabled):
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def teacher_view(image: torch.Tensor) -> torch.Tensor:
    """Differentiable mild EOT for the frozen classifier.

    A fake image that is classified only through a single pixel-level cue will
    not remain reliably classified after these SAR-plausible intensity and
    sub-pixel perturbations.  This is deliberately *not* applied to the
    discriminator, whose job is to inspect the native generated image.
    """
    gain = image.new_empty(len(image), 1, 1, 1).uniform_(.94, 1.06)
    bias = image.new_empty(len(image), 1, 1, 1).uniform_(-.025, .025)
    # In [0,1] intensity space, then map back to the generator's [-1,1] space.
    view = ((image + 1) * .5 * gain + bias).clamp(0, 1)
    return view


def prepare_prototypes(autoencoder, dataset, device):
    sums = torch.zeros(40, 4, 16, 8, 8, device=device); counts = torch.zeros(40, 4, device=device)
    with torch.inference_mode():
        for batch in tqdm(make_loader(dataset, 256, 4, False), desc="v4 latent prototypes"):
            z = autoencoder.encode(batch["sar"].to(device)); labels, dep = batch["class_id"].to(device), batch["depression"].to(device)
            sums.index_put_((labels, dep), z, accumulate=True)
            counts.index_put_((labels, dep), torch.ones_like(labels, dtype=torch.float), accumulate=True)
    return sums / counts[:, :, None, None, None].clamp_min(1)


def prepare_teacher_feature_prototypes(judge, dataset, device):
    """Class/depression centres from real SAR only; no generated sample enters."""
    sums = torch.zeros(40, 4, judge.feature_dim, device=device)
    counts = torch.zeros(40, 4, device=device)
    with torch.inference_mode():
        for batch in tqdm(make_loader(dataset, 256, 4, False), desc="v4 teacher feature prototypes"):
            _, features = judge((batch["sar"].to(device) + 1) * .5, return_features=True)
            labels, dep = batch["class_id"].to(device), batch["depression"].to(device)
            sums.index_put_((labels, dep), features, accumulate=True)
            counts.index_put_((labels, dep), torch.ones_like(labels, dtype=torch.float), accumulate=True)
    return sums / counts[:, :, None].clamp_min(1)


def main():
    args = arguments(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    amp = device.type == "cuda"
    manifest = build_manifest(args.sar_train_root, args.output / "split_manifest.json", args.validation_fraction, args.seed)
    train = V3PairDataset(args.rgb_root, args.sar_train_root, manifest, "train", augment_rgb=True)
    validation = V3PairDataset(args.rgb_root, args.sar_train_root, manifest, "validation")
    train_loader, validation_loader = make_loader(train, args.batch_size, args.workers, True), make_loader(validation, args.batch_size, args.workers, False)
    ae_state = torch.load(args.ae_checkpoint, map_location=device, weights_only=False)
    autoencoder = SARAutoencoder().to(device); autoencoder.load_state_dict(ae_state["model"]); autoencoder.eval()
    judge_state = torch.load(args.native_classifier_checkpoint, map_location=device, weights_only=False)
    judge = SARClassifier64(40).to(device); judge.load_state_dict(judge_state["model"]); judge.eval()
    for model in (autoencoder, judge):
        for parameter in model.parameters(): parameter.requires_grad_(False)
    prototypes = prepare_prototypes(autoencoder, train, device)
    teacher_feature_prototypes = prepare_teacher_feature_prototypes(judge, train, device)
    generator = SARSPADEGenerator().to(device); discriminator = MultiScaleProjectionDiscriminator().to(device)
    generator_opt = torch.optim.Adam(generator.parameters(), lr=args.generator_lr, betas=(.0, .99))
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(.0, .99))
    generator_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    discriminator_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ce = nn.CrossEntropyLoss(label_smoothing=.02); history = args.output / "history.csv"
    start_epoch, best = 1, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(saved["generator"]); discriminator.load_state_dict(saved["discriminator"])
        generator_opt.load_state_dict(saved["generator_optimizer"]); discriminator_opt.load_state_dict(saved["discriminator_optimizer"])
        generator_scaler.load_state_dict(saved.get("generator_scaler", {})); discriminator_scaler.load_state_dict(saved.get("discriminator_scaler", {}))
        start_epoch, best = int(saved["epoch"]) + 1, float(saved["best_quality"])
    if start_epoch == 1:
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(("epoch", "generator", "adversarial", "latent", "class", "teacher_feature", "discriminator", "r1", "fake_accuracy", "feature", "validation_latent", "validation_class_accuracy"))
    for epoch in range(start_epoch, args.epochs + 1):
        generator.train(); discriminator.train(); total = torch.zeros(9, dtype=torch.float64); steps = 0
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"v4 SPADE GAN {epoch}/{args.epochs}")):
            rgb, real = batch["rgb"].to(device), batch["sar"].to(device)
            labels, geometry, dep = batch["class_id"].to(device), batch["condition"].to(device), batch["depression"].to(device)
            noise = torch.randn(len(real), generator.noise_dim, device=device)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                fake = generator(rgb, labels, geometry, noise)
            discriminator_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                real_for_d = real.requires_grad_(args.r1_weight > 0 and (steps % args.r1_every == 0))
                real_score, _ = discriminator(real_for_d, labels, geometry)
                fake_score, _ = discriminator(fake.detach(), labels, geometry)
                wrong_labels = labels.roll(1)
                wrong_label_score, _ = discriminator(real, wrong_labels, geometry)
                wrong_geometry_score, _ = discriminator(real, labels, geometry.roll(1, 0))
                r1 = real_score.new_zeros(())
                if real_for_d.requires_grad:
                    gradient = torch.autograd.grad(real_score.sum(), real_for_d, create_graph=True)[0]
                    r1 = gradient.flatten(1).square().sum(1).mean()
                disc_loss = (F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
                             + .35 * F.relu(1 + wrong_label_score).mean()
                             + .35 * F.relu(1 + wrong_geometry_score).mean()
                             + (args.r1_weight * .5 * args.r1_every) * r1)
            discriminator_scaler.scale(disc_loss).backward(); discriminator_scaler.unscale_(discriminator_opt)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.); discriminator_scaler.step(discriminator_opt); discriminator_scaler.update()
            set_grad(discriminator, False); generator_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                fake_score, fake_features = discriminator(fake, labels, geometry)
                with torch.no_grad():
                    _, real_features = discriminator(real, labels, geometry)
                    real_latent = autoencoder.encode(real)
                fake_latent = autoencoder.encode(fake)
                latent_loss = F.smooth_l1_loss(fake_latent, real_latent) + .25 * F.smooth_l1_loss(fake_latent, prototypes[labels, dep])
                logits, teacher_features = judge(teacher_view(fake), return_features=True)
                class_loss = ce(logits, labels)
                teacher_feature_loss = F.smooth_l1_loss(teacher_features, teacher_feature_prototypes[labels, dep])
                feature_loss = sum(F.l1_loss(f.mean((2, 3)), r.mean((2, 3))) for f, r in zip(fake_features, real_features))
                adv = -fake_score.mean()
                generator_loss = (2 * adv + 3 * latent_loss + args.class_weight * class_loss
                                  + args.teacher_feature_weight * teacher_feature_loss + .5 * feature_loss)
            generator_scaler.scale(generator_loss).backward(); generator_scaler.unscale_(generator_opt)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.); generator_scaler.step(generator_opt); generator_scaler.update(); set_grad(discriminator, True)
            values = (generator_loss, adv, latent_loss, class_loss, teacher_feature_loss, disc_loss, r1,
                      (logits.argmax(1) == labels).float().mean(), feature_loss)
            total += torch.tensor([value.detach().item() for value in values], dtype=torch.float64); steps += 1
            if args.limit_train_batches and batch_index + 1 >= args.limit_train_batches:
                break
        generator.eval(); val_latent = val_correct = val_total = 0.; preview = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(validation_loader):
                rgb, real = batch["rgb"].to(device), batch["sar"].to(device); labels, geometry = batch["class_id"].to(device), batch["condition"].to(device)
                fake = generator(rgb, labels, geometry, torch.zeros(len(real), generator.noise_dim, device=device))
                val_latent += F.smooth_l1_loss(autoencoder.encode(fake), autoencoder.encode(real)).item() * len(real)
                prediction = judge((fake + 1) * .5).argmax(1); val_correct += (prediction == labels).sum().item(); val_total += len(real)
                if preview is None: preview = (rgb, real, fake)
                if args.limit_validation_batches and batch_index + 1 >= args.limit_validation_batches:
                    break
        averages = (total / steps).tolist(); val_latent /= val_total; val_accuracy = val_correct / val_total
        row = (epoch, *averages, val_latent, val_accuracy)
        with history.open("a", newline="", encoding="utf-8") as handle: csv.writer(handle).writerow(row)
        state = {"architecture": "v4_sar_spade_projection_gan", "epoch": epoch, "generator": generator.state_dict(),
                 "discriminator": discriminator.state_dict(), "validation_latent": val_latent, "validation_class_accuracy": val_accuracy,
                 "generator_optimizer": generator_opt.state_dict(), "discriminator_optimizer": discriminator_opt.state_dict(),
                 "generator_scaler": generator_scaler.state_dict(), "discriminator_scaler": discriminator_scaler.state_dict(),
                 "best_quality": min(best, val_latent + .05 * (1 - val_accuracy)),
                 "split_manifest": str(args.output / "split_manifest.json"), "classes": list(SOC40_CLASSES),
                 "loss_policy": {"class_weight": args.class_weight, "teacher_feature_weight": args.teacher_feature_weight,
                                 "r1_weight": args.r1_weight, "r1_every": args.r1_every}}
        torch.save(state, args.output / "latest.pt")
        quality = val_latent + .05 * (1 - val_accuracy)
        if quality < best:
            best = quality; torch.save(state, args.output / "best.pt")
        if epoch == 1 or epoch % 10 == 0:
            assert preview is not None
            save_visual_grid(args.output / f"validation_{epoch:03d}.png", preview[0], preview[1], preview[2])
        print(dict(zip(("epoch", "generator", "adv", "latent", "class", "teacher_feature", "disc", "r1", "fake_acc", "feature", "val_latent", "val_acc"), row)), flush=True)


if __name__ == "__main__":
    main()

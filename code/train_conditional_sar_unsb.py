"""Train the UNSB-inspired conditional, unpaired RGB-to-SAR bridge.

This is the faithful experimental path recommended for this repository:
successive stochastic bridge refinement, a conditional projection PatchD, an
energy estimator for the SB regularizer, and structure-only PatchNCE.  The
RGB/SAR pair is class/acquisition matched but never pixel registered.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from hifc_unpaired_sar_gan import condition_from_batch
from joint_data import JointROIDataset
from unsb_sar_bridge import (
    UNSB_SAR_UNPAIRED_ARCHITECTURE, BridgeDiscriminator, BridgeEnergy,
    BridgePatchEncoder, SilhouetteBridge, bridge_sample, patch_nce_loss,
    sb_energy_loss, soft_silhouette_prior,
)

# All bridge tensors have fixed spatial sizes.  Let cuDNN select and reuse
# the fastest convolution kernels; TF32 only affects float32 matmul paths and
# leaves the AMP training path unchanged.
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UNSB-SAR conditional unpaired bridge")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band", choices=("all", "X", "KU"), default="all")
    parser.add_argument("--polarization", choices=("all", "HH", "HV", "VH", "VV"), default="all")
    parser.add_argument("--depression", choices=("all", "15", "30", "45", "60"), default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epoch-size", type=int, default=24_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--control-base", type=int, default=32)
    parser.add_argument("--discriminator-base", type=int, default=32)
    parser.add_argument("--energy-base", type=int, default=16)
    parser.add_argument("--patch-base", type=int, default=16)
    parser.add_argument("--bridge-steps", type=int, default=5)
    parser.add_argument("--trajectory-noise", type=float, default=.05)
    parser.add_argument("--lambda-gan", type=float, default=1.0)
    parser.add_argument("--lambda-sb", type=float, default=.1)
    parser.add_argument("--lambda-nce", type=float, default=1.0)
    parser.add_argument("--sb-tau", type=float, default=.1)
    parser.add_argument("--identity-loss-weight", type=float, default=.1)
    parser.add_argument("--view-consistency-weight", type=float, default=.25)
    parser.add_argument("--wrong-condition-weight", type=float, default=.25)
    parser.add_argument("--lr-generator", type=float, default=2e-4)
    parser.add_argument("--lr-discriminator", type=float, default=2e-4)
    parser.add_argument("--lr-energy", type=float, default=2e-4)
    parser.add_argument("--lr-patch", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--sample-steps", type=int, default=5)
    parser.add_argument("--sample-temperature", type=float, default=.05)
    parser.add_argument("--preview-every", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if not distributed:
        return False, 1, 0, torch.device(args.device)
    if not torch.cuda.is_available() or local_rank < 0:
        raise RuntimeError("torchrun UNSB-SAR training requires CUDA and LOCAL_RANK")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            device_id=torch.device("cuda", local_rank))
    return True, world_size, rank, torch.device("cuda", local_rank)


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def angle_features(angle: torch.Tensor) -> torch.Tensor:
    radians = angle.float() * (np.pi / 180.0)
    return torch.stack((radians.sin(), radians.cos()), dim=1)


def wrap(module: nn.Module, distributed: bool, device: torch.device) -> nn.Module:
    if not distributed:
        return module
    return DDP(module, device_ids=[device.index], output_device=device.index,
               broadcast_buffers=False, find_unused_parameters=False)


def make_loader(dataset: JointROIDataset, batch_size: int, workers: int,
                distributed: bool, rank: int, world_size: int,
                device: torch.device) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True, drop_last=True)
               if distributed else None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None,
                        sampler=sampler, num_workers=workers, drop_last=True,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    return loader, sampler


def panel(image: torch.Tensor, channels: int = 3) -> np.ndarray:
    value = image.detach().cpu().clamp(-1, 1).numpy()
    if value.ndim == 3:
        value = value[0] if value.shape[0] == 1 else value.transpose(1, 2, 0)
    if value.ndim == 2:
        value = np.repeat(value[..., None], channels, axis=2)
    return ((value + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)


def write_preview(path: Path, rgb: torch.Tensor, mask: torch.Tensor,
                  real: torch.Tensor, fake: torch.Tensor) -> None:
    rows = []
    for index in range(min(len(rgb), len(fake))):
        rgb_panel = F.interpolate(rgb[index:index + 1], (64, 64), mode="bilinear",
                                  align_corners=False)[0]
        mask_panel = F.interpolate(mask[index:index + 1], (64, 64), mode="bilinear",
                                   align_corners=False)[0]
        rows.append(np.concatenate((panel(rgb_panel), panel(mask_panel),
                                    panel(real[index]), panel(fake[index])), axis=1))
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(rows, axis=0), "RGB").save(path)


def checkpoint_state(epoch: int, generator: nn.Module, ema: nn.Module,
                     discriminator: nn.Module, energy: nn.Module,
                     patch_encoder: nn.Module, optimizers: dict[str, torch.optim.Optimizer],
                     scaler: torch.amp.GradScaler, args: argparse.Namespace,
                     data_summary: dict[str, object]) -> dict[str, object]:
    return {
        "architecture": UNSB_SAR_UNPAIRED_ARCHITECTURE,
        "epoch": epoch,
        "generator": unwrap(generator).state_dict(),
        "ema_generator": ema.state_dict(),
        "discriminator": unwrap(discriminator).state_dict(),
        "energy": unwrap(energy).state_dict(),
        "patch_encoder": unwrap(patch_encoder).state_dict(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "scaler": scaler.state_dict(),
        "args": vars(args),
        "data_summary": data_summary,
        "method": {
            "base_inspiration": "cyclomon/UNSB",
            "official_commit": "d1f644f7777e19d5afe5aea3e5cb4bd3afd9b88b",
            "source_endpoint": "soft blurred alpha occupancy, no RGB/SAR pixel alignment",
            "losses": "conditional hinge GAN + 0.1 SB energy + 1.0 structure PatchNCE + RGB-only identity audit",
            "native_sar_classifier_gradient": False,
        },
    }


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def main() -> None:
    args = arguments()
    if args.epochs < 1 or args.epoch_size < 1 or args.batch_size < 1 or args.bridge_steps < 1:
        raise ValueError("epochs, epoch-size, batch-size, and bridge-steps must be positive")
    if min(args.lambda_gan, args.lambda_sb, args.lambda_nce, args.identity_loss_weight,
           args.view_consistency_weight, args.wrong_condition_weight, args.trajectory_noise) < 0:
        raise ValueError("loss weights and trajectory noise must be non-negative")
    if not 0 < args.ema_decay < 1 or args.sample_temperature < 0:
        raise ValueError("ema decay must be in (0,1), sample temperature must be non-negative")
    distributed, world_size, rank, device = setup_distributed(args)
    is_main = rank == 0
    seed_everything(args.seed)
    dataset = JointROIDataset(
        args.rgb_root, args.sar_train_root, rgb_size=128, roi_size=64,
        epoch_size=args.epoch_size, band=args.band, polarization=args.polarization,
        depression=args.depression, augment_rgb=True, source_view_mode="nearest",
        return_rgb_mask=True,
    )
    loader, sampler = make_loader(dataset, args.batch_size, args.workers,
                                  distributed, rank, world_size, device)
    generator_core = SilhouetteBridge(base=args.base, token_dim=args.token_dim,
                                      control_base=args.control_base).to(device)
    discriminator_core = BridgeDiscriminator(token_dim=args.token_dim,
                                             base=args.discriminator_base).to(device)
    energy_core = BridgeEnergy(token_dim=args.token_dim, base=args.energy_base).to(device)
    patch_core = BridgePatchEncoder(base=args.patch_base).to(device)
    ema = copy.deepcopy(generator_core).eval().to(device)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    generator = wrap(generator_core, distributed, device)
    discriminator = wrap(discriminator_core, distributed, device)
    energy = wrap(energy_core, distributed, device)
    patch_encoder = wrap(patch_core, distributed, device)
    optimizers = {
        "generator": torch.optim.AdamW(generator.parameters(), lr=args.lr_generator,
                                        weight_decay=args.weight_decay, betas=(.0, .99)),
        "discriminator": torch.optim.AdamW(discriminator.parameters(), lr=args.lr_discriminator,
                                            weight_decay=args.weight_decay, betas=(.0, .99)),
        "energy": torch.optim.AdamW(energy.parameters(), lr=args.lr_energy,
                                     weight_decay=args.weight_decay, betas=(.0, .99)),
        "patch": torch.optim.AdamW(patch_encoder.parameters(), lr=args.lr_patch,
                                    weight_decay=args.weight_decay, betas=(.0, .99)),
    }
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and not args.no_amp)
    start_epoch = 1
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("architecture") != UNSB_SAR_UNPAIRED_ARCHITECTURE:
            raise RuntimeError("resume checkpoint is not an UNSB-SAR G/D/E checkpoint")
        unwrap(generator).load_state_dict(state["generator"])
        ema.load_state_dict(state["ema_generator"])
        unwrap(discriminator).load_state_dict(state["discriminator"])
        unwrap(energy).load_state_dict(state["energy"])
        unwrap(patch_encoder).load_state_dict(state["patch_encoder"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        scaler.load_state_dict(state.get("scaler", {}))
        start_epoch = int(state["epoch"]) + 1
    seed_everything(args.seed + rank)

    data_summary = {**dataset.summary(),
                    "filters": {"band": args.band, "polarization": args.polarization,
                                "depression": args.depression},
                    "global_batch_size": args.batch_size * world_size}
    if is_main:
        args.output.mkdir(parents=True, exist_ok=True)
        config = {**{key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
                  "architecture": UNSB_SAR_UNPAIRED_ARCHITECTURE,
                  "data_summary": data_summary}
        (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        history = args.output / "history.csv"
        if not (args.resume is not None and history.is_file()):
            with history.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(("epoch", "g_total", "g_adv", "g_sb", "g_nce",
                                             "rgb_identity", "d_loss", "e_loss",
                                             "rgb_class_accuracy", "examples_per_second"))
        print({"device": str(device), "world_size": world_size,
               "dataset": data_summary, "architecture": UNSB_SAR_UNPAIRED_ARCHITECTURE}, flush=True)
    else:
        history = args.output / "history.csv"
    if distributed:
        dist.barrier()

    preview = None
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        generator.train(); discriminator.train(); energy.train(); patch_encoder.train()
        sums = {key: 0.0 for key in ("g_total", "g_adv", "g_sb", "g_nce", "identity", "d", "e", "correct")}
        examples = 0
        started = time.perf_counter()
        iterator = tqdm(loader, desc=f"UNSB-SAR {epoch}/{args.epochs}", disable=not is_main, leave=False)
        for batch_index, batch in enumerate(iterator):
            rgb = batch["rgb"].to(device, non_blocking=True)
            rgb_alt = batch["rgb_alt"].to(device, non_blocking=True)
            mask = batch["rgb_mask"].to(device, non_blocking=True)
            mask_alt = batch["rgb_alt_mask"].to(device, non_blocking=True)
            real = batch["roi"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            depression = batch["depression"].to(device, non_blocking=True)
            labels = batch["class_id"].to(device, non_blocking=True).long()
            acquisition = condition_from_batch(meta, depression)
            source_angle = angle_features(batch["rgb_angle"].to(device, non_blocking=True))
            prior = soft_silhouette_prior(mask)

            # D update: real and fake share the same detached condition.  A
            # shuffled condition is an explicit wrong-pair negative.
            set_requires_grad(discriminator, True)
            optimizers["discriminator"].zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_d, cond_d, trajectory_d = generator(
                    torch.zeros_like(real), torch.zeros(len(real), device=device),
                    rgb, mask, acquisition, source_angle=source_angle,
                    rgb_alt=rgb_alt, mask_alt=mask_alt, return_conditions=True,
                    rollout_steps=args.bridge_steps, rollout_noise=args.trajectory_noise)
            real_score, _ = discriminator(real, cond_d.token.detach(), acquisition, source_angle)
            fake_score, _ = discriminator(fake_d.detach(), cond_d.token.detach(), acquisition, source_angle)
            wrong_score, _ = discriminator(real, cond_d.token.detach().roll(1, 0),
                                            acquisition.roll(1, 0), source_angle.roll(1, 0))
            d_loss = (F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
                      + args.wrong_condition_weight * F.relu(1.0 + wrong_score).mean())
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            optimizers["discriminator"].step()

            # E update on two independent trajectories sharing the same
            # geometry/identity/acquisition condition.  Reuse the detached
            # D trajectory as the first path; it has the exact same sampling
            # distribution and avoids one redundant generator rollout.
            set_requires_grad(energy, True)
            optimizers["energy"].zero_grad(set_to_none=True)
            cond_e = cond_d
            trajectory_e = trajectory_d
            with torch.no_grad():
                _, _, trajectory_e2 = generator(
                    torch.zeros_like(real), torch.zeros(len(real), device=device),
                    rgb, mask, acquisition, source_angle=source_angle,
                    rgb_alt=rgb_alt, mask_alt=mask_alt, return_conditions=True,
                    rollout_steps=args.bridge_steps, rollout_noise=args.trajectory_noise,
                    conditions=cond_e)
            stage = random.randrange(args.bridge_steps)
            e_loss = sb_energy_loss(energy, trajectory_e[stage], trajectory_e2[stage],
                                    cond_e, acquisition, tau=args.sb_tau, detach_negative=False)
            e_loss.backward()
            torch.nn.utils.clip_grad_norm_(energy.parameters(), 5.0)
            optimizers["energy"].step()

            # G + PatchNCE update.  D/E parameters are frozen, but their input
            # derivatives remain available to train the generator output.
            set_requires_grad(discriminator, False)
            set_requires_grad(energy, False)
            set_requires_grad(generator, True)
            set_requires_grad(patch_encoder, True)
            optimizers["generator"].zero_grad(set_to_none=True)
            optimizers["patch"].zero_grad(set_to_none=True)
            context = (torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled())
                       if scaler.is_enabled() else nullcontext())
            with context:
                fake, cond_g, trajectory_g = generator(
                    torch.zeros_like(real), torch.zeros(len(real), device=device),
                    rgb, mask, acquisition, source_angle=source_angle,
                    rgb_alt=rgb_alt, mask_alt=mask_alt, return_conditions=True,
                    rollout_steps=args.bridge_steps, rollout_noise=args.trajectory_noise)
                _, _, trajectory_g2 = generator(
                    torch.zeros_like(real), torch.zeros(len(real), device=device),
                    rgb, mask, acquisition, source_angle=source_angle,
                    rgb_alt=rgb_alt, mask_alt=mask_alt, return_conditions=True,
                    rollout_steps=args.bridge_steps, rollout_noise=args.trajectory_noise,
                    conditions=cond_g)
                fake_score, _ = discriminator(fake, cond_g.token.detach(), acquisition, source_angle)
                sb = sb_energy_loss(energy, trajectory_g[stage], trajectory_g2[stage],
                                    cond_g, acquisition, tau=args.sb_tau, detach_negative=True)
                nce = patch_nce_loss(prior, fake, patch_encoder)
                identity = .5 * (F.cross_entropy(cond_g.primary_logits, labels)
                                 + F.cross_entropy(cond_g.alternate_logits, labels))
                identity = identity + args.view_consistency_weight * (
                    1.0 - F.cosine_similarity(cond_g.primary_token,
                                               cond_g.alternate_token, dim=1).mean())
                g_adv = -fake_score.mean()
                g_total = (args.lambda_gan * g_adv + args.lambda_sb * sb
                           + args.lambda_nce * nce + args.identity_loss_weight * identity)
            scaler.scale(g_total).backward()
            scaler.unscale_(optimizers["generator"])
            scaler.unscale_(optimizers["patch"])
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(patch_encoder.parameters(), 5.0)
            scaler.step(optimizers["generator"])
            scaler.step(optimizers["patch"])
            scaler.update()
            for target_param, source_param in zip(ema.parameters(), unwrap(generator).parameters()):
                target_param.mul_(args.ema_decay).add_(source_param.detach(), alpha=1.0 - args.ema_decay)
            for target_buffer, source_buffer in zip(ema.buffers(), unwrap(generator).buffers()):
                target_buffer.copy_(source_buffer)

            count = len(real)
            sums["g_total"] += float(g_total.detach()) * count
            sums["g_adv"] += float(g_adv.detach()) * count
            sums["g_sb"] += float(sb.detach()) * count
            sums["g_nce"] += float(nce.detach()) * count
            sums["identity"] += float(identity.detach()) * count
            sums["d"] += float(d_loss.detach()) * count
            sums["e"] += float(e_loss.detach()) * count
            sums["correct"] += float((cond_g.primary_logits.argmax(1) == labels).sum().detach())
            examples += count
            if is_main:
                iterator.set_postfix(g=f"{g_total.detach().item():.3f}",
                                     d=f"{d_loss.detach().item():.3f}",
                                     nce=f"{nce.detach().item():.3f}")
                if preview is None:
                    preview = (rgb[:args.preview_count].detach().cpu(),
                               mask[:args.preview_count].detach().cpu(),
                               real[:args.preview_count].detach().cpu(),
                               acquisition[:args.preview_count].detach().cpu(),
                               source_angle[:args.preview_count].detach().cpu(),
                               rgb_alt[:args.preview_count].detach().cpu(),
                               mask_alt[:args.preview_count].detach().cpu())
            if args.limit_train_batches and batch_index + 1 >= args.limit_train_batches:
                break

        values = torch.tensor(tuple(sums[key] for key in ("g_total", "g_adv", "g_sb", "g_nce",
                                                          "identity", "d", "e", "correct"))
                              + (float(examples),),
                              dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        denominator = values[8].clamp_min(1.0)
        metrics = {key: float(values[index] / denominator)
                   for index, key in enumerate(("g_total", "g_adv", "g_sb", "g_nce",
                                                "identity", "d", "e", "correct"))}
        elapsed = max(time.perf_counter() - started, 1e-6)
        if distributed:
            dist.barrier()
        if is_main:
            speed = float(values[8] / elapsed)
            with history.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow((epoch, metrics["g_total"], metrics["g_adv"],
                                             metrics["g_sb"], metrics["g_nce"], metrics["identity"],
                                             metrics["d"], metrics["e"], metrics["correct"], speed))
            state = checkpoint_state(epoch, generator, ema, discriminator, energy, patch_encoder,
                                     optimizers, scaler, args, data_summary)
            torch.save(state, args.output / "latest.pt")
            if epoch % args.save_every == 0 or epoch == args.epochs:
                torch.save(state, args.output / f"epoch_{epoch:03d}.pt")
            if preview is not None and (epoch == 1 or epoch % args.preview_every == 0 or epoch == args.epochs):
                (preview_rgb, preview_mask, preview_real, preview_condition,
                 preview_source_angle, preview_alt, preview_alt_mask) = preview
                preview_rgb = preview_rgb.to(device); preview_mask = preview_mask.to(device)
                preview_real = preview_real.to(device); preview_condition = preview_condition.to(device)
                preview_source_angle = preview_source_angle.to(device)
                preview_alt = preview_alt.to(device); preview_alt_mask = preview_alt_mask.to(device)
                generator_rng = torch.Generator(device=device).manual_seed(args.seed + epoch)
                fake_preview = bridge_sample(
                    ema, preview_rgb, preview_mask, preview_condition,
                    steps=args.sample_steps, temperature=args.sample_temperature,
                    generator=generator_rng, source_angle=preview_source_angle,
                    rgb_alt=preview_alt, mask_alt=preview_alt_mask)
                write_preview(args.output / f"preview_{epoch:03d}.png", preview_rgb,
                              preview_mask, preview_real, fake_preview)
            print({"epoch": epoch, **metrics, "examples_per_second": speed}, flush=True)
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Train a 64x64 RGB-identity/acquisition-conditioned SAR DDPM.

This is intentionally independent of the HiFC and FACT trainers.  It uses no
native SAR classifier, adversarial discriminator, pixel RGB/SAR alignment, or
downstream SAR classification loss.  The only auxiliary objective is
supervised identity classification of the two RGB views, which makes the
available class-level association usable without pretending RGB and SAR are
pixel paired.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from conditional_sar_diffusion import (
    CONDITIONAL_DIFFUSION_ARCHITECTURE, ConditionalSARDDPM, DiffusionSchedule,
    ema_update,
)
from hifc_unpaired_sar_gan import condition_from_batch
from joint_data import JointROIDataset


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="64px RGB-identity/acquisition-conditioned SAR DDPM")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band", choices=("all", "X", "KU"), default="all")
    parser.add_argument("--polarization", choices=("all", "HH", "HV", "VH", "VV"), default="all")
    parser.add_argument("--depression", choices=("all", "15", "30", "45", "60"), default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epoch-size", type=int, default=24_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--rgb-base", type=int, default=32)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=1_000)
    parser.add_argument("--condition-drop-prob", type=float, default=.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--identity-loss-weight", type=float, default=.1)
    parser.add_argument("--view-consistency-weight", type=float, default=.25)
    parser.add_argument("--sample-steps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--preview-every", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260904)
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
        raise RuntimeError("torchrun DDPM training requires CUDA and LOCAL_RANK")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device("cuda", local_rank))
    return True, world_size, rank, torch.device("cuda", local_rank)


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def make_loader(dataset: JointROIDataset, batch_size: int, workers: int,
                distributed: bool, rank: int, world_size: int,
                device: torch.device) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True, drop_last=True)
               if distributed else None)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=workers, pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=workers > 0)
    return loader, sampler


def write_preview(path: Path, rgb: torch.Tensor, real: torch.Tensor, fake: torch.Tensor) -> None:
    rows = []
    for index in range(min(len(rgb), len(fake))):
        rgb_panel = F.interpolate(rgb[index:index + 1], (64, 64), mode="bilinear", align_corners=False)[0]
        rgb_panel = (((rgb_panel.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy()) + 1.0) * 127.5).astype(np.uint8)
        panels = [rgb_panel]
        for image in (real[index, 0], fake[index, 0]):
            panel = (((image.detach().cpu().clamp(-1, 1).numpy()) + 1.0) * 127.5).astype(np.uint8)
            panels.append(np.repeat(panel[..., None], 3, axis=2))
        rows.append(np.concatenate(panels, axis=1))
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(rows, axis=0), "RGB").save(path)


def checkpoint_state(epoch: int, model: nn.Module, ema: nn.Module,
                     optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                     args: argparse.Namespace, data_summary: dict[str, object]) -> dict[str, object]:
    return {
        "architecture": CONDITIONAL_DIFFUSION_ARCHITECTURE,
        "epoch": epoch,
        "model": unwrap(model).state_dict(),
        "ema_model": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
        "data_summary": data_summary,
    }


def main() -> None:
    args = arguments()
    if args.epochs < 1 or args.epoch_size < 1 or args.batch_size < 1:
        raise ValueError("epochs, epoch-size, and batch-size must be positive")
    if not 0.0 <= args.condition_drop_prob < 1.0:
        raise ValueError("condition-drop-prob must be in [0, 1)")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema-decay must be in (0, 1)")
    if args.identity_loss_weight < 0.0 or args.view_consistency_weight < 0.0:
        raise ValueError("identity and view consistency weights must be non-negative")
    if args.preview_every < 1 or args.save_every < 1:
        raise ValueError("preview-every and save-every must be positive")

    distributed, world_size, rank, device = setup_distributed(args)
    is_main = rank == 0
    # All ranks construct identical parameters first; only the subsequent data
    # noise stream is rank-specific.
    seed_everything(args.seed)
    dataset = JointROIDataset(
        args.rgb_root, args.sar_train_root, rgb_size=128, roi_size=64,
        epoch_size=args.epoch_size, band=args.band, polarization=args.polarization,
        depression=args.depression, augment_rgb=True, source_view_mode="random")
    loader, sampler = make_loader(dataset, args.batch_size, args.workers,
                                  distributed, rank, world_size, device)
    model_core = ConditionalSARDDPM(base=args.base, token_dim=args.token_dim,
                                    rgb_base=args.rgb_base, class_conditioning=True).to(device)
    ema = copy.deepcopy(model_core).eval().to(device)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    model: nn.Module = (DDP(model_core, device_ids=[device.index], output_device=device.index,
                            broadcast_buffers=False)
                         if distributed else model_core)
    schedule = DiffusionSchedule(args.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(.9, .999))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and not args.no_amp)
    start_epoch = 1
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("architecture") != CONDITIONAL_DIFFUSION_ARCHITECTURE:
            raise RuntimeError("resume checkpoint is not a conditional SAR DDPM")
        unwrap(model).load_state_dict(state["model"])
        ema.load_state_dict(state["ema_model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_epoch = int(state["epoch"]) + 1
    # Distinct noise draws across ranks after identical parameter initialization.
    seed_everything(args.seed + rank)

    data_summary = {**dataset.summary(), "filters": {"band": args.band, "polarization": args.polarization,
                                                        "depression": args.depression},
                    "global_batch_size": args.batch_size * world_size}
    if is_main:
        args.output.mkdir(parents=True, exist_ok=True)
        config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "architecture": CONDITIONAL_DIFFUSION_ARCHITECTURE, "data_summary": data_summary}
        (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        history = args.output / "history.csv"
        if not (args.resume is not None and history.is_file()):
            with history.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(("epoch", "total_loss", "v_mse", "identity_loss",
                                             "rgb_class_accuracy", "learning_rate",
                                             "examples_per_second"))
        print({"device": str(device), "world_size": world_size, "dataset": data_summary}, flush=True)
    else:
        history = args.output / "history.csv"
    if distributed:
        dist.barrier()

    preview = None
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        v_loss_sum = 0.0
        identity_loss_sum = 0.0
        class_correct = 0.0
        examples = 0
        started = time.perf_counter()
        iterator = tqdm(loader, desc=f"conditional DDPM {epoch}/{args.epochs}", disable=not is_main, leave=False)
        for batch_index, batch in enumerate(iterator):
            rgb = batch["rgb"].to(device, non_blocking=True)
            rgb_alt = batch["rgb_alt"].to(device, non_blocking=True)
            clean = batch["roi"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            depression = batch["depression"].to(device, non_blocking=True)
            labels = batch["class_id"].to(device, non_blocking=True).long()
            acquisition = condition_from_batch(meta, depression)
            timestep = torch.randint(schedule.steps, (len(clean),), device=device)
            dropped = torch.rand(len(clean), device=device) < args.condition_drop_prob
            noisy, noise = schedule.q_sample(clean, timestep)
            velocity = schedule.v_target(clean, timestep, noise)
            optimizer.zero_grad(set_to_none=True)
            context = (torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled())
                       if scaler.is_enabled() else nullcontext())
            with context:
                prediction, primary_logits, alternate_logits, primary_token, alternate_token = model(
                    noisy, timestep, rgb, acquisition, condition_drop=dropped,
                    rgb_alt=rgb_alt, return_identity=True)
                v_loss = F.mse_loss(prediction, velocity)
                if primary_logits is None or alternate_logits is None or alternate_token is None:
                    raise RuntimeError("identity-conditioned training requires two RGB identity outputs")
                identity_ce = .5 * (F.cross_entropy(primary_logits, labels)
                                    + F.cross_entropy(alternate_logits, labels))
                token_cosine = 1.0 - F.cosine_similarity(primary_token, alternate_token, dim=1).mean()
                identity_loss = identity_ce + args.view_consistency_weight * token_cosine
                loss = v_loss + args.identity_loss_weight * identity_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema_update(ema, unwrap(model), args.ema_decay)
            loss_sum += float(loss.detach()) * len(clean)
            v_loss_sum += float(v_loss.detach()) * len(clean)
            identity_loss_sum += float(identity_loss.detach()) * len(clean)
            class_correct += float((primary_logits.argmax(1) == labels).sum().detach())
            examples += len(clean)
            if is_main:
                iterator.set_postfix(v=f"{v_loss.detach().item():.4f}",
                                     id=f"{identity_loss.detach().item():.3f}")
                if preview is None:
                    preview = (rgb[:args.preview_count].detach().cpu(), clean[:args.preview_count].detach().cpu(),
                               acquisition[:args.preview_count].detach().cpu())
            if args.limit_train_batches and batch_index + 1 >= args.limit_train_batches:
                break

        totals = torch.tensor((loss_sum, v_loss_sum, identity_loss_sum, class_correct,
                               float(examples)), dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        global_loss = float(totals[0] / totals[4].clamp_min(1.0))
        global_v_loss = float(totals[1] / totals[4].clamp_min(1.0))
        global_identity_loss = float(totals[2] / totals[4].clamp_min(1.0))
        global_class_accuracy = float(totals[3] / totals[4].clamp_min(1.0))
        elapsed = max(time.perf_counter() - started, 1e-6)
        if distributed:
            dist.barrier()
        if is_main:
            samples_per_second = float(totals[4] / elapsed)
            with history.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow((epoch, global_loss, global_v_loss, global_identity_loss,
                                             global_class_accuracy, optimizer.param_groups[0]["lr"],
                                             samples_per_second))
            state = checkpoint_state(epoch, model, ema, optimizer, scaler, args, data_summary)
            torch.save(state, args.output / "latest.pt")
            if epoch % args.save_every == 0 or epoch == args.epochs:
                torch.save(state, args.output / f"epoch_{epoch:03d}.pt")
            if preview is not None and (epoch == 1 or epoch % args.preview_every == 0 or epoch == args.epochs):
                preview_rgb, preview_real, preview_condition = (value.to(device) for value in preview)
                generator = torch.Generator(device=device).manual_seed(args.seed + epoch)
                fake = schedule.ddim_sample(ema, preview_rgb, preview_condition,
                                            sample_steps=args.sample_steps,
                                            guidance_scale=args.guidance_scale,
                                            generator=generator)
                write_preview(args.output / f"preview_{epoch:03d}.png", preview_rgb, preview_real, fake)
            print({"epoch": epoch, "total_loss": global_loss, "v_mse": global_v_loss,
                   "identity_loss": global_identity_loss,
                   "rgb_class_accuracy": global_class_accuracy,
                   "examples_per_second": samples_per_second}, flush=True)
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

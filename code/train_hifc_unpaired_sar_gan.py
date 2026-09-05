"""Train the HiFC-inspired unpaired RGB-to-SAR model.

This is a separate experiment from all V1 trainers.  It keeps the two ideas
from HiFC-GAN (local texture contrast and deep semantic feature mapping), but
removes the optical-to-SAR pixel-registration assumption.  RGB and SAR are
matched by vehicle class only; acquisition metadata is supplied as a target
condition.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from dual_component_sar_gan import LargeRGBIdentityEncoder
from hifc_unpaired_sar_gan import (
    HIFC_ARCHITECTURE, HIFCConditionedDiscriminator, HIFCUnpairedGenerator,
    ConditionalPrototypeBank, condition_from_batch, condition_group_code,
    conditional_set_sfm_loss, discriminator_hinge, geometry_auxiliary_loss,
    initialise_hifc, local_texture_contrast_loss,
    local_texture_signature, parameter_count, rgb_identity_loss,
    semantic_feature_mapping_loss,
    set_grad, update_ema)
from joint_data import JointROIDataset
from bbox_data import image_tensor
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HiFC-inspired unpaired RGB-to-SAR GAN (no pixel alignment)")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-train-root", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band", choices=("all", "X", "KU"), default="all",
                        help="train conditions; use all to learn band conditioning")
    parser.add_argument("--polarization", choices=("all", "HH", "HV", "VH", "VV"),
                        default="all", help="train conditions; use all to learn polarization")
    parser.add_argument("--depression", choices=("all", "15", "30", "45", "60"),
                        default="all")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--epoch-size", type=int, default=24000,
                        help="random weakly-unpaired samples per training epoch")
    parser.add_argument("--condition-sampler", choices=("record", "class_uniform", "domain_uniform", "support_uniform"),
                        default="record",
                        help=("training-record sampler; record preserves the empirical data "
                              "frequency, domain_uniform balances class and shared "
                              "band/polarization/depression domains while retaining empirical 5-degree azimuth"))
    parser.add_argument("--condition-sampler-seed", type=int, default=20260830,
                        help="seed for deterministic class/acquisition sampling")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--generator-lr", type=float, default=1.5e-4)
    parser.add_argument("--identity-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=1e-4)
    parser.add_argument("--adversarial-weight", type=float, default=1.0)
    parser.add_argument("--adversarial-warmup-epochs", type=int, default=1)
    parser.add_argument("--r1-weight", type=float, default=.25)
    parser.add_argument("--r1-every", type=int, default=16)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--rgb-identity-weight", type=float, default=1.0)
    parser.add_argument("--ltc-weight", type=float, default=2.0)
    parser.add_argument("--sfm-weight", type=float, default=2.0)
    parser.add_argument("--sfm-mode", choices=("batch", "conditional_set_ot"),
                        default="batch",
                        help=("SFM implementation; batch preserves the original "
                              "itemwise baseline, conditional_set_ot uses a "
                              "frozen condition-prototype whitened set loss"))
    parser.add_argument("--sfm-projection-count", type=int, default=64)
    parser.add_argument("--sfm-ltc-cost-weight", type=float, default=.50)
    parser.add_argument("--sfm-anchor-weight", type=float, default=.25)
    parser.add_argument("--sfm-prototype-batch-size", type=int, default=128)
    parser.add_argument("--sfm-prototype-cache", type=Path)
    parser.add_argument("--geometry-weight", type=float, default=.30)
    parser.add_argument(
        "--native-gradient-mode", choices=("full", "embedding_off", "all_off"),
        default="full",
        help=("native SAR teacher gradient route: full keeps the baseline; "
              "embedding_off detaches only SFM teacher embeddings; all_off "
              "also detaches geometry. Teacher metrics are always logged."))
    parser.add_argument("--meta-transfer-weight", type=float, default=0.0,
                        help=("HiFC-TU generated-support -> real-query meta loss "
                              "weight; zero preserves the original baseline"))
    parser.add_argument("--meta-transfer-every", type=int, default=4,
                        help="compute the train-only meta objective every N batches")
    parser.add_argument("--meta-transfer-inner-lr", type=float, default=.1,
                        help="one virtual classifier-head SGD step size")
    parser.add_argument("--meta-transfer-probe-checkpoint", type=Path,
                        help="synthetic-only SARClassifier64 probe checkpoint")
    parser.add_argument("--meta-transfer-query-batch-size", type=int, default=0,
                        help="real X/HH meta-query batch size; zero uses the GAN batch size")
    parser.add_argument("--meta-transfer-seed", type=int, default=1729,
                        help="private RNG seed for meta support noise")
    parser.add_argument("--meta-transfer-audit-events", type=int, default=0,
                        help="record meta diagnostics for this many events (zero disables extra audit)")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1,
                        help="torchrun local rank; normally supplied through the environment")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-validation-batches", type=int, default=0)
    return parser.parse_args()


def autocast_context(device: torch.device, enabled: bool):
    return (torch.amp.autocast(device_type=device.type, enabled=True)
            if enabled else nullcontext())


def make_loader(dataset: JointROIDataset, batch_size: int, workers: int,
                shuffle: bool, device: torch.device,
                sampler: DistributedSampler | None = None) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        drop_last=shuffle)


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int, torch.device]:
    """Initialize single-node DDP when launched by torchrun.

    A normal ``python`` invocation remains single-process.  In DDP mode the
    launcher-provided rank is authoritative, so ``--device`` is only used by
    the non-distributed path.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA; torchrun launched multiple ranks")
        if local_rank < 0:
            raise RuntimeError("torchrun did not provide LOCAL_RANK")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            device_id=torch.device("cuda", local_rank))
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    return distributed, world_size, rank, local_rank, device


def unwrap(module: nn.Module) -> nn.Module:
    """Return the underlying module for DDP-safe checkpoint serialization."""
    return module.module if isinstance(module, DDP) else module


def all_reduce_stats(values: list[float], device: torch.device,
                     distributed: bool) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def split_records(records: list[tuple], root: Path, manifest: Path,
                  fraction: float, seed: int,
                  filters: dict[str, str]) -> tuple[set[str], set[str]]:
    """Make a deterministic class/condition split without pairing RGB pixels."""
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if (saved.get("split_version") == 2
                and saved.get("source_root") == str(root.resolve())
                and saved.get("filters") == filters):
            return set(saved["train"]), set(saved["validation"])
        # A manifest made by an earlier invocation may not contain the filter;
        # retaining it would silently mix all-condition and X/HH experiments.
    groups: dict[tuple[str, str, str, int, int], list[tuple]] = defaultdict(list)
    for record in records:
        meta = record[4]
        groups[(record[2], str(meta["band"]), str(meta["pol"]),
                int(meta["depression"]),
                (int(meta["azimuth"]) + 15) // 30 % 12)].append(record)
    train, validation = [], []
    for group, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda record: hashlib.sha256(
            f"{seed}:{group}:{record[0].relative_to(root)}".encode()).hexdigest())
        # Keep at least one training record in every class/acquisition cell.
        # This makes support_uniform a genuine observed-support sampler rather
        # than silently dropping sparse angle cells into validation.
        count = min(max(1, round(len(ordered) * fraction)), max(0, len(ordered) - 1)) if fraction else 0
        validation.extend(str(item[0].relative_to(root)) for item in ordered[:count])
        train.extend(str(item[0].relative_to(root)) for item in ordered[count:])
    payload = {
        "version": HIFC_ARCHITECTURE, "split_version": 2,
        "source_root": str(root.resolve()),
        "seed": seed, "validation_fraction": fraction,
        "filters": filters,
        "train": sorted(train), "validation": sorted(validation)}
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return set(train), set(validation)


def configure_records(dataset: JointROIDataset, selected: set[str], root: Path,
                      epoch_size: int = 0) -> None:
    dataset.records = [
        record for record in dataset.records
        if str(record[0].relative_to(root)) in selected]
    if not dataset.records:
        raise RuntimeError("empty HiFC train/validation split")
    dataset.epoch_size = epoch_size or len(dataset.records)
    dataset.random_epoch = bool(epoch_size)
    dataset._rebuild_condition_index()


class RealSARPrototypeDataset(Dataset):
    """Read only real SAR records for the frozen SFM prototype cache."""

    def __init__(self, records: list[tuple], roi_size: int = 64) -> None:
        self.records = records
        self.roi_size = roi_size
        self.class_to_id = {name: index for index, name in enumerate(SOC40_CLASSES)}
        self.polarization_to_id = {name: index for index, name in enumerate(
            ("HH", "HV", "VH", "VV"))}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        tif, _, class_name, _, meta, _ = self.records[index]
        with Image.open(tif) as image:
            roi = image_tensor(image, self.roi_size, False)
        class_id = self.class_to_id[class_name]
        band = 0 if str(meta["band"]).upper() == "X" else 1
        polarization = self.polarization_to_id[str(meta["pol"]).upper()]
        depression = (int(meta["depression"]) // 15) - 1
        code = (((class_id * 2 + band) * 4 + polarization) * 4 + depression)
        return roi, code


class MetaRealXHHDataset(Dataset):
    """Train-only real X/HH queries for the HiFC-TU meta objective.

    The records come from the deterministic GAN validation split, never from
    the official test root.  RGB is sampled by class only and is used to make
    a synthetic support example; its associated SAR ROI is the outer query.
    No RGB/SAR pixel correspondence is assumed.
    """

    def __init__(self, base: JointROIDataset, records: list[tuple]) -> None:
        self.base = base
        self.records = [record for record in records
                         if str(record[4]["band"]).upper() == "X"
                         and str(record[4]["pol"]).upper() == "HH"]
        if not self.records:
            raise RuntimeError("HiFC-TU requires real X/HH meta-query records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
        tif, _, class_name, _, meta, _ = self.records[index]
        # The support RGB view is selected independently within the class.
        # This retains the unpaired policy even though the record also carries
        # a convenient nearest-view path.
        rgb_angle = random.choice(self.base.class_rgb_angles[class_name])
        rgb = self.base._rgb(self.base.rgb_paths[class_name, rgb_angle])
        with Image.open(tif) as image:
            query = image_tensor(image, 64, False).add(1.0).mul(.5)
        class_id = torch.tensor(self.base.class_to_id[class_name], dtype=torch.long)
        depression = torch.tensor(int(meta["depression"]), dtype=torch.long)
        azimuth = torch.tensor(int(meta["azimuth"]), dtype=torch.long)
        return rgb, query, class_id, torch.stack((depression, azimuth))


def _prototype_split_digest(records: list[tuple], root: Path) -> str:
    names = sorted(str(record[0].relative_to(root)) for record in records)
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _prototype_cache_signature(args: argparse.Namespace,
                               records: list[tuple]) -> dict[str, object]:
    checkpoint = args.native_classifier_checkpoint
    stat = checkpoint.stat()
    return {
        "version": 1,
        "architecture": HIFC_ARCHITECTURE,
        "source_root": str(args.sar_train_root.resolve()),
        "split_digest": _prototype_split_digest(records, args.sar_train_root),
        "teacher": {"path": str(checkpoint.resolve()), "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns},
        "roi_size": 64,
        "std_floor": .05,
    }


def build_prototype_bank(records: list[tuple], teacher: SARClassifier64,
                         device: torch.device, batch_size: int,
                         workers: int) -> ConditionalPrototypeBank:
    """Compute train-only native/LTC statistics once, without RGB reads."""
    dataset = RealSARPrototypeDataset(records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=max(0, min(workers, 4)),
                        pin_memory=device.type == "cuda",
                        persistent_workers=workers > 0)
    counts: dict[int, int] = defaultdict(int)
    sums: dict[int, torch.Tensor] = {}
    squares: dict[int, torch.Tensor] = {}
    ltc_sums: dict[int, torch.Tensor] = {}
    ltc_squares: dict[int, torch.Tensor] = {}
    total_count = 0
    total_sum = total_square = None
    total_ltc_sum = total_ltc_square = None
    teacher.eval()
    with torch.inference_mode():
        for images, codes in tqdm(loader, desc="SFM real prototype cache", leave=False):
            images = images.to(device, non_blocking=True)
            codes = codes.long()
            _, features = teacher(((images + 1.0) * .5).clamp(0, 1),
                                  return_features=True)
            signatures = local_texture_signature(images)
            features_cpu = features.float().cpu()
            signatures_cpu = signatures.float().cpu()
            if total_sum is None:
                total_sum = torch.zeros(features_cpu.shape[1], dtype=torch.float64)
                total_square = torch.zeros_like(total_sum)
                total_ltc_sum = torch.zeros(signatures_cpu.shape[1], dtype=torch.float64)
                total_ltc_square = torch.zeros_like(total_ltc_sum)
            total_count += len(codes)
            total_sum += features_cpu.double().sum(0)
            total_square += features_cpu.double().square().sum(0)
            total_ltc_sum += signatures_cpu.double().sum(0)
            total_ltc_square += signatures_cpu.double().square().sum(0)
            for code in codes.unique().tolist():
                mask = codes == code
                feature_rows = features_cpu[mask].double()
                ltc_rows = signatures_cpu[mask].double()
                counts[code] += int(mask.sum())
                if code not in sums:
                    sums[code] = torch.zeros_like(total_sum)
                    squares[code] = torch.zeros_like(total_sum)
                    ltc_sums[code] = torch.zeros_like(total_ltc_sum)
                    ltc_squares[code] = torch.zeros_like(total_ltc_sum)
                sums[code] += feature_rows.sum(0)
                squares[code] += feature_rows.square().sum(0)
                ltc_sums[code] += ltc_rows.sum(0)
                ltc_squares[code] += ltc_rows.square().sum(0)
    if not total_count or total_sum is None:
        raise RuntimeError("cannot build SFM prototypes from an empty split")

    floor = .05
    global_mean = total_sum / total_count
    global_std = (total_square / total_count - global_mean.square()).clamp_min(floor ** 2).sqrt()
    global_ltc_mean = total_ltc_sum / total_count
    global_ltc_std = (total_ltc_square / total_count - global_ltc_mean.square()).clamp_min(floor ** 2).sqrt()
    codes_sorted = sorted(counts)
    embedding_mean, embedding_std, ltc_mean, ltc_std = [], [], [], []
    for code in codes_sorted:
        count = counts[code]
        mean = sums[code] / count
        std = (squares[code] / count - mean.square()).clamp_min(floor ** 2).sqrt()
        ltc_mean_row = ltc_sums[code] / count
        ltc_std_row = (ltc_squares[code] / count - ltc_mean_row.square()).clamp_min(floor ** 2).sqrt()
        embedding_mean.append(mean.float())
        embedding_std.append(std.float())
        ltc_mean.append(ltc_mean_row.float())
        ltc_std.append(ltc_std_row.float())
    return ConditionalPrototypeBank(
        torch.tensor(codes_sorted, dtype=torch.long),
        torch.stack(embedding_mean), torch.stack(embedding_std),
        torch.stack(ltc_mean), torch.stack(ltc_std),
        global_mean.float(), global_std.float(),
        global_ltc_mean.float(), global_ltc_std.float())


def differentiable_augment(image: torch.Tensor) -> torch.Tensor:
    """Shared radiometric augmentation; no domain-specific spatial warp."""
    batch = len(image)
    amplitude = ((image + 1.0) * .5).clamp(0, 1)
    gain = amplitude.new_empty(batch, 1, 1, 1).uniform_(.90, 1.10)
    bias = amplitude.new_empty(batch, 1, 1, 1).uniform_(-.03, .03)
    amplitude = amplitude * gain + bias
    # A tiny multiplicative perturbation models acquisition variation without
    # creating an RGB/SAR coordinate correspondence.
    amplitude = amplitude * torch.exp(torch.randn_like(amplitude) * .015)
    return amplitude.clamp(0, 1) * 2.0 - 1.0


def save_preview(path: Path, rgb: torch.Tensor, real: torch.Tensor,
                 fake: torch.Tensor, clean: torch.Tensor) -> None:
    rows = []
    for index in range(min(8, len(fake))):
        rgb_panel = F.interpolate(rgb[index:index + 1], (64, 64),
                                  mode="bilinear", align_corners=False)[0]
        rgb_panel = (((rgb_panel.detach().cpu().clamp(-1, 1)
                       .permute(1, 2, 0).numpy()) + 1) * 127.5).astype(np.uint8)
        panels = [rgb_panel]
        for tensor in (real, clean, fake):
            panel = (((tensor[index, 0].detach().cpu().clamp(-1, 1).numpy())
                      + 1) * 127.5).astype(np.uint8)
            panels.append(np.repeat(panel[..., None], 3, axis=2))
        rows.append(np.concatenate(panels, axis=1))
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0), "RGB").save(path)


def load_teacher(checkpoint: Path, device: torch.device) -> SARClassifier64:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("native classifier class order mismatch")
    teacher = SARClassifier64(len(SOC40_CLASSES)).to(device)
    teacher.load_state_dict(state["model"])
    teacher.eval()
    set_grad(teacher, False)
    return teacher


def _private_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def private_randn(shape: tuple[int, ...], device: torch.device, seed: int) -> torch.Tensor:
    """Draw meta noise without perturbing the main training RNG stream."""
    with torch.random.fork_rng(devices=_private_rng_devices(device)):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        return torch.randn(shape, device=device)


def load_meta_probe(checkpoint: Path, device: torch.device) -> SARClassifier64:
    """Load a frozen synthetic-only probe while preserving caller RNG state."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"meta-transfer probe not found: {checkpoint}")
    with torch.random.fork_rng(devices=_private_rng_devices(device)):
        torch.manual_seed(0)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get("classes") != list(SOC40_CLASSES):
            raise RuntimeError("meta-transfer probe class order mismatch")
        metadata = state.get("meta_probe_metadata", {})
        if metadata and metadata.get("real_images_seen", True):
            raise RuntimeError("meta-transfer probe is not synthetic-only")
        probe = SARClassifier64(len(SOC40_CLASSES)).to(device)
        probe.load_state_dict(state["model"])
    probe.eval()
    set_grad(probe, False)
    return probe


def meta_transfer_head_loss(probe: SARClassifier64,
                            support_images: torch.Tensor,
                            support_labels: torch.Tensor,
                            query_images: torch.Tensor,
                            query_labels: torch.Tensor,
                            inner_lr: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One-step synthetic-support/real-query class-head meta objective.

    The probe backbone is frozen.  A detached copy of its class head is
    updated on generated support features, and the resulting real-query CE
    supplies a second-order gradient only through the support images.
    """
    if support_images.ndim != 4 or query_images.ndim != 4:
        raise ValueError("meta-transfer images must be [B,1,H,W]")
    with torch.autocast(device_type=support_images.device.type, enabled=False):
        support_images = support_images.float().clamp(0, 1)
        query_images = query_images.float().clamp(0, 1)
        _, support_features = probe(support_images, return_features=True)
        with torch.no_grad():
            _, query_features = probe(query_images, return_features=True)
        weight = probe.classifier.weight.detach().clone().requires_grad_(True)
        bias = probe.classifier.bias.detach().clone().requires_grad_(True)
        support_logits = F.linear(support_features, weight, bias)
        support_before = F.cross_entropy(
            support_logits, support_labels, label_smoothing=.03)
        grad_weight, grad_bias = torch.autograd.grad(
            support_before, (weight, bias), create_graph=True)
        updated_weight = weight - float(inner_lr) * grad_weight
        updated_bias = bias - float(inner_lr) * grad_bias
        support_after = F.cross_entropy(
            F.linear(support_features, updated_weight, updated_bias), support_labels)
        query_before = F.cross_entropy(
            F.linear(query_features, weight.detach(), bias.detach()), query_labels)
        query_after = F.cross_entropy(
            F.linear(query_features, updated_weight, updated_bias), query_labels)
    return query_after, {
        "support_before": support_before.detach(),
        "support_after": support_after.detach(),
        "query_before": query_before.detach(),
        "query_after": query_after.detach(),
    }


def _gradient_l2(grads: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
                 device: torch.device | None = None) -> torch.Tensor:
    """Return a finite scalar norm for an allow-unused autograd result."""
    pieces = [gradient.detach().float().square().sum()
              for gradient in grads if gradient is not None]
    if not pieces:
        return torch.zeros((), device=device or torch.device("cpu"))
    return torch.sqrt(torch.stack(pieces).sum().clamp_min(1e-24))


def checkpoint_state(epoch: int, encoder: nn.Module, generator: nn.Module,
                     discriminator: nn.Module, ema_encoder: nn.Module,
                     ema_generator: nn.Module, generator_optimizer: torch.optim.Optimizer,
                     discriminator_optimizer: torch.optim.Optimizer,
                     validation: dict[str, float], args: argparse.Namespace,
                     sfm_prototype_cache: Path | None = None) -> dict:
    return {
        "architecture": HIFC_ARCHITECTURE,
        "epoch": epoch,
        "classes": list(SOC40_CLASSES),
        "condition_dim": 12,
        "identity_encoder": unwrap(encoder).state_dict(),
        "generator": unwrap(generator).state_dict(),
        "discriminator": unwrap(discriminator).state_dict(),
        "ema_identity_encoder": ema_encoder.state_dict(),
        "ema_generator": ema_generator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "validation": validation,
        "training_policy": "class-matched unpaired RGB/SAR; no pixel alignment",
        "native_gradient_mode": args.native_gradient_mode,
        "sfm_mode": args.sfm_mode,
        "meta_transfer_weight": args.meta_transfer_weight,
        "meta_transfer_every": args.meta_transfer_every,
        "meta_transfer_inner_lr": args.meta_transfer_inner_lr,
        "meta_transfer_probe_checkpoint": (
            str(args.meta_transfer_probe_checkpoint.resolve())
            if args.meta_transfer_probe_checkpoint is not None else None),
        "sfm_prototype_cache": (str(sfm_prototype_cache)
                                 if sfm_prototype_cache is not None else None),
        "condition_sampler": args.condition_sampler,
        "condition_sampler_seed": args.condition_sampler_seed,
        "filters": {"band": args.band, "polarization": args.polarization,
                    "depression": args.depression},
    }


def main() -> None:
    args = arguments()
    if args.epochs <= 0 or args.batch_size <= 0 or args.epoch_size <= 0:
        raise ValueError("epochs, batch-size and epoch-size must be positive")
    if args.sfm_projection_count <= 0 or args.sfm_prototype_batch_size <= 0:
        raise ValueError("SFM projection and prototype batch sizes must be positive")
    if args.meta_transfer_weight < 0.0:
        raise ValueError("--meta-transfer-weight must be non-negative")
    if args.meta_transfer_every <= 0 or args.meta_transfer_inner_lr <= 0.0:
        raise ValueError("meta-transfer interval and inner learning rate must be positive")
    if args.meta_transfer_query_batch_size < 0:
        raise ValueError("--meta-transfer-query-batch-size must be non-negative")
    if args.meta_transfer_audit_events < 0:
        raise ValueError("--meta-transfer-audit-events must be non-negative")
    if args.meta_transfer_weight > 0.0 and args.meta_transfer_probe_checkpoint is None:
        raise ValueError("--meta-transfer-weight > 0 requires --meta-transfer-probe-checkpoint")
    distributed, world_size, rank, local_rank, device = setup_distributed(args)
    is_main = rank == 0
    if distributed:
        # Keep model initialization identical on every rank.  Rank-specific
        # RNG streams are installed after DDP broadcasts the parameters.
        seed = args.seed
    else:
        seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if is_main:
        args.output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    use_amp = device.type == "cuda" and not args.no_amp

    train_data = JointROIDataset(
        args.rgb_root, args.sar_train_root, rgb_size=128, roi_size=64,
        epoch_size=0, band=args.band, polarization=args.polarization,
        depression=args.depression, augment_rgb=True, source_view_mode="random",
        condition_sampler=args.condition_sampler,
        condition_sampler_seed=args.condition_sampler_seed)
    manifest = args.output / f"split_manifest__{args.band}_{args.polarization}_{args.depression}.json"
    filters = {"band": args.band, "polarization": args.polarization,
               "depression": args.depression}
    # Only rank 0 creates a new manifest.  Other ranks wait for the file and
    # then read the exact same split, avoiding concurrent writes on the mount.
    if is_main:
        train_keys, validation_keys = split_records(
            train_data.records, args.sar_train_root, manifest,
            args.validation_fraction, args.seed, filters)
    if distributed:
        dist.barrier()
    if not is_main:
        train_keys, validation_keys = split_records(
            train_data.records, args.sar_train_root, manifest,
            args.validation_fraction, args.seed, filters)
    if distributed:
        dist.barrier()
    configure_records(train_data, train_keys, args.sar_train_root, args.epoch_size)
    validation_data = JointROIDataset(
        args.rgb_root, args.sar_train_root, rgb_size=128, roi_size=64,
        epoch_size=0, band=args.band, polarization=args.polarization,
        depression=args.depression, augment_rgb=False, source_view_mode="random")
    configure_records(validation_data, validation_keys, args.sar_train_root)
    train_sampler = (DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True,
        seed=args.seed, drop_last=True) if distributed else None)
    validation_sampler = (DistributedSampler(
        validation_data, num_replicas=world_size, rank=rank, shuffle=False,
        seed=args.seed, drop_last=False) if distributed else None)
    train_loader = make_loader(
        train_data, args.batch_size, args.workers, True, device, train_sampler)
    validation_loader = make_loader(
        validation_data, args.batch_size, args.workers, False, device,
        validation_sampler)

    meta_query_data = None
    meta_query_sampler = None
    meta_query_loader = None
    meta_probe = None
    if args.meta_transfer_weight > 0.0:
        # Validation records are already excluded from GAN/SFM training.  A
        # fixed X/HH subset therefore gives the meta objective real pixels
        # without shrinking the ordinary training set or touching the test.
        meta_query_data = MetaRealXHHDataset(validation_data, validation_data.records)
        meta_query_sampler = (DistributedSampler(
            meta_query_data, num_replicas=world_size, rank=rank, shuffle=True,
            seed=args.meta_transfer_seed, drop_last=True) if distributed else None)
        meta_batch_size = args.meta_transfer_query_batch_size or args.batch_size
        meta_query_loader = DataLoader(
            meta_query_data, batch_size=meta_batch_size,
            shuffle=meta_query_sampler is None, sampler=meta_query_sampler,
            num_workers=args.workers, pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0, drop_last=True)
        meta_probe = load_meta_probe(args.meta_transfer_probe_checkpoint, device)

    teacher = load_teacher(args.native_classifier_checkpoint, device)
    sfm_prototype_cache = None
    sfm_cache_signature = None
    prototype_bank = None
    if args.sfm_mode == "conditional_set_ot":
        sfm_prototype_cache = (args.sfm_prototype_cache or
                               (args.output /
                                f"sfm_prototypes__{args.band}_{args.polarization}_{args.depression}.pt"))
        signature = _prototype_cache_signature(args, train_data.records)
        sfm_cache_signature = signature
        if is_main:
            sfm_prototype_cache.parent.mkdir(parents=True, exist_ok=True)
            cache_valid = False
            if sfm_prototype_cache.is_file():
                try:
                    saved_cache = torch.load(sfm_prototype_cache, map_location="cpu",
                                             weights_only=True)
                    cache_valid = saved_cache.get("signature") == signature
                except Exception:
                    cache_valid = False
            if not cache_valid:
                print("building frozen train-only SFM prototype cache", flush=True)
                built_bank = build_prototype_bank(
                    train_data.records, teacher, device,
                    args.sfm_prototype_batch_size, args.workers)
                payload = {"signature": signature, "bank": built_bank.state_dict()}
                temporary = sfm_prototype_cache.with_suffix(
                    sfm_prototype_cache.suffix + f".{os.getpid()}.tmp")
                torch.save(payload, temporary)
                temporary.replace(sfm_prototype_cache)
        if distributed:
            dist.barrier()
        saved_cache = torch.load(sfm_prototype_cache, map_location="cpu",
                                 weights_only=True)
        if saved_cache.get("signature") != signature:
            raise RuntimeError("SFM prototype cache provenance mismatch")
        prototype_bank = ConditionalPrototypeBank(**saved_cache["bank"], device=device)
    encoder = LargeRGBIdentityEncoder(len(SOC40_CLASSES)).to(device)
    generator = HIFCUnpairedGenerator().to(device)
    discriminator = HIFCConditionedDiscriminator().to(device)
    initialise_hifc(encoder, generator, discriminator)
    ema_encoder = copy.deepcopy(encoder).eval()
    ema_generator = copy.deepcopy(generator).eval()
    set_grad(ema_encoder, False)
    set_grad(ema_generator, False)

    generator_optimizer = torch.optim.AdamW((
        {"params": encoder.parameters(), "lr": args.identity_lr},
        {"params": generator.parameters(), "lr": args.generator_lr},
    ), betas=(0.0, .99), weight_decay=1e-4)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0.0, .99))
    generator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    discriminator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    start_epoch = 1
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("architecture") != HIFC_ARCHITECTURE:
            raise RuntimeError("--resume architecture mismatch")
        saved_sfm_mode = saved.get("sfm_mode", "batch")
        if saved_sfm_mode != args.sfm_mode:
            raise RuntimeError(
                f"--resume SFM mode mismatch: checkpoint={saved_sfm_mode}, "
                f"requested={args.sfm_mode}")
        saved_sampler = saved.get("condition_sampler", "record")
        if saved_sampler != args.condition_sampler:
            raise RuntimeError(
                f"--resume condition sampler mismatch: checkpoint={saved_sampler}, "
                f"requested={args.condition_sampler}")
        encoder.load_state_dict(saved["identity_encoder"])
        generator.load_state_dict(saved["generator"])
        discriminator.load_state_dict(saved["discriminator"])
        ema_encoder.load_state_dict(saved["ema_identity_encoder"])
        ema_generator.load_state_dict(saved["ema_generator"])
        generator_optimizer.load_state_dict(saved["generator_optimizer"])
        discriminator_optimizer.load_state_dict(saved["discriminator_optimizer"])
        start_epoch = int(saved["epoch"]) + 1

    if distributed:
        # Optimizers are restored against the plain modules first.  Wrapping
        # afterwards keeps old single-GPU checkpoints compatible and avoids
        # the ``module.`` prefix in the saved state.
        ddp_kwargs = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "broadcast_buffers": True,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
        }
        encoder = DDP(encoder, **ddp_kwargs)
        generator = DDP(generator, **ddp_kwargs)
        discriminator = DDP(discriminator, **ddp_kwargs)
        # The model was broadcast by DDP.  Use independent streams for data
        # augmentation and per-rank spatial noise after that broadcast.
        rank_seed = args.seed + 100003 * rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)

    counts = {
        "rgb_identity_encoder": parameter_count(encoder),
        "hifc_generator": parameter_count(generator),
        "shared_conditioned_discriminator": parameter_count(discriminator),
    }
    counts["total_trainable"] = sum(counts.values())
    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "architecture": HIFC_ARCHITECTURE,
        "distributed": distributed,
        "world_size": world_size,
        "per_rank_batch_size": args.batch_size,
        "effective_global_batch_size": args.batch_size * world_size,
        "parameters": counts,
        "train_records": len(train_data.records),
        "validation_records": len(validation_data.records),
        "condition_layout": ["azimuth_sin", "azimuth_cos", "dep_15", "dep_30",
                              "dep_45", "dep_60", "band_X", "band_KU",
                              "pol_HH", "pol_HV", "pol_VH", "pol_VV"],
        "losses": {
            "rgb_identity": "two-view class CE + merged cosine invariance",
            "adversarial": "single conditional projection PatchGAN",
            "ltc": "local residual/contrast/Haar batch moments; no pixel matching",
            "sfm": ("condition-prototype whitened sliced-Wasserstein set matching "
                    "+ prototype anchor + D feature moments"
                    if args.sfm_mode == "conditional_set_ot" else
                    "native pre-classifier embedding cosine/moments + D feature moments"),
            "geometry": "frozen native band/polarization/depression/azimuth heads",
            "meta_transfer": ("one-step synthetic-support -> real X/HH query CE "
                              "hypergradient; disabled when weight=0"),
        },
        "sfm_prototype_cache": (str(sfm_prototype_cache)
                                 if sfm_prototype_cache is not None else None),
        "sfm_prototype_signature": sfm_cache_signature,
        "condition_sampler": args.condition_sampler,
        "condition_sampler_seed": args.condition_sampler_seed,
        "meta_transfer_query_records": (len(meta_query_data)
                                          if meta_query_data is not None else 0),
        "gradient_routes": {
            "native_gradient_mode": args.native_gradient_mode,
            "teacher_metrics": "always evaluated; mode only changes E/G gradients",
            "meta_transfer": {
                "weight": args.meta_transfer_weight,
                "every": args.meta_transfer_every,
                "inner_lr": args.meta_transfer_inner_lr,
                "probe": (str(args.meta_transfer_probe_checkpoint.resolve())
                           if args.meta_transfer_probe_checkpoint is not None else None),
                "query": "X/HH records from the deterministic validation split; test root never read",
                "gradient_target": "generator only",
            },
        },
        "unpaired_policy": "same vehicle class only; source RGB view is random and target SAR condition is metadata",
        "condition_sampling_policy": {
            "record": "empirical record frequency",
            "class_uniform": "uniform class, empirical acquisition distribution within class",
            "domain_uniform": "uniform class x shared (band, polarization, depression) domain; empirical 5-degree azimuth within class/domain",
            "support_uniform": "legacy alias of domain_uniform",
        }.get(args.condition_sampler, args.condition_sampler),
        "pixel_alignment_used": False,
        "comparison_to_v1": "new standalone HiFC-inspired path; V1 untouched",
    }
    if is_main:
        (args.output / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print({"parameters": counts, "train": train_data.summary(),
               "validation": validation_data.summary(),
               "condition": config["condition_layout"],
               "distributed": distributed, "world_size": world_size}, flush=True)

    columns = (
        "epoch", "generator", "adversarial", "rgb_identity", "ltc", "sfm",
        "geometry", "meta_transfer", "discriminator", "disc_wrong_class", "disc_wrong_condition",
        "r1", "rgb_accuracy", "native_class_accuracy", "native_band_accuracy",
        "native_polarization_accuracy", "native_depression_accuracy",
        "native_azimuth_accuracy", "validation_ltc", "validation_sfm",
        "validation_geometry", "validation_native_class_accuracy")
    history = args.output / "history.csv"
    if is_main and (start_epoch == 1 or not history.is_file()):
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(columns)

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if meta_query_sampler is not None:
            meta_query_sampler.set_epoch(epoch)
        encoder.train(); generator.train(); discriminator.train()
        set_grad(discriminator, True)
        totals = defaultdict(float)
        steps = 0
        progress = tqdm(
            train_loader, desc=f"HiFC unpaired {epoch}/{args.epochs}",
            disable=not is_main)
        train_data.set_epoch(epoch)
        meta_iterator = iter(meta_query_loader) if meta_query_loader is not None else None
        meta_events = 0
        for batch_index, batch in enumerate(progress):
            rgb = batch["rgb"].to(device, non_blocking=True)
            rgb_alt = batch["rgb_alt"].to(device, non_blocking=True)
            labels = batch["class_id"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            depression = batch["depression"].to(device, non_blocking=True)
            azimuth = batch["azimuth"].to(device, non_blocking=True)
            condition = condition_from_batch(meta, depression)
            group_code = condition_group_code(labels, meta, depression)
            real = batch["roi"].to(device, non_blocking=True)
            real_d = differentiable_augment(real)
            spatial_noise = torch.randn(len(real), 1, 64, 64, device=device)

            # The D step sees a detached sample.  A second G forward below is
            # intentional: it keeps the E/G gradient route explicit.
            with torch.no_grad(), autocast_context(device, use_amp):
                d_identity, _, d_pyramid = encoder(rgb, return_pyramid=True)
                _, _, d_fake, _ = generator(d_identity, condition, d_pyramid,
                                             spatial_noise)
            discriminator_optimizer.zero_grad(set_to_none=True)
            do_r1 = args.r1_weight > 0 and batch_index % args.r1_every == 0
            real_for_d = real_d.detach().requires_grad_(do_r1)
            with autocast_context(device, use_amp):
                real_score, _ = discriminator(real_for_d, labels, condition)
                fake_score, _ = discriminator(d_fake.detach(), labels, condition)
                wrong_class_score, _ = discriminator(
                    real_for_d, labels.roll(1), condition)
                wrong_condition_score, _ = discriminator(
                    real_for_d, labels, condition.roll(1, 0))
                d_main = discriminator_hinge(real_score, fake_score)
                wrong_class = F.relu(1.0 + wrong_class_score).mean()
                wrong_condition = F.relu(1.0 + wrong_condition_score).mean()
                d_loss = d_main + .25 * wrong_class + .25 * wrong_condition
                r1 = real.new_zeros(())
                if do_r1:
                    gradient = torch.autograd.grad(
                        real_score.sum(), real_for_d, create_graph=True)[0]
                    r1 = gradient.flatten(1).square().sum(1).mean()
                    d_loss = d_loss + .5 * args.r1_weight * args.r1_every * r1
            discriminator_scaler.scale(d_loss).backward()
            discriminator_scaler.unscale_(discriminator_optimizer)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()

            set_grad(discriminator, False)
            generator_optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                identity, rgb_logits, pyramid = encoder(rgb, return_pyramid=True)
                alt_identity, alt_logits = encoder(rgb_alt)
                fake_clean, _, fake, _ = generator(
                    identity, condition, pyramid, spatial_noise)
                fake_score, fake_d_feature = discriminator(fake, labels, condition)
                with torch.no_grad():
                    _, real_d_feature = discriminator(real_d, labels, condition)
                adversarial = -fake_score.mean()
                rgb_loss = rgb_identity_loss(
                    rgb_logits, alt_logits, identity, alt_identity, labels)
                fake_for_ltc = differentiable_augment(fake)
                ltc = local_texture_contrast_loss(fake_for_ltc, real_d)

            meta_transfer = real.new_zeros(())
            meta_diagnostics = {
                "support_before": real.new_zeros(()),
                "support_after": real.new_zeros(()),
                "query_before": real.new_zeros(()),
                "query_after": real.new_zeros(()),
            }
            meta_audit = {}
            run_meta = (meta_iterator is not None
                        and batch_index % args.meta_transfer_every == 0)
            if run_meta:
                assert meta_query_loader is not None and meta_probe is not None
                try:
                    meta_batch = next(meta_iterator)
                except StopIteration:
                    meta_iterator = iter(meta_query_loader)
                    meta_batch = next(meta_iterator)
                meta_rgb, meta_query, meta_labels, meta_geometry = meta_batch
                meta_rgb = meta_rgb.to(device, non_blocking=True)
                meta_query = meta_query.to(device, non_blocking=True)
                meta_labels = meta_labels.to(device, non_blocking=True)
                meta_geometry = meta_geometry.to(device, non_blocking=True)
                meta_azimuth = meta_geometry[:, 1]
                meta_depression = meta_geometry[:, 0]
                meta_condition_meta = torch.zeros(
                    len(meta_rgb), 8, dtype=meta_rgb.dtype, device=device)
                meta_condition_meta[:, :2] = torch.stack((
                    torch.sin(meta_azimuth.float() * (math.pi / 180.0)),
                    torch.cos(meta_azimuth.float() * (math.pi / 180.0))), dim=1)
                meta_condition_meta[:, 3] = 1.0  # X band
                meta_condition_meta[:, 4] = 1.0  # HH polarisation
                meta_condition = condition_from_batch(
                    meta_condition_meta, meta_depression)
                with torch.no_grad():
                    meta_identity, _, meta_pyramid = encoder(
                        meta_rgb, return_pyramid=True)
                meta_noise = private_randn(
                    (len(meta_rgb), 1, 64, 64), device,
                    args.meta_transfer_seed + epoch * 100003
                    + batch_index * 97 + rank * 1009)
                with torch.autocast(device_type=device.type, enabled=False):
                    meta_identity = meta_identity.detach().float()
                    meta_pyramid = tuple(value.detach().float()
                                         for value in meta_pyramid)
                    _, _, meta_fake, _ = generator(
                        meta_identity, meta_condition.float(), meta_pyramid,
                        meta_noise.float())
                meta_transfer, meta_diagnostics = meta_transfer_head_loss(
                    meta_probe, (meta_fake.float() + 1.0) * .5,
                    meta_labels, meta_query, meta_labels,
                    args.meta_transfer_inner_lr)
                meta_events += 1

            # Keep the frozen native teacher in FP32.  Its parameters are never
            # trainable, while the selected route controls whether the fake
            # input receives a gradient through the teacher representation.
            teacher_embedding_gradient = args.native_gradient_mode == "full"
            geometry_gradient = args.native_gradient_mode == "full"
            if args.native_gradient_mode == "embedding_off":
                geometry_gradient = True
            with torch.autocast(device_type=device.type, enabled=False):
                if teacher_embedding_gradient or geometry_gradient:
                    fake_logits, fake_teacher_feature = teacher(
                        ((fake.float() + 1.0) * .5).clamp(0, 1),
                        return_features=True)
                else:
                    with torch.no_grad():
                        fake_logits, fake_teacher_feature = teacher(
                            ((fake.float() + 1.0) * .5).clamp(0, 1),
                            return_features=True)
                with torch.no_grad():
                    _, real_teacher_feature = teacher(
                        ((real.float() + 1.0) * .5).clamp(0, 1), return_features=True)
                if args.sfm_mode == "conditional_set_ot":
                    sfm = conditional_set_sfm_loss(
                        fake_teacher_feature, real_teacher_feature,
                        local_texture_signature(fake_for_ltc.float()),
                        local_texture_signature(real_d.float()).detach(),
                        group_code, group_code, prototype_bank,
                        fake_d_feature.float(), real_d_feature.float(),
                        teacher_gradient=teacher_embedding_gradient,
                        projection_count=args.sfm_projection_count,
                        ltc_weight=args.sfm_ltc_cost_weight,
                        anchor_weight=args.sfm_anchor_weight,
                        ddp_global_set=distributed)
                else:
                    sfm = semantic_feature_mapping_loss(
                        fake_teacher_feature, real_teacher_feature,
                        fake_d_feature.float(), real_d_feature.float(),
                        teacher_gradient=teacher_embedding_gradient)
                geometry, geometry_terms = geometry_auxiliary_loss(
                    teacher, fake.float(), meta.float(), depression, azimuth,
                    teacher_gradient=geometry_gradient)
                adv_scale = 0.0 if epoch <= args.adversarial_warmup_epochs else 1.0
                base_g_loss = (adv_scale * args.adversarial_weight * adversarial
                               + args.rgb_identity_weight * rgb_loss
                               + args.ltc_weight * ltc
                               + args.sfm_weight * sfm
                               + args.geometry_weight * geometry)
                g_loss = base_g_loss + args.meta_transfer_weight * meta_transfer
            if (run_meta and args.meta_transfer_audit_events
                    and meta_events <= args.meta_transfer_audit_events):
                generator_parameters = tuple(generator.parameters())
                encoder_parameters = tuple(encoder.parameters())
                base_gradients = torch.autograd.grad(
                    base_g_loss, generator_parameters, retain_graph=True,
                    allow_unused=True)
                meta_gradients = torch.autograd.grad(
                    meta_transfer, generator_parameters, retain_graph=True,
                    allow_unused=True)
                meta_encoder_gradients = torch.autograd.grad(
                    meta_transfer, encoder_parameters, retain_graph=True,
                    allow_unused=True)
                base_norm = _gradient_l2(base_gradients, device)
                raw_meta_norm = _gradient_l2(meta_gradients, device)
                encoder_meta_norm = _gradient_l2(meta_encoder_gradients, device)
                meta_audit = {
                    "base_generator_grad_norm": float(base_norm),
                    "raw_meta_generator_grad_norm": float(raw_meta_norm),
                    "weighted_meta_to_base": float(
                        args.meta_transfer_weight * raw_meta_norm
                        / base_norm.clamp_min(1e-12)),
                    "meta_encoder_grad_norm": float(encoder_meta_norm),
                    "meta_requires_grad": bool(meta_transfer.requires_grad),
                }
            generator_scaler.scale(g_loss).backward()
            generator_scaler.unscale_(generator_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(generator.parameters()), 5.0)
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()
            set_grad(discriminator, True)
            update_ema(ema_encoder, encoder, args.ema_decay)
            update_ema(ema_generator, generator, args.ema_decay)

            with torch.no_grad():
                rgb_accuracy = .5 * (
                    (rgb_logits.argmax(1) == labels).float().mean()
                    + (alt_logits.argmax(1) == labels).float().mean())
                native_class = fake_logits.argmax(1)
                native_aux = teacher.auxiliary_logits(fake_teacher_feature)
                target_band = (1 - meta[:, 3].round().long()).clamp(0, 1)
                target_pol = meta[:, 4:8].argmax(1)
                target_dep = torch.round(depression.float() / 15).long().sub(1).clamp(0, 3)
                target_az = ((azimuth.long() + 15) % 360) // 30
                aux_targets = (target_band, target_pol, target_dep, target_az)
                aux_acc = [
                    (logit.argmax(1) == target).float().mean()
                    for logit, target in zip(native_aux, aux_targets)
                ]
            values = {
                "generator": g_loss, "adversarial": adversarial,
                "rgb_identity": rgb_loss, "ltc": ltc, "sfm": sfm,
                "geometry": geometry, "meta_transfer": meta_transfer,
                "discriminator": d_loss,
                "disc_wrong_class": wrong_class,
                "disc_wrong_condition": wrong_condition, "r1": r1,
                "rgb_accuracy": rgb_accuracy,
                "native_class_accuracy": (native_class == labels).float().mean(),
                "native_band_accuracy": aux_acc[0],
                "native_polarization_accuracy": aux_acc[1],
                "native_depression_accuracy": aux_acc[2],
                "native_azimuth_accuracy": aux_acc[3],
            }
            if run_meta and is_main:
                audit_path = args.output / "meta_transfer_audit.jsonl"
                with audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "epoch": epoch,
                        "batch": batch_index,
                        "event": meta_events,
                        "support_before": float(meta_diagnostics["support_before"]),
                        "support_after": float(meta_diagnostics["support_after"]),
                        "query_before": float(meta_diagnostics["query_before"]),
                        "query_after": float(meta_diagnostics["query_after"]),
                        **meta_audit,
                    }) + "\n")
            for name, value in values.items():
                totals[name] += float(value.detach())
            steps += 1
            progress.set_postfix(
                g=f"{float(g_loss.detach()):.3f}",
                d=f"{float(d_loss.detach()):.3f}",
                ltc=f"{float(ltc.detach()):.3f}",
                meta=(f"{float(meta_transfer.detach()):.3f}"
                      if run_meta else "-"),
                cls=f"{float(values['native_class_accuracy']):.3f}")
            if args.limit_train_batches and batch_index + 1 >= args.limit_train_batches:
                break

        ema_encoder.eval(); ema_generator.eval(); discriminator.eval()
        val_totals = defaultdict(float)
        val_count = 0
        preview = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(validation_loader):
                rgb = batch["rgb"].to(device)
                labels = batch["class_id"].to(device)
                meta = batch["meta"].to(device)
                depression = batch["depression"].to(device)
                condition = condition_from_batch(meta, depression)
                group_code = condition_group_code(labels, meta, depression)
                real = batch["roi"].to(device)
                identity, _, pyramid = ema_encoder(rgb, return_pyramid=True)
                noise = torch.randn(len(real), 1, 64, 64, device=device)
                clean, _, fake, _ = ema_generator(identity, condition, pyramid, noise)
                real_d = differentiable_augment(real)
                _, fake_d_feature = discriminator(fake, labels, condition)
                _, real_d_feature = discriminator(real_d, labels, condition)
                _, fake_teacher_feature = teacher(
                    ((fake + 1.0) * .5).clamp(0, 1), return_features=True)
                _, real_teacher_feature = teacher(
                    ((real + 1.0) * .5).clamp(0, 1), return_features=True)
                if args.sfm_mode == "conditional_set_ot":
                    sfm = conditional_set_sfm_loss(
                        fake_teacher_feature, real_teacher_feature,
                        local_texture_signature(fake),
                        local_texture_signature(real_d),
                        group_code, group_code, prototype_bank,
                        fake_d_feature, real_d_feature,
                        projection_count=args.sfm_projection_count,
                        ltc_weight=args.sfm_ltc_cost_weight,
                        anchor_weight=args.sfm_anchor_weight)
                else:
                    sfm = semantic_feature_mapping_loss(
                        fake_teacher_feature, real_teacher_feature,
                        fake_d_feature, real_d_feature)
                geometry, _ = geometry_auxiliary_loss(
                    teacher, fake, meta, depression, batch["azimuth"].to(device))
                ltc = local_texture_contrast_loss(fake, real_d)
                size = len(real)
                val_totals["ltc"] += float(ltc) * size
                val_totals["sfm"] += float(sfm) * size
                val_totals["geometry"] += float(geometry) * size
                val_totals["native_class_accuracy"] += float(
                    (teacher(((fake + 1.0) * .5).clamp(0, 1)).argmax(1)
                     == labels).float().sum())
                val_count += size
                if preview is None:
                    preview = (rgb, real, fake, clean)
                if (args.limit_validation_batches
                        and batch_index + 1 >= args.limit_validation_batches):
                    break
        metric_names = (
            "generator", "adversarial", "rgb_identity", "ltc", "sfm",
            "geometry", "meta_transfer", "discriminator", "disc_wrong_class",
            "disc_wrong_condition", "r1", "rgb_accuracy",
            "native_class_accuracy", "native_band_accuracy",
            "native_polarization_accuracy", "native_depression_accuracy",
            "native_azimuth_accuracy")
        reduced_totals = all_reduce_stats(
            [totals.get(name, 0.0) for name in metric_names] + [float(steps)],
            device, distributed)
        total_steps = max(reduced_totals[-1], 1.0)
        averages = {
            name: reduced_totals[index] / total_steps
            for index, name in enumerate(metric_names)}
        validation_names = ("ltc", "sfm", "geometry", "native_class_accuracy")
        reduced_validation = all_reduce_stats(
            [val_totals.get(name, 0.0) for name in validation_names]
            + [float(val_count)], device, distributed)
        total_val_count = max(reduced_validation[-1], 1.0)
        validation = {
            name: reduced_validation[index] / total_val_count
            for index, name in enumerate(validation_names)}
        row = (epoch, *[averages.get(name, float("nan")) for name in columns[1:18]],
               validation.get("ltc", float("nan")),
               validation.get("sfm", float("nan")),
               validation.get("geometry", float("nan")),
               validation.get("native_class_accuracy", float("nan")))
        if is_main:
            with history.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(row)
            state = checkpoint_state(
                epoch, encoder, generator, discriminator, ema_encoder, ema_generator,
                generator_optimizer, discriminator_optimizer, validation, args,
                sfm_prototype_cache)
            torch.save(state, args.output / "latest.pt")
            if epoch % 10 == 0 or epoch == args.epochs:
                torch.save(state, args.output / f"epoch_{epoch:03d}.pt")
            if preview is not None and (epoch == 1 or epoch % 5 == 0
                                        or epoch == args.epochs):
                save_preview(args.output / f"validation_{epoch:03d}.png", *preview)
            print(dict(zip(columns, row)), flush=True)
        if distributed:
            # Do not let a fast rank enter the next epoch while rank 0 is
            # still serializing a multi-hundred-MiB checkpoint.
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

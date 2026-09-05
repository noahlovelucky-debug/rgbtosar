"""Train SARClassifier64 solely on ROIs produced by a frozen continuous GAN.

Real train TIFF pixel values are never passed to the classifier.  They provide
only the legal observation condition (class, X/HH, depression, azimuth) used
to ask the frozen GAN for a synthetic training image.  The normal mode
evaluates on a selected held-out real acquisition domain; ``--meta-probe``
intentionally disables that evaluation and emits the registered synthetic-only
MT1 probe artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from dual_component_sar_gan import LargeRGBIdentityEncoder
from fact_sar import FACT_ARCHITECTURE
from hifc_unpaired_sar_gan import HIFC_ARCHITECTURE, HIFCUnpairedGenerator, condition_from_batch
from joint_data import JointROIDataset
from joint_models import (CodebookSpatialROIGenerator, RGBIdentityEncoder, SARStyleEncoder,
                          SpatialROIGenerator, StyleSpatialROIGenerator)
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES
from train_sar_classifier_64 import SARImageDataset


DEPRESSION_TO_ID = {15: 0, 30: 1, 45: 2, 60: 3}
BAND_TO_ID = {"X": 0, "KU": 1}
POLARIZATION_TO_ID = {"HH": 0, "HV": 1, "VH": 2, "VV": 3}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_generated_train_manifest(dataset: GeneratedConditionDataset, path: Path) -> dict[str, object]:
    """Persist the condition corpus used by a synthetic-only probe."""
    expected_records = [str(record[0].relative_to(dataset.base.sar_root))
                        for record in dataset.base.records]
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("records"):
            raise RuntimeError(f"generated train manifest has no records: {path}")
        if set(payload["records"]) != set(expected_records):
            raise RuntimeError("generated train manifest does not match the current X/HH condition corpus")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "root": str(dataset.base.sar_root.resolve()),
        "band": "X",
        "polarization": "HH",
        "records": expected_records,
        "purpose": "MT1 synthetic-only probe condition corpus; no SAR pixels are read",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def gan_condition(meta: torch.Tensor, source_angle: torch.Tensor) -> torch.Tensor:
    """Match the GAN's condition exactly, excluding real annotation-box extent."""
    meta = meta.clone()
    meta[:, -2:] = 0.0
    radians = source_angle.float() * (math.pi / 180.0)
    return torch.cat((meta, radians.sin()[:, None], radians.cos()[:, None]), dim=1)


class GeneratedConditionDataset(Dataset):
    """Conditions plus RGB source images; does not load real SAR image pixels."""

    def __init__(self, rgb_root: Path, sar_root: Path, rgb_size: int = 128,
                 include_style_roi: bool = False, band: str = "X",
                 polarization: str = "HH", depression: str = "all",
                 condition_sampler: str = "record",
                 condition_sampler_seed: int = 20260830) -> None:
        self.include_style_roi = include_style_roi
        self.band, self.polarization, self.depression = band, polarization, depression
        self.base = JointROIDataset(rgb_root, sar_root, rgb_size=rgb_size, epoch_size=0,
                                   band=band, polarization=polarization, depression=depression,
                                   augment_rgb=False, source_view_mode="random",
                                   condition_sampler=condition_sampler,
                                   condition_sampler_seed=condition_sampler_seed)

    def __len__(self) -> int:
        return len(self.base.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record_index = self.base._sample_record_index(index)
        tif, _, class_name, bbox, meta, _ = self.base.records[record_index]
        source_angle = random.choice(self.base.class_rgb_angles[class_name])
        rgb = self.base._rgb(self.base.rgb_paths[class_name, source_angle])
        targets = torch.tensor((self.base.class_to_id[class_name],
                                BAND_TO_ID[str(meta["band"]).upper()],
                                POLARIZATION_TO_ID[str(meta["pol"]).upper()],
                                DEPRESSION_TO_ID[int(meta["depression"])],
                                ((int(meta["azimuth"]) + 15) % 360) // 30), dtype=torch.long)
        if self.include_style_roi:
            with Image.open(tif) as image:
                style_roi = image_tensor(image, 64, False)
        else:
            style_roi = torch.zeros(1, 64, 64)
        return rgb, metadata_vector(meta, bbox), torch.tensor(source_angle, dtype=torch.long), targets, style_roi


class RealConditionTestDataset(Dataset):
    """Real 64x64 SAR ROIs from one held-out band/polarisation condition."""

    def __init__(self, root: Path, band: str = "X", polarization: str = "HH",
                 depression: str = "all") -> None:
        if band not in BAND_TO_ID:
            raise ValueError(f"unsupported test band: {band}")
        if polarization not in POLARIZATION_TO_ID:
            raise ValueError(f"unsupported test polarisation: {polarization}")
        if depression not in {"all", *(str(value) for value in DEPRESSION_TO_ID)}:
            raise ValueError(f"unsupported test depression: {depression}")
        self.band, self.polarization, self.depression = band, polarization, depression
        self.records: list[tuple[Path, int, int, int]] = []
        for class_id, class_name in enumerate(SOC40_CLASSES):
            for path in sorted((Path(root) / class_name).glob(f"{band}_{polarization}_*.tif")):
                try:
                    _, meta = read_annotation(path.with_suffix(".xml"))
                except Exception:
                    continue
                if depression != "all" and int(meta["depression"]) != int(depression):
                    continue
                self.records.append((path, class_id, DEPRESSION_TO_ID[int(meta["depression"])],
                                     ((int(meta["azimuth"]) + 15) % 360) // 30))
        if not self.records:
            raise RuntimeError(f"no {band}/{polarization} TIFFs under {root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, class_id, depression, azimuth = self.records[index]
        with Image.open(path) as image:
            roi = image_tensor(image, 64, False).add(1).mul(.5)
        return roi, torch.tensor((class_id, BAND_TO_ID[self.band], POLARIZATION_TO_ID[self.polarization],
                                  depression, azimuth), dtype=torch.long)


class RealXHHTestDataset(RealConditionTestDataset):
    """Backward-compatible default held-out X/HH test dataset."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, band="X", polarization="HH", depression="all")


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


def save_generated_examples(directory: Path, synthetic: torch.Tensor,
                            targets: torch.Tensor, start: int,
                            limit: int) -> int:
    """Save a small auditable sample of the synthetic classifier corpus."""
    directory.mkdir(parents=True, exist_ok=True)
    count = min(len(synthetic), max(0, limit - start))
    for index in range(count):
        array = (synthetic[index, 0].detach().cpu().clamp(0, 1).numpy()
                 * 255).round().astype("uint8")
        class_name = SOC40_CLASSES[int(targets[index, 0])].replace("/", "_")
        Image.fromarray(array, mode="L").save(
            directory / f"sample_{start + index:05d}_{class_name}.png")
    return count


def evaluate(model: SARClassifier64, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.eval(); total = correct = top5 = azimuth_correct = 0; azimuth_error = 0.0; loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(); by_depression: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    with torch.inference_mode():
        for image, targets in tqdm(loader, desc="real SAR classifier test", leave=False):
            image, targets = image.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            logits, features = model(image, return_features=True)
            labels = targets[:, 0]; prediction = logits.argmax(1)
            azimuth_logits = model.auxiliary_logits(features)[3]
            azimuth_target = targets[:, 4]
            azimuth_prediction = azimuth_logits.argmax(1)
            azimuth_correct += int((azimuth_prediction == azimuth_target).sum().item())
            azimuth_distance = (azimuth_prediction - azimuth_target).abs()
            azimuth_error += float(torch.minimum(azimuth_distance, 12 - azimuth_distance).sum().item() * 30.0)
            loss_sum += criterion(logits, labels).item() * len(labels)
            correct += (prediction == labels).sum().item()
            top5 += (logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item(); total += len(labels)
            for depression in (15, 30, 45, 60):
                mask = targets[:, 3] == DEPRESSION_TO_ID[depression]
                by_depression[depression][0] += int(mask.sum().item())
                by_depression[depression][1] += int((prediction[mask] == labels[mask]).sum().item())
    return ({"loss": loss_sum / total, "top1": correct / total, "top5": top5 / total, "samples": total,
             "azimuth_top1": azimuth_correct / total,
             "azimuth_circular_mae": azimuth_error / total},
            {str(key): {"samples": n, "top1": right / n} for key, (n, right) in by_depression.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SARClassifier64 on frozen-GAN samples, test on real X/HH")
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--gan-weights", choices=("raw", "ema"), default="raw",
                        help="use trainable raw E/G weights or the checkpoint's E/G EMA shadows")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True, help="real SAR train root; metadata only")
    parser.add_argument("--real-test-root", type=Path,
                        help="held-out real SAR root; omitted by the synthetic-only meta-probe mode")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-classifier-checkpoint", type=Path,
                        help="resume classifier weights/state from a previous latest.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoint-selection", choices=("final", "real_test_legacy"), default="final",
                        help="select the final fixed epoch (default) or reproduce the old test-set selection")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-band", choices=("all", "X", "KU"), default="X",
                        help="band conditions used for real and generated classifier training")
    parser.add_argument("--train-polarization", choices=("all", "HH", "HV", "VH", "VV"), default="HH",
                        help="polarization conditions used for real and generated classifier training")
    parser.add_argument("--train-depression", choices=("all", "15", "30", "45", "60"), default="all",
                        help="depression conditions used for real and generated classifier training")
    parser.add_argument("--test-band", choices=("X", "KU"), default="X",
                        help="held-out real SAR band used for evaluation")
    parser.add_argument("--test-polarization", choices=("HH", "HV", "VH", "VV"), default="HH",
                        help="held-out real SAR polarisation used for evaluation")
    parser.add_argument("--test-depression", choices=("all", "15", "30", "45", "60"), default="all",
                        help="held-out real SAR depression used for evaluation")
    parser.add_argument("--real-train-root", type=Path,
                        help="optional real SAR train root to mix with generated samples")
    parser.add_argument("--real-fraction", type=float, default=0.0,
                        help=("fraction of each classifier batch drawn from real train ROIs; "
                              "this is a batch ratio, not a real-dataset shot fraction"))
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help=("optional fixed optimizer updates per epoch; useful for "
                              "compute-matched real/generated comparisons"))
    parser.add_argument("--condition-sampler",
                        choices=("record", "class_uniform", "domain_uniform", "support_uniform"),
                        default="record",
                        help=("sampling distribution for generated classifier conditions; "
                              "domain_uniform balances class and shared band/polarization/depression domains"))
    parser.add_argument("--condition-sampler-seed", type=int, default=20260830)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--aux-weight", type=float, default=.12)
    parser.add_argument("--style-source", choices=("prior", "posterior"), default="prior",
                        help="posterior is a diagnostic upper bound that encodes a real train ROI style")
    parser.add_argument("--meta-probe", action="store_true",
                        help="emit the registered seed-1729 MT1 probe; never load/evaluate real SAR pixels")
    parser.add_argument("--generated-train-manifest", type=Path,
                        help="fixed condition manifest recorded in a synthetic-only probe checkpoint")
    parser.add_argument("--save-generated-dir", type=Path,
                        help="save a small sample of raw generated SAR images for inspection")
    parser.add_argument("--save-generated-count", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=415)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.real_fraction <= 1.0:
        raise ValueError("--real-fraction must be in [0, 1]")
    if args.steps_per_epoch < 0:
        raise ValueError("--steps-per-epoch must be non-negative")
    if args.real_fraction > 0.0 and args.real_train_root is None:
        raise ValueError("--real-fraction > 0 requires --real-train-root")
    if args.meta_probe:
        registered_meta_probe_seeds = (1729, 537236390)
        if args.seed not in registered_meta_probe_seeds:
            raise ValueError(
                "--meta-probe requires one of the registered synthetic-only seeds "
                f"{registered_meta_probe_seeds}")
        if args.epochs != 30:
            raise ValueError("--meta-probe requires exactly 30 fixed epochs")
        if args.checkpoint_selection != "final":
            raise ValueError("--meta-probe requires --checkpoint-selection final")
        if args.style_source != "prior":
            raise ValueError("--meta-probe cannot use a real-ROI posterior style")
        if (args.train_band, args.train_polarization, args.train_depression) != ("X", "HH", "all"):
            raise ValueError("--meta-probe requires the registered X/HH/all condition corpus")
        if (args.test_band, args.test_polarization, args.test_depression) != ("X", "HH", "all"):
            raise ValueError("--meta-probe does not permit a held-out test condition")
        if args.real_test_root is not None:
            raise ValueError("--meta-probe must omit --real-test-root so no real test is touched")
        if args.real_fraction != 0.0 or args.real_train_root is not None:
            raise ValueError("--meta-probe cannot mix real SAR pixels into its synthetic-only support")
        if args.condition_sampler != "record":
            raise ValueError("--meta-probe requires the registered record-frequency condition corpus")
    elif args.real_test_root is None:
        raise ValueError("normal classifier mode requires --real-test-root")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device); args.output.mkdir(parents=True, exist_ok=True)
    if args.resume_classifier_checkpoint is not None and not args.resume_classifier_checkpoint.is_file():
        raise FileNotFoundError(f"classifier resume checkpoint not found: {args.resume_classifier_checkpoint}")

    state = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
    architecture = state.get("architecture")
    if architecture not in {"continuous_spatial_v1", "continuous_spatial_v1_ablation",
                            "continuous_spatial_style_v2",
                            "continuous_spatial_codebook_v3", HIFC_ARCHITECTURE,
                            FACT_ARCHITECTURE} \
            or state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("expected a supported SOC40 GAN checkpoint")
    # FACT keeps the HiFC generator/condition interface but changes the
    # supervision route.  Treat it as HiFC at inference time so the standard
    # generated-to-real classifier protocol remains exactly comparable.
    is_hifc = architecture in {HIFC_ARCHITECTURE, FACT_ARCHITECTURE}
    encoder = (LargeRGBIdentityEncoder(len(SOC40_CLASSES)) if is_hifc
               else RGBIdentityEncoder(len(SOC40_CLASSES))).to(device)
    if is_hifc:
        generator = HIFCUnpairedGenerator().to(device)
        style_encoder = None
        latent_codes = None
        code_lookup = None
    elif architecture == "continuous_spatial_style_v2":
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
    if args.gan_weights == "ema":
        ema_keys = ("ema_identity_encoder", "ema_generator") if is_hifc else \
            ("identity_encoder_ema", "generator_ema")
        if any(key not in state for key in ema_keys):
            raise RuntimeError("--gan-weights ema requires EMA shadows in the GAN checkpoint")
        encoder_state, generator_state = (state[key] for key in ema_keys)
    else:
        encoder_state = state["identity_encoder"]
        generator_state = state["generator"]
    encoder.load_state_dict(encoder_state); generator.load_state_dict(generator_state)
    encoder.eval(); generator.eval()
    for parameter in (*encoder.parameters(), *generator.parameters()): parameter.requires_grad_(False)

    if args.style_source == "posterior" and architecture != "continuous_spatial_style_v2":
        raise ValueError("--style-source posterior requires a continuous_spatial_style_v2 checkpoint")
    generated_train = GeneratedConditionDataset(
        args.rgb_root, args.condition_root,
        include_style_roi=args.style_source == "posterior",
        band=args.train_band, polarization=args.train_polarization,
        depression=args.train_depression,
        condition_sampler=args.condition_sampler,
        condition_sampler_seed=args.condition_sampler_seed)
    manifest_path = None
    if args.meta_probe:
        manifest_path = args.generated_train_manifest or args.output / "generated_train_manifest.json"
        ensure_generated_train_manifest(generated_train, manifest_path)
    real_test = None if args.meta_probe else RealConditionTestDataset(
        args.real_test_root, band=args.test_band, polarization=args.test_polarization,
        depression=args.test_depression)
    real_train = None
    if args.real_fraction > 0.0:
        real_train = SARImageDataset(
            args.real_train_root, train=True,
            band=args.train_band, polarization=args.train_polarization,
            depression=args.train_depression)
    synthetic_batch_size = int(round(args.batch_size * (1.0 - args.real_fraction)))
    real_batch_size = args.batch_size - synthetic_batch_size
    if synthetic_batch_size < 0 or real_batch_size < 0 or synthetic_batch_size + real_batch_size != args.batch_size:
        raise ValueError("invalid real-fraction batch split")
    if synthetic_batch_size == 0 and real_train is None:
        raise ValueError("a pure-real classifier run requires --real-train-root")
    train_loader = (DataLoader(
        generated_train, synthetic_batch_size, shuffle=True, num_workers=args.workers,
        drop_last=True, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0) if synthetic_batch_size else None)
    real_loader = (DataLoader(
        real_train, real_batch_size, shuffle=True, num_workers=args.workers, drop_last=True,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
                   if real_train is not None and real_batch_size else None)
    if args.steps_per_epoch:
        steps_per_epoch = args.steps_per_epoch
    else:
        loader_lengths = []
        if train_loader is not None:
            loader_lengths.append(len(train_loader))
        if real_loader is not None:
            loader_lengths.append(len(real_loader))
        steps_per_epoch = max(loader_lengths, default=0)
    if steps_per_epoch <= 0:
        raise ValueError("no classifier training batches are available")
    test_loader = (DataLoader(real_test, args.batch_size * 2, shuffle=False, num_workers=args.workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
                   if real_test is not None else None)
    classifier = SARClassifier64(len(SOC40_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, min(3, args.epochs // 8))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: ((epoch + 1) / warmup if epoch < warmup else
        .5 * (1 + np.cos(np.pi * (epoch - warmup + 1) / max(1, args.epochs - warmup + 1)))))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and not args.no_amp)
    class_loss, aux_loss = nn.CrossEntropyLoss(label_smoothing=.03), nn.CrossEntropyLoss()
    history = args.output / "history.csv"
    start_epoch = 1
    resume_state = None
    if args.resume_classifier_checkpoint is not None:
        resume_state = torch.load(args.resume_classifier_checkpoint, map_location=device, weights_only=False)
        if "model" not in resume_state or "epoch" not in resume_state:
            raise RuntimeError("classifier resume checkpoint must contain model and epoch")
        classifier.load_state_dict(resume_state["model"])
        start_epoch = int(resume_state["epoch"]) + 1
        if start_epoch > args.epochs:
            raise ValueError(
                f"resume checkpoint is already at epoch {start_epoch - 1}, beyond requested {args.epochs}"
            )
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        else:
            # Old checkpoints did not persist optimizer/scheduler state.  Put
            # the fresh scheduler at the same epoch before continuing.
            scheduler.step(start_epoch - 1)
        if "scaler" in resume_state:
            scaler.load_state_dict(resume_state["scaler"])
        if "torch_rng_state" in resume_state:
            torch.set_rng_state(resume_state["torch_rng_state"])
        if "numpy_rng_state" in resume_state:
            np.random.set_state(resume_state["numpy_rng_state"])
        if "python_rng_state" in resume_state:
            random.setstate(resume_state["python_rng_state"])
    if not (args.resume_classifier_checkpoint is not None and history.is_file()):
        with history.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(("epoch", "mixed_train_loss", "mixed_train_top1", "synthetic_train_top1",
                                         "real_train_top1", "real_test_loss", "real_test_top1", "real_test_top5", "lr"))
    best = -1.0
    saved_examples = 0
    def cycle(loader):
        while loader is not None:
            yielded = False
            for batch in loader:
                yielded = True
                yield batch
            if not yielded:
                raise RuntimeError("classifier DataLoader produced no batches")

    for epoch in range(start_epoch, args.epochs + 1):
        classifier.train(); loss_sum = correct = total = 0
        synthetic_correct = synthetic_total = real_correct = real_total = 0
        synthetic_iterator = cycle(train_loader)
        real_iterator = cycle(real_loader)
        for batch_index in range(steps_per_epoch):
            synthetic = None; targets = None
            if train_loader is not None:
                rgb, meta, source_angle, targets, style_roi = next(synthetic_iterator)
                rgb, meta = rgb.to(device, non_blocking=True), meta.to(device, non_blocking=True)
                source_angle, targets = source_angle.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with torch.inference_mode():
                    identity, _, pyramid = encoder(rgb, return_pyramid=True)
                    if is_hifc:
                        # HiFC uses only target SAR acquisition metadata.  The
                        # classifier's depression target is an ID in 0..3.
                        depression = (targets[:, 3] + 1).mul(15)
                        condition = condition_from_batch(meta, depression)
                        synthetic = generator(identity, condition, pyramid)[2]
                    else:
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
                                _, style, _ = style_encoder(
                                    style_roi.to(device, non_blocking=True), sample=False)
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
                    if args.save_generated_dir is not None and saved_examples < args.save_generated_count:
                        saved_examples += save_generated_examples(
                            args.save_generated_dir, (synthetic + 1) * .5,
                            targets, saved_examples, args.save_generated_count)
                    synthetic = augment((synthetic + 1) * .5)
            classifier_images, classifier_targets = synthetic, targets
            real_targets = None
            if real_loader is not None:
                real_images, real_targets = next(real_iterator)
                real_images = real_images.to(device, non_blocking=True)
                real_targets = real_targets.to(device, non_blocking=True)
                classifier_images = real_images if classifier_images is None else torch.cat((synthetic, real_images), dim=0)
                classifier_targets = real_targets if classifier_targets is None else torch.cat((targets, real_targets), dim=0)
            if classifier_images is None or classifier_targets is None:
                raise RuntimeError("classifier batch contains neither synthetic nor real samples")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits, features = classifier(classifier_images, return_features=True)
                auxiliary = classifier.auxiliary_logits(features)
                loss = class_loss(logits, classifier_targets[:, 0])
                loss += args.aux_weight * sum(
                    aux_loss(logit, target) for logit, target in
                    zip(auxiliary, classifier_targets[:, 1:].unbind(1))) / len(auxiliary)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.)
            scaler.step(optimizer); scaler.update()
            prediction = logits.argmax(1)
            loss_sum += loss.detach().item() * len(classifier_targets)
            correct += (prediction == classifier_targets[:, 0]).sum().item(); total += len(classifier_targets)
            synthetic_count = len(synthetic) if synthetic is not None else 0
            if synthetic_count:
                synthetic_correct += (prediction[:synthetic_count] == targets[:, 0]).sum().item()
                synthetic_total += synthetic_count
            if real_targets is not None:
                real_correct += (prediction[synthetic_count:] == real_targets[:, 0]).sum().item()
                real_total += len(real_targets)
        if real_test is None:
            metrics, by_depression = ({"loss": float("nan"), "top1": float("nan"),
                                       "top5": float("nan"), "samples": 0,
                                       "azimuth_top1": float("nan"),
                                       "azimuth_circular_mae": float("nan")}, {})
        else:
            assert test_loader is not None
            metrics, by_depression = evaluate(classifier, test_loader, device)
        row = (epoch, loss_sum / total, correct / total,
               synthetic_correct / max(1, synthetic_total), real_correct / max(1, real_total),
               metrics["loss"], metrics["top1"], metrics["top5"], optimizer.param_groups[0]["lr"])
        with history.open("a", newline="", encoding="utf-8") as handle: csv.writer(handle).writerow(row)
        scheduler.step()
        saved = {"model": classifier.state_dict(), "epoch": epoch, "classes": list(SOC40_CLASSES), "input_size": 64,
                 "metrics": metrics, "gan_checkpoint": str(args.gan_checkpoint),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "torch_rng_state": torch.get_rng_state(),
                 "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
                 "training_source": (
                     "frozen GAN samples plus real train pixels"
                     if real_train is not None else "frozen GAN samples only"),
                 "real_fraction": args.real_fraction,
                 "real_batch_ratio": args.real_fraction,
                 "synthetic_batch_size": synthetic_batch_size,
                 "real_batch_size": real_batch_size,
                 "steps_per_epoch": steps_per_epoch,
                 "real_train_root": str(args.real_train_root) if args.real_train_root is not None else None,
                 "train_band": args.train_band,
                 "train_polarization": args.train_polarization,
                 "train_depression": args.train_depression,
                 "test_band": args.test_band,
                 "test_polarization": args.test_polarization,
                 "test_depression": args.test_depression}
        if args.meta_probe:
            assert manifest_path is not None
            saved["meta_probe_metadata"] = {
                "role": "meta_probe",
                "seed": int(args.seed),
                "epoch": 30,
                "selection": "fixed_final",
                "real_images_seen": False,
                "test_evaluation_performed": False,
                "band": "X",
                "polarization": "HH",
                "augmentation_version": "generated_classifier_augment_v1",
                "source_parent_sha256": file_sha256(args.gan_checkpoint),
                "generated_train_manifest": str(manifest_path.resolve()),
                "generated_train_manifest_sha256": file_sha256(manifest_path),
            }
        torch.save(saved, args.output / "latest.pt")
        select_checkpoint = (
            args.checkpoint_selection == "real_test_legacy" and metrics["top1"] >= best
        ) or (
            args.checkpoint_selection == "final" and epoch == args.epochs
        )
        if select_checkpoint:
            best = metrics["top1"]
            torch.save(saved, args.output / "best.pt")
            (args.output / "selected_metrics.json").write_text(
                json.dumps({**metrics, "by_depression": by_depression,
                            "checkpoint_selection": args.checkpoint_selection,
                            "meta_probe": args.meta_probe}, indent=2),
                encoding="utf-8")
        print(dict(zip(("epoch", "mixed_loss", "mixed_top1", "synthetic_train_top1", "real_train_top1",
                        "real_loss", "real_top1", "real_top5", "lr"), row)), flush=True)
    config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "gan_architecture": architecture,
        "synthetic_train_samples_per_epoch": len(generated_train),
        "steps_per_epoch": steps_per_epoch,
        "real_train_samples": len(real_train) if real_train is not None else 0,
        "real_fraction": args.real_fraction,
        "real_batch_ratio": args.real_fraction,
        "condition_sampler": args.condition_sampler,
        "condition_sampler_seed": args.condition_sampler_seed,
        "synthetic_batch_size": synthetic_batch_size,
        "real_batch_size": real_batch_size,
        "real_test_samples": len(real_test) if real_test is not None else 0,
        "checkpoint_selection": args.checkpoint_selection,
        "train_band": args.train_band,
        "train_polarization": args.train_polarization,
        "train_depression": args.train_depression,
        "test_band": args.test_band,
        "test_polarization": args.test_polarization,
        "test_depression": args.test_depression,
        "training_policy": (
            "classifier sees generated pixels only; generator samples a frozen empirical spatial-code prior "
            "learned from real train SAR" if architecture == "continuous_spatial_codebook_v3" else
            "classifier sees generated pixels only; posterior diagnostic uses real train ROI only "
            "inside the frozen style encoder" if args.style_source == "posterior" else
            "frozen GAN samples plus real train pixels"
            if real_train is not None else
            "no real SAR pixels in classifier training; real train root supplies condition labels only")}
    if args.meta_probe:
        assert manifest_path is not None
        config["generated_train_manifest"] = str(manifest_path.resolve())
        config["generated_train_manifest_sha256"] = file_sha256(manifest_path)
        config["meta_probe_metadata"] = saved["meta_probe_metadata"]
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

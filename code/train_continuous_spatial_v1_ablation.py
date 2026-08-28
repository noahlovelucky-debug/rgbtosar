"""Reproducible, one-variable loss ablations from the archived V1 GAN.

This trainer intentionally keeps V1's RGB encoder, spatial generator and
conditional PatchGAN.  Loss weights are exposed independently, while a frozen
real-SAR geometry validator is used only for fixed validation, never as a
training gradient.  It is a separate entry point because the original V1
trainer was later replaced by the Fused V2 implementation.
"""
from __future__ import annotations

import argparse
import copy
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

from bbox_data import image_tensor
from joint_data import JointROIDataset
from joint_models import (
    ContinuousROIDiscriminator,
    RGBIdentityEncoder,
    SpatialROIGenerator,
    _align_translation,
    angle_curvature_loss,
    initialise,
    sar_perceptual_pyramid_loss,
    sar_statistics_loss,
    weighted_aligned_structure_loss,
    weighted_physics_prior_loss,
)
from sar_classifier_64 import SARClassifier64
from sar_geometry_validator import DEPRESSION_VALUES, SARGeometryValidator, circular_degree_error
from saratrx import SOC40_CLASSES


DEPRESSION_TO_ID = {15: 0, 30: 1, 45: 2, 60: 3}
LOSS_NAMES = (
    "rgb_identity", "cross_view", "sar_class", "cluster", "structure",
    "statistics", "physics", "perceptual", "angle", "adversarial", "feature_match",
)
WEIGHT_ARGUMENTS = {
    "rgb_identity": "rgb_id_weight",
    "cross_view": "cross_view_weight",
    "sar_class": "sar_class_weight",
    "cluster": "cluster_weight",
    "structure": "structure_weight",
    "statistics": "statistics_weight",
    "physics": "physics_weight",
    "perceptual": "perceptual_weight",
    "angle": "angle_smooth_weight",
    "adversarial": "adversarial_weight",
    "feature_match": "feature_match_weight",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original-V1 one-variable loss ablation trainer")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--initialise-checkpoint", type=Path, required=True,
                        help="continuous_spatial_v1 parent checkpoint")
    parser.add_argument("--parent-epoch", type=int, default=-1,
                        help="epoch represented by the parent; default reads it from the checkpoint")
    parser.add_argument("--geometry-validator-checkpoint", type=Path,
                        help="frozen real-SAR-only validator; never used for GAN gradients")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path,
                        help="shared fixed train/validation partition; defaults beside output")
    parser.add_argument("--validation-proxy-manifest", type=Path,
                        help="shared balanced proxy used when --validation-batches is non-zero")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--epoch-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prototype-batch-size", type=int, default=128)
    parser.add_argument("--prototype-cache", type=Path,
                        help="shared native feature-prototype cache for branches using cluster loss")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rgb-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--identity-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=5e-5)
    parser.add_argument("--discriminator-every", type=int, default=2)
    parser.add_argument("--discriminator-class-mode", choices=("disabled", "real_only"), default="disabled",
                        help="optional class auxiliary head on the existing PatchGAN; real_only uses only real SAR")
    parser.add_argument("--discriminator-class-weight", type=float, default=0.0,
                        help="weight for the real-only PatchGAN class CE; zero preserves V1 behavior")
    parser.add_argument("--generator-discriminator-class-weight", type=float, default=0.0,
                        help="optional G-step CE through the real-trained PatchGAN class head; default is disabled")
    parser.add_argument("--rgb-id-weight", type=float, default=10.0)
    parser.add_argument("--cross-view-weight", type=float, default=2.0)
    parser.add_argument("--sar-class-weight", type=float, default=12.0)
    parser.add_argument("--cluster-weight", type=float, default=5.0)
    parser.add_argument("--structure-weight", type=float, default=20.0)
    parser.add_argument("--statistics-weight", type=float, default=5.0)
    parser.add_argument("--physics-weight", type=float, default=3.0)
    parser.add_argument("--physics-amplitude-weight", type=float, default=1.0)
    parser.add_argument("--physics-scatter-weight", type=float, default=1.0)
    parser.add_argument("--physics-correlation-weight", type=float, default=1.0)
    parser.add_argument("--perceptual-weight", type=float, default=0.0)
    parser.add_argument("--angle-smooth-weight", type=float, default=.2)
    parser.add_argument("--angle-loss-mode", choices=("first_order", "curvature"), default="first_order",
                        help="V1 first-order smoothness or the separate curvature ablation")
    parser.add_argument("--adversarial-weight", type=float, default=2.0)
    parser.add_argument("--feature-match-weight", type=float, default=5.0)
    # These default coefficients exactly reproduce the archived V1 structure objective.
    parser.add_argument("--structure-pixel-64-weight", type=float, default=1.0)
    parser.add_argument("--structure-pixel-32-weight", type=float, default=.5)
    parser.add_argument("--structure-pixel-16-weight", type=float, default=.25)
    parser.add_argument("--structure-edge-weight", type=float, default=.5)
    parser.add_argument("--structure-ssim-weight", type=float, default=1.0)
    parser.add_argument("--wrong-azimuth-discriminator-weight", type=float, default=0.0,
                        help="single-variable PatchGAN wrong-condition negative; V1 default is zero")
    parser.add_argument("--discriminator-condition", choices=("full", "target"), default="full",
                        help="V1's full 12D condition or only target azimuth/depression")
    parser.add_argument("--gradient-routing", choices=("coupled", "generator_only"), default="coupled",
                        help="coupled preserves V1; generator_only blocks SAR-side losses from RGB encoder")
    parser.add_argument("--assert-gradient-routing", action="store_true",
                        help="check the first generator backward for the generator_only route")
    parser.add_argument("--rgb-loss-mode", choices=("separate", "joint_equivalent"), default="separate",
                        help="keep V1 RGB terms separate or report their exact weighted sum as one term")
    parser.add_argument("--speckle-warmup-epochs", type=int, default=8)
    parser.add_argument("--speckle-ramp-epochs", type=int, default=5)
    parser.add_argument("--source-view-mode", choices=("nearest", "random", "mixed"), default="mixed")
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--validation-batches", type=int, default=20,
                        help="0 evaluates the full fixed validation set")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def target_condition(meta: torch.Tensor, source_angle: torch.Tensor) -> torch.Tensor:
    meta = meta.clone()
    meta[:, -2:] = 0.0
    radians = source_angle.float() * (math.pi / 180.0)
    return torch.cat((meta, radians.sin()[:, None], radians.cos()[:, None]), dim=1)


def rotate_target_azimuth(condition: torch.Tensor, degrees: float = 5.0) -> torch.Tensor:
    radians = torch.atan2(condition[:, 0], condition[:, 1]) + math.radians(degrees)
    result = condition.clone()
    result[:, 0], result[:, 1] = radians.sin(), radians.cos()
    return result


def rotate_vector(vector: torch.Tensor, degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    sine, cosine = vector[:, 0], vector[:, 1]
    return torch.stack((sine * math.cos(radians) + cosine * math.sin(radians),
                        cosine * math.cos(radians) - sine * math.sin(radians)), dim=1)


def target_vectors(azimuth: torch.Tensor) -> torch.Tensor:
    radians = azimuth.float() * (math.pi / 180.0)
    return torch.stack((radians.sin(), radians.cos()), dim=1)


def depression_ids(values: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(values, dtype=torch.long)
    for index, value in enumerate(DEPRESSION_VALUES):
        result[values == value] = index
    return result


def set_grad(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def route_generator_inputs(identity: torch.Tensor, pyramid: tuple[torch.Tensor, ...],
                           mode: str) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Select whether SAR-side losses may send gradients into the RGB encoder."""
    if mode == "coupled":
        return identity, pyramid
    if mode == "generator_only":
        return identity.detach(), tuple(value.detach() for value in pyramid)
    raise ValueError(f"unsupported gradient routing mode: {mode}")


def combine_rgb_losses(identity_loss: torch.Tensor, cross_view_loss: torch.Tensor,
                       identity_weight: float, cross_view_weight: float,
                       mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally group RGB terms without changing their weighted objective."""
    if mode == "separate":
        return identity_loss, cross_view_loss
    if mode == "joint_equivalent":
        if identity_weight <= 0:
            raise ValueError("joint_equivalent RGB loss requires a positive rgb identity weight")
        return identity_loss + (cross_view_weight / identity_weight) * cross_view_loss, identity_loss.new_zeros(())
    raise ValueError(f"unsupported RGB loss mode: {mode}")


def discriminator_condition(condition: torch.Tensor, mode: str) -> torch.Tensor:
    """Keep only physical target attributes when testing the discriminator input."""
    if mode == "full":
        return condition
    if mode == "target":
        # The first three entries are target [sin(azimuth), cos(azimuth), depression].
        return condition[:, :3]
    raise ValueError(f"unsupported discriminator condition mode: {mode}")


def load_parent_discriminator(discriminator: ContinuousROIDiscriminator,
                              parent_state: dict[str, torch.Tensor], mode: str) -> str:
    """Migrate V1's PatchGAN without randomising unrelated D weights."""
    target_state = discriminator.state_dict()
    for name, value in parent_state.items():
        if name == "condition.0.weight":
            target_state[name] = (value.clone() if mode == "full" else value[:, :3].clone())
        elif name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value
    # The archived V1 D has no auxiliary classifier.  Keep the new head at
    # zero so adding it is an exact D0 no-op until its real-only loss is enabled.
    if not any(name.startswith("classifier.") for name in parent_state):
        target_state["classifier.weight"] = torch.zeros_like(target_state["classifier.weight"])
        target_state["classifier.bias"] = torch.zeros_like(target_state["classifier.bias"])
    discriminator.load_state_dict(target_state)
    return ("exact V1 state + zero auxiliary class head" if mode == "full" else
            "V1 state; condition.0.weight restricted to target azimuth/depression columns")


class PrototypeROIDataset(Dataset):
    def __init__(self, dataset: JointROIDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tif, _, class_name, bbox, meta, _ = self.dataset.records[index]
        with Image.open(tif) as image:
            source = image if self.dataset.pre_cropped else image.crop(bbox)
            roi = image_tensor(source, 64, False)
        return (roi,
                torch.tensor(self.dataset.class_to_id[class_name], dtype=torch.long),
                torch.tensor(DEPRESSION_TO_ID[int(meta["depression"])], dtype=torch.long))


def prototype_signature(dataset: JointROIDataset, classifier: Path) -> dict[str, object]:
    return {
        "version": 1,
        "records": [str(record[0].relative_to(dataset.sar_root)) for record in dataset.records],
        "classifier": str(classifier.resolve()),
    }


def conditional_prototypes(judge: SARClassifier64, dataset: JointROIDataset,
                           args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    cache = args.prototype_cache or args.output / "native_conditional_prototypes.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    signature = prototype_signature(dataset, args.native_classifier_checkpoint)
    if cache.is_file():
        try:
            saved = torch.load(cache, map_location="cpu", weights_only=True)
            if saved.get("signature") == signature:
                return saved["prototypes"].to(device)
        except Exception:
            pass
    sums = torch.zeros(40, 4, judge.feature_dim, device=device)
    counts = torch.zeros(40, 4, device=device)
    loader = DataLoader(PrototypeROIDataset(dataset), batch_size=args.prototype_batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    with torch.inference_mode():
        for roi, labels, depression in tqdm(loader, desc="V1 ablation conditional prototypes"):
            roi, labels, depression = roi.to(device), labels.to(device), depression.to(device)
            _, features = judge((roi + 1.0) * .5, return_features=True)
            features = F.normalize(features, dim=1)
            sums.index_put_((labels, depression), features, accumulate=True)
            counts.index_put_((labels, depression), torch.ones_like(labels, dtype=torch.float), accumulate=True)
    if (counts == 0).any():
        raise RuntimeError("missing class/depression prototype")
    prototypes = F.normalize(sums / counts[..., None], dim=2)
    torch.save({"prototypes": prototypes.cpu(), "counts": counts.cpu(), "signature": signature}, cache)
    return prototypes


def record_key(record: tuple, root: Path) -> str:
    return str(record[0].relative_to(root))


def build_split(records: list[tuple], root: Path, fraction: float, seed: int,
                manifest: Path) -> tuple[set[str], set[str]]:
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("root") == str(root.resolve()):
            return set(saved["train"]), set(saved["validation"])
    groups: dict[tuple[str, int, int], list[tuple]] = defaultdict(list)
    for record in records:
        class_name, meta = record[2], record[4]
        groups[class_name, int(meta["depression"]), int(meta["azimuth"]) // 30].append(record)
    train, validation = [], []
    for group, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda record: hashlib.sha256(
            f"{seed}:{record_key(record, root)}".encode()).hexdigest())
        count = max(1, round(len(ordered) * fraction)) if len(ordered) > 1 else 0
        validation.extend(record_key(record, root) for record in ordered[:count])
        train.extend(record_key(record, root) for record in ordered[count:])
    if not train or not validation:
        raise RuntimeError("fixed ablation split is empty")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "version": 1, "root": str(root.resolve()), "seed": seed,
        "validation_fraction": fraction, "train": sorted(train), "validation": sorted(validation),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return set(train), set(validation)


def build_balanced_proxy(records: list[tuple], root: Path, count: int, seed: int,
                         manifest: Path) -> list[str]:
    """Return a fixed class-balanced subset for inexpensive repeated validation.

    The record scanner groups paths by class, so using the first N validation
    rows would silently evaluate only a few classes.  This round-robin order
    gives each class comparable representation and rotates its depression /
    azimuth buckets before any bucket is repeated.
    """
    available = sorted(record_key(record, root) for record in records)
    signature = hashlib.sha256("\n".join(available).encode()).hexdigest()
    desired = min(count, len(available))
    if manifest.is_file():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if (saved.get("root") == str(root.resolve())
                    and saved.get("available_signature") == signature
                    and saved.get("count") == desired):
                selected = [str(value) for value in saved["records"]]
                if len(selected) == desired and set(selected).issubset(available):
                    return selected
        except (KeyError, TypeError, json.JSONDecodeError):
            pass

    buckets: dict[str, dict[tuple[int, int], list[tuple]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        meta = record[4]
        buckets[record[2]][int(meta["depression"]), int(meta["azimuth"]) // 30].append(record)
    sequences: dict[str, list[tuple]] = {}
    for class_name in SOC40_CLASSES:
        grouped = buckets[class_name]
        keys = sorted(grouped, key=lambda key: hashlib.sha256(
            f"{seed}:{class_name}:{key}".encode()).hexdigest())
        for key in keys:
            grouped[key].sort(key=lambda record: hashlib.sha256(
                f"{seed}:{record_key(record, root)}".encode()).hexdigest())
        sequence: list[tuple] = []
        maximum = max((len(grouped[key]) for key in keys), default=0)
        for level in range(maximum):
            for key in keys:
                if level < len(grouped[key]):
                    sequence.append(grouped[key][level])
        if sequence:
            sequences[class_name] = sequence
    classes = sorted(sequences, key=lambda name: hashlib.sha256(
        f"{seed}:class:{name}".encode()).hexdigest())
    cursor = {name: 0 for name in classes}
    selected: list[str] = []
    while classes and len(selected) < desired:
        progressed = False
        for class_name in classes:
            index = cursor[class_name]
            if index < len(sequences[class_name]):
                selected.append(record_key(sequences[class_name][index], root))
                cursor[class_name] += 1
                progressed = True
                if len(selected) == desired:
                    break
        if not progressed:
            break
    if len(selected) != desired:
        raise RuntimeError("unable to construct requested balanced validation proxy")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "version": 1, "root": str(root.resolve()), "seed": seed,
        "count": desired, "available_signature": signature, "records": selected,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


def records_from_keys(records: list[tuple], root: Path, keys: list[str]) -> list[tuple]:
    lookup = {record_key(record, root): record for record in records}
    try:
        return [lookup[key] for key in keys]
    except KeyError as error:
        raise RuntimeError(f"validation proxy references missing record: {error.args[0]}") from error


def configure_records(dataset: JointROIDataset, included: set[str], root: Path,
                      epoch_size: int, random_epoch: bool) -> None:
    dataset.records = [record for record in dataset.records if record_key(record, root) in included]
    if not dataset.records:
        raise RuntimeError("ablation split has no records")
    dataset.epoch_size = epoch_size or len(dataset.records)
    dataset.random_epoch = random_epoch and 0 < epoch_size < len(dataset.records)


def save_preview(rgb: torch.Tensor, real: torch.Tensor, fake: torch.Tensor, path: Path) -> None:
    rows = []
    for index in range(min(8, len(fake))):
        rgb_panel = F.interpolate(rgb[index:index + 1], (64, 64), mode="bilinear", align_corners=False)[0]
        rgb_panel = ((rgb_panel.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        panels = [rgb_panel]
        for image in (real[index, 0], fake[index, 0]):
            panel = ((image.detach().cpu().clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
            panels.append(np.repeat(panel[..., None], 3, axis=2))
        rows.append(np.concatenate(panels, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0), "RGB").save(path)


@torch.inference_mode()
def validate_geometry(encoder: RGBIdentityEncoder, generator: SpatialROIGenerator,
                      validator: SARGeometryValidator | None, loader: DataLoader,
                      device: torch.device, use_amp: bool, speckle: float,
                      seed: int, limit_batches: int) -> dict[str, float]:
    if validator is None:
        return {}
    encoder.eval(); generator.eval(); validator.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    for batch_index, batch in enumerate(loader):
        rgb, real = batch["rgb"].to(device), batch["roi"].to(device)
        labels = batch["class_id"].to(device)
        azimuth, depression = batch["azimuth"].to(device), batch["depression"].to(device)
        condition = target_condition(batch["meta"].to(device), batch["rgb_angle"].to(device))
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            identity, _, pyramid = encoder(rgb, return_pyramid=True)
            clean = generator(identity, condition, pyramid, apply_speckle=False)
            devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed + batch_index)
                fake = generator.apply_speckle(clean, speckle)
            offset = generator(identity, rotate_target_azimuth(condition, 30.0), pyramid,
                               apply_speckle=False)
            fake_output = validator((fake + 1.0) * .5)
            real_output = validator((real + 1.0) * .5)
            offset_output = validator((offset + 1.0) * .5)
        target = target_vectors(azimuth)
        aligned_real = _align_translation(clean, real)
        values = {
            "samples": float(len(labels)),
            "generated_identity": float((fake_output.identity_logits.argmax(1) == labels).float().sum()),
            "real_identity": float((real_output.identity_logits.argmax(1) == labels).float().sum()),
            "generated_depression": float((fake_output.depression_logits.argmax(1) == depression_ids(depression)).float().sum()),
            "real_depression": float((real_output.depression_logits.argmax(1) == depression_ids(depression)).float().sum()),
            "generated_azimuth_mae": float(circular_degree_error(fake_output.azimuth_vector, target).sum()),
            "real_azimuth_mae": float(circular_degree_error(real_output.azimuth_vector, target).sum()),
            "pair_30_error": float(circular_degree_error(
                rotate_vector(fake_output.azimuth_vector, 30.0), offset_output.azimuth_vector).sum()),
            "feature_cosine": float(F.cosine_similarity(fake_output.features, real_output.features, dim=1).sum()),
            "aligned_lowpass_l1": float((F.avg_pool2d(clean, 4) - F.avg_pool2d(aligned_real, 4)).abs().mean((1, 2, 3)).sum()),
            "response_lowpass_l1_30": float((F.avg_pool2d(clean, 4) - F.avg_pool2d(offset, 4)).abs().mean((1, 2, 3)).sum()),
        }
        for name, value in values.items():
            totals[name] += value
        if limit_batches and batch_index + 1 >= limit_batches:
            break
    samples = max(totals["samples"], 1.0)
    return {name: (value if name == "samples" else value / samples) for name, value in totals.items()}


def pareto_epochs(rows: list[dict[str, float]]) -> list[int]:
    """Return non-dominated epochs; selection remains explicit rather than a hidden scalar."""
    objectives = (
        ("generated_identity", True), ("generated_depression", True),
        ("feature_cosine", True), ("generated_azimuth_mae", False),
        ("pair_30_error", False), ("aligned_lowpass_l1", False),
    )
    answer = []
    for candidate in rows:
        if not candidate:
            continue
        dominated = False
        for other in rows:
            if not other or other is candidate:
                continue
            comparisons = [
                other[name] >= candidate[name] if maximise else other[name] <= candidate[name]
                for name, maximise in objectives
            ]
            strict = [
                other[name] > candidate[name] if maximise else other[name] < candidate[name]
                for name, maximise in objectives
            ]
            if all(comparisons) and any(strict):
                dominated = True
                break
        if not dominated:
            answer.append(int(candidate["epoch"]))
    return answer


def main() -> None:
    args = arguments()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    manifest = args.split_manifest or args.output.parent / "v1_ablation_split.json"

    full_data = JointROIDataset(args.rgb_root, args.sar_root, rgb_size=args.rgb_size, epoch_size=0,
                                band="X", polarization="HH", depression="all", augment_rgb=True,
                                source_view_mode=args.source_view_mode)
    train_keys, validation_keys = build_split(full_data.records, args.sar_root, args.validation_fraction,
                                              args.seed, manifest)
    train_data = copy.copy(full_data)
    configure_records(train_data, train_keys, args.sar_root, args.epoch_size, random_epoch=True)
    validation_data = copy.copy(full_data)
    validation_data.augment_rgb = False
    validation_data.source_view_mode = "nearest"
    configure_records(validation_data, validation_keys, args.sar_root, 0, random_epoch=False)
    validation_proxy_manifest = None
    if args.validation_batches:
        validation_proxy_manifest = args.validation_proxy_manifest or manifest.with_name(
            "v1_ablation_validation_proxy.json")
        proxy_keys = build_balanced_proxy(
            validation_data.records, args.sar_root,
            args.validation_batches * args.batch_size, args.seed,
            validation_proxy_manifest)
        validation_eval_data = copy.copy(validation_data)
        validation_eval_data.records = records_from_keys(validation_data.records, args.sar_root, proxy_keys)
        validation_eval_data.epoch_size = len(validation_eval_data.records)
        validation_eval_data.random_epoch = False
    else:
        validation_eval_data = validation_data
    loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_eval_data, args.batch_size, shuffle=False, num_workers=args.workers,
                                   persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")

    needs_judge = bool(args.sar_class_weight or args.cluster_weight or args.perceptual_weight)
    judge: SARClassifier64 | None = None
    prototypes: torch.Tensor | None = None
    if needs_judge:
        saved_classifier = torch.load(args.native_classifier_checkpoint, map_location=device, weights_only=False)
        if saved_classifier.get("classes") != list(SOC40_CLASSES):
            raise RuntimeError("native classifier class order does not match SOC40")
        judge = SARClassifier64(40).to(device)
        judge.load_state_dict(saved_classifier["model"]); judge.eval(); set_grad(judge, False)
        if args.cluster_weight:
            prototypes = conditional_prototypes(judge, train_data, args, device)
    validator = None
    if args.geometry_validator_checkpoint:
        validator_state = torch.load(args.geometry_validator_checkpoint, map_location=device, weights_only=False)
        if validator_state.get("architecture") != "sar_geometry_validator_v2":
            raise RuntimeError("geometry validator checkpoint has the wrong architecture")
        validator = SARGeometryValidator(40).to(device)
        validator.load_state_dict(validator_state["model"]); validator.eval(); set_grad(validator, False)

    parent = torch.load(args.initialise_checkpoint, map_location=device, weights_only=False)
    if parent.get("architecture") not in {"continuous_spatial_v1", "continuous_spatial_v1_ablation"}:
        raise RuntimeError("--initialise-checkpoint must be V1 or a V1 ablation checkpoint")
    encoder = RGBIdentityEncoder(40).to(device)
    generator = SpatialROIGenerator(meta_dim=12).to(device)
    discriminator_meta_dim = 12 if args.discriminator_condition == "full" else 3
    discriminator = ContinuousROIDiscriminator(meta_dim=discriminator_meta_dim).to(device)
    if args.discriminator_class_weight < 0:
        raise ValueError("--discriminator-class-weight must be non-negative")
    if args.generator_discriminator_class_weight < 0:
        raise ValueError("--generator-discriminator-class-weight must be non-negative")
    if args.discriminator_class_mode == "disabled" and args.discriminator_class_weight:
        raise ValueError("a non-zero discriminator class weight requires --discriminator-class-mode real_only")
    if args.discriminator_class_mode == "disabled" and args.generator_discriminator_class_weight:
        raise ValueError("a non-zero generator discriminator class weight requires --discriminator-class-mode real_only")
    if (args.generator_discriminator_class_weight
            and not args.discriminator_class_weight):
        raise ValueError("the generator class head requires a non-zero real-only discriminator class loss")
    encoder.apply(initialise); generator.apply(initialise); discriminator.apply(initialise)
    encoder.load_state_dict(parent["identity_encoder"])
    generator.load_state_dict(parent["generator"])
    discriminator_migration = load_parent_discriminator(
        discriminator, parent["discriminator"], args.discriminator_condition)
    parent_epoch = args.parent_epoch if args.parent_epoch >= 0 else int(parent.get("epoch", 0))
    generator_optimizer = torch.optim.Adam(
        ({"params": encoder.parameters(), "lr": args.identity_lr},
         {"params": generator.parameters(), "lr": args.lr}), betas=(.5, .999), foreach=False)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr,
                                                betas=(.5, .999), foreach=False)
    generator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    discriminator_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=.02)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"parent_checkpoint": str(args.initialise_checkpoint.resolve()),
                   "parent_epoch": parent_epoch,
                   "split_manifest": str(manifest.resolve()),
                   "validation_proxy_manifest": str(validation_proxy_manifest.resolve())
                   if validation_proxy_manifest else None,
                   "train_records": len(train_data.records), "validation_records": len(validation_data.records),
                   "validation_proxy_records": len(validation_eval_data.records),
                   "native_judge_loaded": needs_judge,
                   "discriminator_migration": discriminator_migration,
                   "policy": "single-variable V1 ablation; frozen geometry validator only"})
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    header = ("run_epoch", "epoch", "loss_total", "loss_encoder", "loss_generator",
              *(f"loss_{name}" for name in LOSS_NAMES),
              *(f"contribution_{name}" for name in LOSS_NAMES), "loss_discriminator",
              "loss_discriminator_class", "loss_discriminator_wrong_azimuth",
              "discriminator_real_class_accuracy", "discriminator_fake_class_accuracy",
              "loss_generator_discriminator_class", "generator_discriminator_class_accuracy",
              "rgb_identity_accuracy", "native_fake_accuracy", "cluster_cosine", "speckle_strength",
              "validation_samples", "validation_generated_identity",
              "validation_real_identity", "validation_generated_depression", "validation_real_depression",
              "validation_generated_azimuth_mae", "validation_real_azimuth_mae", "validation_pair_30_error",
              "validation_feature_cosine", "validation_aligned_lowpass_l1", "validation_response_lowpass_l1_30")
    history = args.output / "history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(header)
    validation_rows: list[dict[str, float]] = []
    print({"train": train_data.summary(), "validation_records": len(validation_data.records),
           "validation_proxy_records": len(validation_eval_data.records),
           "parent": str(args.initialise_checkpoint), "parent_epoch": parent_epoch,
           "discriminator_migration": discriminator_migration, "weights": {
               key: getattr(args, value) for key, value in WEIGHT_ARGUMENTS.items()}}, flush=True)

    for run_epoch in range(1, args.epochs + 1):
        epoch = parent_epoch + run_epoch
        encoder.train(); generator.train(); discriminator.train()
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        speckle = generator.speckle_strength * min(1.0, max(
            0.0, (epoch - args.speckle_warmup_epochs) / max(1, args.speckle_ramp_epochs)))
        for batch_index, batch in enumerate(tqdm(
                loader, desc=f"V1 ablation {run_epoch}/{args.epochs} (epoch {epoch})")):
            rgb, rgb_alt = batch["rgb"].to(device), batch["rgb_alt"].to(device)
            real, meta, labels = batch["roi"].to(device), batch["meta"].to(device), batch["class_id"].to(device)
            condition = target_condition(meta, batch["rgb_angle"].to(device))
            depression = torch.tensor(
                [DEPRESSION_TO_ID[int(value)] for value in batch["depression"].tolist()], device=device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                identity, rgb_logits, pyramid = encoder(rgb, return_pyramid=True)
                alternate_identity, alternate_logits = encoder(rgb_alt)
                # The coupled route reproduces V1.  The generator-only route
                # prevents SAR teacher/structure/physics/D gradients from
                # changing the RGB identity representation or its pyramid.
                generator_identity, generator_pyramid = route_generator_inputs(
                    identity, pyramid, args.gradient_routing)
                clean = generator(generator_identity, condition, generator_pyramid, apply_speckle=False)
                fake = generator.apply_speckle(clean, speckle)

            discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_meta = discriminator_condition(condition, args.discriminator_condition)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                discriminator_class_loss = real.new_zeros(())
                discriminator_real_class_accuracy = real.new_zeros(())
                discriminator_fake_class_accuracy = real.new_zeros(())
                if args.discriminator_class_mode == "real_only":
                    real_score, _, real_class_logits = discriminator(
                        real, discriminator_meta, return_class_logits=True)
                    fake_score, _, fake_class_logits = discriminator(
                        fake.detach(), discriminator_meta, return_class_logits=True)
                    discriminator_class_loss = cross_entropy(real_class_logits, labels)
                    discriminator_loss = (F.relu(1.0 - real_score).mean()
                                          + F.relu(1.0 + fake_score).mean()
                                          + args.discriminator_class_weight * discriminator_class_loss)
                    discriminator_real_class_accuracy = (real_class_logits.argmax(1) == labels).float().mean()
                    discriminator_fake_class_accuracy = (fake_class_logits.argmax(1) == labels).float().mean()
                else:
                    real_score, _ = discriminator(real, discriminator_meta)
                    fake_score, _ = discriminator(fake.detach(), discriminator_meta)
                    discriminator_loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
                wrong_azimuth_loss = real.new_zeros(())
                if args.wrong_azimuth_discriminator_weight:
                    wrong_condition = rotate_target_azimuth(condition, random.choice((-90., -60., -30., 30., 60., 90.)))
                    wrong_score, _ = discriminator(
                        real, discriminator_condition(wrong_condition, args.discriminator_condition))
                    wrong_azimuth_loss = F.relu(1.0 + wrong_score).mean()
                    discriminator_loss = discriminator_loss + args.wrong_azimuth_discriminator_weight * wrong_azimuth_loss
            if batch_index % args.discriminator_every == 0:
                discriminator_scaler.scale(discriminator_loss).backward()
                discriminator_scaler.unscale_(discriminator_optimizer)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.)
                discriminator_scaler.step(discriminator_optimizer); discriminator_scaler.update()

            set_grad(discriminator, False); generator_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                generator_discriminator_class_loss = real.new_zeros(())
                generator_discriminator_class_accuracy = real.new_zeros(())
                if args.generator_discriminator_class_weight:
                    fake_score, fake_disc_features, fake_class_logits = discriminator(
                        fake, discriminator_meta, return_class_logits=True)
                    generator_discriminator_class_loss = cross_entropy(fake_class_logits, labels)
                    generator_discriminator_class_accuracy = (
                        (fake_class_logits.argmax(1) == labels).float().mean())
                else:
                    fake_score, fake_disc_features = discriminator(fake, discriminator_meta)
                with torch.no_grad():
                    _, real_disc_features = discriminator(real, discriminator_meta)
                sar_logits = sar_features = None
                fake_sar_pyramid = None
                if judge is not None:
                    sar_logits, sar_features, fake_sar_pyramid = judge((fake + 1.0) * .5, return_pyramid=True)
                if args.angle_loss_mode == "first_order":
                    angle_loss = F.l1_loss(F.avg_pool2d(clean, 4), F.avg_pool2d(
                        generator(generator_identity, rotate_target_azimuth(condition), generator_pyramid,
                                  apply_speckle=False), 4))
                else:
                    left = generator(generator_identity, rotate_target_azimuth(condition, -5.0), generator_pyramid,
                                     apply_speckle=False)
                    right = generator(generator_identity, rotate_target_azimuth(condition, 5.0), generator_pyramid,
                                      apply_speckle=False)
                    angle_loss = angle_curvature_loss(left, clean, right)
                rgb_identity_loss = .5 * (cross_entropy(rgb_logits, labels)
                                          + cross_entropy(alternate_logits, labels))
                cross_view_loss = 1.0 - (F.normalize(identity, dim=1)
                                         * F.normalize(alternate_identity, dim=1)).sum(1).mean()
                rgb_identity_loss, cross_view_loss = combine_rgb_losses(
                    rgb_identity_loss, cross_view_loss, args.rgb_id_weight,
                    args.cross_view_weight, args.rgb_loss_mode)
                raw_losses = {
                    "rgb_identity": rgb_identity_loss,
                    "cross_view": cross_view_loss,
                    "sar_class": cross_entropy(sar_logits, labels) if args.sar_class_weight else real.new_zeros(()),
                    "cluster": (1.0 - (F.normalize(sar_features, dim=1)
                                * prototypes[labels, depression]).sum(1).mean())
                    if args.cluster_weight else real.new_zeros(()),
                    "statistics": sar_statistics_loss(fake, real),
                    "angle": angle_loss,
                    "adversarial": -fake_score.mean(),
                    "feature_match": (F.l1_loss(fake_disc_features.mean((2, 3)), real_disc_features.mean((2, 3)))
                                      + F.l1_loss(fake_disc_features.std((2, 3)), real_disc_features.std((2, 3)))),
                }
                raw_losses["structure"], _ = weighted_aligned_structure_loss(
                    clean, real, pixel_64_weight=args.structure_pixel_64_weight,
                    pixel_32_weight=args.structure_pixel_32_weight,
                    pixel_16_weight=args.structure_pixel_16_weight,
                    edge_weight=args.structure_edge_weight, ssim_weight=args.structure_ssim_weight)
                raw_losses["physics"], _ = weighted_physics_prior_loss(
                    clean, real, amplitude_weight=args.physics_amplitude_weight,
                    scatter_weight=args.physics_scatter_weight,
                    correlation_weight=args.physics_correlation_weight)
                if args.perceptual_weight:
                    if judge is None or fake_sar_pyramid is None:
                        raise RuntimeError("perceptual loss requires the native SAR judge")
                    aligned_real = _align_translation(clean, real)
                    with torch.no_grad():
                        _, _, real_sar_pyramid = judge((aligned_real + 1.0) * .5, return_pyramid=True)
                    raw_losses["perceptual"] = sar_perceptual_pyramid_loss(fake_sar_pyramid, real_sar_pyramid)
                else:
                    raw_losses["perceptual"] = real.new_zeros(())
                contributions = {
                    name: raw_losses[name] * getattr(args, WEIGHT_ARGUMENTS[name]) for name in LOSS_NAMES}
                encoder_loss = contributions["rgb_identity"] + contributions["cross_view"]
                generator_loss = sum(contributions[name] for name in LOSS_NAMES
                                     if name not in {"rgb_identity", "cross_view"})
                generator_loss = (generator_loss
                                  + args.generator_discriminator_class_weight
                                  * generator_discriminator_class_loss)
                total_loss = encoder_loss + generator_loss
                if args.assert_gradient_routing and args.gradient_routing == "generator_only" and batch_index == 0:
                    encoder_parameters = tuple(encoder.parameters())
                    generator_parameters = tuple(generator.parameters())
                    sar_to_encoder = torch.autograd.grad(
                        generator_loss, encoder_parameters, retain_graph=True, allow_unused=True)
                    rgb_to_generator = torch.autograd.grad(
                        encoder_loss, generator_parameters, retain_graph=True, allow_unused=True)
                    sar_norm = sum(float(value.detach().abs().max()) for value in sar_to_encoder if value is not None)
                    rgb_norm = sum(float(value.detach().abs().max()) for value in rgb_to_generator if value is not None)
                    if sar_norm > 1e-8 or rgb_norm > 1e-8:
                        raise RuntimeError(
                            f"generator_only gradient routing failed: sar_to_encoder={sar_norm}, "
                            f"rgb_to_generator={rgb_norm}")
            generator_scaler.scale(total_loss).backward()
            generator_scaler.unscale_(generator_optimizer)
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(generator.parameters()), 5.)
            generator_scaler.step(generator_optimizer); generator_scaler.update()
            set_grad(discriminator, True)

            totals["loss_total"] += float(total_loss.detach())
            totals["loss_encoder"] += float(encoder_loss.detach())
            totals["loss_generator"] += float(generator_loss.detach())
            for name in LOSS_NAMES:
                totals[f"loss_{name}"] += float(raw_losses[name].detach())
                totals[f"contribution_{name}"] += float(contributions[name].detach())
            totals["loss_discriminator"] += float(discriminator_loss.detach())
            totals["loss_discriminator_class"] += float(discriminator_class_loss.detach())
            totals["loss_discriminator_wrong_azimuth"] += float(wrong_azimuth_loss.detach())
            totals["discriminator_real_class_accuracy"] += float(discriminator_real_class_accuracy.detach())
            totals["discriminator_fake_class_accuracy"] += float(discriminator_fake_class_accuracy.detach())
            totals["loss_generator_discriminator_class"] += float(
                generator_discriminator_class_loss.detach())
            totals["generator_discriminator_class_accuracy"] += float(
                generator_discriminator_class_accuracy.detach())
            totals["rgb_identity_accuracy"] += float(.5 * (
                (rgb_logits.argmax(1) == labels).float().mean() + (alternate_logits.argmax(1) == labels).float().mean()))
            if sar_logits is not None:
                totals["native_fake_accuracy"] += float((sar_logits.argmax(1) == labels).float().mean())
            if args.cluster_weight:
                totals["cluster_cosine"] += float(1.0 - raw_losses["cluster"].detach())
            batches += 1
            if args.max_train_batches and batch_index + 1 >= args.max_train_batches:
                break

        averages = {key: value / max(batches, 1) for key, value in totals.items()}
        metrics: dict[str, float] = {}
        if validator is not None and epoch % args.validation_every == 0:
            metrics = validate_geometry(encoder, generator, validator, validation_loader, device, use_amp,
                                        speckle, args.seed + epoch * 10000, 0)
            metrics["epoch"] = float(epoch)
            validation_rows.append(metrics)
            (args.output / "validation_pareto.json").write_text(json.dumps({
                "metrics": validation_rows, "pareto_epochs": pareto_epochs(validation_rows),
                "policy": "non-dominated epochs only; do not select by native classifier accuracy",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        row_values = {
            "run_epoch": run_epoch, "epoch": epoch, **averages, "speckle_strength": speckle,
            "validation_samples": metrics.get("samples", float("nan")),
            "validation_generated_identity": metrics.get("generated_identity", float("nan")),
            "validation_real_identity": metrics.get("real_identity", float("nan")),
            "validation_generated_depression": metrics.get("generated_depression", float("nan")),
            "validation_real_depression": metrics.get("real_depression", float("nan")),
            "validation_generated_azimuth_mae": metrics.get("generated_azimuth_mae", float("nan")),
            "validation_real_azimuth_mae": metrics.get("real_azimuth_mae", float("nan")),
            "validation_pair_30_error": metrics.get("pair_30_error", float("nan")),
            "validation_feature_cosine": metrics.get("feature_cosine", float("nan")),
            "validation_aligned_lowpass_l1": metrics.get("aligned_lowpass_l1", float("nan")),
            "validation_response_lowpass_l1_30": metrics.get("response_lowpass_l1_30", float("nan")),
        }
        with history.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([row_values.get(name, float("nan")) for name in header])
        checkpoint = {
            "architecture": "continuous_spatial_v1_ablation", "parent_architecture": parent["architecture"],
            "parent_checkpoint": str(args.initialise_checkpoint), "parent_epoch": parent_epoch,
            "run_epoch": run_epoch, "epoch": epoch,
            "identity_encoder": encoder.state_dict(), "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(), "classes": list(SOC40_CLASSES),
            "args": config, "speckle_strength": speckle, "metrics": metrics,
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if run_epoch % args.save_every == 0 or run_epoch == args.epochs:
            torch.save(checkpoint, args.output / f"epoch_{epoch:04d}.pt")
            save_preview(rgb, real, fake, args.output / f"preview_{epoch:04d}.png")
        print(row_values, flush=True)


if __name__ == "__main__":
    main()

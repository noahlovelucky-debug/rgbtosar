"""Fast generated-to-real transfer probe for V1 loss-ablation finalists.

The GAN is frozen.  A real-SAR geometry validator is frozen too; this script
only fits class/depression centroids and a ridge azimuth readout in that fixed
feature space.  Training the readout on generated features and testing it on
held-out real features (TSTR) is substantially harder to game than asking the
native classifier used in V1's loss to classify its own generator output.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from joint_data import JointROIDataset
from joint_models import RGBIdentityEncoder, SpatialROIGenerator
from sar_geometry_validator import SARGeometryValidator, circular_degree_error
from saratrx import SOC40_CLASSES
from train_continuous_spatial_v1_ablation import (
    build_balanced_proxy,
    records_from_keys,
)


SUPPORTED_ARCHITECTURES = {"continuous_spatial_v1", "continuous_spatial_v1_ablation"}
DEPRESSION_TO_ID = {15: 0, 30: 1, 45: 2, 60: 3}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit cheap generated-to-real transfer probes in frozen real-SAR feature space")
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-validator-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=1920)
    parser.add_argument("--validation-samples", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=.01)
    parser.add_argument("--observed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=451)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def target_condition(meta: torch.Tensor, source_angle: torch.Tensor) -> torch.Tensor:
    meta = meta.clone()
    meta[:, -2:] = 0.0
    radians = source_angle.float() * (math.pi / 180.0)
    return torch.cat((meta, radians.sin()[:, None], radians.cos()[:, None]), dim=1)


def load_manifest(path: Path, root: Path) -> tuple[set[str], set[str]]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("root") != str(root.resolve()):
        raise RuntimeError("split manifest belongs to a different SAR root")
    return set(saved["train"]), set(saved["validation"])


def record_key(record: tuple, root: Path) -> str:
    return str(record[0].relative_to(root))


def select_records(dataset: JointROIDataset, included: set[str], count: int, seed: int,
                   manifest: Path) -> list[tuple]:
    eligible = [record for record in dataset.records if record_key(record, dataset.sar_root) in included]
    keys = build_balanced_proxy(eligible, dataset.sar_root, count, seed, manifest)
    return records_from_keys(eligible, dataset.sar_root, keys)


def make_dataset(base: JointROIDataset, records: list[tuple]) -> JointROIDataset:
    result = copy.copy(base)
    result.records = records
    result.epoch_size = len(records)
    result.random_epoch = False
    result.augment_rgb = False
    result.source_view_mode = "nearest"
    return result


def extract_features(encoder: RGBIdentityEncoder, generator: SpatialROIGenerator,
                     validator: SARGeometryValidator, loader: DataLoader, device: torch.device,
                     observed: bool, speckle: float, seed: int) -> dict[str, torch.Tensor]:
    fake_features: list[torch.Tensor] = []
    real_features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    depressions: list[torch.Tensor] = []
    azimuths: list[torch.Tensor] = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="cross-domain features", leave=False)):
            rgb = batch["rgb"].to(device, non_blocking=True)
            real = batch["roi"].to(device, non_blocking=True)
            condition = target_condition(
                batch["meta"].to(device, non_blocking=True),
                batch["rgb_angle"].to(device, non_blocking=True))
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                identity, _, pyramid = encoder(rgb, return_pyramid=True)
                clean = generator(identity, condition, pyramid, apply_speckle=False)
                if observed:
                    devices = [device.index or 0] if device.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(seed + batch_index)
                        fake = generator.apply_speckle(clean, speckle)
                else:
                    fake = clean
                fake_output = validator((fake + 1.0) * .5)
                real_output = validator((real + 1.0) * .5)
            fake_features.append(fake_output.features.float().cpu())
            real_features.append(real_output.features.float().cpu())
            labels.append(batch["class_id"].long().cpu())
            depressions.append(torch.tensor(
                [DEPRESSION_TO_ID[int(value)] for value in batch["depression"].tolist()], dtype=torch.long))
            radians = batch["azimuth"].float() * (math.pi / 180.0)
            azimuths.append(torch.stack((radians.sin(), radians.cos()), dim=1).cpu())
    return {
        "fake": torch.cat(fake_features), "real": torch.cat(real_features),
        "class": torch.cat(labels), "depression": torch.cat(depressions),
        "azimuth": torch.cat(azimuths),
    }


def centroid_accuracy(train_features: torch.Tensor, train_targets: torch.Tensor,
                      test_features: torch.Tensor, test_targets: torch.Tensor,
                      classes: int) -> float:
    train_features = F.normalize(train_features, dim=1)
    centroids = torch.zeros(classes, train_features.shape[1])
    counts = torch.zeros(classes)
    centroids.index_add_(0, train_targets, train_features)
    counts.index_add_(0, train_targets, torch.ones_like(train_targets, dtype=torch.float))
    if (counts == 0).any():
        raise RuntimeError("transfer probe is missing a target class")
    centroids = F.normalize(centroids / counts[:, None], dim=1)
    predictions = F.normalize(test_features, dim=1) @ centroids.T
    return float((predictions.argmax(1) == test_targets).float().mean())


def ridge_azimuth_error(train_features: torch.Tensor, train_targets: torch.Tensor,
                        test_features: torch.Tensor, test_targets: torch.Tensor,
                        ridge: float) -> float:
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    train_x = torch.cat((train_features, torch.ones(len(train_features), 1)), dim=1)
    test_x = torch.cat((test_features, torch.ones(len(test_features), 1)), dim=1)
    regularizer = torch.eye(train_x.shape[1]) * ridge
    regularizer[-1, -1] = 0.0  # Do not shrink the intercept.
    weights = torch.linalg.solve(train_x.T @ train_x + regularizer, train_x.T @ train_targets)
    prediction = F.normalize(test_x @ weights, dim=1)
    return float(circular_degree_error(prediction, test_targets).mean())


def transfer_metrics(train: dict[str, torch.Tensor], validation: dict[str, torch.Tensor],
                     source: str, target: str, ridge: float) -> dict[str, float]:
    return {
        "identity_top1": centroid_accuracy(
            train[source], train["class"], validation[target], validation["class"], len(SOC40_CLASSES)),
        "depression_top1": centroid_accuracy(
            train[source], train["depression"], validation[target], validation["depression"], 4),
        "azimuth_degree_mae": ridge_azimuth_error(
            train[source], train["azimuth"], validation[target], validation["azimuth"], ridge),
    }


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    state = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(f"unsupported GAN architecture: {state.get('architecture')!r}")
    if state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("GAN class order differs from SOC40")
    validator_state = torch.load(args.geometry_validator_checkpoint, map_location=device, weights_only=False)
    if validator_state.get("architecture") != "sar_geometry_validator_v2":
        raise RuntimeError("geometry validator checkpoint has the wrong architecture")
    encoder = RGBIdentityEncoder(len(SOC40_CLASSES)).to(device)
    generator = SpatialROIGenerator(meta_dim=12).to(device)
    validator = SARGeometryValidator(len(SOC40_CLASSES)).to(device)
    encoder.load_state_dict(state["identity_encoder"])
    generator.load_state_dict(state["generator"])
    validator.load_state_dict(validator_state["model"])
    for module in (encoder, generator, validator):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    train_keys, validation_keys = load_manifest(args.split_manifest, args.sar_root)
    base = JointROIDataset(
        args.rgb_root, args.sar_root, epoch_size=0, band="X", polarization="HH",
        depression="all", augment_rgb=False, source_view_mode="nearest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_records = select_records(
        base, train_keys, args.train_samples, args.seed,
        args.output.with_name(args.output.stem + "_train_proxy.json"))
    validation_records = select_records(
        base, validation_keys, args.validation_samples, args.seed + 1,
        args.output.with_name(args.output.stem + "_validation_proxy.json"))
    train_loader = DataLoader(make_dataset(base, train_records), args.batch_size, shuffle=False,
                              num_workers=args.workers, persistent_workers=args.workers > 0,
                              pin_memory=device.type == "cuda")
    validation_loader = DataLoader(make_dataset(base, validation_records), args.batch_size, shuffle=False,
                                   num_workers=args.workers, persistent_workers=args.workers > 0,
                                   pin_memory=device.type == "cuda")
    speckle = float(state.get("speckle_strength", generator.speckle_strength))
    train = extract_features(encoder, generator, validator, train_loader, device, args.observed,
                             speckle, args.seed + 1000)
    validation = extract_features(encoder, generator, validator, validation_loader, device, args.observed,
                                  speckle, args.seed + 2000)
    report = {
        "architecture": state["architecture"],
        "gan_checkpoint": str(args.gan_checkpoint.resolve()),
        "geometry_validator_checkpoint": str(args.geometry_validator_checkpoint.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "observed": args.observed,
        "speckle_strength": speckle,
        "samples": {"generated_train": len(train_records), "real_validation": len(validation_records)},
        "generated_to_real": transfer_metrics(train, validation, "fake", "real", args.ridge),
        "real_to_real_reference": transfer_metrics(train, validation, "real", "real", args.ridge),
        "real_to_generated": transfer_metrics(train, validation, "real", "fake", args.ridge),
        "paired_validation_feature_cosine": float(F.cosine_similarity(
            validation["fake"], validation["real"], dim=1).mean()),
        "interpretation": {
            "generated_to_real": "primary transfer metric; higher identity/depression and lower azimuth MAE are better",
            "real_to_real_reference": "upper-reference for the same fixed feature/readout procedure",
            "real_to_generated": "diagnostic only; asymmetry indicates domain mismatch",
            "native_classifier": "intentionally absent because it is part of V1's GAN loss",
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

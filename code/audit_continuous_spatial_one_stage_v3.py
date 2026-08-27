"""Independent official-test audit for Continuous Spatial V3."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from continuous_spatial_one_stage_v3 import (
    ARCHITECTURE, ContinuousSpatialOneStageV3, target_geometry)
from joint_data import JointROIDataset
from sar_classifier_64 import SARClassifier64
from sar_geometry_validator import (
    DEPRESSION_VALUES, SARGeometryValidator, circular_degree_error)
from saratrx import SOC40_CLASSES


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--gan-checkpoint", type=Path, required=True)
    result.add_argument(
        "--geometry-validator-checkpoint", type=Path, required=True)
    result.add_argument(
        "--native-classifier-checkpoint", type=Path, required=True)
    result.add_argument("--rgb-root", type=Path, required=True)
    result.add_argument("--sar-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--seed", type=int, default=9871)
    result.add_argument("--max-batches", type=int, default=0)
    result.add_argument(
        "--device",
        default="cuda:2" if torch.cuda.is_available() else "cpu")
    return result


def per_sample_correlation(
        first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.flatten(1)
    second = second.flatten(1)
    first = first - first.mean(1, keepdim=True)
    second = second - second.mean(1, keepdim=True)
    return (
        (first * second).mean(1)
        / (first.std(1) * second.std(1) + 1e-6)).abs()


def high_frequency(image: torch.Tensor) -> torch.Tensor:
    low = F.interpolate(
        F.avg_pool2d(image, 4), image.shape[-2:],
        mode="bilinear", align_corners=False)
    return image - low


def normalize(values: dict[str, float]) -> dict[str, float | int]:
    samples = int(values.get("samples", 0))
    if samples == 0:
        return {"samples": 0}
    return {
        name: samples if name == "samples" else value / samples
        for name, value in values.items()}


def main() -> None:
    args = parser().parse_args()
    device = torch.device(args.device)
    state = torch.load(
        args.gan_checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") != ARCHITECTURE:
        raise RuntimeError("incompatible GAN checkpoint")
    validator_state = torch.load(
        args.geometry_validator_checkpoint,
        map_location=device, weights_only=False)
    native_state = torch.load(
        args.native_classifier_checkpoint,
        map_location=device, weights_only=False)
    if validator_state.get("architecture") != "sar_geometry_validator_v2":
        raise RuntimeError("incompatible geometry validator")
    if native_state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("native classifier class order mismatch")

    model = ContinuousSpatialOneStageV3(
        len(SOC40_CLASSES)).to(device)
    model.encoder.load_state_dict(
        state.get("ema_encoder", state["encoder"]))
    model.generator.load_state_dict(
        state.get("ema_generator", state["generator"]))
    validator = SARGeometryValidator(
        len(SOC40_CLASSES)).to(device)
    validator.load_state_dict(validator_state["model"])
    native = SARClassifier64(len(SOC40_CLASSES)).to(device)
    native.load_state_dict(native_state["model"])
    model.eval()
    validator.eval()
    native.eval()

    dataset = JointROIDataset(
        args.rgb_root, args.sar_root,
        band="X", polarization="HH", depression="all",
        augment_rgb=False, source_view_mode="nearest",
        return_all_views=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda")
    totals: defaultdict[str, float] = defaultdict(float)
    by_depression = {
        int(value): defaultdict(float)
        for value in DEPRESSION_VALUES}

    with torch.inference_mode():
        for batch_index, batch in enumerate(
                tqdm(loader, desc="V3 official test audit")):
            views = batch["rgb_views"].to(device)
            view_angles = batch["rgb_view_angles"].to(device)
            view_mask = batch["rgb_view_mask"].to(device)
            labels = batch["class_id"].to(device)
            real = batch["roi"].to(device)
            azimuth = batch["azimuth"].to(device).float()
            depression = batch["depression"].to(device).float()
            metadata = batch["meta"].to(device)
            geometry = target_geometry(
                metadata, azimuth, depression)
            encoding = model.encode(views, view_mask)
            first_generator = torch.Generator(device=device)
            first_generator.manual_seed(args.seed + batch_index)
            second_generator = torch.Generator(device=device)
            second_generator.manual_seed(
                args.seed + 100000 + batch_index)
            first_field = torch.randn(
                len(real), 3, 64, 64, device=device,
                generator=first_generator)
            second_field = torch.randn(
                len(real), 3, 64, 64, device=device,
                generator=second_generator)
            first = model.generator(
                encoding, view_angles, view_mask,
                azimuth, depression, geometry, first_field)
            second = model.generator(
                encoding, view_angles, view_mask,
                azimuth, depression, geometry, second_field)
            fake_geometry = validator((first.sar + 1.0) * .5)
            real_geometry = validator((real + 1.0) * .5)
            fake_native_logits, fake_native_features = native(
                (first.sar + 1.0) * .5, return_features=True)
            real_native_logits, real_native_features = native(
                (real + 1.0) * .5, return_features=True)
            # The whitened residual is the quantity that should be independent
            # of vehicle identity.  Raw additive differences naturally inherit
            # the target amplitude and are therefore not a valid leakage test.
            residual_logits = native(
                first.whitened_noise.clamp(-3, 3).div(6).add(.5))
            depression_id = (
                depression.div(15).round().long() - 1).clamp(0, 3)
            radians = azimuth * (math.pi / 180.0)
            target_vector = torch.stack(
                (radians.sin(), radians.cos()), 1)
            low_first = F.avg_pool2d(first.sar, 4)
            low_second = F.avg_pool2d(second.sar, 4)
            metrics = {
                "samples": torch.ones_like(depression),
                "generated_identity_correct":
                    (fake_geometry.identity_logits.argmax(1)
                     == labels).float(),
                "real_identity_correct":
                    (real_geometry.identity_logits.argmax(1)
                     == labels).float(),
                "generated_native_identity_correct":
                    (fake_native_logits.argmax(1) == labels).float(),
                "real_native_identity_correct":
                    (real_native_logits.argmax(1) == labels).float(),
                "whitened_residual_identity_correct":
                    (residual_logits.argmax(1) == labels).float(),
                "generated_depression_correct":
                    (fake_geometry.depression_logits.argmax(1)
                     == depression_id).float(),
                "real_depression_correct":
                    (real_geometry.depression_logits.argmax(1)
                     == depression_id).float(),
                "generated_azimuth_cosine":
                    (fake_geometry.azimuth_vector
                     * target_vector).sum(1),
                "real_azimuth_cosine":
                    (real_geometry.azimuth_vector
                     * target_vector).sum(1),
                "generated_azimuth_degree_error":
                    circular_degree_error(
                        fake_geometry.azimuth_vector,
                        target_vector),
                "real_azimuth_degree_error":
                    circular_degree_error(
                        real_geometry.azimuth_vector,
                        target_vector),
                "geometry_feature_cosine_to_real":
                    F.cosine_similarity(
                        fake_geometry.features,
                        real_geometry.features, dim=1),
                "native_feature_cosine_to_real":
                    F.cosine_similarity(
                        fake_native_features,
                        real_native_features, dim=1),
                "seed_correlation":
                    per_sample_correlation(
                        first.whitened_noise,
                        second.whitened_noise),
                "full_seed_l1":
                    (first.sar - second.sar).abs().mean((1, 2, 3)),
                "lowpass_seed_l1":
                    (low_first - low_second).abs().mean((1, 2, 3)),
                "highpass_seed_l1":
                    (high_frequency(first.sar)
                     - high_frequency(second.sar)).abs().mean((1, 2, 3)),
                "sigma": first.sigma.flatten(1).mean(1),
                "receiver_scale":
                    first.receiver_scale.flatten(1).mean(1),
                "attention_nearest_mass":
                    first.attention.max(1).values,
            }
            for name, values in metrics.items():
                totals[name] += float(values.sum())
                for value in DEPRESSION_VALUES:
                    mask = depression == value
                    if mask.any():
                        by_depression[int(value)][name] += float(
                            values[mask].sum())
            if args.max_batches and batch_index + 1 >= args.max_batches:
                break

    overall = normalize(totals)
    report = {
        "architecture": ARCHITECTURE,
        "condition": (
            "X/HH; 12 RGB views; continuous azimuth; "
            "depressions=15,30,45,60"),
        "all": overall,
        "by_depression": {
            str(key): normalize(values)
            for key, values in by_depression.items()},
        "gan_checkpoint": str(args.gan_checkpoint.resolve()),
        "geometry_validator_checkpoint":
            str(args.geometry_validator_checkpoint.resolve()),
        "native_classifier_checkpoint":
            str(args.native_classifier_checkpoint.resolve()),
        "acceptance_targets": {
            "geometry_feature_cosine_to_real": ">=0.5595",
            "identity_gap": "generated >= real - 0.05",
            "seed_correlation": "<0.10",
            "lowpass_seed_l1": "<0.02",
            "whitened_residual_identity_correct": "<0.05",
            "sixty_degree_feature_cosine": ">0.3589",
        },
    }
    if isinstance(overall.get("generated_identity_correct"), float):
        report["acceptance"] = {
            "identity_gap": (
                overall["generated_identity_correct"]
                >= overall["real_identity_correct"] - .05),
            "feature_cosine": (
                overall["geometry_feature_cosine_to_real"] >= .5595),
            "seed_correlation": overall["seed_correlation"] < .10,
            "lowpass_seed_l1": overall["lowpass_seed_l1"] < .02,
            "residual_identity_leakage": (
                overall["whitened_residual_identity_correct"] < .05),
            "sixty_degree_feature": (
                report["by_depression"]["60"]
                .get("geometry_feature_cosine_to_real", -1) > .3589),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Independent test-set audit for geometry and stochasticity of SAR GAN v2."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dual_component_sar_gan_v2 import (
    MultiViewDenoisedSARGenerator, MultiViewRGBEncoder,
    NoiseLeakageClassifier, StochasticSARObservation,
    residual_view, target_geometry)
from joint_data import JointROIDataset
from sar_geometry_validator import (
    DEPRESSION_VALUES, SARGeometryValidator, circular_degree_error)
from saratrx import SOC40_CLASSES
from train_dual_component_sar_gan_v2 import ARCHITECTURE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-validator-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9871)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    gan_state = torch.load(
        args.gan_checkpoint, map_location=device, weights_only=False)
    if gan_state.get("architecture") != ARCHITECTURE:
        raise RuntimeError("incompatible GAN checkpoint")
    validator_state = torch.load(
        args.geometry_validator_checkpoint, map_location=device,
        weights_only=False)
    if validator_state.get("architecture") != "sar_geometry_validator_v2":
        raise RuntimeError("incompatible validator checkpoint")

    encoder = MultiViewRGBEncoder(len(SOC40_CLASSES)).to(device)
    clean_generator = MultiViewDenoisedSARGenerator().to(device)
    observation = StochasticSARObservation().to(device)
    leakage = NoiseLeakageClassifier(len(SOC40_CLASSES)).to(device)
    validator = SARGeometryValidator(len(SOC40_CLASSES)).to(device)
    encoder.load_state_dict(gan_state["ema_encoder"])
    clean_generator.load_state_dict(gan_state["ema_clean_generator"])
    observation.load_state_dict(gan_state["ema_observation"])
    leakage.load_state_dict(gan_state["noise_leakage_classifier"])
    validator.load_state_dict(validator_state["model"])
    for module in (
            encoder, clean_generator, observation, leakage, validator):
        module.eval()

    dataset = JointROIDataset(
        args.rgb_root, args.sar_root, band="X", polarization="HH",
        depression="all", augment_rgb=False, source_view_mode="nearest",
        return_all_views=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=torch.cuda.is_available())
    totals: defaultdict[str, float] = defaultdict(float)
    by_depression: dict[int, defaultdict[str, float]] = {
        value: defaultdict(float) for value in DEPRESSION_VALUES}

    with torch.inference_mode():
        for batch_index, batch in enumerate(
                tqdm(loader, desc="dual v2 official audit")):
            views = batch["rgb_views"].to(device)
            source_angles = batch["rgb_view_angles"].to(device)
            view_mask = batch["rgb_view_mask"].to(device)
            labels = batch["class_id"].to(device)
            real = batch["roi"].to(device)
            azimuth = batch["azimuth"].to(device).float()
            depression = batch["depression"].to(device).float()
            meta = batch["meta"].to(device)
            geometry = target_geometry(
                azimuth, depression, meta[:, 3:8])
            encoding = encoder(views, view_mask)
            clean, attention = clean_generator(
                encoding, source_angles, view_mask,
                azimuth, depression, geometry)
            first_generator = torch.Generator(device=device)
            first_generator.manual_seed(args.seed + batch_index)
            second_generator = torch.Generator(device=device)
            second_generator.manual_seed(
                args.seed + 100000 + batch_index)
            first_field = torch.randn(
                len(real), observation.random_channels, 64, 64,
                device=device, generator=first_generator)
            second_field = torch.randn(
                len(real), observation.random_channels, 64, 64,
                device=device, generator=second_generator)
            first = observation(clean, depression, first_field)
            second = observation(clean, depression, second_field)
            first_residual = residual_view(clean, first.observed)
            fake_output = validator((first.observed + 1) * .5)
            real_output = validator((real + 1) * .5)
            depression_id = (
                depression.div(15).round().long() - 1).clamp(0, 3)
            radians = azimuth * (math.pi / 180)
            target_vector = torch.stack(
                (radians.sin(), radians.cos()), 1)
            def per_sample_correlation(
                    first_noise: torch.Tensor,
                    second_noise: torch.Tensor) -> torch.Tensor:
                first_flat = first_noise.flatten(1)
                second_flat = second_noise.flatten(1)
                first_flat = first_flat - first_flat.mean(1, keepdim=True)
                second_flat = second_flat - second_flat.mean(1, keepdim=True)
                return ((first_flat * second_flat).mean(1) / (
                    first_flat.std(1) * second_flat.std(1) + 1e-6)).abs()

            metrics = {
                "samples": torch.ones_like(depression),
                "generated_identity_correct":
                    (fake_output.identity_logits.argmax(1) == labels).float(),
                "real_identity_correct":
                    (real_output.identity_logits.argmax(1) == labels).float(),
                "generated_depression_correct":
                    (fake_output.depression_logits.argmax(1)
                     == depression_id).float(),
                "real_depression_correct":
                    (real_output.depression_logits.argmax(1)
                     == depression_id).float(),
                "generated_azimuth_cosine":
                    (fake_output.azimuth_vector * target_vector).sum(1),
                "real_azimuth_cosine":
                    (real_output.azimuth_vector * target_vector).sum(1),
                "generated_azimuth_degree_error":
                    circular_degree_error(
                        fake_output.azimuth_vector, target_vector),
                "real_azimuth_degree_error":
                    circular_degree_error(
                        real_output.azimuth_vector, target_vector),
                "feature_cosine_to_real":
                    F.cosine_similarity(
                        fake_output.features, real_output.features, dim=1),
                "noise_seed_correlation":
                    per_sample_correlation(
                        first.log_multiplicative,
                        second.log_multiplicative),
                "noise_seed_l1":
                    (first.log_multiplicative
                     - second.log_multiplicative).abs().mean((1, 2, 3)),
                "full_seed_l1":
                    (first.observed
                     - second.observed).abs().mean((1, 2, 3)),
                "noise_leakage_correct":
                    (leakage(first_residual).argmax(1) == labels).float(),
                "attention_nearest_mass": attention.max(1).values,
            }
            for name, values in metrics.items():
                totals[name] += float(values.sum())
                for value in DEPRESSION_VALUES:
                    mask = depression == value
                    if mask.any():
                        by_depression[value][name] += float(
                            values[mask].sum())
            if args.max_batches and batch_index + 1 >= args.max_batches:
                break

    def normalize(values: dict[str, float]) -> dict[str, float | int]:
        raw_samples = int(values.get("samples", 0))
        if raw_samples == 0:
            return {"samples": 0}
        return {
            name: (
                raw_samples if name == "samples" else value / raw_samples)
            for name, value in values.items()}

    report = {
        "architecture": ARCHITECTURE,
        "condition": "X/HH, 12 RGB views, azimuth continuous, depression 15/30/45/60",
        "all": normalize(totals),
        "by_depression": {
            str(key): normalize(values)
            for key, values in by_depression.items()},
        "gan_checkpoint": str(args.gan_checkpoint.resolve()),
        "geometry_validator_checkpoint":
            str(args.geometry_validator_checkpoint.resolve()),
        "acceptance_targets": {
            "noise_seed_correlation": "<0.30",
            "noise_leakage_accuracy": "<0.075 (40-class chance=0.025)",
            "generated_geometry": "at least 90% of real-validator performance",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

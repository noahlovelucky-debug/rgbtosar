"""Independent geometry audit for continuous-spatial V1 loss ablations.

The geometry validator is trained on real SAR only and is never used by the
GAN trainer.  This keeps its identity, depression, and azimuth measurements
separate from the loss terms being compared.
"""
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

from joint_data import JointROIDataset
from joint_models import RGBIdentityEncoder, SpatialROIGenerator, _align_translation
from sar_geometry_validator import (
    DEPRESSION_VALUES,
    SARGeometryValidator,
    circular_degree_error,
)
from saratrx import SOC40_CLASSES
from train_continuous_spatial_v1_ablation import build_balanced_proxy, records_from_keys


SUPPORTED_ARCHITECTURES = {
    "continuous_spatial_v1",
    "continuous_spatial_v1_ablation",
    "continuous_spatial_fused_v2",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a continuous-spatial checkpoint with a frozen real-SAR geometry validator")
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-validator-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-view-mode", choices=("nearest", "random"), default="nearest")
    parser.add_argument("--observed", action=argparse.BooleanOptionalAction, default=True,
                        help="score the rendered speckled image; --no-observed scores clean G output")
    parser.add_argument("--seed", type=int, default=9871)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--proxy-manifest", type=Path,
                        help="fixed balanced proxy used when --max-batches is non-zero")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def target_condition(meta: torch.Tensor, source_angle: torch.Tensor) -> torch.Tensor:
    """The V1 condition without leaking the target SAR bounding-box extent."""
    meta = meta.clone()
    meta[:, -2:] = 0.0
    radians = source_angle.float() * (math.pi / 180.0)
    return torch.cat((meta, radians.sin()[:, None], radians.cos()[:, None]), dim=1)


def rotate_target_azimuth(condition: torch.Tensor, degrees: float) -> torch.Tensor:
    radians = torch.atan2(condition[:, 0], condition[:, 1]) + math.radians(degrees)
    result = condition.clone()
    result[:, 0], result[:, 1] = radians.sin(), radians.cos()
    return result


def rotate_vector(vector: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate a [sin(theta), cos(theta)] vector by ``degrees``."""
    radians = math.radians(degrees)
    sine, cosine = vector[:, 0], vector[:, 1]
    return torch.stack((sine * math.cos(radians) + cosine * math.sin(radians),
                        cosine * math.cos(radians) - sine * math.sin(radians)), dim=1)


def depression_ids(depression: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(depression, dtype=torch.long)
    for index, value in enumerate(DEPRESSION_VALUES):
        result[depression == value] = index
    return result


def target_vectors(azimuth: torch.Tensor) -> torch.Tensor:
    radians = azimuth.float() * (math.pi / 180.0)
    return torch.stack((radians.sin(), radians.cos()), dim=1)


def add_metrics(destination: defaultdict[str, float], values: dict[str, torch.Tensor]) -> None:
    for name, value in values.items():
        destination[name] += float(value.detach().sum())


def normalise(values: defaultdict[str, float]) -> dict[str, float | int]:
    samples = int(values.get("samples", 0))
    if samples == 0:
        return {"samples": 0}
    return {
        name: samples if name == "samples" else value / samples
        for name, value in values.items()
    }


def main() -> None:
    args = arguments()
    device = torch.device(args.device)
    use_amp = device.type == "cuda"

    state = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(f"unsupported GAN architecture: {state.get('architecture')!r}")
    if state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("GAN class order differs from SOC40 data")
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

    dataset = JointROIDataset(
        args.rgb_root, args.sar_root, epoch_size=0, band="X", polarization="HH",
        depression="all", augment_rgb=False, source_view_mode=args.source_view_mode)
    proxy_manifest = None
    if args.max_batches:
        proxy_manifest = args.proxy_manifest or args.output.with_name(
            args.output.stem + "_proxy.json")
        proxy_keys = build_balanced_proxy(
            dataset.records, args.sar_root, args.max_batches * args.batch_size,
            args.seed, proxy_manifest)
        dataset.records = records_from_keys(dataset.records, args.sar_root, proxy_keys)
        dataset.epoch_size = len(dataset.records)
        dataset.random_epoch = False
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    totals: defaultdict[str, float] = defaultdict(float)
    by_depression: dict[int, defaultdict[str, float]] = {
        value: defaultdict(float) for value in DEPRESSION_VALUES}

    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="continuous spatial independent audit")):
            rgb = batch["rgb"].to(device, non_blocking=True)
            real = batch["roi"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            labels = batch["class_id"].to(device, non_blocking=True)
            azimuth = batch["azimuth"].to(device, non_blocking=True)
            depression = batch["depression"].to(device, non_blocking=True)
            condition = target_condition(meta, batch["rgb_angle"].to(device, non_blocking=True))
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                identity, _, pyramid = encoder(rgb, return_pyramid=True)
                clean = generator(identity, condition, pyramid, apply_speckle=False)
                if args.observed:
                    # Fix the rendering noise per batch so repeated audits compare exactly.
                    devices = [device.index or 0] if device.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(args.seed + batch_index)
                        fake = generator.apply_speckle(clean)
                else:
                    fake = clean
                offset = generator(
                    identity, rotate_target_azimuth(condition, 30.0), pyramid, apply_speckle=False)
                fake_output = validator((fake + 1.0) * 0.5)
                real_output = validator((real + 1.0) * 0.5)
                offset_output = validator((offset + 1.0) * 0.5)

            target = target_vectors(azimuth)
            aligned_real = _align_translation(clean, real)
            lowpass_error = (F.avg_pool2d(clean, 4) - F.avg_pool2d(aligned_real, 4)).abs().mean((1, 2, 3))
            response_l1 = (F.avg_pool2d(clean, 4) - F.avg_pool2d(offset, 4)).abs().mean((1, 2, 3))
            metric_values = {
                "samples": torch.ones_like(azimuth, dtype=torch.float),
                "generated_identity_correct": (fake_output.identity_logits.argmax(1) == labels).float(),
                "real_identity_correct": (real_output.identity_logits.argmax(1) == labels).float(),
                "generated_depression_correct": (
                    fake_output.depression_logits.argmax(1) == depression_ids(depression)).float(),
                "real_depression_correct": (
                    real_output.depression_logits.argmax(1) == depression_ids(depression)).float(),
                "generated_azimuth_degree_error": circular_degree_error(fake_output.azimuth_vector, target),
                "real_azimuth_degree_error": circular_degree_error(real_output.azimuth_vector, target),
                "generated_azimuth_cosine": (fake_output.azimuth_vector * target).sum(1),
                "real_azimuth_cosine": (real_output.azimuth_vector * target).sum(1),
                "pair_30_degree_error": circular_degree_error(
                    rotate_vector(fake_output.azimuth_vector, 30.0), offset_output.azimuth_vector),
                "geometry_feature_cosine_to_real": F.cosine_similarity(
                    fake_output.features, real_output.features, dim=1),
                "aligned_lowpass_l1": lowpass_error,
                "condition_response_lowpass_l1_30": response_l1,
            }
            add_metrics(totals, metric_values)
            for value in DEPRESSION_VALUES:
                mask = depression == value
                if mask.any():
                    add_metrics(by_depression[value], {
                        name: values[mask] for name, values in metric_values.items()})

    report = {
        "architecture": state["architecture"],
        "checkpoint_epoch": state.get("epoch"),
        "condition": "X/HH; target azimuth/depression; source RGB view is nearest",
        "rendered_observation": args.observed,
        "proxy_manifest": str(proxy_manifest.resolve()) if proxy_manifest else None,
        "all": normalise(totals),
        "by_depression": {str(value): normalise(metrics) for value, metrics in by_depression.items()},
        "gan_checkpoint": str(args.gan_checkpoint.resolve()),
        "geometry_validator_checkpoint": str(args.geometry_validator_checkpoint.resolve()),
        "metric_interpretation": {
            "identity/depression": "higher is better; validator is frozen and never receives GAN gradients",
            "azimuth_degree_error": "lower is better; real-SAR reference is reported beside generated SAR",
            "pair_30_degree_error": "lower is better; checks whether a +30 degree target changes output by +30 degrees",
            "geometry_feature_cosine_to_real": "higher is better for the paired real-SAR condition",
            "aligned_lowpass_l1": "lower is better; diagnostic only, using V1 small-translation alignment",
            "condition_response_lowpass_l1_30": "near zero signals azimuth collapse; compare with pair_30_degree_error",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

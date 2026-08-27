"""Benchmark cached-feature inference against Continuous Spatial V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from continuous_spatial_one_stage_v3 import (
    ARCHITECTURE, ContinuousSpatialOneStageV3,
    MultiViewEncoding, target_geometry)
from joint_models import RGBIdentityEncoder, SpatialROIGenerator
from render_continuous_spatial_one_stage_v3 import load_views
from saratrx import SOC40_CLASSES
from train_continuous_spatial_roi_gan import target_condition


def repeat_encoding(
        encoding: MultiViewEncoding, count: int) -> MultiViewEncoding:
    return MultiViewEncoding(
        encoding.identity.repeat(count, 1),
        encoding.logits.repeat(count, 1),
        tuple(feature.repeat(count, 1, 1, 1, 1)
              for feature in encoding.pyramids),
        encoding.per_view_identity.repeat(count, 1, 1),
        tuple(feature.repeat(count, 1, 1)
              for feature in encoding.pooled_pyramids),
        encoding.canonical_complete)


def elapsed(device: torch.device, function,
            warmup: int, repeats: int, trials: int = 5) -> float:
    for _ in range(warmup):
        function()
    values = []
    for _ in range(trials):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        for _ in range(repeats):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append(
            (perf_counter() - start) * 1000.0 / repeats)
    return median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--class-name", default="Buick_GL8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--device",
        default="cuda:2" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    v3_state = torch.load(
        args.v3_checkpoint, map_location=device, weights_only=False)
    v1_state = torch.load(
        args.v1_checkpoint, map_location=device, weights_only=False)
    if v3_state.get("architecture") != ARCHITECTURE:
        raise RuntimeError("incompatible V3 checkpoint")
    if v1_state.get("architecture") != "continuous_spatial_v1":
        raise RuntimeError("incompatible V1 checkpoint")
    v3 = ContinuousSpatialOneStageV3(
        len(SOC40_CLASSES)).to(device).eval()
    v3.encoder.load_state_dict(
        v3_state.get("ema_encoder", v3_state["encoder"]))
    v3.generator.load_state_dict(
        v3_state.get("ema_generator", v3_state["generator"]))
    v1_encoder = RGBIdentityEncoder(
        len(SOC40_CLASSES)).to(device).eval()
    v1_generator = SpatialROIGenerator(
        meta_dim=12).to(device).eval()
    v1_encoder.load_state_dict(v1_state["identity_encoder"])
    v1_generator.load_state_dict(v1_state["generator"])

    views, angles, mask = load_views(
        args.rgb_root, args.class_name)
    views, angles, mask = (
        views.to(device), angles.to(device), mask.to(device))
    metadata = torch.zeros(1, 10, device=device)
    metadata[:, 3] = 1.0
    metadata[:, 4] = 1.0
    with torch.inference_mode():
        v3_encoding = v3.encode(views, mask)
        v1_identity, _, v1_pyramid = v1_encoder(
            views[:, 0], return_pyramid=True)
        azimuth = torch.tensor((30.0,), device=device)
        depression = torch.tensor((30.0,), device=device)
        geometry = target_geometry(
            metadata, azimuth, depression)
        v1_geometry = target_condition(
            metadata, torch.zeros(1, device=device))
        field = torch.randn(1, 3, 64, 64, device=device)

        def v3_single():
            return v3.generator(
                v3_encoding, angles, mask, azimuth,
                depression, geometry, field).sar

        def v1_single():
            return v1_generator(
                v1_identity, v1_geometry, v1_pyramid,
                apply_speckle=True)

        v3_single_ms = elapsed(
            device, v3_single, args.warmup, args.repeats)
        v1_single_ms = elapsed(
            device, v1_single, args.warmup, args.repeats)

        count = 72
        target_azimuth = torch.arange(
            count, device=device).float() * 5.0
        target_depression = torch.full(
            (count,), 30.0, device=device)
        repeated_meta = metadata.repeat(count, 1)
        repeated_geometry = target_geometry(
            repeated_meta, target_azimuth, target_depression)
        repeated_v3 = repeat_encoding(v3_encoding, count)
        repeated_angles = angles.repeat(count, 1)
        repeated_mask = mask.repeat(count, 1)
        repeated_field = torch.randn(
            count, 3, 64, 64, device=device)
        repeated_identity = v1_identity.repeat(count, 1)
        repeated_pyramid = tuple(
            feature.repeat(count, 1, 1, 1)
            for feature in v1_pyramid)
        repeated_v1_geometry = target_condition(
            repeated_meta, target_azimuth)

        def v3_batch():
            return v3.generator(
                repeated_v3, repeated_angles, repeated_mask,
                target_azimuth, target_depression,
                repeated_geometry, repeated_field).sar

        def v1_batch():
            return v1_generator(
                repeated_identity, repeated_v1_geometry,
                repeated_pyramid, apply_speckle=True)

        batch_repeats = max(10, args.repeats // 10)
        v3_batch_ms = elapsed(
            device, v3_batch, 5, batch_repeats)
        v1_batch_ms = elapsed(
            device, v1_batch, 5, batch_repeats)

    report = {
        "device": str(device),
        "features_cached": True,
        "single_angle_ms": {
            "v1": v1_single_ms, "v3": v3_single_ms,
            "ratio": v3_single_ms / max(v1_single_ms, 1e-9)},
        "batch_72_angles_ms": {
            "v1": v1_batch_ms, "v3": v3_batch_ms,
            "ratio": v3_batch_ms / max(v1_batch_ms, 1e-9)},
        "acceptance": {
            "single_ratio_le_1_5": (
                v3_single_ms <= 1.5 * v1_single_ms),
            "batch_ratio_le_1_5": (
                v3_batch_ms <= 1.5 * v1_batch_ms),
        }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

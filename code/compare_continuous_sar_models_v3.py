"""Create a fair real/V1/Wavelet/Dual-V2/V3 visual comparison grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from continuous_spatial_one_stage_v3 import (
    ARCHITECTURE, ContinuousSpatialOneStageV3,
    target_geometry as target_geometry_v3)
from dual_component_sar_gan import LargeRGBIdentityEncoder
from dual_component_sar_gan_v2 import (
    MultiViewDenoisedSARGenerator, MultiViewRGBEncoder,
    StochasticSARObservation, target_geometry as target_geometry_v2)
from joint_data import JointROIDataset
from joint_models import RGBIdentityEncoder, SpatialROIGenerator
from one_stage_wavelet_sar_gan import OneStageWaveletSARGenerator
from saratrx import SOC40_CLASSES
from train_continuous_spatial_roi_gan import target_condition


def grayscale(image: torch.Tensor) -> Image.Image:
    array = (
        (image.detach().cpu().clamp(-1, 1).numpy() + 1.0) * 127.5
    ).astype(np.uint8)
    return Image.fromarray(array, "L").convert("RGB")


def rgb_panel(image: torch.Tensor) -> Image.Image:
    array = (
        (image.detach().cpu().clamp(-1, 1)
         .permute(1, 2, 0).numpy() + 1.0) * 127.5
    ).astype(np.uint8)
    return Image.fromarray(array, "RGB").resize((64, 64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--wavelet-checkpoint", type=Path, required=True)
    parser.add_argument("--dual-v2-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument(
        "--device",
        default="cuda:2" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    v1_state = torch.load(
        args.v1_checkpoint, map_location=device, weights_only=False)
    wavelet_state = torch.load(
        args.wavelet_checkpoint, map_location=device, weights_only=False)
    dual_state = torch.load(
        args.dual_v2_checkpoint, map_location=device, weights_only=False)
    v3_state = torch.load(
        args.v3_checkpoint, map_location=device, weights_only=False)
    if v1_state.get("architecture") != "continuous_spatial_v1":
        raise RuntimeError("invalid V1 checkpoint")
    if wavelet_state.get("architecture") != "one_stage_aliasfree_wavelet_sar_v1":
        raise RuntimeError("invalid Wavelet checkpoint")
    if dual_state.get("architecture") != "dual_component_multiview_stochastic_v2":
        raise RuntimeError("invalid Dual V2 checkpoint")
    if v3_state.get("architecture") != ARCHITECTURE:
        raise RuntimeError("invalid V3 checkpoint")

    v1_encoder = RGBIdentityEncoder(40).to(device)
    v1_generator = SpatialROIGenerator(meta_dim=12).to(device)
    v1_encoder.load_state_dict(v1_state["identity_encoder"])
    v1_generator.load_state_dict(v1_state["generator"])

    wavelet_encoder = LargeRGBIdentityEncoder(40).to(device)
    wavelet_generator = OneStageWaveletSARGenerator().to(device)
    wavelet_encoder.load_state_dict(
        wavelet_state["ema_identity_encoder"])
    wavelet_generator.load_state_dict(
        wavelet_state["ema_generator"])

    dual_encoder = MultiViewRGBEncoder(40).to(device)
    dual_clean = MultiViewDenoisedSARGenerator().to(device)
    dual_observation = StochasticSARObservation().to(device)
    dual_encoder.load_state_dict(dual_state["ema_encoder"])
    dual_clean.load_state_dict(dual_state["ema_clean_generator"])
    dual_observation.load_state_dict(dual_state["ema_observation"])

    v3 = ContinuousSpatialOneStageV3(40).to(device)
    v3.encoder.load_state_dict(
        v3_state.get("ema_encoder", v3_state["encoder"]))
    v3.generator.load_state_dict(
        v3_state.get("ema_generator", v3_state["generator"]))
    modules = (
        v1_encoder, v1_generator, wavelet_encoder, wavelet_generator,
        dual_encoder, dual_clean, dual_observation, v3)
    for module in modules:
        module.eval()

    dataset = JointROIDataset(
        args.rgb_root, args.sar_root,
        band="X", polarization="HH", depression="all",
        augment_rgb=False, source_view_mode="nearest",
        return_all_views=True)
    # Deterministic evenly-spaced records avoid a first-class-only grid.
    indices = torch.linspace(
        0, len(dataset) - 1,
        steps=min(args.samples, len(dataset))).round().long().tolist()
    items = [dataset[index] for index in indices]
    loader = DataLoader(items, batch_size=len(items), shuffle=False)
    batch = next(iter(loader))
    rgb = batch["rgb"].to(device)
    views = batch["rgb_views"].to(device)
    view_angles = batch["rgb_view_angles"].to(device)
    view_mask = batch["rgb_view_mask"].to(device)
    metadata = batch["meta"].to(device)
    source_angle = batch["rgb_angle"].to(device)
    azimuth = batch["azimuth"].to(device).float()
    depression = batch["depression"].to(device).float()
    real = batch["roi"].to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    with torch.inference_mode():
        v1_identity, _, v1_pyramid = v1_encoder(
            rgb, return_pyramid=True)
        v1_output = v1_generator(
            v1_identity, target_condition(metadata, source_angle),
            v1_pyramid, apply_speckle=True)

        wavelet_identity, _, wavelet_pyramid = wavelet_encoder(
            rgb, return_pyramid=True)
        wavelet_noise = torch.randn(
            len(real), 1, 64, 64,
            device=device, generator=generator)
        _, _, wavelet_output, _ = wavelet_generator(
            wavelet_identity,
            target_condition(metadata, source_angle),
            wavelet_pyramid, wavelet_noise)

        dual_encoding = dual_encoder(views, view_mask)
        dual_geometry = target_geometry_v2(
            azimuth, depression, metadata[:, 3:8])
        dual_base, _ = dual_clean(
            dual_encoding, view_angles, view_mask,
            azimuth, depression, dual_geometry)
        dual_field = torch.randn(
            len(real), 2, 64, 64,
            device=device, generator=generator)
        dual_output = dual_observation(
            dual_base, depression, dual_field).observed

        v3_encoding = v3.encode(views, view_mask)
        v3_field = torch.randn(
            len(real), 3, 64, 64,
            device=device, generator=generator)
        v3_output = v3.generator(
            v3_encoding, view_angles, view_mask,
            azimuth, depression,
            target_geometry_v3(metadata, azimuth, depression),
            v3_field).sar

    headers = ("RGB", "Real", "V1", "Wavelet", "Dual V2", "V3")
    cell, label_width, header = 64, 125, 22
    canvas = Image.new(
        "RGB", (label_width + len(headers) * cell,
                header + len(items) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(headers):
        draw.text(
            (label_width + column * cell + 2, 3),
            name, fill="black")
    for row, item in enumerate(items):
        y = header + row * cell
        label = (
            f"{item['class_name']} "
            f"a{item['azimuth']} d{item['depression']}")
        draw.text((2, y + 22), label[:20], fill="black")
        panels = (
            rgb_panel(rgb[row]), grayscale(real[row, 0]),
            grayscale(v1_output[row, 0]),
            grayscale(wavelet_output[row, 0]),
            grayscale(dual_output[row, 0]),
            grayscale(v3_output[row, 0]))
        for column, image in enumerate(panels):
            canvas.paste(image, (label_width + column * cell, y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

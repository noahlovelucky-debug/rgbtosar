"""Render continuous azimuth/depression grids from the one-stage GAN."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from bbox_data import image_tensor
from dual_component_sar_gan import LargeRGBIdentityEncoder
from joint_data import source_rgb_angle
from one_stage_wavelet_sar_gan import OneStageWaveletSARGenerator
from rgb2sar.data import rgba_to_rgb
from saratrx import SOC40_CLASSES
from train_one_stage_wavelet_sar_gan import ARCHITECTURE


def find_rgb(root: Path, class_name: str, angle: int) -> Path:
    images = [
        path for path in (root / class_name).glob("*.png")
        if path.stem.isdigit()]
    degree_names = any(path.stem == "0" for path in images)
    for path in images:
        if source_rgb_angle(path, degree_names) == angle:
            return path
    raise ValueError(f"missing RGB source angle {angle} for {class_name}")


def condition(azimuth: float, depression: int, source: int,
              device: torch.device) -> torch.Tensor:
    azimuth, source = math.radians(azimuth), math.radians(source)
    values = (
        math.sin(azimuth), math.cos(azimuth), depression / 60,
        1., 1., 0., 0., 0., 0., 0.,
        math.sin(source), math.cos(source))
    return torch.tensor(values, device=device, dtype=torch.float32)[None]


def panel(image: torch.Tensor) -> Image.Image:
    data = (((image.detach().cpu().clamp(-1, 1).numpy()) + 1)
            * 127.5).astype(np.uint8)
    return Image.fromarray(data, "L").convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--class-name", choices=SOC40_CLASSES, required=True)
    parser.add_argument("--source-angle", type=int, default=0)
    parser.add_argument(
        "--target-angles",
        default="0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345")
    parser.add_argument("--noise-seed", type=int, default=2718)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda:2" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(
        args.gan_checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") != ARCHITECTURE:
        raise RuntimeError("one-stage checkpoint architecture mismatch")
    encoder = LargeRGBIdentityEncoder(40).to(device)
    generator = OneStageWaveletSARGenerator().to(device)
    encoder.load_state_dict(state["ema_identity_encoder"])
    generator.load_state_dict(state["ema_generator"])
    encoder.eval(); generator.eval()
    rgb_path = find_rgb(
        args.rgb_root, args.class_name, args.source_angle)
    with Image.open(rgb_path) as source:
        rgb = image_tensor(rgba_to_rgb(source), 128, True)[None].to(device)
    angles = [
        float(item) % 360 for item in args.target_angles.split(",")
        if item.strip()]
    depressions = (15, 30, 45, 60)
    outputs = {}
    with torch.inference_mode():
        identity, _, pyramid = encoder(rgb, return_pyramid=True)
        for row, depression in enumerate(depressions):
            for column, angle in enumerate(angles):
                geometry = condition(
                    angle, depression, args.source_angle, device)
                random_generator = torch.Generator(device=device)
                random_generator.manual_seed(
                    args.noise_seed + row * 1000 + column)
                spatial_noise = torch.randn(
                    1, 1, 64, 64, device=device,
                    generator=random_generator)
                clean, _, observed, _ = generator(
                    identity, geometry, pyramid, spatial_noise)
                outputs[depression, angle] = (
                    clean if args.clean else observed)[0, 0]

    cell, header = 64, 18
    canvas = Image.new(
        "RGB", ((len(angles) + 1) * cell,
                (len(depressions) + 1) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    rgb_preview = (((rgb[0].detach().cpu().clamp(-1, 1)
                     .permute(1, 2, 0).numpy()) + 1) * 127.5).astype(np.uint8)
    canvas.paste(
        Image.fromarray(rgb_preview, "RGB").resize((cell, cell)),
        (0, header))
    draw.text((1, 1), f"RGB {args.source_angle} deg", fill="black")
    for column, angle in enumerate(angles, 1):
        draw.text((column * cell + 2, 1), f"{angle:g}", fill="black")
    for row, depression in enumerate(depressions, 1):
        y = row * (cell + header)
        draw.text((1, y + 1), f"{depression} deg", fill="black")
        for column, angle in enumerate(angles, 1):
            canvas.paste(
                panel(outputs[depression, angle]),
                (column * cell, y + header))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"saved {args.output} from {rgb_path}")


if __name__ == "__main__":
    main()

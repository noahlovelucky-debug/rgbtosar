"""Render the dual-component GAN over continuous azimuth/depression grids."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from bbox_data import image_tensor
from dual_component_sar_gan import (
    DenoisedSARGenerator, LargeRGBIdentityEncoder, SARNoiseGenerator,
    compose_sar, noise_view)
from joint_data import source_rgb_angle
from rgb2sar.data import rgba_to_rgb
from saratrx import SOC40_CLASSES
from train_dual_component_sar_gan import ARCHITECTURE


def parse_angles(value: str) -> list[float]:
    angles = [float(item.strip()) % 360
              for item in value.split(",") if item.strip()]
    if not angles:
        raise ValueError("--target-angles is empty")
    return angles


def find_rgb(root: Path, class_name: str, angle: int) -> Path:
    numeric = [
        path for path in (root / class_name).glob("*.png")
        if path.stem.isdigit()]
    degree_names = any(path.stem == "0" for path in numeric)
    for path in numeric:
        if source_rgb_angle(path, degree_names) == angle:
            return path
    available = sorted(
        source_rgb_angle(path, degree_names) for path in numeric)
    raise ValueError(f"{class_name}: source angle {angle} missing; {available=}")


def condition(azimuth: float, depression: int, source_angle: int,
              device: torch.device) -> torch.Tensor:
    azimuth = math.radians(azimuth)
    source = math.radians(source_angle)
    # metadata_vector: target azimuth, depression, X band, HH polarization,
    # no target bbox leakage, followed by RGB source-view sin/cos.
    values = (
        math.sin(azimuth), math.cos(azimuth), depression / 60,
        1., 1., 0., 0., 0., 0., 0.,
        math.sin(source), math.cos(source))
    return torch.tensor(values, dtype=torch.float32, device=device)[None]


def panel(image: torch.Tensor) -> Image.Image:
    data = (((image.detach().cpu().clamp(-1, 1).numpy()) + 1)
            * 127.5).astype(np.uint8)
    return Image.fromarray(data, "L").convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="render continuous dual-component RGB-to-SAR grid")
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--class-name", choices=SOC40_CLASSES, required=True)
    parser.add_argument("--source-angle", type=int, default=0)
    parser.add_argument(
        "--target-angles",
        default="0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345")
    parser.add_argument("--noise-seed", type=int, default=2718)
    parser.add_argument(
        "--noise-mode", choices=("independent", "fixed"),
        default="independent",
        help="fixed reuses one latent for every angle/depression; independent preserves the original renderer")
    parser.add_argument(
        "--component", choices=("full", "clean", "noise"), default="full",
        help="render composed SAR, denoised SAR, or normalized log-noise")
    parser.add_argument("--clean", action="store_true",
                        help="deprecated alias for --component clean")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(
        args.gan_checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") != ARCHITECTURE:
        raise RuntimeError(f"expected {ARCHITECTURE}, got {state.get('architecture')}")
    encoder = LargeRGBIdentityEncoder(40).to(device)
    clean_generator = DenoisedSARGenerator().to(device)
    noise_generator = SARNoiseGenerator().to(device)
    encoder.load_state_dict(state["ema_identity_encoder"])
    clean_generator.load_state_dict(state["ema_clean_generator"])
    noise_generator.load_state_dict(state["ema_noise_generator"])
    encoder.eval(); clean_generator.eval(); noise_generator.eval()

    path = find_rgb(args.rgb_root, args.class_name, args.source_angle)
    with Image.open(path) as image:
        rgb = image_tensor(rgba_to_rgb(image), 128, True)[None].to(device)
    angles = parse_angles(args.target_angles)
    depressions = (15, 30, 45, 60)
    outputs = {}
    with torch.inference_mode():
        identity, _, pyramid = encoder(rgb, return_pyramid=True)
        fixed_generator = torch.Generator(device=device)
        fixed_generator.manual_seed(args.noise_seed)
        fixed_latent = torch.randn(
            1, noise_generator.noise_dim, device=device,
            generator=fixed_generator)
        for row, depression in enumerate(depressions):
            for column, angle in enumerate(angles):
                geometry = condition(
                    angle, depression, args.source_angle, device)
                clean = clean_generator(identity, geometry, pyramid)
                if args.noise_mode == "fixed":
                    latent = fixed_latent
                else:
                    random_generator = torch.Generator(device=device)
                    random_generator.manual_seed(
                        args.noise_seed + row * 1000 + column)
                    latent = torch.randn(
                        1, noise_generator.noise_dim, device=device,
                        generator=random_generator)
                log_noise = noise_generator(
                    clean, geometry, pyramid, latent)
                component = "clean" if args.clean else args.component
                rendered = {
                    "clean": clean,
                    "noise": noise_view(log_noise),
                    "full": compose_sar(clean, log_noise),
                }[component]
                outputs[depression, angle] = rendered[0, 0]

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
    component = "clean" if args.clean else args.component
    draw.text((1, 1), f"{component}/{args.noise_mode}", fill="black")
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
    print(f"saved {args.output} from {path}; component={component}; noise={args.noise_mode}")


if __name__ == "__main__":
    main()

"""Render continuous azimuth/depression grids from the multi-view SAR GAN v2."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from bbox_data import image_tensor
from dual_component_sar_gan_v2 import (
    LOG_NOISE_LIMIT, MultiViewDenoisedSARGenerator, MultiViewRGBEncoder,
    StochasticSARObservation, residual_view, target_geometry)
from joint_data import source_rgb_angle
from rgb2sar.data import rgba_to_rgb
from saratrx import SOC40_CLASSES
from train_dual_component_sar_gan_v2 import ARCHITECTURE


def parse_angles(value: str) -> list[float]:
    result = [
        float(item.strip()) % 360 for item in value.split(",")
        if item.strip()]
    if not result:
        raise ValueError("--target-angles is empty")
    return result


def load_views(root: Path, class_name: str,
               device: torch.device) -> tuple[
                   torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray]]:
    folder = root / class_name
    numeric = [
        path for path in folder.glob("*.png") if path.stem.isdigit()]
    if not numeric:
        raise RuntimeError(f"no RGB views under {folder}")
    degree_names = any(path.stem == "0" for path in numeric)
    paths = {
        source_rgb_angle(path, degree_names): path for path in numeric}
    views, mask, previews = [], [], []
    for angle in range(0, 360, 30):
        path = paths.get(angle)
        if path is None:
            views.append(torch.zeros(3, 128, 128))
            mask.append(0.0)
            previews.append(np.zeros((128, 128, 3), dtype=np.uint8))
        else:
            with Image.open(path) as image:
                rgb = image_tensor(rgba_to_rgb(image), 128, True)
            views.append(rgb)
            mask.append(1.0)
            previews.append(
                (((rgb.permute(1, 2, 0).numpy().clip(-1, 1)) + 1)
                 * 127.5).astype(np.uint8))
    return (
        torch.stack(views)[None].to(device),
        torch.arange(0, 360, 30, dtype=torch.float32, device=device)[None],
        torch.tensor(mask, dtype=torch.float32, device=device)[None],
        previews)


def panel(image: torch.Tensor) -> Image.Image:
    data = (((image.detach().cpu().clamp(-1, 1).numpy()) + 1)
            * 127.5).astype(np.uint8)
    return Image.fromarray(data, "L").convert("RGB")


def view_contact_sheet(previews: list[np.ndarray], size: int = 64) -> Image.Image:
    sheet = Image.new("RGB", (size, size), "#888888")
    width, height = size // 4, size // 3
    for index, preview in enumerate(previews):
        image = Image.fromarray(preview, "RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        x = (index % 4) * width + (width - image.width) // 2
        y = (index // 4) * height + (height - image.height) // 2
        sheet.paste(image, (x, y))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--class-name", choices=SOC40_CLASSES, required=True)
    parser.add_argument(
        "--target-angles",
        default="0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345")
    parser.add_argument(
        "--component", choices=("full", "clean", "noise"), default="full")
    parser.add_argument(
        "--noise-mode", choices=("fixed", "independent"), default="fixed")
    parser.add_argument("--noise-seed", type=int, default=2718)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    saved = torch.load(
        args.gan_checkpoint, map_location=device, weights_only=False)
    if saved.get("architecture") != ARCHITECTURE:
        raise RuntimeError(
            f"expected {ARCHITECTURE}, got {saved.get('architecture')}")
    encoder = MultiViewRGBEncoder(len(SOC40_CLASSES)).to(device)
    clean_generator = MultiViewDenoisedSARGenerator().to(device)
    observation = StochasticSARObservation().to(device)
    encoder.load_state_dict(saved["ema_encoder"])
    clean_generator.load_state_dict(saved["ema_clean_generator"])
    observation.load_state_dict(saved["ema_observation"])
    encoder.eval()
    clean_generator.eval()
    observation.eval()

    views, source_angles, view_mask, previews = load_views(
        args.rgb_root, args.class_name, device)
    angles = parse_angles(args.target_angles)
    depressions = (15, 30, 45, 60)
    acquisition = torch.tensor(
        [[1., 1., 0., 0., 0.]], device=device)
    outputs: dict[tuple[int, float], torch.Tensor] = {}
    attention_rows = []
    fixed_generator = torch.Generator(device=device)
    fixed_generator.manual_seed(args.noise_seed)
    fixed_field = torch.randn(
        1, observation.random_channels, 64, 64,
        device=device, generator=fixed_generator)
    with torch.inference_mode():
        encoding = encoder(views, view_mask)
        for row, depression_value in enumerate(depressions):
            for column, angle_value in enumerate(angles):
                azimuth = torch.tensor([angle_value], device=device)
                depression = torch.tensor(
                    [float(depression_value)], device=device)
                geometry = target_geometry(
                    azimuth, depression, acquisition)
                clean, attention = clean_generator(
                    encoding, source_angles, view_mask,
                    azimuth, depression, geometry)
                if args.noise_mode == "fixed":
                    field = fixed_field
                else:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        args.noise_seed + row * 10000 + column)
                    field = torch.randn(
                        1, observation.random_channels, 64, 64,
                        device=device, generator=generator)
                observed = observation(clean, depression, field)
                rendered = {
                    "clean": clean,
                    "noise": residual_view(clean, observed.observed),
                    "full": observed.observed,
                }[args.component]
                outputs[depression_value, angle_value] = rendered[0, 0]
                attention_rows.append((
                    depression_value, angle_value,
                    *attention[0].detach().cpu().tolist()))

    cell, header = 64, 18
    canvas = Image.new(
        "RGB", ((len(angles) + 1) * cell,
                (len(depressions) + 1) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(view_contact_sheet(previews, cell), (0, header))
    draw.text(
        (1, 1), f"{args.component}/{args.noise_mode}", fill="black")
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
    attention_path = args.output.with_name(
        args.output.stem + "_attention.csv")
    with attention_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "depression", "target_azimuth",
            *[f"rgb_{angle}" for angle in range(0, 360, 30)]))
        writer.writerows(attention_rows)
    print({
        "output": str(args.output),
        "attention": str(attention_path),
        "class": args.class_name,
        "component": args.component,
        "noise_mode": args.noise_mode,
        "noise_limit": LOG_NOISE_LIMIT,
    })


if __name__ == "__main__":
    main()

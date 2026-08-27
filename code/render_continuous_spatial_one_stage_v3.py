"""Render continuous azimuth/depression grids from a V3 checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from bbox_data import image_tensor
from continuous_spatial_one_stage_v3 import (
    ARCHITECTURE, ContinuousSpatialOneStageV3, target_geometry)
from joint_data import source_rgb_angle
from rgb2sar.data import rgba_to_rgb
from saratrx import SOC40_CLASSES


def parse_angles(value: str) -> list[float]:
    angles = [
        float(item.strip()) % 360.0
        for item in value.split(",") if item.strip()]
    if not angles:
        raise ValueError("target angles cannot be empty")
    return angles


def load_views(
        root: Path, class_name: str
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    folder = root / class_name
    numeric = [
        path for path in folder.glob("*.png")
        if path.stem.isdigit()]
    degree_names = any(path.stem == "0" for path in numeric)
    paths = {
        source_rgb_angle(path, degree_names): path
        for path in numeric}
    views, mask = [], []
    for angle in range(0, 360, 30):
        path = paths.get(angle)
        if path is None:
            views.append(torch.zeros(3, 128, 128))
            mask.append(0.0)
        else:
            with Image.open(path) as image:
                views.append(image_tensor(
                    rgba_to_rgb(image), 128, True))
            mask.append(1.0)
    if not any(mask):
        raise RuntimeError(f"{class_name} has no RGB views")
    return (
        torch.stack(views)[None],
        torch.arange(0, 360, 30, dtype=torch.float32)[None],
        torch.tensor(mask, dtype=torch.float32)[None])


def panel(image: torch.Tensor) -> Image.Image:
    array = (
        (image.detach().cpu().clamp(-1, 1).numpy() + 1.0) * 127.5
    ).astype(np.uint8)
    return Image.fromarray(array, "L").convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument(
        "--class-name", choices=SOC40_CLASSES, required=True)
    parser.add_argument(
        "--target-angles",
        default=(
            "0,15,30,45,60,75,90,105,120,135,150,165,"
            "180,195,210,225,240,255,270,285,300,315,330,345"))
    parser.add_argument(
        "--noise-mode", choices=("fixed", "independent"),
        default="fixed")
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda:2" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(
        args.gan_checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") != ARCHITECTURE:
        raise RuntimeError("checkpoint is not Continuous Spatial V3")
    model = ContinuousSpatialOneStageV3(
        len(SOC40_CLASSES)).to(device)
    model.encoder.load_state_dict(
        state.get("ema_encoder", state["encoder"]))
    model.generator.load_state_dict(
        state.get("ema_generator", state["generator"]))
    model.eval()
    views, view_angles, view_mask = load_views(
        args.rgb_root, args.class_name)
    views = views.to(device)
    view_angles = view_angles.to(device)
    view_mask = view_mask.to(device)
    angles = parse_angles(args.target_angles)
    depressions = (15, 30, 45, 60)
    metadata = torch.zeros(1, 10, device=device)
    metadata[:, 3] = 1.0
    metadata[:, 4] = 1.0
    outputs = {}
    with torch.inference_mode():
        encoding = model.encode(views, view_mask)
        fixed_generator = torch.Generator(device=device)
        fixed_generator.manual_seed(args.seed)
        fixed_field = torch.randn(
            1, 3, 64, 64, device=device,
            generator=fixed_generator)
        for row, depression_value in enumerate(depressions):
            for column, angle_value in enumerate(angles):
                azimuth = torch.tensor(
                    (angle_value,), device=device)
                depression = torch.tensor(
                    (float(depression_value),), device=device)
                geometry = target_geometry(
                    metadata, azimuth, depression)
                if args.noise_mode == "fixed":
                    field = fixed_field
                else:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        args.seed + row * 1000 + column)
                    field = torch.randn(
                        1, 3, 64, 64, device=device,
                        generator=generator)
                outputs[depression_value, angle_value] = (
                    model.generator(
                        encoding, view_angles, view_mask,
                        azimuth, depression, geometry, field).sar[0, 0])

    cell, header = 64, 18
    canvas = Image.new(
        "RGB", ((len(angles) + 1) * cell,
                (len(depressions) + 1) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    preview = (
        (views[0, 0].detach().cpu().clamp(-1, 1)
         .permute(1, 2, 0).numpy() + 1.0) * 127.5
    ).astype(np.uint8)
    canvas.paste(
        Image.fromarray(preview, "RGB").resize((cell, cell)),
        (0, header))
    draw.text((1, 1), f"V3/{args.noise_mode}", fill="black")
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
    print(
        f"saved {args.output}; class={args.class_name}; "
        f"noise={args.noise_mode}")


if __name__ == "__main__":
    main()

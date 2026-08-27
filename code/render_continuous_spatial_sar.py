"""Render arbitrary (including unobserved) azimuths for all four depressions."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from bbox_data import image_tensor
from joint_data import source_rgb_angle
from joint_models import RGBIdentityEncoder, SpatialROIGenerator
from rgb2sar.data import rgba_to_rgb
from saratrx import SOC40_CLASSES


def parse_angles(value: str) -> list[float]:
    angles = [float(item.strip()) % 360 for item in value.split(",") if item.strip()]
    if not angles:
        raise ValueError("--target-angles must contain at least one angle")
    return angles


def rgb_path(root: Path, class_name: str, source_angle: int) -> Path:
    folder = root / class_name
    numeric = [path for path in folder.glob("*.png") if path.stem.isdigit()]
    has_degree_names = any(path.stem == "0" for path in numeric)
    matches = [path for path in numeric if source_rgb_angle(path, has_degree_names) == source_angle]
    if not matches:
        available = sorted(source_rgb_angle(path, has_degree_names) for path in numeric)
        raise ValueError(f"{class_name} has RGB source views {available}; requested {source_angle}")
    return matches[0]


def target_condition(azimuth: float, depression: int, source_angle: int, device: torch.device) -> torch.Tensor:
    azimuth_rad, source_rad = math.radians(azimuth), math.radians(source_angle)
    # metadata_vector without annotation-box dimensions: X-band + HH.
    meta = [math.sin(azimuth_rad), math.cos(azimuth_rad), depression / 60.0, 1.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.sin(source_rad), math.cos(source_rad)]
    return torch.tensor(meta, dtype=torch.float32, device=device)[None]


def panel(image: torch.Tensor) -> Image.Image:
    data = ((image.detach().cpu().clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
    return Image.fromarray(data, "L").convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render continuous target azimuths from a single RGB view")
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--class-name", choices=SOC40_CLASSES, required=True)
    parser.add_argument("--source-angle", type=int, default=0, help="available RGB viewpoint, typically 0,30,...330")
    parser.add_argument("--target-angles", default="0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clean", action="store_true", help="disable simulated speckle for structural inspection")
    args = parser.parse_args()
    if not 0 <= args.source_angle < 360:
        raise ValueError("--source-angle must be in [0, 360)")
    device = torch.device(args.device)
    state = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") not in {"continuous_spatial_v1", "continuous_spatial_fused_v2"}:
        raise RuntimeError("checkpoint is not a continuous spatial V1 or fused-v2 model")
    encoder = RGBIdentityEncoder(len(SOC40_CLASSES)).to(device)
    generator = SpatialROIGenerator(meta_dim=12).to(device)
    encoder.load_state_dict(state["identity_encoder"]); generator.load_state_dict(state["generator"])
    encoder.eval(); generator.eval()
    path = rgb_path(args.rgb_root, args.class_name, args.source_angle)
    with Image.open(path) as source:
        rgb = image_tensor(rgba_to_rgb(source), 128, True)[None].to(device)
    angles = parse_angles(args.target_angles)
    depressions = (15, 30, 45, 60)
    with torch.inference_mode():
        identity, _, pyramid = encoder(rgb, return_pyramid=True)
        outputs = {
            (depression, angle): generator(identity, target_condition(angle, depression, args.source_angle, device),
                                           pyramid, apply_speckle=not args.clean)[0, 0]
            for depression in depressions for angle in angles
        }

    cell, header = 64, 18
    canvas = Image.new("RGB", ((len(angles) + 1) * cell, (len(depressions) + 1) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    rgb_preview = ((rgb[0].detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
    canvas.paste(Image.fromarray(rgb_preview, "RGB").resize((cell, cell)), (0, header))
    draw.text((1, 1), f"RGB {args.source_angle}°", fill="black")
    for column, angle in enumerate(angles, start=1):
        draw.text((column * cell + 2, 1), f"{angle:g}°", fill="black")
    for row, depression in enumerate(depressions, start=1):
        y = row * (cell + header)
        draw.text((1, y + 1), f"{depression}°", fill="black")
        for column, angle in enumerate(angles, start=1):
            canvas.paste(panel(outputs[depression, angle]), (column * cell, y + header))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"saved {args.output} from {path}")


if __name__ == "__main__":
    main()

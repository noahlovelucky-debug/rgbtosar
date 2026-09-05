"""Render a target-acquisition sweep from a silhouette bridge checkpoint."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from bbox_data import image_tensor
from joint_data import source_rgb_angle
from rgb2sar.data import rgba_to_rgb
from saratrx import SOC40_CLASSES
from unsb_sar_bridge import (
    UNSB_SAR_BRIDGE_ARCHITECTURE, UNSB_SAR_UNPAIRED_ARCHITECTURE,
    SilhouetteBridge, bridge_sample,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render UNSB-SAR bridge target-angle sweep")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--class-name", choices=SOC40_CLASSES, default="Buick_GL8")
    parser.add_argument("--source-angle", type=int, default=0)
    parser.add_argument("--depression", type=int, choices=(15, 30, 45, 60), default=30)
    parser.add_argument("--band", choices=("X", "KU"), default="X")
    parser.add_argument("--polarization", choices=("HH", "HV", "VH", "VV"), default="HH")
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def find_rgb(root: Path, class_name: str, angle: int) -> Path:
    paths = [path for path in (root / class_name).glob("*.png") if path.stem.isdigit()]
    if not paths:
        raise RuntimeError(f"no RGB views found under {root / class_name}")
    has_degree_names = any(path.stem == "0" for path in paths)
    return min(paths, key=lambda path: min(
        (angle - source_rgb_angle(path, has_degree_names)) % 360,
        (source_rgb_angle(path, has_degree_names) - angle) % 360))


def load_rgb(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        rgb = image_tensor(rgba_to_rgb(rgba), 128, True)[None].to(device)
        mask = (image_tensor(rgba.getchannel("A"), 128, False) + 1.0).mul(.5)[None].to(device)
    return rgb, mask


def acquisition(azimuth: int, depression: int, band: str, polarization: str,
                device: torch.device) -> torch.Tensor:
    result = torch.zeros(12, device=device)
    radians = math.radians(azimuth % 360)
    result[0], result[1] = math.sin(radians), math.cos(radians)
    result[2 + (depression // 15 - 1)] = 1.0
    result[6 if band == "X" else 7] = 1.0
    result[8 + ("HH", "HV", "VH", "VV").index(polarization)] = 1.0
    return result


def angle_features(angle: int, count: int, device: torch.device) -> torch.Tensor:
    radians = torch.full((count,), math.radians(angle % 360), device=device)
    return torch.stack((radians.sin(), radians.cos()), dim=1)


def panel(image: torch.Tensor, channels: int = 3) -> np.ndarray:
    value = image.detach().cpu().clamp(-1, 1).numpy()
    if value.ndim == 3:
        value = value[0] if value.shape[0] == 1 else value.transpose(1, 2, 0)
    if value.ndim == 2:
        value = np.repeat(value[..., None], channels, axis=2)
    return ((value + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)


def main() -> None:
    args = arguments()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") not in {UNSB_SAR_BRIDGE_ARCHITECTURE,
                                          UNSB_SAR_UNPAIRED_ARCHITECTURE}:
        raise RuntimeError("checkpoint is not an UNSB-SAR silhouette bridge checkpoint")
    saved_args = state.get("args", {})
    model = SilhouetteBridge(
        base=int(saved_args.get("base", 64)),
        token_dim=int(saved_args.get("token_dim", 256)),
        control_base=int(saved_args.get("control_base", 32)),
    ).to(device).eval()
    ema_key = "ema_generator" if state.get("architecture") == UNSB_SAR_UNPAIRED_ARCHITECTURE else "ema_model"
    model.load_state_dict(state[ema_key])
    primary_path = find_rgb(args.rgb_root, args.class_name, args.source_angle)
    alternate_path = find_rgb(args.rgb_root, args.class_name, args.source_angle + 30)
    rgb, mask = load_rgb(primary_path, device)
    rgb_alt, mask_alt = load_rgb(alternate_path, device)
    azimuths = list(range(0, 360, 30))
    rgb_batch = rgb.expand(len(azimuths), -1, -1, -1)
    mask_batch = mask.expand(len(azimuths), -1, -1, -1)
    rgb_alt_batch = rgb_alt.expand(len(azimuths), -1, -1, -1)
    mask_alt_batch = mask_alt.expand(len(azimuths), -1, -1, -1)
    conditions = torch.stack([acquisition(angle, args.depression, args.band,
                                           args.polarization, device)
                              for angle in azimuths])
    source_angles = angle_features(args.source_angle, len(azimuths), device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    fake = bridge_sample(model, rgb_batch, mask_batch, conditions,
                         steps=args.sample_steps, temperature=args.sample_temperature,
                         generator=generator, rgb_alt=rgb_alt_batch,
                         mask_alt=mask_alt_batch, source_angle=source_angles)
    rgb_panel = F.interpolate(rgb, (64, 64), mode="bilinear", align_corners=False)[0]
    panels = [panel(rgb_panel), panel(F.interpolate(mask, (64, 64), mode="bilinear", align_corners=False)[0])]
    panels.extend(panel(sample) for sample in fake)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=1), "RGB").save(args.output)
    print({"checkpoint": str(args.checkpoint), "source_rgb": str(primary_path),
           "alternate_rgb": str(alternate_path), "class": args.class_name,
           "azimuths": azimuths, "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()

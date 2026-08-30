"""Render target-condition sweeps from a HiFC-unpaired checkpoint."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F

from bbox_data import image_tensor
from dual_component_sar_gan import LargeRGBIdentityEncoder
from hifc_unpaired_sar_gan import (
    HIFC_ARCHITECTURE, HIFCUnpairedGenerator, condition_from_batch)
from joint_data import source_rgb_angle
from rgb2sar.data import rgba_to_rgb
from saratrx import SOC40_CLASSES


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="render HiFC unpaired target conditions")
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--class-name", default="Buick_GL8", choices=SOC40_CLASSES)
    parser.add_argument("--source-angle", type=int, default=0)
    parser.add_argument("--depression", type=int, default=30, choices=(15, 30, 45, 60))
    parser.add_argument("--band", choices=("X", "KU"), default="X")
    parser.add_argument("--polarization", choices=("HH", "HV", "VH", "VV"), default="HH")
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


def condition(azimuth: int, depression: int, band: str, pol: str) -> torch.Tensor:
    meta = torch.zeros(1, 10)
    radians = math.radians(azimuth % 360)
    meta[0, 0] = math.sin(radians)
    meta[0, 1] = math.cos(radians)
    meta[0, 2] = depression / 60.0
    meta[0, 3] = float(band == "X")
    meta[0, 4 + ("HH", "HV", "VH", "VV").index(pol)] = 1.0
    return condition_from_batch(meta, torch.tensor([depression]))


def main() -> None:
    args = arguments()
    device = torch.device(args.device)
    state = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") != HIFC_ARCHITECTURE:
        raise RuntimeError("checkpoint is not hifc_unpaired_conditioned_v1")
    encoder = LargeRGBIdentityEncoder(len(SOC40_CLASSES)).to(device).eval()
    generator = HIFCUnpairedGenerator().to(device).eval()
    encoder.load_state_dict(state["ema_identity_encoder"])
    generator.load_state_dict(state["ema_generator"])
    path = find_rgb(args.rgb_root, args.class_name, args.source_angle)
    with Image.open(path) as image:
        rgb = image_tensor(rgba_to_rgb(image), 128, True)[None].to(device)
    label = torch.tensor([SOC40_CLASSES.index(args.class_name)], device=device)
    azimuths = list(range(0, 360, 30))
    rows = []
    with torch.inference_mode():
        identity, _, pyramid = encoder(rgb, return_pyramid=True)
        rgb_panel = F.interpolate(rgb, (64, 64), mode="bilinear", align_corners=False)[0]
        rgb_panel = (((rgb_panel.cpu().clamp(-1, 1).permute(1, 2, 0).numpy()) + 1) * 127.5).astype("uint8")
        row = [rgb_panel]
        for azimuth in azimuths:
            geometry = condition(azimuth, args.depression, args.band, args.polarization).to(device)
            noise = torch.zeros(1, 1, 64, 64, device=device)
            _, _, fake, _ = generator(identity, geometry, pyramid, noise)
            panel = (((fake[0, 0].cpu().clamp(-1, 1).numpy()) + 1) * 127.5).astype("uint8")
            row.append(np.repeat(panel[..., None], 3, axis=2))
        rows.append(np.concatenate(row, axis=1))
    array = np.concatenate(rows, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, "RGB").save(args.output)
    print({"checkpoint": str(args.gan_checkpoint), "class": args.class_name,
           "source_view": str(path), "conditions": azimuths,
           "band": args.band, "polarization": args.polarization,
           "depression": args.depression, "output": str(args.output)})


if __name__ == "__main__":
    main()

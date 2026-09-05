"""Render an azimuth sweep from a trained 64px conditional SAR DDPM."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from bbox_data import image_tensor
from conditional_sar_diffusion import (
    CONDITIONAL_DIFFUSION_ARCHITECTURE, LEGACY_CONDITIONAL_DIFFUSION_ARCHITECTURE,
    ConditionalSARDDPM, DiffusionSchedule,
)
from joint_data import source_rgb_angle
from rgb2sar.data import rgba_to_rgb
from saratrx import SOC40_CLASSES


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render conditional DDPM SAR target-angle sweep")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--class-name", choices=SOC40_CLASSES, default="Buick_GL8")
    parser.add_argument("--source-angle", type=int, default=0)
    parser.add_argument("--depression", type=int, choices=(15, 30, 45, 60), default=30)
    parser.add_argument("--band", choices=("X", "KU"), default="X")
    parser.add_argument("--polarization", choices=("HH", "HV", "VH", "VV"), default="HH")
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--independent-noise", action="store_true",
                        help="use a separate initial noise field for each azimuth")
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


def acquisition(azimuth: int, depression: int, band: str, polarization: str,
                device: torch.device) -> torch.Tensor:
    result = torch.zeros(12, device=device)
    radians = math.radians(azimuth % 360)
    result[0], result[1] = math.sin(radians), math.cos(radians)
    result[2 + (depression // 15 - 1)] = 1.0
    result[6 if band == "X" else 7] = 1.0
    result[8 + ("HH", "HV", "VH", "VV").index(polarization)] = 1.0
    return result


def main() -> None:
    args = arguments()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    architecture = state.get("architecture")
    if architecture not in {CONDITIONAL_DIFFUSION_ARCHITECTURE,
                             LEGACY_CONDITIONAL_DIFFUSION_ARCHITECTURE}:
        raise RuntimeError("checkpoint is not a conditional SAR DDPM")
    saved_args = state.get("args", {})
    model = ConditionalSARDDPM(
        base=int(saved_args.get("base", 64)), token_dim=int(saved_args.get("token_dim", 256)),
        rgb_base=int(saved_args.get("rgb_base", 32)),
        class_conditioning=architecture == CONDITIONAL_DIFFUSION_ARCHITECTURE).to(device).eval()
    model.load_state_dict(state["ema_model"])
    schedule = DiffusionSchedule(int(saved_args.get("diffusion_steps", 1_000))).to(device)
    path = find_rgb(args.rgb_root, args.class_name, args.source_angle)
    with Image.open(path) as source:
        rgb = image_tensor(rgba_to_rgb(source), 128, True)[None].to(device)
    azimuths = list(range(0, 360, 30))
    rgb_batch = rgb.expand(len(azimuths), -1, -1, -1)
    conditions = torch.stack([
        acquisition(angle, args.depression, args.band, args.polarization, device)
        for angle in azimuths])
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = None
    if not args.independent_noise:
        # A shared initial field makes a sweep a condition-response diagnostic,
        # rather than a comparison of twelve unrelated random samples.
        initial_noise = torch.randn((1, 1, 64, 64), device=device, dtype=rgb.dtype,
                                    generator=generator).expand(len(azimuths), -1, -1, -1)
    fake = schedule.ddim_sample(model, rgb_batch, conditions, sample_steps=args.sample_steps,
                                guidance_scale=args.guidance_scale, generator=generator,
                                initial_noise=initial_noise,
                                prediction_type="v" if architecture == CONDITIONAL_DIFFUSION_ARCHITECTURE else "epsilon")
    rgb_panel = F.interpolate(rgb, (64, 64), mode="bilinear", align_corners=False)[0]
    panels = [(((rgb_panel.cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8))]
    for sample in fake:
        panel = (((sample[0].cpu().clamp(-1, 1).numpy() + 1.0) * 127.5).astype(np.uint8))
        panels.append(np.repeat(panel[..., None], 3, axis=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=1), "RGB").save(args.output)
    print({"checkpoint": str(args.checkpoint), "source_rgb": str(path), "class": args.class_name,
           "azimuths": azimuths, "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()

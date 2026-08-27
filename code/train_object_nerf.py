"""Sparse-view object NeRF with learnable orbit cameras for one vehicle.

Input is the RGBA directory produced by ``prepare_rgb3d.py``.  Known 30-degree
azimuths initialise a constrained orbit camera; mask centroids initialise each
principal point.  Small azimuth/elevation/radius/intrinsic residuals are then
optimised jointly with a single 3-D radiance field.  Rendering intermediate
cameras therefore queries one shared 3-D representation rather than blending
two images.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def encode(value: torch.Tensor, frequencies: int) -> torch.Tensor:
    outputs = [value]
    for frequency in (2.0 ** torch.arange(frequencies, device=value.device, dtype=value.dtype)):
        outputs.extend((torch.sin(value * frequency * math.pi), torch.cos(value * frequency * math.pi)))
    return torch.cat(outputs, dim=-1)


class RadianceField(nn.Module):
    def __init__(self, width: int = 128, position_frequencies: int = 8,
                 direction_frequencies: int = 4) -> None:
        super().__init__()
        self.width = width
        self.position_frequencies = position_frequencies
        self.direction_frequencies = direction_frequencies
        position_dim = 3 * (1 + 2 * position_frequencies)
        direction_dim = 3 * (1 + 2 * direction_frequencies)
        self.layers = nn.ModuleList()
        for index in range(6):
            input_dim = position_dim if index == 0 else width
            if index == 3: input_dim += position_dim
            self.layers.append(nn.Linear(input_dim, width))
        self.density = nn.Linear(width, 1)
        self.feature = nn.Linear(width, width)
        self.colour = nn.Sequential(nn.Linear(width + direction_dim, width // 2), nn.SiLU(),
                                    nn.Linear(width // 2, 3), nn.Sigmoid())

    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_position = encode(positions, self.position_frequencies)
        hidden = encoded_position
        for index, layer in enumerate(self.layers):
            if index == 3: hidden = torch.cat((hidden, encoded_position), dim=-1)
            hidden = F.silu(layer(hidden))
        sigma = F.softplus(self.density(hidden) - 1.0)
        # Compact support suppresses floaters outside the canonical vehicle box.
        support = torch.sigmoid((1.3 - positions.abs().amax(dim=-1, keepdim=True)) * 20.0)
        sigma = sigma * support
        colour = self.colour(torch.cat((self.feature(hidden),
                                        encode(F.normalize(directions, dim=-1), self.direction_frequencies)), dim=-1))
        return colour, sigma


class OrbitCameras(nn.Module):
    def __init__(self, angles: torch.Tensor, centroids: torch.Tensor, resolution: int,
                 fov_degrees: float = 45.0, elevation_degrees: float = 16.0) -> None:
        super().__init__()
        self.resolution = resolution
        self.register_buffer("base_angles", torch.deg2rad(angles.float()))
        self.register_buffer("base_centroids", centroids.float())
        self.base_elevation = math.radians(elevation_degrees)
        focal = 0.5 * resolution / math.tan(math.radians(fov_degrees) / 2)
        self.register_buffer("base_focal", torch.tensor(focal, dtype=torch.float32))
        count = len(angles)
        self.azimuth_residual = nn.Parameter(torch.zeros(count))
        self.elevation_residual = nn.Parameter(torch.zeros(count))
        self.radius_residual = nn.Parameter(torch.zeros(count))
        self.centre_residual = nn.Parameter(torch.zeros(count, 2))
        self.focal_residual = nn.Parameter(torch.zeros(()))

    def values(self, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        azimuth = self.base_angles[indices] + math.radians(8) * torch.tanh(self.azimuth_residual[indices])
        elevation = self.base_elevation + math.radians(10) * torch.tanh(self.elevation_residual[indices])
        radius = 3.0 * torch.exp(0.18 * torch.tanh(self.radius_residual[indices]))
        centre = self.base_centroids[indices] + self.resolution * 0.05 * torch.tanh(self.centre_residual[indices])
        focal = self.base_focal * torch.exp(0.15 * torch.tanh(self.focal_residual))
        return azimuth, elevation, radius, centre[:, 0], centre[:, 1], focal.expand_as(radius)

    @staticmethod
    def rays_from_values(x: torch.Tensor, y: torch.Tensor, azimuth: torch.Tensor,
                         elevation: torch.Tensor, radius: torch.Tensor, cx: torch.Tensor,
                         cy: torch.Tensor, focal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_elevation = torch.cos(elevation)
        origins = torch.stack((radius * torch.sin(azimuth) * cos_elevation,
                               radius * torch.sin(elevation),
                               radius * torch.cos(azimuth) * cos_elevation), dim=-1)
        forward = F.normalize(-origins, dim=-1)
        world_up = torch.zeros_like(forward); world_up[:, 1] = 1.0
        right = F.normalize(torch.cross(forward, world_up, dim=-1), dim=-1)
        up = F.normalize(torch.cross(right, forward, dim=-1), dim=-1)
        down = -up
        dx, dy = (x - cx) / focal, (y - cy) / focal
        directions = F.normalize(right * dx[:, None] + down * dy[:, None] + forward, dim=-1)
        return origins, directions

    def rays(self, indices: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rays_from_values(x, y, *self.values(indices))

    def regularisation(self) -> torch.Tensor:
        return (self.azimuth_residual.square().mean() + self.elevation_residual.square().mean()
                + self.radius_residual.square().mean() + self.centre_residual.square().mean()
                + self.focal_residual.square())


def load_turntable(root: Path, resolution: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    records = []
    for path in root.glob("*.png"):
        try: angle = int(path.stem)
        except ValueError: continue
        with Image.open(path) as opened:
            image = opened.convert("RGBA").resize((resolution, resolution), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        records.append((angle, array))
    records.sort(key=lambda item: item[0])
    if len(records) < 6: raise RuntimeError(f"need >=6 RGBA views under {root}")
    angles = torch.tensor([record[0] for record in records], device=device)
    images = torch.from_numpy(np.stack([record[1] for record in records])).to(device)
    centroids = []
    for alpha in images[..., 3]:
        ys, xs = torch.where(alpha > 0.05)
        centroids.append(torch.stack((xs.float().mean(), ys.float().mean())) if len(xs)
                         else alpha.new_tensor((resolution / 2, resolution / 2)))
    return images, angles, torch.stack(centroids)


def volume_render(field: RadianceField, origins: torch.Tensor, directions: torch.Tensor,
                  backgrounds: torch.Tensor, samples: int, perturb: bool) -> tuple[torch.Tensor, ...]:
    radius = origins.norm(dim=-1)
    near, far = (radius - 1.65).clamp_min(0.1), radius + 1.65
    unit = torch.linspace(0, 1, samples, device=origins.device)
    depths = near[:, None] * (1 - unit) + far[:, None] * unit
    if perturb:
        midpoint = (depths[:, 1:] + depths[:, :-1]) * 0.5
        lower = torch.cat((depths[:, :1], midpoint), dim=1)
        upper = torch.cat((midpoint, depths[:, -1:]), dim=1)
        depths = lower + (upper - lower) * torch.rand_like(depths)
    points = origins[:, None, :] + directions[:, None, :] * depths[..., None]
    flat_directions = directions[:, None, :].expand_as(points)
    colours, sigma = field(points.reshape(-1, 3), flat_directions.reshape(-1, 3))
    colours, sigma = colours.reshape(*points.shape[:-1], 3), sigma.reshape(*points.shape[:-1])
    intervals = depths[:, 1:] - depths[:, :-1]
    intervals = torch.cat((intervals, intervals[:, -1:]), dim=1)
    alpha = 1.0 - torch.exp(-sigma * intervals)
    transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :1]),
                                             1.0 - alpha + 1e-8), dim=1), dim=1)[:, :-1]
    weights = alpha * transmittance
    opacity = weights.sum(1)
    colour = (weights[..., None] * colours).sum(1) + (1.0 - opacity[..., None]) * backgrounds
    expected_depth = (weights * depths).sum(1) / opacity.clamp_min(1e-6)
    return colour, opacity, expected_depth, weights


def sample_rays(images: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, ...]:
    views, height, width, _ = images.shape
    alpha = images[..., 3]
    probabilities = (0.03 + 0.97 * alpha).flatten()
    flat = torch.multinomial(probabilities, batch_size, replacement=True)
    pixels_per_view = height * width
    view = flat // pixels_per_view
    pixel = flat % pixels_per_view
    y, x = pixel // width, pixel % width
    target = images[view, y, x]
    return view, x.float() + 0.5, y.float() + 0.5, target[..., :3], target[..., 3]


def interpolated_values(cameras: OrbitCameras, angle: int) -> tuple[torch.Tensor, ...]:
    available = [int(round(math.degrees(value) % 360)) for value in cameras.base_angles.tolist()]
    left = min(available, key=lambda source: (angle - source) % 360)
    right = min(available, key=lambda source: (source - angle) % 360)
    left_index, right_index = available.index(left), available.index(right)
    span = (right - left) % 360
    fraction = 0.0 if span == 0 else ((angle - left) % 360) / span
    indices = torch.tensor([left_index, right_index], device=cameras.base_angles.device)
    values = cameras.values(indices)
    # Azimuth itself follows the requested virtual camera; only learned offset is interpolated.
    learned_offsets = values[0] - cameras.base_angles[indices]
    azimuth = torch.tensor([math.radians(angle)], device=indices.device) + (
        (1 - fraction) * learned_offsets[:1] + fraction * learned_offsets[1:])
    result = [azimuth]
    for value in values[1:]: result.append((1 - fraction) * value[:1] + fraction * value[1:])
    return tuple(result)


def render_image(field: RadianceField, cameras: OrbitCameras, angle: int, resolution: int,
                 samples: int, chunk: int) -> np.ndarray:
    device = next(field.parameters()).device
    y, x = torch.meshgrid(torch.arange(resolution, device=device) + 0.5,
                          torch.arange(resolution, device=device) + 0.5, indexing="ij")
    # Scale learned source intrinsics/centres from train resolution to render resolution.
    values = list(interpolated_values(cameras, angle))
    scale = resolution / cameras.resolution
    values[3], values[4], values[5] = values[3] * scale, values[4] * scale, values[5] * scale
    rgba = []
    with torch.inference_mode():
        for start in tqdm(range(0, resolution * resolution, chunk), desc=f"render {angle:03d}", leave=False):
            xs, ys = x.flatten()[start:start + chunk], y.flatten()[start:start + chunk]
            expanded = [value.expand(len(xs)) for value in values]
            origins, directions = OrbitCameras.rays_from_values(xs, ys, *expanded)
            background = torch.zeros(len(xs), 3, device=device)
            colour, opacity, _, _ = volume_render(field, origins, directions, background, samples, False)
            rgba.append(torch.cat((colour, opacity[:, None]), dim=1).cpu())
    return (torch.cat(rgba).reshape(resolution, resolution, 4).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/render a masked sparse-view vehicle NeRF")
    parser.add_argument("--rgba-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--train-resolution", type=int, default=256)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=3072)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--render-samples", type=int, default=96)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--camera-lr", type=float, default=2e-4)
    parser.add_argument("--camera-start", type=int, default=800)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--render-chunk", type=int, default=4096)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" and not args.render_only:
        print("warning: object NeRF training is intended for CUDA")
    images, angles, centroids = load_turntable(args.rgba_root, args.train_resolution, device)
    field = RadianceField().to(device)
    cameras = OrbitCameras(angles, centroids, args.train_resolution).to(device)
    optimizer = torch.optim.Adam(({"params": field.parameters(), "lr": args.lr},
                                  {"params": cameras.parameters(), "lr": args.camera_lr}), betas=(0.9, 0.99))
    start_step = 1
    checkpoint_path = args.output / "latest.pt"
    resume = args.resume or (checkpoint_path if args.render_only else None)
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        field.load_state_dict(saved["field"]); cameras.load_state_dict(saved["cameras"])
        if "optimizer" in saved and not args.render_only: optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved.get("step", 0)) + 1
    if not args.render_only:
        field.train(); cameras.train()
        progress = tqdm(range(start_step, args.steps + 1), desc=f"object NeRF {args.rgba_root.parent.name}")
        for step in progress:
            view, x, y, target_rgb, target_alpha = sample_rays(images, args.batch_size)
            origins, directions = cameras.rays(view, x, y)
            backgrounds = torch.rand(args.batch_size, 3, device=device)
            rendered, opacity, _, weights = volume_render(field, origins, directions, backgrounds,
                                                           args.samples, True)
            target = target_rgb * target_alpha[:, None] + backgrounds * (1 - target_alpha[:, None])
            colour_loss = ((rendered - target).square() * (1 + 2 * target_alpha[:, None])).mean()
            mask_loss = F.binary_cross_entropy(opacity.clamp(1e-5, 1 - 1e-5), target_alpha)
            entropy = (weights * (1 - weights)).mean()
            camera_reg = cameras.regularisation()
            loss = colour_loss + 0.25 * mask_loss + 0.01 * entropy + 1e-3 * camera_reg
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if step < args.camera_start:
                for parameter in cameras.parameters(): parameter.grad = None
            torch.nn.utils.clip_grad_norm_(field.parameters(), 5.0)
            optimizer.step()
            if step % 25 == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}", rgb=f"{colour_loss.item():.4f}",
                                     mask=f"{mask_loss.item():.4f}")
            if step % 500 == 0 or step == args.steps:
                torch.save({"field": field.state_dict(), "cameras": cameras.state_dict(),
                            "optimizer": optimizer.state_dict(), "step": step,
                            "angles": angles.cpu(), "args": vars(args)}, checkpoint_path)
    field.eval(); cameras.eval()
    render_dir = args.output / "renders"; render_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    rendered_arrays: dict[int, np.ndarray] = {}
    for angle in range(0, 360, 15):
        array = render_image(field, cameras, angle, args.render_resolution,
                             args.render_samples, args.render_chunk)
        path = render_dir / f"{angle}.png"
        Image.fromarray(array, "RGBA").save(path, compress_level=4)
        rendered_arrays[angle] = array
        manifest.append({"angle": angle, "file": str(path)})
    validation = []
    for source in sorted(args.rgba_root.glob("*.png")):
        try: angle = int(source.stem)
        except ValueError: continue
        if angle not in rendered_arrays: continue
        with Image.open(source) as opened:
            target = np.asarray(opened.convert("RGBA").resize(
                (args.render_resolution, args.render_resolution), Image.Resampling.LANCZOS),
                dtype=np.float32) / 255.0
        predicted = rendered_arrays[angle].astype(np.float32) / 255.0
        target_pm = np.concatenate((target[..., :3] * target[..., 3:4], target[..., 3:4]), axis=-1)
        predicted_pm = np.concatenate((predicted[..., :3] * predicted[..., 3:4], predicted[..., 3:4]), axis=-1)
        mse = float(np.mean((target_pm - predicted_pm) ** 2))
        target_mask, predicted_mask = target[..., 3] > 0.5, predicted[..., 3] > 0.5
        union = np.logical_or(target_mask, predicted_mask).sum()
        iou = float(np.logical_and(target_mask, predicted_mask).sum() / max(1, union))
        validation.append({"angle": angle, "psnr": float(-10 * math.log10(max(mse, 1e-10))),
                           "alpha_iou": iou})
    mean_psnr = float(np.mean([item["psnr"] for item in validation]))
    mean_iou = float(np.mean([item["alpha_iou"] for item in validation]))
    camera_data = {"source_angles": angles.tolist(), "azimuth_residual_degrees":
                   torch.rad2deg(math.radians(8) * torch.tanh(cameras.azimuth_residual)).detach().cpu().tolist(),
                   "elevation_degrees": torch.rad2deg(cameras.base_elevation + math.radians(10)
                   * torch.tanh(cameras.elevation_residual)).detach().cpu().tolist(),
                   "radius": (3 * torch.exp(.18 * torch.tanh(cameras.radius_residual))).detach().cpu().tolist(),
                   "focal": float(cameras.base_focal * torch.exp(.15 * torch.tanh(cameras.focal_residual))),
                   "renders": manifest, "source_view_validation": validation,
                   "mean_source_psnr": mean_psnr, "mean_source_alpha_iou": mean_iou}
    (args.output / "cameras_and_renders.json").write_text(
        json.dumps(camera_data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

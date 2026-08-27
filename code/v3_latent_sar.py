"""Core data, models and visualisation helpers for the v3.0 RGB-to-SAR system.

v3.0 deliberately separates (1) learning the real SAR image manifold and
(2) learning an RGB/geometry-conditioned distribution in that latent space.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from bbox_data import image_tensor, read_annotation
from joint_data import nearest_available_angle, source_rgb_angle
from rgb2sar.data import rgba_to_rgb
from saratrx import SOC40_CLASSES


DEP_TO_ID = {15: 0, 30: 1, 45: 2, 60: 3}
_RGB_THUMBNAIL_CACHE: dict[tuple[Path, int], dict[Path, torch.Tensor]] = {}


def _record_key(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def build_manifest(sar_train_root: Path, output: Path, val_fraction: float = .15,
                   seed: int = 20260723) -> dict[str, list[str]]:
    """Make a fixed class/depression-stratified validation split.

    The original supplied ``test`` directory is intentionally not inspected or
    used here.  It remains a final, one-shot evaluation set.
    """
    output = Path(output)
    if output.is_file():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved.get("source_root") == str(Path(sar_train_root).resolve()):
            return saved
    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for class_id, name in enumerate(SOC40_CLASSES):
        for path in sorted((Path(sar_train_root) / name).glob("X_HH_*.tif")):
            try:
                _, meta = read_annotation(path.with_suffix(".xml"))
            except Exception:
                continue
            groups[class_id, int(meta["depression"])] .append(path)
    train, validation = [], []
    for group, paths in sorted(groups.items()):
        ordered = sorted(paths, key=lambda p: hashlib.sha256(
            f"{seed}:{group}:{_record_key(p, sar_train_root)}".encode()).hexdigest())
        count = max(1, round(len(ordered) * val_fraction)) if len(ordered) > 2 else 0
        validation.extend(_record_key(path, sar_train_root) for path in ordered[:count])
        train.extend(_record_key(path, sar_train_root) for path in ordered[count:])
    payload = {"version": "v3.0", "seed": seed, "validation_fraction": val_fraction,
               "source_root": str(Path(sar_train_root).resolve()),
               "train": sorted(train), "validation": sorted(validation)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


class V3PairDataset(Dataset):
    """Fixed split of real X/HH ROIs paired only by vehicle identity/view."""

    def __init__(self, rgb_root: Path, sar_root: Path, manifest: dict, split: str,
                 rgb_size: int = 128, augment_rgb: bool = False, load_rgb: bool = True) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(split)
        self.rgb_root, self.sar_root = Path(rgb_root), Path(sar_root)
        self.rgb_size, self.augment_rgb, self.load_rgb = rgb_size, augment_rgb, load_rgb
        requested = set(manifest[split])
        self.rgb_paths: dict[tuple[str, int], Path] = {}
        self.class_angles: dict[str, list[int]] = {}
        for name in SOC40_CLASSES:
            folder = self.rgb_root / name
            numeric = [path for path in folder.glob("*.png") if path.stem.isdigit()]
            degrees = any(path.stem == "0" for path in numeric)
            for path in numeric:
                try:
                    self.rgb_paths[name, source_rgb_angle(path, degrees)] = path
                except ValueError:
                    continue
            self.class_angles[name] = sorted(angle for key, angle in self.rgb_paths if key == name)
        cache_key = (self.rgb_root.resolve(), self.rgb_size)
        if self.load_rgb:
            if cache_key not in _RGB_THUMBNAIL_CACHE:
                cache: dict[Path, torch.Tensor] = {}
                existing = self.rgb_root / f".joint_rgb_cache_{self.rgb_size}.pt"
                if existing.is_file():
                    saved = torch.load(existing, map_location="cpu", weights_only=True)
                    cache = {self.rgb_root / relative: value for relative, value in saved["images"].items()}
                else:
                    for path in sorted(set(self.rgb_paths.values())):
                        with Image.open(path) as image:
                            value = image_tensor(rgba_to_rgb(image), self.rgb_size, True)
                        cache[path] = ((value + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
                _RGB_THUMBNAIL_CACHE[cache_key] = cache
            self.rgb_cache = _RGB_THUMBNAIL_CACHE[cache_key]
        else:
            self.rgb_cache = {}
        self.records: list[tuple[Path, int, dict[str, object], int]] = []
        for relative in sorted(requested):
            path = self.sar_root / relative
            try:
                _, meta = read_annotation(path.with_suffix(".xml"))
                class_id = SOC40_CLASSES.index(path.parent.name)
                angle = nearest_available_angle(int(meta["azimuth"]), self.class_angles[path.parent.name])
            except Exception:
                continue
            self.records.append((path, class_id, meta, angle))
        if not self.records:
            raise RuntimeError(f"empty v3 split {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, class_id, meta, source_angle = self.records[index]
        with Image.open(path) as image:
            sar = image_tensor(image, 64, False)
        if self.load_rgb:
            rgb = self.rgb_cache[self.rgb_paths[SOC40_CLASSES[class_id], source_angle]].float().div(127.5).sub(1.)
        else:
            rgb = torch.zeros(3, self.rgb_size, self.rgb_size)
        if self.augment_rgb and self.load_rgb:
            rgb = (rgb * random.uniform(.92, 1.08) + random.uniform(-.03, .03)
                   + torch.randn_like(rgb) * random.uniform(0, .01)).clamp(-1, 1)
        azimuth = math.radians(int(meta["azimuth"])); source = math.radians(source_angle)
        condition = torch.tensor((math.sin(azimuth), math.cos(azimuth),
                                  int(meta["depression"]) / 60.0,
                                  math.sin(source), math.cos(source)), dtype=torch.float32)
        return {"sar": sar, "rgb": rgb, "class_id": torch.tensor(class_id),
                "condition": condition, "depression": torch.tensor(DEP_TO_ID[int(meta["depression"])]),
                "azimuth": torch.tensor(int(meta["azimuth"]))}


class ResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int = 0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(16, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(16, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Linear(time_dim, channels * 2) if time_dim else None

    def forward(self, x: torch.Tensor, embedding: torch.Tensor | None = None) -> torch.Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        if self.time is not None and embedding is not None:
            scale, shift = self.time(embedding).chunk(2, 1)
            y = y * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        y = self.conv2(F.silu(self.norm2(y)))
        return x + y


class SARAutoencoder(nn.Module):
    """A compact SAR-domain decoder; no real-image skip path reaches output."""

    def __init__(self, latent_channels: int = 16, base: int = 32) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base, 3, 1, 1), ResBlock(base),
            nn.Conv2d(base, base * 2, 4, 2, 1), ResBlock(base * 2),
            nn.Conv2d(base * 2, base * 3, 4, 2, 1), ResBlock(base * 3),
            nn.Conv2d(base * 3, latent_channels, 4, 2, 1), ResBlock(latent_channels),
        )
        self.decoder_in = nn.Conv2d(latent_channels, base * 3, 3, 1, 1)
        self.decoder = nn.ModuleList((
            ResBlock(base * 3), nn.ConvTranspose2d(base * 3, base * 2, 4, 2, 1),
            ResBlock(base * 2), nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            ResBlock(base), nn.ConvTranspose2d(base, base, 4, 2, 1),
            ResBlock(base), nn.Conv2d(base, 1, 3, 1, 1), nn.Tanh(),
        ))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.decoder_in(latent)
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(image)
        return self.decode(latent), latent


class RGBSpatialConditioner(nn.Module):
    """FPN-style RGB encoder producing a spatial condition at latent resolution."""

    def __init__(self, condition_channels: int = 64, classes: int = 40, base: int = 32) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, base, 4, 2, 1), nn.GroupNorm(8, base), nn.SiLU(), ResBlock(base))
        self.stage2 = nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1), nn.GroupNorm(16, base * 2), nn.SiLU(), ResBlock(base * 2))
        self.stage3 = nn.Sequential(nn.Conv2d(base * 2, condition_channels, 4, 2, 1), nn.GroupNorm(16, condition_channels), nn.SiLU(), ResBlock(condition_channels))
        self.class_embedding = nn.Embedding(classes, 32)
        self.meta = nn.Sequential(nn.Linear(5 + 32, condition_channels * 2), nn.SiLU(),
                                  nn.Linear(condition_channels * 2, condition_channels * 2))

    def forward(self, rgb: torch.Tensor, class_id: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.stage3(self.stage2(self.stage1(rgb)))
        x = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False)
        scale, bias = self.meta(torch.cat((condition, self.class_embedding(class_id)), 1)).chunk(2, 1)
        return x * (1 + .25 * torch.tanh(scale)[:, :, None, None]) + bias[:, :, None, None]


def timestep_embedding(timestep: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    frequency = torch.exp(-math.log(10000) * torch.arange(half, device=timestep.device) / max(half - 1, 1))
    values = timestep.float()[:, None] * frequency[None]
    return torch.cat((values.sin(), values.cos()), 1)


class LatentDenoiser(nn.Module):
    def __init__(self, latent_channels: int = 16, condition_channels: int = 64, hidden: int = 96) -> None:
        super().__init__()
        self.input = nn.Conv2d(latent_channels + condition_channels, hidden, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(128, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden))
        self.blocks = nn.ModuleList((ResBlock(hidden, hidden), ResBlock(hidden, hidden), ResBlock(hidden, hidden), ResBlock(hidden, hidden)))
        self.output = nn.Sequential(nn.GroupNorm(16, hidden), nn.SiLU(), nn.Conv2d(hidden, latent_channels, 3, padding=1))

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.input(torch.cat((noisy, condition), 1))
        embedding = self.time(timestep_embedding(timestep))
        for block in self.blocks:
            x = block(x, embedding)
        return self.output(x)


class LatentDiffusion(nn.Module):
    def __init__(self, latent_channels: int = 16, timesteps: int = 50) -> None:
        super().__init__()
        self.timesteps = timesteps
        beta = torch.linspace(1e-4, .02, timesteps)
        alpha = 1 - beta
        self.register_buffer("alpha_bar", torch.cumprod(alpha, 0))

    def noisy(self, clean: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_bar[timestep][:, None, None, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    @torch.no_grad()
    def sample(self, denoiser: LatentDenoiser, condition: torch.Tensor, steps: int = 20,
               seed: int | None = None) -> torch.Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=condition.device).manual_seed(seed)
        x = torch.randn(len(condition), 16, 8, 8, device=condition.device, generator=generator)
        schedule = torch.linspace(self.timesteps - 1, 0, steps, device=condition.device).long().unique_consecutive()
        for index, current in enumerate(schedule):
            timestep = torch.full((len(x),), int(current), device=x.device, dtype=torch.long)
            alpha = self.alpha_bar[timestep][:, None, None, None]
            epsilon = denoiser(x, timestep, condition)
            predicted_clean = (x - (1 - alpha).sqrt() * epsilon) / alpha.sqrt().clamp_min(1e-5)
            if index + 1 == len(schedule):
                x = predicted_clean
            else:
                next_timestep = torch.full_like(timestep, int(schedule[index + 1]))
                next_alpha = self.alpha_bar[next_timestep][:, None, None, None]
                x = next_alpha.sqrt() * predicted_clean + (1 - next_alpha).sqrt() * epsilon
        return x


def sar_reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.smooth_l1_loss(reconstruction, target)
    gradient = F.l1_loss(reconstruction[..., 1:, :] - reconstruction[..., :-1, :],
                         target[..., 1:, :] - target[..., :-1, :])
    log_recon = torch.log(((reconstruction + 1) * .5).clamp_min(1e-4))
    log_target = torch.log(((target + 1) * .5).clamp_min(1e-4))
    moments = (F.l1_loss(log_recon.mean((2, 3)), log_target.mean((2, 3))) +
               F.l1_loss(log_recon.std((2, 3)), log_target.std((2, 3))))
    return l1 + .25 * gradient + .10 * moments, {"l1": l1.detach(), "gradient": gradient.detach(), "moments": moments.detach()}


def save_visual_grid(path: Path, rgb: torch.Tensor, real: torch.Tensor, reconstruction: torch.Tensor,
                     generated: torch.Tensor | None = None) -> None:
    """Rows are samples; columns are RGB / real SAR / reconstruction / generated."""
    tensors = [rgb, real, reconstruction] + ([] if generated is None else [generated])
    count = min(len(real), 8)
    cells = []
    for index in range(count):
        row = []
        for tensor in tensors:
            value = tensor[index].detach().float().cpu()
            if value.shape[0] == 3:
                value = F.interpolate(value[None], (64, 64), mode="bilinear", align_corners=False)[0]
                array = ((value.permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()
            else:
                array = (((value[0] + 1) * 127.5).clamp(0, 255).byte().numpy())
                array = np.repeat(array[..., None], 3, axis=2)
            row.append(array)
        cells.append(np.concatenate(row, axis=1))
    image = Image.fromarray(np.concatenate(cells, axis=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)

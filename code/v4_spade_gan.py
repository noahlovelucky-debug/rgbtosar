"""Fast SPADE conditional GAN for RGB-to-SAR v4.

The generator is one feed-forward pass.  RGB spatial features are injected at
every decoding scale through SPADE, while class and acquisition geometry are
visible to the multi-scale projection discriminator.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def spectral(module: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(module)


class SPADE(nn.Module):
    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(16, channels), channels, affine=False)
        self.condition = nn.Sequential(nn.Conv2d(condition_channels, 64, 3, padding=1), nn.SiLU(),
                                       nn.Conv2d(64, channels * 2, 3, padding=1))

    def forward(self, feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = F.interpolate(condition, feature.shape[-2:], mode="bilinear", align_corners=False)
        scale, bias = self.condition(condition).chunk(2, 1)
        return self.norm(feature) * (1 + scale) + bias


class SPADEResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_channels: int) -> None:
        super().__init__()
        self.spade1 = SPADE(in_channels, condition_channels)
        # Spectral normalisation remains on D.  Removing it from the inference
        # path of G is material for the low-latency deployment target.
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.spade2 = SPADE(out_channels, condition_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        x = self.conv1(F.silu(self.spade1(x, condition)))
        x = self.conv2(F.silu(self.spade2(x, condition)))
        return shortcut + x


class RGBConditionPyramid(nn.Module):
    """RGB FPN with global class/geometry modulation at all spatial scales."""

    def __init__(self, condition_channels: int = 64, classes: int = 40, base: int = 32) -> None:
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(3, base, 4, 2, 1), nn.GroupNorm(8, base), nn.SiLU(),
                                nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
        self.s2 = nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1), nn.GroupNorm(16, base * 2), nn.SiLU(),
                                nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.SiLU())
        self.s3 = nn.Sequential(nn.Conv2d(base * 2, condition_channels, 4, 2, 1), nn.GroupNorm(16, condition_channels), nn.SiLU())
        self.s4 = nn.Sequential(nn.Conv2d(condition_channels, condition_channels, 4, 2, 1), nn.GroupNorm(16, condition_channels), nn.SiLU())
        self.project = nn.ModuleList(nn.Conv2d(channels, condition_channels, 1) for channels in (base, base * 2, condition_channels, condition_channels))
        self.class_embedding = nn.Embedding(classes, 32)
        self.geometry = nn.Sequential(nn.Linear(37, condition_channels * 2), nn.SiLU(),
                                      nn.Linear(condition_channels * 2, condition_channels * 2))

    def forward(self, rgb: torch.Tensor, class_id: torch.Tensor, geometry: torch.Tensor) -> tuple[torch.Tensor, ...]:
        maps = (self.s1(rgb),)
        maps += (self.s2(maps[-1]),)
        maps += (self.s3(maps[-1]),)
        maps += (self.s4(maps[-1]),)
        scale, bias = self.geometry(torch.cat((self.class_embedding(class_id), geometry), 1)).chunk(2, 1)
        return tuple(self.project[index](feature) * (1 + .25 * torch.tanh(scale)[:, :, None, None]) +
                     bias[:, :, None, None] for index, feature in enumerate(maps))


class SARSPADEGenerator(nn.Module):
    """SPADE baseline adapted to 64px SAR generation with a 32-d style noise."""

    def __init__(self, classes: int = 40, geometry_dim: int = 5, base: int = 32, noise_dim: int = 32) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.conditioner = RGBConditionPyramid(64, classes, base)
        self.global_condition = nn.Sequential(nn.Embedding(classes, 32),)
        self.style = nn.Sequential(nn.Linear(noise_dim + 32 + geometry_dim, base * 8 * 4 * 4), nn.SiLU())
        self.block4 = SPADEResBlock(base * 8, base * 8, 64)
        self.block8 = SPADEResBlock(base * 8, base * 4, 64)
        self.block16 = SPADEResBlock(base * 4, base * 2, 64)
        self.block32 = SPADEResBlock(base * 2, base, 64)
        self.block64 = SPADEResBlock(base, base, 64)
        self.output = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, rgb: torch.Tensor, class_id: torch.Tensor, geometry: torch.Tensor,
                noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = rgb.new_zeros(len(rgb), self.noise_dim)
        maps = self.conditioner(rgb, class_id, geometry)  # 64,32,16,8 RGB scales
        x = self.style(torch.cat((noise, self.global_condition[0](class_id), geometry), 1)).reshape(len(rgb), -1, 4, 4)
        x = self.block4(x, maps[3])
        x = self.block8(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), maps[3])
        x = self.block16(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), maps[2])
        x = self.block32(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), maps[1])
        x = self.block64(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), maps[0])
        return torch.tanh(self.output(F.silu(x)))


class ProjectionPatchDiscriminator(nn.Module):
    def __init__(self, classes: int = 40, geometry_dim: int = 5, base: int = 32) -> None:
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8)
        layers = []
        previous = 1
        for channel in channels:
            layers.extend((spectral(nn.Conv2d(previous, channel, 4, 2, 1)), nn.LeakyReLU(.2, inplace=True)))
            previous = channel
        self.features = nn.Sequential(*layers)
        self.score = spectral(nn.Conv2d(channels[-1], 1, 3, padding=1))
        self.class_embedding = nn.Embedding(classes, channels[-1])
        self.geometry = nn.Sequential(nn.Linear(geometry_dim, channels[-1]), nn.SiLU(), nn.Linear(channels[-1], channels[-1]))

    def forward(self, image: torch.Tensor, class_id: torch.Tensor, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(image)
        condition = self.class_embedding(class_id) + self.geometry(geometry)
        projection = (features * condition[:, :, None, None]).sum(1, keepdim=True)
        return (self.score(features) + projection).mean((1, 2, 3)), features


class MultiScaleProjectionDiscriminator(nn.Module):
    def __init__(self, classes: int = 40, geometry_dim: int = 5, base: int = 32) -> None:
        super().__init__()
        self.full = ProjectionPatchDiscriminator(classes, geometry_dim, base)
        self.half_scale = ProjectionPatchDiscriminator(classes, geometry_dim, base)

    def forward(self, image: torch.Tensor, class_id: torch.Tensor, geometry: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        score_a, feature_a = self.full(image, class_id, geometry)
        score_b, feature_b = self.half_scale(F.avg_pool2d(image, 2), class_id, geometry)
        return .5 * (score_a + score_b), (feature_a, feature_b)

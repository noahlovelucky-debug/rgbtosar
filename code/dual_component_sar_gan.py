"""Large dual-component conditional GAN for continuous RGB-to-SAR synthesis.

The observation is factorised in amplitude space as

    observed SAR = denoised reflectivity * exp(log-speckle)

and each term, plus the composed observation, has its own conditional
projection discriminator.  This keeps the visually useful spatial conditioning
of continuous-spatial-v1 without relying on its hand-written speckle renderer.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from v4_spade_gan import ProjectionPatchDiscriminator


LOG_NOISE_LIMIT = 0.8


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels)


class LargeRGBIdentityEncoder(nn.Module):
    """High-capacity identity encoder exposing 64/32/16/8 spatial features."""

    def __init__(self, classes: int = 40, dim: int = 512, base: int = 64) -> None:
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8)
        stages = []
        previous = 3
        for channel in channels:
            stages.append(nn.Sequential(
                nn.Conv2d(previous, channel, 4, 2, 1, bias=False),
                _norm(channel), nn.SiLU(inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
                _norm(channel), nn.SiLU(inplace=True),
            ))
            previous = channel
        self.stages = nn.ModuleList(stages)
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels[-1], dim), nn.LayerNorm(dim), nn.SiLU())
        self.classifier = nn.Linear(dim, classes)

    def forward(self, image: torch.Tensor, return_pyramid: bool = False):
        pyramid = []
        for stage in self.stages:
            image = stage(image)
            pyramid.append(image)
        identity = self.embedding(image)
        logits = self.classifier(identity)
        return (identity, logits, tuple(pyramid)) if return_pyramid else (identity, logits)


class ResidualDecodeBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            _norm(output_channels), nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False),
            _norm(output_channels),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, scale_factor=2, mode="bilinear", align_corners=False)
        return F.silu(self.net(image) + self.skip(image), inplace=True)


class DenoisedSARGenerator(nn.Module):
    """Deterministic RGB/geometry-conditioned denoised SAR generator."""

    def __init__(self, identity_dim: int = 512, geometry_dim: int = 12,
                 rgb_base: int = 64, base: int = 64) -> None:
        super().__init__()
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU())
        self.fc = nn.Linear(identity_dim + 256, base * 8 * 4 * 4)
        channels = (base * 8, base * 8, base * 4, base * 2, base)
        self.blocks = nn.ModuleList(
            ResidualDecodeBlock(channels[index], channels[index + 1])
            for index in range(4))
        rgb_channels = (rgb_base * 8, rgb_base * 4, rgb_base * 2, rgb_base)
        self.rgb_projection = nn.ModuleList(
            nn.Conv2d(source, target, 1)
            for source, target in zip(rgb_channels, channels[1:]))
        self.refine = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            _norm(base), nn.SiLU(inplace=True),
            nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, identity: torch.Tensor, geometry: torch.Tensor,
                pyramid: tuple[torch.Tensor, ...]) -> torch.Tensor:
        latent = torch.cat((identity, self.geometry(geometry)), 1)
        image = self.fc(latent).reshape(len(identity), -1, 4, 4)
        for index, block in enumerate(self.blocks):
            image = block(image)
            spatial = F.interpolate(
                pyramid[3 - index], image.shape[-2:],
                mode="bilinear", align_corners=False)
            image = image + self.rgb_projection[index](spatial)
        return torch.tanh(self.refine(image))


class SARNoiseGenerator(nn.Module):
    """Learn conditional multiplicative speckle and acquisition noise."""

    noise_dim = 64

    def __init__(self, geometry_dim: int = 12, rgb_base: int = 64,
                 base: int = 64) -> None:
        super().__init__()
        self.clean_encoder = nn.ModuleList((
            nn.Sequential(nn.Conv2d(1, base, 3, 1, 1), _norm(base), nn.SiLU()),
            nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1), _norm(base * 2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, 2, 1), _norm(base * 4), nn.SiLU()),
            nn.Sequential(nn.Conv2d(base * 4, base * 6, 4, 2, 1), _norm(base * 6), nn.SiLU()),
        ))
        self.rgb8 = nn.Conv2d(rgb_base * 8, base * 6, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(base * 12, base * 6, 3, padding=1, bias=False),
            _norm(base * 6), nn.SiLU())
        self.style = nn.Sequential(
            nn.Linear(self.noise_dim + geometry_dim, base * 12), nn.SiLU(),
            nn.Linear(base * 12, base * 12))
        self.decode16 = ResidualDecodeBlock(base * 6, base * 4)
        self.decode32 = ResidualDecodeBlock(base * 4, base * 2)
        self.decode64 = ResidualDecodeBlock(base * 2, base)
        self.skip16 = nn.Conv2d(base * 4, base * 4, 1)
        self.skip32 = nn.Conv2d(base * 2, base * 2, 1)
        self.skip64 = nn.Conv2d(base, base, 1)
        self.rgb_projection = nn.ModuleList((
            nn.Conv2d(rgb_base * 4, base * 4, 1),
            nn.Conv2d(rgb_base * 2, base * 2, 1),
            nn.Conv2d(rgb_base, base, 1),
        ))
        self.output = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, clean: torch.Tensor, geometry: torch.Tensor,
                pyramid: tuple[torch.Tensor, ...],
                noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(len(clean), self.noise_dim, device=clean.device,
                                dtype=clean.dtype)
        features = []
        image = clean
        for encoder in self.clean_encoder:
            image = encoder(image)
            features.append(image)
        rgb8 = F.interpolate(pyramid[3], image.shape[-2:],
                             mode="bilinear", align_corners=False)
        image = self.fuse(torch.cat((image, self.rgb8(rgb8)), 1))
        scale, bias = self.style(torch.cat((noise, geometry), 1)).chunk(2, 1)
        image = image * (1 + .25 * torch.tanh(scale)[:, :, None, None])
        image = image + bias[:, :, None, None]
        for decoder, clean_skip, rgb_projection, rgb_feature in (
            (self.decode16, self.skip16(features[2]), self.rgb_projection[0], pyramid[2]),
            (self.decode32, self.skip32(features[1]), self.rgb_projection[1], pyramid[1]),
            (self.decode64, self.skip64(features[0]), self.rgb_projection[2], pyramid[0]),
        ):
            image = decoder(image)
            rgb_feature = F.interpolate(
                rgb_feature, image.shape[-2:], mode="bilinear", align_corners=False)
            image = image + clean_skip + rgb_projection(rgb_feature)
        return LOG_NOISE_LIMIT * torch.tanh(self.output(image))


class DualComponentDiscriminators(nn.Module):
    """Three independent conditional critics for clean, noise and full SAR."""

    def __init__(self, classes: int = 40, geometry_dim: int = 12,
                 base: int = 64) -> None:
        super().__init__()
        self.clean = ProjectionPatchDiscriminator(classes, geometry_dim, base)
        self.noise = ProjectionPatchDiscriminator(classes, geometry_dim, base)
        self.full = ProjectionPatchDiscriminator(classes, geometry_dim, base)


@torch.no_grad()
def decompose_real_sar(observed: torch.Tensor, kernel: int = 7
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge-preserving Lee-style amplitude decomposition.

    Returns denoised amplitude in [-1, 1] and bounded log multiplicative noise.
    The median local variance estimates the per-image noise floor.  A modest
    shrinkage of the Lee weight makes the two learned components identifiable:
    isolated scattering centres remain in ``clean`` while coherent granular
    fluctuation is assigned to ``log_noise``.
    """
    amplitude = ((observed + 1) * .5).clamp(1e-4, 1)
    mean = F.avg_pool2d(amplitude, kernel, 1, kernel // 2)
    variance = (F.avg_pool2d(amplitude.square(), kernel, 1, kernel // 2)
                - mean.square()).clamp_min(0)
    noise_variance = torch.quantile(
        variance.flatten(1).float(), .50, dim=1).to(variance.dtype)
    noise_variance = noise_variance[:, None, None, None]
    weight = .70 * ((variance - noise_variance).clamp_min(0)
                    / variance.clamp_min(1e-6)).clamp(0, 1)
    clean_amplitude = (mean + weight * (amplitude - mean)).clamp(1e-4, 1)
    log_noise = (amplitude.log() - clean_amplitude.log()).clamp(
        -LOG_NOISE_LIMIT, LOG_NOISE_LIMIT)
    return clean_amplitude * 2 - 1, log_noise


def compose_sar(clean: torch.Tensor, log_noise: torch.Tensor) -> torch.Tensor:
    """Differentiable multiplicative SAR observation model."""
    clean_amplitude = ((clean + 1) * .5).clamp(1e-4, 1)
    observed = clean_amplitude * torch.exp(log_noise)
    return observed.clamp(0, 1) * 2 - 1


def noise_view(log_noise: torch.Tensor) -> torch.Tensor:
    return (log_noise / LOG_NOISE_LIMIT).clamp(-1, 1)


def initialise(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

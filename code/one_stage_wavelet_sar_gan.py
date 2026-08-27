"""One-pass reflectivity/speckle GAN with alias-free spatial conditioning.

Unlike the cascaded dual-generator model, one shared decoder produces the
denoised reflectivity and the parameters of a multiplicative speckle field in a
single pass.  A spatial random field supplies stochastic detail without asking
a decoder to invent a complete noise image, which reduces tiled/checkerboard
artefacts.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dual_component_sar_gan import (
    LOG_NOISE_LIMIT, LargeRGBIdentityEncoder, compose_sar)
from v4_spade_gan import ProjectionPatchDiscriminator


def _norm(channels: int, affine: bool = True) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels, affine=affine)


class Blur2d(nn.Module):
    """Fixed separable [1,2,1] low-pass filter used after every resize."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        kernel = image.new_tensor(((1., 2., 1.), (2., 4., 2.), (1., 2., 1.))) / 16
        kernel = kernel[None, None].expand(image.shape[1], 1, 3, 3)
        return F.conv2d(F.pad(image, (1, 1, 1, 1), mode="reflect"),
                        kernel, groups=image.shape[1])


class SpatialModulation(nn.Module):
    """Efficient depthwise-separable SPADE-like RGB modulation."""

    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        self.norm = _norm(channels, affine=False)
        self.affine = nn.Sequential(
            nn.Conv2d(condition_channels, condition_channels, 3, padding=1,
                      groups=condition_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(condition_channels, channels * 2, 1))

    def forward(self, image: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        condition = F.interpolate(condition, image.shape[-2:],
                                  mode="bilinear", align_corners=False)
        scale, bias = self.affine(condition).chunk(2, 1)
        return self.norm(image) * (1 + .25 * torch.tanh(scale)) + bias


class AliasFreeSPADEBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int,
                 condition_channels: int) -> None:
        super().__init__()
        self.blur = Blur2d()
        self.mod1 = SpatialModulation(input_channels, condition_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm2 = _norm(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, image: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, scale_factor=2, mode="bilinear",
                              align_corners=False)
        image = self.blur(image)
        shortcut = self.skip(image)
        image = self.conv1(F.silu(self.mod1(image, condition)))
        image = self.conv2(F.silu(self.norm2(image)))
        return F.silu(image + shortcut, inplace=True)


class OneStageWaveletSARGenerator(nn.Module):
    """Shared-decoder one-stage clean SAR and stochastic log-speckle model."""

    spatial_noise_channels = 1

    def __init__(self, identity_dim: int = 512, geometry_dim: int = 12,
                 rgb_base: int = 64, base: int = 64) -> None:
        super().__init__()
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU())
        self.fc = nn.Linear(identity_dim + 256, base * 8 * 4 * 4)
        channels = (base * 8, base * 8, base * 4, base * 2, base)
        condition_channels = (rgb_base * 8, rgb_base * 4,
                              rgb_base * 2, rgb_base)
        self.blocks = nn.ModuleList(
            AliasFreeSPADEBlock(channels[index], channels[index + 1],
                                condition_channels[index])
            for index in range(4))
        self.clean_head = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(base, 1, 3, padding=1))
        self.noise_features = nn.Sequential(
            nn.Conv2d(base + 2, base, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(inplace=True))
        self.noise_scale = nn.Conv2d(base, 1, 3, padding=1)
        self.noise_bias = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, identity: torch.Tensor, geometry: torch.Tensor,
                pyramid: tuple[torch.Tensor, ...],
                spatial_noise: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = torch.cat((identity, self.geometry(geometry)), 1)
        feature = self.fc(latent).reshape(len(identity), -1, 4, 4)
        for index, block in enumerate(self.blocks):
            feature = block(feature, pyramid[3 - index])
        clean = torch.tanh(self.clean_head(feature))
        if spatial_noise is None:
            spatial_noise = torch.randn(
                len(clean), 1, *clean.shape[-2:], device=clean.device,
                dtype=clean.dtype)
        if spatial_noise.shape[-2:] != clean.shape[-2:]:
            spatial_noise = F.interpolate(
                spatial_noise, clean.shape[-2:], mode="bilinear",
                align_corners=False)
        correlated = F.avg_pool2d(
            spatial_noise, 3, stride=1, padding=1)
        random_field = .70 * spatial_noise + .30 * correlated
        noise_feature = self.noise_features(
            torch.cat((feature, clean, correlated), 1))
        # Strictly positive local scale produces heteroscedastic speckle.
        scale = .04 + .38 * torch.sigmoid(self.noise_scale(noise_feature))
        # Only a weak low-frequency acquisition term is learned deterministically.
        bias = .12 * torch.tanh(self.noise_bias(noise_feature))
        log_noise = scale * random_field + bias
        log_noise = log_noise - log_noise.mean((2, 3), keepdim=True)
        log_noise = log_noise.clamp(-LOG_NOISE_LIMIT, LOG_NOISE_LIMIT)
        observed = compose_sar(clean, log_noise)
        return clean, log_noise, observed, feature


def haar_texture(image: torch.Tensor) -> torch.Tensor:
    """Observable Haar detail energy, mapped to the discriminator range."""
    amplitude = (image + 1) * .5
    a = amplitude[..., 0::2, 0::2]
    b = amplitude[..., 0::2, 1::2]
    c = amplitude[..., 1::2, 0::2]
    d = amplitude[..., 1::2, 1::2]
    horizontal = (a + c - b - d) * .5
    vertical = (a + b - c - d) * .5
    diagonal = (a - b - c + d) * .5
    energy = torch.sqrt(
        horizontal.square() + vertical.square()
        + diagonal.square() + 1e-6)
    return 2 * torch.tanh(4 * energy) - 1


class OneStageWaveletDiscriminators(nn.Module):
    """Clean image, observed image and observable wavelet-texture critics."""

    def __init__(self, classes: int = 40, geometry_dim: int = 12,
                 base: int = 64) -> None:
        super().__init__()
        self.clean = ProjectionPatchDiscriminator(classes, geometry_dim, base)
        self.full = ProjectionPatchDiscriminator(classes, geometry_dim, base)
        self.texture = ProjectionPatchDiscriminator(classes, geometry_dim, base)


def initialise(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

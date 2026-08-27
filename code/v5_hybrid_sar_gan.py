"""v5 RGB-driven hybrid GAN for fast, physically plausible 64px SAR synthesis.

The generator never receives a vehicle class id.  Vehicle identity and shape
must come from the RGB encoder, while class labels are visible only to the
discriminators and frozen real-SAR teacher.  The generator predicts a stable
reflectivity field and a spatial speckle scale; a differentiable observation
layer produces the final SAR amplitude image.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from v4_spade_gan import ProjectionPatchDiscriminator


class DecodeBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, rgb_channels: int) -> None:
        super().__init__()
        self.rgb = nn.Conv2d(rgb_channels, output_channels, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, output_channels), output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, output_channels), output_channels),
            nn.SiLU(),
        )
        self.shortcut = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x: torch.Tensor, rgb_feature: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        shortcut = self.shortcut(x)
        rgb_feature = F.interpolate(rgb_feature, x.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(x) + shortcut + self.rgb(rgb_feature)


class RGBReflectivityGenerator(nn.Module):
    """Decode RGB identity/FPN features into reflectivity and local speckle."""

    def __init__(self, identity_dim: int = 256, geometry_dim: int = 5,
                 noise_dim: int = 32, base: int = 32) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.geometry = nn.Sequential(nn.Linear(geometry_dim, 64), nn.SiLU(),
                                      nn.Linear(64, 64), nn.SiLU())
        self.style = nn.Sequential(nn.Linear(noise_dim, 64), nn.SiLU(),
                                   nn.Linear(64, 64), nn.SiLU())
        self.input = nn.Linear(identity_dim + 128, base * 8 * 4 * 4)
        # RGBIdentityEncoder pyramid channels: 32, 64, 128, 256.
        self.decode8 = DecodeBlock(base * 8, base * 8, base * 8)
        self.decode16 = DecodeBlock(base * 8, base * 4, base * 4)
        self.decode32 = DecodeBlock(base * 4, base * 2, base * 2)
        self.decode64 = DecodeBlock(base * 2, base, base)
        self.reflectivity = nn.Conv2d(base, 1, 3, padding=1)
        self.speckle_scale = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, identity: torch.Tensor, geometry: torch.Tensor,
                pyramid: tuple[torch.Tensor, ...],
                style_noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if len(pyramid) != 4:
            raise ValueError("expected four RGB pyramid features")
        if style_noise is None:
            style_noise = identity.new_zeros(len(identity), self.noise_dim)
        condition = torch.cat((identity, self.geometry(geometry), self.style(style_noise)), 1)
        x = self.input(condition).reshape(len(identity), -1, 4, 4)
        x = self.decode8(x, pyramid[3])
        x = self.decode16(x, pyramid[2])
        x = self.decode32(x, pyramid[1])
        x = self.decode64(x, pyramid[0])
        clean = torch.tanh(self.reflectivity(x))
        # Bounded local log-amplitude scale prevents noise from becoming an
        # unconstrained class code while still allowing spatial heteroscedasticity.
        sigma = .025 + .30 * torch.sigmoid(self.speckle_scale(x))
        return clean, sigma


def sar_observation(clean: torch.Tensor, sigma: torch.Tensor,
                    noise: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable amplitude-domain multiplicative-speckle observation."""
    amplitude = (clean + 1) * .5
    if noise is None:
        noise = torch.randn_like(amplitude)
    correlated = F.avg_pool2d(noise, 3, stride=1, padding=1)
    noise = (.7 * noise + .3 * correlated) / .76
    multiplier = torch.exp(sigma * noise - .5 * sigma.square())
    observed = amplitude * multiplier
    # A weak signal-dependent receiver floor avoids an unnaturally hard black
    # background without injecting a trainable vehicle-class cue.
    receiver = .0035 * torch.tanh(noise)
    return (observed + receiver).clamp(0, 1) * 2 - 1


def highpass_view(image: torch.Tensor) -> torch.Tensor:
    residual = image - F.avg_pool2d(image, 5, stride=1, padding=2)
    return torch.tanh(3 * residual)


def spectrum_view(image: torch.Tensor) -> torch.Tensor:
    amplitude = (image + 1) * .5
    spectrum = torch.log1p(torch.fft.fftshift(torch.fft.fft2(amplitude, norm="ortho"), dim=(-2, -1)).abs())
    mean = spectrum.mean((2, 3), keepdim=True)
    std = spectrum.std((2, 3), keepdim=True).clamp_min(1e-5)
    return torch.tanh((spectrum - mean) / (2 * std))


def raw_highpass_view(image: torch.Tensor) -> torch.Tensor:
    """High-pass SAR view without non-linear saturation."""
    return image - F.avg_pool2d(image, 5, stride=1, padding=2)


def raw_spectrum_view(image: torch.Tensor) -> torch.Tensor:
    """Log Fourier magnitude retaining absolute real-domain energy."""
    amplitude = (image + 1) * .5
    return torch.log1p(torch.fft.fftshift(
        torch.fft.fft2(amplitude, norm="ortho"), dim=(-2, -1)).abs())


class MultiDomainDiscriminator(nn.Module):
    """Conditional discriminators for native, high-pass and Fourier domains."""

    def __init__(self, classes: int = 40, geometry_dim: int = 5, base: int = 32) -> None:
        super().__init__()
        self.spatial = ProjectionPatchDiscriminator(classes, geometry_dim, base)
        self.highpass = ProjectionPatchDiscriminator(classes, geometry_dim, max(16, base // 2))
        self.spectrum = ProjectionPatchDiscriminator(classes, geometry_dim, max(16, base // 2))

    def forward(self, image: torch.Tensor, class_id: torch.Tensor,
                geometry: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        spatial_score, spatial_features = self.spatial(image, class_id, geometry)
        high_score, high_features = self.highpass(highpass_view(image), class_id, geometry)
        spectrum_score, spectrum_features = self.spectrum(spectrum_view(image), class_id, geometry)
        return (spatial_score, high_score, spectrum_score), (
            spatial_features, high_features, spectrum_features)


class CalibratedMultiDomainDiscriminator(nn.Module):
    """Multi-domain D calibrated once on real SAR instead of per-image normalisation.

    The previous tanh-normalised high-pass branch saturated on artificial black
    contours.  Fixed real-domain moments preserve the magnitude information
    needed to distinguish a natural low-return background from a hard mask.
    """

    def __init__(self, highpass_mean: torch.Tensor, highpass_std: torch.Tensor,
                 spectrum_mean: torch.Tensor, spectrum_std: torch.Tensor,
                 classes: int = 40, geometry_dim: int = 5, base: int = 32) -> None:
        super().__init__()
        self.spatial = ProjectionPatchDiscriminator(classes, geometry_dim, base)
        self.highpass = ProjectionPatchDiscriminator(classes, geometry_dim, max(16, base // 2))
        self.spectrum = ProjectionPatchDiscriminator(classes, geometry_dim, max(16, base // 2))
        self.register_buffer("highpass_mean", highpass_mean.reshape(1, 1, 1, 1).float())
        self.register_buffer("highpass_std", highpass_std.reshape(1, 1, 1, 1).float().clamp_min(1e-4))
        self.register_buffer("spectrum_mean", spectrum_mean.reshape(1, 1, 1, 1).float())
        self.register_buffer("spectrum_std", spectrum_std.reshape(1, 1, 1, 1).float().clamp_min(1e-4))

    @staticmethod
    def _bounded_normalise(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # The wide clamp is only a numerical guard; unlike tanh it does not
        # collapse the gradients of ordinary SAR edges or mask boundaries.
        return ((value - mean) / std).clamp(-5, 5) / 5

    def highpass_input(self, image: torch.Tensor) -> torch.Tensor:
        return self._bounded_normalise(raw_highpass_view(image), self.highpass_mean, self.highpass_std)

    def spectrum_input(self, image: torch.Tensor) -> torch.Tensor:
        return self._bounded_normalise(raw_spectrum_view(image), self.spectrum_mean, self.spectrum_std)

    def forward(self, image: torch.Tensor, class_id: torch.Tensor,
                geometry: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        spatial_score, spatial_features = self.spatial(image, class_id, geometry)
        high_score, high_features = self.highpass(self.highpass_input(image), class_id, geometry)
        spectrum_score, spectrum_features = self.spectrum(self.spectrum_input(image), class_id, geometry)
        return (spatial_score, high_score, spectrum_score), (
            spatial_features, high_features, spectrum_features)

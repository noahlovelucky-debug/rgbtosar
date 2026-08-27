"""Dual-stage continuous RGB-to-SAR GAN v2.

The first stage queries all available RGB views with the requested SAR
azimuth/depression and injects the selected spatial evidence at every decoder
scale.  The second stage is a stochastic observation model whose full-resolution
random field reaches the output directly; it therefore cannot collapse to a
deterministic RGB-conditioned "noise" image.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from v4_spade_gan import ProjectionPatchDiscriminator, spectral


AZIMUTH_HARMONICS = 4
GEOMETRY_DIM = 16
LOG_NOISE_LIMIT = 0.8


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels)


def angle_fourier(angle_degrees: torch.Tensor,
                  harmonics: int = AZIMUTH_HARMONICS) -> torch.Tensor:
    """Periodic Fourier encoding, with sin/cos adjacent in the last axis."""
    radians = angle_degrees.float() * (math.pi / 180.0)
    frequencies = torch.arange(
        1, harmonics + 1, device=radians.device, dtype=radians.dtype)
    phase = radians[..., None] * frequencies
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


def depression_features(depression: torch.Tensor) -> torch.Tensor:
    radians = depression.float() * (math.pi / 180.0)
    return torch.stack((
        depression.float() / 60.0, radians.sin(), radians.cos()), dim=-1)


def target_geometry(azimuth: torch.Tensor, depression: torch.Tensor,
                    acquisition: torch.Tensor) -> torch.Tensor:
    """Build target geometry from azimuth, depression, and band/pol metadata.

    ``acquisition`` contains the one X/KU flag followed by four polarization
    one-hot values, matching columns 3:8 of ``bbox_data.metadata_vector``.
    """
    if acquisition.shape[-1] != 5:
        raise ValueError("acquisition must contain one band and four pol values")
    return torch.cat((
        angle_fourier(azimuth), depression_features(depression), acquisition), -1)


class MultiViewEncoding(NamedTuple):
    identity: torch.Tensor
    logits: torch.Tensor
    pyramids: tuple[torch.Tensor, ...]


class MultiViewRGBEncoder(nn.Module):
    """Shared RGB encoder that keeps view-specific maps and invariant identity."""

    def __init__(self, classes: int = 40, identity_dim: int = 512,
                 base: int = 64) -> None:
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8)
        stages, previous = [], 3
        for channel in channels:
            stages.append(nn.Sequential(
                nn.Conv2d(previous, channel, 4, 2, 1, bias=False),
                _norm(channel), nn.SiLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1, bias=False),
                _norm(channel), nn.SiLU(inplace=True)))
            previous = channel
        self.stages = nn.ModuleList(stages)
        self.view_embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels[-1], identity_dim), nn.LayerNorm(identity_dim),
            nn.SiLU())
        self.identity_refine = nn.Sequential(
            nn.Linear(identity_dim, identity_dim), nn.SiLU(),
            nn.LayerNorm(identity_dim))
        self.classifier = nn.Linear(identity_dim, classes)
        self.channels = channels

    def forward(self, views: torch.Tensor,
                view_mask: torch.Tensor) -> MultiViewEncoding:
        if views.ndim != 5:
            raise ValueError("views must have shape [batch, views, 3, H, W]")
        batch, count = views.shape[:2]
        if view_mask.shape != (batch, count):
            raise ValueError("view_mask must have shape [batch, views]")
        image = views.flatten(0, 1)
        pyramids = []
        for stage in self.stages:
            image = stage(image)
            pyramids.append(image.reshape(batch, count, *image.shape[1:]))
        per_view = self.view_embedding(image).reshape(batch, count, -1)
        mask = view_mask.to(per_view.dtype)[..., None]
        identity = (per_view * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        identity = self.identity_refine(identity)
        return MultiViewEncoding(identity, self.classifier(identity),
                                 tuple(pyramids))


class CircularViewAttention(nn.Module):
    """Target-angle query over a circular set of RGB view features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        condition_dim = AZIMUTH_HARMONICS * 2 + 3
        hidden = min(256, channels)
        self.modulation = nn.Sequential(
            nn.Linear(condition_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, channels * 2))
        self.score = nn.Sequential(
            nn.Linear(channels + condition_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    def forward(self, views: torch.Tensor, source_angles: torch.Tensor,
                view_mask: torch.Tensor, target_azimuth: torch.Tensor,
                depression: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, channels = views.shape[:3]
        relative = (
            target_azimuth[:, None].float() - source_angles.float() + 180.0
        ).remainder(360.0) - 180.0
        condition = torch.cat((
            angle_fourier(relative),
            depression_features(depression)[:, None].expand(-1, count, -1)), -1)
        scale, bias = self.modulation(condition).chunk(2, -1)
        modulated = (
            views * (1 + .25 * torch.tanh(scale)[..., None, None])
            + bias[..., None, None])
        pooled = views.mean((-2, -1))
        learned_score = self.score(torch.cat((pooled, condition), -1)).squeeze(-1)
        # A soft circular locality prior stabilises early training while leaving
        # the learned score free to select a non-nearest view when useful.
        locality = 2.0 * torch.cos(relative * (math.pi / 180.0))
        score = learned_score + locality
        score = score.masked_fill(view_mask <= 0, torch.finfo(score.dtype).min)
        weights = score.softmax(1)
        aggregate = (modulated * weights[..., None, None, None]).sum(1)
        return aggregate, weights


class GeometrySPADE(nn.Module):
    def __init__(self, channels: int, condition_channels: int,
                 geometry_dim: int = GEOMETRY_DIM) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(16, channels), channels, affine=False)
        hidden = min(128, max(32, condition_channels))
        self.spatial = nn.Sequential(
            nn.Conv2d(condition_channels, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, channels * 2, 3, padding=1))
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, channels * 2), nn.Tanh())

    def forward(self, feature: torch.Tensor, condition: torch.Tensor,
                geometry: torch.Tensor) -> torch.Tensor:
        condition = F.interpolate(
            condition, feature.shape[-2:], mode="bilinear", align_corners=False)
        spatial_scale, spatial_bias = self.spatial(condition).chunk(2, 1)
        global_scale, global_bias = self.geometry(geometry).chunk(2, 1)
        scale = spatial_scale + .25 * global_scale[..., None, None]
        bias = spatial_bias + global_bias[..., None, None]
        return self.norm(feature) * (1 + scale) + bias


class ViewModulatedDecodeBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int,
                 condition_channels: int) -> None:
        super().__init__()
        self.norm1 = GeometrySPADE(input_channels, condition_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm2 = GeometrySPADE(output_channels, condition_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels else nn.Identity())

    def forward(self, image: torch.Tensor, condition: torch.Tensor,
                geometry: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            image, scale_factor=2, mode="bilinear", align_corners=False)
        shortcut = self.skip(image)
        output = self.conv1(F.silu(
            self.norm1(image, condition, geometry), inplace=True))
        output = self.conv2(F.silu(
            self.norm2(output, condition, geometry), inplace=True))
        return output + shortcut


class MultiViewDenoisedSARGenerator(nn.Module):
    """Target-query multi-view generator with geometry at every spatial scale."""

    def __init__(self, identity_dim: int = 512, rgb_base: int = 64,
                 base: int = 64) -> None:
        super().__init__()
        self.geometry = nn.Sequential(
            nn.Linear(GEOMETRY_DIM, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU())
        self.fc = nn.Linear(identity_dim + 256, base * 8 * 4 * 4)
        view_channels = (rgb_base, rgb_base * 2, rgb_base * 4, rgb_base * 8)
        self.attention = nn.ModuleList(
            CircularViewAttention(channel) for channel in view_channels)
        decoder_channels = (base * 8, base * 8, base * 4, base * 2, base)
        self.blocks = nn.ModuleList(
            ViewModulatedDecodeBlock(
                decoder_channels[index], decoder_channels[index + 1],
                view_channels[3 - index])
            for index in range(4))
        self.refine = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            _norm(base), nn.SiLU(inplace=True),
            nn.Conv2d(base, 1, 3, padding=1))

    def forward(
            self, encoding: MultiViewEncoding, source_angles: torch.Tensor,
            view_mask: torch.Tensor, target_azimuth: torch.Tensor,
            depression: torch.Tensor,
            geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aggregate, attention_weights = [], []
        for module, pyramid in zip(self.attention, encoding.pyramids):
            selected, weights = module(
                pyramid, source_angles, view_mask, target_azimuth, depression)
            aggregate.append(selected)
            attention_weights.append(weights)
        latent = torch.cat((encoding.identity, self.geometry(geometry)), 1)
        image = self.fc(latent).reshape(len(latent), -1, 4, 4)
        for index, block in enumerate(self.blocks):
            image = block(image, aggregate[3 - index], geometry)
        clean = torch.tanh(self.refine(image))
        return clean, torch.stack(attention_weights, 1).mean(1)


class ObservationOutput(NamedTuple):
    log_multiplicative: torch.Tensor
    additive: torch.Tensor
    observed: torch.Tensor
    parameters: torch.Tensor


class StochasticSARObservation(nn.Module):
    """Non-collapsible empirical/physics-inspired 8-bit SAR observation model.

    A two-channel full-resolution random field is always used directly.  The
    learnable network only chooses bounded acquisition parameters; it cannot
    redraw the vehicle or turn the stochastic field off.
    """

    random_channels = 2

    def __init__(self, minimum_log_scale: float = .08) -> None:
        super().__init__()
        self.minimum_log_scale = minimum_log_scale
        # depression features (3) + detached clean mean/std (2)
        self.parameter_net = nn.Sequential(
            nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
            # 5 correlation bases, log-scale, skew, additive scale
            nn.Linear(64, 8))

    @staticmethod
    def _standardize(field: torch.Tensor) -> torch.Tensor:
        return ((field - field.mean((2, 3), keepdim=True))
                / field.std((2, 3), keepdim=True).clamp_min(1e-4))

    def forward(self, clean: torch.Tensor, depression: torch.Tensor,
                random_field: torch.Tensor | None = None) -> ObservationOutput:
        batch, _, height, width = clean.shape
        if random_field is None:
            random_field = torch.randn(
                batch, self.random_channels, height, width,
                device=clean.device, dtype=clean.dtype)
        expected = (batch, self.random_channels, height, width)
        if tuple(random_field.shape) != expected:
            raise ValueError(
                f"random_field must have shape {expected}, got {tuple(random_field.shape)}")
        amplitude = ((clean + 1) * .5).clamp(1e-4, 1)
        clean_stats = torch.cat((
            amplitude.detach().mean((2, 3)),
            amplitude.detach().std((2, 3))), 1)
        condition = torch.cat((
            depression_features(depression), clean_stats), 1)
        raw = self.parameter_net(condition)
        basis_logits, raw_scale, raw_skew, raw_additive = (
            raw[:, :5], raw[:, 5:6], raw[:, 6:7], raw[:, 7:8])

        primary = random_field[:, :1]
        bases = (
            primary,
            F.avg_pool2d(primary, 3, 1, 1),
            F.avg_pool2d(primary, 5, 1, 2),
            F.avg_pool2d(primary, (1, 5), 1, (0, 2)),
            F.avg_pool2d(primary, (5, 1), 1, (2, 0)),
        )
        bases = torch.stack(
            tuple(self._standardize(field) for field in bases), 1)
        weights = basis_logits.softmax(1)[..., None, None, None]
        correlated = self._standardize((bases * weights).sum(1))
        skew = .20 * torch.tanh(raw_skew)[..., None, None]
        correlated = correlated + skew * (correlated.square() - 1)
        correlated = self._standardize(correlated)

        log_scale = (
            self.minimum_log_scale + .27 * torch.sigmoid(raw_scale)
        )[..., None, None]
        # The log-normal correction preserves unit expected multiplicative gain.
        log_multiplicative = (
            log_scale * correlated - .5 * log_scale.square()
        ).clamp(-LOG_NOISE_LIMIT, LOG_NOISE_LIMIT)
        additive_scale = (
            .002 + .023 * torch.sigmoid(raw_additive)
        )[..., None, None]
        additive = additive_scale * self._standardize(random_field[:, 1:2])
        observed = compose_observation(clean, log_multiplicative, additive)
        parameters = torch.cat((
            weights[..., 0, 0, 0], log_scale[..., 0, 0],
            skew[..., 0, 0], additive_scale[..., 0, 0]), 1)
        return ObservationOutput(
            log_multiplicative, additive, observed, parameters)


def compose_observation(clean: torch.Tensor, log_multiplicative: torch.Tensor,
                        additive: torch.Tensor) -> torch.Tensor:
    amplitude = ((clean + 1) * .5).clamp(1e-4, 1)
    observed = amplitude * torch.exp(log_multiplicative) + additive
    return observed.clamp(0, 1) * 2 - 1


def residual_view(clean: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    clean_amplitude = ((clean + 1) * .5).clamp(1e-4, 1)
    observed_amplitude = ((observed + 1) * .5).clamp(1e-4, 1)
    residual = (
        observed_amplitude.log() - clean_amplitude.log()
    ).clamp(-LOG_NOISE_LIMIT, LOG_NOISE_LIMIT)
    return residual / LOG_NOISE_LIMIT


class AcquisitionPatchDiscriminator(nn.Module):
    """Noise critic conditioned only on depression, never vehicle identity."""

    def __init__(self, base: int = 64) -> None:
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8)
        layers, previous = [], 1
        for channel in channels:
            layers.extend((
                spectral(nn.Conv2d(previous, channel, 4, 2, 1)),
                nn.LeakyReLU(.2, inplace=True)))
            previous = channel
        self.features = nn.Sequential(*layers)
        self.score = spectral(nn.Conv2d(channels[-1], 1, 3, padding=1))
        self.condition = nn.Sequential(
            nn.Linear(3, channels[-1]), nn.SiLU(),
            nn.Linear(channels[-1], channels[-1]))

    def forward(self, residual: torch.Tensor,
                depression: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(residual)
        condition = self.condition(depression_features(depression))
        projection = (features * condition[..., None, None]).sum(
            1, keepdim=True)
        return (self.score(features) + projection).mean((1, 2, 3)), features


class NoiseLeakageClassifier(nn.Module):
    """Auditor/adversary for vehicle information leaked into the residual."""

    def __init__(self, classes: int = 40, base: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, 2, 1), _norm(base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), _norm(base * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(base * 4, classes)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(residual))


class DualComponentDiscriminatorsV2(nn.Module):
    def __init__(self, classes: int = 40, base: int = 64) -> None:
        super().__init__()
        self.clean = ProjectionPatchDiscriminator(classes, GEOMETRY_DIM, base)
        self.noise = AcquisitionPatchDiscriminator(base)
        self.full = ProjectionPatchDiscriminator(classes, GEOMETRY_DIM, base)


def initialise(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

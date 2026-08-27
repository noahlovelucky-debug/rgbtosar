"""Continuous Spatial V3: one-stage, physically stochastic RGB-to-SAR GAN.

The public generator produces one observed SAR image.  A latent reflectivity
image is used only inside the observation equation; it is never supervised
against an artificial "denoised SAR" target.  Likewise, the stochastic path is
an analytic, bounded multiplicative-speckle/receiver-noise layer rather than a
second neural generator.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from joint_models import RGBIdentityEncoder, SpatialROIGenerator
from v4_spade_gan import spectral


ARCHITECTURE = "continuous_spatial_one_stage_v3"
GEOMETRY_DIM = 12
DEPRESSION_VALUES = (15.0, 30.0, 45.0, 60.0)
RAYLEIGH_MEAN = math.sqrt(math.pi / 2.0)


def _standardize(image: torch.Tensor) -> torch.Tensor:
    return ((image - image.mean((2, 3), keepdim=True))
            / image.std((2, 3), keepdim=True).clamp_min(1e-4))


def circular_delta(target: torch.Tensor,
                   source: torch.Tensor) -> torch.Tensor:
    return (target[..., None] - source + 180.0).remainder(360.0) - 180.0


def target_geometry(
        metadata: torch.Tensor, azimuth: torch.Tensor,
        depression: torch.Tensor) -> torch.Tensor:
    """Build the V1-compatible 12-D condition without target-box leakage."""
    if metadata.shape[-1] != 10:
        raise ValueError("metadata must contain ten bbox_data features")
    radians = azimuth.float() * (math.pi / 180.0)
    result = metadata.clone()
    result[:, 0] = radians.sin()
    result[:, 1] = radians.cos()
    result[:, 2] = depression.float() / 60.0
    result[:, -2:] = 0.0
    # V1 used the selected source-view angle here.  With a continuous
    # multi-view query, the target direction is the stable limiting value.
    return torch.cat((
        result, radians.sin()[:, None], radians.cos()[:, None]), 1)


class MultiViewEncoding(NamedTuple):
    identity: torch.Tensor
    logits: torch.Tensor
    pyramids: tuple[torch.Tensor, ...]
    per_view_identity: torch.Tensor
    pooled_pyramids: tuple[torch.Tensor, ...]
    canonical_complete: bool


class MultiViewV1Encoder(RGBIdentityEncoder):
    """V1-compatible shared encoder for a masked set of twelve RGB views."""

    def forward(self, views: torch.Tensor,
                view_mask: torch.Tensor) -> MultiViewEncoding:
        if views.ndim != 5:
            raise ValueError("views must be [batch, views, 3, height, width]")
        batch, count = views.shape[:2]
        if view_mask.shape != (batch, count):
            raise ValueError("view_mask must be [batch, views]")
        identity, _, pyramid = super().forward(
            views.flatten(0, 1), return_pyramid=True)
        per_view = identity.reshape(batch, count, -1)
        mask = view_mask.to(per_view.dtype)[..., None]
        invariant = (
            (per_view * mask).sum(1)
            / mask.sum(1).clamp_min(1.0))
        pyramids = tuple(
            feature.reshape(batch, count, *feature.shape[1:])
            for feature in pyramid)
        return MultiViewEncoding(
            invariant, self.classifier(invariant), pyramids, per_view,
            tuple(feature.mean((-2, -1)) for feature in pyramids),
            bool(views.shape[1] == 12
                 and (view_mask > 0).all()) if not self.training else False)


class CircularScaleAttention(nn.Module):
    """Fast continuous circular query with learned interpolation sharpness."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        del channels
        # softplus(4.993) ~= 5: initially blend the nearest circular pair.
        self.raw_temperature = nn.Parameter(torch.tensor(4.993))

    def forward(
            self, views: torch.Tensor, view_angles: torch.Tensor,
            view_mask: torch.Tensor, target_azimuth: torch.Tensor,
            depression: torch.Tensor,
            pooled_views: torch.Tensor | None = None,
            canonical_complete: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor]:
        del depression, pooled_views
        # Only the two best circular neighbours contribute spatial maps.
        # Sparse gathering avoids multiplying twelve full 64px feature maps
        # for every target.
        count = min(2, views.shape[1])
        # Deployment data normally has the canonical complete 0:30:330 set.
        # Avoid top-k kernel overhead for this common cached-feature path.
        if not self.training and canonical_complete:
            position = target_azimuth.remainder(360.0) / 30.0
            lower = position.floor().long().remainder(12)
            upper = (lower + 1).remainder(12)
            indices = torch.stack((lower, upper), 1)
            fraction = position - position.floor()
            selected_weights = torch.stack(
                (1.0 - fraction, fraction), 1).to(views.dtype)
        else:
            relative = circular_delta(target_azimuth, view_angles)
            radians = relative * (math.pi / 180.0)
            temperature = F.softplus(
                self.raw_temperature).clamp(2.0, 12.0)
            logits = temperature * torch.cos(radians)
            logits = logits.masked_fill(
                view_mask <= 0, torch.finfo(logits.dtype).min)
            selected_logits, indices = logits.topk(count, dim=1)
            selected_weights = selected_logits.softmax(1)
        gather_index = indices[..., None, None, None].expand(
            -1, -1, views.shape[2], views.shape[3], views.shape[4])
        candidates = views.gather(1, gather_index)
        selected = (
            candidates
            * selected_weights[..., None, None, None]).sum(1)
        weights = torch.zeros(
            views.shape[:2], device=views.device,
            dtype=views.dtype).scatter(
            1, indices, selected_weights)
        return selected, weights


class MultiScaleGeometryAffine(nn.Module):
    """One zero-init projection supplying modulation bias to all scales."""

    def __init__(self, channels: tuple[int, ...],
                 geometry_dim: int = GEOMETRY_DIM) -> None:
        super().__init__()
        self.channels = channels
        self.affine = nn.Linear(geometry_dim, sum(channels))
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(
            self, geometry: torch.Tensor
            ) -> tuple[torch.Tensor, ...]:
        return self.affine(geometry).split(self.channels, 1)

    @staticmethod
    def apply(image: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return image + .20 * bias[..., None, None]


class AntiAliasBlend(nn.Module):
    """Fixed 3x3 low-pass filter with a scheduled, non-learned blend."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("strength", torch.zeros(()))

    def set_strength(self, value: float) -> None:
        self.strength.fill_(max(0.0, min(1.0, value)))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if float(self.strength) == 0.0:
            return image
        filtered = F.avg_pool2d(
            image, 3, stride=1, padding=1,
            count_include_pad=False)
        return image.lerp(
            filtered.to(image.dtype), self.strength.to(image.dtype))


class GenerationOutput(NamedTuple):
    sar: torch.Tensor
    base: torch.Tensor
    whitened_noise: torch.Tensor
    sigma: torch.Tensor
    receiver_scale: torch.Tensor
    attention: torch.Tensor


class OneStageContinuousSARGenerator(SpatialROIGenerator):
    """V1 decoder with multi-view queries and a bounded observation equation."""

    random_channels = 3

    def __init__(self, identity_dim: int = 256,
                 geometry_dim: int = GEOMETRY_DIM,
                 base: int = 32) -> None:
        # V1 uses the same 12-dimensional geometry vector.
        super().__init__(
            identity_dim=identity_dim, meta_dim=geometry_dim, base=base)
        view_channels = (base, base * 2, base * 4, base * 8)
        output_channels = (
            base * 4, base * 2, base, max(16, base // 2))
        # A single semantic query at the deepest scale supplies one
        # geometrically consistent pair of views to every decoder scale.
        self.view_attention = CircularScaleAttention(view_channels[-1])
        self.geometry_affine = MultiScaleGeometryAffine(
            output_channels, geometry_dim)
        self.antialias = nn.ModuleList(AntiAliasBlend() for _ in range(4))
        # Four depression anchors, linearly interpolated for intermediate d.
        # Start close to the lower physical bounds so a new random seed changes
        # fine texture before it has any measurable effect on low-frequency
        # vehicle structure.  Training may increase either value when real-SAR
        # distribution losses require it.
        initial_sigma = math.log(.0125 / .9875)  # sigma = 0.061
        initial_receiver = math.log(.03125 / .96875)  # receiver = 0.00225
        self.raw_sigma = nn.Parameter(
            torch.full((4,), initial_sigma))
        self.raw_receiver = nn.Parameter(
            torch.full((4,), initial_receiver))

    def set_antialias_strength(self, value: float) -> None:
        for module in self.antialias:
            module.set_strength(value)

    @staticmethod
    def _interpolate_parameter(
            values: torch.Tensor,
            depression: torch.Tensor) -> torch.Tensor:
        position = ((depression - 15.0) / 15.0).clamp(0.0, 3.0)
        lower = position.floor().long()
        upper = (lower + 1).clamp_max(3)
        fraction = position - lower
        return values[lower].lerp(values[upper], fraction)

    def observation_parameters(
            self, depression: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_values = .06 + .08 * self.raw_sigma.sigmoid()
        receiver_values = .002 + .008 * self.raw_receiver.sigmoid()
        sigma = self._interpolate_parameter(
            sigma_values, depression.float())[:, None, None, None]
        receiver = self._interpolate_parameter(
            receiver_values, depression.float())[:, None, None, None]
        return sigma, receiver

    def select_views(
            self, encoding: MultiViewEncoding,
            view_angles: torch.Tensor, view_mask: torch.Tensor,
            target_azimuth: torch.Tensor,
            depression: torch.Tensor
            ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        _, weights = self.view_attention(
            encoding.pyramids[-1], view_angles, view_mask,
            target_azimuth, depression, encoding.pooled_pyramids[-1],
            encoding.canonical_complete)
        count = min(2, weights.shape[1])
        selected_weights, indices = weights.topk(count, 1)
        selected_weights = (
            selected_weights
            / selected_weights.sum(1, keepdim=True).clamp_min(1e-6))
        selected = []
        for pyramid in encoding.pyramids:
            gather_index = indices[..., None, None, None].expand(
                -1, -1, pyramid.shape[2],
                pyramid.shape[3], pyramid.shape[4])
            candidates = pyramid.gather(1, gather_index)
            selected.append((
                candidates
                * selected_weights[..., None, None, None]).sum(1))
        return tuple(selected), weights

    def base_amplitude(
            self, encoding: MultiViewEncoding,
            view_angles: torch.Tensor, view_mask: torch.Tensor,
            target_azimuth: torch.Tensor, depression: torch.Tensor,
            geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if geometry.shape[-1] != GEOMETRY_DIM:
            raise ValueError(
                f"geometry must have {GEOMETRY_DIM} features")
        selected, attention = self.select_views(
            encoding, view_angles, view_mask,
            target_azimuth, depression)
        latent = torch.cat((encoding.identity, self.meta(geometry)), 1)
        image = self.fc(latent).reshape(
            -1, self.fc.out_features // 16, 4, 4)
        geometry_values = self.geometry_affine(geometry)
        for index in range(4):
            block = self.net[index * 7:(index + 1) * 7]
            image = block[0](image)
            for layer in block[1:]:
                image = layer(image)
            condition = F.interpolate(
                selected[3 - index], image.shape[-2:],
                mode="bilinear", align_corners=False)
            image = image + self.spatial_projection[index](condition)
            image = self.geometry_affine.apply(
                image, geometry_values[index])
        base = self.net[28:](image)
        # Filtering the final one-channel reflectivity is materially cheaper
        # than filtering a wide decoder map and directly suppresses output
        # aliasing before stochastic observation.
        base = self.antialias[3](base)
        return base, attention

    def observe(
            self, base: torch.Tensor, depression: torch.Tensor,
            random_field: torch.Tensor,
            stochastic_scale: float = 1.0
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (
            len(base), self.random_channels,
            base.shape[-2], base.shape[-1])
        if tuple(random_field.shape) != expected:
            raise ValueError(
                f"random_field must have shape {expected}, "
                f"got {tuple(random_field.shape)}")
        primary = random_field[:, :1]
        correlated = F.avg_pool2d(primary, 3, 1, 1)
        # For an i.i.d. N(0,1) field this fixed factor gives unit variance for
        # .85*z + .15*mean3x3(z), avoiding per-image reductions at inference.
        epsilon = (.85 * primary + .15 * correlated) / .868
        sigma, receiver = self.observation_parameters(depression)
        scale = max(.0, min(1.0, float(stochastic_scale)))
        # Warm-up may reduce stochasticity, but a nonzero path remains.
        effective_sigma = sigma * (.25 + .75 * scale)
        effective_receiver = receiver * (.25 + .75 * scale)
        amplitude = ((base + 1.0) * .5).clamp(1e-4, 1.0)
        multiplier = torch.exp(
            effective_sigma * epsilon
            - .5 * effective_sigma.square())
        rayleigh = torch.sqrt(
            random_field[:, 1:2].square()
            + random_field[:, 2:3].square() + 1e-8) / RAYLEIGH_MEAN
        observed = (
            amplitude * multiplier
            + effective_receiver * rayleigh).clamp(0.0, 1.0)
        return (
            observed * 2.0 - 1.0, epsilon,
            effective_sigma, effective_receiver)

    def forward(
            self, encoding: MultiViewEncoding,
            view_angles: torch.Tensor, view_mask: torch.Tensor,
            target_azimuth: torch.Tensor, depression: torch.Tensor,
            geometry: torch.Tensor, random_field: torch.Tensor,
            stochastic_scale: float = 1.0) -> GenerationOutput:
        base, attention = self.base_amplitude(
            encoding, view_angles, view_mask,
            target_azimuth, depression, geometry)
        sar, epsilon, sigma, receiver = self.observe(
            base, depression, random_field, stochastic_scale)
        return GenerationOutput(
            sar, base, epsilon, sigma, receiver, attention)


class DiscriminatorOutput(NamedTuple):
    score: torch.Tensor
    features: tuple[torch.Tensor, ...]


class OneStageConditionalDiscriminator(nn.Module):
    """One shared critic over local, global and Fourier observations."""

    def __init__(self, classes: int = 40,
                 geometry_dim: int = GEOMETRY_DIM,
                 base: int = 32) -> None:
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8)
        layers, previous = [], 1
        for channel in channels:
            layers.extend((
                spectral(nn.Conv2d(previous, channel, 4, 2, 1)),
                nn.LeakyReLU(.2, inplace=True)))
            previous = channel
        self.shared = nn.Sequential(*layers)
        self.patch = spectral(nn.Conv2d(channels[-1], 1, 3, padding=1))
        self.global_score = spectral(nn.Linear(channels[-1], 1))
        self.class_embedding = nn.Embedding(classes, channels[-1])
        self.geometry = nn.Sequential(
            spectral(nn.Linear(geometry_dim, channels[-1])),
            nn.LeakyReLU(.2, inplace=True),
            spectral(nn.Linear(channels[-1], channels[-1])))
        self.branch_logits = nn.Parameter(torch.zeros(3))

    @staticmethod
    def fourier_image(image: torch.Tensor) -> torch.Tensor:
        amplitude = (image + 1.0) * .5
        spectrum = torch.log1p(torch.fft.fftshift(
            torch.fft.fft2(amplitude.float(), norm="ortho"),
            dim=(-2, -1)).abs())
        spectrum = spectrum / spectrum.amax(
            (2, 3), keepdim=True).clamp_min(1e-5)
        return spectrum.to(image.dtype) * 2.0 - 1.0

    def _branch(
            self, image: torch.Tensor, labels: torch.Tensor,
            geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.shared(image)
        pooled = feature.mean((2, 3))
        condition = self.class_embedding(labels) + self.geometry(geometry)
        projection = (pooled * condition).sum(1) / math.sqrt(pooled.shape[1])
        score = (
            self.patch(feature).flatten(1).mean(1)
            + self.global_score(pooled).squeeze(1)
            + projection)
        return score, feature

    def forward(
            self, image: torch.Tensor, labels: torch.Tensor,
            geometry: torch.Tensor) -> DiscriminatorOutput:
        inputs = (
            image,
            F.avg_pool2d(image, 2),
            self.fourier_image(image))
        branches = [
            self._branch(branch, labels, geometry)
            for branch in inputs]
        weights = self.branch_logits.softmax(0)
        score = sum(
            weight * branch[0]
            for weight, branch in zip(weights, branches))
        return DiscriminatorOutput(
            score, tuple(branch[1] for branch in branches))


class ContinuousSpatialOneStageV3(nn.Module):
    """Inference wrapper exposing the requested one-call ``generate`` API."""

    def __init__(self, classes: int = 40) -> None:
        super().__init__()
        self.encoder = MultiViewV1Encoder(classes)
        self.generator = OneStageContinuousSARGenerator()

    def encode(self, rgb_views: torch.Tensor,
               view_mask: torch.Tensor) -> MultiViewEncoding:
        return self.encoder(rgb_views, view_mask)

    def generate(
            self, rgb_views: torch.Tensor, view_angles: torch.Tensor,
            view_mask: torch.Tensor, azimuth: torch.Tensor,
            depression: torch.Tensor, geometry: torch.Tensor,
            seed: int) -> torch.Tensor:
        encoding = self.encode(rgb_views, view_mask)
        random = torch.Generator(device=rgb_views.device)
        random.manual_seed(int(seed))
        field = torch.randn(
            len(rgb_views), self.generator.random_channels, 64, 64,
            device=rgb_views.device, dtype=rgb_views.dtype,
            generator=random)
        return self.generator(
            encoding, view_angles, view_mask, azimuth,
            depression, geometry, field).sar

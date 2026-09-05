"""Silhouette-conditioned unpaired SAR bridge diffusion.

This module adapts the *training idea* of UNSB (a stochastic neural bridge
between unpaired image domains) to the RGB/SAR setting in this repository.  It
does not copy UNSB's CycleGAN data assumptions: RGB and SAR are joined only by
class and acquisition metadata, never by pixel coordinates.

The bridge is trained with a conditional flow-matching objective.  A random
real SAR ROI is an endpoint sample from the target distribution and the RGB
silhouette supplies the source-side spatial condition.  At inference the model
integrates a short velocity field from a silhouette prior to a SAR sample.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


UNSB_SAR_BRIDGE_ARCHITECTURE = "unsb_sar_silhouette_bridge64_v1"
UNSB_SAR_UNPAIRED_ARCHITECTURE = "unsb_sar_silhouette_bridge64_gde_v1"
CONDITION_DIM = 12
CLASS_COUNT = 40
SOURCE_ANGLE_DIM = 2


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_embedding(value: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    if half == 0:
        return value.float()[:, None]
    frequencies = torch.exp(
        torch.arange(half, device=value.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    phases = value.float()[:, None] * frequencies[None]
    result = torch.cat((phases.sin(), phases.cos()), dim=1)
    return F.pad(result, (0, 1)) if dimension % 2 else result


class BridgeCondition(NamedTuple):
    """Cached RGB/acquisition features used by the bridge U-Net."""

    token: torch.Tensor
    controls: tuple[torch.Tensor, ...]
    primary_logits: torch.Tensor
    alternate_logits: torch.Tensor | None
    primary_token: torch.Tensor
    alternate_token: torch.Tensor | None
    prior: torch.Tensor
    source_angle: torch.Tensor


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(_groups(out_channels), out_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(image)))


class SpatialControlEncoder(nn.Module):
    """Extract global identity and 64/32/16/8 spatial RGB-silhouette controls."""

    def __init__(self, token_dim: int = 256, base: int = 32) -> None:
        super().__init__()
        channels = (base * 2, base * 4, base * 8, base * 16)
        self.stem = ConvNormAct(4, base, stride=2)
        self.stage64 = nn.Sequential(ConvNormAct(base, channels[0]),
                                      ConvNormAct(channels[0], channels[0]))
        self.down32 = ConvNormAct(channels[0], channels[1], stride=2)
        self.stage32 = nn.Sequential(ConvNormAct(channels[1], channels[1]),
                                     ConvNormAct(channels[1], channels[1]))
        self.down16 = ConvNormAct(channels[1], channels[2], stride=2)
        self.stage16 = nn.Sequential(ConvNormAct(channels[2], channels[2]),
                                     ConvNormAct(channels[2], channels[2]))
        self.down8 = ConvNormAct(channels[2], channels[3], stride=2)
        self.stage8 = nn.Sequential(ConvNormAct(channels[3], channels[3]),
                                    ConvNormAct(channels[3], channels[3]))
        self.token = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels[3], token_dim), nn.LayerNorm(token_dim), nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.classifier = nn.Linear(token_dim, CLASS_COUNT)

    def forward(self, rgb: torch.Tensor, mask: torch.Tensor) -> tuple[
        torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor
    ]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}")
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"mask must be [B,1,H,W], got {tuple(mask.shape)}")
        if rgb.shape[-2:] != mask.shape[-2:]:
            raise ValueError("RGB and alpha mask must have the same spatial size")
        if rgb.shape[-2:] != (128, 128):
            rgb = F.interpolate(rgb, (128, 128), mode="bilinear", align_corners=False)
            mask = F.interpolate(mask, (128, 128), mode="bilinear", align_corners=False)
        image = self.stem(torch.cat((rgb, mask.clamp(0, 1)), dim=1))
        control64 = self.stage64(image)
        control32 = self.stage32(self.down32(control64))
        control16 = self.stage16(self.down16(control32))
        control8 = self.stage8(self.down8(control16))
        token = self.token(control8)
        return token, (control64, control32, control16, control8), self.classifier(token)


class SpatialCrossAttention(nn.Module):
    """Cross-attend a SAR feature map to the RGB spatial control at one scale."""

    def __init__(self, channels: int, control_channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("attention channels must be divisible by heads")
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Linear(control_channels, channels, bias=False)
        self.value = nn.Linear(control_channels, channels, bias=False)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, image: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        if control.shape[-2:] != (height, width):
            control = F.interpolate(control, (height, width), mode="bilinear", align_corners=False)
        query = self.query(self.norm(image)).flatten(2).transpose(1, 2)
        tokens = control.flatten(2).transpose(1, 2)
        key, value = self.key(tokens), self.value(tokens)
        attended, _ = self.attention(query, key, value, need_weights=False)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return image + self.output(attended)


class ConditionedResidualBlock(nn.Module):
    """Residual block with time/acquisition FiLM and zero-start spatial control."""

    def __init__(self, in_channels: int, out_channels: int, context_dim: int,
                 control_channels: int | None = None) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, out_channels * 2))
        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())
        self.control = (nn.Conv2d(control_channels, out_channels, 1)
                        if control_channels is not None else None)
        if self.control is not None:
            nn.init.zeros_(self.control.weight)
            nn.init.zeros_(self.control.bias)

    def forward(self, image: torch.Tensor, context: torch.Tensor,
                control: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(image)))
        scale, bias = self.film(context).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        if self.control is not None and control is not None:
            hidden = hidden + self.control(control)
        return hidden + self.skip(image)


class BridgeUNet(nn.Module):
    """64px velocity U-Net with multi-scale RGB spatial control."""

    def __init__(self, base: int = 64, context_dim: int = 256,
                 control_base: int = 32) -> None:
        super().__init__()
        control_channels = (control_base * 2, control_base * 4,
                            control_base * 8, control_base * 16)
        channels = (base, base * 2, base * 4, base * 8)
        self.time = nn.Sequential(nn.Linear(context_dim, context_dim * 2), nn.SiLU(),
                                  nn.Linear(context_dim * 2, context_dim))
        # target acquisition + four target-angle harmonics + source/relative
        # angle sin/cos.  The source angle is metadata, not an RGB/SAR pixel
        # correspondence assumption.
        self.acquisition = nn.Sequential(nn.Linear(CONDITION_DIM + 12, context_dim), nn.SiLU(),
                                         nn.Linear(context_dim, context_dim))
        self.context = nn.Sequential(nn.Linear(context_dim * 3, context_dim * 2), nn.SiLU(),
                                     nn.Linear(context_dim * 2, context_dim))
        self.input = nn.Conv2d(1, channels[0], 3, padding=1)
        self.block64 = ConditionedResidualBlock(channels[0], channels[0], context_dim,
                                                control_channels[0])
        self.down32 = nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1)
        self.block32 = ConditionedResidualBlock(channels[1], channels[1], context_dim,
                                                control_channels[1])
        self.down16 = nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1)
        self.block16 = ConditionedResidualBlock(channels[2], channels[2], context_dim,
                                                control_channels[2])
        self.down8 = nn.Conv2d(channels[2], channels[3], 4, stride=2, padding=1)
        self.block8 = ConditionedResidualBlock(channels[3], channels[3], context_dim,
                                               control_channels[3])
        self.attn16 = SpatialCrossAttention(channels[2], control_channels[2], heads=4)
        self.attn8 = SpatialCrossAttention(channels[3], control_channels[3], heads=8)
        self.up16 = nn.Conv2d(channels[3], channels[2], 3, padding=1)
        self.up_block16 = ConditionedResidualBlock(channels[2] * 2, channels[2], context_dim,
                                                   control_channels[2])
        self.up16_attn = SpatialCrossAttention(channels[2], control_channels[2], heads=4)
        self.up8 = nn.Conv2d(channels[2], channels[1], 3, padding=1)
        self.up_block32 = ConditionedResidualBlock(channels[1] * 2, channels[1], context_dim,
                                                   control_channels[1])
        self.up32_attn = SpatialCrossAttention(channels[1], control_channels[1], heads=4)
        self.up4 = nn.Conv2d(channels[1], channels[0], 3, padding=1)
        self.up_block64 = ConditionedResidualBlock(channels[0] * 2, channels[0], context_dim,
                                                   control_channels[0])
        self.output = nn.Sequential(nn.GroupNorm(_groups(channels[0]), channels[0]), nn.SiLU(),
                                    nn.Conv2d(channels[0], 1, 3, padding=1))
        # A zero-start velocity makes the first bridge steps stable and lets
        # the data term decide the direction before the model adds texture.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _angle_features(acquisition: torch.Tensor, source_angle: torch.Tensor) -> torch.Tensor:
        angle = torch.atan2(acquisition[:, 0], acquisition[:, 1])
        harmonics = torch.cat([torch.stack((torch.sin(k * angle), torch.cos(k * angle)), dim=1)
                               for k in range(1, 5)], dim=1)
        if source_angle.shape != (len(acquisition), SOURCE_ANGLE_DIM):
            raise ValueError("source_angle must be [B,2] sin/cos")
        source = source_angle / source_angle.norm(dim=1, keepdim=True).clamp_min(1e-6)
        source_theta = torch.atan2(source[:, 0], source[:, 1])
        delta = angle - source_theta
        relative = torch.stack((delta.sin(), delta.cos()), dim=1)
        return torch.cat((acquisition, harmonics, source, relative), dim=1)

    def _context(self, timestep: torch.Tensor, token: torch.Tensor,
                 acquisition: torch.Tensor, source_angle: torch.Tensor) -> torch.Tensor:
        time = self.time(sinusoidal_embedding(timestep, token.shape[1]))
        acq = self.acquisition(self._angle_features(acquisition, source_angle))
        return self.context(torch.cat((time, token, acq), dim=1))

    @staticmethod
    def _resize(image: torch.Tensor, convolution: nn.Conv2d) -> torch.Tensor:
        return convolution(F.interpolate(image, scale_factor=2, mode="bilinear", align_corners=False))

    def forward(self, state: torch.Tensor, timestep: torch.Tensor, token: torch.Tensor,
                acquisition: torch.Tensor, controls: tuple[torch.Tensor, ...],
                source_angle: torch.Tensor) -> torch.Tensor:
        if state.shape[1:] != (1, 64, 64):
            raise ValueError(f"bridge state must be [B,1,64,64], got {tuple(state.shape)}")
        context = self._context(timestep, token, acquisition, source_angle)
        control64, control32, control16, control8 = controls
        skip64 = self.block64(self.input(state), context, control64)
        skip32 = self.block32(self.down32(skip64), context, control32)
        skip16 = self.block16(self.down16(skip32), context, control16)
        image = self.block8(self.down8(skip16), context, control8)
        image = self.attn8(image, control8)
        image = self._resize(image, self.up16)
        image = self.attn16(image, control16)
        image = self.up_block16(torch.cat((image, skip16), dim=1), context, control16)
        image = self.up16_attn(image, control16)
        image = self._resize(image, self.up8)
        image = self.up_block32(torch.cat((image, skip32), dim=1), context, control32)
        image = self.up32_attn(image, control32)
        image = self._resize(image, self.up4)
        image = self.up_block64(torch.cat((image, skip64), dim=1), context, control64)
        return self.output(image)


def soft_silhouette_prior(mask: torch.Tensor, blur_kernel: int = 15) -> torch.Tensor:
    """Create a same-grid soft signed-occupancy source for the bridge.

    The alpha mask is the only spatial RGB/SAR signal.  A broad occupancy
    field, rather than RGB pixels or a learned class map, gives the bridge a
    stable geometric endpoint while leaving scattering appearance to the
    target-domain transport.
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [B,1,H,W]")
    mask = F.interpolate(mask.float().clamp(0, 1), (64, 64), mode="bilinear", align_corners=False)
    padding = blur_kernel // 2
    occupancy = F.avg_pool2d(mask, blur_kernel, stride=1, padding=padding)
    return (occupancy * 2.0 - 1.0) * .35


class SilhouetteBridge(nn.Module):
    """Complete RGB/silhouette -> conditional SAR stochastic bridge."""

    def __init__(self, base: int = 64, token_dim: int = 256,
                 control_base: int = 32) -> None:
        super().__init__()
        self.encoder = SpatialControlEncoder(token_dim=token_dim, base=control_base)
        self.unet = BridgeUNet(base=base, context_dim=token_dim, control_base=control_base)

    def encode_conditions(self, rgb: torch.Tensor, mask: torch.Tensor,
                          acquisition: torch.Tensor, source_angle: torch.Tensor | None = None,
                          rgb_alt: torch.Tensor | None = None,
                          mask_alt: torch.Tensor | None = None) -> BridgeCondition:
        primary_token, controls, primary_logits = self.encoder(rgb, mask)
        if source_angle is None:
            source_angle = torch.zeros((len(rgb), SOURCE_ANGLE_DIM), device=rgb.device,
                                       dtype=rgb.dtype)
            source_angle[:, 1] = 1.0
        if source_angle.shape != (len(rgb), SOURCE_ANGLE_DIM):
            raise ValueError("source_angle must be [B,2] sin/cos")
        alternate_token = None
        alternate_logits = None
        token = primary_token
        if rgb_alt is not None:
            if mask_alt is None:
                raise ValueError("mask_alt is required when rgb_alt is provided")
            alternate_token, alternate_logits = self.encoder(rgb_alt, mask_alt)[0::2]
            # The second view contributes only global identity evidence.  Its
            # spatial map is deliberately excluded to avoid fake pixel pairing.
            token = .5 * (primary_token + alternate_token)
        # Do not feed a 40-way posterior to the SAR generator.  The logits are
        # an RGB-only identity audit; the bridge sees the invariant token and
        # the explicit spatial silhouette, which prevents a frozen SAR
        # classifier from becoming a shortcut condition.
        prior = soft_silhouette_prior(mask)
        return BridgeCondition(token, controls, primary_logits, alternate_logits,
                               primary_token, alternate_token, prior, source_angle)

    def predict(self, state: torch.Tensor, timestep: torch.Tensor,
                acquisition: torch.Tensor, conditions: BridgeCondition) -> torch.Tensor:
        return self.unet(state, timestep, conditions.token, acquisition,
                         conditions.controls, conditions.source_angle)

    def forward(self, state: torch.Tensor, timestep: torch.Tensor, rgb: torch.Tensor,
                mask: torch.Tensor, acquisition: torch.Tensor,
                source_angle: torch.Tensor | None = None,
                rgb_alt: torch.Tensor | None = None,
                mask_alt: torch.Tensor | None = None,
                return_conditions: bool = False, rollout_steps: int = 0,
                rollout_noise: float = 0.0,
                conditions: BridgeCondition | None = None):
        # Conditions can be supplied by the trainer when several independent
        # rollouts share the same RGB/acquisition context.  This avoids
        # recomputing the RGB pyramid while keeping every rollout stochastic
        # and preserving the DDP forward call for gradient synchronization.
        if conditions is None:
            conditions = self.encode_conditions(rgb, mask, acquisition, source_angle,
                                                rgb_alt, mask_alt)
        if rollout_steps:
            state = conditions.prior
            dt = 1.0 / rollout_steps
            trajectory = []
            for index in range(rollout_steps):
                bridge_t = torch.full((len(rgb),), (index + .5) * dt,
                                      device=rgb.device, dtype=rgb.dtype)
                velocity = self.predict(state, bridge_t, acquisition, conditions)
                following = state + dt * velocity
                trajectory.append((state, following, bridge_t))
                state = following
                if rollout_noise > 0 and index + 1 < rollout_steps:
                    state = state + rollout_noise * math.sqrt(dt) * torch.randn_like(state)
            if return_conditions:
                return state.clamp(-1, 1), conditions, trajectory
            return state.clamp(-1, 1)
        prediction = self.predict(state, timestep, acquisition, conditions)
        if return_conditions:
            return prediction, conditions
        return prediction


def bridge_interpolation(prior: torch.Tensor, target: torch.Tensor, timestep: torch.Tensor,
                         noise_scale: float = .12,
                         generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the UNSB-style stochastic bridge path and its velocity target."""
    t = timestep[:, None, None, None].float()
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    sigma = noise_scale * (t * (1.0 - t)).clamp_min(0).sqrt()
    state = (1.0 - t) * prior + t * target + sigma * noise
    velocity = (target - prior).detach()
    return state, velocity


@torch.no_grad()
def bridge_sample(model: SilhouetteBridge, rgb: torch.Tensor, mask: torch.Tensor,
                  acquisition: torch.Tensor, steps: int = 8,
                  temperature: float = .05,
                  generator: torch.Generator | None = None,
                  source_angle: torch.Tensor | None = None,
                  rgb_alt: torch.Tensor | None = None,
                  mask_alt: torch.Tensor | None = None) -> torch.Tensor:
    """Euler-integrate the learned bridge from the RGB prior to SAR."""
    if steps < 1:
        raise ValueError("bridge sampling needs at least one step")
    conditions = model.encode_conditions(rgb, mask, acquisition, source_angle, rgb_alt, mask_alt)
    state = conditions.prior + temperature * torch.randn(
        conditions.prior.shape, device=conditions.prior.device,
        dtype=conditions.prior.dtype, generator=generator)
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full((len(rgb),), (index + .5) * dt, device=rgb.device, dtype=rgb.dtype)
        velocity = model.predict(state, t, acquisition, conditions)
        state = state + dt * velocity
        if index + 1 < steps and temperature > 0:
            state = state + temperature * math.sqrt(dt) * torch.randn(
                state.shape, device=state.device, dtype=state.dtype, generator=generator)
    return state.clamp(-1.0, 1.0)


def bridge_loss(prediction: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Robust velocity matching; no RGB/SAR pixel-alignment term is used."""
    return F.smooth_l1_loss(prediction, velocity, beta=.1)


def identity_loss(primary_logits: torch.Tensor, alternate_logits: torch.Tensor,
                  labels: torch.Tensor, primary_token: torch.Tensor,
                  alternate_token: torch.Tensor, consistency_weight: float = .25) -> torch.Tensor:
    ce = .5 * (F.cross_entropy(primary_logits, labels) +
               F.cross_entropy(alternate_logits, labels))
    cosine = 1.0 - F.cosine_similarity(primary_token, alternate_token, dim=1).mean()
    return ce + consistency_weight * cosine


class BridgeDiscriminator(nn.Module):
    """Projection PatchD for target-domain realism and acquisition matching."""

    def __init__(self, token_dim: int = 256, base: int = 64) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, base, 4, 2, 1), nn.LeakyReLU(.2, True)),
            nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1),
                          nn.GroupNorm(_groups(base * 2), base * 2), nn.LeakyReLU(.2, True)),
            nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, 2, 1),
                          nn.GroupNorm(_groups(base * 4), base * 4), nn.LeakyReLU(.2, True)),
            nn.Sequential(nn.Conv2d(base * 4, base * 8, 4, 2, 1),
                          nn.GroupNorm(_groups(base * 8), base * 8), nn.LeakyReLU(.2, True)),
        ])
        channels = base * 8
        self.image_head = nn.Linear(channels, 1)
        self.condition = nn.Sequential(
            nn.Linear(token_dim + CONDITION_DIM + SOURCE_ANGLE_DIM, channels), nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, image: torch.Tensor, token: torch.Tensor,
                acquisition: torch.Tensor, source_angle: torch.Tensor) -> tuple[
                    torch.Tensor, tuple[torch.Tensor, ...]
                ]:
        features = []
        hidden = image
        for layer in self.layers:
            hidden = layer(hidden)
            features.append(hidden)
        pooled = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
        # The condition is detached by construction at the trainer boundary;
        # D must not teach the RGB encoder a class shortcut.
        condition = self.condition(torch.cat((token, acquisition, source_angle), dim=1))
        score = self.image_head(pooled).squeeze(1)
        score = score + (pooled * condition).sum(1) / math.sqrt(pooled.shape[1])
        return score, tuple(features)


class BridgeEnergy(nn.Module):
    """Small time/condition energy estimator used by the SB regularizer."""

    def __init__(self, token_dim: int = 256, base: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, base, 4, 2, 1), nn.LeakyReLU(.2, True),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.GroupNorm(_groups(base * 2), base * 2),
            nn.LeakyReLU(.2, True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.GroupNorm(_groups(base * 4), base * 4),
            nn.LeakyReLU(.2, True),
        )
        channels = base * 4
        self.context = nn.Sequential(
            nn.Linear(token_dim + CONDITION_DIM + SOURCE_ANGLE_DIM + 1, channels), nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, previous: torch.Tensor, following: torch.Tensor,
                timestep: torch.Tensor, token: torch.Tensor,
                acquisition: torch.Tensor, source_angle: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(self.features(torch.cat((previous, following), dim=1)), 1).flatten(1)
        context = self.context(torch.cat((token.detach(), acquisition, source_angle,
                                           timestep[:, None]), dim=1))
        return self.head(pooled + context).squeeze(1)


class BridgePatchEncoder(nn.Module):
    """Shared source/target feature extractor for structure-only PatchNCE."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvNormAct(1, base, stride=1),
            ConvNormAct(base, base * 2, stride=2),
            ConvNormAct(base * 2, base * 4, stride=2),
        ])

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        hidden = image
        for block in self.blocks:
            hidden = block(hidden)
            outputs.append(hidden)
        return tuple(outputs)


def patch_nce_loss(source: torch.Tensor, target: torch.Tensor,
                   encoder: BridgePatchEncoder, temperature: float = .10,
                   max_patches: int = 128) -> torch.Tensor:
    """Contrast source silhouette and generated structure without SAR pairing."""
    source_features = encoder(source)
    target_features = encoder(target)
    losses = []
    for source_feature, target_feature in zip(source_features, target_features):
        batch, channels, height, width = source_feature.shape
        count = min(max_patches, height * width)
        # Fixed stride sampling is deterministic under DDP and covers the
        # entire map; it is not a pixel correspondence to a real SAR image.
        indices = torch.linspace(0, height * width - 1, count, device=source.device).long()
        key = F.normalize(source_feature.flatten(2)[:, :, indices].transpose(1, 2), dim=-1)
        query = F.normalize(target_feature.flatten(2)[:, :, indices].transpose(1, 2), dim=-1)
        logits = torch.einsum("bpc,bqc->bpq", query, key) / temperature
        labels = torch.arange(count, device=source.device).expand(batch, -1)
        losses.append(F.cross_entropy(logits.reshape(batch * count, count), labels.reshape(-1)))
    return torch.stack(losses).mean()


def bridge_rollout(model: SilhouetteBridge, conditions: BridgeCondition,
                   acquisition: torch.Tensor, steps: int = 5,
                   stochastic_scale: float = 0.0,
                   generator: torch.Generator | None = None) -> tuple[
                       torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                   ]:
    """Successive refinement trajectory used by UNSB-style G/SB objectives."""
    if steps < 1:
        raise ValueError("bridge rollout needs at least one step")
    state = conditions.prior
    dt = 1.0 / steps
    trajectory = []
    for index in range(steps):
        timestep = torch.full((len(state),), (index + .5) * dt,
                              device=state.device, dtype=state.dtype)
        velocity = model.predict(state, timestep, acquisition, conditions)
        following = state + dt * velocity
        trajectory.append((state, following, timestep))
        state = following
        if stochastic_scale > 0 and index + 1 < steps:
            state = state + stochastic_scale * math.sqrt(dt) * torch.randn(
                state.shape, device=state.device, dtype=state.dtype, generator=generator)
    return state.clamp(-1, 1), trajectory


def sb_energy_loss(energy: BridgeEnergy, positive: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   negative: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   conditions: BridgeCondition, acquisition: torch.Tensor,
                   tau: float = .10, detach_negative: bool = False) -> torch.Tensor:
    previous, following, timestep = positive
    negative_previous, negative_following, negative_timestep = negative
    positive_score = energy(previous, following, timestep, conditions.token,
                            acquisition, conditions.source_angle)
    negative_score = energy(negative_previous, negative_following, negative_timestep,
                            conditions.token, acquisition, conditions.source_angle)
    if detach_negative:
        negative_score = negative_score.detach()
    contrast = -(positive_score - negative_score).mean()
    transport = (previous - following).square().mean()
    return tau * (contrast + transport)


def discriminator_hinge(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()

"""RGB-identity- and acquisition-conditioned DDPM for 64x64 SAR ROIs.

The RGB and SAR images are only class-level associated in this project.  The
RGB encoder therefore contributes a global identity token rather than spatial
skips; the U-Net is never asked to align RGB pixels to SAR pixels.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


CONDITIONAL_DIFFUSION_ARCHITECTURE = "rgb_identity_conditioned_sar_ddpm64_v2"
LEGACY_CONDITIONAL_DIFFUSION_ARCHITECTURE = "rgb_conditioned_sar_ddpm64_v1"
CONDITION_DIM = 12
CLASS_COUNT = 40


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    """Standard continuous timestep embedding without mutable state."""
    half = dimension // 2
    if half == 0:
        return timestep.float()[:, None]
    scale = math.log(10_000.0) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=timestep.device, dtype=torch.float32) * -scale)
    phases = timestep.float()[:, None] * frequencies[None]
    embedding = torch.cat((phases.sin(), phases.cos()), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, context_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.context = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, out_channels * 2))
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(image)))
        scale, bias = self.context(context).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(image)


class AttentionBlock(nn.Module):
    """A compact self-attention block used only at the 8x8 bottleneck."""
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("attention channels must divide heads")
        self.heads = heads
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        hidden = self.qkv(self.norm(image)).reshape(batch, 3, self.heads,
                                                     channels // self.heads, height * width)
        query, key, value = hidden.unbind(1)
        weights = torch.softmax(torch.einsum("bhdi,bhdj->bhij", query, key) * (query.shape[2] ** -0.5), dim=-1)
        hidden = torch.einsum("bhij,bhdj->bhdi", weights, value).reshape(batch, channels, height, width)
        return image + self.proj(hidden)


class RGBConditionEncoder(nn.Module):
    """Map one RGB view to an identity-aware token and class logits."""
    def __init__(self, token_dim: int = 256, base: int = 32,
                 classes: int | None = CLASS_COUNT) -> None:
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8)
        layers: list[nn.Module] = []
        previous = 3
        for channel in channels:
            layers.extend((
                nn.Conv2d(previous, channel, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(_groups(channel), channel), nn.SiLU(),
                nn.Conv2d(channel, channel, 3, padding=1, bias=False),
                nn.GroupNorm(_groups(channel), channel), nn.SiLU(),
            ))
            previous = channel
        self.features = nn.Sequential(*layers)
        self.token = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1], token_dim),
            nn.LayerNorm(token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.classifier = nn.Linear(token_dim, classes) if classes is not None else None

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        token = self.token(self.features(rgb))
        logits = self.classifier(token) if self.classifier is not None else None
        return token, logits


class ConditionalSARUNet(nn.Module):
    """64px v-prediction U-Net with global RGB/acquisition FiLM."""
    def __init__(self, base: int = 64, context_dim: int = 256,
                 separate_context: bool = True) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.separate_context = bool(separate_context)
        self.time_mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim * 2), nn.SiLU(),
            nn.Linear(context_dim * 2, context_dim))
        self.acquisition_mlp = nn.Sequential(
            nn.Linear(CONDITION_DIM, context_dim), nn.SiLU(),
            nn.Linear(context_dim, context_dim))
        if self.separate_context:
            self.context_fusion = nn.Sequential(
                nn.Linear(context_dim * 3, context_dim * 2), nn.SiLU(),
                nn.Linear(context_dim * 2, context_dim))
        self.input = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = ResidualBlock(base, base, context_dim)
        self.downsample1 = nn.Conv2d(base, base, 4, stride=2, padding=1)
        self.down2 = ResidualBlock(base, base * 2, context_dim)
        self.downsample2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)
        self.down3 = ResidualBlock(base * 2, base * 4, context_dim)
        self.downsample3 = nn.Conv2d(base * 4, base * 4, 4, stride=2, padding=1)
        self.mid1 = ResidualBlock(base * 4, base * 4, context_dim)
        self.attention = AttentionBlock(base * 4)
        self.mid2 = ResidualBlock(base * 4, base * 4, context_dim)
        self.upsample3 = nn.Conv2d(base * 4, base * 4, 3, padding=1)
        self.up3 = ResidualBlock(base * 8, base * 2, context_dim)
        self.upsample2 = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.up2 = ResidualBlock(base * 4, base, context_dim)
        self.upsample1 = nn.Conv2d(base, base, 3, padding=1)
        self.up1 = ResidualBlock(base * 2, base, context_dim)
        self.output_norm = nn.GroupNorm(_groups(base), base)
        self.output = nn.Conv2d(base, 1, 3, padding=1)

    def context(self, timestep: torch.Tensor, rgb_token: torch.Tensor,
                acquisition: torch.Tensor) -> torch.Tensor:
        time_context = self.time_mlp(sinusoidal_embedding(timestep, self.context_dim))
        acquisition_context = self.acquisition_mlp(acquisition)
        if self.separate_context:
            return self.context_fusion(torch.cat((time_context, rgb_token, acquisition_context), dim=1))
        return time_context + rgb_token + acquisition_context

    @staticmethod
    def _upsample(image: torch.Tensor, convolution: nn.Conv2d) -> torch.Tensor:
        image = F.interpolate(image, scale_factor=2, mode="nearest")
        return convolution(image)

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor, rgb_token: torch.Tensor,
                acquisition: torch.Tensor) -> torch.Tensor:
        context = self.context(timestep, rgb_token, acquisition)
        image = self.input(noisy)
        skip1 = self.down1(image, context)
        image = self.downsample1(skip1)
        skip2 = self.down2(image, context)
        image = self.downsample2(skip2)
        skip3 = self.down3(image, context)
        image = self.downsample3(skip3)
        image = self.mid2(self.attention(self.mid1(image, context)), context)
        image = self.up3(torch.cat((self._upsample(image, self.upsample3), skip3), dim=1), context)
        image = self.up2(torch.cat((self._upsample(image, self.upsample2), skip2), dim=1), context)
        image = self.up1(torch.cat((self._upsample(image, self.upsample1), skip1), dim=1), context)
        return self.output(F.silu(self.output_norm(image)))


class ConditionalSARDDPM(nn.Module):
    """RGB identity encoder plus conditional SAR denoiser.

    The v2 path uses two class-matched RGB views.  A detached soft class
    posterior is projected into the U-Net context, while the explicit RGB CE
    remains the only gradient that shapes the identity classifier.  Passing
    ``class_conditioning=False`` keeps v1 checkpoints renderable.
    """
    def __init__(self, base: int = 64, token_dim: int = 256, rgb_base: int = 32,
                 classes: int = CLASS_COUNT, class_conditioning: bool = True) -> None:
        super().__init__()
        self.class_conditioning = bool(class_conditioning)
        self.rgb_encoder = RGBConditionEncoder(
            token_dim=token_dim, base=rgb_base,
            classes=classes if self.class_conditioning else None)
        if self.class_conditioning:
            self.class_context = nn.Sequential(
                nn.Linear(classes, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.unet = ConditionalSARUNet(
            base=base, context_dim=token_dim, separate_context=self.class_conditioning)

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor, rgb: torch.Tensor,
                acquisition: torch.Tensor, condition_drop: torch.Tensor | None = None,
                rgb_alt: torch.Tensor | None = None,
                return_identity: bool = False) -> torch.Tensor | tuple[torch.Tensor, ...]:
        rgb_token, class_logits = self.rgb_encoder(rgb)
        primary_token = rgb_token
        alt_logits = None
        alt_token = None
        if rgb_alt is not None:
            alt_token, alt_logits = self.rgb_encoder(rgb_alt)
            rgb_token = .5 * (rgb_token + alt_token)
            if class_logits is not None and alt_logits is not None:
                class_logits = .5 * (class_logits + alt_logits)
        if self.class_conditioning:
            if class_logits is None:
                raise RuntimeError("identity-conditioned model needs RGB class logits")
            # The denoiser can use the posterior values, but cannot alter the
            # classifier through the diffusion objective.
            rgb_token = rgb_token + self.class_context(F.softmax(class_logits.detach(), dim=-1))
        acquisition = acquisition.float()
        if condition_drop is not None:
            if condition_drop.ndim != 1 or len(condition_drop) != len(noisy):
                raise ValueError("condition_drop must have shape [B]")
            keep = (~condition_drop.bool()).to(rgb_token.dtype)[:, None]
            rgb_token = rgb_token * keep
            acquisition = acquisition * keep
        prediction = self.unet(noisy, timestep, rgb_token, acquisition)
        if return_identity:
            return prediction, class_logits, alt_logits, primary_token, alt_token
        return prediction

    def identity_logits(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return logits for the auxiliary RGB identity objective."""
        _, logits = self.rgb_encoder(rgb)
        if logits is None:
            raise RuntimeError("identity logits are unavailable for the legacy model")
        return logits


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((points / steps) + offset) / (1.0 + offset) * math.pi * .5).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(.0001, .9999).float()


class DiffusionSchedule(nn.Module):
    """Precomputed DDPM coefficients and deterministic DDIM sampling."""
    def __init__(self, steps: int = 1_000) -> None:
        super().__init__()
        if steps < 2:
            raise ValueError("diffusion needs at least two steps")
        beta = cosine_beta_schedule(steps)
        alpha = 1.0 - beta
        alpha_bar = alpha.cumprod(0)
        self.steps = int(steps)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())

    def q_sample(self, clean: torch.Tensor, timestep: torch.Tensor,
                 noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        alpha = self.sqrt_alpha_bar[timestep][:, None, None, None]
        sigma = self.sqrt_one_minus_alpha_bar[timestep][:, None, None, None]
        return alpha * clean + sigma * noise, noise

    def v_target(self, clean: torch.Tensor, timestep: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Velocity target used by the v-prediction parameterisation."""
        alpha = self.sqrt_alpha_bar[timestep][:, None, None, None]
        sigma = self.sqrt_one_minus_alpha_bar[timestep][:, None, None, None]
        return alpha * noise - sigma * clean

    def predict_clean_from_epsilon(self, noisy: torch.Tensor, timestep: torch.Tensor,
                                   epsilon: torch.Tensor) -> torch.Tensor:
        alpha = self.sqrt_alpha_bar[timestep][:, None, None, None]
        sigma = self.sqrt_one_minus_alpha_bar[timestep][:, None, None, None]
        return (noisy - sigma * epsilon) / alpha.clamp_min(1e-6)

    def predict_clean_from_v(self, noisy: torch.Tensor, timestep: torch.Tensor,
                             velocity: torch.Tensor) -> torch.Tensor:
        alpha = self.sqrt_alpha_bar[timestep][:, None, None, None]
        sigma = self.sqrt_one_minus_alpha_bar[timestep][:, None, None, None]
        return alpha * noisy - sigma * velocity

    def epsilon_from_v(self, noisy: torch.Tensor, timestep: torch.Tensor,
                       velocity: torch.Tensor) -> torch.Tensor:
        alpha = self.sqrt_alpha_bar[timestep][:, None, None, None]
        sigma = self.sqrt_one_minus_alpha_bar[timestep][:, None, None, None]
        return sigma * noisy + alpha * velocity

    @torch.inference_mode()
    def ddim_sample(self, model: ConditionalSARDDPM, rgb: torch.Tensor,
                    acquisition: torch.Tensor, sample_steps: int = 32,
                    guidance_scale: float = 1.0,
                    generator: torch.Generator | None = None,
                    initial_noise: torch.Tensor | None = None,
                    prediction_type: str = "v",
                    rgb_alt: torch.Tensor | None = None) -> torch.Tensor:
        if sample_steps < 2:
            raise ValueError("DDIM sampling needs at least two steps")
        if prediction_type not in {"v", "epsilon"}:
            raise ValueError("prediction_type must be v or epsilon")
        device = rgb.device
        indices = torch.linspace(self.steps - 1, 0, sample_steps, device=device).round().long().unique_consecutive()
        expected_shape = (len(rgb), 1, 64, 64)
        if initial_noise is None:
            image = torch.randn(expected_shape, device=device, dtype=rgb.dtype, generator=generator)
        else:
            if tuple(initial_noise.shape) != expected_shape:
                raise ValueError(f"initial_noise must have shape {expected_shape}, got {tuple(initial_noise.shape)}")
            image = initial_noise.to(device=device, dtype=rgb.dtype).clone()
        for index, timestep in enumerate(indices):
            batch_t = torch.full((len(rgb),), int(timestep), device=device, dtype=torch.long)
            prediction = model(image, batch_t, rgb, acquisition, rgb_alt=rgb_alt)
            if guidance_scale != 1.0:
                dropped = torch.ones(len(rgb), device=device, dtype=torch.bool)
                unconditional = model(image, batch_t, rgb, acquisition,
                                      condition_drop=dropped, rgb_alt=rgb_alt)
                prediction = unconditional + guidance_scale * (prediction - unconditional)
            if prediction_type == "v":
                clean = self.predict_clean_from_v(image, batch_t, prediction)
                epsilon = self.epsilon_from_v(image, batch_t, prediction)
            else:
                epsilon = prediction
                clean = self.predict_clean_from_epsilon(image, batch_t, epsilon)
            clean = clean.clamp(-1.0, 1.0)
            if index + 1 == len(indices):
                image = clean
                continue
            next_t = torch.full((len(rgb),), int(indices[index + 1]), device=device, dtype=torch.long)
            alpha_next = self.sqrt_alpha_bar[next_t][:, None, None, None]
            sigma_next = self.sqrt_one_minus_alpha_bar[next_t][:, None, None, None]
            image = alpha_next * clean + sigma_next * epsilon
        return image.clamp(-1.0, 1.0)


@torch.no_grad()
def ema_update(ema: nn.Module, source: nn.Module, decay: float) -> None:
    for target, value in zip(ema.parameters(), source.parameters()):
        target.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
    for target, value in zip(ema.buffers(), source.buffers()):
        target.copy_(value)

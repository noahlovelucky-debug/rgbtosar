"""Models and differentiable losses for joint identity-conditioned ROI GAN."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CosineClassifier(nn.Module):
    """CosFace class centres with legacy ``weight``/``bias`` checkpoint keys."""

    def __init__(self, features: int, classes: int, scale: float = 20.0, margin: float = .15) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, features))
        # Retained solely so old RGBIdentityEncoder checkpoints remain loadable.
        self.bias = nn.Parameter(torch.zeros(classes))
        self.scale, self.margin = scale, margin
        nn.init.normal_(self.weight, 0.0, .02)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        logits = F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))
        if labels is not None:
            logits = logits.clone()
            logits[torch.arange(len(labels), device=labels.device), labels] -= self.margin
        return logits * self.scale


class RGBIdentityEncoder(nn.Module):
    """Jointly trained vehicle recogniser; its embedding directly conditions G."""

    def __init__(self, num_classes: int = 40, dim: int = 256, base: int = 32) -> None:
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8)
        layers: list[nn.Module] = []
        previous = 3
        for channel in channels:
            layers.extend((
                nn.Conv2d(previous, channel, 4, 2, 1, bias=False),
                nn.GroupNorm(min(16, channel), channel),
                nn.SiLU(inplace=True),
            ))
            previous = channel
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.embedding = nn.Sequential(nn.Linear(channels[-1], dim), nn.LayerNorm(dim), nn.SiLU())
        self.classifier = CosineClassifier(dim, num_classes)

    def class_logits(self, identity: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(identity, labels)

    def forward(self, rgb: torch.Tensor, return_pyramid: bool = False):
        """Return identity logits and, optionally, 64/32/16/8px feature maps.

        Keeping the original ``features.*`` module names preserves old
        checkpoints while exposing spatial information to the new generator.
        """
        x = rgb
        pyramid = []
        for stage in range(4):
            for layer in self.features[stage * 3:(stage + 1) * 3]:
                x = layer(x)
            pyramid.append(x)
        identity = self.embedding(self.features[-1](x).flatten(1))
        logits = self.class_logits(identity)
        return (identity, logits, tuple(pyramid)) if return_pyramid else (identity, logits)


class ROIGenerator(nn.Module):
    def __init__(self, identity_dim: int = 256, meta_dim: int = 10, base: int = 32) -> None:
        super().__init__()
        self.speckle_strength = 0.32
        self.meta = nn.Sequential(nn.Linear(meta_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.fc = nn.Linear(identity_dim + 128, base * 8 * 4 * 4)
        channels = base * 8
        layers: list[nn.Module] = []
        for output in (base * 4, base * 2, base, max(16, base // 2)):
            layers.extend((
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels, output, 3, 1, 1, bias=False),
                nn.GroupNorm(min(16, output), output),
                nn.SiLU(inplace=True),
                nn.Conv2d(output, output, 3, 1, 1, bias=False),
                nn.GroupNorm(min(16, output), output),
                nn.SiLU(inplace=True),
            ))
            channels = output
        layers.extend((nn.Conv2d(channels, 1, 3, 1, 1), nn.Tanh()))
        self.net = nn.Sequential(*layers)

    def apply_speckle(self, clean: torch.Tensor, strength: float | None = None,
                      radiometric_variation: bool = True) -> torch.Tensor:
        """Differentiable stochastic SAR observation model.

        Besides correlated multiplicative speckle, each observation receives a
        weak low-frequency illumination field, calibration/gamma variation and
        a positive receiver-noise floor.  The earlier white-noise-only renderer
        produced one narrow synthetic style per vehicle and transferred poorly
        to real SAR.
        """
        strength = self.speckle_strength if strength is None else strength
        amplitude = (clean + 1.0) * .5
        noise = torch.randn_like(amplitude)
        correlated = F.avg_pool2d(noise, 3, stride=1, padding=1)
        noise = .5 * noise + .5 * correlated
        multiplier = torch.exp(strength * noise - .5 * strength ** 2)
        amplitude = amplitude * multiplier
        if radiometric_variation and strength > 0:
            batch = len(amplitude)
            # Acquisition-to-acquisition radiometric variation broadens the
            # image-level distribution without changing target identity.
            gain_log = torch.randn(batch, 1, 1, 1, device=amplitude.device,
                                   dtype=amplitude.dtype) * .22
            gain = torch.exp(gain_log - .5 * .22 ** 2)
            gamma = torch.exp(torch.randn(batch, 1, 1, 1, device=amplitude.device,
                                          dtype=amplitude.dtype) * .10)
            field = F.avg_pool2d(torch.randn_like(amplitude), 15, stride=1, padding=7)
            field = field / field.std((2, 3), keepdim=True).clamp_min(1e-4)
            field = torch.exp(.10 * field - .5 * .10 ** 2)
            amplitude = amplitude.clamp_min(1e-5).pow(gamma) * gain * field
            receiver_level = amplitude.new_empty(batch, 1, 1, 1).uniform_(.004, .020)
            rayleigh = torch.sqrt(torch.randn_like(amplitude).square()
                                  + torch.randn_like(amplitude).square()) / 1.2533
            amplitude = amplitude + receiver_level * rayleigh
        # A small zero-mean receiver term prevents an unnaturally hard floor.
        receiver_scale = strength / max(self.speckle_strength, 1e-6)
        amplitude = amplitude + (.004 * receiver_scale) * torch.randn_like(amplitude)
        return amplitude.clamp(0, 1) * 2.0 - 1.0

    def forward(self, identity: torch.Tensor, meta: torch.Tensor,
                apply_speckle: bool = True) -> torch.Tensor:
        latent = torch.cat((identity, self.meta(meta)), dim=1)
        clean = self.net(self.fc(latent).reshape(-1, self.fc.out_features // 16, 4, 4))
        return self.apply_speckle(clean) if apply_speckle else clean


class SpatialROIGenerator(ROIGenerator):
    """FPN-conditioned generator for a continuous target SAR observation.

    Identity controls the vehicle class while RGB pyramid maps inject silhouette,
    aspect ratio and view-dependent shape at every SAR decoding scale.
    """

    def __init__(self, identity_dim: int = 256, meta_dim: int = 12, base: int = 32) -> None:
        super().__init__(identity_dim=identity_dim, meta_dim=meta_dim, base=base)
        # Encoder maps are 64/32/16/8 at channels base/2base/4base/8base.
        self.spatial_projection = nn.ModuleList((
            nn.Conv2d(base * 8, base * 4, 1),
            nn.Conv2d(base * 4, base * 2, 1),
            nn.Conv2d(base * 2, base, 1),
            nn.Conv2d(base, max(16, base // 2), 1),
        ))

    def forward(self, identity: torch.Tensor, meta: torch.Tensor,
                pyramid: tuple[torch.Tensor, ...], apply_speckle: bool = True) -> torch.Tensor:
        if len(pyramid) != 4:
            raise ValueError("expected four RGB pyramid maps")
        latent = torch.cat((identity, self.meta(meta)), dim=1)
        x = self.fc(latent).reshape(-1, self.fc.out_features // 16, 4, 4)
        # ``self.net`` consists of four upsampling blocks followed by output.
        for index in range(4):
            block = self.net[index * 7:(index + 1) * 7]
            x = block(x)
            feature = pyramid[3 - index]
            feature = F.interpolate(feature, x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.spatial_projection[index](feature)
        clean = self.net[28:](x)
        return self.apply_speckle(clean) if apply_speckle else clean


class SARStyleEncoder(nn.Module):
    """Variational encoder for real-SAR acquisition/scattering style."""

    def __init__(self, style_dim: int = 32, base: int = 32) -> None:
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8)
        layers: list[nn.Module] = []
        previous = 1
        for channel in channels:
            layers.extend((nn.Conv2d(previous, channel, 4, 2, 1, bias=False),
                           nn.GroupNorm(min(16, channel), channel), nn.SiLU(inplace=True)))
            previous = channel
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mu = nn.Linear(channels[-1], style_dim)
        self.logvar = nn.Linear(channels[-1], style_dim)

    def forward(self, roi: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(roi)
        mu, logvar = self.mu(features), self.logvar(features).clamp(-8, 8)
        style = mu + torch.randn_like(mu) * torch.exp(.5 * logvar) if sample else mu
        return style, mu, logvar


class StyleSpatialROIGenerator(SpatialROIGenerator):
    """Spatial RGB generator with a sampleable real-SAR style latent."""

    def __init__(self, identity_dim: int = 256, meta_dim: int = 12, style_dim: int = 32,
                 base: int = 32) -> None:
        super().__init__(identity_dim=identity_dim, meta_dim=meta_dim, base=base)
        self.style_dim = style_dim
        self.identity_style = nn.Linear(style_dim, identity_dim)
        outputs = (base * 4, base * 2, base, max(16, base // 2))
        self.style_affine = nn.ModuleList(nn.Linear(style_dim, channels * 2) for channels in outputs)

    def reset_style_injection(self) -> None:
        """Start as the warm-started deterministic generator."""
        nn.init.zeros_(self.identity_style.weight); nn.init.zeros_(self.identity_style.bias)
        for layer in self.style_affine:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, identity: torch.Tensor, meta: torch.Tensor,
                pyramid: tuple[torch.Tensor, ...], style: torch.Tensor | None = None,
                apply_speckle: bool = True) -> torch.Tensor:
        if len(pyramid) != 4:
            raise ValueError("expected four RGB pyramid maps")
        if style is None:
            style = identity.new_zeros(len(identity), self.style_dim)
        identity = identity + self.identity_style(style)
        latent = torch.cat((identity, self.meta(meta)), dim=1)
        x = self.fc(latent).reshape(-1, self.fc.out_features // 16, 4, 4)
        for index in range(4):
            x = self.net[index * 7:(index + 1) * 7](x)
            scale, bias = self.style_affine[index](style).chunk(2, dim=1)
            x = x * (1 + .25 * torch.tanh(scale)[:, :, None, None]) + bias[:, :, None, None]
            feature = F.interpolate(pyramid[3 - index], x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.spatial_projection[index](feature)
        clean = self.net[28:](x)
        return self.apply_speckle(clean) if apply_speckle else clean


class SARSpatialCodeEncoder(nn.Module):
    """Encode a real SAR ROI into a compact spatial scattering code."""

    def __init__(self, code_channels: int = 64, base: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1, bias=False), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False), nn.GroupNorm(16, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, code_channels, 4, 2, 1, bias=False),
            nn.GroupNorm(min(16, code_channels), code_channels), nn.SiLU(),
        )

    def forward(self, roi: torch.Tensor) -> torch.Tensor:
        return self.net(roi)


class CodebookSpatialROIGenerator(SpatialROIGenerator):
    """RGB spatial generator conditioned by an 8x8 real-SAR scattering code."""

    def __init__(self, identity_dim: int = 256, meta_dim: int = 12, code_channels: int = 64,
                 base: int = 32) -> None:
        super().__init__(identity_dim=identity_dim, meta_dim=meta_dim, base=base)
        outputs = (base * 4, base * 2, base, max(16, base // 2))
        self.code_channels = code_channels
        self.code_projection = nn.ModuleList(nn.Conv2d(code_channels, channels, 1) for channels in outputs)

    def reset_code_injection(self) -> None:
        for layer in self.code_projection:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, identity: torch.Tensor, meta: torch.Tensor,
                pyramid: tuple[torch.Tensor, ...], code: torch.Tensor,
                apply_speckle: bool = True) -> torch.Tensor:
        if code.ndim != 4 or code.shape[1] != self.code_channels:
            raise ValueError(f"expected Bx{self.code_channels}x8x8 SAR code")
        latent = torch.cat((identity, self.meta(meta)), dim=1)
        x = self.fc(latent).reshape(-1, self.fc.out_features // 16, 4, 4)
        for index in range(4):
            x = self.net[index * 7:(index + 1) * 7](x)
            rgb_feature = F.interpolate(pyramid[3 - index], x.shape[-2:], mode="bilinear", align_corners=False)
            code_feature = F.interpolate(code, x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.spatial_projection[index](rgb_feature) + self.code_projection[index](code_feature)
        clean = self.net[28:](x)
        return self.apply_speckle(clean) if apply_speckle else clean


class ROIDiscriminator(nn.Module):
    def __init__(self, base: int = 32) -> None:
        super().__init__()
        channels = (base, base * 2, base * 4, base * 8)
        layers: list[nn.Module] = []
        previous = 1
        for index, channel in enumerate(channels):
            conv = nn.utils.spectral_norm(nn.Conv2d(previous, channel, 4, 2, 1))
            layers.extend((conv, nn.LeakyReLU(0.2, inplace=True)))
            previous = channel
        self.features = nn.Sequential(*layers)
        # A 4x4 PatchGAN score map keeps the discriminator sensitive to local
        # SAR texture; the former single global score accepted smooth swirls.
        self.score = nn.utils.spectral_norm(nn.Conv2d(channels[-1], 1, 3, 1, 1))

    def forward(self, roi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(roi)
        return self.score(features).flatten(1), features


class ContinuousROIDiscriminator(ROIDiscriminator):
    """Projection PatchGAN with an optional real-SAR class auxiliary head.

    The class head reads the same spatial feature tensor as the PatchGAN but
    does not participate in the score map.  It is initialized to zero so an
    old V1 checkpoint has exactly the same adversarial behavior when the
    auxiliary loss is disabled.  The trainer can later enable a real-only
    class loss without introducing a separate classifier backbone.
    """

    def __init__(self, meta_dim: int = 12, base: int = 32, classes: int = 40,
                 class_pool: str = "mean") -> None:
        super().__init__(base=base)
        self.condition = nn.Sequential(nn.Linear(meta_dim, base * 8), nn.SiLU(), nn.Linear(base * 8, base * 8))
        self.classes = classes
        if class_pool not in {"mean", "mean_max"}:
            raise ValueError(f"unsupported class pooling mode: {class_pool}")
        self.class_pool = class_pool
        pooled_channels = base * 8 * (2 if class_pool == "mean_max" else 1)
        self.classifier = nn.Linear(pooled_channels, classes)
        # D0 must be numerically equivalent to the archived V1 discriminator.
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, roi: torch.Tensor, meta: torch.Tensor,
                return_class_logits: bool = False):
        features = self.features(roi)
        projection = (features * self.condition(meta)[:, :, None, None]).sum(1, keepdim=True)
        score = (self.score(features) + projection).flatten(1)
        if return_class_logits:
            pooled = features.mean((2, 3))
            if self.class_pool == "mean_max":
                pooled = torch.cat((pooled, features.amax((2, 3))), dim=1)
            return score, features, self.classifier(pooled)
        return score, features


def _sobel(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = image.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).reshape(1, 1, 3, 3) / 8
    kernel_y = kernel_x.transpose(2, 3)
    return F.conv2d(image, kernel_x, padding=1), F.conv2d(image, kernel_y, padding=1)


def _ssim_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    # Global SSIM is intentionally used: ROIs are class/angle matched but not
    # pixel registered, so local-window SSIM would over-penalise small shifts.
    x, y = (fake + 1) * 0.5, (real + 1) * 0.5
    dims = (2, 3)
    mx, my = x.mean(dims, keepdim=True), y.mean(dims, keepdim=True)
    vx = ((x - mx) ** 2).mean(dims, keepdim=True)
    vy = ((y - my) ** 2).mean(dims, keepdim=True)
    covariance = ((x - mx) * (y - my)).mean(dims, keepdim=True)
    score = ((2 * mx * my + 0.01 ** 2) * (2 * covariance + 0.03 ** 2))
    score = score / ((mx.square() + my.square() + 0.01 ** 2) * (vx + vy + 0.03 ** 2))
    return (1.0 - score.clamp(-1, 1)).mean()


def _align_translation(fake: torch.Tensor, real: torch.Tensor,
                       max_shift: int = 4, stride: int = 2) -> torch.Tensor:
    """Select the best small translation for each unregistered real SAR ROI.

    The shift is a nuisance alignment operation on the supervision target; it
    is not a transform applied to generated output.  Selection is discrete,
    while the subsequent structure loss remains differentiable with respect
    to ``fake``.
    """
    padded = F.pad(real, (max_shift,) * 4, value=-1.0)
    candidates = []
    height, width = real.shape[-2:]
    for dy in range(-max_shift, max_shift + 1, stride):
        for dx in range(-max_shift, max_shift + 1, stride):
            y0, x0 = max_shift + dy, max_shift + dx
            candidates.append(padded[..., y0:y0 + height, x0:x0 + width])
    shifted = torch.stack(candidates, dim=1)
    with torch.no_grad():
        small_fake = F.avg_pool2d(fake, 4)
        batch, count = shifted.shape[:2]
        small_real = F.avg_pool2d(shifted.flatten(0, 1), 4).reshape(batch, count, 1,
                                                                      height // 4, width // 4)
        costs = (small_real - small_fake[:, None]).abs().mean((2, 3, 4))
        choice = costs.argmin(1)
    return shifted[torch.arange(len(fake), device=fake.device), choice]


def multiscale_structure_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Translation-aligned low-frequency, edge and global-SSIM loss."""
    real = _align_translation(fake, real)
    total = fake.new_zeros(())
    x, y = fake, real
    for weight in (1.0, 0.5, 0.25):
        total = total + weight * F.l1_loss(x, y)
        if min(x.shape[-2:]) >= 16:
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    fx, fy = _sobel(fake)
    rx, ry = _sobel(real)
    fake_gradient = torch.sqrt(fx.square() + fy.square() + 1e-6)
    real_gradient = torch.sqrt(rx.square() + ry.square() + 1e-6)
    return total / 1.75 + 0.5 * F.l1_loss(fake_gradient, real_gradient) + _ssim_loss(fake, real)


def aligned_structure_terms(fake: torch.Tensor, real: torch.Tensor) -> dict[str, torch.Tensor]:
    """Expose the original V1 structure terms for one-variable ablations.

    The real target is weakly paired, so the small translation selection is
    retained.  Keeping the individual terms separate lets an experiment remove
    only 64px L1, SSIM, or edge placement without silently changing the others.
    """
    aligned_real = _align_translation(fake, real)
    pixel_64 = F.l1_loss(fake, aligned_real)
    pixel_32 = F.l1_loss(F.avg_pool2d(fake, 2), F.avg_pool2d(aligned_real, 2))
    pixel_16 = F.l1_loss(F.avg_pool2d(fake, 4), F.avg_pool2d(aligned_real, 4))
    fx, fy = _sobel(fake)
    rx, ry = _sobel(aligned_real)
    fake_gradient = torch.sqrt(fx.square() + fy.square() + 1e-6)
    real_gradient = torch.sqrt(rx.square() + ry.square() + 1e-6)
    return {
        "pixel_64": pixel_64,
        "pixel_32": pixel_32,
        "pixel_16": pixel_16,
        "edge": F.l1_loss(fake_gradient, real_gradient),
        "ssim": _ssim_loss(fake, aligned_real),
    }


def weighted_aligned_structure_loss(
        fake: torch.Tensor, real: torch.Tensor, *, pixel_64_weight: float = 1.0,
        pixel_32_weight: float = 0.5, pixel_16_weight: float = 0.25,
        edge_weight: float = 0.5, ssim_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """V1 structure loss with independently switchable spatial subterms.

    Default coefficients reproduce :func:`multiscale_structure_loss` exactly.
    The pixel denominator intentionally remains the original 1.75 when a term
    is disabled: an ablation should reveal both its spatial and loss-magnitude
    effects instead of silently re-scaling the remaining objectives.
    """
    terms = aligned_structure_terms(fake, real)
    total = (
        (pixel_64_weight * terms["pixel_64"]
         + pixel_32_weight * terms["pixel_32"]
         + pixel_16_weight * terms["pixel_16"]) / 1.75
        + edge_weight * terms["edge"]
        + ssim_weight * terms["ssim"]
    )
    return total, terms


def sar_statistics_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Match radiometry and edge-energy moments without assuming registration."""
    fake_x, real_x = (fake + 1) * .5, (real + 1) * .5
    dims = (2, 3)
    intensity = (F.l1_loss(fake_x.mean(dims), real_x.mean(dims))
                 + F.l1_loss(fake_x.std(dims), real_x.std(dims)))
    fx, fy = _sobel(fake_x); rx, ry = _sobel(real_x)
    fake_edge = torch.sqrt(fx.square() + fy.square() + 1e-6)
    real_edge = torch.sqrt(rx.square() + ry.square() + 1e-6)
    edges = (F.l1_loss(fake_edge.mean(dims), real_edge.mean(dims))
             + F.l1_loss(fake_edge.std(dims), real_edge.std(dims)))
    return intensity + edges


def _distribution_signature(image: torch.Tensor) -> torch.Tensor:
    """Per-image distribution descriptors with no spatial correspondence."""
    dims = (2, 3)
    values = [image.mean(dims), image.std(dims)]
    values.extend(torch.sigmoid((image - threshold) / .08).mean(dims)
                  for threshold in (.15, .35, .55, .75))
    return torch.cat(values, dim=1)


def _residual_signature(image: torch.Tensor) -> torch.Tensor:
    amplitude = (image + 1) * .5
    dims = (2, 3)
    values = []
    for kernel in (3, 7):
        residual = amplitude - F.avg_pool2d(amplitude, kernel, stride=1, padding=kernel // 2)
        magnitude = residual.abs()
        values.extend((magnitude.mean(dims), magnitude.std(dims)))
    grad_x, grad_y = _sobel(amplitude)
    energy = grad_x.square() + grad_y.square()
    values.extend((energy.mean(dims), energy.std(dims)))
    return torch.cat(values, dim=1)


def distributional_structure_loss(clean_fake: torch.Tensor, fake: torch.Tensor, real: torch.Tensor,
                                  fake_pyramid: tuple[torch.Tensor, ...],
                                  real_pyramid: tuple[torch.Tensor, ...]
                                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-registered structural similarity from image and discriminator moments.

    No term compares matching pixels or aligns either image. Low-frequency
    radiometry uses the clean generator output; residual and feature statistics
    use the observed (speckled) output.
    """
    if len(fake_pyramid) != len(real_pyramid):
        raise ValueError("fake and real discriminator pyramids must have equal depth")
    low = clean_fake.new_zeros(())
    for kernel in (4, 8):
        low_fake = F.avg_pool2d((clean_fake + 1) * .5, kernel)
        low_real = F.avg_pool2d((real + 1) * .5, kernel)
        low = low + F.smooth_l1_loss(_distribution_signature(low_fake), _distribution_signature(low_real))
    low = low / 2
    residual = F.smooth_l1_loss(_residual_signature(fake), _residual_signature(real))
    features = fake.new_zeros(())
    for fake_map, real_map in zip(fake_pyramid, real_pyramid):
        dims = (2, 3)
        fake_moments = torch.cat((fake_map.mean(dims), fake_map.std(dims)), dim=1)
        real_moments = torch.cat((real_map.mean(dims), real_map.std(dims)), dim=1).detach()
        features = features + F.smooth_l1_loss(fake_moments, real_moments)
    features = features / len(fake_pyramid)
    return .3 * low + .3 * residual + .4 * features, low, residual, features


def angle_curvature_loss(left: torch.Tensor, centre: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Penalise angular jumps without rewarding a constant azimuth response."""
    left, centre, right = (F.avg_pool2d(value, 4) for value in (left, centre, right))
    return F.smooth_l1_loss(left + right, 2 * centre)


def sar_perceptual_pyramid_loss(fake_pyramid: tuple[torch.Tensor, ...],
                                real_pyramid: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Match real-SAR low/mid-level content and feature distributions.

    Global classifier logits are easy for G to exploit.  Stage-wise normalized
    content plus channel moments constrain local scattering geometry and texture
    at several receptive-field sizes.
    """
    if len(fake_pyramid) != len(real_pyramid):
        raise ValueError("fake and real SAR feature pyramids must have equal depth")
    total = fake_pyramid[0].new_zeros(())
    weights = (1.0, .75, .5, .25)
    for weight, fake, real in zip(weights, fake_pyramid, real_pyramid):
        real = real.detach()
        fake_content = F.normalize(fake, dim=1)
        real_content = F.normalize(real, dim=1)
        content = F.l1_loss(fake_content, real_content)
        dims = (2, 3)
        moments = (F.l1_loss(fake.mean(dims), real.mean(dims))
                   + F.l1_loss(fake.std(dims), real.std(dims)))
        total = total + weight * (content + .5 * moments)
    return total / sum(weights)


def aligned_physics_terms(fake: torch.Tensor, real: torch.Tensor) -> dict[str, torch.Tensor]:
    """Expose V1 physics-prior subterms for controlled ablations.

    This intentionally retains V1's weak translation selection.  In
    particular, ``scatter`` is the position-sensitive scattering-map term
    that must be tested independently rather than being removed together with
    the amplitude-distribution and local-correlation priors.
    """
    real = _align_translation(fake, real)
    fake_a, real_a = (fake + 1) * .5, (real + 1) * .5
    fake_log, real_log = torch.log(fake_a.clamp_min(1e-4)), torch.log(real_a.clamp_min(1e-4))
    dims = (2, 3)
    amplitude = (F.l1_loss(fake_log.mean(dims), real_log.mean(dims))
                 + F.l1_loss(fake_log.std(dims), real_log.std(dims)))
    # Local positive residuals represent bright attributed scattering centres.
    fake_scatter = F.relu(fake_a - F.avg_pool2d(fake_a, 9, stride=1, padding=4))
    real_scatter = F.relu(real_a - F.avg_pool2d(real_a, 9, stride=1, padding=4))
    scatter = 0.0
    for size, weight in ((1, 1.0), (2, .5), (4, .25)):
        if size > 1:
            x, y = F.avg_pool2d(fake_scatter, size), F.avg_pool2d(real_scatter, size)
        else:
            x, y = fake_scatter, real_scatter
        # Separate centre placement from total return strength.
        x = x / x.sum(dims, keepdim=True).clamp_min(1e-5)
        y = y / y.sum(dims, keepdim=True).clamp_min(1e-5)
        scatter = scatter + weight * F.l1_loss(x, y)

    def correlation(image: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        shifted = image[..., dy:, dx:]
        source = image[..., :image.shape[-2] - dy, :image.shape[-1] - dx]
        source = source - source.mean(dims, keepdim=True)
        shifted = shifted - shifted.mean(dims, keepdim=True)
        return (source * shifted).mean(dims) / (source.std(dims) * shifted.std(dims) + 1e-5)

    correlation_loss = sum(F.l1_loss(correlation(fake_log, dy, dx), correlation(real_log, dy, dx))
                           for dy, dx in ((0, 1), (1, 0), (1, 1))) / 3
    return {
        "amplitude": amplitude,
        "scatter": scatter,
        "correlation": correlation_loss,
    }


def weighted_physics_prior_loss(
        fake: torch.Tensor, real: torch.Tensor, *, amplitude_weight: float = 1.0,
        scatter_weight: float = 1.0, correlation_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The V1 physics prior with independently switchable subterms.

    Default weights reproduce :func:`sar_physics_prior_loss` exactly.  The
    caller can therefore isolate the exact scattering-map L1 without changing
    the broader amplitude and short-range speckle assumptions.
    """
    terms = aligned_physics_terms(fake, real)
    total = (amplitude_weight * terms["amplitude"]
             + scatter_weight * terms["scatter"]
             + correlation_weight * terms["correlation"])
    return total, terms


def sar_physics_prior_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Physics-inspired amplitude, scattering-centre and speckle constraints."""
    total, _ = weighted_physics_prior_loss(fake, real)
    return total


def initialise(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

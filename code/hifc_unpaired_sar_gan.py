"""HiFC-inspired RGB-to-SAR building blocks for unregistered data.

The original HiFC-GAN paper describes two useful ideas for optical-to-SAR
translation: shallow local-texture contrast (LTC) and deep semantic feature
mapping (SFM).  This module adapts those ideas to the SOC data where an RGB
side view and a SAR ROI are not pixel registered.

The implementation deliberately keeps the image comparison terms spatially
agnostic.  It compares local-contrast moments and deep feature statistics; it
never computes an HxW-to-HxW RGB/SAR reconstruction loss.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from dual_component_sar_gan import LargeRGBIdentityEncoder
from joint_models import initialise
from one_stage_wavelet_sar_gan import OneStageWaveletSARGenerator, haar_texture
from v4_spade_gan import ProjectionPatchDiscriminator


HIFC_ARCHITECTURE = "hifc_unpaired_conditioned_v1"
CONDITION_DIM = 12
DEPRESSION_VALUES = (15, 30, 45, 60)


def condition_group_code(class_id: torch.Tensor, meta: torch.Tensor,
                         depression: torch.Tensor) -> torch.Tensor:
    """Encode the non-angular SAR acquisition group as one stable integer.

    The group deliberately excludes azimuth: azimuth is continuous/circular
    and is supervised by the geometry heads.  Grouping by class, band,
    polarization and depression supplies enough real examples for a set loss
    without pretending that an RGB/SAR pair is pixel registered.
    """
    band = (1 - meta[:, 3].round().long()).clamp(0, 1)
    polarization = meta[:, 4:8].argmax(1)
    dep = torch.round(depression.float() / 15.0).long().sub(1).clamp(0, 3)
    return (((class_id.long() * 2 + band) * 4 + polarization) * 4 + dep)


class ConditionalPrototypeBank:
    """Frozen real-SAR statistics used by the conditional set SFM loss.

    ``codes`` are the values returned by :func:`condition_group_code`.  The
    bank stores native-teacher and LTC means/stds, plus global fallbacks for a
    condition that is absent from a particular filtered training split.
    """

    def __init__(self, codes: torch.Tensor, embedding_mean: torch.Tensor,
                 embedding_std: torch.Tensor, ltc_mean: torch.Tensor,
                 ltc_std: torch.Tensor, global_embedding_mean: torch.Tensor,
                 global_embedding_std: torch.Tensor,
                 global_ltc_mean: torch.Tensor, global_ltc_std: torch.Tensor,
                 device: torch.device | None = None) -> None:
        self.codes = codes.long()
        self.embedding_mean = embedding_mean.float()
        self.embedding_std = embedding_std.float()
        self.ltc_mean = ltc_mean.float()
        self.ltc_std = ltc_std.float()
        self.global_embedding_mean = global_embedding_mean.float()
        self.global_embedding_std = global_embedding_std.float()
        self.global_ltc_mean = global_ltc_mean.float()
        self.global_ltc_std = global_ltc_std.float()
        self._index = {int(code): index for index, code in enumerate(self.codes.tolist())}
        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "ConditionalPrototypeBank":
        for name in ("codes", "embedding_mean", "embedding_std", "ltc_mean",
                     "ltc_std", "global_embedding_mean", "global_embedding_std",
                     "global_ltc_mean", "global_ltc_std"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def lookup(self, codes: torch.Tensor) -> tuple[torch.Tensor, ...]:
        indices = torch.tensor([self._index.get(int(code), -1)
                                for code in codes.detach().cpu().tolist()],
                               dtype=torch.long, device=codes.device)
        present = indices >= 0
        safe = indices.clamp_min(0)
        embedding_mean = self.embedding_mean[safe].clone()
        embedding_std = self.embedding_std[safe].clone()
        ltc_mean = self.ltc_mean[safe].clone()
        ltc_std = self.ltc_std[safe].clone()
        if (~present).any():
            embedding_mean[~present] = self.global_embedding_mean
            embedding_std[~present] = self.global_embedding_std
            ltc_mean[~present] = self.global_ltc_mean
            ltc_std[~present] = self.global_ltc_std
        return embedding_mean, embedding_std, ltc_mean, ltc_std

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "codes": self.codes.detach().cpu(),
            "embedding_mean": self.embedding_mean.detach().cpu(),
            "embedding_std": self.embedding_std.detach().cpu(),
            "ltc_mean": self.ltc_mean.detach().cpu(),
            "ltc_std": self.ltc_std.detach().cpu(),
            "global_embedding_mean": self.global_embedding_mean.detach().cpu(),
            "global_embedding_std": self.global_embedding_std.detach().cpu(),
            "global_ltc_mean": self.global_ltc_mean.detach().cpu(),
            "global_ltc_std": self.global_ltc_std.detach().cpu(),
        }


def _fixed_sfm_projections(dim: int, count: int, device: torch.device) -> torch.Tensor:
    """Return deterministic normalized projections without touching training RNG."""
    key = (dim, count, str(device))
    cache = getattr(_fixed_sfm_projections, "_cache", {})
    if key not in cache:
        generator = torch.Generator(device="cpu").manual_seed(20260901 + dim + count)
        projection = torch.randn(count, dim, generator=generator)
        projection = F.normalize(projection, dim=1)
        cache[key] = projection.to(device)
        _fixed_sfm_projections._cache = cache
    return cache[key]


def _ddp_set_gather(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Gather a set while retaining the local fake branch's autograd path."""
    if not dist.is_available() or not dist.is_initialized():
        return value, False
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    # ``all_gather`` itself has no autograd support.  Replacing the local slot
    # with the original tensor gives this rank the correct fake gradient; DDP
    # averages the corresponding partial gradients across ranks.
    dist.all_gather(gathered, value.detach())
    gathered[dist.get_rank()] = value
    return torch.cat(gathered, dim=0), True


class HIFCUnpairedGenerator(OneStageWaveletSARGenerator):
    """One-pass clean reflectivity plus stochastic observed SAR generator.

    RGB features are injected at all decoder scales.  The geometry vector is
    target-only: target azimuth, depression, band and polarisation.  The
    source RGB view angle and SAR bounding-box dimensions are intentionally
    absent, because neither is a reliable target-domain condition here.
    """

    def __init__(self, identity_dim: int = 512, base: int = 64) -> None:
        super().__init__(identity_dim=identity_dim, geometry_dim=CONDITION_DIM,
                         rgb_base=64, base=base)


class HIFCConditionedDiscriminator(nn.Module):
    """One shared conditional critic for realism and class/geometry matching."""

    def __init__(self, classes: int = 40, base: int = 64) -> None:
        super().__init__()
        self.critic = ProjectionPatchDiscriminator(
            classes=classes, geometry_dim=CONDITION_DIM, base=base)

    def forward(self, image: torch.Tensor, class_id: torch.Tensor,
                condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic(image, class_id, condition)


def condition_from_batch(meta: torch.Tensor,
                          depression: torch.Tensor) -> torch.Tensor:
    """Encode only target SAR acquisition variables.

    ``metadata_vector`` stores X as ``band_is_x=1`` and the four one-hot
    polarisation entries in HH/HV/VH/VV order.  The classifier convention is
    X=0, KU=1, so the band one-hot below is [X, KU].
    """
    if meta.ndim != 2 or meta.shape[1] < 8:
        raise ValueError(f"expected metadata [B,>=8], got {tuple(meta.shape)}")
    azimuth = meta[:, :2]
    dep = depression.to(device=meta.device).long()
    dep_index = torch.round(dep.float() / 15.0).long().sub(1).clamp(0, 3)
    dep_one_hot = F.one_hot(dep_index, num_classes=4).to(meta.dtype)
    x_indicator = meta[:, 3:4].clamp(0, 1)
    band_one_hot = torch.cat((x_indicator, 1.0 - x_indicator), dim=1)
    polarisation = meta[:, 4:8].clamp(0, 1)
    condition = torch.cat((azimuth, dep_one_hot, band_one_hot, polarisation), dim=1)
    if condition.shape[1] != CONDITION_DIM:
        raise RuntimeError(f"condition encoder produced {condition.shape[1]} channels")
    return condition


def local_texture_signature(image: torch.Tensor) -> torch.Tensor:
    """Return per-image local contrast statistics, without spatial matching."""
    amplitude = ((image + 1.0) * .5).clamp(0, 1)
    local3 = F.avg_pool2d(amplitude, 3, stride=1, padding=1)
    local7 = F.avg_pool2d(amplitude, 7, stride=1, padding=3)
    residual3 = amplitude - local3
    residual7 = amplitude - local7
    contrast3 = residual3 / (local3.abs() + .04)
    contrast7 = residual7 / (local7.abs() + .04)
    haar = haar_texture(image)

    def moments(value: torch.Tensor) -> torch.Tensor:
        dims = (2, 3)
        return torch.cat((value.mean(dims), value.std(dims, unbiased=False),
                          value.abs().mean(dims)), dim=1)

    # Each row is a global signature for one image; no pixel coordinate is
    # compared to the corresponding coordinate in a real image.
    return torch.cat((moments(residual3), moments(residual7),
                      moments(contrast3), moments(contrast7),
                      moments(haar)), dim=1)


def local_texture_contrast_loss(fake: torch.Tensor,
                                real: torch.Tensor) -> torch.Tensor:
    """HiFC shallow LTC adapted to unregistered RGB/SAR observations."""
    fake_signature = local_texture_signature(fake)
    real_signature = local_texture_signature(real).detach()
    fake_batch = torch.cat((fake_signature.mean(0),
                            fake_signature.std(0, unbiased=False)))
    real_batch = torch.cat((real_signature.mean(0),
                            real_signature.std(0, unbiased=False)))
    return F.smooth_l1_loss(fake_batch, real_batch)


def feature_moment_loss(fake: torch.Tensor,
                        real: torch.Tensor) -> torch.Tensor:
    """Match deep feature distributions using channel moments only."""
    if fake.ndim != 4 or real.ndim != 4 or fake.shape[1:] != real.shape[1:]:
        raise ValueError("feature moment inputs must have equal [B,C,H,W] shape")
    dims = (2, 3)
    fake_stats = torch.cat((fake.mean(dims), fake.std(dims, unbiased=False)), 1)
    real_stats = torch.cat((real.detach().mean(dims),
                            real.detach().std(dims, unbiased=False)), 1)
    return F.smooth_l1_loss(fake_stats, real_stats)


def semantic_feature_mapping_loss(fake_teacher_feature: torch.Tensor,
                                  real_teacher_feature: torch.Tensor,
                                  fake_discriminator_feature: torch.Tensor | None = None,
                                  real_discriminator_feature: torch.Tensor | None = None,
                                  teacher_gradient: bool = True
                                  ) -> torch.Tensor:
    """HiFC deep SFM without an image-alignment assumption.

    The native SAR teacher contributes its pre-classifier embedding rather
    than a hard class CE.  A conditional discriminator feature-moment term is
    added when available.  Both terms are global; the native teacher parameters
    and real features are detached by the caller.
    """
    # The native teacher is a useful real-SAR representation, but its
    # decision boundary can become a generator shortcut.  ``teacher_gradient``
    # therefore controls only the path through its embedding; discriminator
    # feature moments below remain differentiable in either mode.
    fake_embedding = (fake_teacher_feature if teacher_gradient
                      else fake_teacher_feature.detach())
    fake_norm = F.normalize(fake_embedding, dim=1)
    real_norm = F.normalize(real_teacher_feature.detach(), dim=1)
    cosine = 1.0 - (fake_norm * real_norm).sum(1).mean()
    batch_mean = F.smooth_l1_loss(fake_norm.mean(0), real_norm.mean(0))
    total = cosine + .5 * batch_mean
    if fake_discriminator_feature is not None and real_discriminator_feature is not None:
        total = total + .5 * feature_moment_loss(
            fake_discriminator_feature, real_discriminator_feature)
    return total


def conditional_set_sfm_loss(
        fake_teacher_feature: torch.Tensor,
        real_teacher_feature: torch.Tensor,
        fake_ltc_signature: torch.Tensor,
        real_ltc_signature: torch.Tensor,
        fake_group_code: torch.Tensor,
        real_group_code: torch.Tensor,
        prototypes: ConditionalPrototypeBank,
        fake_discriminator_feature: torch.Tensor | None = None,
        real_discriminator_feature: torch.Tensor | None = None,
        teacher_gradient: bool = True,
        projection_count: int = 16,
        ltc_weight: float = .50,
        anchor_weight: float = .25,
        ddp_global_set: bool = False) -> torch.Tensor:
    """Conditional set-level SFM for genuinely unpaired RGB/SAR examples.

    A conventional SFM term compares item ``i`` with item ``i``.  That is an
    implicit pairing assumption and can teach the generator a shortcut when
    the two items only share a class label.  This replacement first removes a
    frozen real-SAR prototype for each acquisition group and whitens both
    domains.  A fixed sliced-Wasserstein distance then compares the sorted
    projected *sets*; no image coordinate or batch position is matched.  The
    prototype cosine anchor retains the class/condition identity signal.

    The discriminator feature-moment term is intentionally unchanged and is
    included when the shared critic supplies features.  Real tensors and the
    prototype bank are detached, so only fake/E/G receives this loss gradient.
    """
    if fake_teacher_feature.ndim != 2 or real_teacher_feature.ndim != 2:
        raise ValueError("teacher features must have shape [B,C]")
    if fake_teacher_feature.shape[0] != fake_ltc_signature.shape[0]:
        raise ValueError("fake teacher/LTC batch sizes differ")
    fake_embedding = (fake_teacher_feature if teacher_gradient
                      else fake_teacher_feature.detach())
    real_embedding = real_teacher_feature.detach()
    fake_ltc = fake_ltc_signature if teacher_gradient else fake_ltc_signature.detach()
    real_ltc = real_ltc_signature.detach()
    fake_mu, fake_std, fake_ltc_mu, fake_ltc_std = prototypes.lookup(fake_group_code)
    real_mu, real_std, real_ltc_mu, real_ltc_std = prototypes.lookup(real_group_code)
    eps = 1e-3
    fake_e = (fake_embedding - fake_mu) / fake_std.clamp_min(eps)
    real_e = (real_embedding - real_mu) / real_std.clamp_min(eps)
    fake_t = (fake_ltc - fake_ltc_mu) / fake_ltc_std.clamp_min(eps)
    real_t = (real_ltc - real_ltc_mu) / real_ltc_std.clamp_min(eps)

    # Equalize the contribution of the 384-D teacher vector and the 15-D LTC
    # signature before projecting.  Whitening makes the comparison invariant
    # to per-channel native-teacher scale while retaining residual structure.
    fake_joint = torch.cat((fake_e / math.sqrt(fake_e.shape[1]),
                            ltc_weight * fake_t / math.sqrt(fake_t.shape[1])), dim=1)
    real_joint = torch.cat((real_e / math.sqrt(real_e.shape[1]),
                            ltc_weight * real_t / math.sqrt(real_t.shape[1])), dim=1)
    if ddp_global_set:
        fake_joint, gathered = _ddp_set_gather(fake_joint)
        real_joint, real_gathered = _ddp_set_gather(real_joint)
        if gathered != real_gathered:
            raise RuntimeError("SFM DDP gather state differs between fake and real sets")
    else:
        gathered = False
    projections = _fixed_sfm_projections(fake_joint.shape[1], projection_count,
                                         fake_joint.device).to(fake_joint.dtype)
    fake_projected = fake_joint @ projections.t()
    real_projected = real_joint @ projections.t()
    # In one dimension the optimal transport plan is sorting.  smooth_l1 is a
    # robust approximation of the 1-Wasserstein cost and remains stable for
    # the small per-rank DDP batch sizes used by this trainer.
    fake_sorted = fake_projected.sort(dim=0).values
    real_sorted = real_projected.sort(dim=0).values
    sliced = F.smooth_l1_loss(fake_sorted, real_sorted)
    if gathered:
        # DDP averages gradients across ranks.  The global set loss is already
        # a mean over all ranks, so restore the baseline gradient scale.
        sliced = sliced * dist.get_world_size()

    fake_norm = F.normalize(fake_embedding, dim=1)
    prototype_norm = F.normalize(fake_mu.detach(), dim=1)
    anchor = 1.0 - (fake_norm * prototype_norm).sum(1).mean()
    total = sliced + anchor_weight * anchor
    if fake_discriminator_feature is not None and real_discriminator_feature is not None:
        total = total + .5 * feature_moment_loss(
            fake_discriminator_feature, real_discriminator_feature)
    return total


def geometry_auxiliary_loss(teacher: nn.Module, fake: torch.Tensor,
                            meta: torch.Tensor,
                            depression: torch.Tensor,
                            azimuth: torch.Tensor,
                            teacher_gradient: bool = True
                            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teach band/polarisation/depression/azimuth from a real-only teacher.

    This is deliberately separate from the class objective.  A class CE would
    encourage the old native-classifier shortcut; the geometry heads add
    acquisition information that a class-only proxy cannot provide.
    """
    if teacher_gradient:
        teacher_logits, features = teacher((fake + 1.0) * .5,
                                           return_features=True)
    else:
        with torch.no_grad():
            teacher_logits, features = teacher(
                ((fake + 1.0) * .5).clamp(0, 1), return_features=True)
    del teacher_logits
    band = (1 - meta[:, 3].round().long()).clamp(0, 1)
    pol = meta[:, 4:8].argmax(1)
    dep = torch.round(depression.float() / 15.0).long().sub(1).clamp(0, 3)
    az = ((azimuth.long() + 15) % 360) // 30
    band_logits, pol_logits, dep_logits, az_logits = teacher.auxiliary_logits(features)
    losses = {
        "band": F.cross_entropy(band_logits, band),
        "polarization": F.cross_entropy(pol_logits, pol),
        "depression": F.cross_entropy(dep_logits, dep),
        "azimuth": F.cross_entropy(az_logits, az),
    }
    return sum(losses.values()) / len(losses), losses


def rgb_identity_loss(logits: torch.Tensor, alt_logits: torch.Tensor,
                      identity: torch.Tensor, alt_identity: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
    """One merged RGB identity term for two independent source views."""
    class_loss = .5 * (F.cross_entropy(logits, labels, label_smoothing=.03)
                       + F.cross_entropy(alt_logits, labels, label_smoothing=.03))
    view_invariance = 1.0 - (
        F.normalize(identity, dim=1) * F.normalize(alt_identity, dim=1)
    ).sum(1).mean()
    return class_loss + .5 * view_invariance


def differentiable_radiometry(image: torch.Tensor) -> torch.Tensor:
    """Light label-preserving augmentation shared by real and fake D inputs."""
    batch = len(image)
    amplitude = ((image + 1.0) * .5).clamp(0, 1)
    gain = amplitude.new_empty(batch, 1, 1, 1).uniform_(.92, 1.08)
    bias = amplitude.new_empty(batch, 1, 1, 1).uniform_(-.025, .025)
    return (amplitude * gain + bias).clamp(0, 1) * 2.0 - 1.0


def discriminator_hinge(real_score: torch.Tensor,
                        fake_score: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def set_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.lerp_(source_parameter, 1.0 - decay)
    for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
        target_buffer.copy_(source_buffer)


def initialise_hifc(encoder: nn.Module, generator: nn.Module,
                    discriminator: nn.Module) -> None:
    encoder.apply(initialise)
    generator.apply(initialise)
    discriminator.apply(initialise)

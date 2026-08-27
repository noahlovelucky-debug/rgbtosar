"""Independent real-SAR identity, azimuth, and depression validator."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from sar_classifier_64 import SARClassifier64


AZIMUTH_BINS = 72
AZIMUTH_STEP = 360.0 / AZIMUTH_BINS
DEPRESSION_VALUES = (15, 30, 45, 60)


class GeometryValidatorOutput(NamedTuple):
    identity_logits: torch.Tensor
    depression_logits: torch.Tensor
    azimuth_logits: torch.Tensor
    azimuth_vector: torch.Tensor
    features: torch.Tensor


class SARGeometryValidator(nn.Module):
    """Image-only judge trained exclusively on real 64x64 SAR.

    This model is frozen while training the GAN.  It supplies identity features
    and explicit continuous/discrete geometry targets without receiving any
    metadata as input.
    """

    def __init__(self, classes: int = 40) -> None:
        super().__init__()
        self.backbone = SARClassifier64(classes)
        feature_dim = self.backbone.feature_dim
        self.depression_head = nn.Linear(feature_dim, len(DEPRESSION_VALUES))
        self.azimuth_head = nn.Linear(feature_dim, AZIMUTH_BINS)
        self.azimuth_vector_head = nn.Linear(feature_dim, 2)

    def forward(self, image: torch.Tensor) -> GeometryValidatorOutput:
        identity_logits, features = self.backbone(
            image, return_features=True)
        azimuth_vector = F.normalize(
            self.azimuth_vector_head(features), dim=1, eps=1e-6)
        return GeometryValidatorOutput(
            identity_logits,
            self.depression_head(features),
            self.azimuth_head(features),
            azimuth_vector,
            features)


def circular_bin_distance(logits: torch.Tensor,
                          target_bin: torch.Tensor) -> torch.Tensor:
    predicted = logits.argmax(1)
    direct = (predicted - target_bin).abs()
    return torch.minimum(direct, AZIMUTH_BINS - direct)


def circular_degree_error(prediction: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
    """Angular error for [sin(theta), cos(theta)] vectors."""
    cosine = (prediction * target).sum(1).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cosine))


def circular_soft_cross_entropy(
        logits: torch.Tensor, target_bin: torch.Tensor,
        sigma_bins: float = 1.25) -> torch.Tensor:
    """Circular Gaussian labels avoid treating 355° and 0° as unrelated."""
    bins = torch.arange(
        AZIMUTH_BINS, device=logits.device, dtype=logits.dtype)[None]
    centre = target_bin.to(logits.dtype)[:, None]
    distance = (bins - centre).abs()
    distance = torch.minimum(distance, AZIMUTH_BINS - distance)
    weights = torch.exp(-.5 * (distance / sigma_bins).square())
    weights = weights / weights.sum(1, keepdim=True)
    return -(weights * logits.log_softmax(1)).sum(1).mean()


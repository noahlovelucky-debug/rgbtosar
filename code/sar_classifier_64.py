"""Native 64x64 image-only SAR classifier used as an independent GAN judge."""
from __future__ import annotations

from math import gcd

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    return max(1, min(16, gcd(channels, 16)))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(16, channels // 8)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                 nn.SiLU(inplace=True), nn.Conv2d(hidden, channels, 1),
                                 nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(output_channels), output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(output_channels), output_channels)
        self.se = SqueezeExcite(output_channels)
        self.skip = (nn.Identity() if stride == 1 and input_channels == output_channels else
                     nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, stride, bias=False),
                                   nn.GroupNorm(_groups(output_channels), output_channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)), inplace=True)
        x = self.se(self.norm2(self.conv2(x)))
        return F.silu(x + residual, inplace=True)


class SARClassifier64(nn.Module):
    """Residual image classifier; metadata heads are training-only auxiliaries.

    The network is deliberately fed only the one-channel SAR intensity image.
    Band/polarisation/depression/azimuth labels are targets, never inputs.
    """

    feature_dim = embedding_dim = 384

    def __init__(self, classes: int = 40) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 48, 3, 1, 1, bias=False),
                                  nn.GroupNorm(_groups(48), 48), nn.SiLU(inplace=True))
        channels = (48, 96, 192, 384)
        depths = (2, 2, 3, 2)
        stages: list[nn.Module] = []
        previous = 48
        for index, (channel, depth) in enumerate(zip(channels, depths)):
            blocks = [ResidualBlock(previous, channel, 1 if index == 0 else 2)]
            blocks.extend(ResidualBlock(channel, channel) for _ in range(depth - 1))
            stages.append(nn.Sequential(*blocks))
            previous = channel
        self.stages = nn.Sequential(*stages)
        self.embedding = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.LayerNorm(channels[-1]), nn.Dropout(0.15))
        self.classifier = nn.Linear(channels[-1], classes)
        self.band_head = nn.Linear(channels[-1], 2)
        self.polarization_head = nn.Linear(channels[-1], 4)
        self.depression_head = nn.Linear(channels[-1], 4)
        self.azimuth_head = nn.Linear(channels[-1], 12)

    def forward(self, image: torch.Tensor, return_features: bool = False,
                return_pyramid: bool = False):
        x = self.stem(image)
        pyramid = []
        for stage in self.stages:
            x = stage(x)
            pyramid.append(x)
        features = self.embedding(x)
        logits = self.classifier(features)
        if return_pyramid:
            return logits, features, tuple(pyramid)
        if return_features:
            return logits, features
        return logits

    def auxiliary_logits(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.band_head(features), self.polarization_head(features),
                self.depression_head(features), self.azimuth_head(features))


class SARClassDiscriminator64(SARClassifier64):
    """A K+1 SAR classifier-discriminator initialised from ``SARClassifier64``.

    The first ``classes`` logits describe real vehicle classes and the final
    logit describes synthetic SAR. Classification and discrimination therefore
    use one SAR backbone. The condition is target SAR geometry only: target
    azimuth sin/cos and depression; source RGB-view angle is deliberately not
    exposed to this network.
    """

    def __init__(self, classes: int = 40, condition_dim: int = 3) -> None:
        super().__init__(classes)
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, self.feature_dim), nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim * 2),
        )
        self.fake_classifier = nn.Linear(self.feature_dim, 1)
        # A zero FiLM adapter preserves the native classifier at transfer time.
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        nn.init.zeros_(self.fake_classifier.weight)
        nn.init.constant_(self.fake_classifier.bias, -2.0)

    def forward(self, image: torch.Tensor, condition: torch.Tensor,
                return_pyramid: bool = False):
        x = self.stem(image)
        pyramid = []
        for stage in self.stages:
            x = stage(x)
            pyramid.append(x)
        features = self.embedding(x)
        scale, bias = self.condition(condition).chunk(2, dim=1)
        features = features * (1.0 + .1 * torch.tanh(scale)) + bias
        logits = torch.cat((self.classifier(features), self.fake_classifier(features)), dim=1)
        if return_pyramid:
            return logits, features, tuple(pyramid)
        return logits, features

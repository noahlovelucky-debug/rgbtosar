"""Regression checks for V1-compatible loss decomposition."""
from __future__ import annotations

import torch

from joint_models import (
    multiscale_structure_loss,
    sar_physics_prior_loss,
    weighted_aligned_structure_loss,
    weighted_physics_prior_loss,
)


def test_default_structure_decomposition_matches_v1() -> None:
    torch.manual_seed(17)
    fake = torch.rand(3, 1, 64, 64) * 2 - 1
    real = torch.rand(3, 1, 64, 64) * 2 - 1
    original = multiscale_structure_loss(fake, real)
    decomposed, terms = weighted_aligned_structure_loss(fake, real)
    torch.testing.assert_close(decomposed, original, atol=1e-7, rtol=1e-6)
    assert set(terms) == {"pixel_64", "pixel_32", "pixel_16", "edge", "ssim"}


def test_default_physics_decomposition_matches_v1() -> None:
    torch.manual_seed(19)
    fake = torch.rand(3, 1, 64, 64) * 2 - 1
    real = torch.rand(3, 1, 64, 64) * 2 - 1
    original = sar_physics_prior_loss(fake, real)
    decomposed, terms = weighted_physics_prior_loss(fake, real)
    torch.testing.assert_close(decomposed, original, atol=1e-7, rtol=1e-6)
    assert set(terms) == {"amplitude", "scatter", "correlation"}

"""Regression checks for V1-compatible loss decomposition."""
from __future__ import annotations

import torch
from torch import nn

from joint_models import (
    ContinuousROIDiscriminator,
    multiscale_structure_loss,
    sar_physics_prior_loss,
    weighted_aligned_structure_loss,
    weighted_physics_prior_loss,
)
from train_continuous_spatial_v1_ablation import (
    combine_rgb_losses,
    load_parent_discriminator,
    route_generator_inputs,
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


def test_generator_only_routing_blocks_encoder_gradient() -> None:
    torch.manual_seed(23)
    encoder = nn.Linear(3, 4)
    generator = nn.Linear(4, 1)
    identity = encoder(torch.randn(5, 3))
    pyramid = (identity[:, :, None, None],)
    routed_identity, routed_pyramid = route_generator_inputs(identity, pyramid, "generator_only")
    output = generator(routed_identity) + routed_pyramid[0].mean()
    output.mean().backward()
    assert encoder.weight.grad is None
    assert generator.weight.grad is not None
    assert float(generator.weight.grad.abs().sum()) > 0.0


def test_coupled_routing_preserves_encoder_gradient() -> None:
    torch.manual_seed(29)
    encoder = nn.Linear(3, 4)
    generator = nn.Linear(4, 1)
    identity = encoder(torch.randn(5, 3))
    pyramid = (identity[:, :, None, None],)
    routed_identity, routed_pyramid = route_generator_inputs(identity, pyramid, "coupled")
    output = generator(routed_identity) + routed_pyramid[0].mean()
    output.mean().backward()
    assert encoder.weight.grad is not None
    assert float(encoder.weight.grad.abs().sum()) > 0.0


def test_joint_rgb_loss_is_weighted_equivalent_to_v1_terms() -> None:
    identity = torch.tensor(0.17, requires_grad=True)
    cross_view = torch.tensor(0.031, requires_grad=True)
    separate_identity, separate_cross = combine_rgb_losses(identity, cross_view, 10.0, 2.0, "separate")
    joint_identity, joint_cross = combine_rgb_losses(identity, cross_view, 10.0, 2.0, "joint_equivalent")
    separate_total = 10.0 * separate_identity + 2.0 * separate_cross
    joint_total = 10.0 * joint_identity + 2.0 * joint_cross
    torch.testing.assert_close(joint_total, separate_total)
    torch.testing.assert_close(joint_cross, torch.zeros_like(joint_cross))


def test_patchgan_auxiliary_class_head_does_not_change_score() -> None:
    torch.manual_seed(31)
    discriminator = ContinuousROIDiscriminator(meta_dim=12)
    discriminator.eval()
    roi = torch.randn(4, 1, 64, 64)
    meta = torch.randn(4, 12)
    score_without_head, features = discriminator(roi, meta)
    score_with_head, same_features, logits = discriminator(roi, meta, return_class_logits=True)
    torch.testing.assert_close(score_with_head, score_without_head)
    torch.testing.assert_close(same_features, features)
    torch.testing.assert_close(logits, torch.zeros_like(logits))


def test_archived_discriminator_migration_zeros_missing_class_head() -> None:
    torch.manual_seed(37)
    source = ContinuousROIDiscriminator(meta_dim=12)
    archived_state = {name: value.clone() for name, value in source.state_dict().items()
                      if not name.startswith("classifier.")}
    target = ContinuousROIDiscriminator(meta_dim=12)
    with torch.no_grad():
        target.classifier.weight.fill_(1.0)
        target.classifier.bias.fill_(1.0)
    migration = load_parent_discriminator(target, archived_state, "full")
    assert "zero auxiliary class head" in migration
    torch.testing.assert_close(target.classifier.weight, torch.zeros_like(target.classifier.weight))
    torch.testing.assert_close(target.classifier.bias, torch.zeros_like(target.classifier.bias))

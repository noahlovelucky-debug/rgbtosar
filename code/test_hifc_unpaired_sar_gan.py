"""Fast tensor-level checks for the HiFC unpaired path."""
from __future__ import annotations

import torch

from hifc_unpaired_sar_gan import (
    CONDITION_DIM, HIFCConditionedDiscriminator, HIFCUnpairedGenerator,
    condition_from_batch, geometry_auxiliary_loss,
    local_texture_contrast_loss, semantic_feature_mapping_loss)
from dual_component_sar_gan import LargeRGBIdentityEncoder
from sar_classifier_64 import SARClassifier64


def main() -> None:
    torch.manual_seed(7)
    batch = 1
    encoder = LargeRGBIdentityEncoder(40)
    generator = HIFCUnpairedGenerator()
    discriminator = HIFCConditionedDiscriminator()
    teacher = SARClassifier64(40).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    rgb = torch.randn(batch, 3, 128, 128)
    rgb_alt = torch.randn_like(rgb)
    meta = torch.zeros(batch, 10)
    meta[:, 0] = 1.0
    meta[:, 3] = 1.0  # X
    meta[:, 4] = 1.0  # HH
    labels = torch.tensor([3])
    depression = torch.tensor([30])
    azimuth = torch.tensor([0])
    condition = condition_from_batch(meta, depression)
    assert condition.shape == (batch, CONDITION_DIM)
    identity, logits, pyramid = encoder(rgb, return_pyramid=True)
    alt_identity, alt_logits = encoder(rgb_alt)
    clean, _, fake, _ = generator(identity, condition, pyramid,
                                   torch.randn(batch, 1, 64, 64))
    score, d_feature = discriminator(fake, labels, condition)
    with torch.no_grad():
        _, real_teacher_feature = teacher(torch.rand_like(fake), return_features=True)
    _, fake_teacher_feature = teacher((fake + 1) * .5, return_features=True)
    loss = (
        score.mean() * 0.0
        + local_texture_contrast_loss(fake, torch.rand_like(fake))
        + semantic_feature_mapping_loss(fake_teacher_feature,
                                         real_teacher_feature, d_feature,
                                         d_feature.detach())
        + .5 * (1 - torch.nn.functional.cosine_similarity(
            identity, alt_identity).mean())
        + .5 * (logits.square().mean() + alt_logits.square().mean())
    )
    loss.backward()
    assert clean.shape == fake.shape == (batch, 1, 64, 64)
    assert score.shape == (batch,)
    assert d_feature.ndim == 4
    assert any(parameter.grad is not None for parameter in generator.parameters())
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    print("hifc unpaired tensor smoke: PASS")


def test_sfm_teacher_gradient_switch() -> None:
    """Detaching the native embedding must preserve the D-feature route."""
    torch.manual_seed(11)
    fake_teacher = torch.randn(2, 384, requires_grad=True)
    real_teacher = torch.randn(2, 384)
    fake_d = torch.randn(2, 8, 4, 4, requires_grad=True)
    real_d = torch.randn(2, 8, 4, 4)
    loss = semantic_feature_mapping_loss(
        fake_teacher, real_teacher, fake_d, real_d, teacher_gradient=False)
    loss.backward()
    assert fake_teacher.grad is None or torch.allclose(
        fake_teacher.grad, torch.zeros_like(fake_teacher.grad))
    assert fake_d.grad is not None and torch.isfinite(fake_d.grad).all()


def test_geometry_teacher_gradient_switch() -> None:
    """The all-off route keeps geometry as a metric without G/E gradients."""
    class ToyTeacher(torch.nn.Module):
        def forward(self, image, return_features=False):
            features = image.mean((2, 3)).repeat(1, 384 // image.shape[1])
            logits = features[:, :40]
            return (logits, features) if return_features else logits

        def auxiliary_logits(self, features):
            return (features[:, :2], features[:, :4], features[:, :4],
                    features[:, :12])

    fake = torch.randn(2, 1, 8, 8, requires_grad=True)
    meta = torch.zeros(2, 8)
    meta[:, 3] = 1.0
    depression = torch.tensor([15, 60])
    azimuth = torch.tensor([0, 30])
    loss, _ = geometry_auxiliary_loss(
        ToyTeacher(), fake, meta, depression, azimuth, teacher_gradient=False)
    assert not loss.requires_grad


if __name__ == "__main__":
    main()

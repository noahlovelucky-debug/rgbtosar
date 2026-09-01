"""Fast tensor-level checks for the HiFC unpaired path."""
from __future__ import annotations

import torch

from hifc_unpaired_sar_gan import (
    CONDITION_DIM, ConditionalPrototypeBank, HIFCConditionedDiscriminator,
    HIFCUnpairedGenerator, condition_from_batch, conditional_set_sfm_loss,
    geometry_auxiliary_loss, local_texture_contrast_loss,
    semantic_feature_mapping_loss)
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


def test_conditional_set_sfm_is_permutation_invariant_and_differentiable() -> None:
    """C2OT compares sets, while only fake features receive its gradient."""
    torch.manual_seed(13)
    batch, embedding_dim, signature_dim = 8, 384, 15
    group_code = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    codes = group_code.unique()
    embedding_mean = torch.randn(len(codes), embedding_dim)
    embedding_std = torch.rand(len(codes), embedding_dim) + .5
    ltc_mean = torch.randn(len(codes), signature_dim)
    ltc_std = torch.rand(len(codes), signature_dim) + .5
    bank = ConditionalPrototypeBank(
        codes, embedding_mean, embedding_std, ltc_mean, ltc_std,
        embedding_mean.mean(0), embedding_std.mean(0),
        ltc_mean.mean(0), ltc_std.mean(0))
    fake = torch.randn(batch, embedding_dim, requires_grad=True)
    fake_ltc = torch.randn(batch, signature_dim, requires_grad=True)
    real = torch.randn(batch, embedding_dim)
    real_ltc = torch.randn(batch, signature_dim)
    loss = conditional_set_sfm_loss(
        fake, real, fake_ltc, real_ltc, group_code, group_code, bank,
        projection_count=8)
    loss.backward()
    assert torch.isfinite(loss)
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
    assert fake_ltc.grad is not None and torch.isfinite(fake_ltc.grad).all()
    permutation = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    with torch.no_grad():
        permuted = conditional_set_sfm_loss(
            fake.detach()[permutation], real[permutation],
            fake_ltc.detach()[permutation], real_ltc[permutation],
            group_code[permutation], group_code[permutation], bank,
            projection_count=8)
    assert torch.allclose(loss.detach(), permuted, atol=1e-6)


if __name__ == "__main__":
    main()

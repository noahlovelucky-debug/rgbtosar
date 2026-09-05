import torch
from torch.nn import functional as F

from conditional_sar_diffusion import ConditionalSARDDPM, DiffusionSchedule


def test_conditional_ddpm_forward_and_backward() -> None:
    torch.manual_seed(7)
    model = ConditionalSARDDPM(base=16, token_dim=64, rgb_base=8)
    schedule = DiffusionSchedule(steps=32)
    clean = torch.randn(2, 1, 64, 64).clamp(-1, 1)
    rgb = torch.randn(2, 3, 128, 128).clamp(-1, 1)
    condition = torch.randn(2, 12)
    timestep = torch.tensor((3, 21), dtype=torch.long)
    noisy, noise = schedule.q_sample(clean, timestep)
    prediction = model(noisy, timestep, rgb, condition, condition_drop=torch.tensor((False, True)))
    assert prediction.shape == clean.shape
    loss = F.mse_loss(prediction, noise)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ddim_sampling_is_finite_and_bounded() -> None:
    torch.manual_seed(11)
    model = ConditionalSARDDPM(base=16, token_dim=64, rgb_base=8).eval()
    schedule = DiffusionSchedule(steps=32)
    rgb = torch.randn(1, 3, 128, 128).clamp(-1, 1)
    condition = torch.randn(1, 12)
    sample = schedule.ddim_sample(model, rgb, condition, sample_steps=4)
    assert sample.shape == (1, 1, 64, 64)
    assert torch.isfinite(sample).all()
    assert sample.min() >= -1 and sample.max() <= 1


def test_ddim_sampling_accepts_shared_initial_noise() -> None:
    torch.manual_seed(19)
    model = ConditionalSARDDPM(base=16, token_dim=64, rgb_base=8).eval()
    schedule = DiffusionSchedule(steps=32)
    rgb = torch.randn(2, 3, 128, 128).clamp(-1, 1)
    condition = torch.randn(2, 12)
    noise = torch.randn(1, 1, 64, 64).expand(2, -1, -1, -1)
    sample = schedule.ddim_sample(model, rgb, condition, sample_steps=4, initial_noise=noise)
    assert sample.shape == (2, 1, 64, 64)
    assert torch.isfinite(sample).all()

import torch

from unsb_sar_bridge import (
    SilhouetteBridge, bridge_interpolation, bridge_loss, bridge_sample,
    soft_silhouette_prior,
)


def _inputs(batch=2):
    rgb = torch.randn(batch, 3, 128, 128)
    mask = torch.rand(batch, 1, 128, 128)
    alt = torch.randn(batch, 3, 128, 128)
    alt_mask = torch.rand(batch, 1, 128, 128)
    acquisition = torch.randn(batch, 12)
    acquisition[:, :2] = torch.tensor((0.0, 1.0))
    source_angle = torch.tensor((0.0, 1.0)).repeat(batch, 1)
    target = torch.randn(batch, 1, 64, 64)
    return rgb, mask, alt, alt_mask, acquisition, source_angle, target


def test_bridge_shapes_and_gradient():
    torch.manual_seed(7)
    model = SilhouetteBridge(base=8, token_dim=32, control_base=4)
    rgb, mask, alt, alt_mask, acquisition, source_angle, target = _inputs()
    timestep = torch.rand(len(target))
    prior = soft_silhouette_prior(mask)
    state, velocity = bridge_interpolation(prior, target, timestep)
    prediction, conditions = model(
        state, timestep, rgb, mask, acquisition, source_angle=source_angle,
        rgb_alt=alt, mask_alt=alt_mask, return_conditions=True)
    loss = bridge_loss(prediction, velocity) + .1 * (
        conditions.primary_logits.square().mean() + conditions.alternate_logits.square().mean())
    loss.backward()
    assert prediction.shape == target.shape
    assert prior.shape == target.shape
    assert [tuple(value.shape[-2:]) for value in conditions.controls] == [(64, 64), (32, 32), (16, 16), (8, 8)]
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.parameters())


def test_sampling_is_periodic_conditioned_and_bounded():
    torch.manual_seed(11)
    model = SilhouetteBridge(base=8, token_dim=32, control_base=4).eval()
    rgb, mask, alt, alt_mask, acquisition, source_angle, _ = _inputs(batch=1)
    batch_rgb = rgb.expand(3, -1, -1, -1)
    batch_mask = mask.expand(3, -1, -1, -1)
    batch_alt = alt.expand(3, -1, -1, -1)
    batch_alt_mask = alt_mask.expand(3, -1, -1, -1)
    batch_acquisition = acquisition.expand(3, -1).clone()
    batch_acquisition[:, 0:2] = torch.tensor(((0.0, 1.0), (1.0, 0.0), (0.0, -1.0)))
    batch_source = source_angle.expand(3, -1)
    sample = bridge_sample(model, batch_rgb, batch_mask, batch_acquisition,
                           source_angle=batch_source, steps=2, temperature=0,
                           rgb_alt=batch_alt, mask_alt=batch_alt_mask)
    assert sample.shape == (3, 1, 64, 64)
    assert torch.isfinite(sample).all()
    assert sample.abs().max() <= 1.0


"""Regression tests for Continuous Spatial V3."""
from __future__ import annotations

import io
import unittest

import torch
from torch.nn import functional as F

from continuous_spatial_one_stage_v3 import (
    CircularScaleAttention, ContinuousSpatialOneStageV3,
    OneStageContinuousSARGenerator, target_geometry)


class ContinuousSpatialV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.metadata = torch.zeros(1, 10)
        self.metadata[:, 3] = 1.0
        self.metadata[:, 4] = 1.0
        self.azimuth = torch.tensor((359.0,))
        self.depression = torch.tensor((30.0,))
        self.geometry = target_geometry(
            self.metadata, self.azimuth, self.depression)

    def test_geometry_is_v1_compatible_and_hides_bbox(self) -> None:
        metadata = self.metadata.clone()
        metadata[:, -2:] = 99.0
        geometry = target_geometry(
            metadata, torch.tensor((0.0,)),
            torch.tensor((15.0,)))
        self.assertEqual(tuple(geometry.shape), (1, 12))
        self.assertTrue(torch.equal(
            geometry[:, 8:10], torch.zeros(1, 2)))
        self.assertAlmostEqual(float(geometry[0, 10]), 0.0, places=6)
        self.assertAlmostEqual(float(geometry[0, 11]), 1.0, places=6)

    def test_attention_masks_missing_views_and_wraps(self) -> None:
        module = CircularScaleAttention(4)
        features = torch.randn(1, 12, 4, 8, 8)
        angles = torch.arange(0, 360, 30).float()[None]
        mask = torch.ones(1, 12)
        mask[:, 0] = 0.0
        _, weights_359 = module(
            features, angles, mask,
            torch.tensor((359.0,)), self.depression)
        _, weights_zero = module(
            features, angles, mask,
            torch.tensor((0.0,)), self.depression)
        self.assertEqual(float(weights_359[0, 0]), 0.0)
        self.assertEqual(float(weights_zero[0, 0]), 0.0)
        self.assertLess(
            float((weights_359 - weights_zero).abs().mean()), .01)

    def test_observation_bounds_and_random_path(self) -> None:
        generator = OneStageContinuousSARGenerator()
        base = torch.zeros(4, 1, 64, 64)
        depression = torch.tensor((15., 30., 45., 60.))
        first = torch.randn(4, 3, 64, 64)
        second = torch.randn(4, 3, 64, 64)
        first_sar, _, sigma, receiver = generator.observe(
            base, depression, first)
        second_sar, _, _, _ = generator.observe(
            base, depression, second)
        self.assertGreaterEqual(float(sigma.min()), .06)
        self.assertLessEqual(float(sigma.max()), .14)
        self.assertGreaterEqual(float(receiver.min()), .002)
        self.assertLessEqual(float(receiver.max()), .010)
        self.assertGreater(
            float(F.l1_loss(first_sar, second_sar)), .01)
        self.assertGreaterEqual(float(first_sar.min()), -1.0)
        self.assertLessEqual(float(first_sar.max()), 1.0)

    def test_generate_seed_is_reproducible_and_structure_stable(self) -> None:
        model = ContinuousSpatialOneStageV3().eval()
        views = torch.randn(1, 12, 3, 128, 128)
        angles = torch.arange(0, 360, 30).float()[None]
        mask = torch.ones(1, 12)
        with torch.inference_mode():
            first = model.generate(
                views, angles, mask, self.azimuth,
                self.depression, self.geometry, seed=11)
            repeat = model.generate(
                views, angles, mask, self.azimuth,
                self.depression, self.geometry, seed=11)
            second = model.generate(
                views, angles, mask, self.azimuth,
                self.depression, self.geometry, seed=12)
        self.assertTrue(torch.equal(first, repeat))
        self.assertGreater(float(F.l1_loss(first, second)), .005)
        self.assertLess(
            float(F.l1_loss(
                F.avg_pool2d(first, 4),
                F.avg_pool2d(second, 4))), .02)

    def test_checkpoint_round_trip(self) -> None:
        model = ContinuousSpatialOneStageV3()
        buffer = io.BytesIO()
        torch.save({
            "encoder": model.encoder.state_dict(),
            "generator": model.generator.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(
            buffer, map_location="cpu", weights_only=True)
        restored = ContinuousSpatialOneStageV3()
        restored.encoder.load_state_dict(saved["encoder"])
        restored.generator.load_state_dict(saved["generator"])
        for left, right in zip(
                model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()

"""Measure original V1 loss values and actual gradient pressure without training.

Raw loss magnitudes are not comparable across objectives.  This probe reports
each loss's weighted scalar contribution plus the L2 norm of its gradient on
the RGB encoder and SAR generator.  It is intended to rank ablation candidates
before spending time on a full training run.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bbox_data import image_tensor
from joint_data import JointROIDataset
from joint_models import (
    ContinuousROIDiscriminator,
    RGBIdentityEncoder,
    SpatialROIGenerator,
    _align_translation,
    multiscale_structure_loss,
    sar_perceptual_pyramid_loss,
    sar_physics_prior_loss,
    sar_statistics_loss,
)
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES


DEPRESSION_TO_ID = {15: 0, 30: 1, 45: 2, 60: 3}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the weighted loss gradients of a V1 checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--source-view-mode", choices=("nearest", "random", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class PrototypeROIDataset(Dataset):
    """SAR-only view used to construct original V1 class/depression centres."""

    def __init__(self, dataset: JointROIDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tif, _, class_name, bbox, meta, _ = self.dataset.records[index]
        with torch.no_grad():
            from PIL import Image
            with Image.open(tif) as image:
                source = image if self.dataset.pre_cropped else image.crop(bbox)
                roi = image_tensor(source, 64, False)
        return (
            roi,
            torch.tensor(self.dataset.class_to_id[class_name], dtype=torch.long),
            torch.tensor(DEPRESSION_TO_ID[int(meta["depression"])], dtype=torch.long),
        )


def target_condition(meta: torch.Tensor, source_angle: torch.Tensor) -> torch.Tensor:
    meta = meta.clone()
    meta[:, -2:] = 0.0
    radians = source_angle.float() * (math.pi / 180.0)
    return torch.cat((meta, radians.sin()[:, None], radians.cos()[:, None]), dim=1)


def rotate_target_azimuth(condition: torch.Tensor, degrees: float = 5.0) -> torch.Tensor:
    radians = torch.atan2(condition[:, 0], condition[:, 1]) + math.radians(degrees)
    result = condition.clone()
    result[:, 0], result[:, 1] = radians.sin(), radians.cos()
    return result


def conditional_prototypes(judge: SARClassifier64, dataset: JointROIDataset,
                           checkpoint: Path, device: torch.device, workers: int) -> torch.Tensor:
    """Load the original cache when valid, otherwise construct V1 centres."""
    cache = checkpoint.parent / "native_conditional_prototypes.pt"
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        prototypes = saved.get("prototypes")
        if isinstance(prototypes, torch.Tensor) and prototypes.shape == (40, 4, judge.feature_dim):
            return prototypes.to(device)

    sums = torch.zeros(40, 4, judge.feature_dim, device=device)
    counts = torch.zeros(40, 4, device=device)
    loader = DataLoader(PrototypeROIDataset(dataset), batch_size=128, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    with torch.inference_mode():
        for roi, labels, depressions in tqdm(loader, desc="rebuilding V1 prototypes"):
            roi, labels, depressions = roi.to(device), labels.to(device), depressions.to(device)
            _, features = judge((roi + 1.0) * .5, return_features=True)
            features = F.normalize(features, dim=1)
            sums.index_put_((labels, depressions), features, accumulate=True)
            counts.index_put_((labels, depressions), torch.ones_like(labels, dtype=torch.float), accumulate=True)
    if (counts == 0).any():
        raise RuntimeError("missing V1 class/depression prototype")
    return F.normalize(sums / counts[..., None], dim=2)


def gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared = sum((gradient.detach().float().square().sum() for gradient in gradients if gradient is not None),
                  loss.new_zeros((), dtype=torch.float))
    return float(squared.sqrt())


def main() -> None:
    args = arguments()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") != "continuous_spatial_v1":
        raise RuntimeError("--checkpoint must be an original continuous_spatial_v1 checkpoint")
    weights = dict(state.get("args", {}))
    # The archived V1 checkpoint predates persistence of this zero-default flag.
    weights.setdefault("perceptual_weight", 0.0)
    required_weights = (
        "rgb_id_weight", "cross_view_weight", "sar_class_weight", "cluster_weight",
        "structure_weight", "statistics_weight", "physics_weight", "perceptual_weight",
        "angle_smooth_weight", "adversarial_weight", "feature_match_weight",
    )
    missing = [key for key in required_weights if key not in weights]
    if missing:
        raise RuntimeError(f"checkpoint has no V1 loss weights: {missing}")

    dataset = JointROIDataset(args.rgb_root, args.sar_root, epoch_size=0, band="X", polarization="HH",
                              depression="all", augment_rgb=True, source_view_mode=args.source_view_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    judge_state = torch.load(args.native_classifier_checkpoint, map_location=device, weights_only=False)
    judge = SARClassifier64(40).to(device)
    judge.load_state_dict(judge_state["model"]); judge.eval()
    for parameter in judge.parameters():
        parameter.requires_grad_(False)
    prototypes = conditional_prototypes(judge, dataset, args.checkpoint, device, args.workers)

    encoder = RGBIdentityEncoder(40).to(device)
    generator = SpatialROIGenerator(meta_dim=12).to(device)
    discriminator = ContinuousROIDiscriminator(meta_dim=12).to(device)
    encoder.load_state_dict(state["identity_encoder"])
    generator.load_state_dict(state["generator"])
    discriminator.load_state_dict(state["discriminator"])
    encoder.train(); generator.train(); discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=.02)
    sums: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    names = (
        ("rgb_identity", "rgb_id_weight"),
        ("cross_view", "cross_view_weight"),
        ("sar_class", "sar_class_weight"),
        ("cluster", "cluster_weight"),
        ("structure", "structure_weight"),
        ("statistics", "statistics_weight"),
        ("physics", "physics_weight"),
        ("perceptual", "perceptual_weight"),
        ("angle", "angle_smooth_weight"),
        ("adversarial", "adversarial_weight"),
        ("feature_match", "feature_match_weight"),
    )

    for batch_index, batch in enumerate(tqdm(loader, desc="V1 loss-gradient probe")):
        rgb, rgb_alt = batch["rgb"].to(device), batch["rgb_alt"].to(device)
        real = batch["roi"].to(device)
        labels = batch["class_id"].to(device)
        meta = batch["meta"].to(device)
        source_angle = batch["rgb_angle"].to(device)
        depression_id = torch.tensor(
            [DEPRESSION_TO_ID[int(value)] for value in batch["depression"].tolist()], device=device)
        condition = target_condition(meta, source_angle)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            identity, rgb_logits, pyramid = encoder(rgb, return_pyramid=True)
            alternate_identity, alternate_logits = encoder(rgb_alt)
            clean = generator(identity, condition, pyramid, apply_speckle=False)
            fake = generator.apply_speckle(clean, float(state.get("speckle_strength", generator.speckle_strength)))
            fake_score, fake_disc_features = discriminator(fake, condition)
            with torch.no_grad():
                _, real_disc_features = discriminator(real, condition)
            sar_logits, sar_features, fake_sar_pyramid = judge((fake + 1.0) * .5, return_pyramid=True)
            rgb_loss = .5 * (cross_entropy(rgb_logits, labels) + cross_entropy(alternate_logits, labels))
            cross_loss = 1.0 - (F.normalize(identity, dim=1) * F.normalize(alternate_identity, dim=1)).sum(1).mean()
            class_loss = cross_entropy(sar_logits, labels)
            cosine = (F.normalize(sar_features, dim=1) * prototypes[labels, depression_id]).sum(1).mean()
            cluster_loss = 1.0 - cosine
            structure_loss = multiscale_structure_loss(clean, real)
            statistics_loss = sar_statistics_loss(fake, real)
            physics_loss = sar_physics_prior_loss(clean, real)
            aligned_real = _align_translation(clean, real)
            with torch.no_grad():
                _, _, real_sar_pyramid = judge((aligned_real + 1.0) * .5, return_pyramid=True)
            perceptual_loss = sar_perceptual_pyramid_loss(fake_sar_pyramid, real_sar_pyramid)
            neighbour = generator(identity, rotate_target_azimuth(condition), pyramid, apply_speckle=False)
            angle_loss = F.l1_loss(F.avg_pool2d(clean, 4), F.avg_pool2d(neighbour, 4))
            adversarial_loss = -fake_score.mean()
            feature_loss = (
                F.l1_loss(fake_disc_features.mean((2, 3)), real_disc_features.mean((2, 3)))
                + F.l1_loss(fake_disc_features.std((2, 3)), real_disc_features.std((2, 3))))
        losses = {
            "rgb_identity": rgb_loss, "cross_view": cross_loss, "sar_class": class_loss,
            "cluster": cluster_loss, "structure": structure_loss, "statistics": statistics_loss,
            "physics": physics_loss, "perceptual": perceptual_loss, "angle": angle_loss,
            "adversarial": adversarial_loss, "feature_match": feature_loss,
        }
        for name, weight_name in names:
            weighted = losses[name] * float(weights[weight_name])
            sums[name]["raw"] += float(losses[name].detach())
            sums[name]["weight"] = float(weights[weight_name])
            sums[name]["weighted_contribution"] += float(weighted.detach())
            sums[name]["encoder_gradient_l2"] += gradient_norm(weighted, list(encoder.parameters()))
            sums[name]["generator_gradient_l2"] += gradient_norm(weighted, list(generator.parameters()))
        sums["diagnostics"]["native_class_accuracy"] += float((sar_logits.argmax(1) == labels).float().mean())
        sums["diagnostics"]["cluster_cosine"] += float(cosine.detach())
        if batch_index + 1 >= args.batches:
            break

    count = min(args.batches, len(loader))
    losses_report = {
        name: {
            key: value if key == "weight" else value / max(count, 1)
            for key, value in values.items()
        }
        for name, values in sums.items() if name != "diagnostics"
    }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "batches": count,
        "losses": losses_report,
        "diagnostics": {key: value / max(count, 1) for key, value in sums["diagnostics"].items()},
        "interpretation": {
            "weighted_contribution": "weight times mean raw loss; not sufficient by itself",
            "encoder_gradient_l2": "actual weighted gradient norm delivered to RGB encoder",
            "generator_gradient_l2": "actual weighted gradient norm delivered to SAR generator",
            "selection": "low-value candidates have both tiny weighted contribution and tiny generator gradient; test one change at a time",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

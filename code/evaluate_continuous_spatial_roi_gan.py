"""Evaluate the four-depression continuous-azimuth spatial GAN on real SAR ROIs."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from joint_data import JointROIDataset
from joint_models import RGBIdentityEncoder, SpatialROIGenerator, distributional_structure_loss, sar_physics_prior_loss
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES


def condition(meta: torch.Tensor, rgb_angle: torch.Tensor) -> torch.Tensor:
    meta = meta.clone()
    meta[:, -2:] = 0.0  # do not leak real-SAR annotation-box extent to G
    radians = rgb_angle.float() * (math.pi / 180.0)
    return torch.cat((meta, radians.sin()[:, None], radians.cos()[:, None]), dim=1)


def rotate_target_azimuth(target: torch.Tensor, degrees: float) -> torch.Tensor:
    angle = torch.atan2(target[:, 0], target[:, 1]) + math.radians(degrees)
    result = target.clone()
    result[:, 0], result[:, 1] = angle.sin(), angle.cos()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit continuous-spatial RGB-to-SAR GAN")
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source-view-mode", choices=("nearest", "random"), default="nearest")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    state = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") not in {"continuous_spatial_v1", "continuous_spatial_fused_v2"}:
        raise RuntimeError("checkpoint is not a continuous spatial V1 or fused-v2 model")
    if state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("GAN class order differs from SOC40 data")
    encoder = RGBIdentityEncoder(len(SOC40_CLASSES)).to(device)
    generator = SpatialROIGenerator(meta_dim=12).to(device)
    encoder.load_state_dict(state["identity_encoder"]); generator.load_state_dict(state["generator"])
    encoder.eval(); generator.eval()

    judge_state = torch.load(args.classifier_checkpoint, map_location=device, weights_only=False)
    if judge_state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("classifier class order differs from SOC40 data")
    judge = SARClassifier64(len(SOC40_CLASSES)).to(device)
    judge.load_state_dict(judge_state["model"]); judge.eval()

    dataset = JointROIDataset(args.rgb_root, args.sar_root, epoch_size=0, augment_rgb=False,
                              band="X", polarization="HH", depression="all",
                              source_view_mode=args.source_view_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    totals = defaultdict(float)
    per_depression: dict[int, defaultdict[str, float]] = {d: defaultdict(float) for d in (15, 30, 45, 60)}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="continuous spatial GAN audit"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            real = batch["roi"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            labels = batch["class_id"].to(device, non_blocking=True)
            target = condition(meta, batch["rgb_angle"].to(device, non_blocking=True))
            identity, rgb_logits, pyramid = encoder(rgb, return_pyramid=True)
            clean = generator(identity, target, pyramid, apply_speckle=False)
            fake = generator.apply_speckle(clean)
            fake_logits, fake_features, fake_pyramid = judge((fake + 1) * .5, return_pyramid=True)
            real_logits, real_features, real_pyramid = judge((real + 1) * .5, return_pyramid=True)
            left = generator(identity, rotate_target_azimuth(target, -5.), pyramid, apply_speckle=False)
            right = generator(identity, rotate_target_azimuth(target, 5.), pyramid, apply_speckle=False)
            far = generator(identity, rotate_target_azimuth(target, 30.), pyramid, apply_speckle=False)
            structure, _, _, _ = distributional_structure_loss(clean, fake, real, fake_pyramid, real_pyramid)
            near_delta = .5 * (F.l1_loss(F.avg_pool2d(left, 4), F.avg_pool2d(clean, 4)) +
                                F.l1_loss(F.avg_pool2d(right, 4), F.avg_pool2d(clean, 4)))
            far_delta = F.l1_loss(F.avg_pool2d(far, 4), F.avg_pool2d(clean, 4))
            size = len(labels)
            measures = {
                "samples": size,
                "rgb_identity_top1": (rgb_logits.argmax(1) == labels).sum().item(),
                "generated_native_sar_top1": (fake_logits.argmax(1) == labels).sum().item(),
                "real_native_sar_top1": (real_logits.argmax(1) == labels).sum().item(),
                "feature_cosine_to_real": F.cosine_similarity(fake_features, real_features).sum().item(),
                "structure_loss": structure.item() * size,
                "physics_prior_loss": sar_physics_prior_loss(fake, real).item() * size,
                "angle_near_delta": near_delta.item() * size,
                "angle_far_delta": far_delta.item() * size,
            }
            for key, value in measures.items():
                totals[key] += value
            for depression in (15, 30, 45, 60):
                mask = batch["depression"] == depression
                count = int(mask.sum().item())
                if not count:
                    continue
                result = per_depression[depression]
                result["samples"] += count
                result["rgb_identity_top1"] += (rgb_logits[mask].argmax(1) == labels[mask]).sum().item()
                result["generated_native_sar_top1"] += (fake_logits[mask].argmax(1) == labels[mask]).sum().item()
                result["real_native_sar_top1"] += (real_logits[mask].argmax(1) == labels[mask]).sum().item()
                result["feature_cosine_to_real"] += F.cosine_similarity(fake_features[mask], real_features[mask]).sum().item()

    def normalise(values: defaultdict[str, float], include_losses: bool) -> dict[str, float | int]:
        samples = int(values["samples"])
        keys = ("rgb_identity_top1", "generated_native_sar_top1", "real_native_sar_top1", "feature_cosine_to_real")
        answer: dict[str, float | int] = {"samples": samples}
        answer.update({key: values[key] / max(samples, 1) for key in keys})
        if include_losses:
            answer["structure_loss"] = values["structure_loss"] / max(samples, 1)
            answer["physics_prior_loss"] = values["physics_prior_loss"] / max(samples, 1)
            answer["angle_near_delta"] = values["angle_near_delta"] / max(samples, 1)
            answer["angle_far_delta"] = values["angle_far_delta"] / max(samples, 1)
            answer["angle_far_to_near"] = answer["angle_far_delta"] / max(answer["angle_near_delta"], 1e-8)
        return answer

    result = {
        "condition": "X/HH; target azimuth is sin/cos conditioned; depressions=15,30,45,60",
        "source_view_mode": args.source_view_mode,
        "all": normalise(totals, True),
        "by_depression": {str(key): normalise(value, False) for key, value in per_depression.items()},
        "gan_checkpoint": str(args.gan_checkpoint), "classifier_checkpoint": str(args.classifier_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Independently audit RGB-to-SAR outputs with the native image-only classifier."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from joint_data import JointROIDataset
from joint_models import RGBIdentityEncoder, ROIGenerator
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated SAR with an independent native classifier")
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--band", default="all", choices=("all", "X", "KU"))
    parser.add_argument("--polarization", default="all", choices=("all", "HH", "HV", "VH", "VV"))
    parser.add_argument("--depression", default="all", choices=("all", "15", "30", "45", "60"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)

    gan = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
    base = 16 if gan.get("args", {}).get("tiny", False) else 32
    encoder, generator = RGBIdentityEncoder(len(SOC40_CLASSES), base=base).to(device), ROIGenerator(base=base).to(device)
    encoder.load_state_dict(gan["identity_encoder"]); generator.load_state_dict(gan["generator"])
    encoder.eval(); generator.eval()
    classifier_state = torch.load(args.classifier_checkpoint, map_location=device, weights_only=False)
    classifier = SARClassifier64(len(SOC40_CLASSES)).to(device)
    classifier.load_state_dict(classifier_state["model"]); classifier.eval()
    if classifier_state.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("native classifier class order differs from GAN class order")

    dataset = JointROIDataset(args.rgb_root, args.sar_root, epoch_size=0, augment_rgb=False,
                              band=args.band, polarization=args.polarization, depression=args.depression)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    generated_correct = real_correct = rgb_correct = total = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="native classifier GAN audit"):
            rgb, real = batch["rgb"].to(device, non_blocking=True), batch["roi"].to(device, non_blocking=True)
            meta, labels = batch["meta"].to(device, non_blocking=True), batch["class_id"].to(device, non_blocking=True)
            identity, rgb_logits = encoder(rgb)
            fake = generator(identity, meta)
            generated_correct += (classifier((fake + 1) * .5).argmax(1) == labels).sum().item()
            real_correct += (classifier((real + 1) * .5).argmax(1) == labels).sum().item()
            rgb_correct += (rgb_logits.argmax(1) == labels).sum().item()
            total += len(labels)
    result = {"samples": total, "rgb_identity_top1": rgb_correct / total,
              "generated_native_sar_top1": generated_correct / total,
              "real_native_sar_top1": real_correct / total,
              "gan_checkpoint": str(args.gan_checkpoint), "classifier_checkpoint": str(args.classifier_checkpoint),
              "condition": {"band": args.band, "polarization": args.polarization, "depression": args.depression}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

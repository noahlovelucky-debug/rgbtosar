"""Evaluate identity, SARATR-X class/cluster, structure and GAN realism on test ROIs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from joint_data import JointROIDataset
from joint_models import RGBIdentityEncoder, ROIDiscriminator, ROIGenerator, multiscale_structure_loss
from saratrx import SOC40_CLASSES, load_saratrx, saratrx_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a joint identity ROI GAN checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-cache", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--saratrx-checkpoint", type=Path, required=True)
    parser.add_argument("--saratrx-input-size", type=int, default=64, choices=(64,))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=5000, help="0 evaluates the complete test set")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pre-cropped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--band", default="all", choices=("all", "X", "KU"))
    parser.add_argument("--polarization", default="all", choices=("all", "HH", "HV", "VH", "VV"))
    parser.add_argument("--depression", default="all", choices=("all", "15", "30", "45", "60"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if saved.get("classes") != list(SOC40_CLASSES):
        raise RuntimeError("checkpoint class order does not match SOC_40classes.pth")
    config = saved.get("args", {})
    base = 16 if config.get("tiny", False) else 32
    encoder = RGBIdentityEncoder(len(SOC40_CLASSES), base=base).to(device)
    generator = ROIGenerator(base=base).to(device)
    discriminator = ROIDiscriminator(base=base).to(device)
    encoder.load_state_dict(saved["identity_encoder"])
    generator.load_state_dict(saved["generator"])
    discriminator.load_state_dict(saved["discriminator"])
    encoder.eval(); generator.eval(); discriminator.eval()
    saratrx = load_saratrx(args.saratrx_checkpoint, device=device, freeze=True,
                           input_size=args.saratrx_input_size)
    prototype_data = torch.load(args.prototype_cache, map_location="cpu", weights_only=True)
    prototypes = prototype_data["prototypes"].to(device)
    dataset = JointROIDataset(args.rgb_root, args.sar_root, epoch_size=args.samples,
                              pre_cropped=args.pre_cropped, band=args.band,
                              polarization=args.polarization, depression=args.depression,
                              augment_rgb=False)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    totals = torch.zeros(9, dtype=torch.float64)
    per_class = {name: {"count": 0, "rgb_correct": 0, "fake_correct": 0,
                        "real_correct": 0, "fake_cluster_cosine_sum": 0.0}
                 for name in SOC40_CLASSES}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="joint GAN evaluation"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            real = batch["roi"].to(device, non_blocking=True)
            meta = batch["meta"].to(device, non_blocking=True)
            labels = batch["class_id"].to(device, non_blocking=True)
            identity, rgb_logits = encoder(rgb)
            fake = generator(identity, meta)
            fake_logits, fake_features = saratrx(saratrx_input(fake, args.saratrx_input_size),
                                                  return_features=True)
            real_logits, real_features = saratrx(saratrx_input(real, args.saratrx_input_size),
                                                  return_features=True)
            fake_score, _ = discriminator(fake)
            real_score, _ = discriminator(real)
            fake_cosine = (F.normalize(fake_features, dim=1) * prototypes[labels]).sum(1)
            real_cosine = (F.normalize(real_features, dim=1) * prototypes[labels]).sum(1)
            structure = multiscale_structure_loss(fake, real)
            rgb_correct = rgb_logits.argmax(1) == labels
            fake_correct = fake_logits.argmax(1) == labels
            real_correct = real_logits.argmax(1) == labels
            count = labels.numel()
            values = (
                rgb_correct.sum(), fake_correct.sum(), real_correct.sum(), fake_cosine.sum(),
                real_cosine.sum(), fake_score.sum(), real_score.sum(), structure * count,
            )
            totals[:8] += torch.tensor([value.item() for value in values], dtype=torch.float64)
            totals[8] += count
            for index, label in enumerate(labels.tolist()):
                item = per_class[SOC40_CLASSES[label]]
                item["count"] += 1
                item["rgb_correct"] += int(rgb_correct[index])
                item["fake_correct"] += int(fake_correct[index])
                item["real_correct"] += int(real_correct[index])
                item["fake_cluster_cosine_sum"] += float(fake_cosine[index])
    count = totals[8].item()
    metrics = {
        "samples": int(count),
        "rgb_identity_top1": totals[0].item() / count,
        "generated_saratrx_top1": totals[1].item() / count,
        "real_saratrx_top1": totals[2].item() / count,
        "generated_cluster_cosine": totals[3].item() / count,
        "real_cluster_cosine": totals[4].item() / count,
        "generated_discriminator_score": totals[5].item() / count,
        "real_discriminator_score": totals[6].item() / count,
        "structure_loss": totals[7].item() / count,
        "per_class": {},
    }
    for name, item in per_class.items():
        n = item.pop("count")
        metrics["per_class"][name] = ({"samples": n,
            "rgb_identity_top1": item["rgb_correct"] / n,
            "generated_saratrx_top1": item["fake_correct"] / n,
            "real_saratrx_top1": item["real_correct"] / n,
            "generated_cluster_cosine": item["fake_cluster_cosine_sum"] / n} if n else {"samples": 0})
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2))
    output = args.output or args.checkpoint.parent / "test_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

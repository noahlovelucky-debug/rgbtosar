"""Train an RGB-conditioned SAR decoder with a spatial scattering-code prior."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bbox_data import image_tensor, read_annotation
from joint_data import JointROIDataset
from joint_models import (CodebookSpatialROIGenerator, ContinuousROIDiscriminator,
                          RGBIdentityEncoder, SARSpatialCodeEncoder, _align_translation,
                          initialise, multiscale_structure_loss, sar_perceptual_pyramid_loss,
                          sar_physics_prior_loss, sar_statistics_loss)
from sar_classifier_64 import SARClassifier64
from saratrx import SOC40_CLASSES
from train_continuous_spatial_roi_gan import save_preview, set_grad, target_condition


DEP_TO_ID = {15: 0, 30: 1, 45: 2, 60: 3}


class CodebookDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.records: list[tuple[Path, int, int, int]] = []
        for class_id, class_name in enumerate(SOC40_CLASSES):
            for path in sorted((Path(root) / class_name).glob("X_HH_*.tif")):
                _, meta = read_annotation(path.with_suffix(".xml"))
                self.records.append((path, class_id, DEP_TO_ID[int(meta["depression"])],
                                     ((int(meta["azimuth"]) + 15) % 360) // 30))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, class_id, depression, azimuth_bin = self.records[index]
        with Image.open(path) as image:
            roi = image_tensor(image, 64, False)
        return roi, class_id, depression, azimuth_bin


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial scattering-code RGB-to-SAR generator")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--native-classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--base-gan-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epoch-size", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--code-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=8e-5)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=9973)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-codebook", action="store_true",
                        help="skip the final latent-code export (intended for smoke tests)")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); use_amp = device.type == "cuda" and not args.no_amp

    data = JointROIDataset(args.rgb_root, args.sar_root, epoch_size=args.epoch_size,
                           band="X", polarization="HH", depression="all",
                           augment_rgb=True, source_view_mode="mixed")
    loader = DataLoader(data, args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    base = torch.load(args.base_gan_checkpoint, map_location=device, weights_only=False)
    encoder = RGBIdentityEncoder(len(SOC40_CLASSES)).to(device)
    encoder.load_state_dict(base["identity_encoder"]); encoder.eval()
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    generator = CodebookSpatialROIGenerator(meta_dim=12, code_channels=args.code_channels).to(device)
    generator.apply(initialise)
    missing, unexpected = generator.load_state_dict(base["generator"], strict=False)
    if unexpected or any(not key.startswith("code_projection.") for key in missing):
        raise RuntimeError(f"incompatible base generator: missing={missing}, unexpected={unexpected}")
    generator.reset_code_injection()
    code_encoder = SARSpatialCodeEncoder(args.code_channels).to(device); code_encoder.apply(initialise)
    discriminator = ContinuousROIDiscriminator(meta_dim=12).to(device)
    discriminator.load_state_dict(base["discriminator"])
    judge_state = torch.load(args.native_classifier_checkpoint, map_location=device, weights_only=False)
    judge = SARClassifier64(len(SOC40_CLASSES)).to(device)
    judge.load_state_dict(judge_state["model"]); judge.eval()
    for parameter in judge.parameters(): parameter.requires_grad_(False)

    gen_optimizer = torch.optim.Adam((*generator.parameters(), *code_encoder.parameters()),
                                     lr=args.lr, betas=(.5, .999), foreach=False)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr,
                                      betas=(.5, .999), foreach=False)
    gen_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    disc_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=.02)
    header = ("epoch", "total", "class", "structure", "statistics", "physics", "perceptual",
              "adversarial", "feature", "code_reconstruction", "discriminator", "fake_accuracy")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(header)
    best_quality = float("inf")
    for epoch in range(1, args.epochs + 1):
        generator.train(); code_encoder.train(); discriminator.train()
        totals = torch.zeros(len(header), dtype=torch.float64)
        for batch in tqdm(loader, desc=f"codebook spatial GAN {epoch}/{args.epochs}"):
            rgb, real = batch["rgb"].to(device), batch["roi"].to(device)
            meta, labels = batch["meta"].to(device), batch["class_id"].to(device)
            condition = target_condition(meta, batch["rgb_angle"].to(device))
            with torch.no_grad():
                identity, _, pyramid = encoder(rgb, return_pyramid=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                code = code_encoder(real)
                fake_clean = generator(identity, condition, pyramid, code, apply_speckle=False)
                fake = generator.apply_speckle(fake_clean)

            disc_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                real_score, _ = discriminator(real, condition)
                fake_score, _ = discriminator(fake.detach(), condition)
                disc_loss = F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
            disc_scaler.scale(disc_loss).backward(); disc_scaler.step(disc_optimizer); disc_scaler.update()

            set_grad(discriminator, False); gen_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fake_score, fake_disc = discriminator(fake, condition)
                with torch.no_grad():
                    _, real_disc = discriminator(real, condition)
                logits, _, fake_features = judge((fake + 1) * .5, return_pyramid=True)
                class_loss = criterion(logits, labels)
                structure = multiscale_structure_loss(fake_clean, real)
                statistics = sar_statistics_loss(fake, real)
                physics = sar_physics_prior_loss(fake_clean, real)
                aligned = _align_translation(fake_clean, real)
                with torch.no_grad():
                    _, _, real_features = judge((aligned + 1) * .5, return_pyramid=True)
                perceptual = sar_perceptual_pyramid_loss(fake_features, real_features)
                adversarial = -fake_score.mean()
                feature = (F.l1_loss(fake_disc.mean((2, 3)), real_disc.mean((2, 3)))
                           + F.l1_loss(fake_disc.std((2, 3)), real_disc.std((2, 3))))
                recovered_code = code_encoder(fake_clean)
                code_reconstruction = F.l1_loss(recovered_code, code.detach())
                total = (4 * class_loss + 24 * structure + 6 * statistics + 6 * physics
                         + 10 * perceptual + 4 * adversarial + 8 * feature
                         + 3 * code_reconstruction)
            gen_scaler.scale(total).backward(); gen_scaler.unscale_(gen_optimizer)
            torch.nn.utils.clip_grad_norm_((*generator.parameters(), *code_encoder.parameters()), 5.)
            gen_scaler.step(gen_optimizer); gen_scaler.update(); set_grad(discriminator, True)
            values = (total, class_loss, structure, statistics, physics, perceptual, adversarial,
                      feature, code_reconstruction, disc_loss,
                      (logits.argmax(1) == labels).float().mean())
            totals[1:] += torch.tensor([value.detach().item() for value in values], dtype=torch.float64)
            totals[0] += 1
        averages = (totals[1:] / totals[0]).tolist(); row = (epoch, *averages)
        with (args.output / "history.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        quality = averages[2] + .5 * averages[3] + .25 * averages[4] + .5 * averages[5] + .1 * averages[7]
        checkpoint = {"epoch": epoch, "identity_encoder": encoder.state_dict(),
                      "generator": generator.state_dict(), "code_encoder": code_encoder.state_dict(),
                      "discriminator": discriminator.state_dict(), "classes": list(SOC40_CLASSES),
                      "architecture": "continuous_spatial_codebook_v3",
                      "code_channels": args.code_channels, "quality": quality}
        torch.save(checkpoint, args.output / "latest.pt")
        if averages[10] >= .85 and quality < best_quality:
            best_quality = quality; torch.save(checkpoint, args.output / "best.pt")
        if epoch == 1 or epoch % 5 == 0:
            save_preview(rgb, real, fake, args.output / f"preview_{epoch:04d}.png")
        print(dict(zip(header, row)), flush=True)

    if args.skip_codebook:
        return
    best = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    code_encoder.load_state_dict(best["code_encoder"]); code_encoder.eval()
    code_data = CodebookDataset(args.sar_root)
    code_loader = DataLoader(code_data, 256, shuffle=False, num_workers=args.workers,
                             pin_memory=device.type == "cuda")
    codes, classes, depressions, azimuths = [], [], [], []
    with torch.inference_mode():
        for roi, labels, deps, azimuth_bins in tqdm(code_loader, desc="building spatial SAR codebook"):
            codes.append(code_encoder(roi.to(device, non_blocking=True)).half().cpu())
            classes.append(labels); depressions.append(deps); azimuths.append(azimuth_bins)
    best["latent_codes"] = torch.cat(codes)
    best["latent_class"] = torch.cat(classes)
    best["latent_depression"] = torch.cat(depressions)
    best["latent_azimuth_bin"] = torch.cat(azimuths)
    best["latent_prior_kind"] = "empirical_class_depression_azimuth_codebook"
    torch.save(best, args.output / "best_with_codebook.pt")
    print({"codebook": len(best["latent_codes"]), "output": str(args.output / "best_with_codebook.pt")})


if __name__ == "__main__":
    main()

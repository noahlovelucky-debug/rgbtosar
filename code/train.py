from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
from rgb2sar.data import DirectionDataset
from rgb2sar.models import Discriminator, Generator, init_weights

def save_gray(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(array, "L").save(path)

def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one direction-specific unpaired RGB<->SAR CycleGAN")
    p.add_argument("--rgb-root", type=Path, required=True); p.add_argument("--sar-root", type=Path, required=True)
    p.add_argument("--rgb-index", type=int, required=True, choices=range(1, 13))
    p.add_argument("--angle-offset", type=int, default=0, help="Azimuth represented by RGB 1.png")
    p.add_argument("--angle-tolerance", type=int, default=15)
    p.add_argument("--band", default="all", choices=["all", "X", "KU"])
    p.add_argument("--polarization", default="all", choices=["all", "HH", "HV", "VH", "VV"])
    p.add_argument("--depression", default="all", choices=["all", "15", "30", "45", "60"])
    p.add_argument("--output", type=Path, default=Path("runs/direction_demo"))
    p.add_argument("--image-size", type=int, default=128); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--epoch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--cycle-weight", type=float, default=10.0)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tiny", action="store_true", help="Small network for CPU smoke tests")
    return p.parse_args()

def main() -> None:
    args = arguments(); random.seed(args.seed); torch.manual_seed(args.seed); args.output.mkdir(parents=True, exist_ok=True)
    dataset = DirectionDataset(args.rgb_root, args.sar_root, args.rgb_index, args.image_size, args.angle_offset,
        args.angle_tolerance, args.band, args.polarization, args.depression, args.epoch_size)
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.workers)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
        torch.backends.mkldnn.enabled = False
    base, blocks = (16, 1) if args.tiny else (64, 6)
    g_ab, g_ba = Generator(3, 1, base, blocks).to(device), Generator(1, 3, base, blocks).to(device)
    d_a, d_b = Discriminator(3, base).to(device), Discriminator(1, base).to(device)
    for model in (g_ab, g_ba, d_a, d_b): model.apply(init_weights)
    # foreach=False avoids a Windows CPU crash seen in some recent PyTorch builds.
    opt_g = torch.optim.Adam(list(g_ab.parameters()) + list(g_ba.parameters()), lr=args.lr, betas=(0.5, .999), foreach=False)
    opt_d = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()), lr=args.lr, betas=(0.5, .999), foreach=False)
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update(dataset.summary())
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("dataset:", dataset.summary(), "device:", device)
    history_path = args.output / "history.csv"
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(["epoch", "loss_g", "loss_d"])
    for epoch in range(1, args.epochs + 1):
        total_g = total_d = 0.0
        bar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in bar:
            rgb, sar = batch["rgb"].to(device), batch["sar"].to(device)
            opt_g.zero_grad(set_to_none=True); fake_sar, fake_rgb = g_ab(rgb), g_ba(sar)
            pred_fs, pred_fr = d_b(fake_sar), d_a(fake_rgb)
            adversarial = mse(pred_fs, torch.ones_like(pred_fs)) + mse(pred_fr, torch.ones_like(pred_fr))
            cycle = l1(g_ba(fake_sar), rgb) + l1(g_ab(fake_rgb), sar)
            loss_g = adversarial + args.cycle_weight * cycle; loss_g.backward(); opt_g.step()
            opt_d.zero_grad(set_to_none=True)
            rs, fs, rr, fr = d_b(sar), d_b(fake_sar.detach()), d_a(rgb), d_a(fake_rgb.detach())
            loss_d = .5 * (mse(rs, torch.ones_like(rs)) + mse(fs, torch.zeros_like(fs)) +
                           mse(rr, torch.ones_like(rr)) + mse(fr, torch.zeros_like(fr)))
            loss_d.backward(); opt_d.step(); bar.set_postfix(g=f"{loss_g.item():.3f}", d=f"{loss_d.item():.3f}")
            total_g += loss_g.item(); total_d += loss_d.item()
        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([epoch, total_g / len(loader), total_d / len(loader)])
        checkpoint = {"epoch": epoch, "generator": g_ab.state_dict(), "args": vars(args), "summary": dataset.summary()}
        torch.save(checkpoint, args.output / "latest.pt")
        if epoch == 1 or epoch % 10 == 0:
            torch.save(checkpoint, args.output / f"epoch_{epoch:04d}.pt")
            save_gray((fake_sar[:1].detach().cpu() + 1) / 2, args.output / f"preview_{epoch:04d}.png")

if __name__ == "__main__": main()

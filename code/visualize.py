from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from rgb2sar.data import DirectionDataset
from rgb2sar.models import Generator

def to_image(tensor: torch.Tensor, rgb: bool) -> Image.Image:
    array = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().numpy()
    if rgb: return Image.fromarray(array.transpose(1, 2, 0), "RGB")
    return Image.fromarray(array[0], "L").convert("RGB")

def main() -> None:
    p = argparse.ArgumentParser(description="Make RGB/fake-SAR/real-SAR comparison and loss plot")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples", type=int, default=8); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(); checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False); cfg = checkpoint["args"]
    base, blocks = (16, 1) if cfg.get("tiny") else (64, 6)
    model = Generator(3, 1, base, blocks).to(args.device); model.load_state_dict(checkpoint["generator"]); model.eval()
    dataset = DirectionDataset(Path(cfg["rgb_root"]), Path(cfg["sar_root"]), cfg["rgb_index"], cfg["image_size"],
        cfg["angle_offset"], cfg["angle_tolerance"], cfg["band"], cfg["polarization"], cfg["depression"], args.samples)
    size, header = cfg["image_size"], 28; canvas = Image.new("RGB", (size * 3, header + size * args.samples), "white")
    draw = ImageDraw.Draw(canvas); draw.text((4, 7), "Input RGB", fill="black"); draw.text((size + 4, 7), "Generated SAR", fill="black"); draw.text((size * 2 + 4, 7), "Real SAR (same class/angle)", fill="black")
    class_positions = np.linspace(0, len(dataset.classes) - 1, min(args.samples, len(dataset.classes)), dtype=int)
    sample_indices = [next(i for i, path in enumerate(dataset.sar_paths) if path.parent.name == dataset.classes[pos]) for pos in class_positions]
    with torch.inference_mode():
        for row, sample_index in enumerate(sample_indices):
            item = dataset[sample_index]; fake = model(item["rgb"].unsqueeze(0).to(args.device))[0]
            canvas.paste(to_image(item["rgb"], True), (0, header + row * size)); canvas.paste(to_image(fake, False), (size, header + row * size)); canvas.paste(to_image(item["sar"], False), (size * 2, header + row * size))
    args.output_dir.mkdir(parents=True, exist_ok=True); canvas.save(args.output_dir / "comparison.png")
    history = args.checkpoint.parent / "history.csv"
    if history.exists():
        rows = list(csv.DictReader(history.open(encoding="utf-8"))); width, height = 800, 420
        plot = Image.new("RGB", (width, height), "white"); d = ImageDraw.Draw(plot); margin = 45
        values = [float(r[k]) for r in rows for k in ("loss_g", "loss_d")]; maximum = max(values, default=1.0)
        d.line((margin, 10, margin, height-margin, width-10, height-margin), fill="black", width=2)
        for key, color in (("loss_g", "red"), ("loss_d", "blue")):
            points=[]
            for i,row in enumerate(rows):
                x=margin+(width-margin-15)*(i/max(1,len(rows)-1)); y=height-margin-(height-margin-20)*float(row[key])/maximum; points.append((x,y))
            if len(points)>1: d.line(points, fill=color, width=3)
        d.text((60, 15), "Generator loss (red), Discriminator loss (blue)", fill="black"); plot.save(args.output_dir / "loss_curve.png")
    print(args.output_dir / "comparison.png")
if __name__ == "__main__": main()

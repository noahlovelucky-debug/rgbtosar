from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from rgb2sar.data import rgba_to_rgb
from rgb2sar.models import Generator

def main() -> None:
    p = argparse.ArgumentParser(description="Generate SAR from one RGB image")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False); config = checkpoint["args"]
    base, blocks = (16, 1) if config.get("tiny") else (64, 6)
    model = Generator(3, 1, base, blocks).to(args.device); model.load_state_dict(checkpoint["generator"]); model.eval()
    with Image.open(args.input) as image:
        image = rgba_to_rgb(image).resize((config["image_size"], config["image_size"]), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()
        tensor = (torch.from_numpy(array) / 127.5 - 1).unsqueeze(0).to(args.device)
    with torch.inference_mode(): generated = (model(tensor).cpu() + 1) / 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array = (generated[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(array, "L").save(args.output)
    print(f"wrote {args.output}; azimuth={checkpoint['summary']['azimuth']} degrees")
if __name__ == "__main__": main()

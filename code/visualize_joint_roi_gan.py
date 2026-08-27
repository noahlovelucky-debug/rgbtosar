"""Create an auditable RGB | real SAR | generated SAR contact sheet."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from joint_data import JointROIDataset
from joint_models import RGBIdentityEncoder, ROIGenerator
from saratrx import SOC40_CLASSES, load_saratrx, saratrx_input


def tensor_rgb(image: torch.Tensor, size: int) -> np.ndarray:
    image = F.interpolate(image[None], (size, size), mode="bilinear", align_corners=False)[0]
    return ((image.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)


def tensor_gray(image: torch.Tensor, size: int) -> np.ndarray:
    image = F.interpolate(image[None], (size, size), mode="bilinear", align_corners=False)[0, 0]
    array = ((image.detach().cpu().clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
    return np.repeat(array[..., None], 3, axis=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise RGB-to-SAR joint GAN results")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--saratrx-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--panel-size", type=int, default=128)
    parser.add_argument("--band", default="all", choices=("all", "X", "KU"))
    parser.add_argument("--polarization", default="all", choices=("all", "HH", "HV", "VH", "VV"))
    parser.add_argument("--depression", default="all", choices=("all", "15", "30", "45", "60"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = saved.get("args", {})
    base = 16 if config.get("tiny", False) else 32
    encoder = RGBIdentityEncoder(len(SOC40_CLASSES), base=base).to(device)
    generator = ROIGenerator(base=base).to(device)
    encoder.load_state_dict(saved["identity_encoder"])
    generator.load_state_dict(saved["generator"])
    encoder.eval(); generator.eval()
    classifier = load_saratrx(args.saratrx_checkpoint, device=device, freeze=True, input_size=64)
    dataset = JointROIDataset(args.rgb_root, args.sar_root,
                              rgb_size=int(config.get("rgb_size", 128)),
                              roi_size=64, epoch_size=args.samples,
                              pre_cropped=True, band=args.band,
                              polarization=args.polarization, depression=args.depression,
                              augment_rgb=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    panel, label_height, header_height = args.panel_size, 34, 28
    rows: list[Image.Image] = []
    generated_dir = args.output.parent / f"{args.output.stem}_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    records = []
    font = ImageFont.load_default()
    with torch.inference_mode():
        item_index = 0
        for batch in loader:
            rgb = batch["rgb"].to(device)
            real = batch["roi"].to(device)
            meta = batch["meta"].to(device)
            labels = batch["class_id"].to(device)
            identity, rgb_logits = encoder(rgb)
            fake = generator(identity, meta)
            fake_logits = classifier(saratrx_input(fake, 64))
            rgb_probability = rgb_logits.softmax(1)
            fake_probability = fake_logits.softmax(1)
            for index in range(len(rgb)):
                rgb_pred = int(rgb_logits[index].argmax())
                fake_pred = int(fake_logits[index].argmax())
                true_id = int(labels[index])
                panels = np.concatenate((tensor_rgb(rgb[index], panel),
                                         tensor_gray(real[index], panel),
                                         tensor_gray(fake[index], panel)), axis=1)
                row = Image.new("RGB", (panel * 3, panel + label_height), "white")
                row.paste(Image.fromarray(panels, "RGB"), (0, 0))
                draw = ImageDraw.Draw(row)
                text = (f"{SOC40_CLASSES[true_id][:24]} az={int(batch['azimuth'][index])} "
                        f"rgb={int(batch['rgb_angle'][index])} | RGB:{SOC40_CLASSES[rgb_pred][:15]} "
                        f"{float(rgb_probability[index, rgb_pred]):.2f} | "
                        f"SAR:{SOC40_CLASSES[fake_pred][:15]} {float(fake_probability[index, fake_pred]):.2f}")
                draw.text((3, panel + 3), text, fill="black", font=font)
                rows.append(row)
                fake_array = tensor_gray(fake[index], 64)[..., 0]
                generated_path = generated_dir / f"{item_index:03d}_{SOC40_CLASSES[true_id]}_az{int(batch['azimuth'][index])}.png"
                Image.fromarray(fake_array, "L").save(generated_path)
                records.append({"index": item_index, "class": SOC40_CLASSES[true_id],
                                "azimuth": int(batch["azimuth"][index]),
                                "rgb_angle": int(batch["rgb_angle"][index]),
                                "rgb_prediction": SOC40_CLASSES[rgb_pred],
                                "rgb_confidence": float(rgb_probability[index, rgb_pred]),
                                "sar_prediction": SOC40_CLASSES[fake_pred],
                                "sar_confidence": float(fake_probability[index, fake_pred]),
                                "generated": str(generated_path)})
                item_index += 1
    header = Image.new("RGB", (panel * 3, header_height), "white")
    header_draw = ImageDraw.Draw(header)
    for column, title in enumerate(("INPUT RGB", "MATCHED REAL SAR", "GENERATED SAR")):
        header_draw.text((column * panel + 4, 7), title, fill="black", font=font)
    sheet = Image.new("RGB", (panel * 3, header_height + sum(row.height for row in rows)), "white")
    sheet.paste(header, (0, 0)); y = header_height
    for row in rows:
        sheet.paste(row, (0, y)); y += row.height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    rgb_accuracy = sum(item["rgb_prediction"] == item["class"] for item in records) / len(records)
    sar_accuracy = sum(item["sar_prediction"] == item["class"] for item in records) / len(records)
    print({"output": str(args.output), "samples": len(records),
           "rgb_top1": rgb_accuracy, "generated_sar_top1": sar_accuracy})


if __name__ == "__main__":
    main()

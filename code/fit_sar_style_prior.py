"""Fit a standalone depression-conditional Gaussian to learned real-SAR styles."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bbox_data import image_tensor, read_annotation
from joint_models import SARStyleEncoder
from saratrx import SOC40_CLASSES


DEPRESSION_TO_ID = {15: 0, 30: 1, 45: 2, 60: 3}


class StyleROIDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.records: list[tuple[Path, int, int]] = []
        for class_id, class_name in enumerate(SOC40_CLASSES):
            for path in sorted((Path(root) / class_name).glob("X_HH_*.tif")):
                _, meta = read_annotation(path.with_suffix(".xml"))
                self.records.append((path, class_id, DEPRESSION_TO_ID[int(meta["depression"])]))
        if not self.records:
            raise RuntimeError(f"no X/HH style ROIs under {root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        path, class_id, depression = self.records[index]
        with Image.open(path) as image:
            return image_tensor(image, 64, False), class_id, depression


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit style prior and save an inference-ready GAN checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shrinkage", type=float, default=.08)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if state.get("architecture") != "continuous_spatial_style_v2":
        raise RuntimeError("checkpoint is not continuous_spatial_style_v2")
    encoder = SARStyleEncoder(int(state["style_dim"])).to(device)
    encoder.load_state_dict(state["style_encoder"]); encoder.eval()
    dataset = StyleROIDataset(args.sar_root)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    styles: list[list[list[torch.Tensor]]] = [[[] for _ in range(4)] for _ in SOC40_CLASSES]
    with torch.inference_mode():
        for roi, class_id, depression in tqdm(loader, desc="encoding real SAR styles"):
            _, mu, _ = encoder(roi.to(device, non_blocking=True), sample=False)
            for label in range(len(SOC40_CLASSES)):
                for dep in range(4):
                    mask = (class_id == label) & (depression == dep)
                    if mask.any():
                        styles[label][dep].append(mu[mask.to(device)].cpu())
    means, cholesky, counts = [], [], []
    for class_values in styles:
        class_means, class_cholesky, class_counts = [], [], []
        for values in class_values:
            values = torch.cat(values).float()
            mean = values.mean(0); centered = values - mean
            covariance = centered.T @ centered / max(1, len(values) - 1)
            diagonal = torch.diag(torch.diag(covariance))
            covariance = (1 - args.shrinkage) * covariance + args.shrinkage * diagonal
            covariance = covariance + torch.eye(len(mean)) * 1e-4
            class_means.append(mean); class_cholesky.append(torch.linalg.cholesky(covariance))
            class_counts.append(len(values))
        means.append(torch.stack(class_means)); cholesky.append(torch.stack(class_cholesky))
        counts.append(class_counts)
    state["style_prior_mean"] = torch.stack(means)
    state["style_prior_cholesky"] = torch.stack(cholesky)
    state["style_prior_counts"] = counts
    state["style_prior_kind"] = "class_depression_conditional_gaussian"
    state["style_prior_source"] = str(args.sar_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print({"output": str(args.output), "counts": counts, "style_dim": int(state["style_dim"])})


if __name__ == "__main__":
    main()

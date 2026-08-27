from __future__ import annotations
import random
import re
from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

SAR_RE = re.compile(r"^(X|KU)_(HH|HV|VH|VV)_(15|30|45|60)_(\d{1,3})_\d+$", re.I)

def angular_distance(a: int, b: int) -> int:
    d = abs(a % 360 - b % 360)
    return min(d, 360 - d)

def rgb_index_to_azimuth(index: int, offset: int = 0) -> int:
    if not 1 <= index <= 12:
        raise ValueError("RGB direction index must be in [1, 12]")
    return (offset + (index - 1) * 30) % 360

def parse_sar(path: Path) -> dict[str, object] | None:
    match = SAR_RE.match(path.stem)
    if not match:
        return None
    band, pol, depression, azimuth = match.groups()
    return {"band": band.upper(), "pol": pol.upper(), "depression": int(depression), "azimuth": int(azimuth)}

def rgba_to_rgb(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")

class DirectionDataset(Dataset):
    """Class-matched but pixel-unaligned RGB/SAR samples at one azimuth."""
    def __init__(self, rgb_root: Path, sar_root: Path, rgb_index: int, image_size: int = 128,
                 angle_offset: int = 0, angle_tolerance: int = 15, band: str = "all",
                 polarization: str = "all", depression: str = "all", epoch_size: int = 0) -> None:
        self.azimuth = rgb_index_to_azimuth(rgb_index, angle_offset)
        self.rgb_by_class: dict[str, Path] = {}
        self.sar_by_class: dict[str, list[Path]] = defaultdict(list)
        for class_dir in sorted(rgb_root.iterdir()):
            candidate = class_dir / f"{rgb_index}.png"
            if class_dir.is_dir() and candidate.is_file():
                self.rgb_by_class[class_dir.name] = candidate
        for path in sar_root.rglob("*.tif"):
            meta = parse_sar(path)
            if meta is None or angular_distance(int(meta["azimuth"]), self.azimuth) > angle_tolerance:
                continue
            if band != "all" and meta["band"] != band.upper(): continue
            if polarization != "all" and meta["pol"] != polarization.upper(): continue
            if depression != "all" and int(meta["depression"]) != int(depression): continue
            if path.parent.name in self.rgb_by_class:
                self.sar_by_class[path.parent.name].append(path)
        self.classes = sorted(set(self.rgb_by_class) & set(self.sar_by_class))
        if not self.classes:
            raise RuntimeError(f"No shared class has RGB direction {rgb_index} and SAR near {self.azimuth} degrees")
        self.sar_paths = [p for c in self.classes for p in self.sar_by_class[c]]
        self.epoch_size = epoch_size or len(self.sar_paths)
        self.image_size = image_size

    def _tensor(self, image: Image.Image, channels: int) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
        if channels == 1: array = array[:, :, None]
        return torch.from_numpy(array.transpose(2, 0, 1).copy()) / 127.5 - 1.0

    def __len__(self) -> int: return self.epoch_size

    def __getitem__(self, index: int) -> dict[str, object]:
        sar_path = self.sar_paths[index % len(self.sar_paths)]
        class_name = sar_path.parent.name
        rgb_path = self.rgb_by_class[class_name]
        with Image.open(rgb_path) as image:
            # Source PNGs can be very large; shrink before RGBA compositing to avoid multi-GB peaks.
            image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            rgb = self._tensor(rgba_to_rgb(image), 3)
        sar_path = random.choice(self.sar_by_class[class_name])
        with Image.open(sar_path) as image:
            sar = self._tensor(image.convert("L"), 1)
        return {"rgb": rgb, "sar": sar, "class": class_name, "rgb_path": str(rgb_path), "sar_path": str(sar_path)}

    def summary(self) -> dict[str, int]:
        return {"classes": len(self.classes), "rgb_images": len(self.classes), "sar_images": len(self.sar_paths), "azimuth": self.azimuth}

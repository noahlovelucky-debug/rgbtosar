"""Paired-by-identity-and-azimuth data for joint RGB->SAR ROI training."""
from __future__ import annotations

import random
import hashlib
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from bbox_data import image_tensor, metadata_vector, read_annotation
from rgb2sar.data import rgba_to_rgb
from saratrx import SOC40_CLASSES


def source_rgb_angle(path: Path, has_degree_names: bool) -> int:
    """Map either ``0.png`` degree names or original ``1.png=0°`` names."""
    value = int(path.stem)
    if has_degree_names:
        if not 0 <= value < 360:
            raise ValueError(path)
        return value
    if not 1 <= value <= 12:
        raise ValueError(path)
    return (value - 1) * 30


def nearest_available_angle(azimuth: int, available: list[int]) -> int:
    return min(available, key=lambda angle: min((azimuth - angle) % 360,
                                                 (angle - azimuth) % 360))


class JointROIDataset(Dataset):
    """One real SAR ROI and its class/angle-matched RGB reference per item.

    SAR and RGB are not assumed to be pixel registered.  The real ROI is used
    only for distribution/structure supervision, not as a strict paired target.
    """

    def __init__(self, rgb_root: Path, sar_root: Path, rgb_size: int = 128,
                 roi_size: int = 64, epoch_size: int = 0, pre_cropped: bool = True,
                 band: str = "all", polarization: str = "all", depression: str = "all",
                 augment_rgb: bool = True, preload_rgb: bool = True,
                 source_view_mode: str = "nearest",
                 return_all_views: bool = False, return_rgb_mask: bool = False,
                 cache_dir: Path | None = None) -> None:
        self.rgb_root = Path(rgb_root)
        self.sar_root = Path(sar_root)
        self.rgb_size = rgb_size
        self.roi_size = roi_size
        self.pre_cropped = pre_cropped
        self.augment_rgb = augment_rgb
        self.return_all_views = return_all_views
        self.return_rgb_mask = return_rgb_mask
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir()) / "rgb2sar_cache"
        if source_view_mode not in {"nearest", "random", "mixed"}:
            raise ValueError("source_view_mode must be nearest, random, or mixed")
        self.source_view_mode = source_view_mode
        self.classes = list(SOC40_CLASSES)
        self.class_to_id = {name: index for index, name in enumerate(self.classes)}
        self.rgb_paths: dict[tuple[str, int], Path] = {}
        self.class_rgb_angles: dict[str, list[int]] = {}
        naming_formats = set()
        for class_name in self.classes:
            folder = self.rgb_root / class_name
            numeric = [path for path in folder.glob("*.png") if path.stem.isdigit()]
            has_degree_names = any(path.stem == "0" for path in numeric)
            naming_formats.add("degrees" if has_degree_names else "original_1_to_12")
            for path in numeric:
                try:
                    angle = source_rgb_angle(path, has_degree_names)
                except ValueError:
                    continue
                self.rgb_paths[class_name, angle] = path
            self.class_rgb_angles[class_name] = sorted(
                angle for (name, angle) in self.rgb_paths if name == class_name)
        missing_classes = [name for name in self.classes if not any(key[0] == name for key in self.rgb_paths)]
        if missing_classes:
            raise RuntimeError(f"RGB is missing classes: {missing_classes}")
        self.rgb_naming = ",".join(sorted(naming_formats))

        records = self._load_sar_records(band, polarization, depression)
        if not records:
            raise RuntimeError(f"no valid joint records under {self.sar_root}")
        self.records = records
        self.epoch_size = epoch_size or len(records)
        self.random_epoch = 0 < epoch_size < len(records)
        self._rgb_cache: dict[Path, torch.Tensor] = {}
        self._rgb_mask_cache: dict[Path, torch.Tensor] = {}
        # Populate once in the parent process. DataLoader fork workers then
        # share these read-only base tensors via copy-on-write instead of each
        # worker decoding the same 466 PNG files independently.
        if preload_rgb:
            self._preload_rgb_cache(sorted(set(self.rgb_paths.values())))
            if self.return_rgb_mask:
                self._preload_rgb_mask_cache(sorted(set(self.rgb_paths.values())))

    def _record_cache_signature(self, band: str, polarization: str,
                                depression: str) -> dict[str, object]:
        def directory_mtime(root: Path, name: str) -> int | None:
            folder = root / name
            try:
                return folder.stat().st_mtime_ns
            except OSError:
                return None

        return {
            "version": 1,
            "sar_root": str(self.sar_root.resolve()),
            "rgb_root": str(self.rgb_root.resolve()),
            "band": band,
            "polarization": polarization,
            "depression": depression,
            "sar_class_mtimes": [directory_mtime(self.sar_root, name) for name in self.classes],
            "rgb_class_mtimes": [directory_mtime(self.rgb_root, name) for name in self.classes],
        }

    def _record_cache_file(self, band: str, polarization: str, depression: str) -> Path:
        key = "|".join((str(self.sar_root.resolve()), str(self.rgb_root.resolve()),
                        band, polarization, depression, "joint-sar-records-v1"))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"joint_sar_records_{digest}.pt"

    def _decode_cached_records(self, records: list[dict[str, object]]) -> list[tuple]:
        return [
            (
                self.sar_root / str(record["tif"]),
                self.rgb_root / str(record["rgb"]),
                str(record["class_name"]),
                tuple(record["bbox"]),
                dict(record["meta"]),
                int(record["rgb_angle"]),
            )
            for record in records
        ]

    def _load_sar_records(self, band: str, polarization: str,
                          depression: str) -> list[tuple]:
        """Reuse parsed XML records across short ablation processes.

        XML parsing on the network-mounted SOC dataset is much slower than an
        epoch of GPU training.  The cache stores metadata only and is invalidated
        when any participating class directory changes.
        """
        signature = self._record_cache_signature(band, polarization, depression)
        cache_file = self._record_cache_file(band, polarization, depression)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.is_file():
            try:
                saved = torch.load(cache_file, map_location="cpu", weights_only=True)
                if saved.get("signature") == signature:
                    return self._decode_cached_records(saved["records"])
            except Exception:
                pass

        records = []
        for class_name in self.classes:
            folder = self.sar_root / class_name
            if not folder.is_dir():
                continue
            for tif in folder.glob("*.tif"):
                xml = tif.with_suffix(".xml")
                if not xml.is_file():
                    continue
                try:
                    bbox, meta = read_annotation(xml)
                except Exception:
                    continue
                if band != "all" and meta["band"] != band.upper():
                    continue
                if polarization != "all" and meta["pol"] != polarization.upper():
                    continue
                if depression != "all" and int(meta["depression"]) != int(depression):
                    continue
                rgb_angle = nearest_available_angle(int(meta["azimuth"]),
                                                    self.class_rgb_angles[class_name])
                rgb_path = self.rgb_paths.get((class_name, rgb_angle))
                if rgb_path is not None:
                    records.append((tif, rgb_path, class_name, bbox, meta, rgb_angle))
        cache_records = [
            {
                "tif": str(tif.relative_to(self.sar_root)),
                "rgb": str(rgb.relative_to(self.rgb_root)),
                "class_name": class_name,
                "bbox": tuple(bbox),
                "meta": dict(meta),
                "rgb_angle": rgb_angle,
            }
            for tif, rgb, class_name, bbox, meta, rgb_angle in records
        ]
        temporary = cache_file.with_suffix(cache_file.suffix + f".{os.getpid()}.tmp")
        torch.save({"signature": signature, "records": cache_records}, temporary)
        temporary.replace(cache_file)
        return records

    def __len__(self) -> int:
        return self.epoch_size

    def _decode_rgb_uint8(self, path: Path) -> tuple[Path, torch.Tensor]:
        with Image.open(path) as image:
            tensor = image_tensor(rgba_to_rgb(image), self.rgb_size, True)
        return path, ((tensor + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)

    def _decode_rgb_mask_uint8(self, path: Path) -> tuple[Path, torch.Tensor]:
        """Decode the source PNG alpha channel as a cached 0..255 mask.

        RGB compositing intentionally remains unchanged for existing trainers.
        The mask is an optional geometric signal for the unpaired bridge model;
        images without an alpha channel are treated as fully foreground.
        """
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            tensor = image_tensor(alpha, self.rgb_size, False)
        return path, ((tensor + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)

    def _preload_rgb_cache(self, paths: list[Path]) -> None:
        # The dataset may live under a non-ASCII or read-only mount. PyTorch's
        # zip writer needs an ASCII writable path, so keep derived cache data
        # outside the source dataset and namespace it by resolved RGB root.
        source_hash = hashlib.sha256(str(self.rgb_root.resolve()).encode("utf-8")).hexdigest()[:16]
        cache_file = self.cache_dir / f"joint_rgb_cache_{source_hash}_{self.rgb_size}.pt"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        signature = [(str(path.relative_to(self.rgb_root)), path.stat().st_size,
                      path.stat().st_mtime_ns) for path in paths]
        if cache_file.is_file():
            try:
                saved = torch.load(cache_file, map_location="cpu", weights_only=True)
                if saved.get("signature") == signature:
                    self._rgb_cache = {self.rgb_root / name: tensor
                                       for name, tensor in saved["images"].items()}
                    return
            except Exception:
                pass
        print(f"building one-time RGB thumbnail cache: {cache_file} ({len(paths)} images)",
              flush=True)
        with ThreadPoolExecutor(max_workers=min(16, max(1, len(paths)))) as executor:
            for path, tensor in executor.map(self._decode_rgb_uint8, paths):
                self._rgb_cache[path] = tensor
        payload = {"signature": signature,
                   "images": {str(path.relative_to(self.rgb_root)): tensor
                              for path, tensor in self._rgb_cache.items()}}
        temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(cache_file)

    def _preload_rgb_mask_cache(self, paths: list[Path]) -> None:
        source_hash = hashlib.sha256(str(self.rgb_root.resolve()).encode("utf-8")).hexdigest()[:16]
        cache_file = self.cache_dir / f"joint_rgb_mask_cache_{source_hash}_{self.rgb_size}.pt"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        signature = [(str(path.relative_to(self.rgb_root)), path.stat().st_size,
                      path.stat().st_mtime_ns) for path in paths]
        if cache_file.is_file():
            try:
                saved = torch.load(cache_file, map_location="cpu", weights_only=True)
                if saved.get("signature") == signature:
                    self._rgb_mask_cache = {self.rgb_root / name: tensor
                                            for name, tensor in saved["masks"].items()}
                    return
            except Exception:
                pass
        print(f"building one-time RGB alpha-mask cache: {cache_file} ({len(paths)} images)",
              flush=True)
        with ThreadPoolExecutor(max_workers=min(16, max(1, len(paths)))) as executor:
            for path, tensor in executor.map(self._decode_rgb_mask_uint8, paths):
                self._rgb_mask_cache[path] = tensor
        payload = {"signature": signature,
                   "masks": {str(path.relative_to(self.rgb_root)): tensor
                             for path, tensor in self._rgb_mask_cache.items()}}
        temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(cache_file)

    def _rgb_base(self, path: Path) -> torch.Tensor:
        cached = self._rgb_cache.get(path)
        if cached is None:
            _, cached = self._decode_rgb_uint8(path)
            self._rgb_cache[path] = cached
        return cached.float().div(127.5).sub(1.0)

    def _rgb(self, path: Path) -> torch.Tensor:
        cached = self._rgb_base(path)
        rgb = cached.clone()
        if self.augment_rgb:
            gain = random.uniform(0.9, 1.1)
            bias = random.uniform(-0.04, 0.04)
            noise = torch.randn_like(rgb) * random.uniform(0.0, 0.015)
            rgb = (rgb * gain + bias + noise).clamp(-1.0, 1.0)
        return rgb

    def _rgb_mask(self, path: Path) -> torch.Tensor:
        cached = self._rgb_mask_cache.get(path)
        if cached is None:
            _, cached = self._decode_rgb_mask_uint8(path)
            self._rgb_mask_cache[path] = cached
        return cached.float().div(255.0)

    def __getitem__(self, index: int) -> dict[str, object]:
        if self.random_epoch:
            index = random.randrange(len(self.records))
        tif, nearest_rgb_path, class_name, bbox, meta, nearest_angle = self.records[index % len(self.records)]
        alternatives = self.class_rgb_angles[class_name]
        if self.source_view_mode == "random" or (self.source_view_mode == "mixed" and random.random() < .5):
            rgb_angle = random.choice(alternatives)
        else:
            rgb_angle = nearest_angle
        rgb_path = self.rgb_paths[class_name, rgb_angle]
        alternate_angle = random.choice([angle for angle in alternatives if angle != rgb_angle] or alternatives)
        alternate_path = self.rgb_paths[class_name, alternate_angle]
        with Image.open(tif) as image:
            source = image if self.pre_cropped else image.crop(bbox)
            roi = image_tensor(source, self.roi_size, False)
        item = {
            "rgb": self._rgb(rgb_path),
            "rgb_alt": self._rgb(alternate_path),
            "roi": roi,
            "meta": metadata_vector(meta, bbox),
            "class_id": self.class_to_id[class_name],
            "class_name": class_name,
            "bbox": torch.tensor(bbox, dtype=torch.long),
            "azimuth": int(meta["azimuth"]),
            "depression": int(meta["depression"]),
            "rgb_angle": rgb_angle,
            "rgb_alt_angle": alternate_angle,
            "rgb_path": str(rgb_path),
            "sar_path": str(tif),
        }
        if self.return_rgb_mask:
            item.update({
                "rgb_mask": self._rgb_mask(rgb_path),
                "rgb_alt_mask": self._rgb_mask(alternate_path),
            })
        if self.return_all_views:
            # Keep the original 12-view convention (1.png=0°, ..., 12.png=330°)
            # without synthesising missing RGB views.  A fixed shape and mask
            # make the representation safe for DataLoader batching.
            views, masks = [], []
            for angle in range(0, 360, 30):
                path = self.rgb_paths.get((class_name, angle))
                if path is None:
                    views.append(torch.zeros(3, self.rgb_size, self.rgb_size))
                    masks.append(0.0)
                else:
                    views.append(self._rgb(path))
                    masks.append(1.0)
            item.update({
                "rgb_views": torch.stack(views),
                "rgb_view_angles": torch.arange(0, 360, 30, dtype=torch.float32),
                "rgb_view_mask": torch.tensor(masks, dtype=torch.float32),
            })
        return item

    def summary(self) -> dict[str, object]:
        return {
            "classes": len(self.classes),
            "records": len(self.records),
            "epoch_size": self.epoch_size,
            "rgb_views": len(self.rgb_paths),
            "rgb_naming": self.rgb_naming,
            "source_view_mode": self.source_view_mode,
            "return_all_views": self.return_all_views,
            "return_rgb_mask": self.return_rgb_mask,
            "pre_cropped": self.pre_cropped,
        }

"""Segment/normalise one vehicle turntable and estimate cameras with COLMAP SfM."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def angle_from_source(path: Path) -> int:
    index = int(path.stem)
    if not 1 <= index <= 12:
        raise ValueError(path)
    return (index - 1) * 30


def on_canvas(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGBA")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas.alpha_composite(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def square_union_crop(images: dict[int, Image.Image], margin: float) -> tuple[int, int, int, int]:
    boxes = [image.getchannel("A").getbbox() for image in images.values()]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        raise RuntimeError("all source alpha masks are empty")
    x0, y0 = min(box[0] for box in boxes), min(box[1] for box in boxes)
    x1, y1 = max(box[2] for box in boxes), max(box[3] for box in boxes)
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return tuple(round(value) for value in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))


def prepare(source: Path, output: Path, resolution: int, margin: float,
            overwrite: bool) -> dict[str, object]:
    rgba_dir, images_dir, masks_dir = output / "rgba", output / "images", output / "masks"
    if overwrite and output.exists():
        shutil.rmtree(output)
    for folder in (rgba_dir, images_dir, masks_dir):
        folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for path in source.glob("*.png"):
        try: paths[angle_from_source(path)] = path
        except (ValueError, TypeError): continue
    if len(paths) < 6:
        raise RuntimeError(f"only {len(paths)} usable views in {source}")
    canvas_images = {angle: on_canvas(path, (1024, 768)) for angle, path in paths.items()}
    crop = square_union_crop(canvas_images, margin)
    frames = []
    for angle, image in sorted(canvas_images.items()):
        rgba = image.crop(crop).resize((resolution, resolution), Image.Resampling.LANCZOS)
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        # Close tiny segmentation holes, then erode one pixel so SIFT does not
        # lock onto the artificial cut-out boundary.
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        feature_mask = cv2.erode((alpha > 8).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        rgba_array = np.asarray(rgba, dtype=np.uint8).copy()
        rgba_array[..., 3] = alpha
        rgba = Image.fromarray(rgba_array, "RGBA")
        filename = f"{angle}.png"
        rgba.save(rgba_dir / filename, compress_level=4)
        background = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
        Image.alpha_composite(background, rgba).convert("RGB").save(images_dir / filename)
        Image.fromarray(feature_mask, "L").save(masks_dir / f"{filename}.png")
        frames.append({"angle": angle, "source": str(paths[angle]), "rgba": str(rgba_dir / filename)})
    manifest = {"class": source.name, "resolution": resolution, "canvas": [1024, 768],
                "shared_crop": list(crop), "views": len(frames), "frames": frames}
    (output / "prepare.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_sfm(output: Path, min_registered: int) -> dict[str, object]:
    try:
        import pycolmap
    except ImportError as error:
        raise RuntimeError("pycolmap is required: pip install pycolmap") from error
    database, sparse = output / "database.db", output / "sparse"
    if database.exists(): database.unlink()
    if sparse.exists(): shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    reader = pycolmap.ImageReaderOptions()
    reader.mask_path = str((output / "masks").resolve())
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = 1600
    extraction.sift.max_num_features = 16384
    extraction.sift.peak_threshold = 0.003
    pycolmap.extract_features(str(database), str(output / "images"),
                              camera_mode=pycolmap.CameraMode.SINGLE,
                              camera_model="SIMPLE_RADIAL", reader_options=reader,
                              extraction_options=extraction, device=pycolmap.Device.cpu)
    matching = pycolmap.FeatureMatchingOptions()
    matching.guided_matching = True
    matching.sift.max_ratio = 0.9
    matching.sift.max_distance = 0.8
    pycolmap.match_exhaustive(str(database), matching_options=matching, device=pycolmap.Device.cpu)
    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.min_model_size = min(6, min_registered)
    mapping.multiple_models = True
    mapping.mapper.init_min_num_inliers = 40
    mapping.mapper.abs_pose_min_num_inliers = 20
    mapping.mapper.abs_pose_min_inlier_ratio = 0.15
    reconstructions = pycolmap.incremental_mapping(str(database), str(output / "images"),
                                                    str(sparse), options=mapping)
    if not reconstructions:
        report = {"success": False, "registered": 0, "required": min_registered,
                  "reason": "COLMAP produced no model"}
    else:
        model_id, reconstruction = max(reconstructions.items(),
                                       key=lambda item: item[1].num_reg_images())
        registered = reconstruction.num_reg_images()
        names = sorted(image.name for image in reconstruction.images.values() if image.has_pose)
        report = {"success": registered >= min_registered, "model_id": model_id,
                  "registered": registered, "required": min_registered,
                  "points3D": reconstruction.num_points3D(), "registered_images": names,
                  "model_path": str(sparse / str(model_id))}
    (output / "camera_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RGBA turntable views and run masked COLMAP SfM")
    parser.add_argument("--rgb-class", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--min-registered", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-sfm", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.rgb_class, args.output, args.resolution, args.margin, args.overwrite)
    print("prepared", manifest["class"], manifest["views"], "views")
    if not args.skip_sfm:
        print("camera estimation:", run_sfm(args.output, min(args.min_registered, manifest["views"])))


if __name__ == "__main__":
    main()

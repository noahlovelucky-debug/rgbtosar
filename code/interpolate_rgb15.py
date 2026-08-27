"""Build a 15-degree RGB reference set with alpha-aware affine morphing.

The source convention is 1.png=0 degrees, ..., 12.png=330 degrees.  Exact
source views are normalised onto a common transparent canvas.  Other views
(including missing 30-degree source views) are generated from the closest
clockwise bracketing source images.  Both sources are affinely aligned to an
interpolated foreground box before premultiplied-alpha blending.

This is a deterministic geometric baseline, not a physically exact novel-view
renderer.  The manifest written beside the images records which two source
views and interpolation fraction were used for every output.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


TARGET_ANGLES = tuple(range(0, 360, 15))


def source_angle(path: Path) -> int:
    index = int(path.stem)
    if not 1 <= index <= 12:
        raise ValueError(f"RGB source index must be 1..12: {path}")
    return (index - 1) * 30


def clockwise_delta(start: int, end: int) -> int:
    return (end - start) % 360


def bracket(available: list[int], target: int) -> tuple[int, int, float]:
    """Return circular left/right source angles and clockwise fraction."""
    if target in available:
        return target, target, 0.0
    left = min(available, key=lambda angle: clockwise_delta(angle, target))
    right = min(available, key=lambda angle: clockwise_delta(target, angle))
    span = clockwise_delta(left, right)
    if span == 0:
        return left, right, 0.0
    return left, right, clockwise_delta(left, target) / span


def alpha_bbox(array: np.ndarray) -> tuple[float, float, float, float]:
    alpha = array[..., 3]
    ys, xs = np.nonzero(alpha > 2)
    if len(xs) == 0:
        return 0.0, 0.0, float(array.shape[1]), float(array.shape[0])
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def normalise_rgba(path: Path, canvas: tuple[int, int], margin: float) -> Image.Image:
    """Crop the alpha foreground and place it on a stable common canvas."""
    with Image.open(path) as opened:
        image = opened.convert("RGBA")
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError(f"empty alpha channel: {path}")
    foreground = image.crop(bbox)
    width, height = canvas
    usable_w = max(1, round(width * (1.0 - 2.0 * margin)))
    usable_h = max(1, round(height * (1.0 - 2.0 * margin)))
    scale = min(usable_w / foreground.width, usable_h / foreground.height)
    size = (max(1, round(foreground.width * scale)), max(1, round(foreground.height * scale)))
    foreground = foreground.resize(size, Image.Resampling.LANCZOS)
    result = Image.new("RGBA", canvas, (0, 0, 0, 0))
    # Keep a small downward bias so wheels/base remain spatially stable.
    x = (width - size[0]) // 2
    y = round((height - size[1]) * 0.58)
    result.alpha_composite(foreground, (x, y))
    return result


def warp_box(image: Image.Image, source: tuple[float, float, float, float],
             target: tuple[float, float, float, float]) -> Image.Image:
    """Affine-warp source foreground bounding box to target bounding box."""
    sx0, sy0, sx1, sy1 = source
    tx0, ty0, tx1, ty1 = target
    scale_x = max(1e-6, (tx1 - tx0) / max(1e-6, sx1 - sx0))
    scale_y = max(1e-6, (ty1 - ty0) / max(1e-6, sy1 - sy0))
    # PIL expects output -> input (inverse) affine coefficients.
    coefficients = (
        1.0 / scale_x, 0.0, sx0 - tx0 / scale_x,
        0.0, 1.0 / scale_y, sy0 - ty0 / scale_y,
    )
    return image.transform(image.size, Image.Transform.AFFINE, coefficients,
                           resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))


def premultiplied_blend(left: Image.Image, right: Image.Image, fraction: float) -> Image.Image:
    a = np.asarray(left, dtype=np.float32) / 255.0
    b = np.asarray(right, dtype=np.float32) / 255.0
    aa, ba = a[..., 3:4], b[..., 3:4]
    alpha = (1.0 - fraction) * aa + fraction * ba
    colour = (1.0 - fraction) * a[..., :3] * aa + fraction * b[..., :3] * ba
    colour = colour / np.maximum(alpha, 1e-6)
    out = np.concatenate((colour, alpha), axis=-1)
    return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGBA")


def detail_preserving_blend(left: Image.Image, right: Image.Image, fraction: float) -> Image.Image:
    """Blend low frequencies but retain one source's crisp high frequencies.

    A plain cross-dissolve duplicates headlights, wheels and windows because a
    3-D viewpoint change is not a global affine motion.  The closest source
    supplies alpha and fine detail; both sources still determine the affine
    target box and low-frequency appearance.
    """
    a = np.asarray(left, dtype=np.float32) / 255.0
    b = np.asarray(right, dtype=np.float32) / 255.0
    chosen = a if fraction <= 0.5 else b
    radius = max(4.0, min(left.size) / 48.0)
    low_a = np.asarray(left.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    low_b = np.asarray(right.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    other = b if fraction <= 0.5 else a
    low_chosen = low_a if fraction <= 0.5 else low_b
    low_other = low_b if fraction <= 0.5 else low_a
    # Only transfer smooth appearance where both warped silhouettes overlap.
    # Keeping the correction deliberately small avoids low-alpha ringing.
    overlap = np.minimum(chosen[..., 3:4], other[..., 3:4])
    strength = (4.0 * fraction * (1.0 - fraction)) * 0.22
    colour = chosen[..., :3] + strength * overlap * (low_other[..., :3] - low_chosen[..., :3])
    alpha = chosen[..., 3:4]
    output = np.concatenate((np.clip(colour, 0, 1), alpha), axis=-1)
    return Image.fromarray(np.clip(output * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGBA")


def morph(left: Image.Image, right: Image.Image, fraction: float) -> Image.Image:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    left_box, right_box = alpha_bbox(left_array), alpha_bbox(right_array)
    target_box = tuple((1.0 - fraction) * x + fraction * y for x, y in zip(left_box, right_box))
    return detail_preserving_blend(warp_box(left, left_box, target_box),
                                   warp_box(right, right_box, target_box), fraction)


def build_class(source_dir: Path, output_dir: Path, canvas: tuple[int, int], margin: float,
                overwrite: bool) -> list[dict[str, object]]:
    sources: dict[int, Path] = {}
    for path in source_dir.glob("*.png"):
        try:
            sources[source_angle(path)] = path
        except (ValueError, TypeError):
            continue
    if len(sources) < 2:
        raise RuntimeError(f"need at least two valid source views in {source_dir}")
    available = sorted(sources)
    normalised = {angle: normalise_rgba(path, canvas, margin) for angle, path in sources.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for target in TARGET_ANGLES:
        destination = output_dir / f"{target}.png"
        left_angle, right_angle, fraction = bracket(available, target)
        if overwrite or not destination.exists():
            if left_angle == right_angle:
                result = normalised[left_angle]
                method = "source_affine_normalised"
            else:
                result = morph(normalised[left_angle], normalised[right_angle], fraction)
                method = "affine_alpha_morph"
            temporary = destination.with_suffix(".tmp.png")
            result.save(temporary, compress_level=4)
            temporary.replace(destination)
        rows.append({
            "class": source_dir.name,
            "target_angle": target,
            "left_angle": left_angle,
            "right_angle": right_angle,
            "fraction": f"{fraction:.6f}",
            "left_file": sources[left_angle].name,
            "right_file": sources[right_angle].name,
            "method": "source_affine_normalised" if left_angle == right_angle else "affine_detail_morph",
            "output": str(destination),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 0..345 degree RGB references at 15 degree spacing")
    parser.add_argument("--input", type=Path, required=True, help="RGB root containing one folder per vehicle")
    parser.add_argument("--output", type=Path, required=True, help="destination RGB_15 root")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--margin", type=float, default=0.06)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.margin < 0.45:
        raise ValueError("--margin must be in [0, 0.45)")
    classes = sorted(path for path in args.input.iterdir() if path.is_dir())
    if not classes:
        raise RuntimeError(f"no class folders under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, source_dir in enumerate(classes, 1):
        print(f"[{index:02d}/{len(classes):02d}] {source_dir.name}", flush=True)
        rows.extend(build_class(source_dir, args.output / source_dir.name,
                                (args.width, args.height), args.margin, args.overwrite))
    fields = ["class", "target_angle", "left_angle", "right_angle", "fraction",
              "left_file", "right_file", "method", "output"]
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} images for {len(classes)} classes to {args.output}")


if __name__ == "__main__":
    main()

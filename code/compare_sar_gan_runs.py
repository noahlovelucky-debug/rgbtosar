"""Create a fair progress comparison for the two-stage and one-stage runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


METRICS = (
    ("validation_clean_structure", "clean structure (lower)"),
    ("validation_clean_statistics", "clean statistics (lower)"),
    ("validation_noise_statistics", "noise statistics (lower)"),
    ("validation_full_statistics", "full SAR statistics (lower)"),
    ("validation_spectrum", "spectrum gap (lower)"),
    ("validation_teacher_accuracy", "real-SAR classifier accuracy (higher)"),
)


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)]


def matching_preview_epoch(run_a: Path, run_b: Path) -> int | None:
    def epochs(folder: Path) -> set[int]:
        return {
            int(path.stem.rsplit("_", 1)[-1])
            for path in folder.glob("validation_*.png")}
    common = epochs(run_a) & epochs(run_b)
    return max(common) if common else None


def side_by_side(run_a: Path, run_b: Path, epoch: int,
                 output: Path, names: tuple[str, str]) -> None:
    images = [
        Image.open(folder / f"validation_{epoch:03d}.png").convert("RGB")
        for folder in (run_a, run_b)]
    header = 26
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + header
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    offset = 0
    for image, name in zip(images, names):
        draw.text((offset + 4, 5), f"{name} - epoch {epoch}", fill="black")
        canvas.paste(image, (offset, header))
        offset += image.width
    canvas.save(output)


def draw_curves(runs: dict[str, list[dict[str, float]]],
                output: Path) -> None:
    """Dependency-free 2x3 comparison chart."""
    cell_width, cell_height = 480, 280
    margin_left, margin_top, margin_bottom = 58, 34, 38
    canvas = Image.new(
        "RGB", (cell_width * 3, cell_height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    colours = {"two-stage": (36, 96, 180), "one-stage": (220, 80, 55)}
    for metric_index, (key, title) in enumerate(METRICS):
        column, row = metric_index % 3, metric_index // 3
        ox, oy = column * cell_width, row * cell_height
        left, right = ox + margin_left, ox + cell_width - 16
        top, bottom = oy + margin_top, oy + cell_height - margin_bottom
        values = [
            item[key] for rows in runs.values() for item in rows]
        epochs = [
            item["epoch"] for rows in runs.values() for item in rows]
        low, high = min(values), max(values)
        padding = max((high - low) * .08, 1e-5)
        low, high = low - padding, high + padding
        min_epoch, max_epoch = min(epochs), max(epochs)
        max_epoch = max(max_epoch, min_epoch + 1)
        draw.line((left, top, left, bottom), fill="black")
        draw.line((left, bottom, right, bottom), fill="black")
        draw.text((ox + 8, oy + 6), title, fill="black")
        draw.text((ox + 2, top - 4), f"{high:.3f}", fill="black")
        draw.text((ox + 2, bottom - 10), f"{low:.3f}", fill="black")
        draw.text((left, bottom + 8), f"epoch {int(min_epoch)}", fill="black")
        draw.text((right - 58, bottom + 8), f"{int(max_epoch)}", fill="black")
        for name, rows in runs.items():
            points = []
            for item in rows:
                x = left + (item["epoch"] - min_epoch) / (
                    max_epoch - min_epoch) * (right - left)
                y = bottom - (item[key] - low) / (
                    high - low) * (bottom - top)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=colours[name], width=3)
            elif points:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3),
                             fill=colours[name])
            legend_x = ox + cell_width - 155
            legend_y = oy + 7 + 13 * list(runs).index(name)
            draw.line((legend_x, legend_y + 4, legend_x + 18,
                       legend_y + 4), fill=colours[name], width=3)
            draw.text((legend_x + 23, legend_y), name, fill="black")
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-stage", type=Path, required=True)
    parser.add_argument("--one-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {
        "two-stage": read_history(args.two_stage / "history.csv"),
        "one-stage": read_history(args.one_stage / "history.csv"),
    }
    draw_curves(runs, args.output / "comparison_curves.png")

    summary = {}
    for name, rows in runs.items():
        latest = rows[-1]
        summary[name] = {
            "epoch": int(latest["epoch"]),
            **{key: latest[key] for key, _ in METRICS}}
    epoch = matching_preview_epoch(args.two_stage, args.one_stage)
    if epoch is not None:
        side_by_side(
            args.two_stage, args.one_stage, epoch,
            args.output / f"comparison_epoch_{epoch:03d}.png",
            ("two-stage dual generator", "one-stage wavelet"))
        summary["matching_preview_epoch"] = epoch
        summary["matched_metrics"] = {
            name: {
                key: next(
                    row[key] for row in rows
                    if int(row["epoch"]) == epoch)
                for key, _ in METRICS}
            for name, rows in runs.items()}
    (args.output / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

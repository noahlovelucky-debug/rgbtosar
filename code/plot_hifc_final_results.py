"""Create reproducible figures for the final HiFC-unpaired run.

The script only reads a training history and TSTR JSON files.  It writes
diagnostic figures and a compact summary that can be published alongside the
Chinese workflow document without publishing the large checkpoint itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]


def read_tstr(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("hifc_tstr_epoch120_classifier*/selected_metrics.json")):
        match = re.search(r"classifier(\d+)$", path.parent.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["seed"] = int(match.group(1))
        rows.append(payload)
    if not rows:
        raise RuntimeError(f"no final TSTR metrics under {root}")
    return sorted(rows, key=lambda row: int(row["seed"]))


def style_axes(axes: np.ndarray) -> None:
    for axis in np.asarray(axes).ravel():
        axis.grid(True, alpha=.2, linewidth=.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)


def plot_training(history: list[dict[str, float]], output: Path) -> None:
    epoch = np.array([row["epoch"] for row in history])
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    style_axes(axes)
    colors = {
        "generator": "#1769aa", "discriminator": "#c62828", "adversarial": "#ef6c00",
        "rgb_identity": "#6a1b9a", "ltc": "#00897b", "sfm": "#2e7d32",
        "geometry": "#8d6e63", "validation_sfm": "#1b5e20",
        "validation_geometry": "#5d4037", "validation_ltc": "#00695c",
    }

    axes[0, 0].plot(epoch, [row["generator"] for row in history], label="G total",
                    color=colors["generator"], linewidth=2)
    axes[0, 0].plot(epoch, [row["discriminator"] for row in history], label="D total",
                    color=colors["discriminator"], linewidth=2)
    axes[0, 0].plot(epoch, [row["adversarial"] for row in history], label="adversarial",
                    color=colors["adversarial"], linewidth=1.5)
    axes[0, 0].set_title("Optimized totals")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend(ncol=3, fontsize=9)

    for name in ("rgb_identity", "sfm", "geometry"):
        axes[0, 1].plot(epoch, [row[name] for row in history], label=name,
                        color=colors[name], linewidth=1.8)
    axes[0, 1].plot(epoch, np.array([row["ltc"] for row in history]) * 1000,
                    label="ltc ×1000", color=colors["ltc"], linewidth=1.8)
    axes[0, 1].set_title("Generator components")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("raw value (LTC shown ×1000)")
    axes[0, 1].legend(ncol=2, fontsize=9)

    for name in ("validation_sfm", "validation_geometry"):
        axes[1, 0].plot(epoch, [row[name] for row in history], label=name,
                        color=colors[name], linewidth=1.8)
    axes[1, 0].plot(epoch, np.array([row["validation_ltc"] for row in history]) * 1000,
                    label="validation_ltc ×1000", color=colors["validation_ltc"], linewidth=1.8)
    axes[1, 0].set_title("Held-out validation losses")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("loss (LTC shown ×1000)")
    axes[1, 0].legend(fontsize=9)

    metric_specs = (
        ("native_class_accuracy", "native class", "#263238"),
        ("validation_native_class_accuracy", "val native class", "#455a64"),
        ("native_band_accuracy", "band", "#1565c0"),
        ("native_polarization_accuracy", "polarization", "#ad1457"),
        ("native_depression_accuracy", "depression", "#6d4c41"),
        ("native_azimuth_accuracy", "azimuth", "#00838f"),
    )
    for name, label, color in metric_specs:
        axes[1, 1].plot(epoch, np.array([row[name] for row in history]) * 100,
                        label=label, color=color, linewidth=1.5)
    axes[1, 1].set_title("Frozen native-teacher diagnostics")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("accuracy (%)")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].legend(ncol=2, fontsize=8)

    for axis in axes.ravel():
        axis.axvline(16, color="#757575", linestyle="--", linewidth=1, alpha=.8)
    axes[0, 0].text(16.5, axes[0, 0].get_ylim()[1] * .96, "DDP continuation",
                     color="#616161", fontsize=9, va="top")
    fig.suptitle("HiFC unpaired conditioned v1 | final 120-epoch diagnostics", fontsize=16)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def plot_tstr(rows: list[dict[str, object]], output: Path, summary_path: Path) -> None:
    seeds = [str(row["seed"]) for row in rows]
    top1 = np.array([float(row["top1"]) for row in rows]) * 100
    top5 = np.array([float(row["top5"]) for row in rows]) * 100
    depression_values = (15, 30, 45, 60)
    depression = np.array([
        np.mean([float(row["by_depression"][str(value)]["top1"]) for row in rows]) * 100
        for value in depression_values
    ])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    style_axes(axes)
    x = np.arange(len(rows))
    width = .36
    bars1 = axes[0].bar(x - width / 2, top1, width, label="Top-1", color="#1565c0")
    bars5 = axes[0].bar(x + width / 2, top5, width, label="Top-5", color="#80cbc4")
    axes[0].axhline(14.7465, color="#c62828", linestyle="--", linewidth=1.4,
                    label="old V1 Top-1 = 14.75%")
    axes[0].axhline(39.0621, color="#ef6c00", linestyle="--", linewidth=1.4,
                    label="old V1 Top-5 = 39.06%")
    axes[0].set_xticks(x, [f"seed {seed}" for seed in seeds])
    axes[0].set_ylim(0, 82)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("Generated-only training → real X/HH test")
    axes[0].legend(fontsize=8, loc="upper left")
    for bars in (bars1, bars5):
        axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    bars = axes[1].bar([str(value) + "°" for value in depression_values], depression,
                       color=["#90caf9", "#42a5f5", "#1e88e5", "#ef9a9a"])
    axes[1].set_ylim(0, 70)
    axes[1].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_title("Mean real-test Top-1 by depression")
    axes[1].bar_label(bars, fmt="%.1f", padding=2, fontsize=9)

    summary = {
        "samples": int(rows[0]["samples"]),
        "seeds": [int(row["seed"]) for row in rows],
        "top1_mean": float(np.mean(top1) / 100),
        "top1_std": float(np.std(top1) / 100),
        "top5_mean": float(np.mean(top5) / 100),
        "top5_std": float(np.std(top5) / 100),
        "azimuth_top1_mean": float(np.mean([float(row["azimuth_top1"]) for row in rows])),
        "azimuth_circular_mae_mean_deg": float(np.mean([float(row["azimuth_circular_mae"]) for row in rows])),
        "by_depression_mean_top1": {
            str(value): float(score / 100) for value, score in zip(depression_values, depression)
        },
        "comparison_baseline": {
            "old_v1_top1": 0.147465,
            "old_v1_top5": 0.390621,
            "hifc_epoch005_seed415_top1": 0.2944866920152091,
            "hifc_epoch005_seed415_top5": 0.5876425855513308,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fig.suptitle("Final TSTR audit | 3 classifier seeds", fontsize=16)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def add_box(axis, x: float, y: float, width: float, height: float,
            text: str, color: str, fontsize: int = 10) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=1.3, edgecolor="#37474f", facecolor=color)
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              fontsize=fontsize, color="#263238", wrap=True)
    return x, y, width, height


def arrow(axis, source: tuple[float, float, float, float], target: tuple[float, float, float, float],
          color: str = "#455a64", style: str = "-|>") -> None:
    sx, sy, sw, sh = source
    tx, ty, tw, th = target
    start = (sx + sw, sy + sh / 2) if sx + sw <= tx else (sx + sw / 2, sy)
    end = (tx, ty + th / 2) if sx + sw <= tx else (tx + tw / 2, ty + th)
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13,
                                   linewidth=1.3, color=color, connectionstyle="arc3,rad=0.0"))


def plot_workflow(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    axis.set_xlim(0, 16); axis.set_ylim(0, 9); axis.axis("off")
    axis.set_title("HiFC unpaired conditioned v1 | architecture and gradient workflow", fontsize=16, pad=12)

    rgb = add_box(axis, .4, 6.7, 2.0, 1.0, "RGB view A/B\n[B,3,128,128]", "#dcedc8")
    enc = add_box(axis, 3.0, 6.7, 2.3, 1.0, "LargeRGBIdentityEncoder\nz [512] + pyramid", "#c5e1a5")
    cond_src = add_box(axis, .4, 4.95, 2.0, 1.0, "SAR XML metadata\nclass + az/dep/band/pol", "#ffe0b2")
    cond = add_box(axis, 3.0, 4.95, 2.3, 1.0, "target condition\n12D sin/cos + one-hot", "#ffcc80")
    real = add_box(axis, .4, 3.2, 2.0, 1.0, "real SAR ROI\n[B,1,64,64]", "#bbdefb")
    gen = add_box(axis, 6.0, 5.75, 2.7, 1.35, "HIFCUnpairedGenerator\n4 alias-free SPADE blocks\nclean + log-speckle → observed", "#b3e5fc", 9)
    disc = add_box(axis, 9.5, 3.65, 2.5, 1.2, "shared conditional\nprojection PatchGAN D", "#ffcdd2")
    teacher = add_box(axis, 9.5, 6.3, 2.5, 1.2, "frozen native\nSARClassifier64 teacher", "#d1c4e9")
    ltc = add_box(axis, 13.0, 5.95, 2.3, .85, "LTC\nlocal moments", "#b2dfdb", 9)
    sfm = add_box(axis, 13.0, 4.75, 2.3, .85, "SFM\nembedding + D moments", "#c8e6c9", 9)
    geom = add_box(axis, 13.0, 3.55, 2.3, .85, "geometry\n4 auxiliary CE", "#d7ccc8", 9)
    rgb_loss = add_box(axis, 6.0, 7.45, 2.7, .8, "L_rgb_identity\n2-view CE + cosine", "#e1bee7", 9)
    d_loss = add_box(axis, 9.5, 1.85, 2.5, .9, "L_D\nh hinge + wrong c/cond + R1", "#ef9a9a", 9)
    g_loss = add_box(axis, 13.0, 1.85, 2.3, .9, "L_G\nweighted sum → update E/G", "#90caf9", 9)

    arrow(axis, rgb, enc); arrow(axis, cond_src, cond); arrow(axis, enc, gen); arrow(axis, cond, gen)
    arrow(axis, real, disc); arrow(axis, real, teacher); arrow(axis, gen, disc); arrow(axis, gen, teacher)
    arrow(axis, gen, ltc); arrow(axis, real, ltc); arrow(axis, gen, sfm); arrow(axis, teacher, sfm)
    arrow(axis, gen, geom); arrow(axis, teacher, geom); arrow(axis, enc, rgb_loss)
    arrow(axis, disc, d_loss); arrow(axis, real, d_loss); arrow(axis, g_loss, gen)
    arrow(axis, ltc, g_loss); arrow(axis, sfm, g_loss); arrow(axis, geom, g_loss)
    arrow(axis, rgb_loss, g_loss); arrow(axis, disc, g_loss)
    axis.text(6.0, .75, "No RGB↔SAR pixel L1, no translation alignment, no real SAR pixels in TSTR classifier training",
              fontsize=10, color="#37474f")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--tstr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = read_history(args.history)
    rows = read_tstr(args.tstr_root)
    plot_training(history, args.output_dir / "training_curves.png")
    plot_tstr(rows, args.output_dir / "tstr_final_results.png",
              args.output_dir / "tstr_summary.json")
    plot_workflow(args.output_dir / "workflow_overview.png")


if __name__ == "__main__":
    main()

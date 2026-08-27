"""Render a compact comparison figure for completed V1 ablation screens."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, filename = value.partition("=")
        if not separator or not label or not filename:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        result[label] = Path(filename)
    return result


def last_history(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty history: {path}")
    return {key: float(value) for key, value in rows[-1].items() if value not in (None, "")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="LABEL=history.csv")
    parser.add_argument("--transfers", nargs="+", required=True, help="LABEL=transfer.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="V1 single-variable ablation")
    args = parser.parse_args()
    histories = {label: last_history(path) for label, path in parse_mapping(args.runs).items()}
    transfers = {label: json.loads(path.read_text(encoding="utf-8")) for label, path in parse_mapping(args.transfers).items()}
    labels = list(histories)
    if set(labels) != set(transfers):
        raise ValueError("run labels and transfer labels must match")
    positions = np.arange(len(labels))
    colors = ["#4c78a8", "#59a14f", "#e15759", "#f28e2b"][:len(labels)]

    figure, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    entries = (
        ("TSTR identity top-1", [transfers[label]["generated_to_real"]["identity_top1"] for label in labels], True),
        ("TSTR depression top-1", [transfers[label]["generated_to_real"]["depression_top1"] for label in labels], True),
        ("TSTR azimuth MAE (deg)", [transfers[label]["generated_to_real"]["azimuth_degree_mae"] for label in labels], False),
        ("Frozen geometry azimuth MAE (deg)", [histories[label]["validation_generated_azimuth_mae"] for label in labels], False),
        ("Frozen feature cosine", [histories[label]["validation_feature_cosine"] for label in labels], True),
        ("+30 deg lowpass response", [histories[label]["validation_response_lowpass_l1_30"] for label in labels], True),
    )
    for axis, (title, values, higher_is_better) in zip(axes.flat, entries):
        bars = axis.bar(positions, values, color=colors)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", alpha=.2)
        if higher_is_better and max(values) <= 1.1:
            axis.set_ylim(0, max(1.0, max(values) * 1.12))
        for bar, value in zip(bars, values):
            text = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text,
                      ha="center", va="bottom", fontsize=9)
    figure.suptitle(args.title, fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

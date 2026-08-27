"""Plot paired generated-to-real transfer deltas from a multi-seed report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("tstr_identity_top1", "TSTR identity top-1 delta", True),
    ("tstr_depression_top1", "TSTR depression top-1 delta", True),
    ("tstr_azimuth_degree_mae", "TSTR azimuth MAE delta (deg)", False),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="V1 paired multi-seed ablation")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    pairs = report["pairs"]
    seeds = [row["seed"] for row in pairs]
    figure, axes = plt.subplots(1, len(METRICS), figsize=(12, 4.1), constrained_layout=True)
    positions = np.arange(len(seeds))
    for axis, (key, title, higher_is_better) in zip(axes, METRICS):
        values = np.array([
            row["transfer_delta_candidate_minus_control"][key] for row in pairs
        ])
        mean = float(values.mean())
        stddev = float(values.std())
        colors = ["#59a14f" if (value >= 0 if higher_is_better else value <= 0) else "#e15759"
                  for value in values]
        axis.axhline(0, color="#555555", linewidth=1)
        axis.scatter(positions, values, c=colors, s=60, zorder=3)
        axis.errorbar(len(seeds), mean, yerr=stddev, color="#4c78a8", marker="D", capsize=5,
                      label="mean +/- std")
        axis.set_title(title, fontsize=10.5)
        axis.set_xticks([*positions, len(seeds)], [*seeds, "mean"])
        axis.grid(axis="y", alpha=.25)
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_xlabel(direction, fontsize=9)
        for position, value in zip(positions, values):
            axis.annotate(f"{value:+.3f}", (position, value), xytext=(0, 6),
                          textcoords="offset points", ha="center", fontsize=8)
    gates = "passed" if report["all_screen_gates_pass"] else "failed"
    transfer = "passed" if report["all_primary_nonregressing"] else "failed"
    figure.suptitle(f"{args.title} | geometry gates: {gates}; transfer non-regression: {transfer}", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

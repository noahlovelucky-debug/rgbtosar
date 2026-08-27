"""Select a deployment checkpoint after the full SAR-speckle curriculum.

Identity is treated as a constraint: only checkpoints whose RGB and generated
SAR accuracies are already high are eligible.  Among them, generation quality
(structure, radiometry and discriminator feature matching) selects the model.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an identity-safe RGB-to-SAR checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-rgb-accuracy", type=float, default=0.98)
    parser.add_argument("--min-fake-accuracy", type=float, default=0.95)
    args = parser.parse_args()

    history_path = args.run_dir / "history.csv"
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty training history: {history_path}")

    milestones = {
        int(path.stem.rsplit("_", 1)[1]): path
        for path in args.run_dir.glob("milestone_*.pt")
    }
    full_speckle = max(float(row["speckle_strength"]) for row in rows)
    candidates: list[tuple[float, int, dict[str, str], Path]] = []
    for row in rows:
        epoch = int(row["epoch"])
        if epoch not in milestones:
            continue
        if float(row["speckle_strength"]) < full_speckle - 1e-6:
            continue
        if float(row["rgb_identity_accuracy"]) < args.min_rgb_accuracy:
            continue
        if float(row["fake_saratrx_accuracy"]) < args.min_fake_accuracy:
            continue
        quality = (
            float(row["loss_structure"])
            + 0.5 * float(row["loss_statistics"])
            + 0.1 * float(row["loss_feature_match"])
        )
        candidates.append((quality, epoch, row, milestones[epoch]))
    if not candidates:
        raise RuntimeError(
            "no full-speckle milestone satisfies the identity thresholds; "
            "inspect history.csv instead of silently selecting a warm-up model"
        )

    quality, epoch, row, source = min(candidates, key=lambda item: (item[0], -item[1]))
    output = args.output or args.run_dir / "selected.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    report = {
        "selected_epoch": epoch,
        "source": str(source),
        "output": str(output),
        "rgb_identity_accuracy": float(row["rgb_identity_accuracy"]),
        "fake_saratrx_accuracy": float(row["fake_saratrx_accuracy"]),
        "speckle_strength": float(row["speckle_strength"]),
        "quality_score": quality,
        "loss_structure": float(row["loss_structure"]),
        "loss_statistics": float(row["loss_statistics"]),
        "loss_feature_match": float(row["loss_feature_match"]),
    }
    report_path = output.with_suffix(".selection.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

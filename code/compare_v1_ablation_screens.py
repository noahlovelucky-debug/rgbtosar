"""Apply fixed regression gates to V1 one-variable ablation screens.

This deliberately does not collapse conditional SAR quality into a scalar.
The frozen geometry validator supplies the comparison metrics; the native SAR
classifier accuracy is reported only as a shortcut diagnostic because V1 uses
that classifier directly in its training objective.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = (
    "validation_generated_identity",
    "validation_generated_depression",
    "validation_generated_azimuth_mae",
    "validation_pair_30_error",
    "validation_feature_cosine",
    "validation_aligned_lowpass_l1",
    "validation_response_lowpass_l1_30",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed-length V1 loss-ablation screens against one control")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def last_validation_row(folder: Path) -> dict[str, float]:
    history = folder / "history.csv"
    if not history.is_file():
        raise FileNotFoundError(history)
    rows: list[dict[str, float]] = []
    with history.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                row = {key: float(value) for key, value in raw.items() if value not in (None, "")}
            except ValueError:
                continue
            if all(math.isfinite(row.get(key, float("nan"))) for key in METRICS):
                rows.append(row)
    if not rows:
        raise RuntimeError(f"{history} has no completed frozen-validator row")
    return rows[-1]


def config(folder: Path) -> dict[str, object]:
    path = folder / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def gate(candidate: dict[str, float], control: dict[str, float]) -> dict[str, dict[str, object]]:
    """Hard regressions stop a branch; passing is not an automatic winner."""
    definitions = (
        ("identity", "validation_generated_identity", ">=", control["validation_generated_identity"] - .02),
        ("depression", "validation_generated_depression", ">=", control["validation_generated_depression"] - .05),
        ("azimuth_mae", "validation_generated_azimuth_mae", "<=", control["validation_generated_azimuth_mae"] + 5.0),
        ("pair_30_error", "validation_pair_30_error", "<=", control["validation_pair_30_error"] + 5.0),
        ("feature_cosine", "validation_feature_cosine", ">=", control["validation_feature_cosine"] - .03),
        ("aligned_lowpass", "validation_aligned_lowpass_l1", "<=", control["validation_aligned_lowpass_l1"] + .03),
        ("angle_response", "validation_response_lowpass_l1_30", ">=",
         .5 * control["validation_response_lowpass_l1_30"]),
    )
    result: dict[str, dict[str, object]] = {}
    for name, key, relation, threshold in definitions:
        value = candidate[key]
        passed = value >= threshold if relation == ">=" else value <= threshold
        result[name] = {
            "metric": key, "value": value, "relation": relation,
            "threshold": threshold, "passed": passed,
        }
    return result


def candidate_report(folder: Path, control: dict[str, float]) -> dict[str, object]:
    row = last_validation_row(folder)
    gates = gate(row, control)
    saved = config(folder)
    return {
        "directory": str(folder.resolve()),
        "run_epoch": int(row.get("run_epoch", -1)),
        "epoch": int(row.get("epoch", -1)),
        "metrics": {name: row[name] for name in METRICS},
        "native_fake_accuracy_diagnostic": row.get("native_fake_accuracy"),
        "changed_configuration": {
            key: saved.get(key) for key in (
                "sar_class_weight", "cluster_weight", "structure_weight",
                "structure_pixel_64_weight", "structure_pixel_32_weight",
                "structure_pixel_16_weight", "structure_edge_weight",
                "structure_ssim_weight", "physics_scatter_weight",
                "angle_loss_mode", "angle_smooth_weight",
                "discriminator_condition", "wrong_azimuth_discriminator_weight",
                "cross_view_weight", "adversarial_weight") if key in saved},
        "gates": gates,
        "screen_pass": all(bool(value["passed"]) for value in gates.values()),
    }


def main() -> None:
    args = arguments()
    control = last_validation_row(args.control)
    candidates = [candidate_report(path, control) for path in args.candidates]
    report = {
        "control": {
            "directory": str(args.control.resolve()),
            "run_epoch": int(control.get("run_epoch", -1)),
            "epoch": int(control.get("epoch", -1)),
            "metrics": {name: control[name] for name in METRICS},
            "native_fake_accuracy_diagnostic": control.get("native_fake_accuracy"),
        },
        "candidates": candidates,
        "policy": {
            "screen": "all gates must pass before a candidate can advance",
            "selection": "among passing candidates, retain only Pareto improvements; do not rank native classifier accuracy",
            "confirmation": "retrain each winner and its parent from the same early V1 checkpoint with at least three seeds",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

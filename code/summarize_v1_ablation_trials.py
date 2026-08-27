"""Aggregate paired multi-seed V1 ablation screens without using native CE accuracy.

Each pair must start from the same checkpoint and use the same seed.  The
result deliberately reports a metric vector rather than inventing one scalar:
we need generated-to-real transfer to improve without a meaningful regression
in the frozen conditional-geometry checks.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from compare_v1_ablation_screens import METRICS, gate


HIGHER_IS_BETTER = {
    "tstr_identity_top1": True,
    "tstr_depression_top1": True,
    "tstr_azimuth_degree_mae": False,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired V1 ablation results across matched seeds")
    parser.add_argument(
        "--pairs", nargs="+", required=True,
        help="SEED=CONTROL_RUN,CANDIDATE_RUN; each run directory needs history.csv and transfer.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def final_history(folder: Path) -> dict[str, float]:
    with (folder / "history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty history: {folder / 'history.csv'}")
    row = rows[-1]
    return {name: float(row[name]) for name in METRICS}


def transfer(folder: Path) -> dict[str, float]:
    path = folder / "transfer.json"
    if not path.is_file():
        # Earlier reports used shortened human names (for example L1a_transfer)
        # instead of a transfer.json within the run directory. Match their
        # embedded checkpoint instead of assuming a filename convention.
        resolved_folder = folder.resolve()
        candidates = []
        for candidate in folder.parent.glob("*_transfer.json"):
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            checkpoint = payload.get("gan_checkpoint")
            if checkpoint and Path(checkpoint).resolve().parent == resolved_folder:
                candidates.append(candidate)
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected {path} or exactly one checkpoint-matched transfer report")
        path = candidates[0]
    result = json.loads(path.read_text(encoding="utf-8"))["generated_to_real"]
    return {
        "tstr_identity_top1": float(result["identity_top1"]),
        "tstr_depression_top1": float(result["depression_top1"]),
        "tstr_azimuth_degree_mae": float(result["azimuth_degree_mae"]),
    }


def parse_pair(value: str) -> tuple[str, Path, Path]:
    seed, equals, paths = value.partition("=")
    control, comma, candidate = paths.partition(",")
    if not equals or not comma or not seed or not control or not candidate:
        raise ValueError(f"expected SEED=CONTROL_RUN,CANDIDATE_RUN, got {value!r}")
    return seed, Path(control), Path(candidate)


def summary(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean": mean(values),
        "population_stddev": pstdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "per_seed": values,
    }


def primary_regression(metric: str, delta: float) -> bool:
    """Reject a useful-looking result that damages another transfer axis."""
    tolerances = {
        "tstr_identity_top1": -0.01,
        "tstr_depression_top1": -0.02,
        "tstr_azimuth_degree_mae": 2.0,
    }
    return delta < tolerances[metric] if HIGHER_IS_BETTER[metric] else delta > tolerances[metric]


def main() -> None:
    args = arguments()
    rows: list[dict[str, object]] = []
    deltas: dict[str, list[float]] = {name: [] for name in HIGHER_IS_BETTER}
    all_screen_gates_pass = True
    all_primary_nonregressing = True
    for raw_pair in args.pairs:
        seed, control_dir, candidate_dir = parse_pair(raw_pair)
        control_history = final_history(control_dir)
        candidate_history = final_history(candidate_dir)
        control_transfer = transfer(control_dir)
        candidate_transfer = transfer(candidate_dir)
        gates = gate(candidate_history, control_history)
        screen_pass = all(bool(value["passed"]) for value in gates.values())
        all_screen_gates_pass = all_screen_gates_pass and screen_pass
        pair_deltas = {
            name: candidate_transfer[name] - control_transfer[name] for name in HIGHER_IS_BETTER
        }
        primary_pass = not any(primary_regression(name, delta) for name, delta in pair_deltas.items())
        all_primary_nonregressing = all_primary_nonregressing and primary_pass
        for name, delta in pair_deltas.items():
            deltas[name].append(delta)
        rows.append({
            "seed": seed,
            "control": str(control_dir.resolve()),
            "candidate": str(candidate_dir.resolve()),
            "screen_pass": screen_pass,
            "screen_gates": gates,
            "transfer_control": control_transfer,
            "transfer_candidate": candidate_transfer,
            "transfer_delta_candidate_minus_control": pair_deltas,
            "transfer_nonregression_pass": primary_pass,
        })
    delta_summary = {name: summary(values) for name, values in deltas.items()}
    # A directionally consistent gain is evidence; any tradeoff remains a
    # finalist to investigate rather than an automatic replacement.
    consistent_improvements = {
        name: all(delta > 0 for delta in values) if HIGHER_IS_BETTER[name]
        else all(delta < 0 for delta in values)
        for name, values in deltas.items()
    }
    report = {
        "pairs": rows,
        "transfer_delta_candidate_minus_control": delta_summary,
        "all_screen_gates_pass": all_screen_gates_pass,
        "all_primary_nonregressing": all_primary_nonregressing,
        "consistent_primary_improvements": consistent_improvements,
        "decision": (
            "advance to a longer confirmation only when all gates and transfer non-regression pass; "
            "choose an automatic replacement only with a consistent primary improvement"
        ),
        "native_classifier": "intentionally excluded from all selection logic",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Offline rescheduling what-if analysis over frozen calibrated predictions.

This script is read-only with respect to the frozen evaluation: it fits,
selects and calibrates nothing. It loads the calibrated prediction CSVs written
by ``flight_delay.evaluation.evaluate`` (the selected calibration variant per
model, discovered from ``selection.json``), applies an explicit operational
cost model, and writes a what-if report, a cost-ratio sensitivity sweep, a
net-benefit curve and a figure per model.

No operational cost has been approved; every quantity is a planning-stage
what-if expressed relative to the delay cost. By default the analysis runs on
the validation (development) partition, consistent with the project's frozen
final-test discipline. Point it at a ``*_test.csv`` only with the deliberate
decision to spend the one-time held-out test, which this script does not make.

Run from the repository root (after a validation run):
    python scripts/rescheduling_analysis.py --eval-dir reports/evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flight_delay.evaluation.predictions import load_prediction_csv
from flight_delay.evaluation.rescheduling import (
    CostModel,
    net_benefit_curve,
    rescheduling_analysis,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def _curve_figure(curve, cost: CostModel, frozen: float | None, title: str, path: Path):
    """Net-benefit curve with the status-quo line and reference thresholds."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve.threshold, curve.net_savings_per_flight, label="Model policy")
    ax.axhline(0.0, color="k", ls="--", lw=1, label="No-action baseline")
    tau = cost.optimal_threshold
    if tau is not None:
        ax.axvline(tau, color="#4285f4", ls=":", label=f"Cost-optimal τ*={tau:.3f}")
    if frozen is not None:
        ax.axvline(frozen, color="#9aa0a6", ls=":", label=f"Frozen F1 τ={frozen:.3f}")
    ax.set(
        xlabel="Action threshold (act if calibrated P(delay) ≥ τ)",
        ylabel="Net savings per flight (units of delay cost)",
        title=title,
        xlim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def analyze_model(
    name: str,
    predictions: Path,
    frozen_threshold: float | None,
    cost: CostModel,
    output: Path,
) -> dict:
    """Run and persist the what-if analysis for one model's calibrated frame."""
    frame = load_prediction_csv(predictions)
    report = rescheduling_analysis(
        frame, cost=cost, frozen_threshold=frozen_threshold
    )
    curve = net_benefit_curve(frame.y_true, frame.y_probability, cost)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / f"{name}_rescheduling.json", report)
    curve.to_csv(output / f"{name}_net_benefit_curve.csv", index=False)
    import pandas as pd

    pd.DataFrame(report["cost_ratio_sweep"]).to_csv(
        output / f"{name}_cost_ratio_sweep.csv", index=False
    )
    _curve_figure(
        curve,
        cost,
        frozen_threshold,
        f"{name}: rescheduling net benefit (ratio {cost.cost_ratio:g}, recovery {cost.recovery:g})",
        output / f"{name}_net_benefit.png",
    )
    return report


def run(eval_dir: Path, cost: CostModel, predictions: Path | None) -> dict:
    """Discover selected calibrated frames from selection.json, or use one CSV."""
    output = eval_dir / "rescheduling"
    if predictions is not None:
        return {"custom": analyze_model("custom", predictions, None, cost, output)}
    manifest = json.loads((eval_dir / "selection.json").read_text())
    reports = {}
    for name, choice in manifest["models"].items():
        csv = eval_dir / f"{name}_{choice['calibration']}_validation.csv"
        reports[name] = analyze_model(
            name, csv, choice["threshold"], cost, output
        )
    _write_json(
        output / "rescheduling_summary.json",
        {
            "assumptions": next(iter(reports.values()))["assumptions"],
            "partition": "validation",
            "models": {
                n: {
                    "population": r["population"],
                    "baselines_per_flight": r["baselines_per_flight"],
                    "policy_at_optimal_threshold": r["policy_at_optimal_threshold"],
                    "policy_at_frozen_threshold": r.get(
                        "policy_at_frozen_threshold"
                    ),
                }
                for n, r in reports.items()
            },
        },
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Analyse a single prediction CSV instead of the selected frames.",
    )
    parser.add_argument("--cost-delay", type=float, default=1.0)
    parser.add_argument(
        "--cost-ratio",
        type=float,
        default=0.25,
        help="Assumed action cost as a fraction of delay cost (C_a / C_d).",
    )
    parser.add_argument("--recovery", type=float, default=0.8)
    args = parser.parse_args()
    cost = CostModel(
        cost_delay=args.cost_delay,
        cost_action=args.cost_ratio * args.cost_delay,
        recovery=args.recovery,
    )
    reports = run(args.eval_dir, cost, args.predictions)
    print("Offline rescheduling what-if (NOT an approved operational saving):")
    print(json.dumps(cost.__dict__ | {"optimal_threshold": cost.optimal_threshold}, indent=2))
    for name, report in reports.items():
        policy = report["policy_at_optimal_threshold"]
        if policy is None:
            print(f"\n{name}: acting is never cost-justified at this ratio/recovery.")
            continue
        print(
            f"\n{name} @ cost-optimal τ*={policy['threshold']:.3f}: "
            f"acts on {policy['predicted_positive_rate']:.1%} of flights, "
            f"net {policy['net_savings_per_flight']:+.4f}/flight "
            f"({policy['savings_fraction_of_status_quo']:+.1%} vs status quo), "
            f"oracle efficiency "
            + (
                f"{policy['oracle_efficiency']:.1%}"
                if policy["oracle_efficiency"] is not None
                else "n/a"
            )
        )
    print(f"\nArtifacts written under {args.eval_dir / 'rescheduling'}")


if __name__ == "__main__":
    main()

"""Phase 2 validation-only AML/DL comparison with guarded, aligned predictions.

Usage from the repository root:
    python scripts/compare_aml_dl.py --bootstrap 100

Consumes committed prediction CSVs and only train_calibration targets for MLP
calibration. This script NEVER loads, evaluates or selects from the test set.
The output directory is deliberately separate from frozen Phase 1 reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
PROBABILITY_COLUMNS = ("flight_id", "flight_date", "y_true", "y_probability", "partition")
MODEL_NAMES = {
    "logistic_raw": "Logistic Regression (raw)",
    "logistic_isotonic": "Logistic Regression (isotonic)",
    "xgboost_raw": "XGBoost (raw)",
    "xgboost_isotonic": "XGBoost (isotonic)",
    "mlp_raw": "Embedding MLP (raw)",
    "mlp_sigmoid": "Embedding MLP (sigmoid)",
    "mlp_isotonic": "Embedding MLP (isotonic)",
    "hybrid": "XGBoost + MLP stack",
}


@dataclass(frozen=True)
class Sources:
    """Existing validation artifacts; no test path is accepted by design."""

    logistic_raw: Path
    logistic_isotonic: Path
    xgboost_raw: Path
    xgboost_isotonic: Path
    mlp_raw: Path
    hybrid: Path
    mlp_train_calibration: Path

    @classmethod
    def from_root(cls, root: Path) -> Sources:
        aml = root / "reports/evaluation_v1_noCongestion"
        dl = root / "reports/dl"
        return cls(
            aml / "logistic_balanced_raw_validation.csv",
            aml / "logistic_balanced_isotonic_validation.csv",
            aml / "xgboost_tuned_full_raw_validation.csv",
            aml / "xgboost_tuned_full_isotonic_validation.csv",
            dl / "validation_predictions.csv",
            dl / "hybrid_validation_predictions.csv",
            dl / "train_calibration_predictions.csv",
        )


def load_predictions(path: Path, partition: str) -> pd.DataFrame:
    """Read losslessly: IDs remain strings; thresholds are never rounded."""
    frame = pd.read_csv(path, dtype={"flight_id": "string"}, float_precision="round_trip")
    missing = set(PROBABILITY_COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"{path}: missing prediction columns {sorted(missing)} or empty")
    frame = frame.loc[:, list(PROBABILITY_COLUMNS)].copy()
    if frame.isna().any().any():
        raise ValueError(f"{path}: missing prediction values")
    if frame.flight_id.str.strip().eq("").any() or frame.flight_id.duplicated().any():
        raise ValueError(f"{path}: blank or duplicate flight ID")
    if not frame.partition.eq(partition).all():
        raise ValueError(f"{path}: expected partition {partition!r}")
    if not frame.y_true.isin([0, 1]).all():
        raise ValueError(f"{path}: expected binary targets")
    dates = pd.to_datetime(frame.flight_date, format="%Y-%m-%d", errors="raise")
    if dates.isna().any():
        raise ValueError(f"{path}: missing dates")
    p = pd.to_numeric(frame.y_probability, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{path}: invalid probabilities")
    frame["y_true"] = frame.y_true.astype(np.int64)
    frame["y_probability"] = p
    return frame.sort_values("flight_id").reset_index(drop=True)


def verify_alignment(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    reference = next(iter(frames.values()))
    keys = list(PROBABILITY_COLUMNS[:3])
    for name, frame in frames.items():
        if not frame.loc[:, keys].equals(reference.loc[:, keys]):
            raise ValueError(f"Unaligned flight ID/date/label set for {name}")
    return reference


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


def mlp_calibrations(cal: pd.DataFrame, val: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit sigmoid/isotonic on train_calibration labels, not on validation."""
    if pd.to_datetime(cal.flight_date).max() >= pd.to_datetime(val.flight_date).min():
        raise ValueError("Calibration dates must strictly precede validation")
    if set(cal.flight_id) & set(val.flight_id):
        raise ValueError("Calibration/validation flight IDs overlap")
    y = cal.y_true.to_numpy(dtype=int)
    p = cal.y_probability.to_numpy(dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("MLP calibration requires both binary target classes")
    out = {"mlp_raw": val.y_probability.to_numpy(dtype=float)}
    logistic = LogisticRegression(C=1e10, max_iter=1000, random_state=42)
    logistic.fit(_logit(p), y)
    out["mlp_sigmoid"] = logistic.predict_proba(_logit(out["mlp_raw"]))[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    out["mlp_isotonic"] = isotonic.predict(out["mlp_raw"])
    return out


def threshold_at_min_precision(y: np.ndarray, p: np.ndarray, target: float) -> dict:
    """Highest recall among nonempty rules achieving the specified precision."""
    if not 0 < target <= 1:
        raise ValueError("Minimum precision must be in (0, 1]")
    precision, recall, thresholds = precision_recall_curve(y, p)
    indices = np.flatnonzero(precision[:-1] >= target)
    if len(indices) == 0:
        return {"attainable": False, "threshold": None, "precision": None,
                "recall": None, "flagged": 0}
    i = max(indices, key=lambda i: (recall[i], precision[i], thresholds[i]))
    predicted = p >= thresholds[i]
    return {
        "attainable": True,
        "threshold": float(thresholds[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "flagged": int(predicted.sum()),
    }


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Deterministic equal-width reliability bins, including p=1."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return (pd.DataFrame({"bin": bins, "probability": p, "observed": y})
            .groupby("bin", as_index=False)
            .agg(mean_prediction=("probability", "mean"),
                 observed_delay_rate=("observed", "mean"),
                 count=("observed", "size")))


def metrics(y: np.ndarray, p: np.ndarray, targets: tuple[float, ...]) -> dict:
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Validation must have two classes")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Prediction probabilities must be finite in [0,1]")
    bins = reliability(y, p)
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_predicted_probability": float(p.mean()),
        "calibration_bias": float(p.mean() - y.mean()),
        "ece_10_equal_width_bins": float(
            (bins["count"] * (bins.mean_prediction - bins.observed_delay_rate).abs())
            .sum() / len(y)
        ),
        "recall_at_min_precision": {
            f"{value:.2f}": threshold_at_min_precision(y, p, value)
            for value in targets
        },
    }


def bootstrap_ap_difference(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, dates: np.ndarray,
    *, n_boot: int, seed: int = 42,
) -> dict:
    """Paired resampling of calendar days; exploratory, not independent test CI."""
    if n_boot < 0:
        raise ValueError("Bootstrap count cannot be negative")
    if n_boot == 0:
        return {"n_boot": 0, "interval": None}
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(dates == d) for d in np.unique(dates)]
    deltas = []
    for _ in range(n_boot):
        index = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[index])) != 2:
            continue
        deltas.append(float(average_precision_score(y[index], b[index])
                            - average_precision_score(y[index], a[index])))
    if not deltas:
        raise ValueError("Bootstrap produced no valid binary-class samples")
    return {"n_boot": n_boot, "valid": len(deltas),
            "interval": [float(z) for z in np.quantile(deltas, [0.025, 0.975])],
            "difference": float(average_precision_score(y, b)
                                - average_precision_score(y, a))}


def create_plots(out: Path, y: np.ndarray, predictions: dict[str, np.ndarray],
                 selected: str, summary: dict) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    display = ["logistic_raw", "xgboost_raw", "mlp_raw", "hybrid"]
    if selected not in display:
        display.append(selected)
    for curve in ("roc", "precision_recall"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in display:
            if curve == "roc":
                x, z, _ = roc_curve(y, predictions[name])
            else:
                z, x, _ = precision_recall_curve(y, predictions[name])
            ax.plot(x, z, label=MODEL_NAMES[name])
        if curve == "roc":
            ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
            ax.set(xlabel="False positive rate", ylabel="True positive rate",
                   title="Validation ROC curves")
        else:
            ax.axhline(float(y.mean()), linestyle="--", label="Delay prevalence")
            ax.set(xlabel="Recall", ylabel="Precision",
                   title="Validation precision–recall curves")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"{curve}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    for name in ["logistic_isotonic", "xgboost_isotonic", selected, "hybrid"]:
        table = reliability(y, predictions[name])
        ax.plot(table.mean_prediction, table.observed_delay_rate,
                marker="o", label=MODEL_NAMES[name])
    ax.set(xlabel="Mean predicted delay probability", ylabel="Observed delayed fraction",
           xlim=(0, 1), ylim=(0, 1), title="Validation reliability (10 bins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "reliability.png", dpi=150)
    plt.close(fig)
    for key, title, filename in (
        ("average_precision", "Validation Average Precision (higher is better)", "ap_comparison.png"),
        ("brier", "Validation Brier score (lower is better)", "brier_comparison.png"),
    ):
        names = list(summary["models"])
        values = [summary["models"][name][key] for name in names]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([MODEL_NAMES[name] for name in names], values)
        ax.invert_yaxis()
        ax.set(xlabel=key.replace("_", " "), title=title)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)
    for name in ("logistic_isotonic", "xgboost_isotonic", selected, "hybrid"):
        # Fixed 0.5 threshold is illustrative, NOT a tuned operating policy.
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y, predictions[name] >= 0.5,
                                                labels=[0, 1]),
                               display_labels=["Not delayed", "Delayed"]).plot(
                                   ax=ax, colorbar=False)
        ax.set_title(f"{MODEL_NAMES[name]} @ 0.5 (illustrative)")
        fig.tight_layout()
        fig.savefig(figures / f"confusion_{name}.png", dpi=150)
        plt.close(fig)


def sha256_file(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def compare(sources: Sources, output: Path, *,
            precision_targets: tuple[float, ...] = (0.20, 0.25, 0.30),
            bootstrap: int = 0, make_plots: bool = True) -> dict:
    """Pure validation-stage comparison, retaining immutable AML outputs."""
    if not precision_targets or any(not 0 < v <= 1 for v in precision_targets):
        raise ValueError("Provide valid precision thresholds in (0,1]")
    if len(set(precision_targets)) != len(precision_targets):
        raise ValueError("Duplicate precision targets")
    paths = {
        "logistic_raw": sources.logistic_raw,
        "logistic_isotonic": sources.logistic_isotonic,
        "xgboost_raw": sources.xgboost_raw,
        "xgboost_isotonic": sources.xgboost_isotonic,
        "mlp_raw": sources.mlp_raw,
        "hybrid": sources.hybrid,
    }
    frames = {k: load_predictions(p, "validation") for k, p in paths.items()}
    baseline = verify_alignment(frames)
    calibration = load_predictions(sources.mlp_train_calibration, "train_calibration")
    calibrated = mlp_calibrations(calibration, frames["mlp_raw"])
    for name in ("mlp_sigmoid", "mlp_isotonic"):
        frames[name] = frames["mlp_raw"].assign(y_probability=calibrated[name])
    verify_alignment(frames)
    y = baseline.y_true.to_numpy(dtype=int)
    probabilities = {k: v.y_probability.to_numpy(dtype=float) for k, v in frames.items()}
    stats = {k: metrics(y, v, precision_targets) for k, v in probabilities.items()}
    selected = min(("mlp_raw", "mlp_sigmoid", "mlp_isotonic"),
                   key=lambda k: (stats[k]["brier"], stats[k]["log_loss"]))
    cal_prevalence = float(calibration.y_true.mean())
    report = {
        "stage": "validation_only",
        "target": "arrival delay >= 15 minutes",
        "selection_rule": "MLP calibration selected on validation Brier, with log loss tie-break",
        "test_used": False,
        "n": int(len(y)),
        "validation_prevalence": float(y.mean()),
        "train_calibration_n": int(len(calibration)),
        "train_calibration_prevalence": cal_prevalence,
        "calibration_window_constant_brier": float(brier_score_loss(
            y, np.full(len(y), cal_prevalence))),
        "selected_mlp_calibration": selected,
        "precision_targets": list(precision_targets),
        "models": stats,
        "artifact_sha256": {key: sha256_file(path) for key, path in
                            {**paths, "mlp_train_calibration": sources.mlp_train_calibration}.items()},
        "paired_ap_bootstrap": {
            "mlp_raw_minus_logistic_raw": bootstrap_ap_difference(
                y, probabilities["logistic_raw"], probabilities["mlp_raw"],
                baseline.flight_date.to_numpy(), n_boot=bootstrap),
            "mlp_selected_minus_xgboost_isotonic": bootstrap_ap_difference(
                y, probabilities["xgboost_isotonic"], probabilities[selected],
                baseline.flight_date.to_numpy(), n_boot=bootstrap),
        },
        "limitations": [
            "Validation-only results; calibration and threshold selection here are not independent final-test claims",
            "Historical previous-date outcomes have unresolved exact real-time availability",
            "Precision-target recall is marked unattainable instead of fabricating a threshold",
            "Paired date bootstrap treats days as exchangeable and is descriptive, not a serial-block CI",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, m in stats.items():
        rows.append({"model": name, "average_precision": m["average_precision"],
                     "roc_auc": m["roc_auc"], "brier": m["brier"],
                     "log_loss": m["log_loss"], "ece_10_bins": m["ece_10_equal_width_bins"],
                     **{f"recall_at_precision_{target:.2f}":
                        m["recall_at_min_precision"][f"{target:.2f}"]["recall"]
                        for target in precision_targets}})
        reliability(y, probabilities[name]).to_csv(
            output / f"reliability_{name}.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2,
                                                  allow_nan=False) + "\n", encoding="utf-8")
    if make_plots:
        create_plots(output, y, probabilities, selected, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    parser.add_argument("--output", type=Path, default=None,
                        help="Phase 2 evaluation output directory "
                             "(default: reports/dl, figures in reports/dl/figures)")
    parser.add_argument("--precision-targets", type=float, nargs="+",
                        default=[0.20, 0.25, 0.30])
    parser.add_argument("--bootstrap", type=int, default=100,
                        help="Paired date bootstrap repetitions; 0 disables")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or (root / "reports/dl")
    report = compare(Sources.from_root(root), output,
                     precision_targets=tuple(args.precision_targets),
                     bootstrap=args.bootstrap, make_plots=not args.no_plots)
    print(json.dumps({"validation_rows": report["n"],
                      "selected_mlp_calibration": report["selected_mlp_calibration"],
                      "model_metrics": {name: {
                          "AP": d["average_precision"], "ROC_AUC": d["roc_auc"],
                          "Brier": d["brier"]} for name, d in report["models"].items()},
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

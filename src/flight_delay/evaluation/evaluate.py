"""Explicit validation/final-test stages using frozen training configurations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from numpy.typing import ArrayLike

from flight_delay.data.pipeline import build_data_pipeline
from flight_delay.features.model_features import select_model_features
from flight_delay.features.selection import split_features_and_target
from flight_delay.models.logistic import fit_logistic_baseline
from flight_delay.models.xgboost_model import fit_xgboost_baseline

from .calibration import ScoreCalibrator, check_windows
from .metrics import align_predictions, evaluate, paired_bootstrap
from .plots import render, summary_plots
from .thresholds import select_thresholds, threshold_table

# Existing selected parameters, unchanged from scripts/compare_models.py.
XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.03,
    "min_child_weight": 10,
    "subsample": 0.90,
    "colsample_bytree": 0.75,
    "reg_lambda": 10.0,
    "scale_pos_weight": 2.859,
}


def dates(frame: pd.DataFrame) -> pd.Series:
    """Reconstruct calendar dates from the upstream schema."""
    return pd.to_datetime(frame[["year", "month", "day"]])


def fingerprint(path: Path) -> str:
    """Hash input bytes for reproducible identity checks."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write strict machine-readable JSON."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def prediction_frame(
    frame: pd.DataFrame, probabilities: ArrayLike, partition: str
) -> pd.DataFrame:
    """Keep identifiers and provenance attached to every prediction."""
    return pd.DataFrame(
        {
            "flight_id": frame.id.to_numpy(),
            "flight_date": dates(frame).dt.strftime("%Y-%m-%d").to_numpy(),
            "y_true": frame.is_delayed.to_numpy(),
            "y_probability": probabilities,
            "partition": partition,
        }
    )


def run_validation(
    data: Path, output: Path, n_jobs: int = 4, n_boot: int = 300
) -> dict:
    """Refit early-training bases, calibrate on late training, freeze validation choices."""
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise ValueError("Frozen selection exists; use a new output directory")
    train, val, _ = build_data_pipeline(data)
    days = sorted(dates(train).unique())
    cutoff = days[int(len(days) * 0.8)]
    base = train.loc[dates(train) < cutoff]
    cal = train.loc[dates(train) >= cutoff]
    check_windows(dates(base), dates(cal), dates(val))
    combined_ids = pd.concat([base.id, cal.id, val.id])
    if combined_ids.isna().any() or combined_ids.duplicated().any():
        raise ValueError(
            "Training/calibration/validation IDs must be unique and disjoint"
        )
    X, y = split_features_and_target(base)
    models = {
        "logistic_balanced": fit_logistic_baseline(X, y, class_weight="balanced"),
        "xgboost_tuned_full": fit_xgboost_baseline(
            X, y, n_jobs=n_jobs, model_params=XGB_PARAMS
        ),
    }
    selected = {}
    all_metrics = {}
    all_variants = {}
    tables = {}
    frozen = {}
    frames = {}
    for name, model in models.items():
        cp = model.predict_proba(select_model_features(cal))[:, 1]
        vp = model.predict_proba(select_model_features(val))[:, 1]
        variants = {}
        calibrators = {}
        for method in ["raw", "sigmoid", "isotonic"]:
            calibrator = ScoreCalibrator(method).fit(
                cal.is_delayed, cp, partition="train_calibration"
            )
            frame = prediction_frame(val, calibrator.predict(vp), "validation")
            variants[method] = frame
            calibrators[method] = calibrator
            all_metrics[f"{name}_{method}"] = evaluate(
                frame.y_true, frame.y_probability
            )
        method = min(
            variants,
            key=lambda k: (
                all_metrics[f"{name}_{k}"]["brier"],
                all_metrics[f"{name}_{k}"]["log_loss"],
            ),
        )
        frame = variants[method]
        table = threshold_table(frame.y_true, frame.y_probability)
        choices = select_thresholds(table)
        threshold = choices["max_f1"]["threshold"]
        selected[name] = {
            "calibration": method,
            "threshold": threshold,
            "strategies": choices,
        }
        frozen[name] = {
            "model": model,
            "calibrator": calibrators[method],
            "threshold": threshold,
        }
        frames[name] = frame
        tables[name] = table
        all_variants[name] = variants
        table.to_csv(output / f"{name}_thresholds.csv", index=False)
        for variant, f in variants.items():
            f.to_csv(output / f"{name}_{variant}_validation.csv", index=False)
    frames = align_predictions(frames, "validation")
    metrics = {
        name: evaluate(f.y_true, f.y_probability, selected[name]["threshold"])
        for name, f in frames.items()
    }
    names = list(frames)
    a, b = [frames[n] for n in names]
    uncertainty = paired_bootstrap(
        a.y_true,
        a.y_probability,
        b.y_probability,
        a.flight_date,
        thresholds=tuple(selected[n]["threshold"] for n in names),
        n_boot=n_boot,
    )
    manifest = {
        "schema_version": 1,
        "data_sha256": fingerprint(data),
        "seed": 42,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "xgboost"]
        },
        "windows": {
            k: {
                "n": len(f),
                "first": str(dates(f).min().date()),
                "last": str(dates(f).max().date()),
            }
            for k, f in [
                ("base_train", base),
                ("train_calibration", cal),
                ("validation", val),
            ]
        },
        "feature_columns": list(select_model_features(base).columns),
        "xgboost_parameters": XGB_PARAMS,
        "calibration_rule": "minimum validation Brier, log loss tie-break",
        "threshold_rule": "maximum validation F1; higher threshold tie-break",
        "precision_constraints": "illustrative sensitivity only; no operational constraint approved",
        "models": selected,
        "winner": max(metrics, key=lambda n: metrics[n]["average_precision"]),
    }
    artifact = output / "frozen.joblib"
    joblib.dump(frozen, artifact)
    manifest["artifact_sha256"] = fingerprint(artifact)
    write_json(output / "validation_metrics.json", metrics)
    write_json(output / "calibration_metrics.json", all_metrics)
    write_json(output / "validation_bootstrap.json", uncertainty)
    render(
        frames,
        output / "figures" / "validation",
        thresholds=tables,
        variants=all_variants,
    )
    summary_plots(all_metrics, output / "figures" / "calibration")
    summary_plots(metrics, output / "figures" / "validation")
    write_json(output / "selection.json", manifest)
    return manifest


def run_final_test(data: Path, output: Path) -> dict:
    """Only apply frozen estimators/calibrators/thresholds; never refit or select."""
    manifest = json.loads((output / "selection.json").read_text())
    if (output / "test_metrics.json").exists():
        raise ValueError("Final test already evaluated; refusing repeat")
    if (
        fingerprint(data) != manifest["data_sha256"]
        or fingerprint(output / "frozen.joblib") != manifest["artifact_sha256"]
    ):
        raise ValueError("Dataset or frozen artifact identity mismatch")
    frozen = joblib.load(output / "frozen.joblib")
    _, _, test = build_data_pipeline(data)
    if dates(test).min() <= pd.Timestamp(manifest["windows"]["validation"]["last"]):
        raise ValueError("Test dates overlap validation")
    frames = {}
    for name, item in frozen.items():
        if item["threshold"] != manifest["models"][name]["threshold"]:
            raise ValueError("Frozen threshold mismatch")
        p = item["calibrator"].predict(
            item["model"].predict_proba(select_model_features(test))[:, 1]
        )
        frames[name] = prediction_frame(test, p, "test")
    frames = align_predictions(frames, "test")
    metrics = {
        n: evaluate(f.y_true, f.y_probability, frozen[n]["threshold"])
        for n, f in frames.items()
    }
    for n, f in frames.items():
        f.to_csv(output / f"{n}_test.csv", index=False)
    render(frames, output / "figures" / "test")
    summary_plots(metrics, output / "figures" / "test")
    write_json(output / "test_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validation", "final-test"])
    parser.add_argument("--data", type=Path, default=Path("data/raw/flights.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    result = (
        run_validation(args.data, args.output_dir, args.n_jobs, args.bootstrap)
        if args.stage == "validation"
        else run_final_test(args.data, args.output_dir)
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

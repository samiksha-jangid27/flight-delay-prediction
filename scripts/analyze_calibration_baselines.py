"""Compare fixed training-prevalence baselines on saved validation predictions.

Run from root with ``python -m scripts.analyze_calibration_baselines``. No test
metrics are read, no estimator is refitted, and a new exclusive JSON file records
this supplementary analysis without modifying the frozen experiment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import uuid
from pathlib import Path

import numpy as np
from sklearn import metrics

from flight_delay.evaluation.baselines import fit_prevalence
from scripts.audit_evaluation import (
    load_json,
    read_predictions,
    rebuild_partitions,
    sha256,
)


def analyze(data: Path, output_dir: Path, *, diagnose_logistic: bool = False) -> dict:
    """Use pre-validation labels for constants and validation labels only for scoring."""
    manifest_path = output_dir / "selection.json"
    manifest = load_json(manifest_path)
    data_hash = sha256(data)
    if data_hash != manifest["data_sha256"]:
        raise ValueError("Dataset SHA-256 does not match frozen selection")
    partitions = rebuild_partitions(data)
    validation = partitions["validation"].assign(partition="validation")
    constants = {}
    for name in ("base_train", "train_calibration"):
        train = partitions[name].assign(partition=name)
        observed_window = {
            "n": len(train),
            "first": train.flight_date.min(),
            "last": train.flight_date.max(),
        }
        if manifest["windows"][name] != observed_window:
            raise ValueError(f"Training window mismatch: {name}")
        baseline = fit_prevalence(train, partition=name)
        constants[name] = {
            "fit": baseline.metadata(),
            "validation": baseline.score_validation(validation),
        }
    models = {}
    input_hashes = {"selection.json": sha256(manifest_path)}
    for name, choice in manifest["models"].items():
        variants = {}
        for method in ("raw", "sigmoid", "isotonic"):
            path = output_dir / f"{name}_{method}_validation.csv"
            frame = read_predictions(path, partitions["validation"], "validation")
            input_hashes[path.name] = sha256(path)
            y, p = frame.y_true.to_numpy(), frame.y_probability.to_numpy()
            positive = p[y == 1]
            negative = p[y == 0]
            # Shared positive/negative scores expose cross-class isotonic ties.
            shared = np.intersect1d(positive, negative)
            cross_class_tied_pairs = sum(
                int((positive == value).sum()) * int((negative == value).sum())
                for value in shared
            )
            variants[method] = {
                "n": len(y),
                "brier": float(metrics.brier_score_loss(y, p)),
                "log_loss": float(metrics.log_loss(y, p, labels=[0, 1])),
                "average_precision": float(metrics.average_precision_score(y, p)),
                "roc_auc": float(metrics.roc_auc_score(y, p)),
                "mean_probability": float(p.mean()),
                "observed_prevalence": float(y.mean()),
                "calibration_bias": float(p.mean() - y.mean()),
                "unique_scores": len(np.unique(p)),
                "cross_class_tied_pairs": cross_class_tied_pairs,
            }
        selected = variants[choice["calibration"]]
        models[name] = {
            "frozen_calibration": choice["calibration"],
            "variants": variants,
            "selected_minus_raw": {
                key: selected[key] - variants["raw"][key]
                for key in ("brier", "log_loss", "average_precision", "roc_auc")
            },
            "selected_minus_constants": {
                baseline: {
                    key: selected[key] - value["validation"][key]
                    for key in ("brier", "log_loss", "average_precision")
                }
                for baseline, value in constants.items()
            },
        }
    result = {
        "data_sha256": data_hash,
        "analysis_source_sha256": sha256(Path(__file__)),
        "input_hashes": input_hashes,
        "constants": constants,
        "models": models,
        "policy": "Supplementary validation-only analysis; no selection or frozen-experiment changes; lower Brier/log loss is better.",
        "limitations": [
            "Predicted-prevalence shift can reflect seasonal/class-prior change and feature extrapolation; this analysis does not establish causal attribution.",
            "The validation-prevalence oracle is descriptive only; neither constant here is fitted on validation or test labels.",
        ],
    }
    result["package_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "xgboost")
    }
    if diagnose_logistic:
        input_hashes["frozen.joblib"] = sha256(output_dir / "frozen.joblib")
        result["logistic_shift"] = diagnose_logistic_shift(data, output_dir)
    for filename, digest in input_hashes.items():
        if sha256(output_dir / filename) != digest:
            raise ValueError(f"Input changed during baseline analysis: {filename}")
    path = output_dir / f"calibration_baselines_{uuid.uuid4().hex}.json"
    result["report_path"] = str(path.resolve())
    with path.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def diagnose_logistic_shift(data: Path, output_dir: Path) -> dict:
    """Decompose existing LR link scores on training/calibration/validation only.

    Transform/predict calls inspect the frozen estimator; nothing is fitted or
    selected. Mean link contributions explain score movement algebraically,
    without asserting causality or equating mean logit with logit(mean score).
    """
    import joblib
    import pandas as pd

    from flight_delay.data.pipeline import build_data_pipeline
    from flight_delay.features.model_features import select_model_features

    manifest = load_json(output_dir / "selection.json")
    if (
        sha256(data) != manifest["data_sha256"]
        or sha256(output_dir / "frozen.joblib") != manifest["artifact_sha256"]
    ):
        raise ValueError("Frozen diagnostic input hash mismatch")
    frozen = joblib.load(output_dir / "frozen.joblib")["logistic_balanced"]
    model = frozen["model"]
    train, validation, _test_not_scored = build_data_pipeline(data)
    dates = pd.to_datetime(train[["year", "month", "day"]])
    cutoff = pd.Timestamp(manifest["windows"]["train_calibration"]["first"])
    windows = {
        "base_train": train.loc[dates < cutoff],
        "train_calibration": train.loc[dates >= cutoff],
        "validation": validation,
    }
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    counts = np.array([name.endswith("_historical_flight_count") for name in names])
    rates = np.array([name.endswith("_historical_delay_rate") for name in names])
    result, contributions = {}, {}
    for name, frame in windows.items():
        features = select_model_features(frame)
        transformed = model.named_steps["preprocess"].transform(features)
        contribution = np.asarray(transformed.mean(axis=0)).ravel() * coefficients
        contributions[name] = contribution
        scores = model.predict_proba(features)[:, 1]
        calibrated = frozen["calibrator"].predict(scores)
        intercept = float(model.named_steps["model"].intercept_[0])
        result[name] = {
            "n": len(frame),
            "prevalence": float(frame.is_delayed.mean()),
            "mean_raw_probability": float(scores.mean()),
            "mean_selected_probability": float(calibrated.mean()),
            "historical_counts_logit_contribution": float(contribution[counts].sum()),
            "historical_rates_logit_contribution": float(contribution[rates].sum()),
            "other_features_logit_contribution": float(
                contribution[~(counts | rates)].sum()
            ),
            "intercept": intercept,
            "mean_logit": float(contribution.sum() + intercept),
        }
    shift = contributions["validation"] - contributions["train_calibration"]
    order = np.argsort(np.abs(shift))[::-1][:10]
    result["largest_validation_minus_calibration_link_contributions"] = {
        str(names[i]): float(shift[i]) for i in order
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/flights.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument(
        "--diagnose-logistic",
        action="store_true",
        help="Inspect frozen LR link contributions without fitting or test scoring",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.data, args.output_dir, diagnose_logistic=args.diagnose_logistic
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

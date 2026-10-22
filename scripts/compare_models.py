"""Model-to-model comparison on one shared chronological validation split.

This script is owned by the model authors. It does *not* run calibration,
threshold selection, rescheduling, or held-out test evaluation. All models
are refitted on the identical training rows for a reproducible comparison.

Run from the repository root:
    python scripts/compare_models.py --n-jobs 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from flight_delay.data.pipeline import build_data_pipeline
from flight_delay.features.model_features import select_model_features
from flight_delay.features.selection import split_features_and_target
from flight_delay.models.logistic import fit_logistic_baseline
from flight_delay.models.xgboost_model import fit_xgboost_baseline

# Parameters previously selected using training-only chronological CV and
# reported in docs/xgboost_results.md. They are NOT reoptimized on validation.
SELECTED_XGBOOST_PARAMS = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.03,
    "min_child_weight": 10,
    "subsample": 0.90,
    "colsample_bytree": 0.75,
    "reg_lambda": 10.0,
    "scale_pos_weight": 2.859,
}


def compare_model_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    n_jobs: int = 4,
    xgb_params: dict | None = None,
) -> pd.DataFrame:
    """Fit two LR configurations and the frozen tuned XGB on identical rows.

    Only validation AP and ROC-AUC are compared. The upstream builder
    constructs the test partition, but it is not fitted, scored or inspected
    by this comparison; no calibration is attempted.
    """
    X_train, y_train = split_features_and_target(train)
    X_validation, y_validation = split_features_and_target(validation)
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("Train/validation partitions must both be nonempty")
    if "id" not in validation.columns or validation["id"].duplicated().any():
        raise ValueError("Validation must contain unique flight IDs")
    if set(y_validation.dropna().unique()) != {0, 1}:
        raise ValueError("Both classes are required in validation for ROC-AUC")
    if pd.to_datetime(train[["year", "month", "day"]]).max() >= pd.to_datetime(
        validation[["year", "month", "day"]]
    ).min():
        raise ValueError("Training dates must strictly precede validation dates")

    X_val_full = select_model_features(X_validation, "full")
    models = {
        "logistic_unweighted": fit_logistic_baseline(
            X_train, y_train, feature_set="full", C=1.0,
        ),
        "logistic_balanced": fit_logistic_baseline(
            X_train, y_train, feature_set="full", C=1.0,
            class_weight="balanced",
        ),
        "xgboost_tuned_full": fit_xgboost_baseline(
            X_train, y_train, feature_set="full", n_jobs=n_jobs,
            model_params=SELECTED_XGBOOST_PARAMS if xgb_params is None else xgb_params,
        ),
    }
    results = []
    for name, model in models.items():
        probabilities = model.predict_proba(X_val_full)[:, 1]
        ap = float(average_precision_score(y_validation, probabilities))
        auc = float(roc_auc_score(y_validation, probabilities))
        results.append({
            "model": name,
            "training_flights": len(y_train),
            "validation_flights": len(y_validation),
            "validation_prevalence": float(y_validation.mean()),
            "average_precision": ap,
            "roc_auc": auc,
            "ap_lift_over_no_skill": ap / float(y_validation.mean()),
        })
    return pd.DataFrame(results).sort_values(
        "average_precision", ascending=False
    ).reset_index(drop=True)


def run_model_comparison(
    data_path: str | Path = "data/raw/flights.csv",
    output_path: str | Path = "reports/results/model_comparison_validation.csv",
    *, n_jobs: int = 4,
) -> pd.DataFrame:
    """Build the input dataset once so all candidates use identical flight rows."""
    train, validation, _test_untouched = build_data_pipeline(data_path)
    results = compare_model_candidates(train, validation, n_jobs=n_jobs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/flights.csv")
    parser.add_argument(
        "--output", default="reports/results/model_comparison_validation.csv"
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    result = run_model_comparison(args.data, args.output, n_jobs=args.n_jobs)
    print("Model comparison (chronological validation only; NOT final test):")
    print(result.to_string(index=False))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

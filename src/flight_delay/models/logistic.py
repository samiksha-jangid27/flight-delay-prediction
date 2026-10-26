
"""Interpretable Logistic Regression baseline for flight delay prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flight_delay.data.pipeline import build_data_pipeline
from flight_delay.features.selection import split_features_and_target
from flight_delay.features.model_features import (
    FEATURE_SETS,
    CATEGORICAL_FEATURES,
    select_model_features,
)

MODEL_NAME = "logistic_regression"
DEFAULT_SEED = 42


def make_logistic_pipeline(
    feature_set: str = "full",
    *,
    random_state: int = DEFAULT_SEED,
    C: float = 1.0,
    class_weight: str | None = None,
) -> Pipeline:
    """Create preprocessing and Logistic Regression in one pipeline."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; "
            f"expected {list(FEATURE_SETS)}"
        )

    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be a positive, finite number")
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be None or 'balanced'")

    fields = FEATURE_SETS[feature_set]
    numeric_fields = [
        field for field in fields if field not in CATEGORICAL_FEATURES
    ]
    categorical_fields = [
        field for field in fields if field in CATEGORICAL_FEATURES
    ]

    transformers = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_fields,
        ),
        (
            "categorical",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="__MISSING__",
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            ),
            categorical_fields,
        ),
    ]

    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(transformers, remainder="drop"),
            ),
            (
                "model",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    feature_set: str = "full",
    random_state: int = DEFAULT_SEED,
    C: float = 1.0,
    class_weight: str | None = None,
) -> Pipeline:
    """Fit the model using training rows only."""
    labels = set(pd.Series(y_train).dropna().unique().tolist())

    if labels != {0, 1} or pd.Series(y_train).isna().any():
        raise ValueError("Training labels must include both binary classes 0 and 1")

    pipeline = make_logistic_pipeline(
        feature_set,
        random_state=random_state,
        C=C,
        class_weight=class_weight,
    )
    X_selected = select_model_features(X_train, feature_set)
    pipeline.fit(X_selected, y_train)

    return pipeline


def run_experiment(
    data_path: str | Path = "data/raw/flights.csv",
    *,
    output_dir: str | Path = "reports/results",
    model_dir: str | Path = "models/artifacts",
    feature_set: str = "full",
    random_state: int = DEFAULT_SEED,
    C: float = 1.0,
    class_weight: str | None = None,
) -> dict[str, object]:
    """Train, evaluate and export validation predictions."""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Single pipeline execution retains matching identifiers, labels and features.
    train, validation, _test_split = build_data_pipeline(data_path)
    X_train, y_train = split_features_and_target(train)
    X_validation, y_validation = split_features_and_target(validation)
    validation_dates = pd.to_datetime(
        validation[["year", "month", "day"]]
    ).dt.strftime("%Y-%m-%d")

    if "id" not in validation.columns:
        raise ValueError("Validation data is missing the raw 'id' column")
    if validation["id"].duplicated().any():
        raise ValueError("Validation flight IDs must be unique")

    if len(X_validation) != len(validation):
        raise ValueError("Validation features and identifiers are misaligned")

    model = fit_logistic_baseline(
        X_train,
        y_train,
        feature_set=feature_set,
        random_state=random_state,
        C=C,
        class_weight=class_weight,
    )

    X_selected = select_model_features(X_validation, feature_set)
    probabilities = model.predict_proba(X_selected)[:, 1]

    metrics = {
        "average_precision": float(
            average_precision_score(y_validation, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_validation, probabilities)),
    }

    predictions = pd.DataFrame(
        {
            "flight_id": validation["id"].to_numpy(),
            "flight_date": validation_dates.to_numpy(),
            "y_true": y_validation.to_numpy(),
            "y_probability": probabilities,
        }
    )

    output_path = output_dir / "logistic_validation_predictions.csv"
    predictions.to_csv(output_path, index=False)

    pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "feature_set": feature_set,
                "C": C,
                "class_weight": class_weight,
                **metrics,
            }
        ]
    ).to_csv(output_dir / "logistic_experiment_results.csv", index=False)

    # Preserve unweighted/balanced model artifacts separately.
    suffix = "balanced" if class_weight == "balanced" else "unweighted"
    model_path = model_dir / f"logistic_pipeline_{suffix}_C{C:g}.joblib"
    joblib.dump(model, model_path)

    return {
        "model": model,
        "model_path": model_path,
        "predictions": predictions,
        "metrics": metrics,
        "output_path": output_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Logistic Regression baseline."
    )
    parser.add_argument("--data", default="data/raw/flights.csv")
    parser.add_argument("--results", default="reports/results")
    parser.add_argument("--models", default="models/artifacts")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="full")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--class-weight",
        choices=["balanced"],
        default=None,
    )
    args = parser.parse_args()

    result = run_experiment(
        data_path=args.data,
        output_dir=args.results,
        model_dir=args.models,
        feature_set=args.feature_set,
        random_state=args.seed,
        C=args.C,
        class_weight=args.class_weight,
    )

    print("Validation metrics:")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value:.6f}")
    print("Predictions saved to:", result["output_path"])
    print("Model saved to:", result["model_path"])


if __name__ == "__main__":
    main()

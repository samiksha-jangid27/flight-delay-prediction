"""Leakage-conscious, reproducible XGBoost experiments for flight delays.

Input contract: frames produced by flight_delay.data.prepare.prepare_model_data
or flight_delay.features.selection.split_features_and_target.

Only scheduling-time and prior-date historical features are admitted to the
model. This module deliberately never reads the held-out test partition for
model selection, tuning, or evaluation.

Example (run from repository root after ``pip install -e .``)::

    python -m flight_delay.models.xgboost_model \
        --data data/raw/flights.csv --n-iter 8 --cv-folds 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Explicit allowlist: introducing new columns upstream cannot silently introduce
# outcome fields into a fitted model.
SCHEDULE_FEATURES: tuple[str, ...] = (
    "month", "day", "sched_dep_time", "sched_arr_time", "hour", "minute",
    "carrier", "origin", "dest", "distance", "day_of_week", "is_weekend",
    "season", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos",
)
ROUTE_FEATURES: tuple[str, ...] = (
    "route", "distance_band", "is_long_haul", "distance_log",
)
HISTORICAL_FEATURES: tuple[str, ...] = (
    "carrier_historical_delay_rate", "carrier_historical_flight_count",
    "origin_historical_delay_rate", "origin_historical_flight_count",
    "destination_historical_delay_rate", "destination_historical_flight_count",
    "route_historical_delay_rate", "route_historical_flight_count",
)
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "schedule": SCHEDULE_FEATURES,
    "route": SCHEDULE_FEATURES + ROUTE_FEATURES,
    "full": SCHEDULE_FEATURES + ROUTE_FEATURES + HISTORICAL_FEATURES,
}
CATEGORICAL_FEATURES: frozenset[str] = frozenset(
    {"carrier", "origin", "dest", "route", "distance_band"}
)
MODEL_NAME = "xgboost"
DEFAULT_SEED = 42


def select_xgboost_features(X: pd.DataFrame, feature_set: str = "full") -> pd.DataFrame:
    """Return a validated feature-only frame, never using arbitrary extra fields.

    High-cardinality ``tailnum``/``flight`` and free-form ``time_hour`` are
    deliberately excluded. The target, IDs, and post-departure columns are
    forbidden even if an upstream caller accidentally passes them in ``X``.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; expected {list(FEATURE_SETS)}")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with named columns")

    fields = FEATURE_SETS[feature_set]
    missing = [field for field in fields if field not in X.columns]
    if missing:
        raise ValueError(f"Required {feature_set!r} features missing: {missing}")

    selected = X.loc[:, list(fields)].copy()
    for field in fields:
        if field in CATEGORICAL_FEATURES:
            # CategoricalDtype works with ordinary object/string values after conversion.
            selected[field] = selected[field].astype("object")
        else:
            selected[field] = pd.to_numeric(selected[field], errors="raise")
    return selected


def make_xgboost_pipeline(
    feature_set: str = "full", *, random_state: int = DEFAULT_SEED,
    n_jobs: int = 4, model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Create one fitted-on-training-only preprocessing + XGB pipeline.

    Numeric missing values use -1 as a distinguished sentinel; all historical
    delay rates are nonnegative, and historical counts are nonnegative. The
    accompanying flight counts distinguish cold-start/missing history.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    fields = FEATURE_SETS[feature_set]
    numeric_fields = [f for f in fields if f not in CATEGORICAL_FEATURES]
    categorical_fields = [f for f in fields if f in CATEGORICAL_FEATURES]

    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", SimpleImputer(strategy="constant", fill_value=-1), numeric_fields)
    ]
    if categorical_fields:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_fields,
        ))

    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.08,
        "min_child_weight": 5,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "importance_type": "gain",
    }
    if model_params:
        params.update(model_params)
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers, remainder="drop")),
        ("model", XGBClassifier(**params)),
    ])


def _validate_binary_labels(y: pd.Series) -> None:
    labels = set(pd.Series(y).dropna().unique().tolist())
    if labels != {0, 1} or pd.Series(y).isna().any():
        raise ValueError("Training labels must include both binary classes 0 and 1")


def fit_xgboost_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, *,
    feature_set: str = "full", random_state: int = DEFAULT_SEED,
    n_jobs: int = 4, model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Fit a transparent untuned baseline on the supplied training rows only."""
    _validate_binary_labels(y_train)
    pipeline = make_xgboost_pipeline(
        feature_set, random_state=random_state, n_jobs=n_jobs,
        model_params=model_params,
    )
    pipeline.fit(select_xgboost_features(X_train, feature_set), y_train)
    return pipeline


def chronological_date_cv(
    train_dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    *, n_splits: int = 3, gap_dates: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding chronological folds that never split the same date.

    ``gap_dates`` is an embargo measured in whole calendar dates. Each fold's
    training dates strictly precede its validation dates (with a date gap).
    It is not a substitute for knowing when past-flight outcomes arrived.
    """
    date_values = pd.to_datetime(pd.Series(train_dates), errors="raise").dt.normalize()
    if date_values.isna().any():
        raise ValueError("Training date input has missing values")
    unique_dates = pd.DatetimeIndex(date_values.unique()).sort_values()
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_dates)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for training_days, validation_days in splitter.split(unique_dates):
        in_train = date_values.isin(unique_dates[training_days]).to_numpy()
        in_validation = date_values.isin(unique_dates[validation_days]).to_numpy()
        train_indices = np.flatnonzero(in_train)
        validation_indices = np.flatnonzero(in_validation)
        if not len(train_indices) or not len(validation_indices):
            raise ValueError("Date-grouped CV yielded an empty fold")
        assert date_values.iloc[train_indices].max() < date_values.iloc[validation_indices].min()
        folds.append((train_indices, validation_indices))
    return folds


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    *, feature_set: str = "full", n_iter: int = 8,
    cv_folds: int = 3, random_state: int = DEFAULT_SEED,
    n_jobs: int = 4,
    param_distributions: dict[str, Any] | None = None,
) -> RandomizedSearchCV:
    """Tune AP with train-only, expanding date-based CV; never pass holdout rows.

    ``RandomizedSearchCV`` ranks candidates by cross-validated Average
    Precision (AP). This is commonly reported as PR-AUC but is distinct from
    trapezoidal PR curve area. Its best estimator is refitted on all training.
    """
    _validate_binary_labels(y_train)
    if n_iter < 1:
        raise ValueError("n_iter must be positive")
    if len(X_train) != len(y_train) or len(train_dates) != len(X_train):
        raise ValueError("X_train, y_train, and train_dates must align by row position")

    positives = int((pd.Series(y_train) == 1).sum())
    negatives = int((pd.Series(y_train) == 0).sum())
    imbalance = round(negatives / positives, 4)
    if param_distributions is None:
        param_distributions = {
            "model__n_estimators": [150, 250, 350],
            "model__max_depth": [3, 4, 6],
            "model__learning_rate": [0.03, 0.06, 0.10],
            "model__min_child_weight": [1, 5, 10],
            "model__subsample": [0.75, 0.90, 1.0],
            "model__colsample_bytree": [0.75, 0.90, 1.0],
            "model__reg_lambda": [1.0, 5.0, 10.0],
            "model__scale_pos_weight": sorted({1.0, imbalance}),
        }

    pipeline = make_xgboost_pipeline(feature_set, random_state=random_state, n_jobs=n_jobs)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=chronological_date_cv(train_dates, n_splits=cv_folds),
        random_state=random_state,
        refit=True,
        n_jobs=1,  # Avoid massive nested parallelism with XGBoost's n_jobs.
        error_score="raise",
        return_train_score=False,
        verbose=0,
    )
    search.fit(select_xgboost_features(X_train, feature_set), y_train)
    return search


def evaluate_probabilities(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    """Probability-ranking metrics, leaving calibration and thresholds to Lakshya."""
    scores = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(scores):
        raise ValueError("Labels and probabilities are different lengths")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model probabilities must be finite values in [0, 1]")
    _validate_binary_labels(y_true)
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def feature_importance_table(pipeline: Pipeline) -> pd.DataFrame:
    """Gain-based importance per transformed feature (not causal attribution)."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    return pd.DataFrame({
        "encoded_feature": names,
        "gain_importance": importance,
    }).sort_values("gain_importance", ascending=False).reset_index(drop=True)


def run_experiments(
    data_path: str | Path = "data/raw/flights.csv", *,
    output_dir: str | Path = "reports/results",
    model_dir: str | Path = "models/artifacts",
    n_iter: int = 8,
    cv_folds: int = 3,
    random_state: int = DEFAULT_SEED,
    n_jobs: int = 4,
    run_tuning: bool = True,
) -> dict[str, Any]:
    """Run ablations and tuned XGBoost, export validation-only deliverables.

    The official held-out test data is *never used here*. Evaluation/calibration
    lead should perform final test evaluation with the frozen chosen model.
    Outputs are ignored by git under the team's existing .gitignore rules.
    """
    from flight_delay.data.pipeline import build_data_pipeline
    from flight_delay.features.selection import split_features_and_target

    data_path, output_dir, model_dir = map(Path, (data_path, output_dir, model_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train, validation, _test_untouched = build_data_pipeline(data_path)
    X_train, y_train = split_features_and_target(train)
    X_validation, y_validation = split_features_and_target(validation)
    training_dates = pd.to_datetime(train[["year", "month", "day"]])
    validation_dates = pd.to_datetime(validation[["year", "month", "day"]])

    results: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    models: dict[str, Pipeline] = {}

    for feature_set in FEATURE_SETS:
        variant = f"baseline_{feature_set}"
        model = fit_xgboost_baseline(
            X_train, y_train, feature_set=feature_set,
            random_state=random_state, n_jobs=n_jobs,
        )
        probabilities = model.predict_proba(
            select_xgboost_features(X_validation, feature_set)
        )[:, 1]
        metrics = evaluate_probabilities(y_validation, probabilities)
        models[variant] = model
        predictions[variant] = probabilities
        results.append({
            "experiment": variant,
            "feature_set": feature_set,
            "tuned": False,
            **metrics,
            "cv_average_precision": None,
        })

    search: RandomizedSearchCV | None = None
    if run_tuning:
        search = tune_xgboost(
            X_train, y_train, training_dates,
            n_iter=n_iter, cv_folds=cv_folds,
            random_state=random_state, n_jobs=n_jobs,
        )
        probabilities = search.best_estimator_.predict_proba(
            select_xgboost_features(X_validation, "full")
        )[:, 1]
        metrics = evaluate_probabilities(y_validation, probabilities)
        models["tuned_full"] = search.best_estimator_
        predictions["tuned_full"] = probabilities
        results.append({
            "experiment": "tuned_full", "feature_set": "full", "tuned": True,
            **metrics, "cv_average_precision": float(search.best_score_),
        })
        pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score"
        ).to_csv(output_dir / "xgboost_tuning_cv_results.csv", index=False)

    leaderboard = pd.DataFrame(results).sort_values(
        "average_precision", ascending=False
    ).reset_index(drop=True)
    leaderboard.to_csv(output_dir / "xgboost_experiment_results.csv", index=False)

    # Choose the best *validation* experiment. Validation is consequently a
    # model-selection set, not a final unbiased generalization estimate.
    winner = str(leaderboard.iloc[0]["experiment"])
    winning_pipeline = models[winner]
    winning_predictions = predictions[winner]

    feature_importance_table(winning_pipeline).to_csv(
        output_dir / "xgboost_feature_importance.csv", index=False
    )
    joblib.dump(winning_pipeline, model_dir / "xgboost_pipeline.joblib")

    # Raw id is a join key only, never a feature. Persist row identity so
    # evaluation can align this prediction stream with Logistic Regression.
    if "id" not in validation.columns:
        raise ValueError("Validation data is missing the raw 'id' join key")
    if validation["id"].duplicated().any():
        raise ValueError("Validation raw 'id' is not unique")
    validation_output = pd.DataFrame({
        "flight_id": validation["id"].to_numpy(),
        "flight_date": validation_dates.dt.strftime("%Y-%m-%d").to_numpy(),
        "y_true": y_validation.to_numpy(),
        "y_probability": winning_predictions,
        "model_name": MODEL_NAME,
        "experiment": winner,
    })
    validation_output.to_csv(output_dir / "xgboost_validation_predictions.csv", index=False)

    manifest = {
        "data_path": str(data_path),
        "selected_experiment": winner,
        "selection_metric": "validation average_precision (sklearn AP)",
        "cv_scoring": "average_precision",
        "cv_folds": cv_folds if run_tuning else None,
        "cv_gap_dates": 1 if run_tuning else None,
        "n_iter": n_iter if run_tuning else 0,
        "seed": random_state,
        "n_jobs": n_jobs,
        "train_rows": len(y_train),
        "validation_rows": len(y_validation),
        "test_policy": "untouched; reserved for independent final evaluation",
        "train_first_date": str(training_dates.min().date()),
        "train_last_date": str(training_dates.max().date()),
        "validation_first_date": str(validation_dates.min().date()),
        "validation_last_date": str(validation_dates.max().date()),
        "rolling_history_policy": "previous calendar dates; see docs/data_contract.md",
        "selected_feature_set": next(r["feature_set"] for r in results if r["experiment"] == winner),
        "selected_model_parameters": winning_pipeline.named_steps["model"].get_params(),
        "best_cv_parameters": search.best_params_ if search is not None else None,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "caveats": [
            "AP is average precision, not trapezoidal area under PR curve.",
            "Prior-day historical labels must be available at the actual forecast time.",
            "Validation is used for model selection; final test metrics need separate evaluation.",
            "No probability calibration or operational threshold has been applied.",
        ],
    }
    (output_dir / "xgboost_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return {
        "leaderboard": leaderboard,
        "winner": winner,
        "validation_predictions": validation_output,
        "model": winning_pipeline,
        "search": search,
        "result_directory": output_dir,
        "model_directory": model_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/flights.csv")
    parser.add_argument("--results", default="reports/results")
    parser.add_argument("--models", default="models/artifacts")
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()
    result = run_experiments(
        data_path=args.data,
        output_dir=args.results,
        model_dir=args.models,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        run_tuning=not args.skip_tuning,
    )
    print("\nExperiment ranking (validation; not held-out test):")
    print(result["leaderboard"].to_string(index=False))
    print("\nSelected experiment:", result["winner"])
    print("Saved results to:", result["result_directory"])
    print("Saved model to:", result["model_directory"])


if __name__ == "__main__":
    main()

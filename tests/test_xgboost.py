"""Fast synthetic contract tests for Shubhaang's XGBoost implementation.

No external dataset is required for these tests. To run the complete project's
real-data tests, execute ``pytest`` from the repository root after installing
``pip install -e .`` and ensuring data/raw/flights.csv is present.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from flight_delay.models.xgboost_model import (
    FEATURE_SETS,
    chronological_date_cv,
    evaluate_probabilities,
    feature_importance_table,
    fit_xgboost_baseline,
    select_xgboost_features,
    tune_xgboost,
)


@pytest.fixture
def training_sample():
    """Synthetic feature-only data matching the team's prepared-data schema."""
    rng = np.random.default_rng(42)
    count_per_date = 8
    days = 16
    count = count_per_date * days
    dates = pd.Series(pd.date_range("2013-01-01", periods=days).repeat(count_per_date))
    carriers = np.resize(np.array(["AA", "UA", "DL", "B6"], dtype=object), count)
    origins = np.resize(np.array(["JFK", "LGA", "EWR", "JFK"], dtype=object), count)
    dests = np.resize(np.array(["MIA", "BOS", "ORD", "ATL"], dtype=object), count)
    distances = np.resize(np.array([1000, 300, 800, 700]), count)
    hours = np.arange(count) % 24
    delayed = ((np.arange(count) * 7) % 13 > 7).astype(int)
    assert set(delayed) == {0, 1}

    frame = pd.DataFrame({
        "month": dates.dt.month.to_numpy(), "day": dates.dt.day.to_numpy(),
        "sched_dep_time": hours * 100 + 15,
        "sched_arr_time": (hours * 100 + 400) % 2400,
        "hour": hours, "minute": np.repeat(15, count),
        "carrier": carriers, "origin": origins, "dest": dests,
        "distance": distances,
        "day_of_week": dates.dt.dayofweek.to_numpy(),
        "is_weekend": (dates.dt.dayofweek >= 5).astype(int).to_numpy(),
        "season": np.repeat(1, count),
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
        "month_sin": np.sin(2 * np.pi / 12) * np.ones(count),
        "month_cos": np.cos(2 * np.pi / 12) * np.ones(count),
        "day_of_week_sin": np.sin(2 * np.pi * dates.dt.dayofweek.to_numpy() / 7),
        "day_of_week_cos": np.cos(2 * np.pi * dates.dt.dayofweek.to_numpy() / 7),
        "route": [f"{o}>{d}" for o, d in zip(origins, dests)],
        "distance_band": np.resize(np.array(["750-1000", "250-500", "750-1000", "500-750"], dtype=object), count),
        "is_long_haul": distances >= 1500,
        "distance_log": np.log1p(distances),
        "carrier_historical_delay_rate": rng.random(count),
        "carrier_historical_flight_count": rng.integers(0, 100, size=count),
        "origin_historical_delay_rate": rng.random(count),
        "origin_historical_flight_count": rng.integers(0, 100, size=count),
        "destination_historical_delay_rate": rng.random(count),
        "destination_historical_flight_count": rng.integers(0, 100, size=count),
        "route_historical_delay_rate": rng.random(count),
        "route_historical_flight_count": rng.integers(0, 100, size=count),
        "origin_hourly_departures": rng.integers(1, 40, size=count),
        "dest_hourly_arrivals": rng.integers(1, 10, size=count),
        "dep_delay": delayed * 1_000,  # obvious target leak: must never appear.
        "arr_delay": delayed * 999,
        "is_delayed": delayed,
        "id": np.arange(count),
        "tailnum": "DONT_USE",
        "flight": np.arange(count),
        "time_hour": dates.dt.strftime("%Y-%m-%d %H:%M:%S"),
    })
    frame.loc[0, "carrier_historical_delay_rate"] = np.nan
    return frame, pd.Series(delayed), dates


def test_feature_allowlist_excludes_leakage(training_sample):
    X, _, _ = training_sample
    for name, expected_columns in FEATURE_SETS.items():
        selected = select_xgboost_features(X, feature_set=name)
        assert tuple(selected.columns) == expected_columns
        assert not set(["dep_delay", "arr_delay", "is_delayed", "id", "tailnum", "flight", "time_hour"]) & set(selected)


def test_missing_required_column_rejected(training_sample):
    X, _, _ = training_sample
    with pytest.raises(ValueError, match="missing"):
        select_xgboost_features(X.drop(columns="carrier"))


def test_unknown_feature_set_rejected(training_sample):
    X, _, _ = training_sample
    with pytest.raises(ValueError, match="Unknown"):
        select_xgboost_features(X, feature_set="invalid")


def test_baseline_predicts_probabilities_and_handles_unseen_categories(training_sample):
    X, y, _ = training_sample
    model = fit_xgboost_baseline(
        X.iloc[:96], y.iloc[:96], feature_set="full", n_jobs=1,
        model_params={"n_estimators": 12, "max_depth": 2, "min_child_weight": 1},
    )
    assert isinstance(model, Pipeline)
    held_out = X.iloc[96:].copy()
    held_out.loc[96, "carrier"] = "UNSEEN"
    held_out.loc[97, "route_historical_delay_rate"] = np.nan
    probabilities = model.predict_proba(select_xgboost_features(held_out))[:, 1]
    assert len(probabilities) == len(held_out)
    assert np.isfinite(probabilities).all()
    assert np.logical_and(probabilities >= 0, probabilities <= 1).all()
    assert 0 <= evaluate_probabilities(y.iloc[96:], probabilities)["average_precision"] <= 1
    assert len(feature_importance_table(model)) > 0


def test_cv_splits_entire_calendar_dates_without_future_leakage(training_sample):
    _, _, dates = training_sample
    for tr, va in chronological_date_cv(dates, n_splits=3, gap_dates=1):
        tr_dates = set(dates.iloc[tr])
        val_dates = set(dates.iloc[va])
        assert not tr_dates.intersection(val_dates)
        assert max(tr_dates) < min(val_dates)
        assert (min(val_dates) - max(tr_dates)).days >= 2


def test_train_only_hyperparameter_search(training_sample):
    X, y, dates = training_sample
    tuning = tune_xgboost(
        X, y, dates, n_iter=2, cv_folds=2, n_jobs=1,
        param_distributions={
            "model__n_estimators": [8, 12],
            "model__max_depth": [2],
            "model__min_child_weight": [1],
        },
    )
    assert tuning.best_estimator_ is not None
    assert set(tuning.cv_results_["param_model__n_estimators"]) == {8, 12}
    assert np.isfinite(tuning.best_score_)


def test_saved_pipeline_reproduces_predictions(training_sample, tmp_path):
    X, y, _ = training_sample
    model = fit_xgboost_baseline(
        X.iloc[:96], y.iloc[:96], n_jobs=1,
        model_params={"n_estimators": 10, "max_depth": 2, "min_child_weight": 1},
    )
    out = tmp_path / "xgboost_pipeline.joblib"
    joblib.dump(model, out)
    restored = joblib.load(out)  # Trusted local test file only.
    frame = select_xgboost_features(X.iloc[96:])
    np.testing.assert_allclose(model.predict_proba(frame), restored.predict_proba(frame))


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError, match="finite"):
        evaluate_probabilities(pd.Series([0, 1]), np.array([0.1, np.nan]))


def test_runner_exports_compatible_validation_artifacts(training_sample, monkeypatch, tmp_path):
    """Validate the runner's file contract without downloading the 2013 CSV."""
    import json
    import sys
    from types import ModuleType

    from flight_delay.features.model_features import FEATURE_SETS
    from flight_delay.models.xgboost_model import run_experiments

    X, y, dates = training_sample
    raw = X.copy()
    raw["year"] = 2013
    # Date values are already embedded in day/month; these match "dates".
    train = raw.iloc[:96].copy()
    validation = raw.iloc[96:].copy()

    def fake_build_data_pipeline(path):
        # Test must remain uninspected; it is not part of training/evaluation.
        return train, validation, object()

    def fake_split_features_and_target(df):
        return df.drop(columns=["is_delayed", "arr_delay", "dep_delay", "id", "year"]), df["is_delayed"]

    # Install small stubs under names that exist in the real team repository.
    data_pkg = ModuleType("flight_delay.data")
    data_pkg.__path__ = []
    feature_pkg = ModuleType("flight_delay.features")
    feature_pkg.__path__ = []
    pipeline_mod = ModuleType("flight_delay.data.pipeline")
    pipeline_mod.build_data_pipeline = fake_build_data_pipeline
    selection_mod = ModuleType("flight_delay.features.selection")
    selection_mod.split_features_and_target = fake_split_features_and_target
    for name, module in (
        ("flight_delay.data", data_pkg),
        ("flight_delay.features", feature_pkg),
        ("flight_delay.data.pipeline", pipeline_mod),
        ("flight_delay.features.selection", selection_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    result = run_experiments(
        data_path="synthetic.csv", output_dir=tmp_path / "results",
        model_dir=tmp_path / "models", n_iter=2, cv_folds=2,
        n_jobs=1,
    )
    outputs = tmp_path / "results"
    assert (outputs / "xgboost_experiment_results.csv").exists()
    assert (outputs / "xgboost_validation_predictions.csv").exists()
    assert (outputs / "xgboost_feature_importance.csv").exists()
    assert (outputs / "xgboost_tuning_cv_results.csv").exists()
    assert (outputs / "xgboost_manifest.json").exists()
    assert (tmp_path / "models" / "xgboost_pipeline.joblib").exists()

    predictions = pd.read_csv(outputs / "xgboost_validation_predictions.csv")
    assert predictions["flight_id"].tolist() == validation["id"].tolist()
    assert predictions["y_true"].tolist() == validation["is_delayed"].tolist()
    assert predictions["y_probability"].between(0, 1).all()
    assert len(result["leaderboard"]) == len(FEATURE_SETS) + 1
    assert set(result["leaderboard"]["feature_set"]) == {*FEATURE_SETS, "full"}
    manifest = json.loads((outputs / "xgboost_manifest.json").read_text())
    assert manifest["test_policy"] == "not scored by this validation-only component run"
    assert manifest["train_rows"] == 96
    assert manifest["validation_rows"] == 32

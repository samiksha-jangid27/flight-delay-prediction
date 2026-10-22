
import numpy as np
import pandas as pd
import pytest

from flight_delay.models.logistic import (
    fit_logistic_baseline,
    make_logistic_pipeline,
)


def make_training_data(n_rows=40):
    """Build a small synthetic dataset with the required schedule features."""
    rng = np.random.default_rng(42)

    X = pd.DataFrame({
        "month": rng.integers(1, 13, n_rows),
        "day": rng.integers(1, 29, n_rows),
        "sched_dep_time": rng.integers(500, 2300, n_rows),
        "sched_arr_time": rng.integers(600, 2359, n_rows),
        "hour": rng.integers(0, 24, n_rows),
        "minute": rng.integers(0, 60, n_rows),
        "carrier": rng.choice(["AA", "DL", "UA"], n_rows),
        "origin": rng.choice(["JFK", "LGA", "EWR"], n_rows),
        "dest": rng.choice(["ORD", "LAX", "BOS"], n_rows),
        "distance": rng.integers(100, 3000, n_rows),
        "day_of_week": rng.integers(0, 7, n_rows),
        "is_weekend": rng.integers(0, 2, n_rows),
        "season": rng.integers(1, 5, n_rows),
        "hour_sin": rng.uniform(-1, 1, n_rows),
        "hour_cos": rng.uniform(-1, 1, n_rows),
        "month_sin": rng.uniform(-1, 1, n_rows),
        "month_cos": rng.uniform(-1, 1, n_rows),
        "day_of_week_sin": rng.uniform(-1, 1, n_rows),
        "day_of_week_cos": rng.uniform(-1, 1, n_rows),
    })

    y = pd.Series(
        np.tile([0, 1], n_rows // 2),
        name="is_delayed",
    )

    return X, y


def test_make_logistic_pipeline():
    pipeline = make_logistic_pipeline(feature_set="schedule")

    assert "preprocess" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_fit_and_predict_probabilities():
    X, y = make_training_data()

    model = fit_logistic_baseline(
        X,
        y,
        feature_set="schedule",
    )

    probabilities = model.predict_proba(X)[:, 1]

    assert len(probabilities) == len(y)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_unknown_feature_set_raises_error():
    with pytest.raises(ValueError, match="Unknown feature set"):
        make_logistic_pipeline(feature_set="invalid")


def test_missing_required_feature_raises_error():
    X, y = make_training_data()
    X = X.drop(columns=["hour"])

    with pytest.raises(ValueError, match="features missing"):
        fit_logistic_baseline(X, y, feature_set="schedule")


def test_training_requires_both_classes():
    X, _ = make_training_data()
    y = pd.Series(np.zeros(len(X), dtype=int))

    with pytest.raises(ValueError, match="both binary classes"):
        fit_logistic_baseline(X, y, feature_set="schedule")


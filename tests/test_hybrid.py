"""Stacked hybrid: alignment, leakage safety, schema, and mechanism tests.

These tests exercise the stacker on synthetic base-model predictions so they run
without PyTorch, the dataset, or the frozen XGBoost artifact. They verify that
the implementation is correct and that it *can* add value when the base models
are genuinely complementary; whether value materializes on the real data is a
separate empirical question answered in docs/hybrid_approach.md.
"""

import numpy as np
import pandas as pd
import pytest

from flight_delay.deep_learning.hybrid import (
    PREDICTION_COLUMNS,
    error_diversity,
    fit_meta_learner,
    predict_hybrid,
)
from flight_delay.evaluation.metrics import evaluate


def _frame(partition, flight_ids, labels, probs):
    return pd.DataFrame(
        {
            "flight_id": pd.array([str(f) for f in flight_ids], dtype="string"),
            "flight_date": "2013-10-01",
            "y_true": np.asarray(labels, dtype=np.int64),
            "y_probability": np.asarray(probs, dtype=np.float64),
            "partition": partition,
        }
    )


def _complementary_split(n=400, seed=0):
    """Two weak base models whose errors are on disjoint halves of the data."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    half = n // 2
    # xgb is informative on the first half, noise on the second; mlp vice versa.
    xgb = np.where(np.arange(n) < half, 0.5 + 0.4 * (y - 0.5) * 2, rng.uniform(0.3, 0.7, n))
    mlp = np.where(np.arange(n) >= half, 0.5 + 0.4 * (y - 0.5) * 2, rng.uniform(0.3, 0.7, n))
    return y, np.clip(xgb, 0, 1), np.clip(mlp, 0, 1)


def test_hybrid_improves_when_base_models_are_complementary():
    y, xgb_p, mlp_p = _complementary_split()
    ids = range(len(y))
    cal_xgb = _frame("train_calibration", ids, y, xgb_p)
    cal_mlp = _frame("train_calibration", ids, y, mlp_p)
    val_xgb = _frame("validation", ids, y, xgb_p)
    val_mlp = _frame("validation", ids, y, mlp_p)

    meta = fit_meta_learner(cal_xgb, cal_mlp, seed=0)
    hybrid = predict_hybrid(meta, val_xgb, val_mlp)

    ap_hybrid = evaluate(hybrid.y_true, hybrid.y_probability)["average_precision"]
    ap_xgb = evaluate(y, xgb_p)["average_precision"]
    ap_mlp = evaluate(y, mlp_p)["average_precision"]
    assert ap_hybrid > ap_xgb and ap_hybrid > ap_mlp


def test_hybrid_output_matches_project_schema_and_rows():
    y, xgb_p, mlp_p = _complementary_split(n=50)
    ids = range(len(y))
    meta = fit_meta_learner(
        _frame("train_calibration", ids, y, xgb_p),
        _frame("train_calibration", ids, y, mlp_p),
    )
    hybrid = predict_hybrid(
        meta,
        _frame("validation", ids, y, xgb_p),
        _frame("validation", ids, y, mlp_p),
    )
    assert hybrid.columns.tolist() == list(PREDICTION_COLUMNS)
    assert hybrid.partition.eq("validation").all()
    assert hybrid.flight_id.is_unique
    assert hybrid.y_probability.between(0, 1).all()
    assert len(hybrid) == len(y)


def test_fit_is_reproducible():
    y, xgb_p, mlp_p = _complementary_split(n=120)
    ids = range(len(y))
    args = (_frame("train_calibration", ids, y, xgb_p), _frame("train_calibration", ids, y, mlp_p))
    a = fit_meta_learner(*args, seed=7)
    b = fit_meta_learner(*args, seed=7)
    np.testing.assert_array_equal(a.coef_, b.coef_)


def test_misaligned_flight_ids_are_rejected():
    y, xgb_p, mlp_p = _complementary_split(n=20)
    cal_xgb = _frame("train_calibration", range(20), y, xgb_p)
    cal_mlp = _frame("train_calibration", range(100, 120), y, mlp_p)  # disjoint ids
    with pytest.raises(ValueError, match="align one-to-one"):
        fit_meta_learner(cal_xgb, cal_mlp)


def test_label_disagreement_is_rejected():
    y, xgb_p, mlp_p = _complementary_split(n=20)
    cal_xgb = _frame("train_calibration", range(20), y, xgb_p)
    flipped = 1 - y
    cal_mlp = _frame("train_calibration", range(20), flipped, mlp_p)
    with pytest.raises(ValueError, match="labels disagree"):
        fit_meta_learner(cal_xgb, cal_mlp)


def test_error_diversity_detects_redundant_models():
    y, xgb_p, _ = _complementary_split(n=200)
    ids = range(len(y))
    identical = _frame("validation", ids, y, xgb_p)
    # Two identical models have perfectly correlated probabilities and residuals.
    diversity = error_diversity(identical, identical.copy(), "validation")
    assert diversity["probability_pearson"] == pytest.approx(1.0, abs=1e-9)
    assert diversity["residual_pearson"] == pytest.approx(1.0, abs=1e-9)


def test_wrong_partition_is_rejected():
    y, xgb_p, mlp_p = _complementary_split(n=20)
    # Pass validation-labelled frames where calibration is required.
    cal_xgb = _frame("validation", range(20), y, xgb_p)
    cal_mlp = _frame("validation", range(20), y, mlp_p)
    with pytest.raises(ValueError, match="partition"):
        fit_meta_learner(cal_xgb, cal_mlp)

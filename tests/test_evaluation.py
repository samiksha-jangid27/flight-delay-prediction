"""Synthetic contracts for calibrated evaluation and controlled finalization."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from flight_delay.evaluation.calibration import (
    ScoreCalibrator,
    check_windows,
    reliability,
)
from flight_delay.evaluation.metrics import (
    align_predictions,
    evaluate,
    paired_bootstrap,
)
from flight_delay.evaluation.thresholds import select_thresholds, threshold_table


def frame():
    return pd.DataFrame(
        {
            "flight_id": [1, 2, 3, 4],
            "flight_date": ["2013-09-01"] * 4,
            "y_true": [0, 1, 0, 1],
            "y_probability": [0.1, 0.9, 0.7, 0.4],
            "partition": "validation",
        }
    )


def test_known_metrics():
    r = evaluate([0, 1, 0, 1], [0.1, 0.9, 0.7, 0.4])
    assert [r[k] for k in ["tn", "fp", "fn", "tp"]] == [1, 1, 1, 1]
    assert r["precision"] == r["recall"] == r["f1"] == 0.5
    assert r["roc_auc"] == roc_auc_score([0, 1, 0, 1], [0.1, 0.9, 0.7, 0.4])
    assert r["average_precision"] == average_precision_score(
        [0, 1, 0, 1], [0.1, 0.9, 0.7, 0.4]
    )
    assert r["class_metrics"]["0"]["support"] == 2


@pytest.mark.parametrize(
    "y,p",
    [
        ([0, 1], [0.1, 1.1]),
        ([0, 1], [np.nan, 0.5]),
        ([0, None], [0.1, 0.2]),
        ([], []),
        ([0], [0.1, 0.2]),
        ([2], [0.1]),
    ],
)
def test_invalid_metrics(y, p):
    with pytest.raises(ValueError):
        evaluate(y, p)


@pytest.mark.parametrize("y", [[0, 0], [1, 1]])
def test_single_class(y):
    r = evaluate(y, [0.5, 0.5])
    assert r["roc_auc"] is None
    assert r["average_precision"] is None and np.isfinite(r["log_loss"])


def test_constant_and_boundaries():
    assert evaluate([0, 1], [0.5, 0.5])["roc_auc"] == 0.5
    assert evaluate([0, 1], [0, 1], 0)["tp"] == 1
    assert evaluate([0, 1], [0, 1], 1)["fp"] == 0
    with pytest.raises(ValueError):
        evaluate([0, 1], [0, 1], 1.1)


@pytest.mark.parametrize(
    "change", ["duplicate", "id", "label", "date", "partition", "missing"]
)
def test_alignment_failure(change):
    a = frame()
    b = a.copy()
    if change == "duplicate":
        b.loc[1, "flight_id"] = 1
    if change == "id":
        b.loc[1, "flight_id"] = 9
    if change == "label":
        b.loc[1, "y_true"] = 0
    if change == "date":
        b.loc[1, "flight_date"] = "2013-09-02"
    if change == "partition":
        b["partition"] = "test"
    if change == "missing":
        b.loc[1, "y_probability"] = np.nan
    with pytest.raises(ValueError):
        align_predictions({"a": a, "b": b}, "validation")


def test_alignment_reorders():
    a = frame()
    aligned = align_predictions({"a": a, "b": a.iloc[::-1]}, "validation")
    pd.testing.assert_frame_equal(aligned["a"], aligned["b"])


@pytest.mark.parametrize("method", ["raw", "sigmoid", "isotonic"])
def test_calibration_serialization(method, tmp_path):
    c = ScoreCalibrator(method).fit(
        [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], partition="train_calibration"
    )
    path = tmp_path / "c.joblib"
    joblib.dump(c, path)
    np.testing.assert_allclose(
        c.predict([0, 0.5, 1]), joblib.load(path).predict([0, 0.5, 1])
    )
    assert ((c.predict([0, 0.5, 1]) >= 0) & (c.predict([0, 0.5, 1]) <= 1)).all()


@pytest.mark.parametrize("partition", ["validation", "test", "train"])
def test_no_holdout_calibration(partition):
    with pytest.raises(ValueError):
        ScoreCalibrator("sigmoid").fit([0, 1], [0.1, 0.9], partition=partition)


@pytest.mark.parametrize(
    "windows",
    [
        (["2013-01-01"], ["2013-01-01"], ["2013-03-01"]),
        (["2013-02-01"], ["2013-01-01"], ["2013-03-01"]),
        (["2013-01-01"], ["2013-03-01"], ["2013-02-01"]),
        ([], ["2013-02-01"], ["2013-03-01"]),
        ([None], ["2013-02-01"], ["2013-03-01"]),
    ],
)
def test_chronology_rejected(windows):
    with pytest.raises(ValueError):
        check_windows(*windows)


def test_valid_windows_and_bins():
    check_windows(["2013-01-01"], ["2013-02-01"], ["2013-03-01"])
    bins = reliability([0, 1], [0, 1])
    assert bins["count"].sum() == 2
    assert bins["bin"].tolist() == [0, 9]


def test_thresholds():
    t = threshold_table([0, 1, 0, 1], [0.1, 0.9, 0.7, 0.4])
    choices = select_thresholds(t)
    assert choices["max_f1"]["f1"] == pytest.approx(0.8)
    for row in t.itertuples():
        assert evaluate([0, 1, 0, 1], [0.1, 0.9, 0.7, 0.4], row.threshold)[
            "f1"
        ] == pytest.approx(row.f1)
    with pytest.raises(ValueError):
        threshold_table([0, 1], [0, 1], partition="test")


def test_deterministic_bootstrap():
    args = (
        [0, 1, 0, 1],
        [0.1, 0.9, 0.7, 0.4],
        [0.2, 0.8, 0.3, 0.6],
        ["a", "a", "b", "b"],
    )
    assert paired_bootstrap(*args, n_boot=20) == paired_bootstrap(*args, n_boot=20)


def test_end_to_end(tmp_path, monkeypatch):
    from sklearn.dummy import DummyClassifier

    from flight_delay.evaluation import evaluate as runner

    n = 40
    ds = pd.Series(pd.date_range("2013-01-01", periods=n))
    df = pd.DataFrame(
        {
            "id": range(n),
            "year": ds.dt.year,
            "month": ds.dt.month,
            "day": ds.dt.day,
            "is_delayed": np.arange(n) % 2,
            "x": np.arange(n),
        }
    )
    train, val, test = df.iloc[:24], df.iloc[24:32], df.iloc[32:]
    monkeypatch.setattr(runner, "build_data_pipeline", lambda p: (train, val, test))
    monkeypatch.setattr(runner, "select_model_features", lambda f: f[["x"]])

    def fit(X, y, **kwargs):
        return DummyClassifier(strategy="prior").fit(X, y)

    monkeypatch.setattr(runner, "fit_logistic_baseline", fit)
    monkeypatch.setattr(runner, "fit_xgboost_baseline", fit)
    data = tmp_path / "mock.csv"
    df.to_csv(data, index=False)
    output = tmp_path / "run"
    manifest = runner.run_validation(data, output, n_boot=5)
    assert not (output / "test_metrics.json").exists()
    # Any fitting during finalization must fail this integration test.
    monkeypatch.setattr(
        ScoreCalibrator, "fit", lambda *args, **kwargs: pytest.fail("test fitting")
    )
    results = runner.run_final_test(data, output)
    assert all(r["n"] == 8 for r in results.values())
    assert json.loads((output / "selection.json").read_text()) == manifest
    with pytest.raises(ValueError, match="already"):
        runner.run_final_test(data, output)
    with pytest.raises(ValueError, match="Frozen"):
        runner.run_validation(data, output, n_boot=5)

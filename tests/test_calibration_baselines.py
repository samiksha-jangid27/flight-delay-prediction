"""Synthetic training-only prevalence and frozen validation scoring contracts."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from flight_delay.evaluation.baselines import fit_prevalence


def training(partition="base_train"):
    return pd.DataFrame(
        {
            "flight_date": ["2013-01-01"] * 4,
            "partition": [partition] * 4,
            "y_true": [0, 0, 0, 1],
        }
    )


def validation():
    return pd.DataFrame(
        {
            "flight_date": ["2013-02-01"] * 4,
            "partition": ["validation"] * 4,
            "y_true": [0, 1, 0, 1],
        }
    )


@pytest.mark.parametrize("partition", ["base_train", "train_calibration"])
def test_valid_training_constant(partition):
    fitted = fit_prevalence(training(partition), partition=partition)
    result = fitted.score_validation(validation())
    assert fitted.probability == 0.25 and fitted.n_fit == 4
    assert result["brier"] == brier_score_loss([0, 1, 0, 1], [0.25] * 4)
    assert result["log_loss"] == log_loss([0, 1, 0, 1], [0.25] * 4, labels=[0, 1])
    assert result["average_precision"] == 0.5


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_cannot_fit_heldout_labels(partition):
    with pytest.raises(ValueError, match="training"):
        fit_prevalence(training(partition), partition=partition)
    with pytest.raises(ValueError, match="tags"):
        fit_prevalence(training(partition), partition="base_train")


def test_heldout_changes_cannot_update_fitted_probability():
    # fit() receives only training; scoring cannot update the frozen dataclass.
    base = training()
    fitted = fit_prevalence(base, partition="base_train")
    heldout = validation()
    fitted.score_validation(heldout)
    heldout["y_true"] = 1 - heldout.y_true
    heldout["y_true"] = [1, 1, 1, 0]
    fitted.score_validation(heldout)
    assert fitted.probability == 0.25
    pd.testing.assert_frame_equal(base, training())
    with pytest.raises(TypeError):
        fit_prevalence(base, partition="base_train", test_labels=np.ones(4))
    with pytest.raises(TypeError):
        fit_prevalence(base, partition="base_train", validation_labels=np.ones(4))


@pytest.mark.parametrize(
    "mutation", ["overlap", "missing_date", "wrong_partition", "missing_label"]
)
def test_invalid_evaluation(mutation):
    fitted = fit_prevalence(training(), partition="base_train")
    frame = validation()
    if mutation == "overlap":
        frame["flight_date"] = "2013-01-01"
    if mutation == "missing_date":
        frame.loc[0, "flight_date"] = None
    if mutation == "wrong_partition":
        frame["partition"] = "test"
    if mutation == "missing_label":
        frame.loc[0, "y_true"] = np.nan
    with pytest.raises(ValueError):
        fitted.score_validation(frame)


@pytest.mark.parametrize("values", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_endpoint_training_prevalence(values):
    frame = training()
    frame["y_true"] = values
    result = fit_prevalence(frame, partition="base_train").score_validation(
        validation()
    )
    assert np.isfinite(result["log_loss"])


def test_empty_training_rejected():
    with pytest.raises(ValueError):
        fit_prevalence(training().iloc[:0], partition="base_train")


def test_analysis_never_opens_test_metrics_or_changes_frozen_files(tmp_path):
    import json

    from scripts.analyze_calibration_baselines import analyze
    from scripts.audit_evaluation import rebuild_partitions, sha256

    dates = pd.Series(pd.date_range("2013-01-01", periods=40).repeat(2)).reset_index(
        drop=True
    )
    raw = pd.DataFrame(
        {
            "id": range(len(dates)),
            "year": dates.dt.year,
            "month": dates.dt.month,
            "day": dates.dt.day,
            "arr_delay": np.tile([0, 30], 40),
        }
    )
    data = tmp_path / "flights.csv"
    raw.to_csv(data, index=False)
    parts = rebuild_partitions(data)
    output = tmp_path / "evaluation"
    output.mkdir()
    metadata = {
        "data_sha256": sha256(data),
        "windows": {
            k: {
                "n": len(parts[k]),
                "first": parts[k].flight_date.min(),
                "last": parts[k].flight_date.max(),
            }
            for k in ("base_train", "train_calibration")
        },
        "models": {"fixture": {"calibration": "raw"}},
    }
    manifest = output / "selection.json"
    manifest.write_text(json.dumps(metadata))
    for method in ("raw", "sigmoid", "isotonic"):
        frame = parts["validation"].assign(partition="validation")
        frame["y_probability"] = np.where(frame.y_true == 1, 0.8, 0.2)
        frame.to_csv(output / f"fixture_{method}_validation.csv", index=False)
    poison = output / "test_metrics.json"
    poison.write_text("invalid JSON: must never be opened")
    frozen = output / "frozen.joblib"
    frozen.write_bytes(b"original artifact: must never be loaded or changed")
    original = {p: p.read_bytes() for p in output.iterdir()}
    first = analyze(data, output)
    for path, content in original.items():
        assert path.read_bytes() == content
    # Alter test outcomes only, preserving dataset dates and eligibility.
    raw.loc[raw.index >= 68, "arr_delay"] = 30
    raw.to_csv(data, index=False)
    metadata["data_sha256"] = sha256(data)
    manifest.write_text(json.dumps(metadata))
    second = analyze(data, output)
    assert first["constants"] == second["constants"]
    assert first["models"] == second["models"]
    # Validation outcomes may change scores, never the fitted constants.
    raw.loc[(raw.index >= 56) & (raw.index < 68), "arr_delay"] = np.tile(
        [30, 30, 30, 0], 3
    )
    raw.to_csv(data, index=False)
    metadata["data_sha256"] = sha256(data)
    manifest.write_text(json.dumps(metadata))
    targets = rebuild_partitions(data)["validation"].set_index("flight_id").y_true
    for method in ("raw", "sigmoid", "isotonic"):
        prediction_path = output / f"fixture_{method}_validation.csv"
        prediction = pd.read_csv(
            prediction_path, dtype={"flight_id": "string"}, float_precision="round_trip"
        )
        prediction["y_true"] = prediction.flight_id.map(targets)
        prediction.to_csv(prediction_path, index=False)
    third = analyze(data, output)
    assert {k: v["fit"] for k, v in first["constants"].items()} == {
        k: v["fit"] for k, v in third["constants"].items()
    }
    assert (
        first["constants"]["base_train"]["validation"]
        != third["constants"]["base_train"]["validation"]
    )


def test_baseline_analysis_rejects_incomplete_validation_before_loading(tmp_path):
    from scripts.analyze_calibration_baselines import analyze

    (tmp_path / ".validation_started.json").write_text("{}")
    with pytest.raises(ValueError, match="Incomplete validation"):
        analyze(tmp_path / "not-opened.csv", tmp_path)

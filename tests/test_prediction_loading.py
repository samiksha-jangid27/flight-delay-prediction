"""Lossless prediction CSV boundaries, identifiers and input contracts."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.data.load import load_flights
from flight_delay.evaluation.predictions import load_prediction_csv

THRESHOLD = 0.34357316087308004


def frame():
    return pd.DataFrame(
        {
            "flight_id": ["001", "002", "010", "011"],
            "flight_date": ["2013-09-13"] * 4,
            "y_true": [0, 1, 1, 0],
            "y_probability": [0.1, THRESHOLD, THRESHOLD, 0.7],
            "partition": "validation",
        }
    )


def test_exact_isotonic_ties_and_string_ids_round_trip(tmp_path):
    original = frame()
    path = tmp_path / "predictions.csv"
    original.to_csv(path, index=False)
    restored = load_prediction_csv(path, partition="validation")
    assert restored.flight_id.tolist() == ["001", "002", "010", "011"]
    np.testing.assert_array_equal(
        restored.y_probability.to_numpy(), original.y_probability.to_numpy()
    )
    np.testing.assert_array_equal(
        restored.y_probability >= THRESHOLD, original.y_probability >= THRESHOLD
    )
    assert restored.y_probability.iloc[1] == THRESHOLD
    # The core loader must not rewrite its source.
    before = path.read_bytes()
    load_prediction_csv(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, np.inf])
def test_bad_probabilities(tmp_path, bad):
    f = frame()
    f.loc[0, "y_probability"] = bad
    path = tmp_path / "p.csv"
    f.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_prediction_csv(path, partition="validation")


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "missing_field",
        "missing_id",
        "bad_label",
        "bad_date",
        "bad_partition",
        "empty",
    ],
)
def test_invalid_prediction_schema(tmp_path, change):
    f = frame()
    if change == "duplicate":
        f.loc[1, "flight_id"] = "001"
    if change == "missing_field":
        f = f.drop(columns="y_true")
    if change == "missing_id":
        f.loc[1, "flight_id"] = None
    if change == "bad_label":
        f.loc[1, "y_true"] = 2
    if change == "bad_date":
        f.loc[1, "flight_date"] = "not-a-date"
    if change == "bad_partition":
        f.loc[1, "partition"] = "test"
    if change == "empty":
        f = f.iloc[:0]
    path = tmp_path / "p.csv"
    f.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_prediction_csv(path, partition="validation")


def test_legacy_four_column_notebook_predictions(tmp_path):
    path = tmp_path / "legacy.csv"
    frame().drop(columns="partition").to_csv(path, index=False)
    assert len(load_prediction_csv(path)) == 4
    with pytest.raises(ValueError, match="Missing"):
        load_prediction_csv(path, partition="validation")


def test_raw_dataset_loader_preserves_identifier_zeros(tmp_path):
    path = tmp_path / "flights.csv"
    pd.DataFrame({"id": ["001", "010"], "arr_delay": [0, 30]}).to_csv(path, index=False)
    assert load_flights(path).id.tolist() == ["001", "010"]

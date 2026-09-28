import pandas as pd
import pytest

from flight_delay.features.selection import (
    split_features_and_target,
)


def test_leakage_columns_are_removed():

    df = pd.DataFrame(
        {
            "carrier": ["AA", "UA"],
            "origin": ["JFK", "EWR"],
            "dest": ["MIA", "IAH"],
            "distance": [1000, 1200],
            "dep_time": [500, 600],
            "dep_delay": [10, 20],
            "arr_time": [800, 900],
            "arr_delay": [15, 30],
            "air_time": [120, 130],
            "is_delayed": [1, 1],
        }
    )

    X, y = split_features_and_target(df)

    assert "carrier" in X.columns
    assert "origin" in X.columns
    assert "dest" in X.columns
    assert "distance" in X.columns

    assert "dep_time" not in X.columns
    assert "dep_delay" not in X.columns
    assert "arr_time" not in X.columns
    assert "arr_delay" not in X.columns
    assert "air_time" not in X.columns
    assert "is_delayed" not in X.columns


def test_target_is_returned_separately():

    df = pd.DataFrame(
        {
            "carrier": ["AA", "UA"],
            "is_delayed": [1, 0],
        }
    )

    X, y = split_features_and_target(df)

    assert y.tolist() == [1, 0]
    assert "is_delayed" not in X.columns


def test_missing_target_raises_error():

    df = pd.DataFrame(
        {
            "carrier": ["AA", "UA"],
        }
    )

    with pytest.raises(ValueError):
        split_features_and_target(df)
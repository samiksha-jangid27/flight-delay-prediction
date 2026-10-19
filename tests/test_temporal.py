import numpy as np
import pandas as pd
import pytest

from flight_delay.features.temporal import add_temporal_features


def _sample() -> pd.DataFrame:
    # 2013-01-01 is a Tuesday; 2013-01-05 is a Saturday.
    return pd.DataFrame(
        {
            "year": [2013, 2013],
            "month": [1, 1],
            "day": [1, 5],
            "hour": [0, 23],
        }
    )


def test_day_of_week_and_weekend():
    result = add_temporal_features(_sample())

    # Tuesday = 1, Saturday = 5
    assert result["day_of_week"].tolist() == [1, 5]
    assert result["is_weekend"].tolist() == [0, 1]


def test_season_mapping():
    df = pd.DataFrame(
        {
            "year": [2013] * 4,
            "month": [1, 4, 7, 10],  # Jan, Apr, Jul, Oct
            "day": [1, 1, 1, 1],
            "hour": [8, 8, 8, 8],
        }
    )

    result = add_temporal_features(df)

    # winter, spring, summer, autumn
    assert result["season"].tolist() == [1, 2, 3, 4]


def test_cyclic_hour_wraps_around():
    result = add_temporal_features(_sample())

    # Hour 0 and hour 24 would be identical; hour 23 should be close to hour 0.
    h0 = result.iloc[0][["hour_sin", "hour_cos"]].to_numpy(dtype=float)
    h23 = result.iloc[1][["hour_sin", "hour_cos"]].to_numpy(dtype=float)

    # Distance on the unit circle between hour 0 and hour 23 is small.
    assert np.linalg.norm(h0 - h23) < 0.3

    # hour 0 encodes to (sin, cos) = (0, 1)
    assert np.isclose(h0[0], 0.0)
    assert np.isclose(h0[1], 1.0)


def test_cyclic_columns_exist():
    result = add_temporal_features(_sample())

    for col in (
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ):
        assert col in result.columns


def test_original_rows_preserved():
    df = _sample()
    result = add_temporal_features(df)

    assert len(result) == len(df)
    # Source frame is not mutated in place.
    assert "day_of_week" not in df.columns


def test_missing_columns_raise():
    df = pd.DataFrame({"year": [2013], "month": [1]})

    with pytest.raises(ValueError):
        add_temporal_features(df)

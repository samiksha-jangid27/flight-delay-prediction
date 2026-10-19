import numpy as np
import pandas as pd
import pytest

from flight_delay.features.route import (
    LONG_HAUL_MILES,
    add_route_features,
)


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["JFK", "LGA", "EWR"],
            "dest": ["MIA", "ORD", "SFO"],
            "distance": [1089, 733, 2565],
        }
    )


def test_route_identifier():
    result = add_route_features(_sample())

    assert result["route"].tolist() == ["JFK>MIA", "LGA>ORD", "EWR>SFO"]


def test_is_long_haul_threshold():
    result = add_route_features(_sample())

    expected = [
        1089 >= LONG_HAUL_MILES,
        733 >= LONG_HAUL_MILES,
        2565 >= LONG_HAUL_MILES,
    ]
    assert result["is_long_haul"].tolist() == [int(x) for x in expected]


def test_distance_log_is_monotonic():
    result = add_route_features(_sample())

    assert np.allclose(result["distance_log"], np.log1p([1089, 733, 2565]))


def test_distance_band_assigned():
    result = add_route_features(_sample())

    # 733 miles falls in the 500-750 band; 2565 falls in 2000+.
    assert result.loc[1, "distance_band"] == "500-750"
    assert result.loc[2, "distance_band"] == "2000+"
    assert result["distance_band"].notna().all()


def test_original_frame_not_mutated():
    df = _sample()
    add_route_features(df)

    assert "route" not in df.columns


def test_missing_columns_raise():
    df = pd.DataFrame({"origin": ["JFK"], "dest": ["MIA"]})

    with pytest.raises(ValueError):
        add_route_features(df)

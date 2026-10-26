import pandas as pd
import pytest

from flight_delay.features.congestion import add_congestion_features


def _frame():
    # Three EWR departures in hour 8 on the same date; one in hour 9.
    # Destinations/arrival hours chosen to exercise dest_hourly_arrivals.
    return pd.DataFrame(
        {
            "year": [2013] * 5,
            "month": [1] * 5,
            "day": [1, 1, 1, 1, 2],
            "origin": ["EWR", "EWR", "EWR", "EWR", "EWR"],
            "dest": ["ORD", "ORD", "MIA", "ORD", "ORD"],
            "hour": [8, 8, 8, 9, 8],
            "sched_arr_time": [1030, 1045, 1200, 1130, 1030],
        }
    )


def test_origin_hourly_departures_counts_same_hour_same_date_origin():
    out = add_congestion_features(_frame())
    # Day 1, EWR, hour 8 -> 3 flights; hour 9 -> 1 flight. Day 2 hour 8 -> 1.
    assert out["origin_hourly_departures"].tolist() == [3, 3, 3, 1, 1]


def test_dest_hourly_arrivals_counts_same_arrival_hour_same_date_dest():
    out = add_congestion_features(_frame())
    # Day 1: ORD arrivals in hour 10 -> rows 0,1 (count 2); MIA hour 12 -> 1;
    # ORD hour 11 -> 1. Day 2: ORD hour 10 -> 1.
    assert out["dest_hourly_arrivals"].tolist() == [2, 2, 1, 1, 1]


def test_features_are_nonnegative_integers():
    out = add_congestion_features(_frame())
    for col in ("origin_hourly_departures", "dest_hourly_arrivals"):
        assert (out[col] >= 1).all()
        assert out[col].dtype.kind == "i"


def test_does_not_mutate_input():
    df = _frame()
    before = df.copy()
    add_congestion_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        add_congestion_features(pd.DataFrame({"year": [2013]}))

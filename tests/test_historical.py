import pandas as pd

from flight_delay.features.historical import (
    add_historical_delay_features,
)


def test_same_day_observations_are_not_used():
    df = pd.DataFrame(
        {
            "year": [2013, 2013, 2013, 2013],
            "month": [1, 1, 1, 1],
            "day": [1, 1, 2, 2],
            "carrier": ["AA", "AA", "AA", "AA"],
            "origin": ["JFK", "JFK", "JFK", "JFK"],
            "dest": ["MIA", "MIA", "MIA", "MIA"],
            "is_delayed": [1, 0, 1, 0],
        }
    )

    result = add_historical_delay_features(df)

    # Jan 1 has no previous-day history.
    jan_1 = result[result["day"] == 1]

    assert jan_1["carrier_historical_flight_count"].tolist() == [0, 0]

    # For Jan 2, only Jan 1 counts:
    # 1 delayed out of 2 flights = 0.5
    jan_2 = result[result["day"] == 2]

    assert jan_2["carrier_historical_flight_count"].tolist() == [2, 2]
    assert jan_2["carrier_historical_delay_rate"].tolist() == [0.5, 0.5]


def test_history_grows_only_from_past_dates():
    df = pd.DataFrame(
        {
            "year": [2013, 2013, 2013],
            "month": [1, 1, 1],
            "day": [1, 2, 3],
            "carrier": ["AA", "AA", "AA"],
            "origin": ["JFK", "JFK", "JFK"],
            "dest": ["MIA", "MIA", "MIA"],
            "is_delayed": [1, 0, 1],
        }
    )

    result = add_historical_delay_features(df)

    jan_3 = result[result["day"] == 3].iloc[0]

    # Jan 1 + Jan 2 are the only historical observations.
    assert jan_3["carrier_historical_flight_count"] == 2
    assert jan_3["carrier_historical_delay_rate"] == 0.5


def test_same_day_and_future_outcomes_do_not_change_historical_features():
    df = pd.DataFrame(
        {
            "year": [2013] * 6,
            "month": [1] * 6,
            "day": [1, 1, 2, 2, 3, 3],
            "carrier": ["AA"] * 6,
            "origin": ["JFK"] * 6,
            "dest": ["MIA"] * 6,
            "is_delayed": [1, 0, 1, 0, 0, 1],
        }
    )

    changed = df.copy()
    # Change all outcomes for the final date. These labels must not affect
    # historical features on that date or on any earlier date.
    changed.loc[changed["day"] == 3, "is_delayed"] = [1, 1]

    baseline_features = add_historical_delay_features(df)
    changed_features = add_historical_delay_features(changed)

    historical_columns = [
        column
        for column in baseline_features.columns
        if column.endswith(("_historical_delay_rate", "_historical_flight_count"))
    ]

    pd.testing.assert_frame_equal(
        baseline_features[historical_columns],
        changed_features[historical_columns],
    )

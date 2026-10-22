"""Prior-date exclusion and rolling updates do not establish event-time availability."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from flight_delay.data.split import chronological_split
from flight_delay.features.historical import add_historical_delay_features
from flight_delay.features.model_features import HISTORICAL_FEATURES


@pytest.fixture
def flights():
    dates = pd.Series(pd.date_range("2013-01-01", periods=20).repeat(3))
    frame = pd.DataFrame(
        {
            "id": range(len(dates)),
            "year": dates.dt.year,
            "month": dates.dt.month,
            "day": dates.dt.day,
            "carrier": np.tile(["AA", "AA", "UA"], 20),
            "origin": np.tile(["JFK", "JFK", "EWR"], 20),
            "dest": np.tile(["MIA", "BOS", "MIA"], 20),
            "is_delayed": np.arange(len(dates)) % 2,
        }
    )
    return frame.sample(frac=1, random_state=42).reset_index(drop=True)


def history(frame):
    return add_historical_delay_features(frame).set_index("id").sort_index()


def test_all_groups_match_independent_strict_past_aggregation(flights):
    expected = flights.copy()
    expected["date"] = pd.to_datetime(expected[["year", "month", "day"]])
    expected["route"] = expected.origin + ">" + expected.dest
    result = history(flights)
    for row in expected.itertuples():
        for key, prefix in [
            ("carrier", "carrier"),
            ("origin", "origin"),
            ("dest", "destination"),
            ("route", "route"),
        ]:
            previous = expected[
                (expected.date < row.date) & (expected[key] == getattr(row, key))
            ]
            assert result.loc[row.id, prefix + "_historical_flight_count"] == len(
                previous
            )
            actual = result.loc[row.id, prefix + "_historical_delay_rate"]
            assert (
                pd.isna(actual)
                if previous.empty
                else actual == pytest.approx(previous.is_delayed.mean())
            )


@pytest.mark.parametrize("day", [1, 7, 14, 17, 20])
def test_current_future_outcomes_do_not_change_current_or_earlier_history(flights, day):
    original = history(flights)
    changed = flights.copy()
    changed.loc[changed.day >= day, "is_delayed"] = (
        1 - changed.loc[changed.day >= day, "is_delayed"]
    )
    updated = history(changed)
    ids = original.index[original.day <= day]
    pd.testing.assert_frame_equal(
        original.loc[ids, list(HISTORICAL_FEATURES)],
        updated.loc[ids, list(HISTORICAL_FEATURES)],
    )


def test_own_label_never_changes_own_features(flights):
    original = history(flights)
    for index in [0, 12, 30, 59]:
        changed = flights.copy()
        flight_id = changed.loc[index, "id"]
        changed.loc[index, "is_delayed"] = 1 - changed.loc[index, "is_delayed"]
        updated = history(changed)
        pd.testing.assert_series_equal(
            original.loc[flight_id, list(HISTORICAL_FEATURES)],
            updated.loc[flight_id, list(HISTORICAL_FEATURES)],
        )


def test_rolling_heldout_label_updates_only_subsequent_dates(flights):
    # 20-date split: train days 1..14, validation 15..17, test 18..20.
    original = history(flights)
    changed = flights.copy()
    changed.loc[changed.day == 16, "is_delayed"] = 1
    updated = history(changed)
    before = original.index[original.day <= 16]
    pd.testing.assert_frame_equal(
        original.loc[before, list(HISTORICAL_FEATURES)],
        updated.loc[before, list(HISTORICAL_FEATURES)],
    )
    assert not original.loc[original.day > 16, list(HISTORICAL_FEATURES)].equals(
        updated.loc[updated.day > 16, list(HISTORICAL_FEATURES)]
    )
    changed = flights.copy()
    changed.loc[changed.day == 18, "is_delayed"] = 0
    updated = history(changed)
    before = original.index[original.day <= 18]
    pd.testing.assert_frame_equal(
        original.loc[before, list(HISTORICAL_FEATURES)],
        updated.loc[before, list(HISTORICAL_FEATURES)],
    )
    assert not original.loc[original.day > 18, list(HISTORICAL_FEATURES)].equals(
        updated.loc[updated.day > 18, list(HISTORICAL_FEATURES)]
    )


def test_train_calibration_validation_test_dates_are_disjoint(flights):
    train, val, test = chronological_split(flights)
    date = lambda f: pd.to_datetime(f[["year", "month", "day"]])
    days = sorted(date(train).unique())
    cutoff = days[int(len(days) * 0.8)]
    base = train.loc[date(train) < cutoff]
    cal = train.loc[date(train) >= cutoff]
    windows = [base, cal, val, test]
    for left, right in pairwise(windows):
        assert date(left).max() < date(right).min()
        assert set(left.id).isdisjoint(right.id)
    assert sum(map(len, windows)) == len(flights)

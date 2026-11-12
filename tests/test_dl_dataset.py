"""DL adapter contract, train-only preprocessing and upstream leakage properties."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flight_delay.data.pipeline import build_data_pipeline
from flight_delay.deep_learning.dataset import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    build_dl_dataset,
)
from flight_delay.features.model_features import HISTORICAL_FEATURES


@pytest.fixture
def raw():
    dates = pd.Series(pd.date_range("2013-01-01", periods=40).repeat(2))
    return pd.DataFrame(
        {
            "id": [f"{i:04d}" for i in range(len(dates))],
            "year": dates.dt.year,
            "month": dates.dt.month,
            "day": dates.dt.day,
            "carrier": "AA",
            "origin": "JFK",
            "dest": "MIA",
            "distance": 1000.0,
            "sched_dep_time": 900,
            "sched_arr_time": 1200,
            "hour": 9,
            "minute": 0,
            "arr_delay": np.tile([0.0, 30.0], 40),
            "dep_time": 999,
            "dep_delay": 999,
            "arr_time": 999,
            "air_time": 999,
        }
    )


def save(tmp_path, raw):
    path = tmp_path / "flights.csv"
    raw.to_csv(path, index=False)
    return path


def test_schedule_and_route_exclude_outcome_history_on_the_same_rows(tmp_path, raw):
    path = save(tmp_path, raw)
    full = build_dl_dataset(path)
    route = build_dl_dataset(path, feature_set="route")
    assert set(full.numeric_features) - set(route.numeric_features) == set(HISTORICAL_FEATURES)
    assert route.categorical_features == full.categorical_features
    for name, partition in route.partitions.items():
        pd.testing.assert_frame_equal(partition.metadata, full.partitions[name].metadata)


def assert_alignment(result, original):
    combined_train = pd.concat(
        [
            result.partitions["train"].metadata,
            result.partitions["train_calibration"].metadata,
        ]
    )
    for metadata, frame in zip(
        [
            combined_train,
            result.partitions["validation"].metadata,
            result.partitions["test"].metadata,
        ],
        original,
    ):
        assert set(metadata.flight_id) == set(frame.id)
        if len(metadata.partition.unique()) == 1:
            assert metadata.flight_id.tolist() == frame.id.tolist()
        else:
            for name in ("train", "train_calibration"):
                actual = metadata.loc[metadata.partition == name, "flight_id"].tolist()
                assert actual == frame.loc[frame.id.isin(actual), "id"].tolist()
        frame = frame.set_index("id").loc[metadata.flight_id].reset_index()
        np.testing.assert_array_equal(metadata.y_true, frame.is_delayed)
        np.testing.assert_array_equal(
            metadata.flight_date, pd.to_datetime(frame[["year", "month", "day"]])
        )
    windows = [p.metadata.flight_date for p in result.partitions.values()]
    assert all(left.max() < right.min() for left, right in pairwise(windows))
    ids = pd.concat([p.metadata.flight_id for p in result.partitions.values()])
    assert ids.is_unique


def test_schema_alignment_reproducibility_and_no_raw_outcomes(tmp_path, raw):
    path = save(tmp_path, raw.sample(frac=1, random_state=7))
    result = build_dl_dataset(path)
    again = build_dl_dataset(path)
    assert_alignment(result, build_data_pipeline(path))
    assert len(FEATURES) == 31
    assert len(NUMERIC) == 26 and len(CATEGORICAL) == 5
    assert {"carrier", "origin", "dest"} <= set(CATEGORICAL)
    assert not {
        "id",
        "is_delayed",
        "arr_delay",
        "dep_delay",
        "dep_time",
        "arr_time",
        "air_time",
        "origin_hourly_departures",
        "dest_hourly_arrivals",
    } & set(FEATURES)
    for name, part in result.partitions.items():
        assert part.numeric.shape == (len(part.labels), 26)
        assert part.categorical.shape == (len(part.labels), 5)
        assert part.numeric.dtype == np.float32
        assert part.categorical.dtype == np.int64
        assert np.isfinite(part.numeric).all()
        assert set(part.metadata.partition) == {name}
        np.testing.assert_array_equal(part.numeric, again.partitions[name].numeric)
        np.testing.assert_array_equal(
            part.categorical, again.partitions[name].categorical
        )
        pd.testing.assert_frame_equal(part.metadata, again.partitions[name].metadata)
        for i, f in enumerate(CATEGORICAL):
            assert part.categorical[:, i].max() < result.embedding_sizes[f]
    assert result.partitions["train"].metadata.flight_date.nunique() == 22
    assert result.partitions["train_calibration"].metadata.flight_date.nunique() == 6


def test_missing_values_unseen_categories_and_train_only_fit(tmp_path, raw):
    raw.loc[:, "distance"] = np.nan  # all-missing numeric fallback
    raw.loc[0, "carrier"] = None
    raw.loc[raw.index >= 44, "carrier"] = "UNSEEN"
    raw.loc[raw.index >= 44, "sched_dep_time"] = 2300
    result = build_dl_dataset(save(tmp_path, raw))
    assert result.medians["distance"] == 0
    assert result.means["sched_dep_time"] == 900
    assert result.scales["sched_dep_time"] == 1
    assert "UNSEEN" not in result.vocabularies["carrier"]
    carrier = CATEGORICAL.index("carrier")
    assert result.partitions["train"].categorical[0, carrier] == 0
    assert (result.partitions["train_calibration"].categorical[:, carrier] == 1).all()
    for part in result.partitions.values():
        assert np.isfinite(part.numeric).all()


@pytest.mark.parametrize(
    "kind", ["duplicate", "missing_id", "date", "schema", "infinity"]
)
def test_invalid_inputs_fail(tmp_path, raw, kind):
    if kind == "duplicate":
        raw.loc[79, "id"] = raw.loc[0, "id"]
    elif kind == "missing_id":
        raw.loc[0, "id"] = None
    elif kind == "date":
        raw.loc[0, "month"] = 13
    elif kind == "schema":
        raw = raw.drop(columns="distance")
    else:
        raw.loc[0, "distance"] = np.inf
    with pytest.raises(ValueError):
        build_dl_dataset(save(tmp_path, raw))


def test_undefined_outcomes_excluded_without_other_row_loss(tmp_path, raw):
    raw.loc[0, "arr_delay"] = np.nan
    path = save(tmp_path, raw)
    result = build_dl_dataset(path)
    assert_alignment(result, build_data_pipeline(path))
    assert sum(len(p.labels) for p in result.partitions.values()) == len(raw) - 1
    assert "0000" not in set(
        pd.concat([p.metadata.flight_id for p in result.partitions.values()])
    )


def test_same_day_future_outcomes_and_post_departure_values_cannot_change_inputs(
    tmp_path, raw
):
    path = save(tmp_path, raw)
    before = build_dl_dataset(path)
    # Perturb all outcomes from day 32 onward, including current-day peers.
    cutoff = pd.Timestamp("2013-02-01")
    dates = pd.to_datetime(raw[["year", "month", "day"]])
    raw.loc[dates >= cutoff, "arr_delay"] = 100
    raw.loc[:, ["dep_delay", "dep_time", "arr_time", "air_time"]] = -999
    after = build_dl_dataset(save(tmp_path, raw))
    for name, part in before.partitions.items():
        mask = part.metadata.flight_date <= cutoff
        np.testing.assert_array_equal(
            part.numeric[mask], after.partitions[name].numeric[mask]
        )
        np.testing.assert_array_equal(
            part.categorical[mask], after.partitions[name].categorical[mask]
        )
    # Later dates DO receive rolling history, an intentional availability limitation.
    index = NUMERIC.index(HISTORICAL_FEATURES[0])
    assert not np.array_equal(
        before.partitions["test"].numeric[:, index],
        after.partitions["test"].numeric[:, index],
    )


def test_real_population_and_original_partitions():
    path = Path("data/raw/flights.csv")
    if not path.exists():
        pytest.skip("Local real dataset unavailable; synthetic adapter tests still run")
    result = build_dl_dataset(path)
    assert_alignment(result, build_data_pipeline(path))
    assert sum(len(p.labels) for p in result.partitions.values()) == 327_346


@pytest.mark.parametrize("kind", ["label", "missing_feature", "overlap", "empty"])
def test_rejects_corrupt_upstream_contract(tmp_path, raw, monkeypatch, kind):
    from flight_delay.deep_learning import dataset

    path = save(tmp_path, raw)
    train, val, test = build_data_pipeline(path)
    if kind == "label":
        val.loc[val.index[0], "is_delayed"] = np.nan
    elif kind == "missing_feature":
        val = val.drop(columns="hour_sin")
    elif kind == "overlap":
        val.loc[:, ["year", "month", "day"]] = [2013, 1, 1]
    else:
        test = test.iloc[:0]
    monkeypatch.setattr(dataset, "build_data_pipeline", lambda _: (train, val, test))
    with pytest.raises(ValueError):
        build_dl_dataset(path)


def test_insufficient_training_dates_fail(tmp_path, raw):
    with pytest.raises(ValueError):
        build_dl_dataset(save(tmp_path, raw.iloc[:4]))

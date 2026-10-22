import pandas as pd
import pytest

from flight_delay.data.split import chronological_split


def test_chronological_split():
    df = pd.DataFrame(
        {
            "year": [2013] * 10,
            "month": [1] * 10,
            "day": list(range(1, 11)),
            "value": range(10),
        }
    )

    train, validation, test = chronological_split(
        df,
        train_ratio=0.6,
        validation_ratio=0.2,
    )

    train_last = pd.to_datetime(
        train[["year", "month", "day"]]
    ).max()

    validation_first = pd.to_datetime(
        validation[["year", "month", "day"]]
    ).min()

    validation_last = pd.to_datetime(
        validation[["year", "month", "day"]]
    ).max()

    test_first = pd.to_datetime(
        test[["year", "month", "day"]]
    ).min()

    assert train_last < validation_first
    assert validation_last < test_first


def test_split_preserves_all_rows():
    df = pd.DataFrame(
        {
            "year": [2013] * 10,
            "month": [1] * 10,
            "day": list(range(1, 11)),
        }
    )

    train, validation, test = chronological_split(
        df,
        train_ratio=0.6,
        validation_ratio=0.2,
    )

    assert len(train) + len(validation) + len(test) == len(df)


def test_invalid_ratios():
    df = pd.DataFrame(
        {
            "year": [2013],
            "month": [1],
            "day": [1],
        }
    )

    with pytest.raises(ValueError):
        chronological_split(
            df,
            train_ratio=0.8,
            validation_ratio=0.3,
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("year", None),
        ("month", float("nan")),
        ("day", pd.NaT),
        ("month", 13),
        ("day", 32),
        ("day", 0),
        ("year", "not-a-year"),
        ("month", 1.5),
        ("day", float("inf")),
    ],
)
def test_invalid_date_components_rejected(column, value):
    df = pd.DataFrame(
        {"year": [2013] * 10, "month": [1] * 10, "day": list(range(1, 11))}
    ).astype(object)
    df.loc[0, column] = value
    with pytest.raises(ValueError, match="date"):
        chronological_split(df)


def test_valid_split_matches_original_date_rule_and_preserves_whole_dates():
    dates = (
        pd.Series(pd.date_range("2013-01-01", periods=20).repeat(3))
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    df = pd.DataFrame(
        {
            "year": dates.dt.year,
            "month": dates.dt.month,
            "day": dates.dt.day,
            "id": range(len(dates)),
        }
    )
    train, val, test = chronological_split(df)
    days = sorted(dates.unique())
    n = len(days)
    expected = [
        days[: int(n * 0.7)],
        days[int(n * 0.7) : int(n * (0.7 + 0.15))],
        days[int(n * (0.7 + 0.15)) :],
    ]
    for actual, window in zip([train, val, test], expected):
        pd.testing.assert_frame_equal(actual, df.loc[dates.isin(window)])
    sets = [
        set(pd.to_datetime(f[["year", "month", "day"]])) for f in (train, val, test)
    ]
    assert sets[0].isdisjoint(sets[1] | sets[2]) and sets[1].isdisjoint(sets[2])

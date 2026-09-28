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
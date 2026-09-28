import pandas as pd
import pytest

from flight_delay.features.target import create_delay_target


def test_delay_target():
    df = pd.DataFrame(
        {
            "arr_delay": [-5, 10, 15, 30, None],
        }
    )

    result = create_delay_target(df, threshold_minutes=15)

    assert len(result) == 4
    assert result["is_delayed"].tolist() == [0, 0, 1, 1]


def test_missing_arr_delay_is_removed():
    df = pd.DataFrame(
        {
            "arr_delay": [10, None, 20],
        }
    )

    result = create_delay_target(df)

    assert result["arr_delay"].notna().all()


def test_missing_arr_delay_column():
    df = pd.DataFrame({"dep_delay": [1, 2, 3]})

    with pytest.raises(ValueError):
        create_delay_target(df)
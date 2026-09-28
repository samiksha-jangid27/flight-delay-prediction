from pathlib import Path

from flight_delay.data.prepare import prepare_model_data


DATA_PATH = Path("data/raw/flights.csv")


def test_prepare_model_data():

    data = prepare_model_data(DATA_PATH)

    X_train, y_train = data["train"]
    X_validation, y_validation = data["validation"]
    X_test, y_test = data["test"]

    assert len(X_train) == len(y_train)
    assert len(X_validation) == len(y_validation)
    assert len(X_test) == len(y_test)

    assert len(X_train) > 0
    assert len(X_validation) > 0
    assert len(X_test) > 0


def test_no_target_or_leakage_columns():

    data = prepare_model_data(DATA_PATH)

    X_train, _ = data["train"]

    forbidden = {
        "is_delayed",
        "arr_delay",
        "dep_delay",
        "dep_time",
        "arr_time",
        "air_time",
    }

    assert forbidden.isdisjoint(X_train.columns)
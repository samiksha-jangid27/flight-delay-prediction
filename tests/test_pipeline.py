from pathlib import Path

from flight_delay.data.pipeline import build_data_pipeline


DATA_PATH = Path("data/raw/flights.csv")


def test_pipeline_runs():
    train, validation, test = build_data_pipeline(DATA_PATH)

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0

    assert "is_delayed" in train.columns
    assert "is_delayed" in validation.columns
    assert "is_delayed" in test.columns


def test_pipeline_preserves_row_count_after_target_filter():
    train, validation, test = build_data_pipeline(DATA_PATH)

    total = len(train) + len(validation) + len(test)

    # 9,430 rows have missing arrival delay and are excluded
    # from the classification population.
    assert total == 327_346
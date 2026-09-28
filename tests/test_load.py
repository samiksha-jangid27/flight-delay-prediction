from pathlib import Path

from flight_delay.data.load import load_flights


DATA_PATH = Path("data/raw/flights.csv")


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_load_flights():
    df = load_flights(DATA_PATH)

    assert not df.empty
    assert len(df.columns) > 0
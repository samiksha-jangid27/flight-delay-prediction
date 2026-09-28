from __future__ import annotations

from pathlib import Path

import pandas as pd

from flight_delay.data.load import load_flights
from flight_delay.data.split import chronological_split
from flight_delay.features.historical import add_historical_delay_features
from flight_delay.features.target import create_delay_target


def build_data_pipeline(
    path: str | Path,
    target_threshold: float = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the leakage-safe flight classification dataset.

    Steps:
    1. Load raw flights.
    2. Create the binary delay target.
    3. Add leakage-safe historical features.
    4. Split chronologically into train, validation, and test.
    """

    df = load_flights(path)

    df = create_delay_target(
        df,
        threshold_minutes=target_threshold,
    )

    df = add_historical_delay_features(df)

    train, validation, test = chronological_split(df)

    return train, validation, test

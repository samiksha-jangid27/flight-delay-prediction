from __future__ import annotations

from pathlib import Path

import pandas as pd

from flight_delay.data.load import load_flights
from flight_delay.data.split import chronological_split
from flight_delay.features.congestion import add_congestion_features
from flight_delay.features.historical import add_historical_delay_features
from flight_delay.features.route import add_route_features
from flight_delay.features.target import create_delay_target
from flight_delay.features.temporal import add_temporal_features


def build_data_pipeline(
    path: str | Path,
    target_threshold: float = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the flight-delay dataset with engineered features.

    Steps:
    1. Load raw flight data.
    2. Add schedule-based airport-congestion features on the COMPLETE timetable
       (before any rows are dropped), so counts reflect every scheduled flight,
       including those whose arrival outcome is later undefined.
    3. Create the binary delay target (drops rows with undefined arrival outcome).
    4. Add temporal features.
    5. Add route features.
    6. Add historical features using only previous dates.
    7. Split chronologically into train, validation, and test.
    """

    df = load_flights(path)

    # Congestion is a property of the published schedule, so it must be counted
    # over the full timetable BEFORE create_delay_target removes undefined-outcome
    # rows; otherwise cancelled/diverted flights would silently drop out of the
    # scheduled-load counts.
    df = add_congestion_features(df)

    df = create_delay_target(
        df,
        threshold_minutes=target_threshold,
    )

    df = add_temporal_features(df)
    df = add_route_features(df)
    df = add_historical_delay_features(df)

    train, validation, test = chronological_split(df)

    return train, validation, test
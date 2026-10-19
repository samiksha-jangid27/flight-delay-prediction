from __future__ import annotations

import numpy as np
import pandas as pd


# Periods used for cyclic (sin/cos) encodings.
HOUR_PERIOD = 24
MONTH_PERIOD = 12
WEEKDAY_PERIOD = 7


def _cyclic_encode(
    df: pd.DataFrame,
    column: str,
    period: int,
) -> pd.DataFrame:
    """
    Add sine and cosine columns for a cyclic integer feature.

    A cyclic encoding keeps neighbouring values close in feature space
    (for example hour 23 and hour 0), which a plain integer cannot.
    """

    result = df.copy()

    radians = 2 * np.pi * result[column] / period

    result[f"{column}_sin"] = np.sin(radians)
    result[f"{column}_cos"] = np.cos(radians)

    return result


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-safe temporal features derived from the schedule only.

    All features come from the calendar date (year/month/day) and the
    scheduled departure hour. No post-departure information is used.

    Added features:
    - day_of_week          (0 = Monday ... 6 = Sunday)
    - is_weekend           (1 if Saturday or Sunday)
    - season               (1 = winter, 2 = spring, 3 = summer, 4 = autumn)
    - hour_sin / hour_cos          cyclic scheduled-hour encoding
    - month_sin / month_cos        cyclic month encoding
    - day_of_week_sin / day_of_week_cos  cyclic weekday encoding
    """

    required = {"year", "month", "day", "hour"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    flight_date = pd.to_datetime(
        {
            "year": result["year"],
            "month": result["month"],
            "day": result["day"],
        }
    )

    result["day_of_week"] = flight_date.dt.dayofweek.astype("int8")
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int8")

    # Meteorological season from month: keeps a slow calendar signal
    # (holiday travel, winter weather) that month_sin/cos smooths over.
    result["season"] = (
        result["month"] % 12 // 3 + 1
    ).astype("int8")

    result = _cyclic_encode(result, "hour", HOUR_PERIOD)
    result = _cyclic_encode(result, "month", MONTH_PERIOD)
    result = _cyclic_encode(result, "day_of_week", WEEKDAY_PERIOD)

    return result

from __future__ import annotations

import pandas as pd

"""Airport congestion features derived purely from the published schedule.

These count how many flights are *scheduled* to use an airport in the same hour,
which is known before any flight departs (it is a property of the timetable, not
of any outcome). They therefore introduce no leakage and directly represent the
"airport congestion" signal requested in the project brief, as a time-specific
scheduled load rather than only a historical volume proxy.
"""


def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add schedule-based airport-congestion features.

    Added features:
    - origin_hourly_departures : flights scheduled to depart the same origin in
      the same (calendar date, scheduled-departure hour).
    - dest_hourly_arrivals     : flights scheduled to arrive at the same
      destination in the same (calendar date, scheduled-arrival hour).

    Both are counted from the timetable only (origin/dest, date, and the
    scheduled departure/arrival hours), so every value is available before
    departure and no post-departure or outcome information is used.
    """

    required = {"year", "month", "day", "origin", "dest", "hour", "sched_arr_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()

    # Scheduled arrival hour from HHMM (e.g. 819 -> 8, 2359 -> 23).
    arr_hour = (
        pd.to_numeric(result["sched_arr_time"], errors="coerce") // 100
    ).clip(lower=0, upper=23)

    result["origin_hourly_departures"] = (
        result.groupby(["year", "month", "day", "origin", "hour"])["hour"]
        .transform("size")
        .astype("int32")
    )

    result["_arr_hour"] = arr_hour
    result["dest_hourly_arrivals"] = (
        result.groupby(["year", "month", "day", "dest", "_arr_hour"])["_arr_hour"]
        .transform("size")
        .astype("int32")
    )
    result.drop(columns="_arr_hour", inplace=True)

    return result

from __future__ import annotations

import pandas as pd


HISTORICAL_KEYS = {
    "carrier": "carrier",
    "origin": "origin",
    "destination": "dest",
}


def _add_date(df: pd.DataFrame) -> pd.DataFrame:
    """Create a calendar-date column from year, month, and day."""

    result = df.copy()

    result["_flight_date"] = pd.to_datetime(
        {
            "year": result["year"],
            "month": result["month"],
            "day": result["day"],
        }
    )

    return result


def _historical_group_features(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    feature_prefix: str,
) -> pd.DataFrame:
    """
    Calculate historical delay rate and historical observation count.

    Only previous calendar dates are used. Current-day observations are
    excluded from the historical statistics.
    """

    daily = (
        df.groupby(["_flight_date", group_col], observed=True)[target_col]
        .agg(
            delayed_count="sum",
            total_count="count",
        )
        .reset_index()
    )

    grouped = daily.groupby(group_col, observed=True)

    daily["past_delayed_count"] = (
        grouped["delayed_count"].cumsum() - daily["delayed_count"]
    )

    daily["past_total_count"] = (
        grouped["total_count"].cumsum() - daily["total_count"]
    )

    daily[f"{feature_prefix}_historical_delay_rate"] = (
        daily["past_delayed_count"]
        / daily["past_total_count"]
    )

    daily[f"{feature_prefix}_historical_flight_count"] = (
        daily["past_total_count"]
    )

    return daily[
        [
            "_flight_date",
            group_col,
            f"{feature_prefix}_historical_delay_rate",
            f"{feature_prefix}_historical_flight_count",
        ]
    ]


def add_historical_delay_features(
    df: pd.DataFrame,
    target_col: str = "is_delayed",
) -> pd.DataFrame:
    """
    Add leakage-safe historical delay features.

    Historical statistics are computed from previous calendar dates only.

    Added features:
    - carrier historical delay rate/count
    - origin historical delay rate/count
    - destination historical delay rate/count
    - route historical delay rate/count
    """

    required = {
        "year",
        "month",
        "day",
        "carrier",
        "origin",
        "dest",
        target_col,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = _add_date(df)

    result["_route"] = (
        result["origin"].astype(str)
        + ">"
        + result["dest"].astype(str)
    )

    group_specs = [
        ("carrier", "carrier"),
        ("origin", "origin"),
        ("dest", "destination"),
        ("_route", "route"),
    ]

    for group_col, prefix in group_specs:
        historical = _historical_group_features(
            result,
            group_col=group_col,
            target_col=target_col,
            feature_prefix=prefix,
        )

        result = result.merge(
            historical,
            on=["_flight_date", group_col],
            how="left",
        )

    result.drop(
        columns=["_flight_date", "_route"],
        inplace=True,
    )

    return result
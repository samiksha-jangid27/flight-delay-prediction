"""Shared, pre-departure feature contract for AML classification models.

Both Logistic Regression and XGBoost import this module; neither classifier
needs to import the other. The allowlist blocks accidental admission of
post-departure columns or raw labels into the design matrix.
"""
from __future__ import annotations

import pandas as pd

SCHEDULE_FEATURES: tuple[str, ...] = (
    "month", "day", "sched_dep_time", "sched_arr_time", "hour", "minute",
    "carrier", "origin", "dest", "distance", "day_of_week", "is_weekend",
    "season", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos",
)
ROUTE_FEATURES: tuple[str, ...] = (
    "route", "distance_band", "is_long_haul", "distance_log",
)
HISTORICAL_FEATURES: tuple[str, ...] = (
    "carrier_historical_delay_rate", "carrier_historical_flight_count",
    "origin_historical_delay_rate", "origin_historical_flight_count",
    "destination_historical_delay_rate", "destination_historical_flight_count",
    "route_historical_delay_rate", "route_historical_flight_count",
)
CONGESTION_FEATURES: tuple[str, ...] = (
    "origin_hourly_departures", "dest_hourly_arrivals",
)
# `full` is the canonical 31-feature model for the frozen evaluation.
# `full_congestion` (33 features) is used only in a validation-only ablation;
# it has no held-out test score.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "schedule": SCHEDULE_FEATURES,
    "route": SCHEDULE_FEATURES + ROUTE_FEATURES,
    "full": SCHEDULE_FEATURES + ROUTE_FEATURES + HISTORICAL_FEATURES,
    "full_congestion": (
        SCHEDULE_FEATURES + ROUTE_FEATURES + HISTORICAL_FEATURES + CONGESTION_FEATURES
    ),
}
CATEGORICAL_FEATURES: frozenset[str] = frozenset(
    {"carrier", "origin", "dest", "route", "distance_band"}
)


def select_model_features(X: pd.DataFrame, feature_set: str = "full") -> pd.DataFrame:
    """Select only approved predictors and normalize dtypes without mutating X.

    Raw outcomes, flight IDs, tail numbers and actual arrival/departure values
    can be present in the input frame but will never enter the output frame.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; expected {list(FEATURE_SETS)}")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with named columns")

    fields = FEATURE_SETS[feature_set]
    missing = [field for field in fields if field not in X.columns]
    if missing:
        raise ValueError(f"Required {feature_set!r} features missing: {missing}")

    selected = X.loc[:, list(fields)].copy()
    for field in fields:
        if field in CATEGORICAL_FEATURES:
            selected[field] = selected[field].astype("object")
        else:
            selected[field] = pd.to_numeric(selected[field], errors="raise")
    return selected

from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "is_delayed"

EXCLUDED_COLUMNS = {
    # Target/outcome
    "is_delayed",
    "arr_delay",

    # Post-departure information
    "dep_time",
    "dep_delay",
    "arr_time",
    "air_time",

    # Identifier / redundant columns
    "id",
    "name",
    "year",
}


def split_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features from the target.

    Only pre-departure candidate features are returned in X.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    y = df[TARGET_COLUMN].copy()

    columns_to_drop = [
        column
        for column in EXCLUDED_COLUMNS
        if column in df.columns
    ]

    X = df.drop(columns=columns_to_drop).copy()

    return X, y
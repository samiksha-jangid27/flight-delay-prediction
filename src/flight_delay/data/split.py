import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split flight records chronologically by calendar date.

    The earliest dates are assigned to training,
    the next dates to validation, and the latest dates to testing.
    """

    required_columns = {"year", "month", "day"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required date columns: {sorted(missing)}"
        )

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    result = df.copy()

    result["_flight_date"] = pd.to_datetime(
        {
            "year": result["year"],
            "month": result["month"],
            "day": result["day"],
        }
    )

    unique_dates = (
        result["_flight_date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n_dates = len(unique_dates)

    train_end = int(n_dates * train_ratio)
    validation_end = int(
        n_dates * (train_ratio + validation_ratio)
    )

    train_dates = unique_dates.iloc[:train_end]
    validation_dates = unique_dates.iloc[train_end:validation_end]
    test_dates = unique_dates.iloc[validation_end:]

    train = result[result["_flight_date"].isin(train_dates)].copy()
    validation = result[
        result["_flight_date"].isin(validation_dates)
    ].copy()
    test = result[
        result["_flight_date"].isin(test_dates)
    ].copy()

    train.drop(columns="_flight_date", inplace=True)
    validation.drop(columns="_flight_date", inplace=True)
    test.drop(columns="_flight_date", inplace=True)

    return train, validation, test
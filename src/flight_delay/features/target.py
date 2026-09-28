import pandas as pd


def create_delay_target(
    df: pd.DataFrame,
    threshold_minutes: float = 15,
) -> pd.DataFrame:
    """
    Create a binary arrival-delay target.

    Rows with missing arrival delay are excluded because their
    arrival-delay outcome is undefined.
    """

    if "arr_delay" not in df.columns:
        raise ValueError("Dataset must contain 'arr_delay'.")

    result = df.loc[df["arr_delay"].notna()].copy()

    result["is_delayed"] = (
        result["arr_delay"] >= threshold_minutes
    ).astype("int8")

    return result
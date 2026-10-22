"""Lossless saved-prediction ingestion; thresholds must be applied without rounding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prediction_csv(
    path: str | Path, *, partition: str | None = None
) -> pd.DataFrame:
    """Read exact binary floats and string IDs, validating the prediction schema.

    Core columns are flight_id, flight_date, y_true, y_probability. The original
    model notebooks use this four-column format. Passing ``partition`` requires
    an explicit matching partition column, as in frozen evaluation artifacts.
    Never round the returned probabilities or a frozen decision threshold.
    """
    frame = pd.read_csv(
        path, dtype={"flight_id": "string"}, float_precision="round_trip"
    )
    required = {"flight_id", "flight_date", "y_true", "y_probability"}
    if partition is not None:
        required.add("partition")
    if not required <= set(frame) or frame.empty:
        raise ValueError(f"Missing prediction fields or empty CSV: {path}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Prediction columns contain missing values")
    if frame.flight_id.duplicated().any() or frame.flight_id.str.strip().eq("").any():
        raise ValueError("Flight identifiers must be nonempty and unique")
    dates = pd.to_datetime(frame.flight_date, format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or not frame.y_true.isin([0, 1]).all():
        raise ValueError("Prediction dates and binary labels must be valid")
    probabilities = pd.to_numeric(frame.y_probability, errors="raise").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if partition is not None and not frame.partition.eq(partition).all():
        raise ValueError(f"Prediction partition must equal {partition!r}")
    if "partition" in frame and (
        frame.partition.isna().any() or frame.partition.nunique() != 1
    ):
        raise ValueError("Prediction CSV must contain exactly one nonmissing partition")
    return frame

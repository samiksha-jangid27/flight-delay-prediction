from __future__ import annotations

from pathlib import Path

import pandas as pd

from flight_delay.data.pipeline import build_data_pipeline
from flight_delay.features.selection import split_features_and_target


def prepare_model_data(
    path: str | Path,
    target_threshold: float = 15,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Build the complete leakage-safe model-ready dataset.

    Returns:
        Dictionary containing train, validation, and test
        feature/target pairs.
    """

    train, validation, test = build_data_pipeline(
        path,
        target_threshold=target_threshold,
    )

    X_train, y_train = split_features_and_target(train)
    X_validation, y_validation = split_features_and_target(validation)
    X_test, y_test = split_features_and_target(test)

    return {
        "train": (X_train, y_train),
        "validation": (X_validation, y_validation),
        "test": (X_test, y_test),
    }
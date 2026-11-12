"""Train-fitted NumPy inputs for DL; no model fitting or tensor-framework dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flight_delay.data.pipeline import build_data_pipeline
from flight_delay.features.model_features import (
    CATEGORICAL_FEATURES,
    FEATURE_SETS,
    select_model_features,
)

FEATURES = FEATURE_SETS["full"]
CATEGORICAL = tuple(f for f in FEATURES if f in CATEGORICAL_FEATURES)
NUMERIC = tuple(f for f in FEATURES if f not in CATEGORICAL_FEATURES)


@dataclass
class DLPartition:
    """Position-aligned arrays and comparison metadata, in upstream row order."""

    numeric: np.ndarray
    categorical: np.ndarray
    metadata: pd.DataFrame

    @property
    def labels(self) -> np.ndarray:
        return self.metadata["y_true"].to_numpy(dtype=np.int64, copy=True)


@dataclass
class DLDataset:
    partitions: dict[str, DLPartition]
    medians: pd.Series
    means: pd.Series
    scales: pd.Series
    vocabularies: dict[str, dict[str, int]]
    numeric_features: tuple[str, ...] = NUMERIC
    categorical_features: tuple[str, ...] = CATEGORICAL

    @property
    def embedding_sizes(self) -> dict[str, int]:
        """0 = missing, 1 = unseen, known training values start at 2."""
        return {f: len(v) + 2 for f, v in self.vocabularies.items()}


def build_dl_dataset(path: str | Path, *, feature_set: str = "full") -> DLDataset:
    """Reuse Phase 1 population, target (>=15m), and whole-date partitions.

    Split only original training into early train / late train_calibration,
    exactly as Phase 1 evaluation does. Fit all transformations on early train.
    Missing numeric values use its median (zero for all-missing columns), then
    population-standardize; constant columns have scale one. Fail on infinity.
    Use `route` for the schedule-and-route comparison without historical outcomes.
    """
    if feature_set not in ("full", "route"):
        raise ValueError("DL feature_set must be 'full' or 'route'")
    numeric_features = tuple(
        field for field in FEATURE_SETS[feature_set] if field not in CATEGORICAL_FEATURES
    )
    train, validation, test = build_data_pipeline(path)
    days = sorted(pd.to_datetime(train[["year", "month", "day"]]).unique())
    if len(days) < 2:
        raise ValueError("Need at least two training dates for calibration")
    cutoff = days[int(len(days) * 0.8)]
    dates = pd.to_datetime(train[["year", "month", "day"]])
    frames = {
        "train": train.loc[dates < cutoff],
        "train_calibration": train.loc[dates >= cutoff],
        "validation": validation,
        "test": test,
    }
    selected = {}
    metadata = {}
    previous_max = None
    for name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"Empty partition: {name}")
        flight_dates = pd.to_datetime(frame[["year", "month", "day"]], errors="raise")
        if flight_dates.isna().any() or (
            previous_max is not None and previous_max >= flight_dates.min()
        ):
            raise ValueError(f"Invalid chronological boundary: {name}")
        previous_max = flight_dates.max()
        ids = frame["id"].astype("string")
        if ids.isna().any() or ids.str.strip().eq("").any():
            raise ValueError(f"Missing flight IDs: {name}")
        if not frame["is_delayed"].isin([0, 1]).all():
            raise ValueError(f"Missing or invalid labels: {name}")
        selected[name] = select_model_features(frame, feature_set).reset_index(drop=True)
        numeric = selected[name].loc[:, list(numeric_features)].to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise ValueError(f"Infinite numeric features: {name}")
        metadata[name] = pd.DataFrame(
            {
                "flight_id": ids.to_numpy(),
                "flight_date": flight_dates.to_numpy(),
                "y_true": frame["is_delayed"].to_numpy(dtype=np.int64),
                "partition": name,
            }
        )
    if pd.concat([m.flight_id for m in metadata.values()]).duplicated().any():
        raise ValueError("Flight IDs must be globally unique")
    base = selected["train"]
    medians = base.loc[:, list(numeric_features)].median().fillna(0.0)
    imputed = base.loc[:, list(numeric_features)].fillna(medians)
    means = imputed.mean()
    scales = imputed.std(ddof=0).replace(0.0, 1.0)
    vocabularies = {
        f: {
            value: i + 2
            for i, value in enumerate(sorted(base[f].dropna().astype(str).unique()))
        }
        for f in CATEGORICAL
    }
    partitions = {}
    for name, features in selected.items():
        numeric = (
            (features.loc[:, list(numeric_features)].fillna(medians) - means) / scales
        ).to_numpy(dtype=np.float32)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Nonfinite transformed numeric features: {name}")
        categorical = np.column_stack(
            [
                features[f]
                .astype("string")
                .map(vocabularies[f])
                .fillna(1)
                .mask(features[f].isna(), 0)
                .to_numpy(dtype=np.int64)
                for f in CATEGORICAL
            ]
        )
        partitions[name] = DLPartition(numeric, categorical, metadata[name])
    return DLDataset(
        partitions, medians, means, scales, vocabularies, numeric_features=numeric_features
    )

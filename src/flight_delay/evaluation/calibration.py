"""Training-only chronological score calibration and reliability diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .metrics import validate


def check_windows(
    base_dates: ArrayLike, calibration_dates: ArrayLike, validation_dates: ArrayLike
) -> None:
    """Reject missing dates, overlap and chronology violations."""
    windows = [
        pd.to_datetime(pd.Series(d), errors="raise")
        for d in (base_dates, calibration_dates, validation_dates)
    ]
    if any(w.empty or w.isna().any() for w in windows) or not (
        windows[0].max() < windows[1].min() and windows[1].max() < windows[2].min()
    ):
        raise ValueError(
            "Base training must precede calibration, which must precede evaluation"
        )


class ScoreCalibrator:
    """Sigmoid over clipped log odds or monotone isotonic probability map."""

    def __init__(self, method: str):
        if method not in ("raw", "sigmoid", "isotonic"):
            raise ValueError("Unknown calibration method")
        self.method = method

    @staticmethod
    def _logit(p: ArrayLike) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, y: ArrayLike, p: ArrayLike, *, partition: str) -> ScoreCalibrator:
        """Fit only explicitly tagged training calibration observations."""
        if partition != "train_calibration":
            raise ValueError("Only train_calibration labels may fit calibrators")
        y, p = validate(y, p)
        if len(np.unique(y)) != 2:
            raise ValueError("Calibration needs both classes")
        self.estimator = None
        if self.method == "sigmoid":
            self.estimator = LogisticRegression(
                C=1e10, max_iter=1000, random_state=42
            ).fit(self._logit(p), y)
        elif self.method == "isotonic":
            self.estimator = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p: ArrayLike) -> np.ndarray:
        """Map held-out probabilities using the fitted map."""
        _, p = validate(np.zeros(len(p)), p)
        if not hasattr(self, "estimator"):
            raise ValueError("Calibrator is not fitted")
        if self.method == "raw":
            return p
        if self.method == "sigmoid":
            return self.estimator.predict_proba(self._logit(p))[:, 1]
        return self.estimator.predict(p)


def reliability(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bins including endpoints; empty bins omitted, counts retained."""
    y, p = validate(y, p)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return (
        pd.DataFrame({"bin": bins, "probability": p, "observed": y})
        .groupby("bin")
        .agg(
            mean_probability=("probability", "mean"),
            observed_rate=("observed", "mean"),
            count=("observed", "size"),
        )
        .reset_index()
    )

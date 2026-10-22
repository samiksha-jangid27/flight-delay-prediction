"""Training-only constant-probability references for validation calibration audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


@dataclass(frozen=True)
class PrevalenceBaseline:
    """A constant estimated from one explicitly identified pre-validation window."""

    probability: float
    partition: str
    n_fit: int
    first_date: str
    last_date: str

    def metadata(self) -> dict:
        """Return JSON-ready fitting provenance."""
        return asdict(self)

    def score_validation(self, validation: pd.DataFrame) -> dict:
        """Score a frozen constant; never update its prevalence from evaluation labels."""
        required = {"flight_date", "y_true", "partition"}
        if not required <= set(validation) or validation.empty:
            raise ValueError("Validation needs dates, binary labels and partition tags")
        dates = pd.to_datetime(validation.flight_date, errors="raise")
        if dates.isna().any() or dates.min() <= pd.Timestamp(self.last_date):
            raise ValueError("Validation must strictly follow baseline fitting dates")
        if not validation.partition.eq("validation").all():
            raise ValueError("This analysis scores validation only")
        y = validation.y_true.to_numpy()
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Validation labels must be nonmissing binary values")
        p = np.full(len(y), self.probability)
        return {
            "n": len(y),
            "constant_probability": self.probability,
            "observed_prevalence": float(y.mean()),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "average_precision": float(average_precision_score(y, p))
            if len(np.unique(y)) == 2
            else None,
        }


def fit_prevalence(training: pd.DataFrame, *, partition: str) -> PrevalenceBaseline:
    """Fit only base_train or train_calibration records; no held-out input accepted.

    Caller must supply the actual original training window, not relabel held-out
    records. The baseline accepts no validation/test label parameter and has no
    selection step. Later scoring verifies chronology against this stored window.
    """
    if partition not in ("base_train", "train_calibration"):
        raise ValueError("Prevalence fitting accepts training partitions only")
    required = {"flight_date", "y_true", "partition"}
    if not required <= set(training) or training.empty:
        raise ValueError("Training requires dates, labels and explicit partition tags")
    if not training.partition.eq(partition).all():
        raise ValueError("Training rows have incorrect partition tags")
    y = training.y_true.to_numpy()
    dates = pd.to_datetime(training.flight_date, errors="raise")
    if not np.isin(y, [0, 1]).all() or dates.isna().any():
        raise ValueError("Training labels/dates must be nonmissing and valid")
    return PrevalenceBaseline(
        float(y.mean()),
        partition,
        len(y),
        str(dates.min().date()),
        str(dates.max().date()),
    )

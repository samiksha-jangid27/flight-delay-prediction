"""Validation-only operating-point sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .metrics import validate


def threshold_table(
    y: ArrayLike, p: ArrayLike, *, partition: str = "validation"
) -> pd.DataFrame:
    """Evaluate every distinct decision boundary plus a dense reference grid."""
    if partition != "validation":
        raise ValueError("Threshold selection requires validation")
    y, p = validate(y, p)
    ts = np.unique(np.r_[np.linspace(0, 1, 201), p])
    order = np.argsort(p)
    scores = p[order]
    labels = y[order]
    cumulative = np.r_[0, np.cumsum(labels)]
    cut = np.searchsorted(scores, ts, side="left")
    tp = y.sum() - cumulative[cut]
    fp = len(y) - cut - tp
    fn = y.sum() - tp
    tn = len(y) - y.sum() - fp
    divide = lambda a, b: np.divide(
        a, b, out=np.zeros_like(a, dtype=float), where=b != 0
    )
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    return pd.DataFrame(
        {
            "threshold": ts,
            "precision": precision,
            "recall": recall,
            "f1": divide(2 * tp, 2 * tp + fp + fn),
            "f2": divide(5 * tp, 5 * tp + 4 * fn + fp),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "predicted_positive_rate": (tp + fp) / len(y),
            "false_alarm_rate": divide(fp, fp + tn),
            "missed_delay_rate": divide(fn, fn + tp),
        }
    )


def select_thresholds(
    table: pd.DataFrame, precision_floors=(0.2, 0.25, 0.3, 0.4)
) -> dict:
    """F1 recommendation; precision constraints are illustrative, not approved costs."""
    best = lambda column: (
        table.sort_values([column, "threshold"], ascending=[False, False])
        .iloc[0]
        .to_dict()
    )
    choices = {
        "max_f1": best("f1"),
        "max_f2": best("f2"),
        "reference_0.5": table.iloc[(table.threshold - 0.5).abs().argmin()].to_dict(),
        "precision_sensitivity": {},
    }
    for floor in precision_floors:
        eligible = table[(table.precision >= floor) & ((table.tp + table.fp) > 0)]
        choices["precision_sensitivity"][str(floor)] = (
            None
            if eligible.empty
            else eligible.sort_values(
                ["recall", "precision", "threshold"], ascending=[False, False, False]
            )
            .iloc[0]
            .to_dict()
        )
    return choices

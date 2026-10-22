"""Validated classification, ranking and proper scoring metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn import metrics as m


def validate(y: ArrayLike, p: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Require nonempty, aligned binary labels and finite probabilities."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors")
    if (
        not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Require binary labels and finite probabilities in [0, 1]")
    return y.astype(int), p


def evaluate(y: ArrayLike, p: ArrayLike, threshold: float = 0.5) -> dict:
    """Compute metrics; undefined one-class ranking metrics are null."""
    y, p = validate(y, p)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    pred = p >= threshold
    tn, fp, fn, tp = m.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ratio = lambda a, b: float(a / b) if b else 0.0
    return {
        "threshold": float(threshold),
        "precision": m.precision_score(y, pred, zero_division=0),
        "recall": m.recall_score(y, pred, zero_division=0),
        "f1": m.f1_score(y, pred, zero_division=0),
        "f2": m.fbeta_score(y, pred, beta=2, zero_division=0),
        "accuracy": m.accuracy_score(y, pred),
        "balanced_accuracy": (ratio(tp, tp + fn) + ratio(tn, tn + fp)) / 2,
        "specificity": ratio(tn, tn + fp),
        "false_positive_rate": ratio(fp, tn + fp),
        "false_negative_rate": ratio(fn, tp + fn),
        "predicted_positive_rate": float(pred.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "prevalence": float(y.mean()),
        "n": len(y),
        "roc_auc": float(m.roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(m.average_precision_score(y, p))
        if len(np.unique(y)) == 2
        else None,
        "brier": float(m.brier_score_loss(y, p)),
        "log_loss": float(m.log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(p.mean()),
        "calibration_bias": float(p.mean() - y.mean()),
        "class_metrics": m.classification_report(
            y, pred, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }


def align_predictions(
    frames: dict[str, pd.DataFrame], partition: str
) -> dict[str, pd.DataFrame]:
    """Align exact ID sets, labels, dates and explicit partition without dropping rows."""
    aligned = {}
    reference = None
    for name, frame in frames.items():
        cols = ["flight_id", "flight_date", "y_true", "y_probability", "partition"]
        if not set(cols) <= set(frame) or frame[cols].isna().any().any():
            raise ValueError("Missing prediction columns or values")
        if (
            frame.flight_id.duplicated().any()
            or not frame.partition.eq(partition).all()
        ):
            raise ValueError("Duplicate identifiers or partition mismatch")
        frame = frame.sort_values("flight_id").reset_index(drop=True)
        validate(frame.y_true, frame.y_probability)
        if reference is not None and not frame[
            ["flight_id", "flight_date", "y_true"]
        ].equals(reference):
            raise ValueError("Mismatched record IDs, dates or labels")
        reference = frame[["flight_id", "flight_date", "y_true"]]
        aligned[name] = frame
    if not aligned:
        raise ValueError("No prediction frames")
    return aligned


def paired_bootstrap(
    y, a, b, dates, thresholds=(0.5, 0.5), n_boot: int = 300, seed: int = 42
) -> dict:
    """Paired date-block percentile CIs for B minus A; assumes exchangeable days."""
    y, a = validate(y, a)
    _, b = validate(y, b)
    dates = np.asarray(dates)
    if len(dates) != len(y) or pd.isna(dates).any() or n_boot < 1:
        raise ValueError("Require aligned dates and positive bootstrap count")
    groups = [np.flatnonzero(dates == d) for d in np.unique(dates)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        ix = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        if len(np.unique(y[ix])) < 2:
            continue
        values.append(
            [
                m.average_precision_score(y[ix], b[ix])
                - m.average_precision_score(y[ix], a[ix]),
                m.roc_auc_score(y[ix], b[ix]) - m.roc_auc_score(y[ix], a[ix]),
                m.f1_score(y[ix], b[ix] >= thresholds[1], zero_division=0)
                - m.f1_score(y[ix], a[ix] >= thresholds[0], zero_division=0),
            ]
        )
    return {
        "method": "paired calendar-date block percentile; exchangeable dates; not serial block bootstrap",
        "seed": seed,
        "requested": n_boot,
        "valid": len(values),
        "confidence": 0.95,
        "differences": {
            k: np.quantile(np.asarray(values)[:, i], [0.025, 0.975]).tolist()
            for i, k in enumerate(["average_precision", "roc_auc", "f1"])
        }
        if values
        else {},
    }

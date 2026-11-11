"""Leakage-safe stacked ensemble of the frozen XGBoost model and the MLP.

Rationale
---------
XGBoost splits on tabular operational features (schedule, congestion, calendar,
historical delay rates) and captures sharp threshold interactions. The embedding
MLP instead learns dense latent representations for carrier/origin/dest and smooth
combinations of the standardized numeric block. The two model families therefore
make partly independent errors. A small logistic meta-learner stacked on top of
their out-of-sample probabilities can recover signal that neither model alone
expresses, which is the "operational tabular features to learned route/temporal
patterns" hybrid the project brief asks for.

Leakage discipline
-------------------
Both base models are trained only on the early-training partition. Their
probabilities on the later ``train_calibration`` rows are therefore genuinely
out-of-sample, so the meta-learner is fitted there without seeing any row its
inputs were trained on. The meta-learner is evaluated on ``validation`` and the
held-out ``test`` partition is never loaded, scored, or used for selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from flight_delay.data.pipeline import build_data_pipeline
from flight_delay.features.model_features import select_model_features

CALIBRATION_FRACTION: float = 0.8
META_FEATURES: tuple[str, ...] = ("xgb", "mlp")
PREDICTION_COLUMNS: tuple[str, ...] = (
    "flight_id",
    "flight_date",
    "y_true",
    "y_probability",
    "partition",
)


def _dates(frame: pd.DataFrame) -> pd.Series:
    """Reconstruct calendar dates from the upstream year/month/day schema."""
    return pd.to_datetime(frame[["year", "month", "day"]])


def _prediction_frame(
    frame: pd.DataFrame, probabilities: np.ndarray, partition: str
) -> pd.DataFrame:
    """Assemble the canonical project prediction schema for one partition."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(frame),):
        raise ValueError("Probabilities must align one-to-one with the frame rows")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Base probabilities must be finite and in [0, 1]")
    return pd.DataFrame(
        {
            "flight_id": frame["id"].astype("string").to_numpy(),
            "flight_date": _dates(frame).dt.strftime("%Y-%m-%d").to_numpy(),
            "y_true": frame["is_delayed"].to_numpy(dtype=np.int64),
            "y_probability": probabilities,
            "partition": partition,
        }
    )


def xgb_raw_probabilities(
    data_path: str | Path, frozen_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the frozen XGBoost on the exact calibration and validation rows.

    The calibration cutoff reproduces the Phase 1 evaluation split and the DL
    dataset split (last 20% of training dates), so the returned rows line up
    one-to-one with the MLP's exported calibration/validation predictions.
    """
    train, validation, _test = build_data_pipeline(data_path)
    days = sorted(_dates(train).unique())
    if len(days) < 2:
        raise ValueError("Need at least two training dates for a calibration split")
    cutoff = days[int(len(days) * CALIBRATION_FRACTION)]
    calibration = train.loc[_dates(train) >= cutoff]
    if calibration.empty:
        raise ValueError("Calibration partition is empty")

    bundle = joblib.load(frozen_path)
    if "xgboost_tuned_full" not in bundle:
        raise ValueError("Frozen artifact does not contain 'xgboost_tuned_full'")
    model = bundle["xgboost_tuned_full"]["model"]

    calibration_probs = model.predict_proba(select_model_features(calibration))[:, 1]
    validation_probs = model.predict_proba(select_model_features(validation))[:, 1]
    return (
        _prediction_frame(calibration, calibration_probs, "train_calibration"),
        _prediction_frame(validation, validation_probs, "validation"),
    )


def _merge_base_predictions(
    xgb: pd.DataFrame, mlp: pd.DataFrame, partition: str
) -> pd.DataFrame:
    """Join XGBoost and MLP probabilities on flight_id with strict agreement checks."""
    for name, frame in (("xgb", xgb), ("mlp", mlp)):
        missing = set(PREDICTION_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions missing columns: {sorted(missing)}")
        if not frame["partition"].eq(partition).all():
            raise ValueError(f"{name} predictions are not all partition {partition!r}")
        if frame["flight_id"].duplicated().any():
            raise ValueError(f"{name} predictions contain duplicate flight_id values")

    left = xgb.rename(columns={"y_probability": "xgb"})
    right = mlp.rename(columns={"y_probability": "mlp"})[
        ["flight_id", "y_true", "mlp"]
    ]
    merged = left.merge(
        right, on="flight_id", how="inner", suffixes=("", "_mlp"), validate="one_to_one"
    )
    if len(merged) != len(xgb) or len(merged) != len(mlp):
        raise ValueError(
            f"{partition} base predictions do not align one-to-one on flight_id"
        )
    if not (merged["y_true"] == merged["y_true_mlp"]).all():
        raise ValueError(f"{partition} labels disagree between XGBoost and MLP")
    return merged.drop(columns="y_true_mlp")


def fit_meta_learner(
    calibration_xgb: pd.DataFrame,
    calibration_mlp: pd.DataFrame,
    *,
    seed: int = 42,
) -> LogisticRegression:
    """Fit the logistic stacker on out-of-sample calibration probabilities only."""
    merged = _merge_base_predictions(
        calibration_xgb, calibration_mlp, "train_calibration"
    )
    labels = merged["y_true"].to_numpy(dtype=np.int64)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Calibration labels must contain both binary classes")
    features = merged.loc[:, list(META_FEATURES)].to_numpy(dtype=np.float64)
    meta = LogisticRegression(max_iter=1000, random_state=seed)
    meta.fit(features, labels)
    return meta


def error_diversity(
    xgb: pd.DataFrame, mlp: pd.DataFrame, partition: str = "validation"
) -> dict[str, float]:
    """Quantify how independently the two base models err on the same rows.

    Stacking can only add value when base models make different mistakes. This
    returns the probability correlation, the rank (Spearman) correlation, and
    the correlation of their residuals ``y - p``. Residual correlations close to
    1 mean the models are redundant and no ensemble can help.
    """
    merged = _merge_base_predictions(xgb, mlp, partition)
    y = merged["y_true"].to_numpy(dtype=np.float64)
    px = merged["xgb"].to_numpy(dtype=np.float64)
    pm = merged["mlp"].to_numpy(dtype=np.float64)
    return {
        "probability_pearson": float(np.corrcoef(px, pm)[0, 1]),
        "probability_spearman": float(
            pd.Series(px).corr(pd.Series(pm), method="spearman")
        ),
        "residual_pearson": float(np.corrcoef(y - px, y - pm)[0, 1]),
    }


def predict_hybrid(
    meta: LogisticRegression,
    validation_xgb: pd.DataFrame,
    validation_mlp: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the fitted stacker to validation base probabilities."""
    merged = _merge_base_predictions(validation_xgb, validation_mlp, "validation")
    features = merged.loc[:, list(META_FEATURES)].to_numpy(dtype=np.float64)
    probabilities = meta.predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "flight_id": merged["flight_id"].astype("string").to_numpy(),
            "flight_date": merged["flight_date"].to_numpy(),
            "y_true": merged["y_true"].to_numpy(dtype=np.int64),
            "y_probability": np.asarray(probabilities, dtype=np.float64),
            "partition": "validation",
        }
    )

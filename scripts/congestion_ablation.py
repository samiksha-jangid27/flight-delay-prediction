"""Validation-only ablation: does airport congestion add anything?

Trains the two classifiers on the base-training window with and without the
schedule-based congestion features, and reports validation Average Precision and
ROC-AUC for each. It exports validation-only results and does not train, tune,
or score against the test partition. It measures congestion's marginal
contribution without changing the canonical 31-feature evaluation.

Run from repo root:  python scripts/congestion_ablation.py
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from flight_delay.data.pipeline import build_data_pipeline  # noqa: E402
from flight_delay.features.selection import split_features_and_target  # noqa: E402
from flight_delay.features.model_features import select_model_features  # noqa: E402
from flight_delay.models.logistic import fit_logistic_baseline  # noqa: E402
from flight_delay.models.xgboost_model import fit_xgboost_baseline  # noqa: E402

XGB_PARAMS = {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.03,
              "min_child_weight": 10, "subsample": 0.90, "colsample_bytree": 0.75,
              "reg_lambda": 10.0, "scale_pos_weight": 2.859}


def _dates(f: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(f[["year", "month", "day"]])


def main() -> None:
    train, val, _ = build_data_pipeline(ROOT / "data" / "raw" / "flights.csv")
    days = sorted(_dates(train).unique())
    cutoff = days[int(len(days) * 0.8)]
    base = train.loc[_dates(train) < cutoff]           # same base-train window as the frozen run
    Xb, yb = split_features_and_target(base)
    Xv, yv = split_features_and_target(val)
    yv_np = yv.to_numpy()

    rows = []
    feature_columns: dict[str, list[str]] = {}
    for fs in ("full", "full_congestion"):
        feature_columns[fs] = list(select_model_features(Xb, fs).columns)
        lr = fit_logistic_baseline(Xb, yb, feature_set=fs, class_weight="balanced")
        xgb = fit_xgboost_baseline(Xb, yb, feature_set=fs, model_params=XGB_PARAMS)
        for name, model in (("Balanced LR", lr), ("Tuned XGBoost", xgb)):
            p = model.predict_proba(select_model_features(Xv, fs))[:, 1]
            rows.append({
                "feature_set": fs, "n_features": len(select_model_features(Xb, fs).columns),
                "model": name,
                "val_average_precision": round(float(average_precision_score(yv_np, p)), 4),
                "val_roc_auc": round(float(roc_auc_score(yv_np, p)), 4),
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    # deltas
    print("\nMarginal effect of congestion (full_congestion - full), validation:")
    for name in ("Balanced LR", "Tuned XGBoost"):
        a = df[(df.model == name) & (df.feature_set == "full")].iloc[0]
        b = df[(df.model == name) & (df.feature_set == "full_congestion")].iloc[0]
        print(f"  {name}: ΔAP = {b.val_average_precision - a.val_average_precision:+.4f}, "
              f"ΔROC-AUC = {b.val_roc_auc - a.val_roc_auc:+.4f}")
    data_path = ROOT / "data" / "raw" / "flights.csv"
    base_dates = _dates(base)
    val_dates = _dates(val)
    manifest = {
        "schema_version": 1,
        "experiment": "validation-only congestion ablation",
        "test_used": False,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "seed": 42,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "windows": {
            "outer_training": {"n": int(len(train))},
            "base_training": {
                "n": int(len(base)),
                "first": str(base_dates.min().date()),
                "last": str(base_dates.max().date()),
            },
            "validation": {
                "n": int(len(val)),
                "first": str(val_dates.min().date()),
                "last": str(val_dates.max().date()),
            },
        },
        "feature_columns": feature_columns,
        "models": {
            "Balanced LR": {
                "C": 1.0,
                "class_weight": "balanced",
                "random_state": 42,
                "max_iter": 1000,
                "calibration": None,
                "threshold": None,
            },
            "Tuned XGBoost": {
                **XGB_PARAMS,
                "random_state": 42,
                "calibration": None,
                "threshold": None,
            },
        },
        "metrics": {
            "ranking": "scikit-learn Average Precision (AP); not trapezoidal PR area",
            "selection": "No feature/model selection from the held-out test; validation-only comparison",
            "results": rows,
        },
    }
    out = ROOT / "docs" / "congestion_ablation.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

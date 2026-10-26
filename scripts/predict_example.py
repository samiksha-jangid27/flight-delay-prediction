"""Concrete sample-prediction demo for the frozen flight-delay model.

Loads the frozen XGBoost pipeline + isotonic calibrator + operating threshold,
then shows, for a few real validation flights, exactly how a pre-departure delay
probability is produced and turned into a decision:

    raw model score  ->  isotonic calibration  ->  compare to frozen threshold.

This demonstrates the *source* of every displayed probability. It never touches
the held-out test partition and fits nothing.

Run from the repo root:  python scripts/predict_example.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.data.pipeline import build_data_pipeline  # noqa: E402
from flight_delay.features.model_features import select_model_features  # noqa: E402

FROZEN = ROOT / "reports" / "evaluation" / "frozen.joblib"
DATA = ROOT / "data" / "raw" / "flights.csv"
SHOW_COLS = ["carrier", "origin", "dest", "hour", "sched_dep_time", "distance",
             "carrier_historical_delay_rate", "route_historical_delay_rate"]


def main() -> None:
    frozen = joblib.load(FROZEN)["xgboost_tuned_full"]
    model, calibrator, threshold = frozen["model"], frozen["calibrator"], frozen["threshold"]

    # This example predicts validation flights only; it does not score the test split.
    _, validation, _ = build_data_pipeline(DATA)
    X = select_model_features(validation, "full")

    raw = model.predict_proba(X)[:, 1]            # uncalibrated model score
    cal = calibrator.predict(raw)                 # isotonic-calibrated probability
    y_true = validation["is_delayed"].to_numpy()

    print(f"Frozen model: Tuned XGBoost  |  calibrator: {calibrator.method}  "
          f"|  operating threshold: {threshold:.3f}\n")

    # Show one likely-delayed and one likely-on-time example, plus the top-risk flight.
    import numpy as np
    picks = {
        "Highest-risk flight": int(np.argmax(cal)),
        "A borderline flight (near threshold)": int(np.argmin(np.abs(cal - threshold))),
        "A low-risk flight": int(np.argmin(cal)),
    }
    for label, i in picks.items():
        row = validation.iloc[i]
        decision = "ACT (predict delayed)" if cal[i] >= threshold else "no action (predict on-time)"
        print(f"--- {label} ---")
        for c in SHOW_COLS:
            if c in row:
                print(f"    {c:32s} = {row[c]}")
        print(f"    raw model score                  = {raw[i]:.3f}")
        print(f"    calibrated P(delay)  <-- shown    = {cal[i]:.3f}")
        print(f"    threshold                        = {threshold:.3f}")
        print(f"    DECISION                         = {decision}")
        print(f"    actual outcome                   = {'DELAYED' if y_true[i]==1 else 'on-time'}\n")


if __name__ == "__main__":
    main()

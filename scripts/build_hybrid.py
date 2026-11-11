"""Build and evaluate the leakage-safe XGBoost + MLP stacked hybrid.

This produces the hybrid's validation predictions in the canonical project
schema and a raw metrics comparison against each base model. It intentionally
does NOT touch the Priority-5 evaluation deliverables (compare_aml_dl.py,
docs/dl_evaluation.md, reports/dl/figures/); the exported hybrid CSV is meant to
be consumed by that comparison as one more model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from flight_delay.deep_learning.hybrid import (
    error_diversity,
    fit_meta_learner,
    predict_hybrid,
    xgb_raw_probabilities,
)
from flight_delay.evaluation.metrics import evaluate
from flight_delay.evaluation.predictions import load_prediction_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/flights.csv")
    parser.add_argument("--frozen", default="reports/evaluation/frozen.joblib")
    parser.add_argument("--dl-dir", default="reports/dl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dl_dir = Path(args.dl_dir)
    mlp_cal = load_prediction_csv(
        dl_dir / "train_calibration_predictions.csv", partition="train_calibration"
    )
    mlp_val = load_prediction_csv(
        dl_dir / "validation_predictions.csv", partition="validation"
    )

    xgb_cal, xgb_val = xgb_raw_probabilities(args.data, args.frozen)
    xgb_cal.to_csv(dl_dir / "xgb_train_calibration_predictions.csv", index=False)
    xgb_val.to_csv(dl_dir / "xgb_validation_predictions.csv", index=False)

    meta = fit_meta_learner(xgb_cal, mlp_cal, seed=args.seed)
    hybrid_val = predict_hybrid(meta, xgb_val, mlp_val)
    hybrid_val.to_csv(dl_dir / "hybrid_validation_predictions.csv", index=False)

    comparison = {
        "xgboost_raw": evaluate(xgb_val.y_true, xgb_val.y_probability),
        "mlp_raw": evaluate(mlp_val.y_true, mlp_val.y_probability),
        "hybrid": evaluate(hybrid_val.y_true, hybrid_val.y_probability),
    }
    summary = {
        name: {
            "average_precision": metrics["average_precision"],
            "roc_auc": metrics["roc_auc"],
            "brier": metrics["brier"],
            "log_loss": metrics["log_loss"],
        }
        for name, metrics in comparison.items()
    }
    summary["meta_learner_coefficients"] = {
        "xgb": float(meta.coef_[0][0]),
        "mlp": float(meta.coef_[0][1]),
        "intercept": float(meta.intercept_[0]),
    }
    summary["error_diversity"] = error_diversity(xgb_val, mlp_val)
    (dl_dir / "hybrid_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

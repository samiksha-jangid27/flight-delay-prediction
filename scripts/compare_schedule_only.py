"""Compare AML, MLP and hybrid on schedule and route inputs without outcome history."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from flight_delay.data.pipeline import build_data_pipeline
from flight_delay.deep_learning.dataset import build_dl_dataset
from flight_delay.deep_learning.encoding import DLTensorEncoder
from flight_delay.deep_learning.hybrid import fit_meta_learner, predict_hybrid
from flight_delay.deep_learning.train import (
    _fit_trial, _model, _predict_probabilities, _prediction_frame, _read_config,
    _validate_trial,
)
from flight_delay.evaluation.calibration import ScoreCalibrator
from flight_delay.evaluation.evaluate import XGB_PARAMS, dates, prediction_frame
from flight_delay.evaluation.metrics import align_predictions
from flight_delay.evaluation.rescheduling import CostModel, policy_outcome
from flight_delay.features.model_features import select_model_features
from flight_delay.models.logistic import fit_logistic_baseline
from flight_delay.models.xgboost_model import fit_xgboost_baseline
from scripts.compare_aml_dl import metrics


def run(data: Path, config: dict, frozen_trial: dict, n_jobs: int) -> pd.DataFrame:
    feature_set = "route"
    device = torch.device("cpu")
    dataset = build_dl_dataset(data, feature_set=feature_set)
    encoder = DLTensorEncoder(dataset)
    train = encoder.transform_partition("train")
    calibration = encoder.transform_partition("train_calibration")
    validation = encoder.transform_partition("validation")
    trial = _validate_trial({
        key: frozen_trial[key] for key in config["search"]["trials"][0]
    })
    settings = config["training"]
    fit = _fit_trial(
        encoder, train, validation, trial, seed=frozen_trial["seed"],
        epochs=settings["epochs"], patience=settings["patience"],
        min_delta=settings["min_delta"], optimizer_name=settings["optimizer"], device=device,
    )
    mlp = _model(encoder, trial)
    mlp.load_state_dict(fit["state_dict"])
    mlp_predictions = [
        _prediction_frame(batch, _predict_probabilities(mlp, batch, trial["batch_size"], device))
        for batch in (calibration, validation)
    ]

    full_train, val, _ = build_data_pipeline(data)
    cutoff = calibration.metadata.flight_date.min()
    base = full_train.loc[dates(full_train) < cutoff]
    cal = full_train.loc[dates(full_train) >= cutoff]
    if not base.id.astype(str).reset_index(drop=True).equals(
        train.metadata.flight_id.astype(str).reset_index(drop=True)
    ):
        raise ValueError("AML and MLP training rows differ")
    x_base = select_model_features(base, feature_set)
    models = {
        "logistic": fit_logistic_baseline(
            x_base, base.is_delayed, feature_set=feature_set, class_weight="balanced"
        ),
        "xgboost": fit_xgboost_baseline(
            x_base, base.is_delayed, feature_set=feature_set,
            n_jobs=n_jobs, model_params=XGB_PARAMS,
        ),
    }
    cal_predictions = {"mlp": mlp_predictions[0]}
    val_predictions = {"mlp": mlp_predictions[1]}
    for name, model in models.items():
        cal_predictions[name] = prediction_frame(
            cal, model.predict_proba(select_model_features(cal, feature_set))[:, 1],
            "train_calibration",
        )
        val_predictions[name] = prediction_frame(
            val, model.predict_proba(select_model_features(val, feature_set))[:, 1],
            "validation",
        )
    for predictions in (cal_predictions, val_predictions):
        for frame in predictions.values():
            frame["flight_id"] = frame.flight_id.astype("string")
            frame["flight_date"] = frame.flight_date.astype("string")
            frame["y_true"] = frame.y_true.astype("int64")
    cal_predictions = align_predictions(cal_predictions, "train_calibration")
    val_predictions = align_predictions(val_predictions, "validation")
    meta = fit_meta_learner(cal_predictions["xgboost"], cal_predictions["mlp"])
    hybrid = predict_hybrid(meta, val_predictions["xgboost"], val_predictions["mlp"])

    probabilities = {"hybrid": hybrid.y_probability.to_numpy()}
    for name, cal_frame in cal_predictions.items():
        calibrator = ScoreCalibrator("isotonic").fit(
            cal_frame.y_true, cal_frame.y_probability, partition="train_calibration"
        )
        probabilities[f"{name}_isotonic"] = calibrator.predict(val_predictions[name].y_probability)
    y = hybrid.y_true.to_numpy()
    cost = CostModel()
    rows = []
    for name, probability in probabilities.items():
        result = metrics(y, probability, (0.25,))
        outcome = policy_outcome(y, probability, cost.optimal_threshold, cost)
        rows.append({
            "model": name, "n": result["n"],
            **{key: result[key] for key in (
                "average_precision", "pr_auc_trapezoidal", "roc_auc", "brier", "log_loss"
            )},
            **result["validation_max_f1"],
            "recall_at_precision_0.25": result["recall_at_min_precision"]["0.25"]["recall"],
            "what_if_action_cost_ratio": cost.cost_ratio,
            "what_if_recovery": cost.recovery,
            "what_if_threshold": outcome["threshold"],
            "what_if_net_savings_per_flight": outcome["net_savings_per_flight"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dl_config.yaml"))
    parser.add_argument("--trial", type=Path, default=Path("reports/dl/best_hyperparameters.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/dl/schedule_only_metrics.csv"))
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    torch.set_num_threads(args.n_jobs)
    config = _read_config(args.config)
    frozen_trial = yaml.safe_load(args.trial.read_text())
    data = args.config.resolve().parent.parent / config["data_path"]
    results = run(data, config, frozen_trial, args.n_jobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

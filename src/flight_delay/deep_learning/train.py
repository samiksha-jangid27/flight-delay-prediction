"""Reproducible MLP tuning, training, checkpointing, and prediction export."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch import Tensor, nn

from flight_delay.deep_learning.dataset import build_dl_dataset
from flight_delay.deep_learning.embeddings import build_embedding_tables
from flight_delay.deep_learning.encoding import DLTensorEncoder, TensorBatch
from flight_delay.deep_learning.mlp import FlightDelayMLP

_HYPERPARAMETERS = (
    "learning_rate",
    "batch_size",
    "hidden_dims",
    "dropout",
    "weight_decay",
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic operations."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _validate_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    if set(trial) != set(_HYPERPARAMETERS):
        raise ValueError(f"Each trial must specify exactly {_HYPERPARAMETERS}")
    learning_rate = trial["learning_rate"]
    batch_size = trial["batch_size"]
    dropout = trial["dropout"]
    weight_decay = trial["weight_decay"]
    hidden_dims = trial["hidden_dims"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be positive")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if (
        not isinstance(hidden_dims, Sequence)
        or isinstance(hidden_dims, (str, bytes))
        or not hidden_dims
        or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in hidden_dims
        )
    ):
        raise ValueError("hidden_dims must be a nonempty sequence of positive integers")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not (
        0 <= dropout < 1
    ):
        raise ValueError("dropout must be in [0, 1)")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be nonnegative")
    return {
        "learning_rate": float(learning_rate),
        "batch_size": batch_size,
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
    }


def _model(encoder: DLTensorEncoder, trial: Mapping[str, Any]) -> FlightDelayMLP:
    tables = build_embedding_tables(encoder.embedding_sizes)
    return FlightDelayMLP(
        encoder.dense_dim,
        carrier_embedding=tables["carrier"],
        origin_embedding=tables["origin"],
        dest_embedding=tables["dest"],
        hidden_dims=trial["hidden_dims"],
        dropout=trial["dropout"],
    )


def _validation_ap(
    model: FlightDelayMLP,
    batch: TensorBatch,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, batch.size, batch_size):
            stop = min(start + batch_size, batch.size)
            logits = model(
                batch.x_dense[start:stop].to(device),
                batch.x_cat[start:stop].to(device),
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return float(
        average_precision_score(
            batch.labels.cpu().numpy(), np.concatenate(probabilities)
        )
    )


def _fit_trial(
    encoder: DLTensorEncoder,
    train: TensorBatch,
    validation: TensorBatch,
    trial: Mapping[str, Any],
    *,
    seed: int,
    epochs: int,
    patience: int,
    min_delta: float,
    optimizer_name: str,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed)
    model = _model(encoder, trial).to(device)
    optimizer_type = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(
        model.parameters(),
        lr=trial["learning_rate"],
        weight_decay=trial["weight_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_ap = -np.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train.size, generator=generator)
        for start in range(0, train.size, trial["batch_size"]):
            indices = order[start : start + trial["batch_size"]]
            x_dense = train.x_dense.index_select(0, indices).to(device)
            x_cat = train.x_cat.index_select(0, indices).to(device)
            labels = train.labels.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_dense, x_cat), labels)
            loss.backward()
            optimizer.step()

        validation_ap = _validation_ap(
            model, validation, trial["batch_size"], device
        )
        if validation_ap > best_ap + min_delta:
            best_ap = validation_ap
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training completed without a finite validation checkpoint")
    return {
        "state_dict": best_state,
        "validation_average_precision": float(best_ap),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
    }


def load_best_checkpoint(
    checkpoint_path: str | Path,
    encoder: DLTensorEncoder,
    *,
    device: str | torch.device = "cpu",
) -> FlightDelayMLP:
    """Reconstruct the recorded architecture and strictly restore its weights."""
    target_device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=target_device, weights_only=True
    )
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported or missing DL checkpoint schema version")
    if (
        checkpoint.get("dense_dim") != encoder.dense_dim
        or checkpoint.get("embedding_sizes") != encoder.embedding_sizes
        or checkpoint.get("dense_feature_names") != list(encoder.dense_feature_names)
        or checkpoint.get("preprocessing") != _preprocessing_state(encoder)
    ):
        raise ValueError("Checkpoint preprocessing schema does not match the encoder")
    trial = _validate_trial(checkpoint["hyperparameters"])
    model = _model(encoder, trial).to(target_device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _preprocessing_state(encoder: DLTensorEncoder) -> dict[str, Any]:
    dataset = encoder.dataset
    return {
        "numeric_features": list(dataset.numeric_features),
        "categorical_features": list(dataset.categorical_features),
        "medians": {name: float(value) for name, value in dataset.medians.items()},
        "means": {name: float(value) for name, value in dataset.means.items()},
        "scales": {name: float(value) for name, value in dataset.scales.items()},
        "vocabularies": {
            field: {str(value): int(index) for value, index in vocabulary.items()}
            for field, vocabulary in dataset.vocabularies.items()
        },
    }


def _predict_probabilities(
    model: FlightDelayMLP,
    batch: TensorBatch,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, batch.size, batch_size):
            stop = min(start + batch_size, batch.size)
            logits = model(
                batch.x_dense[start:stop].to(device),
                batch.x_cat[start:stop].to(device),
            )
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(predictions).astype(np.float64)
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise RuntimeError("Model produced invalid prediction probabilities")
    return probabilities


def _prediction_frame(batch: TensorBatch, probabilities: np.ndarray) -> pd.DataFrame:
    metadata = batch.metadata
    return pd.DataFrame(
        {
            "flight_id": metadata["flight_id"].astype("string").to_numpy(),
            "flight_date": pd.to_datetime(metadata["flight_date"])
            .dt.strftime("%Y-%m-%d")
            .to_numpy(),
            "y_true": metadata["y_true"].to_numpy(dtype=np.int64),
            "y_probability": probabilities,
            "partition": metadata["partition"].to_numpy(),
        }
    )


def _read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("DL configuration must be a YAML mapping")
    required = {"data_path", "output_dir", "seed", "device", "training", "search"}
    if not required <= set(config):
        raise ValueError(f"DL configuration is missing keys: {sorted(required - set(config))}")
    if not isinstance(config["training"], dict) or not isinstance(config["search"], dict):
        raise ValueError("training and search configuration values must be mappings")
    training = config["training"]
    search = config["search"]
    required_training = {"optimizer", "epochs", "patience", "min_delta"}
    if not required_training <= set(training):
        raise ValueError(
            f"training configuration is missing keys: "
            f"{sorted(required_training - set(training))}"
        )
    if "trials" not in search or not isinstance(search["trials"], list) or not search["trials"]:
        raise ValueError("search.trials must be a nonempty list")
    config["search"]["trials"] = [
        _validate_trial(trial) for trial in search["trials"]
    ]
    if not isinstance(training["optimizer"], str) or training["optimizer"] not in {
        "adam",
        "adamw",
    }:
        raise ValueError("training.optimizer must be 'adam' or 'adamw'")
    for key in ("epochs", "patience"):
        if isinstance(training[key], bool) or not isinstance(training[key], int) or (
            training[key] < 1
        ):
            raise ValueError(f"training.{key} must be a positive integer")
    min_delta = training["min_delta"]
    if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)) or (
        min_delta < 0
    ):
        raise ValueError("training.min_delta must be nonnegative")
    seed_everything(config["seed"])
    if not isinstance(config["device"], str):
        raise ValueError("device must be a string")
    return config


def run_training(
    config_path: str | Path = "configs/dl_config.yaml",
    *,
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Tune on early training/validation, then export calibration/validation scores.

    The held-out test partition is never transformed, scored, or used for model
    selection. Validation AP drives both early stopping and hyperparameter choice.
    """
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    config = _read_config(config_file)
    input_path = Path(data_path or config["data_path"])
    if not input_path.is_absolute():
        input_path = project_root / input_path
    output_path = Path(output_dir or config["output_dir"])
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = build_dl_dataset(input_path)
    encoder = DLTensorEncoder(dataset)
    train = encoder.transform_partition("train")
    validation = encoder.transform_partition("validation")
    if set(train.labels.unique().tolist()) != {0.0, 1.0}:
        raise ValueError("Training labels must contain both binary classes")
    if set(validation.labels.unique().tolist()) != {0.0, 1.0}:
        raise ValueError("Validation labels must contain both binary classes")

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    settings = config["training"]
    trials = config["search"]["trials"]
    trial_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for trial_index, trial in enumerate(trials):
        result = _fit_trial(
            encoder,
            train,
            validation,
            trial,
            seed=config["seed"] + trial_index,
            epochs=settings["epochs"],
            patience=settings["patience"],
            min_delta=float(settings["min_delta"]),
            optimizer_name=settings["optimizer"],
            device=device,
        )
        record = {
            "trial": trial_index,
            **trial,
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
            "validation_average_precision": result["validation_average_precision"],
        }
        trial_results.append(record)
        if best is None or (
            record["validation_average_precision"]
            > best["record"]["validation_average_precision"]
        ):
            best = {"record": record, "result": result, "trial": trial}

    if best is None:
        raise RuntimeError("Hyperparameter search produced no result")

    checkpoint_path = output_path / "best_model.pt"
    checkpoint = {
        "schema_version": 1,
        "state_dict": best["result"]["state_dict"],
        "hyperparameters": best["trial"],
        "dense_dim": encoder.dense_dim,
        "dense_feature_names": list(encoder.dense_feature_names),
        "embedding_sizes": encoder.embedding_sizes,
        "preprocessing": _preprocessing_state(encoder),
        "seed": config["seed"] + best["record"]["trial"],
        "optimizer": settings["optimizer"],
        "best_epoch": best["record"]["best_epoch"],
        "validation_average_precision": best["record"][
            "validation_average_precision"
        ],
    }
    torch.save(checkpoint, checkpoint_path)
    best_hyperparameters = {
        **best["trial"],
        "optimizer": settings["optimizer"],
        "seed": checkpoint["seed"],
        "best_epoch": best["record"]["best_epoch"],
        "validation_average_precision": best["record"][
            "validation_average_precision"
        ],
    }
    (output_path / "best_hyperparameters.yaml").write_text(
        yaml.safe_dump(best_hyperparameters, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(trial_results).to_csv(output_path / "tuning_results.csv", index=False)
    (output_path / "run_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )

    restored = load_best_checkpoint(checkpoint_path, encoder, device=device)
    calibration = encoder.transform_partition("train_calibration")
    calibration_predictions = _prediction_frame(
        calibration,
        _predict_probabilities(
            restored, calibration, best["trial"]["batch_size"], device
        ),
    )
    validation_predictions = _prediction_frame(
        validation,
        _predict_probabilities(
            restored, validation, best["trial"]["batch_size"], device
        ),
    )
    calibration_predictions.to_csv(
        output_path / "train_calibration_predictions.csv", index=False
    )
    validation_predictions.to_csv(
        output_path / "validation_predictions.csv", index=False
    )
    summary_output_path = output_path.resolve()
    if summary_output_path.is_relative_to(project_root):
        summary_output_path = summary_output_path.relative_to(project_root)
    summary = {
        "seed": config["seed"],
        "selection_metric": "validation average precision",
        "test_partition_used_for_selection": False,
        "best_hyperparameters": best_hyperparameters,
        "checkpoint": (summary_output_path / checkpoint_path.name).as_posix(),
        "validation_predictions": (
            summary_output_path / "validation_predictions.csv"
        ).as_posix(),
        "train_calibration_predictions": (
            summary_output_path / "train_calibration_predictions.csv"
        ).as_posix(),
    }
    (output_path / "training_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "summary": summary,
        "trial_results": pd.DataFrame(trial_results),
        "calibration_predictions": calibration_predictions,
        "validation_predictions": validation_predictions,
        "checkpoint": checkpoint_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dl_config.yaml")
    parser.add_argument("--data", default=None, help="Override configured raw data path")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_training(args.config, data_path=args.data, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

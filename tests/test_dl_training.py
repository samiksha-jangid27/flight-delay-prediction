"""Training, early stopping, checkpoint restoration, and prediction exports."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from flight_delay.deep_learning.dataset import DLDataset, DLPartition
from flight_delay.deep_learning.encoding import DLTensorEncoder
from flight_delay.deep_learning.train import (
    _fit_trial,
    _predict_probabilities,
    load_best_checkpoint,
    run_training,
)
from flight_delay.evaluation.predictions import load_prediction_csv


def _partition(name, labels, *, malformed=False):
    labels = np.asarray(labels, dtype=np.int64)
    n_rows = len(labels)
    metadata = pd.DataFrame(
        {
            "flight_id": [f"{name}-{i}" for i in range(n_rows)],
            "flight_date": pd.date_range("2013-01-01", periods=n_rows),
            "y_true": labels,
            "partition": name,
        }
    )
    if malformed:
        metadata = metadata.drop(columns="y_true")
    values = np.arange(n_rows, dtype=np.float32) % 2
    numeric = values.reshape(-1, 1)
    categorical = np.tile(np.array([[2, 2, 2, 2, 2]], dtype=np.int64), (n_rows, 1))
    return DLPartition(numeric, categorical, metadata)


def _dataset():
    vocabularies = {
        name: {"known-a": 2, "known-b": 3}
        for name in ("carrier", "origin", "dest", "route", "distance_band")
    }
    return DLDataset(
        partitions={
            "train": _partition("train", [0, 1] * 8),
            "train_calibration": _partition("train_calibration", [0, 1, 0, 1]),
            "validation": _partition("validation", [0, 1] * 4),
            # Missing test labels ensure the trainer never transforms this partition.
            "test": _partition("test", [0, 1], malformed=True),
        },
        medians=pd.Series({"num": 0.0}),
        means=pd.Series({"num": 0.0}),
        scales=pd.Series({"num": 1.0}),
        vocabularies=vocabularies,
        numeric_features=("num",),
        categorical_features=(
            "carrier",
            "origin",
            "dest",
            "route",
            "distance_band",
        ),
    )


def _trial():
    return {
        "learning_rate": 0.01,
        "batch_size": 4,
        "hidden_dims": [8, 4],
        "dropout": 0.0,
        "weight_decay": 0.0,
    }


def test_early_stopping_restores_best_validation_ap(monkeypatch):
    from flight_delay.deep_learning import train

    data = _dataset()
    encoder = DLTensorEncoder(data)
    train_batch = encoder.transform_partition("train")
    validation = encoder.transform_partition("validation")
    monkeypatch.setattr(train, "_validation_ap", lambda *args: 0.5)
    result = _fit_trial(
        encoder,
        train_batch,
        validation,
        _trial(),
        seed=17,
        epochs=8,
        patience=2,
        min_delta=0.0,
        optimizer_name="adam",
        device=torch.device("cpu"),
    )

    assert result["best_epoch"] == 1
    assert result["epochs_run"] == 3
    assert result["validation_average_precision"] == 0.5
    assert result["state_dict"]


def test_training_reloads_checkpoint_and_exports_project_schema(tmp_path, monkeypatch):
    from flight_delay.deep_learning import train

    dataset = _dataset()
    monkeypatch.setattr(train, "build_dl_dataset", lambda _: dataset)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    output_dir = tmp_path / "reports" / "dl"
    config = {
        "data_path": "ignored.csv",
        "output_dir": str(output_dir),
        "seed": 42,
        "device": "cpu",
        "training": {
            "optimizer": "adamw",
            "epochs": 3,
            "patience": 2,
            "min_delta": 0.0,
        },
        "search": {"trials": [_trial()]},
    }
    config_path = config_dir / "dl_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_training(config_path)
    repeat = run_training(config_path, output_dir=tmp_path / "repeat")
    np.testing.assert_array_equal(
        result["validation_predictions"].y_probability,
        repeat["validation_predictions"].y_probability,
    )
    expected_columns = [
        "flight_id",
        "flight_date",
        "y_true",
        "y_probability",
        "partition",
    ]
    for key, partition in (
        ("calibration_predictions", "train_calibration"),
        ("validation_predictions", "validation"),
    ):
        predictions = result[key]
        assert predictions.columns.tolist() == expected_columns
        assert predictions.partition.eq(partition).all()
        assert predictions.flight_id.is_unique
        assert predictions.y_probability.between(0, 1).all()
        assert predictions.y_true.isin([0, 1]).all()
        saved = pd.read_csv(output_dir / f"{partition}_predictions.csv")
        pd.testing.assert_frame_equal(saved, predictions, check_dtype=False)
        round_trip = load_prediction_csv(
            output_dir / f"{partition}_predictions.csv", partition=partition
        )
        np.testing.assert_array_equal(
            round_trip.y_probability.to_numpy(),
            predictions.y_probability.to_numpy(),
        )

    encoder = DLTensorEncoder(dataset)
    restored = load_best_checkpoint(result["checkpoint"], encoder)
    validation_batch = encoder.transform_partition("validation")
    restored_predictions = _predict_probabilities(
        restored, validation_batch, batch_size=4, device=torch.device("cpu")
    )
    np.testing.assert_array_equal(
        restored_predictions, result["validation_predictions"].y_probability
    )
    assert (output_dir / "best_hyperparameters.yaml").is_file()
    assert (output_dir / "tuning_results.csv").is_file()
    summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    for key, filename in (
        ("checkpoint", "best_model.pt"),
        ("validation_predictions", "validation_predictions.csv"),
        ("train_calibration_predictions", "train_calibration_predictions.csv"),
    ):
        assert summary[key] == f"reports/dl/{filename}"
        assert (tmp_path / summary[key]).is_file()

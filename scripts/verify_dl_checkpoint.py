"""Read-only, real-data replay of the committed MLP checkpoint (no retraining).

Run in the existing repository with the DL extra installed:
    python -m pip install -e '.[dl]'
    python scripts/verify_dl_checkpoint.py

This reconstructs the frozen *validation* predictions. It does NOT open or
score the final test partition. It writes no model or training artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from flight_delay.deep_learning.dataset import build_dl_dataset
from flight_delay.deep_learning.encoding import DLTensorEncoder
from flight_delay.deep_learning.train import load_best_checkpoint, _predict_probabilities
from flight_delay.evaluation.predictions import load_prediction_csv

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATA_SHA256 = "8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa"


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root: Path = ROOT, *, batch_size: int = 1024,
           atol: float = 1e-6, rtol: float = 1e-5) -> dict:
    """Replay validation without any re-fitting or test scoring."""
    root = Path(root)
    data_file = root / "data/raw/flights.csv"
    ckpt = root / "reports/dl/best_model.pt"
    expected_file = root / "reports/dl/validation_predictions.csv"
    digest = _hash(data_file)
    if digest != ORIGINAL_DATA_SHA256:
        raise ValueError(f"Dataset SHA-256 differs from canonical file: {digest}")
    dataset = build_dl_dataset(data_file)
    encoder = DLTensorEncoder(dataset)
    model = load_best_checkpoint(ckpt, encoder, device="cpu")
    validation = encoder.transform_partition("validation")
    with torch.inference_mode():
        probabilities = _predict_probabilities(
            model, validation, batch_size, torch.device("cpu"))
    actual = pd.DataFrame({
        "flight_id": validation.metadata["flight_id"].astype("string").to_numpy(),
        "flight_date": pd.to_datetime(validation.metadata["flight_date"])
        .dt.strftime("%Y-%m-%d").to_numpy(),
        "y_true": validation.labels.numpy().astype(int),
        "y_probability": probabilities,
    }).sort_values("flight_id").reset_index(drop=True)
    reference = load_prediction_csv(expected_file, partition="validation")
    reference = reference.sort_values("flight_id").reset_index(drop=True)
    for name in ("flight_id", "flight_date", "y_true"):
        if not np.array_equal(actual[name].to_numpy(), reference[name].to_numpy()):
            raise AssertionError(f"Saved vs recomputed validation {name} misaligned")
    expected = reference["y_probability"].to_numpy(dtype=float)
    observed = actual["y_probability"].to_numpy(dtype=float)
    np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol)
    y = actual["y_true"].to_numpy(dtype=int)
    ap = float(average_precision_score(y, observed))
    roc = float(roc_auc_score(y, observed))
    summary_file = root / "reports/dl/training_summary.json"
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        stored_ap = float(summary["best_hyperparameters"]["validation_average_precision"])
        if not np.isclose(ap, stored_ap, atol=1e-5, rtol=0):
            raise AssertionError(f"Replayed AP {ap} does not match recorded AP {stored_ap}")
    return {
        "stage": "validation_only_checkpoint_replay",
        "test_scored": False,
        "rows": len(y),
        "dataset_sha256": digest,
        "checkpoint_sha256": _hash(ckpt),
        "validation_ap": ap,
        "validation_roc_auc": roc,
        "maximum_probability_error": float(np.max(np.abs(observed - expected))),
        "mean_probability_error": float(np.mean(np.abs(observed - expected))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()

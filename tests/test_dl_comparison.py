"""Regression tests for Topic 5: leakage-safe AML/DL validation comparisons."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_aml_dl import (  # noqa: E402
    Sources,
    compare,
    load_predictions,
    threshold_at_min_precision,
)


def _write(path: Path, labels: np.ndarray, probabilities: np.ndarray,
           *, partition: str, start_id: int, start_date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(labels)
    pd.DataFrame({
        "flight_id": [f"{start_id + i:06d}" for i in range(count)],
        "flight_date": pd.date_range(start_date, periods=count, freq="D")
        .strftime("%Y-%m-%d"),
        "y_true": labels,
        "y_probability": probabilities,
        "partition": partition,
    }).to_csv(path, index=False)


def _fixture(tmp_path: Path) -> Sources:
    rng = np.random.default_rng(17)
    labels = np.array([0, 1] * 20, dtype=int)
    p = np.clip(0.20 + labels * 0.38 + rng.normal(0, .12, len(labels)), .01, .99)
    cal_labels = np.array([1, 0] * 10, dtype=int)
    cal_p = np.clip(0.25 + cal_labels * 0.38 + rng.normal(0, .12, len(cal_labels)), .01, .99)
    s = Sources.from_root(tmp_path)
    for key in ("logistic_raw", "logistic_isotonic", "xgboost_raw",
                "xgboost_isotonic", "mlp_raw", "hybrid"):
        base = np.clip(p + {"logistic_raw": .05, "logistic_isotonic": -.06,
                            "xgboost_raw": .10, "xgboost_isotonic": -.03,
                            "mlp_raw": 0, "hybrid": -.02}[key], .01, .99)
        _write(getattr(s, key), labels, base, partition="validation",
               start_id=10000, start_date="2013-03-01")
    _write(s.mlp_train_calibration, cal_labels, cal_p,
           partition="train_calibration", start_id=2000, start_date="2013-02-01")
    return s


def test_full_comparison_writes_validated_metrics_and_figures(tmp_path):
    s = _fixture(tmp_path)
    output = tmp_path / "reports/dl/evaluation_shubhaang"
    result = compare(s, output, bootstrap=4, make_plots=True)
    assert result["stage"] == "validation_only"
    assert result["test_used"] is False
    assert result["n"] == 40
    assert result["train_calibration_n"] == 20
    assert result["selected_mlp_calibration"] in (
        "mlp_raw", "mlp_sigmoid", "mlp_isotonic")
    assert len(result["models"]) == 8
    for name in ("summary.json", "metrics.csv", "figures/precision_recall.png",
                 "figures/roc.png", "figures/reliability.png",
                 "figures/ap_comparison.png", "figures/brier_comparison.png"):
        assert (output / name).is_file(), name
    assert len(list((output / "figures").glob("confusion_*.png"))) >= 3
    pd.read_csv(output / "metrics.csv").set_index("model").loc["mlp_raw"]
    assert result["paired_ap_bootstrap"]["mlp_raw_minus_logistic_raw"]["valid"] > 0


def test_rejects_a_mislabeled_validation_flight(tmp_path):
    s = _fixture(tmp_path)
    bad = pd.read_csv(s.hybrid)
    bad.loc[0, "y_true"] = 1 - bad.loc[0, "y_true"]
    bad.to_csv(s.hybrid, index=False)
    with pytest.raises(ValueError, match="Unaligned"):
        compare(s, tmp_path / "out", bootstrap=0, make_plots=False)


def test_rejects_calibration_and_validation_overlap(tmp_path):
    s = _fixture(tmp_path)
    cal = pd.read_csv(s.mlp_train_calibration, dtype={"flight_id": "string"})
    val = pd.read_csv(s.mlp_raw, dtype={"flight_id": "string"})
    cal.loc[0, "flight_id"] = val.loc[0, "flight_id"]
    cal.to_csv(s.mlp_train_calibration, index=False)
    with pytest.raises(ValueError, match="overlap"):
        compare(s, tmp_path / "out", bootstrap=0, make_plots=False)


def test_precision_unattainable_is_explicit_not_fabricated():
    y = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    p = np.full(len(y), 0.2)
    result = threshold_at_min_precision(y, p, 0.25)
    assert result == {"attainable": False, "threshold": None,
                      "precision": None, "recall": None, "flagged": 0}


def test_nonfinite_or_duplicate_predictions_fail(tmp_path):
    s = _fixture(tmp_path)
    csv = pd.read_csv(s.mlp_raw)
    csv.loc[2, "y_probability"] = np.inf
    csv.to_csv(s.mlp_raw, index=False)
    with pytest.raises(ValueError, match="invalid probabilities"):
        load_predictions(s.mlp_raw, "validation")
    s = _fixture(tmp_path)
    csv = pd.read_csv(s.mlp_raw, dtype={"flight_id": "string"})
    csv.loc[2, "flight_id"] = csv.loc[1, "flight_id"]
    csv.to_csv(s.mlp_raw, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_predictions(s.mlp_raw, "validation")


def test_script_ignores_test_partition_even_if_present_and_corrupt(tmp_path):
    s = _fixture(tmp_path)
    bad_test = tmp_path / "reports/dl/test_predictions.csv"
    bad_test.write_text("THIS FILE MUST NEVER BE OPENED", encoding="utf-8")
    result = compare(s, tmp_path / "out", bootstrap=0, make_plots=False)
    assert not result["test_used"]
    assert result["n"] == 40

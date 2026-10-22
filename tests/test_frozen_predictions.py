"""Validation-only frozen replay using tiny fitted fixtures, never real training."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from flight_delay.evaluation.calibration import ScoreCalibrator
from flight_delay.features.model_features import select_model_features
from scripts import verify_frozen_predictions as verifier


@pytest.fixture
def frozen_fixture(tmp_path):
    dates = pd.Series(pd.date_range("2013-01-01", periods=40).repeat(2)).reset_index(
        drop=True
    )
    n = len(dates)
    hour = np.tile([5, 7], 40)
    raw = pd.DataFrame(
        {
            "id": [f"{i:04d}" for i in range(n)],
            "year": dates.dt.year,
            "month": dates.dt.month,
            "day": dates.dt.day,
            "arr_delay": np.tile([0, 30], 40),
            "hour": hour,
            "minute": 0,
            "sched_dep_time": hour * 100,
            "sched_arr_time": hour * 100 + 200,
            "carrier": np.tile(["AA", "UA"], 40),
            "origin": "JFK",
            "dest": "BOS",
            "distance": 200,
        }
    )
    data = tmp_path / "flights.csv"
    raw.to_csv(data, index=False)
    train, val, _ = verifier.build_data_pipeline(data)
    X = select_model_features(train.iloc[:44])
    y = train.iloc[:44].is_delayed
    numeric = X.select_dtypes(include="number").columns.tolist()
    output = tmp_path / "evaluation"
    output.mkdir()
    manifest = {
        "schema_version": 1,
        "data_sha256": verifier.sha256(data),
        "seed": 42,
        "versions": {
            p: verifier.importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "xgboost"]
        },
        "feature_columns": list(X),
        "models": {},
        "model_parameters": {},
    }
    frozen = {}
    for name, estimator in [
        ("fixture_constant", DummyClassifier(strategy="prior", random_state=42)),
        ("fixture_varying", DecisionTreeClassifier(max_depth=2, random_state=42)),
    ]:
        model = Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [
                            (
                                "numeric",
                                SimpleImputer(strategy="constant", fill_value=-1),
                                numeric,
                            )
                        ]
                    ),
                ),
                ("model", estimator),
            ]
        ).fit(X, y)
        calibrator = ScoreCalibrator("raw").fit(
            [0, 1], [0.2, 0.8], partition="train_calibration"
        )
        frozen[name] = {"model": model, "calibrator": calibrator, "threshold": 0.5}
        manifest["models"][name] = {
            "calibration": "raw",
            "threshold": 0.5,
            "strategies": {"max_f1": {"threshold": 0.5}},
        }
        manifest["model_parameters"][name] = {"random_state": 42}
        frame = pd.DataFrame(
            {
                "flight_id": val.id,
                "flight_date": pd.to_datetime(
                    val[["year", "month", "day"]]
                ).dt.strftime("%Y-%m-%d"),
                "y_true": val.is_delayed.astype(int),
                "y_probability": model.predict_proba(select_model_features(val))[:, 1],
                "partition": "validation",
            }
        )
        # Reverse row order to ensure replay aligns by IDs, not CSV row position.
        frame.iloc[::-1].to_csv(output / f"{name}_raw_validation.csv", index=False)
    joblib.dump(frozen, output / "frozen.joblib")
    manifest["artifact_sha256"] = verifier.sha256(output / "frozen.joblib")
    (output / "selection.json").write_text(json.dumps(manifest))
    (output / "test_metrics.json").write_text("must not be read")
    return data, output


def snapshot(data, output):
    files = [data, *[p for p in output.rglob("*") if p.is_file()]]
    return {p: p.read_bytes() for p in files}


def test_deterministic_read_only_replay_never_fits_or_reads_test_metrics(
    frozen_fixture, monkeypatch
):
    data, output = frozen_fixture
    before = snapshot(data, output)

    def forbidden(*a, **k):
        pytest.fail("Verification attempted fitting")

    for cls, method in [
        (Pipeline, "fit"),
        (DummyClassifier, "fit"),
        (DecisionTreeClassifier, "fit"),
        (ColumnTransformer, "fit_transform"),
        (ScoreCalibrator, "fit"),
    ]:
        monkeypatch.setattr(cls, method, forbidden)
    original = Path.open

    def guarded(path, *a, **k):
        if path.name == "test_metrics.json":
            pytest.fail("Verification opened test metrics")
        return original(path, *a, **k)

    monkeypatch.setattr(Path, "open", guarded)
    first = verifier.verify(data, output, sample_size=5)
    second = verifier.verify(data, output, sample_size=5)
    assert first == second and first["sample_size"] == 5 and first["status"] == "PASS"
    assert len(set(first["sample_ids"])) == 5
    assert all(
        value["classes_match"] and value["max_absolute_probability_error"] == 0
        for value in first["models"].values()
    )
    # Guard the forbidden file while checking other files, then restore for byte checks.
    monkeypatch.setattr(Path, "open", original)
    assert snapshot(data, output) == before


@pytest.mark.parametrize(
    "change", ["probabilities", "labels", "ids", "duplicates", "dates", "partition"]
)
def test_saved_prediction_corruption_fails(frozen_fixture, change):
    data, output = frozen_fixture
    path = output / "fixture_constant_raw_validation.csv"
    frame = pd.read_csv(
        path, dtype={"flight_id": "string"}, float_precision="round_trip"
    )
    if change == "probabilities":
        frame.loc[0, "y_probability"] = 0.7
    if change == "labels":
        frame.loc[0, "y_true"] = 1 - frame.loc[0, "y_true"]
    if change == "ids":
        frame.loc[0, "flight_id"] = "9999"
    if change == "duplicates":
        frame.loc[1, "flight_id"] = frame.loc[0, "flight_id"]
    if change == "dates":
        frame.loc[0, "flight_date"] = "2013-01-01"
    if change == "partition":
        frame.loc[0, "partition"] = "test"
    frame.to_csv(path, index=False)
    before = snapshot(data, output)
    with pytest.raises(ValueError):
        verifier.verify(data, output, sample_size=100)
    assert snapshot(data, output) == before


def test_class_comparison_has_no_tolerance(frozen_fixture):
    data, output = frozen_fixture
    path = output / "fixture_constant_raw_validation.csv"
    frame = pd.read_csv(
        path, dtype={"flight_id": "string"}, float_precision="round_trip"
    )
    frame.loc[0, "y_probability"] = np.nextafter(0.5, 0.0)
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="classes disagree"):
        verifier.verify(data, output, sample_size=100)


@pytest.mark.parametrize("field", ["data_sha256", "artifact_sha256"])
def test_hashes_checked_before_model_loading(frozen_fixture, field, monkeypatch):
    data, output = frozen_fixture
    path = output / "selection.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "0" * 64
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        verifier.joblib, "load", lambda *a: pytest.fail("Loaded hash-mismatched model")
    )
    with pytest.raises(ValueError, match="SHA-256"):
        verifier.verify(data, output)


@pytest.mark.parametrize(
    "change", ["threshold", "calibration", "feature_schema", "model_setting"]
)
def test_frozen_settings_corruption_fails(frozen_fixture, change):
    data, output = frozen_fixture
    path = output / "selection.json"
    manifest = json.loads(path.read_text())
    if change == "threshold":
        manifest["models"]["fixture_constant"]["threshold"] = 0.6
    if change == "calibration":
        manifest["models"]["fixture_constant"]["calibration"] = "isotonic"
    if change == "feature_schema":
        manifest["feature_columns"] = manifest["feature_columns"][::-1]
    if change == "model_setting":
        manifest["model_parameters"]["fixture_constant"]["random_state"] = 123
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch|differs"):
        verifier.verify(data, output)


def test_verifier_rejects_test_partition_and_nonpositive_sample(frozen_fixture):
    data, output = frozen_fixture
    with pytest.raises(ValueError, match="validation"):
        verifier.verify(data, output, partition="test")
    with pytest.raises(ValueError, match="positive"):
        verifier.verify(data, output, sample_size=0)


def test_cli_pass_and_nonzero_failure(frozen_fixture, capsys):
    data, output = frozen_fixture
    args = [
        "--data",
        str(data),
        "--output-dir",
        str(output),
        "--partition",
        "validation",
        "--sample-size",
        "100",
    ]
    assert verifier.main(args) == 0
    (output / "frozen.joblib").unlink()
    assert verifier.main(args) == 1
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("change", ["empty_models", "empty_versions"])
def test_verifier_rejects_empty_required_provenance(frozen_fixture, change):
    data, output = frozen_fixture
    path = output / "selection.json"
    manifest = json.loads(path.read_text())
    if change == "empty_models":
        joblib.dump({}, output / "frozen.joblib")
        manifest["models"] = {}
        manifest["artifact_sha256"] = verifier.sha256(output / "frozen.joblib")
    else:
        manifest["versions"] = {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model|versions"):
        verifier.verify(data, output)

"""Independent-auditor contracts using small synthetic saved-run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from flight_delay.evaluation.metrics import evaluate
from flight_delay.evaluation.thresholds import select_thresholds, threshold_table
from scripts import audit_evaluation as audit


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


@pytest.fixture
def saved_run(tmp_path):
    """Produce actual-schema fixture files, not a real flight-model training run."""
    output = tmp_path / "evaluation"
    output.mkdir()
    days = pd.Series(pd.date_range("2013-01-01", periods=40).repeat(2)).reset_index(
        drop=True
    )
    raw = pd.DataFrame(
        {
            "id": np.arange(len(days)),
            "year": days.dt.year,
            "month": days.dt.month,
            "day": days.dt.day,
            "arr_delay": np.tile([0, 30], len(days) // 2),
        }
    )
    data = tmp_path / "flights.csv"
    raw.to_csv(data, index=False)
    # Independent, explicit original date windows for this 40-date fixture.
    canonical = pd.DataFrame(
        {
            "flight_id": raw.id.astype("string"),
            "flight_date": days.dt.strftime("%Y-%m-%d"),
            "y_true": np.tile([0, 1], 40),
        }
    )
    partitions = {
        "base_train": canonical.iloc[:44],
        "train_calibration": canonical.iloc[44:56],
        "validation": canonical.iloc[56:68],
        "test": canonical.iloc[68:],
    }
    windows = {
        k: {"n": len(f), "first": f.flight_date.min(), "last": f.flight_date.max()}
        for k, f in partitions.items()
        if k != "test"
    }
    manifest = {
        "schema_version": 1,
        "data_sha256": audit.sha256(data),
        "versions": {
            p: audit.importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "xgboost"]
        },
        "feature_columns": ["x"],
        "windows": windows,
        "calibration_rule": "minimum validation Brier, log loss tie-break",
        "threshold_rule": "maximum validation F1; higher threshold tie-break",
        "models": {},
        "winner": "logistic_balanced",
    }
    calibration_metrics, validation_metrics, test_metrics, frozen = {}, {}, {}, {}
    for index, name in enumerate(["logistic_balanced", "xgboost_tuned_full"]):
        val = partitions["validation"].copy().reset_index(drop=True)
        val["partition"] = "validation"
        variants = {
            "raw": np.where(val.y_true == 1, 0.65 + index * 0.1, 0.15 + index * 0.1),
            "sigmoid": np.where(val.y_true == 1, 0.6, 0.4),
            "isotonic": np.where(val.y_true == 1, 0.55, 0.45),
        }
        for method, p in variants.items():
            frame = val.assign(y_probability=p)
            frame.to_csv(output / f"{name}_{method}_validation.csv", index=False)
            # Existing producer builds fixture reports; the auditor must not call it.
            calibration_metrics[f"{name}_{method}"] = evaluate(
                frame.y_true, frame.y_probability
            )
        frame = val.assign(y_probability=variants["raw"])
        strategies = select_thresholds(
            threshold_table(frame.y_true, frame.y_probability)
        )
        threshold = strategies["max_f1"]["threshold"]
        assert threshold == pytest.approx(0.65 + index * 0.1)
        manifest["models"][name] = {
            "calibration": "raw",
            "threshold": threshold,
            "strategies": strategies,
        }
        validation_metrics[name] = evaluate(
            frame.y_true, frame.y_probability, threshold
        )
        test = partitions["test"].copy().reset_index(drop=True)
        test["partition"] = "test"
        test["y_probability"] = np.where(test.y_true == 1, 0.8, 0.2)
        test.to_csv(output / f"{name}_test.csv", index=False)
        test_metrics[name] = evaluate(test.y_true, test.y_probability, threshold)
        # Only a tiny synthetic dummy is fitted; no project classifier is trained.
        model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
        frozen[name] = {
            "model": model,
            "calibrator": SimpleNamespace(method="raw", estimator=None),
            "threshold": threshold,
        }
    joblib.dump(frozen, output / "frozen.joblib")
    manifest["artifact_sha256"] = audit.sha256(output / "frozen.joblib")
    for name, value in [
        ("selection.json", manifest),
        ("calibration_metrics.json", calibration_metrics),
        ("validation_metrics.json", validation_metrics),
        ("test_metrics.json", test_metrics),
    ]:
        write_json(output / name, value)
    return data, output


def failed_names(report):
    return [c["name"] for c in report["checks"] if not c["passed"]]


def mutate_csv(path, mutation):
    frame = pd.read_csv(path, float_precision="round_trip")
    changed = mutation(frame)
    (frame if changed is None else changed).to_csv(path, index=False)


def test_correct_report_and_independence(saved_run, monkeypatch, capsys):
    import flight_delay.evaluation.metrics as implementation

    monkeypatch.setattr(
        implementation,
        "evaluate",
        lambda *a, **k: pytest.fail("Auditor called producer metric helper"),
    )
    data, output = saved_run
    report = audit.run_audit(data, output)
    assert report["status"] == "PASS" and not report["metric_discrepancies"]
    assert not failed_names(report)
    assert report["observed_row_counts"]["dataset"] == {
        "base_train": 44,
        "train_calibration": 12,
        "validation": 12,
        "test": 12,
    }
    assert report["dataset_sha256"] == audit.sha256(data)
    assert report["artifact_sha256"] == audit.sha256(output / "frozen.joblib")
    assert json.loads(Path(report["report_path"]).read_text()) == report
    assert "UNVERIFIED" in capsys.readouterr().out


def test_tampered_in_range_probabilities(saved_run):
    data, output = saved_run
    path = output / "logistic_balanced_raw_validation.csv"
    mutate_csv(path, lambda f: f.assign(y_probability=np.full(len(f), 0.2)))
    report = audit.run_audit(data, output)
    assert report["status"] == "FAIL" and report["metric_discrepancies"]


@pytest.mark.parametrize("value", [-0.1, 1.1, np.inf, np.nan])
def test_invalid_probabilities(saved_run, value):
    data, output = saved_run

    def change(frame):
        frame.loc[0, "y_probability"] = value

    mutate_csv(output / "logistic_balanced_test.csv", change)
    report = audit.run_audit(data, output)
    assert report[
        "status"
    ] == "FAIL" and "test_predictions:logistic_balanced" in failed_names(report)


@pytest.mark.parametrize(
    "change",
    [
        "labels",
        "missing_labels",
        "dates",
        "partition",
        "missing_rows",
        "duplicate_ids",
        "extra_ids",
    ],
)
def test_prediction_identity_failures(saved_run, change):
    data, output = saved_run

    def mutate(frame):
        if change == "labels":
            frame.loc[0, "y_true"] = 1 - frame.loc[0, "y_true"]
        elif change == "missing_labels":
            frame.loc[0, "y_true"] = np.nan
        elif change == "dates":
            frame.loc[0, "flight_date"] = "2014-01-01"
        elif change == "partition":
            frame.loc[0, "partition"] = "validation"
        elif change == "missing_rows":
            return frame.iloc[:-1]
        elif change == "duplicate_ids":
            frame.loc[1, "flight_id"] = frame.loc[0, "flight_id"]
        elif change == "extra_ids":
            frame.loc[0, "flight_id"] = 9999

    mutate_csv(output / "xgboost_tuned_full_test.csv", mutate)
    report = audit.run_audit(data, output)
    assert report[
        "status"
    ] == "FAIL" and "test_predictions:xgboost_tuned_full" in failed_names(report)


@pytest.mark.parametrize(
    "missing",
    [
        "selection.json",
        "frozen.joblib",
        "validation_metrics.json",
        "test_metrics.json",
        "calibration_metrics.json",
        "logistic_balanced_test.csv",
        "xgboost_tuned_full_sigmoid_validation.csv",
    ],
)
def test_missing_files_fail_and_report(saved_run, missing):
    data, output = saved_run
    (output / missing).unlink()
    report = audit.run_audit(data, output)
    assert report["status"] == "FAIL" and Path(report["report_path"]).exists()
    assert any(
        missing in c["name"] or missing in c["detail"]
        for c in report["checks"]
        if not c["passed"]
    )


@pytest.mark.parametrize("field", ["data_sha256", "artifact_sha256"])
def test_mismatched_hashes(saved_run, field, monkeypatch):
    data, output = saved_run
    path = output / "selection.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "0" * 64
    write_json(path, manifest)
    if field == "artifact_sha256":
        monkeypatch.setattr(
            audit.joblib,
            "load",
            lambda *a: pytest.fail("Deserialized mismatched artifact"),
        )
    report = audit.run_audit(data, output)
    assert report["status"] == "FAIL"
    assert (
        "dataset_hash_matches_selection"
        if field == "data_sha256"
        else "frozen_hash_matches_selection"
    ) in failed_names(report)


def test_altered_selection_threshold(saved_run):
    data, output = saved_run
    path = output / "selection.json"
    manifest = json.loads(path.read_text())
    manifest["models"]["logistic_balanced"]["threshold"] = 0.9
    write_json(path, manifest)
    report = audit.run_audit(data, output)
    assert report["status"] == "FAIL"
    assert "frozen_configuration_and_fitted_state" in failed_names(report)
    assert "validation_only_selection:logistic_balanced" in failed_names(report)


def test_altered_frozen_threshold_even_with_updated_hash(saved_run):
    data, output = saved_run
    path = output / "frozen.joblib"
    frozen = joblib.load(path)
    frozen["logistic_balanced"]["threshold"] = 0.9
    joblib.dump(frozen, path)
    manifest = json.loads((output / "selection.json").read_text())
    manifest["artifact_sha256"] = audit.sha256(path)
    write_json(output / "selection.json", manifest)
    report = audit.run_audit(data, output)
    assert report[
        "status"
    ] == "FAIL" and "frozen_configuration_and_fitted_state" in failed_names(report)


def test_wrong_calibration_choice(saved_run):
    data, output = saved_run
    path = output / "selection.json"
    manifest = json.loads(path.read_text())
    manifest["models"]["logistic_balanced"]["calibration"] = "isotonic"
    write_json(path, manifest)
    report = audit.run_audit(data, output)
    assert report[
        "status"
    ] == "FAIL" and "validation_only_selection:logistic_balanced" in failed_names(
        report
    )


def test_no_overwrites_on_success_failure_or_repeat(saved_run):
    data, output = saved_run
    sentinel = output / "audit_report_existing.json"
    sentinel.write_text("preserve original report")
    before = {p: p.read_bytes() for p in output.iterdir() if p.is_file()}
    before[data] = data.read_bytes()
    first = audit.run_audit(data, output)
    second = audit.run_audit(data, output)
    assert first["status"] == second["status"] == "PASS"
    assert first["report_path"] != second["report_path"]
    for path, original in before.items():
        assert path.read_bytes() == original
    mutate_csv(output / "logistic_balanced_test.csv", lambda f: f.iloc[:-1])
    modified = {p: p.read_bytes() for p in output.iterdir() if p.is_file()}
    assert audit.run_audit(data, output)["status"] == "FAIL"
    for path, original in modified.items():
        assert path.read_bytes() == original


def test_test_metrics_not_read_until_validation_replay(saved_run, monkeypatch):
    data, output = saved_run
    decisions = []
    original_check = audit.check_decision
    original_load = audit.load_json

    def check(*args, **kwargs):
        result = original_check(*args, **kwargs)
        decisions.append(args[0])
        return result

    def load(path):
        if path.name == "test_metrics.json":
            assert len(decisions) == 2
        return original_load(path)

    monkeypatch.setattr(audit, "check_decision", check)
    monkeypatch.setattr(audit, "load_json", load)
    # A deliberately false test metric must fail arithmetic, not alter decisions.
    path = output / "test_metrics.json"
    values = json.loads(path.read_text())
    values["logistic_balanced"]["f1"] = 0.123
    write_json(path, values)
    report = audit.run_audit(data, output)
    assert report["status"] == "FAIL" and len(decisions) == 2
    assert all(
        c["passed"]
        for c in report["checks"]
        if c["name"].startswith("validation_only_")
    )
    assert "test_metrics:logistic_balanced.f1" in failed_names(report)


def test_cli_exit_codes(saved_run):
    data, output = saved_run
    args = ["--data", str(data), "--output-dir", str(output)]
    assert audit.main(args) == 0
    (output / "test_metrics.json").unlink()
    assert audit.main(args) == 1


def test_round_trip_probability_boundary(tmp_path):
    threshold = 0.34357316087308004
    expected = pd.DataFrame(
        {
            "flight_id": pd.Series(["1", "2"], dtype="string"),
            "flight_date": ["2013-01-01"] * 2,
            "y_true": [0, 1],
        }
    )
    frame = expected.assign(partition="validation", y_probability=[0.1, threshold])
    path = tmp_path / "boundary.csv"
    frame.to_csv(path, index=False)
    restored = audit.read_predictions(path, expected, "validation")
    assert restored.y_probability.iloc[1] == threshold
    assert audit.calculate_metrics(restored, threshold)["f1"] == 1.0


def test_brier_tie_uses_log_loss():
    frame = pd.DataFrame(
        {
            "partition": ["validation"] * 4,
            "y_true": [0, 1, 0, 1],
            "y_probability": [0.2, 0.8, 0.2, 0.8],
        }
    )
    threshold, _ = audit.validation_f1_choice(frame)
    actual = audit.calculate_metrics(frame, threshold)
    selected = {
        "calibration": "sigmoid",
        "threshold": threshold,
        "strategies": {"max_f1": actual},
    }
    candidates = {
        "raw": {"brier": 0.1, "log_loss": 0.5},
        "sigmoid": {"brier": 0.1, "log_loss": 0.4},
        "isotonic": {"brier": 0.2, "log_loss": 0.3},
    }
    assert audit.check_decision("fixture", selected, candidates, frame)["f1"] == 1.0


def test_validation_replay_rejects_test_partition():
    frame = pd.DataFrame(
        {"partition": ["test"] * 2, "y_true": [0, 1], "y_probability": [0.2, 0.8]}
    )
    with pytest.raises(ValueError, match="validation"):
        audit.validation_f1_choice(frame)


@pytest.mark.parametrize("filename", ["validation_metrics.json", "selection.json"])
def test_corrupt_json_produces_failure_report(saved_run, filename):
    data, output = saved_run
    (output / filename).write_text('{"broken":')
    report = audit.run_audit(data, output)
    assert report["status"] == "FAIL" and Path(report["report_path"]).exists()
    assert filename in failed_names(report)


def test_malformed_window_metadata_fails_cleanly(saved_run):
    data, output = saved_run
    path = output / "selection.json"
    manifest = json.loads(path.read_text())
    manifest["windows"] = []
    write_json(path, manifest)
    report = audit.run_audit(data, output)
    assert report["status"] == "FAIL"
    assert "selection_schema_and_rules" in failed_names(report)
    assert Path(report["report_path"]).exists()


def test_report_exclusive_create_preserves_existing_file(saved_run, monkeypatch):
    data, output = saved_run
    sentinel = output / "audit_report_collision.json"
    sentinel.write_bytes(b"original report must survive")
    monkeypatch.setattr(audit.uuid, "uuid4", lambda: SimpleNamespace(hex="collision"))
    report = audit.run_audit(data, output)
    assert report["status"] == "FAIL"
    assert "new_report_written" in failed_names(report)
    assert sentinel.read_bytes() == b"original report must survive"


def test_modified_dataset_hash_fails(saved_run):
    data, output = saved_run
    with data.open("a") as handle:
        handle.write("\n")
    report = audit.run_audit(data, output)
    assert report["status"] == "FAIL"
    assert "dataset_hash_matches_selection" in failed_names(report)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_metrics_convention(label):
    frame = pd.DataFrame({"y_true": [label] * 2, "y_probability": [0.5, 0.5]})
    values = audit.calculate_metrics(frame, 0.5)
    assert values["roc_auc"] is None and values["average_precision"] is None
    assert np.isfinite(values["log_loss"])
    assert values["n"] == 2


@pytest.mark.parametrize("phase", ["validation", "final_test"])
def test_auditor_rejects_incomplete_publication(saved_run, phase):
    data, output = saved_run
    marker = output / f".{phase}_started.json"
    write_json(marker, {"phase": phase, "status": "started"})
    report = audit.run_audit(data, output)
    assert report["status"] == "FAIL"
    assert f"publication_state:{phase}" in failed_names(report)


def test_auditor_accepts_completed_phase_markers(saved_run):
    data, output = saved_run
    for phase in ("validation", "final_test"):
        for status in ("started", "complete"):
            write_json(
                output / f".{phase}_{status}.json", {"phase": phase, "status": status}
            )
    report = audit.run_audit(data, output)
    assert report["status"] == "PASS"

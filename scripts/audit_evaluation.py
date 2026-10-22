"""Independently verify saved evaluation artifacts without fitting or predicting.

Run from the repository root with the project's environment::

    python scripts/audit_evaluation.py --data data/raw/flights.csv \
        --output-dir reports/evaluation

Rebuilds the original target/date partitions directly from raw CSV columns,
without importing evaluation helpers. CSV probabilities use round-trip parsing.
Every invocation exclusively creates a new audit_report_<uuid>.json; existing
files are only read. Load only trusted local joblib artifacts. Validation-only
replay establishes consistency of recorded decisions, not an immutable history
of which labels a previous process actually consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils.validation import check_is_fitted

METHODS = ("raw", "sigmoid", "isotonic")
PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")
ATOL = 1e-12
RTOL = 1e-10


def sha256(path: Path) -> str:
    """Hash a file in bounded-memory chunks; never open it for writing."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Raise a clear contract error rather than continuing with partial data."""
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    """Read strict JSON objects, rejecting NaN, infinity and duplicate keys."""

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    with path.open() as handle:
        value = json.load(handle, object_pairs_hook=pairs, parse_constant=invalid)
    require(
        isinstance(value, dict) and bool(value),
        f"Expected nonempty JSON object: {path}",
    )
    return value


def rebuild_partitions(data: Path) -> dict[str, pd.DataFrame]:
    """Reconstruct original eligibility, >=15 target and whole-date boundaries.

    No model features or probabilities are needed to verify observation identity.
    Uses the original 70%/15% date rule, then the original 80% training-date base
    fit cutoff. Date/ID errors fail before any rows can be silently excluded.
    """
    raw = pd.read_csv(data, dtype={"id": "string"}, float_precision="round_trip")
    required = {"id", "year", "month", "day", "arr_delay"}
    require(
        required <= set(raw), f"Dataset missing columns: {sorted(required - set(raw))}"
    )
    frame = raw.loc[raw.arr_delay.notna()].copy()
    require(not frame.empty, "No eligible arrival-delay observations")
    require(
        not frame.id.isna().any() and frame.id.is_unique,
        "Eligible dataset IDs must be nonmissing and unique",
    )
    require(
        not frame[["year", "month", "day"]].isna().any().any(),
        "Dataset contains missing date components",
    )
    dates = pd.to_datetime(frame[["year", "month", "day"]], errors="raise")
    require(not dates.isna().any(), "Dataset contains NaT dates")
    arrival = pd.to_numeric(frame.arr_delay, errors="raise")
    require(np.isfinite(arrival).all(), "Arrival delays must be finite")
    frame = pd.DataFrame(
        {
            "flight_id": frame.id,
            "flight_date": dates.dt.strftime("%Y-%m-%d"),
            "y_true": (arrival >= 15).astype(int),
        }
    )
    days = sorted(frame.flight_date.unique())
    train_end, validation_end = int(len(days) * 0.70), int(len(days) * (0.70 + 0.15))
    calibration_start = int(train_end * 0.8)
    windows = {
        "base_train": days[:calibration_start],
        "train_calibration": days[calibration_start:train_end],
        "validation": days[train_end:validation_end],
        "test": days[validation_end:],
    }
    require(
        all(windows.values()),
        "Insufficient dates for nonempty chronological partitions",
    )
    return {
        name: frame.loc[frame.flight_date.isin(window)]
        .sort_values("flight_id")
        .reset_index(drop=True)
        for name, window in windows.items()
    }


def read_predictions(
    path: Path, expected: pd.DataFrame, partition: str
) -> pd.DataFrame:
    """Validate the complete prediction population against original IDs/labels."""
    frame = pd.read_csv(
        path, dtype={"flight_id": "string"}, float_precision="round_trip"
    )
    columns = {"flight_id", "flight_date", "y_true", "y_probability", "partition"}
    require(
        columns <= set(frame),
        f"Missing prediction columns in {path.name}: {sorted(columns - set(frame))}",
    )
    require(
        not frame[list(columns)].isna().any().any(),
        f"Missing labels, IDs, dates, partition or probabilities in {path.name}",
    )
    require(frame.flight_id.is_unique, f"Duplicate flight IDs in {path.name}")
    require(
        frame.partition.eq(partition).all(),
        f"Incorrect partition tags in {path.name}; expected {partition}",
    )
    require(
        set(frame.flight_id) == set(expected.flight_id),
        f"Missing/extra prediction rows in {path.name}: expected {len(expected)}, observed {len(frame)}",
    )
    frame = frame.sort_values("flight_id").reset_index(drop=True)
    require(frame.y_true.isin([0, 1]).all(), f"Labels must be binary in {path.name}")
    require(
        np.array_equal(frame.y_true, expected.y_true),
        f"Incorrect labels against original dataset in {path.name}",
    )
    parsed = pd.to_datetime(frame.flight_date, format="%Y-%m-%d", errors="raise")
    require(
        not parsed.isna().any() and frame.flight_date.equals(expected.flight_date),
        f"Incorrect dates against original dataset in {path.name}",
    )
    probability = pd.to_numeric(frame.y_probability, errors="raise").to_numpy(
        dtype=float
    )
    require(
        np.isfinite(probability).all()
        and ((probability >= 0) & (probability <= 1)).all(),
        f"Probabilities must be finite and in [0, 1] in {path.name}",
    )
    return frame


def calculate_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Independent sklearn calculation with the original binary conventions."""
    require(
        isinstance(threshold, (int, float))
        and np.isfinite(threshold)
        and 0 <= threshold <= 1,
        "Threshold must be finite and in [0, 1]",
    )
    y = frame.y_true.to_numpy(dtype=int)
    p = frame.y_probability.to_numpy(dtype=float)
    prediction = p >= threshold
    tn, fp, fn, tp = metrics.confusion_matrix(y, prediction, labels=[0, 1]).ravel()

    def ratio(a, b):
        return float(a / b) if b else 0.0

    two_classes = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(metrics.precision_score(y, prediction, zero_division=0)),
        "recall": float(metrics.recall_score(y, prediction, zero_division=0)),
        "f1": float(metrics.f1_score(y, prediction, zero_division=0)),
        "f2": float(metrics.fbeta_score(y, prediction, beta=2, zero_division=0)),
        "accuracy": float(metrics.accuracy_score(y, prediction)),
        "balanced_accuracy": (ratio(tp, tp + fn) + ratio(tn, tn + fp)) / 2,
        "specificity": ratio(tn, tn + fp),
        "false_positive_rate": ratio(fp, tn + fp),
        "false_negative_rate": ratio(fn, tp + fn),
        "predicted_positive_rate": float(prediction.mean()),
        "n": len(y),
        "prevalence": float(y.mean()),
        "roc_auc": float(metrics.roc_auc_score(y, p)) if two_classes else None,
        "average_precision": float(metrics.average_precision_score(y, p))
        if two_classes
        else None,
        "brier": float(metrics.brier_score_loss(y, p)),
        "log_loss": float(metrics.log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(p.mean()),
        "calibration_bias": float(p.mean() - y.mean()),
        "class_metrics": metrics.classification_report(
            y, prediction, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }


def validation_f1_choice(frame: pd.DataFrame) -> tuple[float, float]:
    """Replay the documented F1 rule using sklearn's PR curve, never test data."""
    require(
        frame.partition.eq("validation").all(),
        "F1 replay accepts validation observations only",
    )
    y, p = frame.y_true.to_numpy(), frame.y_probability.to_numpy()
    candidates = np.unique(np.r_[np.linspace(0, 1, 201), p])
    precision, recall, score_thresholds = metrics.precision_recall_curve(y, p)
    positions = np.searchsorted(score_thresholds, candidates, side="left")
    pp = np.where(positions < len(score_thresholds), precision[positions], 0.0)
    rr = recall[positions]
    f1 = np.divide(2 * pp * rr, pp + rr, out=np.zeros_like(pp), where=(pp + rr) != 0)
    best = float(f1.max())
    # Same rational F1 may differ by machine rounding across equivalent PR points.
    threshold = float(candidates[np.isclose(f1, best, rtol=0, atol=1e-14)][-1])
    return threshold, best


class Audit:
    """Accumulate explicit checks and discrepancies without suppressing failures."""

    def __init__(self) -> None:
        self.report: dict[str, Any] = {
            "checks": [],
            "metric_discrepancies": [],
            "observed_row_counts": {},
            "file_hashes": {},
            "package_versions": {},
            "unverified": [
                "Historical non-use of test labels cannot be proved from mutable artifacts. This audit replays decisions with validation inputs only, before reading final-test metrics.",
                "Original fitting history and historical-feature outcome availability are not established by these artifacts.",
            ],
        }

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        """Record and print every performed check, preserving failures."""
        self.report["checks"].append(
            {"name": name, "passed": bool(passed), "detail": detail}
        )
        print(
            f"{'PASS' if passed else 'FAIL'} {name}" + (f": {detail}" if detail else "")
        )

    def attempt(self, name: str, action: Callable[[], Any]) -> Any:
        """Capture artifact errors as failed checks so a failure report is saved."""
        try:
            result = action()
        except Exception as error:  # noqa: BLE001 -- preserve failures in the audit report
            self.check(name, False, f"{type(error).__name__}: {error}")
            return None
        self.check(name, True)
        return result

    def compare_metrics(self, name: str, observed: dict, saved: dict) -> None:
        """Compare each required metric, including nested class-wise results."""

        def compare(prefix, values, expected):
            for key, value in values.items():
                label = f"{prefix}.{key}"
                reference = expected.get(key) if isinstance(expected, dict) else None
                if isinstance(value, dict):
                    compare(label, value, reference)
                    continue
                present = isinstance(expected, dict) and key in expected
                if value is None:
                    same = present and reference is None
                elif isinstance(value, int):
                    same = (
                        present
                        and isinstance(reference, (int, float))
                        and reference == value
                    )
                else:
                    same = (
                        present
                        and isinstance(reference, (int, float))
                        and np.isfinite(reference)
                        and np.isclose(value, reference, atol=ATOL, rtol=RTOL)
                    )
                self.check(
                    label,
                    bool(same),
                    "" if same else f"saved={reference!r}, calculated={value!r}",
                )
                if not same:
                    self.report["metric_discrepancies"].append(
                        {"metric": label, "saved": reference, "calculated": value}
                    )

        compare(name, observed, saved)


def validate_manifest(manifest: dict) -> list[str]:
    """Accept only the existing, explicitly documented selection protocol."""
    require(
        manifest.get("schema_version") == 1,
        "Unsupported/missing selection schema_version",
    )
    require(
        manifest.get("calibration_rule")
        == "minimum validation Brier, log loss tie-break",
        "Unsupported calibration selection rule",
    )
    require(
        manifest.get("threshold_rule")
        == "maximum validation F1; higher threshold tie-break",
        "Unsupported threshold rule; expected validation-only F1",
    )
    models = manifest.get("models")
    require(isinstance(models, dict) and bool(models), "Missing model selections")
    for name, item in models.items():
        require(
            isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None,
            "Invalid model filename identifier",
        )
        require(
            isinstance(item, dict) and item.get("calibration") in METHODS,
            f"Invalid calibration for {name}",
        )
        threshold = item.get("threshold")
        require(
            isinstance(threshold, (int, float))
            and np.isfinite(threshold)
            and 0 <= threshold <= 1,
            f"Invalid threshold for {name}",
        )
        require(
            isinstance(item.get("strategies", {}).get("max_f1"), dict),
            f"Missing recorded validation max_f1 for {name}",
        )
    require(manifest.get("winner") in models, "Missing/unknown validation winner")
    require(
        isinstance(manifest.get("versions"), dict) and bool(manifest["versions"]),
        "Missing recorded package versions",
    )
    require(
        isinstance(manifest.get("feature_columns"), list)
        and bool(manifest["feature_columns"]),
        "Missing recorded feature schema",
    )
    windows = manifest.get("windows")
    require(
        isinstance(windows, dict)
        and all(
            isinstance(windows.get(k), dict)
            for k in ("base_train", "train_calibration", "validation")
        ),
        "Missing/malformed chronological window metadata",
    )
    return list(models)


def check_frozen(frozen: dict, manifest: dict) -> bool:
    """Inspect fitted object identity without invoking fit or predict methods."""
    require(
        isinstance(frozen, dict) and set(frozen) == set(manifest["models"]),
        "Frozen model names do not match selection",
    )
    for name, item in frozen.items():
        selection = manifest["models"][name]
        require(
            item["threshold"] == selection["threshold"],
            f"Altered frozen threshold for {name}",
        )
        require(
            item["calibrator"].method == selection["calibration"],
            f"Frozen calibration mismatch for {name}",
        )
        check_is_fitted(item["model"])
        if hasattr(item["model"], "feature_names_in_"):
            require(
                list(item["model"].feature_names_in_) == manifest["feature_columns"],
                f"Frozen feature schema mismatch for {name}",
            )
        require(
            hasattr(item["calibrator"], "estimator"), f"Unfitted calibrator for {name}"
        )
        if selection["calibration"] != "raw":
            check_is_fitted(item["calibrator"].estimator)
    return True


def check_decision(
    name: str, selection: dict, candidates: dict[str, dict], frame: pd.DataFrame
) -> dict:
    """Validate only validation-derived scores/thresholds; no test input accepted."""
    method = min(
        METHODS, key=lambda k: (candidates[k]["brier"], candidates[k]["log_loss"])
    )
    require(
        selection["calibration"] == method,
        f"{name}: calibration is not validation Brier/log-loss minimum (expected {method})",
    )
    threshold, best = validation_f1_choice(frame)
    require(
        selection["threshold"] == threshold,
        f"{name}: threshold differs from validation maximum-F1 choice (expected {threshold!r})",
    )
    recorded = selection["strategies"]["max_f1"]
    require(
        recorded.get("threshold") == threshold,
        f"{name}: recorded validation threshold altered",
    )
    actual = calculate_metrics(frame, threshold)
    require(
        np.isclose(actual["f1"], best, atol=ATOL, rtol=RTOL),
        f"{name}: selected threshold does not attain validation max F1",
    )
    for key in (
        "precision",
        "recall",
        "f1",
        "f2",
        "tp",
        "fp",
        "tn",
        "fn",
        "predicted_positive_rate",
    ):
        require(
            key in recorded
            and isinstance(recorded[key], (int, float))
            and np.isclose(recorded[key], actual[key], atol=ATOL, rtol=RTOL),
            f"{name}: recorded validation choice metric {key} mismatch",
        )
    return actual


def _inspect_artifacts(
    audit: Audit, data: Path, output_dir: Path, before: dict[Path, str]
) -> None:
    """Run checks without modifying inputs; caller always finalizes a report."""

    def remember(path: Path) -> str:
        digest = sha256(path)
        before[path] = digest
        audit.report["file_hashes"][str(path)] = digest
        return digest

    def read_document(name: str) -> dict:
        path = output_dir / name
        remember(path)
        return load_json(path)

    for package in PACKAGES:
        version = audit.attempt(
            f"installed_version:{package}",
            lambda p=package: importlib.metadata.version(p),
        )
        audit.report["package_versions"][package] = version
    dataset_hash = audit.attempt("dataset_file_sha256", lambda: remember(data))
    frozen_hash = audit.attempt(
        "frozen_file_sha256", lambda: remember(output_dir / "frozen.joblib")
    )
    audit.report.update(
        {"dataset_sha256": dataset_hash, "artifact_sha256": frozen_hash}
    )
    manifest = audit.attempt("selection.json", lambda: read_document("selection.json"))
    names = (
        audit.attempt("selection_schema_and_rules", lambda: validate_manifest(manifest))
        if manifest is not None
        else None
    )
    partitions = (
        audit.attempt("original_dataset_partitions", lambda: rebuild_partitions(data))
        if dataset_hash
        else None
    )
    if partitions is not None:
        audit.report["observed_row_counts"]["dataset"] = {
            k: len(v) for k, v in partitions.items()
        }
    frozen = None
    if names is not None:
        data_match = dataset_hash is not None and dataset_hash == manifest.get(
            "data_sha256"
        )
        model_match = frozen_hash is not None and frozen_hash == manifest.get(
            "artifact_sha256"
        )
        audit.check("dataset_hash_matches_selection", data_match)
        audit.check("frozen_hash_matches_selection", model_match)
        if model_match:
            frozen = audit.attempt(
                "trusted_frozen_deserialization",
                lambda: joblib.load(output_dir / "frozen.joblib"),
            )
        if frozen is not None:
            audit.attempt(
                "frozen_configuration_and_fitted_state",
                lambda: check_frozen(frozen, manifest),
            )
        for package, recorded in manifest["versions"].items():
            audit.check(
                f"recorded_version:{package}",
                audit.report["package_versions"].get(package) == recorded,
                f"recorded={recorded}, installed={audit.report['package_versions'].get(package)}",
            )
        if partitions is not None:
            for key in ("base_train", "train_calibration", "validation"):
                frame = partitions[key]
                expected = {
                    "n": len(frame),
                    "first": frame.flight_date.min(),
                    "last": frame.flight_date.max(),
                }
                audit.check(
                    f"chronological_window:{key}",
                    manifest.get("windows", {}).get(key) == expected,
                    json.dumps(expected),
                )
    # Replay ALL selections before final-test metrics are even opened.
    saved_validation = audit.attempt(
        "validation_metrics.json", lambda: read_document("validation_metrics.json")
    )
    saved_calibration = audit.attempt(
        "calibration_metrics.json", lambda: read_document("calibration_metrics.json")
    )
    observed_validation: dict[str, dict] = {}

    def prediction(path: Path, expected: pd.DataFrame, partition: str) -> pd.DataFrame:
        remember(path)
        frame = read_predictions(path, expected, partition)
        audit.report["observed_row_counts"][path.name] = len(frame)
        return frame

    if names is not None and partitions is not None:
        if saved_validation is not None:
            audit.check("validation_model_names", set(saved_validation) == set(names))
        if saved_calibration is not None:
            audit.check(
                "calibration_variant_names",
                set(saved_calibration) == {f"{n}_{m}" for n in names for m in METHODS},
            )
        for name in names:
            candidates, frames = {}, {}
            for method in METHODS:
                key = f"{name}_{method}"
                frame = audit.attempt(
                    f"validation_predictions:{key}",
                    lambda k=key: prediction(
                        output_dir / f"{k}_validation.csv",
                        partitions["validation"],
                        "validation",
                    ),
                )
                if frame is None:
                    continue
                actual = calculate_metrics(frame, 0.5)
                candidates[method], frames[method] = actual, frame
                if saved_calibration is not None:
                    audit.compare_metrics(
                        f"calibration_metrics:{key}",
                        actual,
                        saved_calibration.get(key, {}),
                    )
            if len(candidates) == len(METHODS):
                choice = manifest["models"][name]
                selected = frames[choice["calibration"]]
                actual = audit.attempt(
                    f"validation_only_selection:{name}",
                    lambda n=name, c=choice, v=candidates, f=selected: check_decision(
                        n, c, v, f
                    ),
                )
                if actual is not None:
                    observed_validation[name] = actual
                    if saved_validation is not None:
                        audit.compare_metrics(
                            f"validation_metrics:{name}",
                            actual,
                            saved_validation.get(name, {}),
                        )
            else:
                audit.check(
                    f"validation_only_selection:{name}",
                    False,
                    "Cannot replay: missing or invalid validation predictions",
                )
        audit.check(
            "identical_validation_population_all_models",
            all(
                any(
                    c["name"] == f"validation_predictions:{n}_{m}" and c["passed"]
                    for c in audit.report["checks"]
                )
                for n in names
                for m in METHODS
            ),
            "Every stream must equal the same rebuilt original ID/label/date population",
        )
        if set(observed_validation) == set(names):
            if all(
                v["average_precision"] is not None for v in observed_validation.values()
            ):
                winner = max(
                    names, key=lambda n: observed_validation[n]["average_precision"]
                )
                audit.check(
                    "validation_only_model_winner",
                    manifest["winner"] == winner,
                    f"Validation AP winner: {winner}",
                )
            else:
                audit.check(
                    "validation_only_model_winner",
                    False,
                    "Cannot verify AP model selection with a one-class validation partition",
                )
        else:
            audit.check(
                "validation_only_model_winner",
                False,
                "Validation selection prerequisites failed",
            )
    else:
        audit.check(
            "validation_only_selection_replay",
            False,
            "Dataset or manifest prerequisites failed",
        )
    saved_test = audit.attempt(
        "test_metrics.json", lambda: read_document("test_metrics.json")
    )
    if names is not None and partitions is not None:
        if saved_test is not None:
            audit.check("test_model_names", set(saved_test) == set(names))
        test_ok = True
        for name in names:
            frame = audit.attempt(
                f"test_predictions:{name}",
                lambda n=name: prediction(
                    output_dir / f"{n}_test.csv", partitions["test"], "test"
                ),
            )
            if frame is None:
                test_ok = False
                continue
            actual = calculate_metrics(frame, manifest["models"][name]["threshold"])
            if saved_test is not None:
                audit.compare_metrics(
                    f"test_metrics:{name}", actual, saved_test.get(name, {})
                )
        audit.check(
            "identical_test_population_all_models",
            test_ok,
            "Every stream must equal the same rebuilt original ID/label/date population",
        )


def run_audit(data: Path, output_dir: Path) -> dict[str, Any]:
    """Verify a complete saved run and exclusively create a new JSON report."""
    data, output_dir = Path(data), Path(output_dir)
    audit = Audit()
    audit.report.update(
        {
            "data": str(data.resolve()),
            "output_dir": str(output_dir.resolve()),
            "auditor_sha256": sha256(Path(__file__)),
            "python": platform.python_version(),
            "tolerances": {"atol": ATOL, "rtol": RTOL},
            "started_utc": datetime.now(UTC).isoformat(),
        }
    )
    before: dict[Path, str] = {}
    try:
        _inspect_artifacts(audit, data, output_dir, before)
    except Exception as error:  # noqa: BLE001 -- unexpected corruption still needs a failure report
        audit.check("audit_execution", False, f"{type(error).__name__}: {error}")
    for path, digest in before.items():
        current = audit.attempt(f"unchanged:{path.name}", lambda p=path: sha256(p))
        audit.check(f"read_only:{path.name}", current == digest)
    audit.report["status"] = (
        "PASS" if all(c["passed"] for c in audit.report["checks"]) else "FAIL"
    )
    audit.report["finished_utc"] = datetime.now(UTC).isoformat()
    # Exclusive creation protects all existing artifacts and all previous reports.
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"audit_report_{uuid.uuid4().hex}.json"
        audit.report["report_path"] = str(report_path.resolve())
        with report_path.open("x") as handle:
            audit.check("new_report_written", True, str(report_path))
            json.dump(audit.report, handle, indent=2, allow_nan=False)
            handle.write("\n")

    except OSError as error:
        audit.check("new_report_written", False, str(error))
        audit.report["status"] = "FAIL"
    print(
        f"{audit.report['status']} evaluation audit ({len(audit.report['checks'])} checks)"
    )
    for limitation in audit.report["unverified"]:
        print(f"UNVERIFIED {limitation}")
    return audit.report


def main(argv: list[str] | None = None) -> int:
    """Run the audit and return nonzero on any failed verification or report write."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/flights.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    args = parser.parse_args(argv)
    return 0 if run_audit(args.data, args.output_dir)["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

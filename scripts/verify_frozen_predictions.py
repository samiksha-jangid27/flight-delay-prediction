"""Read-only, validation-only replay of a deterministic frozen-model sample.

Run from root: python scripts/verify_frozen_predictions.py --data
 data/raw/flights.csv --output-dir reports/evaluation --partition validation
 --sample-size 100. Only trusted local, hash-verified joblib files are loaded.
No fitting, threshold selection, test-metric reads or report writes occur.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flight_delay.data.pipeline import build_data_pipeline
from flight_delay.evaluation.predictions import load_prediction_csv
from flight_delay.evaluation.run_state import require_completed_validation
from flight_delay.features.model_features import select_model_features

if __package__:
    from scripts.audit_evaluation import load_json, rebuild_partitions, require, sha256
else:  # Direct `python scripts/verify_frozen_predictions.py` invocation.
    from audit_evaluation import load_json, rebuild_partitions, require, sha256

ATOL = 1e-12
RTOL = 1e-12


def _settings(name: str, item: dict, selection: dict, manifest: dict) -> None:
    """Check frozen schema/map/threshold and settings recorded for this study."""
    require(
        item["threshold"] == selection["threshold"],
        f"{name}: frozen threshold mismatch",
    )
    require(
        item["calibrator"].method == selection["calibration"],
        f"{name}: calibration method mismatch",
    )
    recorded = selection.get("strategies", {}).get("max_f1", {}).get("threshold")
    require(
        recorded == selection["threshold"],
        f"{name}: recorded validation threshold mismatch",
    )
    model = item["model"]
    require(
        list(model.feature_names_in_) == manifest["feature_columns"],
        f"{name}: fitted feature schema mismatch",
    )
    require(
        np.array_equal(model.classes_, [0, 1]),
        f"{name}: expected binary class order [0, 1]",
    )
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    params = estimator.get_params(deep=False)
    if name == "xgboost_tuned_full":
        expected = manifest["xgboost_parameters"]
    elif name == "logistic_balanced":
        # The original manifest does not contain full LR parameters; these are
        # the unchanged producer configuration, checked in addition to its hash.
        expected = {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
        }
    else:
        expected = manifest.get("model_parameters", {}).get(name, {})
    for key, value in expected.items():
        require(
            key in params and params[key] == value,
            f"{name}: model setting {key} mismatch",
        )
    if "random_state" in params and "seed" in manifest:
        require(
            params["random_state"] == manifest["seed"], f"{name}: random_state mismatch"
        )


def verify(
    data: Path,
    output_dir: Path,
    *,
    partition: str = "validation",
    sample_size: int = 100,
) -> dict:
    """Compare exact IDs/classes and strict-tolerance probabilities; never write."""
    require(partition == "validation", "Only validation verification is permitted")
    require(
        isinstance(sample_size, int) and sample_size > 0,
        "sample_size must be a positive integer",
    )
    data, output_dir = Path(data), Path(output_dir)
    require_completed_validation(output_dir)
    manifest_path, artifact = (
        output_dir / "selection.json",
        output_dir / "frozen.joblib",
    )
    before = {path: sha256(path) for path in (data, manifest_path, artifact)}
    manifest = load_json(manifest_path)
    require(manifest.get("schema_version") == 1, "Unsupported selection schema")
    require(before[data] == manifest["data_sha256"], "Dataset SHA-256 mismatch")
    require(
        before[artifact] == manifest["artifact_sha256"],
        "Frozen artifact SHA-256 mismatch",
    )
    require(
        isinstance(manifest.get("models"), dict) and bool(manifest["models"]),
        "Frozen model selections must be nonempty",
    )
    require(
        isinstance(manifest.get("versions"), dict)
        and {"numpy", "pandas", "scikit-learn", "xgboost"} <= set(manifest["versions"]),
        "Missing complete package versions in frozen manifest",
    )
    versions = {
        package: importlib.metadata.version(package) for package in manifest["versions"]
    }
    require(
        versions == manifest["versions"],
        "Runtime package versions differ from frozen manifest",
    )
    print("PASS dataset and frozen-artifact SHA-256; recorded runtime versions")
    frozen = joblib.load(artifact)
    require(
        isinstance(frozen, dict)
        and bool(frozen)
        and set(frozen) == set(manifest["models"]),
        "Frozen model identities mismatch",
    )
    canonical = rebuild_partitions(data)["validation"]
    _train_unused, validation, _test_never_scored = build_data_pipeline(data)
    required = {"id", "is_delayed", "year", "month", "day"}
    require(
        required <= set(validation), "Rebuilt validation has missing identity columns"
    )
    ids = validation.id.astype("string")
    require(
        not ids.isna().any() and ids.is_unique,
        "Rebuilt validation IDs must be nonmissing and unique",
    )
    identity = (
        pd.DataFrame(
            {
                "flight_id": ids,
                "flight_date": pd.to_datetime(
                    validation[["year", "month", "day"]]
                ).dt.strftime("%Y-%m-%d"),
                "y_true": validation.is_delayed.astype(int),
            }
        )
        .sort_values("flight_id")
        .reset_index(drop=True)
    )
    require(
        identity.equals(canonical),
        "Rebuilt validation IDs, dates or labels disagree with canonical raw partitions",
    )
    frame = validation.assign(id=ids).set_index("id")
    population = canonical.flight_id.to_numpy()
    rng = np.random.default_rng(42)
    chosen = np.sort(
        rng.choice(population, size=min(sample_size, len(population)), replace=False)
    )
    features = select_model_features(frame.loc[chosen])
    require(
        list(features) == manifest["feature_columns"],
        "Current feature allowlist differs from frozen schema",
    )
    result = {
        "status": "PASS",
        "partition": partition,
        "seed": 42,
        "sample_size": len(chosen),
        "sample_ids": chosen.tolist(),
        "dataset_sha256": before[data],
        "artifact_sha256": before[artifact],
        "versions": versions,
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "models": {},
    }
    for name, selection in manifest["models"].items():
        require(
            isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None,
            "Invalid model identifier",
        )
        threshold = selection["threshold"]
        require(
            isinstance(threshold, (int, float))
            and np.isfinite(threshold)
            and 0 <= threshold <= 1,
            f"{name}: invalid frozen threshold",
        )
        require(
            selection["calibration"] in ("raw", "sigmoid", "isotonic"),
            f"{name}: unknown calibration method",
        )
        item = frozen[name]
        _settings(name, item, selection, manifest)
        path = output_dir / f"{name}_{selection['calibration']}_validation.csv"
        before[path] = sha256(path)
        saved = (
            load_prediction_csv(path, partition="validation")
            .sort_values("flight_id")
            .reset_index(drop=True)
        )
        require(
            saved[["flight_id", "flight_date", "y_true"]].equals(canonical),
            f"{name}: saved IDs, dates or labels mismatch",
        )
        raw = np.asarray(item["model"].predict_proba(features), dtype=float)
        require(raw.shape == (len(chosen), 2), f"{name}: predict_proba shape mismatch")
        require(
            np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all(),
            f"{name}: invalid raw predict_proba probabilities",
        )
        calculated = np.asarray(item["calibrator"].predict(raw[:, 1]), dtype=float)
        require(
            calculated.shape == (len(chosen),)
            and np.isfinite(calculated).all()
            and ((calculated >= 0) & (calculated <= 1)).all(),
            f"{name}: invalid calculated probabilities",
        )
        expected = saved.set_index("flight_id").loc[chosen, "y_probability"].to_numpy()
        require(
            np.allclose(calculated, expected, atol=ATOL, rtol=RTOL),
            f"{name}: frozen probabilities disagree; maximum absolute error={np.max(np.abs(calculated - expected))!r}",
        )
        require(
            np.array_equal(calculated >= threshold, expected >= threshold),
            f"{name}: exact frozen-threshold classes disagree (no tolerance allowed)",
        )
        result["models"][name] = {
            "max_absolute_probability_error": float(
                np.max(np.abs(calculated - expected))
            ),
            "classes_match": True,
            "threshold": threshold,
            "calibration": selection["calibration"],
        }
        print(
            f"PASS {name}: model settings, sampled probabilities, exact classes and full saved population"
        )
    for path, digest in before.items():
        require(
            sha256(path) == digest,
            f"Input changed during read-only verification: {path}",
        )
    print("PASS read-only input preservation")
    return result


def main(argv: list[str] | None = None) -> int:
    """Print findings; return nonzero for any missing, corrupt or disagreeing input."""
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/flights.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--partition", choices=["validation"], default="validation")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        result = verify(
            args.data,
            args.output_dir,
            partition=args.partition,
            sample_size=args.sample_size,
        )
    except Exception as error:  # noqa: BLE001 -- CLI must clearly fail on corrupt trusted artifacts
        print(f"FAIL {type(error).__name__}: {error}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

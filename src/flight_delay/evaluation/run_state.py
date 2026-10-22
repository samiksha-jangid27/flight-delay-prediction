"""Exclusive, durable phase markers and non-overwriting staged publication.

Failed phases intentionally retain their start marker and staging tree. A failed
validation requires a fresh empty directory. A started final-test phase must not
be retried automatically: preserve it for manual review; no unlock/retry API is
provided. Existing unmarked completed validation runs remain readable.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

RECOVERY = "Preserve all markers and staging files for review. Use a fresh empty directory for validation; do not retry a started final-test phase or delete its marker to bypass protection."


def _marker(path: Path, phase: str, status: str) -> None:
    """Exclusively create and flush a marker; never replace any existing file."""
    try:
        with path.open("x") as handle:
            json.dump(
                {
                    "phase": phase,
                    "status": status,
                    "pid": os.getpid(),
                    "utc": datetime.now(UTC).isoformat(),
                },
                handle,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ValueError(f"{phase} already started or completed. {RECOVERY}") from error


def _publish(source: Path, target: Path) -> None:
    """Create each destination exclusively; retain staging on any partial failure."""
    if source.is_symlink():
        raise ValueError("Refusing symlink in staged artifacts")
    if source.is_dir():
        target.mkdir()  # Existing destinations, even empty ones, must fail.
        for child in source.iterdir():
            _publish(child, target / child.name)
    else:
        os.link(source, target)  # Same filesystem; atomic and fails if target exists.


def _has_test_outputs(output: Path) -> bool:
    return (
        any(output.glob("*_test.csv"))
        or (output / "test_metrics.json").exists()
        or (output / "figures" / "test").exists()
        or any(output.glob(".final_test_*"))
    )


def require_completed_validation(output: Path) -> None:
    """Accept legacy frozen runs, but reject any incomplete new-style validation."""
    started, complete = (
        output / ".validation_started.json",
        output / ".validation_complete.json",
    )
    if started.exists() or complete.exists():
        try:
            if not started.is_file() or not complete.is_file():
                raise ValueError("Missing phase marker")
            for path, status in ((started, "started"), (complete, "complete")):
                value = json.loads(path.read_text())
                if value.get("phase") != "validation" or value.get("status") != status:
                    raise ValueError("Invalid phase marker")
        except (OSError, ValueError, AttributeError) as error:
            raise ValueError(
                f"Incomplete validation publication. {RECOVERY}"
            ) from error


@contextmanager
def staged_run(output: Path, phase: str) -> Iterator[Path]:
    """Claim one phase, generate in staging, publish, then mark completion.

    A start marker is an at-most-once claim, not an automatically expiring lock.
    Staged files are removed only after a successful, marked publication.
    """
    output = Path(output)
    if phase not in ("validation", "final_test"):
        raise ValueError("Unknown evaluation phase")
    output.mkdir(parents=True, exist_ok=True)
    if phase == "validation":
        if any(output.iterdir()):
            raise ValueError(
                f"Frozen or partial validation output exists; new validation requires an empty directory. {RECOVERY}"
            )
    else:
        if _has_test_outputs(output):
            raise ValueError(
                f"Final test already evaluated, started, or partially written. {RECOVERY}"
            )
        require_completed_validation(output)
        if (
            not (output / "selection.json").is_file()
            or not (output / "frozen.joblib").is_file()
        ):
            raise ValueError(
                "Final test requires a completed frozen validation experiment"
            )
    _marker(output / f".{phase}_started.json", phase, "started")
    stage = output / f".{phase}_staging"
    stage.mkdir()
    yield stage
    for child in stage.iterdir():
        if phase == "final_test" and child.name == "figures":
            destination = output / "figures"
            if destination.is_symlink():
                raise ValueError("Refusing symlink figure destination")
            destination.mkdir(exist_ok=True)
            for figure_partition in child.iterdir():
                _publish(figure_partition, destination / figure_partition.name)
        else:
            _publish(child, output / child.name)
    _marker(output / f".{phase}_complete.json", phase, "complete")
    shutil.rmtree(stage)

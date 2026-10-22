"""Synthetic failure injection and filesystem claims; no real final-test scoring."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from flight_delay.evaluation import evaluate as runner
from flight_delay.evaluation import run_state


def frozen_directory(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    data = tmp_path / "data.csv"
    data.write_bytes(b"synthetic")
    artifact = output / "frozen.joblib"
    artifact.write_bytes(b"not deserialized by these guard tests")
    manifest = {
        "data_sha256": runner.fingerprint(data),
        "artifact_sha256": runner.fingerprint(artifact),
    }
    (output / "selection.json").write_text(json.dumps(manifest))
    return data, output


def snapshot(output):
    return {
        str(p.relative_to(output)): p.read_bytes()
        for p in output.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    "existing",
    [
        "selection.json",
        "frozen.joblib",
        "validation_metrics.json",
        ".validation_started.json",
        "notes.txt",
    ],
)
def test_validation_rejects_every_nonempty_directory(tmp_path, monkeypatch, existing):
    output = tmp_path / "run"
    output.mkdir()
    (output / existing).write_bytes(b"preserve")
    monkeypatch.setattr(
        runner,
        "_run_validation",
        lambda *a: pytest.fail("Existing run reached training"),
    )
    before = snapshot(output)
    with pytest.raises(ValueError, match="empty directory"):
        runner.run_validation(tmp_path / "unused.csv", output)
    assert snapshot(output) == before


def test_validation_partial_write_is_retained_and_retry_blocked(tmp_path, monkeypatch):
    output = tmp_path / "run"
    calls = []

    def fail(data, stage, *args):
        calls.append(1)
        (stage / "frozen.joblib").write_bytes(b"partial model")
        raise OSError("injected artifact write failure")

    monkeypatch.setattr(runner, "_run_validation", fail)
    with pytest.raises(OSError, match="injected"):
        runner.run_validation(tmp_path / "unused.csv", output)
    assert (output / ".validation_started.json").exists() and not (
        output / ".validation_complete.json"
    ).exists()
    assert not (output / "frozen.joblib").exists()
    assert (
        output / ".validation_staging/frozen.joblib"
    ).read_bytes() == b"partial model"
    before = snapshot(output)
    with pytest.raises(ValueError):
        runner.run_validation(tmp_path / "unused.csv", output)
    assert len(calls) == 1 and snapshot(output) == before


def test_validation_completion_after_exclusive_publication(tmp_path):
    output = tmp_path / "run"
    with run_state.staged_run(output, "validation") as stage:
        (stage / "frozen.joblib").write_bytes(b"fixture")
        (stage / "selection.json").write_text("{}")
        assert not (output / "selection.json").exists()
        assert not (output / ".validation_complete.json").exists()
    assert (output / "frozen.joblib").read_bytes() == b"fixture"
    assert (output / ".validation_complete.json").exists()
    assert not (output / ".validation_staging").exists()
    with pytest.raises(ValueError), run_state.staged_run(output, "validation"):
        pytest.fail("repeated validation")


@pytest.mark.parametrize(
    "existing",
    [
        "test_metrics.json",
        "fixture_test.csv",
        "figures/test",
        ".final_test_started.json",
        ".final_test_staging",
    ],
)
def test_legacy_and_partial_test_outputs_block_before_deserialization(
    tmp_path, monkeypatch, existing
):
    data, output = frozen_directory(tmp_path)
    path = output / existing
    if existing in ("figures/test", ".final_test_staging"):
        path.mkdir(parents=True)
    else:
        path.write_bytes(b"preserve")
    monkeypatch.setattr(
        runner.joblib, "load", lambda *a: pytest.fail("repeat reached model loading")
    )
    before = snapshot(output)
    with pytest.raises(ValueError, match="already"):
        runner.run_final_test(data, output)
    assert snapshot(output) == before


def test_final_failure_cannot_repeat_and_keeps_originals(tmp_path, monkeypatch):
    data, output = frozen_directory(tmp_path)
    original = snapshot(output)
    calls = []

    def fail(data, frozen_output, stage, manifest):
        calls.append(1)
        (stage / "test_metrics.json").write_text("{}")
        raise RuntimeError("injected plot failure after synthetic scoring")

    monkeypatch.setattr(runner, "_run_final_test", fail)
    with pytest.raises(RuntimeError, match="injected"):
        runner.run_final_test(data, output)
    assert (output / ".final_test_started.json").exists() and not (
        output / ".final_test_complete.json"
    ).exists()
    assert not (output / "test_metrics.json").exists()
    for name, content in original.items():
        assert (output / name).read_bytes() == content
    before = snapshot(output)
    with pytest.raises(ValueError, match="already"):
        runner.run_final_test(data, output)
    assert len(calls) == 1 and snapshot(output) == before


@pytest.mark.parametrize("phase", ["validation", "final_test"])
def test_exclusive_claim_blocks_concurrent_invocation(tmp_path, phase):
    _, output = frozen_directory(tmp_path)
    if phase == "validation":
        output = tmp_path / "fresh"
    entered, release = Event(), Event()

    def first():
        with run_state.staged_run(output, phase) as stage:
            (stage / "phase.json").write_text("{}")
            entered.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(first)
        assert entered.wait(10)
        try:
            with pytest.raises(ValueError), run_state.staged_run(output, phase):
                pytest.fail("Concurrent duplicate phase")
        finally:
            release.set()
        worker.result(timeout=10)
    assert (output / f".{phase}_complete.json").exists()


def test_publication_failure_never_replaces_existing_file(tmp_path):
    output = tmp_path / "run"
    with (
        pytest.raises(FileExistsError),
        run_state.staged_run(output, "validation") as stage,
    ):
        (stage / "artifact.json").write_text("new")
        (output / "artifact.json").write_text("concurrent existing content")
    assert (output / "artifact.json").read_text() == "concurrent existing content"
    assert (output / ".validation_staging/artifact.json").read_text() == "new"
    assert not (output / ".validation_complete.json").exists()


def test_final_publication_merges_only_existing_figures_parent(tmp_path, monkeypatch):
    data, output = frozen_directory(tmp_path)
    (output / "figures/validation").mkdir(parents=True)
    (output / "figures/validation/original.png").write_bytes(b"keep")
    original = snapshot(output)

    def generate(data, source, stage, manifest):
        (stage / "figures/test").mkdir(parents=True)
        (stage / "figures/test/mock.png").write_bytes(b"synthetic plot")
        (stage / "test_metrics.json").write_text("{}")
        return {"mock": True}

    monkeypatch.setattr(runner, "_run_final_test", generate)
    assert runner.run_final_test(data, output) == {"mock": True}
    for name, value in original.items():
        assert (output / name).read_bytes() == value
    assert (output / ".final_test_complete.json").exists()
    assert (output / "figures/test/mock.png").read_bytes() == b"synthetic plot"


def test_incomplete_validation_blocks_final_phase(tmp_path, monkeypatch):
    data, output = frozen_directory(tmp_path)
    (output / ".validation_started.json").write_text("{}")
    before = snapshot(output)
    monkeypatch.setattr(
        runner,
        "_run_final_test",
        lambda *a: pytest.fail("Incomplete validation scored"),
    )
    with pytest.raises(ValueError, match="Incomplete"):
        runner.run_final_test(data, output)
    assert snapshot(output) == before

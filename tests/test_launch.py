from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import launch as launcher
from experiments.runtime import configure_runtime_environment


def test_plot_stage_uses_the_batch_plot_entrypoint():
    assert launcher.STAGE_MODULES["plot"] == "experiments.run_plot"


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("eval-df", ["experiments.eval_df"]),
        ("eval-phi", ["experiments.eval_phi"]),
        ("plot-df", ["experiments.plot_df"]),
        ("plot-phi", ["experiments.plot_phi"]),
        ("all", ["experiments.run", "all"]),
        ("df-pipeline", ["experiments.run", "df"]),
        ("phi-pipeline", ["experiments.run", "phi"]),
    ],
)
def test_new_stage_command_vectors(stage: str, expected: list[str], tmp_path: Path):
    config_path = tmp_path / "run.yaml"

    assert launcher._command(stage, config_path) == [
        sys.executable,
        "-u",
        "-m",
        *expected,
        str(config_path),
    ]


def test_runtime_environment_sets_default_without_overriding_user_value():
    default_env: dict[str, str] = {}
    configure_runtime_environment(default_env)
    assert default_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"

    custom_env = {"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}
    configure_runtime_environment(custom_env)
    assert custom_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"


def test_launch_detaches_worker_and_writes_run_scoped_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = tmp_path / "run.yaml"
    config_path.write_text("schema: test\n", encoding="utf-8")
    logs_dir = tmp_path / "artifacts" / "logs"
    spec = SimpleNamespace(
        name="test_run",
        source_path=config_path,
        logs_dir=logs_dir,
    )
    prepared: list[object] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(launcher, "load_run_spec", lambda path: spec)

    def fake_prepare_run(value):
        prepared.append(value)
        logs_dir.mkdir(parents=True, exist_ok=True)

    class DummyProcess:
        pid = 4321

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return DummyProcess()

    monkeypatch.setattr(launcher, "prepare_run", fake_prepare_run)
    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    result = launcher.launch("phi", config_path)

    assert prepared == [spec]
    assert captured["command"] == [
        sys.executable,
        "-u",
        "-m",
        "experiments.run_phi",
        str(config_path),
    ]
    assert captured["stdin"] is launcher.subprocess.DEVNULL
    assert captured["stderr"] is launcher.subprocess.STDOUT
    assert captured["start_new_session"] is True
    assert captured["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "CUDA_VISIBLE_DEVICES" not in captured["env"]
    assert result.pid == 4321
    assert result.log_path == logs_dir / "phi.log"
    assert result.pid_path.read_text(encoding="utf-8") == "4321\n"
    log_text = result.log_path.read_text(encoding="utf-8")
    assert "stage=phi run=test_run" in log_text
    assert "CUDA_VISIBLE_DEVICES=all visible devices" in log_text


def test_launch_rejects_duplicate_active_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "df.pid").write_text("123\n", encoding="utf-8")
    spec = SimpleNamespace(
        name="test_run",
        source_path=tmp_path / "run.yaml",
        logs_dir=logs_dir,
    )
    monkeypatch.setattr(launcher, "load_run_spec", lambda path: spec)
    monkeypatch.setattr(launcher, "prepare_run", lambda value: None)
    monkeypatch.setattr(launcher, "_process_is_running", lambda pid: pid == 123)

    with pytest.raises(RuntimeError, match="already running"):
        launcher.launch("df", spec.source_path)


@pytest.mark.parametrize(
    ("active_stage", "requested_stage"),
    [
        ("df-pipeline", "df"),
        ("df-pipeline", "eval-df"),
        ("df-pipeline", "plot-df"),
        ("df-pipeline", "plot-phi"),
        ("phi-pipeline", "phi"),
        ("phi-pipeline", "eval-phi"),
        ("phi-pipeline", "plot-phi"),
        ("phi-pipeline", "plot-df"),
        ("plot-phi", "df-pipeline"),
        ("plot-df", "phi-pipeline"),
        ("all", "df"),
        ("all", "phi"),
        ("eval", "df-pipeline"),
        ("plot", "phi-pipeline"),
        ("df-pipeline", "phi-pipeline"),
    ],
)
def test_pipeline_and_related_stage_are_mutually_exclusive(
    active_stage: str,
    requested_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / f"{active_stage}.pid").write_text("123\n", encoding="utf-8")
    spec = SimpleNamespace(
        name="test_run",
        source_path=tmp_path / "run.yaml",
        logs_dir=logs_dir,
    )
    monkeypatch.setattr(launcher, "load_run_spec", lambda path: spec)
    monkeypatch.setattr(launcher, "prepare_run", lambda value: None)
    monkeypatch.setattr(launcher, "_process_is_running", lambda pid: pid == 123)

    with pytest.raises(RuntimeError, match=active_stage):
        launcher.launch(requested_stage, spec.source_path)


@pytest.mark.parametrize(
    ("active_stage", "requested_stage"),
    [
        ("eval-df", "plot-df"),
        ("plot-df", "eval-df"),
        ("eval-phi", "plot-phi"),
        ("plot-phi", "eval-phi"),
        ("plot-df", "plot-phi"),
        ("plot", "plot-df"),
        ("eval", "eval-df"),
        ("eval-phi", "plot"),
    ],
)
def test_evaluation_and_plot_writers_are_mutually_exclusive(
    active_stage: str,
    requested_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / f"{active_stage}.pid").write_text("123\n", encoding="utf-8")
    spec = SimpleNamespace(
        name="test_run",
        source_path=tmp_path / "run.yaml",
        logs_dir=logs_dir,
    )
    monkeypatch.setattr(launcher, "load_run_spec", lambda path: spec)
    monkeypatch.setattr(launcher, "prepare_run", lambda value: None)
    monkeypatch.setattr(launcher, "_process_is_running", lambda pid: pid == 123)

    with pytest.raises(RuntimeError, match=active_stage):
        launcher.launch(requested_stage, spec.source_path)


def test_unrelated_standalone_stages_keep_existing_non_conflicting_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "df.pid").write_text("123\n", encoding="utf-8")
    spec = SimpleNamespace(
        name="test_run",
        source_path=tmp_path / "run.yaml",
        logs_dir=logs_dir,
    )

    class DummyProcess:
        pid = 456

    monkeypatch.setattr(launcher, "load_run_spec", lambda path: spec)
    monkeypatch.setattr(launcher, "prepare_run", lambda value: None)
    monkeypatch.setattr(launcher, "_process_is_running", lambda pid: pid == 123)
    monkeypatch.setattr(
        launcher.subprocess, "Popen", lambda command, **kwargs: DummyProcess()
    )

    result = launcher.launch("phi", spec.source_path)

    assert result.pid == 456
    assert result.log_path == logs_dir / "phi.log"


def test_experiments_import_applies_runtime_default():
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" in os.environ

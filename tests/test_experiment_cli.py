from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


STAGE_MODULES = (
    "experiments.eval_df",
    "experiments.eval_phi",
    "experiments.plot_df",
    "experiments.plot_phi",
)


@pytest.mark.parametrize("module_name", STAGE_MODULES)
def test_flat_stage_cli_accepts_exactly_one_config_argument(
    module_name,
    monkeypatch,
):
    module = importlib.import_module(module_name)
    captured = {}

    def fake_run(config_path):
        captured["config_path"] = config_path
        return {}

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [module_name, "config.yaml"])

    assert module.main() == 0
    assert captured["config_path"] == "config.yaml"


@pytest.mark.parametrize("module_name", STAGE_MODULES)
@pytest.mark.parametrize(
    "arguments",
    [(), ("config.yaml", "--stage", "df")],
    ids=("missing-config", "no-public-stage-option"),
)
def test_flat_stage_cli_rejects_invalid_argument_shapes(
    module_name,
    arguments,
    monkeypatch,
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(
        module,
        "run",
        lambda *args, **kwargs: pytest.fail("invalid CLI must not call run"),
    )
    monkeypatch.setattr(sys, "argv", [module_name, *arguments])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 2


@pytest.mark.parametrize("module_name", STAGE_MODULES)
def test_flat_stage_cli_help_exits_successfully(module_name, monkeypatch, capsys):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 0
    assert "CONFIG" not in capsys.readouterr().err


def _workflow_spec(*, external_df: bool = False):
    df_dir = Path("/run/df")
    return SimpleNamespace(
        source_path=Path("/configs/run.yaml"),
        evaluation={"df": {"enabled": True}, "phi": {"enabled": True}},
        df_dir=df_dir,
        phi_df_dir=Path("/external/df") if external_df else df_dir,
    )


@pytest.mark.parametrize(
    ("workflow", "expected_modules"),
    [
        (
            "df",
            (
                "experiments.run_df",
                "experiments.eval_df",
                "experiments.plot_df",
            ),
        ),
        (
            "phi",
            (
                "experiments.run_phi",
                "experiments.eval_phi",
                "experiments.plot_phi",
            ),
        ),
        (
            "all",
            (
                "experiments.run_df",
                "experiments.eval_df",
                "experiments.plot_df",
                "experiments.run_phi",
                "experiments.eval_phi",
                "experiments.plot_phi",
            ),
        ),
    ],
)
def test_composed_workflows_run_modules_in_the_declared_order(
    workflow,
    expected_modules,
    monkeypatch,
):
    import experiments.run as runner

    spec = _workflow_spec()
    calls = []
    checkpoint_checks = []
    monkeypatch.setattr(runner, "load_run_spec", lambda path: spec)
    monkeypatch.setattr(
        runner,
        "require_checkpoint",
        lambda path, label: checkpoint_checks.append((path, label)),
    )
    monkeypatch.setattr(
        runner,
        "_run_module",
        lambda module, config: calls.append((module, config)),
    )

    assert runner.run("source.yaml", workflow=workflow) is None

    assert calls == [(module, spec.source_path) for module in expected_modules]
    expected_checks = [(spec.phi_df_dir, "DF")] if workflow == "phi" else []
    assert checkpoint_checks == expected_checks


def test_composed_workflow_stops_after_the_first_failed_child(monkeypatch):
    import experiments.run as runner

    spec = _workflow_spec()
    calls = []
    monkeypatch.setattr(runner, "load_run_spec", lambda path: spec)

    def fake_run_module(module, config):
        calls.append(module)
        if module == "experiments.eval_df":
            raise subprocess.CalledProcessError(1, [module])

    monkeypatch.setattr(runner, "_run_module", fake_run_module)

    with pytest.raises(subprocess.CalledProcessError):
        runner.run("source.yaml", workflow="all")

    assert calls == ["experiments.run_df", "experiments.eval_df"]


def test_all_workflow_rejects_an_external_df_before_starting(monkeypatch):
    import experiments.run as runner

    spec = _workflow_spec(external_df=True)
    monkeypatch.setattr(runner, "load_run_spec", lambda path: spec)
    monkeypatch.setattr(
        runner,
        "_run_module",
        lambda *args, **kwargs: pytest.fail("invalid all workflow must not start"),
    )

    with pytest.raises(ValueError, match="requires Phi to use this run's DF"):
        runner.run("source.yaml", workflow="all")


def test_phi_workflow_allows_and_checks_an_external_df(monkeypatch):
    import experiments.run as runner

    spec = _workflow_spec(external_df=True)
    calls = []
    checkpoint_checks = []
    monkeypatch.setattr(runner, "load_run_spec", lambda path: spec)
    monkeypatch.setattr(
        runner,
        "require_checkpoint",
        lambda path, label: checkpoint_checks.append((path, label)),
    )
    monkeypatch.setattr(
        runner,
        "_run_module",
        lambda module, config: calls.append(module),
    )

    runner.run("source.yaml", workflow="phi")

    assert checkpoint_checks == [(Path("/external/df"), "DF")]
    assert calls == list(runner.WORKFLOW_MODULES["phi"])


@pytest.mark.parametrize("workflow", ["df", "phi", "all"])
def test_composed_workflow_rejects_disabled_evaluation_before_starting(
    workflow,
    monkeypatch,
):
    import experiments.run as runner

    spec = _workflow_spec()
    disabled_stage = "df" if workflow in {"df", "all"} else "phi"
    spec.evaluation[disabled_stage]["enabled"] = False
    monkeypatch.setattr(runner, "load_run_spec", lambda path: spec)
    monkeypatch.setattr(
        runner,
        "_run_module",
        lambda *args, **kwargs: pytest.fail("disabled workflow must not start"),
    )

    with pytest.raises(ValueError, match=rf"evaluation\.{disabled_stage}\.enabled"):
        runner.run("source.yaml", workflow=workflow)


def test_child_process_uses_current_python_project_cwd_and_configured_environment(
    tmp_path,
    monkeypatch,
):
    import experiments.run as runner

    captured = {}
    config_path = tmp_path / "run.yaml"
    monkeypatch.setenv("DPJAX_PARENT_SENTINEL", "present")
    monkeypatch.setattr(runner.sys, "executable", "/env/bin/python")

    def fake_configure(environment):
        environment["DPJAX_CHILD_SENTINEL"] = "configured"

    def fake_subprocess_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)

    monkeypatch.setattr(runner, "configure_runtime_environment", fake_configure)
    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    runner._run_module("experiments.eval_df", config_path)

    assert captured["command"] == [
        "/env/bin/python",
        "-u",
        "-m",
        "experiments.eval_df",
        str(config_path),
    ]
    assert captured["cwd"] == runner.PROJECT_ROOT
    assert captured["check"] is True
    assert captured["env"]["DPJAX_PARENT_SENTINEL"] == "present"
    assert captured["env"]["DPJAX_CHILD_SENTINEL"] == "configured"

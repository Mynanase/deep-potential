from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")

from experiments.diagnostics import load_df_diagnostics, resolve_figure_artifact
from experiments.diagnostics.validation_artifacts import load_validation_metrics
from experiments.list_runs import collect_runs
from experiments.plotting import render_figure, write_figure
from experiments.run_plot import run as run_plot
from experiments.workflows.config import load_run_spec, prepare_run


def _write_config(tmp_path: Path) -> Path:
    df_model = tmp_path / "df.yaml"
    phi_model = tmp_path / "phi.yaml"
    df_model.write_text(
        yaml.safe_dump(
            {
                "schema": "dpjax.model.v1",
                "kind": "df",
                "flow": {"type": "ffjord", "dim": 6},
                "train": {"optimizer": "radam"},
            }
        ),
        encoding="utf-8",
    )
    phi_model.write_text(
        yaml.safe_dump(
            {
                "schema": "dpjax.model.v1",
                "kind": "phi",
                "potential": {"hidden_sizes": [8]},
                "train": {"optimizer": "radam"},
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "source.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "dpjax.run.v2",
                "name": "plot-test",
                "case": "generic",
                "output_dir": str(tmp_path / "runs" / "group" / "plot-test"),
                "data": {"path": str(tmp_path / "data.h5"), "dataset": "eta"},
                "df": {"model": str(df_model)},
                "phi": {"model": str(phi_model)},
                "execution": {},
                "evaluation": {},
                "plots": {"formats": ["png", "pdf"], "dpi": 40},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_score_artifact(spec) -> None:
    values = np.array([[[1.0, -1.0], [0.5, -0.5]]])
    np.savez_compressed(
        spec.result_data_dir / "df_diagnostics.npz",
        score_field_r_edges=np.array([0.0, 1.0, 2.0]),
        score_field_v_edges=np.array([0.0, 1.0, 2.0]),
        score_field_effective_count=np.full((2, 2), 30.0),
        score_field_min_effective_count=np.asarray(20.0),
        score_field_r_median=values,
        score_field_v_median=-values,
    )


def test_new_artifact_path_precedes_legacy_fallback(tmp_path):
    run = tmp_path / "run"
    (run / "results" / "data").mkdir(parents=True)
    (run / "eval").mkdir()
    np.savez(run / "eval" / "df_diagnostics.npz", marker=np.array([1]))
    np.savez(run / "results" / "data" / "df_diagnostics.npz", marker=np.array([2]))

    assert load_df_diagnostics(run)["marker"].item() == 2
    (run / "results" / "data" / "df_diagnostics.npz").unlink()
    assert load_df_diagnostics(run)["marker"].item() == 1

    legacy_validation = run / "validation" / "plummer"
    legacy_validation.mkdir(parents=True)
    (legacy_validation / "metrics.json").write_text(
        json.dumps({"legacy": True}), encoding="utf-8"
    )
    assert load_validation_metrics(run, kind="plummer") == {"legacy": True}

    (run / "plots").mkdir()
    legacy_figure = run / "plots" / "old.png"
    legacy_figure.write_bytes(b"legacy")
    assert resolve_figure_artifact(run, "old.png") == legacy_figure


def test_debug_and_official_writes_have_isolated_manifest_effects(tmp_path):
    config = _write_config(tmp_path)
    spec = load_run_spec(config)
    prepare_run(spec)
    _write_score_artifact(spec)

    debug_figure = render_figure(spec, "df_score_field_rv", {"min_effective_count": 10})
    debug_paths = write_figure(debug_figure, target="debug", formats=("png",), dpi=40)
    assert debug_paths[0].parent == spec.debug_dir
    assert not spec.manifest_path.exists()

    official_figure = render_figure(
        spec,
        "df_score_field_rv",
        {"length_unit": "kpc", "min_effective_count": 15},
    )
    official_paths = write_figure(official_figure, target="official", dpi=40)
    assert {path.suffix for path in official_paths} == {".png", ".pdf"}
    manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["figures"]) == {"df_score_field_rv"}
    assert manifest["run"]["name"] == "plot-test"
    assert "results/data/df_diagnostics.npz" in manifest["inputs"]
    parameters = manifest["figures"]["df_score_field_rv"]["parameters"]
    assert parameters["min_effective_count"] == 15
    assert "length_unit" not in parameters


def test_run_plot_only_df_writes_report_and_list_runs_reads_state(tmp_path):
    config = _write_config(tmp_path)
    spec = load_run_spec(config)
    prepare_run(spec)
    _write_score_artifact(spec)

    outputs = run_plot(config, sections=["df"])
    assert set(outputs) == {"df_score_field_rv"}
    assert spec.report_path.is_file()
    assert (spec.figures_dir / "df_score_field_rv.png").is_file()
    assert (spec.figures_dir / "df_score_field_rv.pdf").is_file()

    records = collect_runs(tmp_path / "runs")
    assert len(records) == 1
    assert records[0]["name"] == "plot-test"
    assert records[0]["plot"] is True


def test_list_runs_falls_back_when_new_result_data_is_empty(tmp_path):
    config = _write_config(tmp_path)
    spec = load_run_spec(config)
    prepare_run(spec)
    legacy_eval = spec.output_dir / "eval"
    legacy_eval.mkdir()
    (legacy_eval / "df_metrics.json").write_text("{}", encoding="utf-8")

    records = collect_runs(tmp_path / "runs")
    assert records[0]["eval"] is True


def test_figure_notebook_contains_only_individual_saved_artifact_calls():
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "figure_debug.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    combined = "\n".join(code)
    assert "render_all" not in combined
    assert "run_eval" not in combined
    assert "run_df" not in combined
    assert "run_phi" not in combined
    assert 'target="debug"' in combined
    assert 'target="official"' in combined
    for cell in code:
        assert cell.count("render_figure(") <= 1

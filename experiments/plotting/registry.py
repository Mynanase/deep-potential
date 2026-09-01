"""Run-aware figure registry and deterministic artifact writer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.diagnostics.df_artifacts import load_df_diagnostics
from experiments.diagnostics.phi_artifacts import load_phi_diagnostics
from experiments.diagnostics.training_artifacts import (
    load_metrics,
    plot_training_loss,
    plot_training_residual_stats,
    plot_training_score_stats,
)
from experiments.paths import resolve_path
from experiments.plotting.df_diagnostics import (
    plot_cylindrical_marginals_by_radius,
    plot_cylindrical_rz_density,
    plot_density_profile,
    plot_radial_speed_density,
    plot_score_distribution,
    plot_score_field_rv,
    plot_score_slices_by_radius,
    plot_velocity_marginals,
)
from experiments.plotting.phi_diagnostics import (
    plot_mass_density_profile,
    plot_mass_density_slice,
    plot_potential_profile,
    plot_potential_slice,
    plot_radial_acceleration_profile,
)
from experiments.plotting.style import PAPER_STYLE, label_with_unit
from experiments.workflows.config import RunSpec, load_run_spec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _manifest_plot_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Keep rendering controls while excluding display-only unit strings."""
    return {
        str(key): _json_safe(value)
        for key, value in parameters.items()
        if key != "display_units" and not str(key).endswith("_unit")
    }


def resolve_run(run: RunSpec | str | Path) -> RunSpec:
    """Resolve a source config, run directory, snapshot, or RunSpec."""
    if isinstance(run, RunSpec):
        return run
    path = resolve_path(run)
    snapshot = path / "run.yaml" if path.is_dir() else path
    if snapshot.name == "run.yaml" and snapshot.is_file():
        raw = yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}
        source = raw.get("_meta", {}).get("source_config")
        if not source:
            raise ValueError(f"Run snapshot does not record its source config: {snapshot}")
        return load_run_spec(source)
    return load_run_spec(snapshot)


@dataclass(frozen=True)
class FigureSpec:
    """One named figure and the saved artifacts its builder consumes."""

    name: str
    section: str
    loader: Callable[[RunSpec], Any]
    builder: Callable[[Any, Mapping[str, Any]], Any]


def _unit(overrides: Mapping[str, Any], key: str) -> str:
    value = overrides.get(key, "")
    return "" if value is None else str(value)


def _df_arrays(spec: RunSpec) -> dict[str, np.ndarray]:
    return load_df_diagnostics(spec.output_dir)


def _phi_arrays(spec: RunSpec) -> dict[str, np.ndarray]:
    return load_phi_diagnostics(spec.output_dir)


def _training_metrics(stage: str):
    return lambda spec: load_metrics(getattr(spec, f"{stage}_dir") / "metrics.csv")


def _training(metrics: Mapping[str, np.ndarray], overrides: Mapping[str, Any]):
    return plot_training_loss(
        dict(metrics),
        title=overrides.get("title"),
        dpi=int(overrides.get("dpi", 200)),
    )


def _training_score(metrics: Mapping[str, np.ndarray], overrides: Mapping[str, Any]):
    return plot_training_score_stats(
        dict(metrics),
        title=overrides.get("title"),
        dpi=int(overrides.get("dpi", 200)),
    )


def _training_residual(metrics: Mapping[str, np.ndarray], overrides: Mapping[str, Any]):
    return plot_training_residual_stats(
        dict(metrics),
        title=overrides.get("title"),
        dpi=int(overrides.get("dpi", 200)),
    )


def _density(arrays: Mapping[str, np.ndarray], o: Mapping[str, Any]):
    unit = _unit(o, "length_unit")
    return plot_density_profile(
        arrays,
        radius_label=label_with_unit("r", unit),
        title=o.get("title"),
        dpi=int(o.get("dpi", 200)),
    )


def _cylindrical_rz(arrays: Mapping[str, np.ndarray], o: Mapping[str, Any]):
    unit = _unit(o, "length_unit")
    return plot_cylindrical_rz_density(
        arrays,
        radius_label=label_with_unit("R", unit),
        height_label=label_with_unit("z", unit),
        title=o.get("title"),
        dpi=int(o.get("dpi", 200)),
    )


def _velocity(coordinate: str):
    def builder(arrays: Mapping[str, np.ndarray], o: Mapping[str, Any]):
        return plot_velocity_marginals(
            arrays,
            coordinate,
            velocity_unit=_unit(o, "velocity_unit"),
            title=o.get("title"),
            dpi=int(o.get("dpi", 200)),
        )

    return builder


def _cylindrical_marginals(arrays: Mapping[str, np.ndarray], o: Mapping[str, Any]):
    return plot_cylindrical_marginals_by_radius(
        arrays,
        length_unit=_unit(o, "length_unit"),
        velocity_unit=_unit(o, "velocity_unit"),
        title=o.get("title"),
        dpi=int(o.get("dpi", 200)),
    )


def _score_field(arrays: Mapping[str, np.ndarray], o: Mapping[str, Any]):
    threshold = o.get("min_effective_count")
    return plot_score_field_rv(
        arrays,
        length_unit=_unit(o, "length_unit"),
        velocity_unit=_unit(o, "velocity_unit"),
        min_effective_count=None if threshold is None else float(threshold),
        title=o.get("title"),
        dpi=int(o.get("dpi", 200)),
    )


def _score_slices(arrays: Mapping[str, np.ndarray], o: Mapping[str, Any]):
    threshold = o.get("min_effective_count")
    return plot_score_slices_by_radius(
        arrays,
        length_unit=_unit(o, "length_unit"),
        velocity_unit=_unit(o, "velocity_unit"),
        min_effective_count=None if threshold is None else float(threshold),
        title=o.get("title"),
        dpi=int(o.get("dpi", 200)),
    )


def _score_distribution(arrays: Mapping[str, np.ndarray], o: Mapping[str, Any]):
    return plot_score_distribution(
        arrays,
        max_points=int(o.get("max_points", 2_000)),
        seed=int(o.get("seed", 0)),
        title=o.get("title"),
        dpi=int(o.get("dpi", 200)),
    )


def _radial_speed(arrays: Mapping[str, np.ndarray], o: Mapping[str, Any]):
    return plot_radial_speed_density(
        arrays,
        length_unit=_unit(o, "length_unit"),
        velocity_unit=_unit(o, "velocity_unit"),
        title=o.get("title"),
        dpi=int(o.get("dpi", 200)),
    )


def _phi_profile(kind: str):
    def builder(arrays: Mapping[str, np.ndarray], o: Mapping[str, Any]):
        radius = arrays["radial_r"]
        length_unit = _unit(o, "length_unit")
        radius_label = label_with_unit("r", length_unit)
        dpi = int(o.get("dpi", 200))
        if kind == "potential":
            unit = _unit(o, "potential_unit")
            return plot_potential_profile(
                radius,
                arrays["radial_phi_shifted"],
                truth_potential=arrays.get("radial_truth_phi_shifted"),
                radius_label=radius_label,
                potential_label=label_with_unit(r"$\Phi$", unit),
                title=o.get("title"),
                dpi=dpi,
            )
        if kind == "acceleration":
            unit = _unit(o, "acceleration_unit")
            return plot_radial_acceleration_profile(
                radius,
                arrays["radial_acceleration"],
                truth_acceleration=arrays.get("radial_truth_acceleration"),
                radius_label=radius_label,
                acceleration_label=label_with_unit(r"$a_r$", unit),
                title=o.get("title"),
                dpi=dpi,
            )
        unit = _unit(o, "density_unit")
        return plot_mass_density_profile(
            radius,
            arrays["radial_density"],
            truth_density=arrays.get("radial_truth_density"),
            radius_label=radius_label,
            density_label=label_with_unit(r"$\rho$", unit),
            title=o.get("title"),
            dpi=dpi,
        )

    return builder


def _phi_slice(kind: str):
    def builder(arrays: Mapping[str, np.ndarray], o: Mapping[str, Any]):
        unit = _unit(o, "length_unit")

        def axis(symbol: str) -> str:
            return label_with_unit(symbol, unit)

        dpi = int(o.get("dpi", 200))
        if kind == "potential":
            potential_unit = _unit(o, "potential_unit")
            return plot_potential_slice(
                arrays["slice_x"],
                arrays["slice_y"],
                arrays["slice_phi"],
                truth_potential=arrays.get("slice_truth_phi"),
                x_label=axis("x"),
                y_label=axis("y"),
                potential_label=label_with_unit(r"$\Phi$", potential_unit),
                title=o.get("title"),
                dpi=dpi,
            )
        density_unit = _unit(o, "density_unit")
        return plot_mass_density_slice(
            arrays["slice_x"],
            arrays["slice_y"],
            arrays["slice_density"],
            truth_density=arrays.get("slice_truth_density"),
            x_label=axis("x"),
            y_label=axis("y"),
            density_label=label_with_unit(r"$\rho$", density_unit),
            title=o.get("title"),
            dpi=dpi,
        )

    return builder


class FigureRegistry:
    """Registry that loads artifacts and calls pure in-memory builders."""

    def __init__(self) -> None:
        self._specs: dict[str, FigureSpec] = {}

    def register(
        self,
        name: str,
        section: str,
        loader: Callable[[RunSpec], Any],
        builder: Callable[[Any, Mapping[str, Any]], Any],
    ) -> None:
        if name in self._specs:
            raise ValueError(f"Figure {name!r} is already registered.")
        self._specs[name] = FigureSpec(name, section, loader, builder)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    @staticmethod
    def _parameters(
        spec: RunSpec,
        figure_spec: FigureSpec,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        parameters = {
            key: value
            for key, value in spec.plots.items()
            if key
            not in {
                "formats",
                "display_units",
                "training",
                "df",
                "phi",
                "validation",
            }
        }
        parameters.update(dict(spec.plots.get("display_units", {})))
        parameters.update(dict(spec.plots.get(figure_spec.section, {})))
        if overrides:
            parameters.update(dict(overrides))
        return parameters

    @staticmethod
    def _render_loaded(
        spec: RunSpec,
        figure_spec: FigureSpec,
        arrays: Any,
        parameters: Mapping[str, Any],
    ) -> Any:
        import matplotlib.pyplot as plt

        with plt.rc_context(PAPER_STYLE):
            figure = figure_spec.builder(arrays, parameters)
        figure._dpjax_run_spec = spec
        figure._dpjax_figure_name = figure_spec.name
        figure._dpjax_figure_parameters = dict(parameters)
        return figure

    def render_figure(
        self,
        run: RunSpec | str | Path,
        figure_name: str,
        overrides: Mapping[str, Any] | None = None,
    ) -> Any:
        if figure_name not in self._specs:
            raise KeyError(f"Unknown figure {figure_name!r}. Available: {', '.join(self.names)}")
        spec = resolve_run(run)
        figure_spec = self._specs[figure_name]
        parameters = self._parameters(spec, figure_spec, overrides)
        arrays = figure_spec.loader(spec)
        return self._render_loaded(spec, figure_spec, arrays, parameters)

    def render_all(
        self,
        run: RunSpec | str | Path,
        sections: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        spec = resolve_run(run)
        selected = None if sections is None else set(sections)
        figures: dict[str, Any] = {}
        loaded: dict[Callable[[RunSpec], Any], Any] = {}
        missing_loaders: set[Callable[[RunSpec], Any]] = set()
        for name, figure_spec in self._specs.items():
            if selected is not None and figure_spec.section not in selected:
                continue
            if figure_spec.loader in missing_loaders:
                continue
            try:
                if figure_spec.loader not in loaded:
                    loaded[figure_spec.loader] = figure_spec.loader(spec)
                parameters = self._parameters(spec, figure_spec)
                figures[name] = self._render_loaded(
                    spec,
                    figure_spec,
                    loaded[figure_spec.loader],
                    parameters,
                )
            except FileNotFoundError:
                missing_loaders.add(figure_spec.loader)
            except KeyError:
                continue
        return figures

    def render_many(
        self,
        run: RunSpec | str | Path,
        figure_names: Sequence[str],
    ) -> dict[str, Any]:
        """Strictly render named figures while sharing loaded artifacts."""
        unknown = [name for name in figure_names if name not in self._specs]
        if unknown:
            raise KeyError(
                f"Unknown figure(s): {', '.join(unknown)}. "
                f"Available: {', '.join(self.names)}"
            )
        spec = resolve_run(run)
        figures: dict[str, Any] = {}
        loaded: dict[Callable[[RunSpec], Any], Any] = {}
        try:
            for name in figure_names:
                figure_spec = self._specs[name]
                if figure_spec.loader not in loaded:
                    loaded[figure_spec.loader] = figure_spec.loader(spec)
                parameters = self._parameters(spec, figure_spec)
                figures[name] = self._render_loaded(
                    spec,
                    figure_spec,
                    loaded[figure_spec.loader],
                    parameters,
                )
        except Exception:
            import matplotlib.pyplot as plt

            for figure in figures.values():
                plt.close(figure)
            raise
        return figures


DEFAULT_REGISTRY = FigureRegistry()
DEFAULT_REGISTRY.register("training_df", "training", _training_metrics("df"), _training)
DEFAULT_REGISTRY.register("training_phi", "training", _training_metrics("phi"), _training)
DEFAULT_REGISTRY.register(
    "training_df_score_stats",
    "training",
    _training_metrics("df"),
    _training_score,
)
DEFAULT_REGISTRY.register(
    "training_phi_residual_stats",
    "training",
    _training_metrics("phi"),
    _training_residual,
)
DEFAULT_REGISTRY.register("df_density_profile", "df", _df_arrays, _density)
DEFAULT_REGISTRY.register("df_cylindrical_rz_density", "df", _df_arrays, _cylindrical_rz)
DEFAULT_REGISTRY.register("df_velocity_marginals_by_r", "df", _df_arrays, _velocity("r"))
DEFAULT_REGISTRY.register("df_velocity_marginals_by_theta", "df", _df_arrays, _velocity("theta"))
DEFAULT_REGISTRY.register("df_velocity_marginals_by_phi", "df", _df_arrays, _velocity("phi"))
DEFAULT_REGISTRY.register("df_score_distribution", "df", _df_arrays, _score_distribution)
DEFAULT_REGISTRY.register("df_cylindrical_marginals_by_R", "df", _df_arrays, _cylindrical_marginals)
DEFAULT_REGISTRY.register("df_score_field_rv", "df", _df_arrays, _score_field)
DEFAULT_REGISTRY.register("df_score_slices_by_R", "df", _df_arrays, _score_slices)
DEFAULT_REGISTRY.register("df_radial_speed_density", "df", _df_arrays, _radial_speed)
DEFAULT_REGISTRY.register("phi_potential_profile", "phi", _phi_arrays, _phi_profile("potential"))
DEFAULT_REGISTRY.register(
    "phi_acceleration_profile",
    "phi",
    _phi_arrays,
    _phi_profile("acceleration"),
)
DEFAULT_REGISTRY.register("phi_density_profile", "phi", _phi_arrays, _phi_profile("density"))
DEFAULT_REGISTRY.register("phi_potential_slice", "phi", _phi_arrays, _phi_slice("potential"))
DEFAULT_REGISTRY.register("phi_density_slice", "phi", _phi_arrays, _phi_slice("density"))


def render_figure(
    run: RunSpec | str | Path,
    figure_name: str,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    return DEFAULT_REGISTRY.render_figure(run, figure_name, overrides)


def render_all(
    run: RunSpec | str | Path,
    sections: Sequence[str] | None = None,
) -> dict[str, Any]:
    return DEFAULT_REGISTRY.render_all(run, sections)


def render_many(
    run: RunSpec | str | Path,
    figure_names: Sequence[str],
) -> dict[str, Any]:
    return DEFAULT_REGISTRY.render_many(run, figure_names)


class FigureWriter:
    """Write one figure and update the formal manifest only when requested."""

    def __init__(
        self,
        run: RunSpec | str | Path,
        *,
        formats: Sequence[str] | None = None,
        dpi: int = 200,
        overwrite: bool = True,
    ) -> None:
        self.spec = resolve_run(run)
        configured = self.spec.plots.get("formats", ["png", "pdf"])
        self.formats = tuple(str(value).lower() for value in (formats or configured))
        unsupported = set(self.formats) - {"png", "pdf"}
        if unsupported:
            raise ValueError(f"Unsupported figure formats: {sorted(unsupported)}")
        self.dpi = int(dpi)
        self.overwrite = bool(overwrite)

    def _manifest(self) -> dict[str, Any]:
        if self.spec.manifest_path.is_file():
            return json.loads(self.spec.manifest_path.read_text(encoding="utf-8"))
        snapshot = (
            yaml.safe_load(self.spec.snapshot_path.read_text(encoding="utf-8"))
            if self.spec.snapshot_path.is_file()
            else {}
        )
        return {
            "schema": "dpjax.plot-manifest.v1",
            "run": {
                "name": self.spec.name,
                "case": self.spec.case,
                "path": str(self.spec.output_dir),
                "git_commit": (snapshot or {}).get("_meta", {}).get("git_commit"),
            },
            "inputs": {},
            "figures": {},
        }

    def _input_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for path in (self.spec.df_dir / "metrics.csv", self.spec.phi_dir / "metrics.csv"):
            if path.is_file():
                hashes[str(path.relative_to(self.spec.output_dir))] = _sha256(path)
        for directory in (self.spec.result_data_dir, self.spec.output_dir / "eval"):
            if not directory.is_dir():
                continue
            found_result_artifact = False
            for path in sorted(directory.glob("*")):
                if path.is_file() and path.suffix in {".json", ".npz", ".yaml"}:
                    hashes[str(path.relative_to(self.spec.output_dir))] = _sha256(path)
                    found_result_artifact = True
            if found_result_artifact:
                break
        return hashes

    def write(
        self,
        figure: Any,
        figure_name: str | None = None,
        *,
        target: str = "official",
    ) -> tuple[Path, ...]:
        import matplotlib.pyplot as plt

        if target not in {"official", "debug"}:
            raise ValueError("target must be 'official' or 'debug'.")
        name = figure_name or getattr(figure, "_dpjax_figure_name", None)
        if not name:
            raise ValueError("figure_name is required for an unregistered Figure.")
        output_dir = self.spec.figures_dir if target == "official" else self.spec.debug_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = [output_dir / f"{name}.{format_name}" for format_name in self.formats]
        if not self.overwrite:
            existing = [path for path in outputs if path.exists()]
            if existing:
                raise FileExistsError(existing[0])
        try:
            for path, format_name in zip(outputs, self.formats):
                figure.savefig(
                    path,
                    format=format_name,
                    dpi=self.dpi,
                    metadata={"Title": name, "Creator": "deep-potential"},
                )
        finally:
            plt.close(figure)
        if target == "official":
            manifest = self._manifest()
            manifest["inputs"] = self._input_hashes()
            generated_at = datetime.now(timezone.utc).isoformat()
            manifest["figures"][name] = {
                "parameters": _manifest_plot_parameters(
                    getattr(figure, "_dpjax_figure_parameters", {})
                ),
                "outputs": [
                    {
                        "path": str(path.relative_to(self.spec.output_dir)),
                        "format": path.suffix.lstrip("."),
                        "sha256": _sha256(path),
                    }
                    for path in outputs
                ],
                "generated_at": generated_at,
            }
            manifest["updated_at"] = generated_at
            self.spec.results_dir.mkdir(parents=True, exist_ok=True)
            self.spec.manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return tuple(outputs)


def write_figure(
    figure: Any,
    run: RunSpec | str | Path | None = None,
    figure_name: str | None = None,
    *,
    target: str = "debug",
    formats: Sequence[str] | None = None,
    dpi: int = 200,
    overwrite: bool = True,
) -> tuple[Path, ...]:
    resolved_run = run or getattr(figure, "_dpjax_run_spec", None)
    if resolved_run is None:
        raise ValueError("run is required for an unregistered Figure.")
    return FigureWriter(
        resolved_run,
        formats=formats,
        dpi=dpi,
        overwrite=overwrite,
    ).write(figure, figure_name, target=target)

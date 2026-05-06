from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from dpjax.plotting.diagnostics import plot_plummer_figure3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DF_RUN = PROJECT_ROOT / "runs" / "plummer" / "df_v5"
DEFAULT_PHI_RUN = PROJECT_ROOT / "runs" / "plummer" / "phi_v4a_mse2"


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _ensure_slice(
    df_run_dir: Path,
    phi_run_dir: Path,
    out_dir: Path,
    *,
    rmax: float,
    grid: int,
    batch: int,
    recompute: bool,
) -> Path:
    slice_path = out_dir / "phi_slice_xy.npz"
    if slice_path.exists() and not recompute:
        return slice_path

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "plot_phi_slice.py"),
        "--df-run-dir",
        str(df_run_dir),
        "--phi-run-dir",
        str(phi_run_dir),
        "--out-dir",
        str(out_dir),
        "--z",
        "0.0",
        "--rmax",
        str(float(rmax)),
        "--grid",
        str(int(grid)),
        "--batch",
        str(int(batch)),
    ]
    subprocess.run(cmd, check=True)
    if not slice_path.exists():
        raise FileNotFoundError(f"Expected slice cache was not created: {slice_path}")
    return slice_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Plummer Figure 3 style potential/density diagnostic.")
    parser.add_argument("--df-run-dir", type=str, default=str(DEFAULT_DF_RUN))
    parser.add_argument("--phi-run-dir", type=str, default=str(DEFAULT_PHI_RUN))
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--rmax", type=float, default=5.0)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--recompute-slice", action="store_true")
    args = parser.parse_args()

    df_run_dir = _resolve_path(args.df_run_dir)
    phi_run_dir = _resolve_path(args.phi_run_dir)
    eval_dir = phi_run_dir / "eval"
    out_dir = _resolve_path(args.out_dir) if args.out_dir else eval_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    radial_path = eval_dir / "radial_curves_plummer.npz"
    if not radial_path.exists():
        raise FileNotFoundError(f"Missing radial cache: {radial_path}")

    slice_path = _ensure_slice(
        df_run_dir,
        phi_run_dir,
        out_dir,
        rmax=float(args.rmax),
        grid=int(args.grid),
        batch=int(args.batch),
        recompute=bool(args.recompute_slice),
    )

    radial = np.load(radial_path)
    slice_data = np.load(slice_path)

    plot_plummer_figure3(
        radial["r"],
        radial["phi_true"],
        radial["phi_learned_shift"],
        radial["rho_analytic"],
        radial["rho_learned"],
        slice_data["x"],
        slice_data["y"],
        slice_data["phi"],
        slice_data["rho"],
        fig_dir=out_dir,
        fig_fmt=tuple(args.formats),
        dpi=int(args.dpi),
    )

    written = [out_dir / f"plummer_figure3.{fmt}" for fmt in args.formats]
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

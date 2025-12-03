"""Run VSM on the Belloc 2015 paraglider and compare CL/CD over alpha.

Workflow:
1) Parse geometry + per-section polars from allPolarsVSM.dat (LE/TE + CL, CD, CM).
2) Build a Wing/BodyAerodynamics instance directly from those arrays.
3) Plot the geometry (lightweight custom chord viewer to avoid plot_geometry issues).
4) Sweep angle of attack and plot CL–alpha, CD–alpha, and CL/CD–alpha.
   Experimental data are overlaid if the .ods dependency is available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import matplotlib

# Avoid Qt backend warnings on Wayland; fall back to a non-Qt backend if needed.
try:
    if str(matplotlib.get_backend()).lower().startswith("qt"):
        matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.core.WingGeometry import Wing


def parse_all_polars(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Parse the Belloc allPolarsVSM.dat format into LE/TE arrays and per-section polars.

    File format (repeats for each section):
        panel LE : x, y, z, TE : x, y, z
        alpha_deg, cl, cd, cm
        ...

    Returns:
        le_arr: (N, 3) leading edge coordinates
        te_arr: (N, 3) trailing edge coordinates
        polar_data: list of (M_i, 4) arrays [alpha_rad, cl, cd, cm] per section
    """
    header_re = re.compile(
        r"panel LE : ([^,]+), ([^,]+), ([^,]+), TE : ([^,]+), ([^,]+), ([^,]+)"
    )

    le_list, te_list, polar_list = [], [], []
    current_rows: List[list] = []
    with file_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("panel LE"):
                # flush previous block
                if current_rows:
                    polar_list.append(_rows_to_polar_array(current_rows))
                    current_rows = []
                match = header_re.search(line)
                if not match:
                    raise ValueError(f"Could not parse header line: {line}")
                le_list.append([float(match.group(i)) for i in range(1, 4)])
                te_list.append([float(match.group(i)) for i in range(4, 7)])
            else:
                current_rows.append([float(v.strip()) for v in line.split(",")])

        if current_rows:
            polar_list.append(_rows_to_polar_array(current_rows))

    le_arr = np.array(le_list, dtype=float)
    te_arr = np.array(te_list, dtype=float)
    return le_arr, te_arr, polar_list


def _rows_to_polar_array(rows: List[list]) -> np.ndarray:
    """Convert raw rows [[alpha_deg, cl, cd, cm], ...] into (N,4) ndarray with alpha in rad."""
    arr = np.array(rows, dtype=float)
    alpha_rad = np.deg2rad(arr[:, 0])
    return np.column_stack((alpha_rad, arr[:, 1:]))


def load_experimental_points(data_path: Path):
    """
    Load experimental CL/CD vs alpha from the provided file (prefers .ods, falls back to .csv).
    Returns None on failure so the script still runs.
    """
    df = None
    if data_path.suffix.lower() == ".ods":
        try:
            df = pd.read_excel(data_path, engine="odf")  # odfpy is needed for .ods
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"Skipping experimental overlay ({exc})")
            return None
    elif data_path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(data_path)
        except Exception as exc:
            print(f"Skipping experimental overlay ({exc})")
            return None

    if df is None:
        return None

    # Heuristic: pick first columns containing alpha/alfa/cl/cd strings (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    alpha_col = next(
        (c for k, c in cols_lower.items() if ("alpha" in k) or ("alfa" in k)), None
    )
    cl_col = next((c for k, c in cols_lower.items() if k.startswith("cl")), None)
    cd_col = next((c for k, c in cols_lower.items() if k.startswith("cd")), None)
    if not all([alpha_col, cl_col, cd_col]):
        print(
            "Experimental data found but columns were not recognized; skipping overlay."
        )
        return None

    exp = df[[alpha_col, cl_col, cd_col]].dropna()
    exp = exp.rename(columns={alpha_col: "alpha_deg", cl_col: "cl", cd_col: "cd"})
    exp["cl_over_cd"] = exp["cl"] / exp["cd"]
    return exp


def simple_geometry_plot(le_arr: np.ndarray, te_arr: np.ndarray, title: str = ""):
    """
    Very light geometry viewer: plots LE/TE points and the chord line per section.
    Avoids reliance on plot_geometry to keep dependencies minimal.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title or "Geometry")

    # chord lines per section
    for idx, (le, te) in enumerate(zip(le_arr, te_arr)):
        xs = [le[0], te[0]]
        ys = [le[1], te[1]]
        zs = [le[2], te[2]]
        ax.plot(
            xs,
            ys,
            zs,
            color="tab:blue",
            linewidth=1.5,
            alpha=0.8,
            label="Chord line" if idx == 0 else None,
        )
        ax.scatter(
            [le[0]],
            [le[1]],
            [le[2]],
            color="k",
            s=12,
            label="LE" if idx == 0 else None,
        )
        ax.scatter(
            [te[0]],
            [te[1]],
            [te[2]],
            color="red",
            s=12,
            label="TE" if idx == 0 else None,
        )

    # quick span/cg sizing
    all_pts = np.vstack([le_arr, te_arr])
    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)
    xrange, yrange, zrange = xmax - xmin, ymax - ymin, zmax - zmin
    max_range = max(xrange, yrange, zrange)
    cx, cy, cz = all_pts.mean(axis=0)
    ax.set_xlim(cx - max_range / 2, cx + max_range / 2)
    ax.set_ylim(cy - max_range / 2, cy + max_range / 2)
    ax.set_zlim(cz - max_range / 2, cz + max_range / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def main():
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data" / "paraglider_Belloc_2015"
    all_polars_path = data_dir / "allPolarsVSM.dat"
    experimental_path = data_dir / "experimental_data.ods"
    if not experimental_path.exists():
        alt_path = data_dir / "experimental_data.csv"
        experimental_path = alt_path if alt_path.exists() else experimental_path

    # Quick settings
    N_PANELS = 50  # downsample from the raw 252 sections to speed plotting/solve
    SPANWISE_DISTRIBUTION = "uniform"
    SHOW_GEOMETRY = False  # set True to pop up the simple 3D chord plot

    # 1) Build BodyAerodynamics directly from the bundled geometry/polars
    le_arr, te_arr, polar_data = parse_all_polars(all_polars_path)
    wing = Wing(
        n_panels=N_PANELS,
        spanwise_panel_distribution=SPANWISE_DISTRIBUTION,
    )
    wing.update_wing_from_points(
        le_arr=le_arr,
        te_arr=te_arr,
        aero_input_type="reuse_initial_polar_data",
        polar_data_arr=polar_data,
    )
    body = BodyAerodynamics([wing])

    # 2) Plot geometry (simple viewer)
    if SHOW_GEOMETRY:
        simple_geometry_plot(le_arr, te_arr, title="Paraglider Belloc 2015 Geometry")

    # 3) Alpha sweep
    alpha_min_deg = -1.0
    alpha_max_deg = 19.0
    alpha_sweep_deg = np.arange(alpha_min_deg, alpha_max_deg + 1e-6, 1.0)
    Umag = 10.0  # matches Re used for pre-generated polars
    solver = Solver(reference_point=np.array([0.0, 0.0, 0.0]))

    cl_list, cd_list = [], []
    for alpha in alpha_sweep_deg:
        body.va_initialize(
            Umag=Umag, angle_of_attack=alpha, side_slip=0.0, yaw_rate=0.0
        )
        results = solver.solve(body)
        cl_list.append(results["cl"])
        cd_list.append(results["cd"])

    cl_arr = np.array(cl_list)
    cd_arr = np.array(cd_list)
    cl_over_cd = np.divide(
        cl_arr, cd_arr, out=np.full_like(cl_arr, np.nan), where=cd_arr != 0
    )

    # 4) Optional experimental overlay
    exp_df = load_experimental_points(experimental_path)

    # 5) Plot CL/CD curves
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    line_label = "VSM"
    exp_label = "WT Belloc"

    axes[0].plot(alpha_sweep_deg, cl_arr, label=line_label)
    axes[0].set_xlabel("Angle of attack [deg]")
    axes[0].set_ylabel("CL")
    axes[0].grid(True)
    if exp_df is not None:
        axes[0].scatter(
            exp_df["alpha_deg"], exp_df["cl"], color="k", s=16, label=exp_label
        )

    axes[1].plot(alpha_sweep_deg, cd_arr, label=line_label)
    axes[1].set_xlabel("Angle of attack [deg]")
    axes[1].set_ylabel("CD")
    axes[1].grid(True)
    if exp_df is not None:
        axes[1].scatter(
            exp_df["alpha_deg"], exp_df["cd"], color="k", s=16, label=exp_label
        )

    axes[2].plot(alpha_sweep_deg, cl_over_cd, label=line_label)
    axes[2].set_xlabel("Angle of attack [deg]")
    axes[2].set_ylabel("CL/CD")
    axes[2].grid(True)
    if exp_df is not None:
        axes[2].scatter(
            exp_df["alpha_deg"],
            exp_df["cl_over_cd"],
            color="k",
            s=16,
            label=exp_label,
        )

    # Single legend combining first axis entries (assumes consistent handles across axes)
    handles, labels = axes[0].get_legend_handles_labels()
    if exp_df is None:
        handles, labels = axes[0].get_legend_handles_labels()
    else:
        # Merge handles/labels across axes to ensure both series appear
        for ax in axes[1:]:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        # remove duplicates while preserving order
        uniq = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l not in uniq_labels:
                uniq.append(h)
                uniq_labels.append(l)
        handles, labels = uniq, uniq_labels

    fig.legend(handles, labels, loc="upper center", ncol=len(handles))
    fig.suptitle("Paraglider Belloc 2015 – VSM vs. experiment")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

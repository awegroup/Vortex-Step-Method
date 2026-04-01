"""Compare AoA-corrected vs non-corrected VSM against wind-tunnel data
for both the TU Delft V3 kite and the Belloc 2015 paraglider.

Outputs a 2×3 PDF with CL–alpha, CD–alpha, and CL/CD–alpha for each case:
  Row 1: V3 (WT vs VSM with and without AoA correction)
  Row 2: Belloc (WT vs VSM with and without AoA correction)
"""

from __future__ import annotations

import re
from pathlib import Path
import sys
from typing import List, Tuple

import matplotlib

# Use a non-Qt backend to avoid Wayland/QThread issues when saving figures.
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure we use the local source (not an older site-packages install)
PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.core.WingGeometry import Wing

from VSM.plot_styling import set_plot_style


def parse_all_polars(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Parse Belloc allPolarsVSM.dat into LE/TE arrays and per-section polars."""
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
    arr = np.array(rows, dtype=float)
    alpha_rad = np.deg2rad(arr[:, 0])
    return np.column_stack((alpha_rad, arr[:, 1:]))


def load_experimental_belloc(data_path: Path):
    """Load Belloc WT data (alfa, Cl, Cd). Returns None on failure."""
    try:
        df = (
            pd.read_excel(data_path, engine="odf")
            if data_path.suffix == ".ods"
            else pd.read_csv(data_path)
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Skipping Belloc experimental overlay ({exc})")
        return None

    cols_lower = {c.lower(): c for c in df.columns}
    alpha_col = next(
        (c for k, c in cols_lower.items() if ("alpha" in k) or ("alfa" in k)), None
    )
    cl_col = next((c for k, c in cols_lower.items() if k.startswith("cl")), None)
    cd_col = next((c for k, c in cols_lower.items() if k.startswith("cd")), None)
    if not all([alpha_col, cl_col, cd_col]):
        print(
            "Belloc experimental data found but columns were not recognized; skipping overlay."
        )
        return None

    exp = df[[alpha_col, cl_col, cd_col]].dropna()
    exp = exp.rename(columns={alpha_col: "alpha_deg", cl_col: "cl", cd_col: "cd"})
    exp["cl_over_cd"] = exp["cl"] / exp["cd"]
    return exp


def run_alpha_sweep(
    body: BodyAerodynamics, solver: Solver, angles_deg: np.ndarray, Umag: float
):
    """Sweep angle of attack and return cl, cd arrays."""
    cl_vals, cd_vals = [], []
    for alpha in angles_deg:
        body.va_initialize(
            Umag=Umag, angle_of_attack=float(alpha), side_slip=0.0, yaw_rate=0.0
        )
        res = solver.solve(body)
        cl_vals.append(res["cl"])
        cd_vals.append(res["cd"])
    cl_arr = np.array(cl_vals)
    cd_arr = np.array(cd_vals)
    cl_arr[~np.isfinite(cl_arr)] = np.nan
    cd_arr[~np.isfinite(cd_arr)] = np.nan
    cl_over_cd = np.divide(
        cl_arr, cd_arr, out=np.full_like(cl_arr, np.nan), where=cd_arr != 0
    )
    return cl_arr, cd_arr, cl_over_cd


def plot_valid(ax, x, y, **kwargs):
    """Plot only finite y-values to avoid broken lines when solver fails."""
    mask = np.isfinite(y)
    if np.any(mask):
        ax.plot(np.asarray(x)[mask], np.asarray(y)[mask], **kwargs)


def make_solver(is_aoa_corrected: bool) -> Solver:
    """Create Solver, tolerating older installs without is_aoa_corrected kwarg."""
    return Solver(
        reference_point=np.array([0.0, 0.0, 0.0]),
        is_aoa_corrected=is_aoa_corrected,
    )


def main(
    color_no="blue",
    color_yes="red",
    color_scatter="black",
    labels_V3=(
        r"VSM no corr ($3/4c$)",
        r"VSM $\alpha$ corr ($1/4c$)",
        r"WT LEI kite",
    ),
    labels_Belloc=(
        r"VSM no corr ($3/4c$)",
        r"VSM $\alpha$ corr ($1/4c$)",
        r"WT ram-air kite",
    ),
    cl_label=r"$C_{\mathrm{L}}$",
    cd_label=r"$C_{\mathrm{D}}$",
    clcd_label=r"$C_{\mathrm{L}}/C_{\mathrm{D}}$",
    alpha_label=r"$\alpha$ ($^\circ$)",
    figsize=(15, 8),
    n_panels: int = 50,
    n_vsm_data_points: int = 20,
    alpha_range: Tuple[float, float] = (-2, 17),
):
    set_plot_style()
    project_dir = Path(__file__).resolve().parents[2]

    # ---------------------- V3 setup ----------------------
    v3_cad_dir = project_dir / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
    v3_body = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=v3_cad_dir / "aero_geometry_CAD_CFD_polars.yaml",
        # ml_models_dir=project_dir / "data" / "ml_models",
        spanwise_panel_distribution="uniform",
    )
    solver_v3_no = make_solver(False)
    solver_v3_yes = make_solver(True)
    v3_alpha = np.linspace(alpha_range[0], alpha_range[1], n_vsm_data_points)
    Umag_v3 = 3.15

    v3_cl_no, v3_cd_no, v3_clcd_no = run_alpha_sweep(
        v3_body, solver_v3_no, v3_alpha, Umag_v3
    )
    v3_cl_yes, v3_cd_yes, v3_clcd_yes = run_alpha_sweep(
        v3_body, solver_v3_yes, v3_alpha, Umag_v3
    )

    v3_wt_path = (
        project_dir
        / "data"
        / "TUDELFT_V3_KITE"
        / "3D_polars_literature"
        / "V3_CL_CD_CS_alpha_sweep_for_beta_0_WindTunnel_Poland_2025_Rey_560e4.csv"
    )
    v3_wt = pd.read_csv(v3_wt_path)
    v3_wt_cl = v3_wt["CL"].values
    v3_wt_cd = v3_wt["CD"].values
    v3_wt_alpha = v3_wt["alpha"].values
    v3_wt_clcd = np.divide(
        v3_wt_cl, v3_wt_cd, out=np.full_like(v3_wt_cl, np.nan), where=v3_wt_cd != 0
    )

    # ---------------------- Belloc setup ----------------------
    belloc_dir = project_dir / "data" / "paraglider_Belloc_2015"
    le_arr, te_arr, polar_data = parse_all_polars(belloc_dir / "allPolarsVSM.dat")
    belloc_wing = Wing(n_panels=n_panels, spanwise_panel_distribution="cosine")
    belloc_wing.update_wing_from_points(
        le_arr=le_arr,
        te_arr=te_arr,
        aero_input_type="reuse_initial_polar_data",
        polar_data_arr=polar_data,
    )
    belloc_body = BodyAerodynamics([belloc_wing])
    solver_belloc_no = make_solver(False)
    solver_belloc_yes = make_solver(True)

    belloc_alpha = np.linspace(alpha_range[0], alpha_range[1], n_vsm_data_points)
    Umag_belloc = 10.0
    belloc_cl_yes, belloc_cd_yes, belloc_clcd_yes = run_alpha_sweep(
        belloc_body, solver_belloc_yes, belloc_alpha, Umag_belloc
    )
    belloc_cl_no, belloc_cd_no, belloc_clcd_no = run_alpha_sweep(
        belloc_body, solver_belloc_no, belloc_alpha, Umag_belloc
    )

    belloc_exp = load_experimental_belloc(belloc_dir / "experimental_data.ods")

    # ---------------------- Plotting ----------------------
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=False)

    # Row 1: V3
    # CL
    ax = axes[0, 0]
    ax.plot(
        v3_wt_alpha,
        v3_wt_cl,
        color=color_scatter,
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=1,
        label=labels_V3[2],
    )
    plot_valid(ax, v3_alpha, v3_cl_no, label=labels_V3[0], color=color_no)
    plot_valid(ax, v3_alpha, v3_cl_yes, label=labels_V3[1], color=color_yes)
    ax.set_ylabel(cl_label)
    ax.set_xlabel(alpha_label)
    ax.set_xlim(alpha_range[0], alpha_range[1])
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True)
    # CD
    ax = axes[0, 1]
    ax.plot(
        v3_wt_alpha,
        v3_wt_cd,
        color=color_scatter,
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=1,
        label=labels_V3[2],
    )
    plot_valid(ax, v3_alpha, v3_cd_no, label=labels_V3[0], color=color_no)
    plot_valid(ax, v3_alpha, v3_cd_yes, label=labels_V3[1], color=color_yes)
    ax.set_ylabel(cd_label)
    ax.set_xlabel(alpha_label)
    ax.set_xlim(alpha_range[0], alpha_range[1])
    ax.set_ylim(0, 0.4)
    ax.grid(True)
    # CL/CD
    ax = axes[0, 2]
    ax.plot(
        v3_wt_alpha,
        v3_wt_clcd,
        color=color_scatter,
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=1,
        label=labels_V3[2],
    )
    plot_valid(ax, v3_alpha, v3_clcd_no, label=labels_V3[0], color=color_no)
    plot_valid(ax, v3_alpha, v3_clcd_yes, label=labels_V3[1], color=color_yes)
    ax.set_ylabel(clcd_label)
    ax.set_xlabel(alpha_label)
    ax.set_xlim(alpha_range[0], alpha_range[1])
    ax.set_ylim(0, 16)
    ax.grid(True)

    # Row 2: Belloc
    ax = axes[1, 0]
    if belloc_exp is not None:
        ax.plot(
            belloc_exp["alpha_deg"],
            belloc_exp["cl"],
            color=color_scatter,
            marker="o",
            markersize=4,
            linestyle="-",
            label=labels_Belloc[2],
        )
    plot_valid(ax, belloc_alpha, belloc_cl_no, label=labels_Belloc[0], color=color_no)
    plot_valid(ax, belloc_alpha, belloc_cl_yes, label=labels_Belloc[1], color=color_yes)
    ax.set_ylabel(cl_label)
    ax.set_xlabel(alpha_label)
    ax.set_xlim(alpha_range[0], alpha_range[1])
    ax.set_ylim(-0.2, 1.4)
    ax.grid(True)

    ax = axes[1, 1]
    if belloc_exp is not None:
        ax.plot(
            belloc_exp["alpha_deg"],
            belloc_exp["cd"],
            color=color_scatter,
            marker="o",
            markersize=4,
            linestyle="-",
            label=labels_Belloc[2],
        )
    plot_valid(ax, belloc_alpha, belloc_cd_no, label=labels_Belloc[0], color=color_no)
    plot_valid(ax, belloc_alpha, belloc_cd_yes, label=labels_Belloc[1], color=color_yes)
    ax.set_ylabel(cd_label)
    ax.set_xlabel(alpha_label)
    ax.set_xlim(alpha_range[0], alpha_range[1])
    ax.grid(True)

    ax = axes[1, 2]
    if belloc_exp is not None:
        ax.plot(
            belloc_exp["alpha_deg"],
            belloc_exp["cl_over_cd"],
            color=color_scatter,
            marker="o",
            markersize=4,
            linestyle="-",
            label=labels_Belloc[2],
        )
    plot_valid(ax, belloc_alpha, belloc_clcd_no, label=labels_Belloc[0], color=color_no)
    plot_valid(
        ax, belloc_alpha, belloc_clcd_yes, label=labels_Belloc[1], color=color_yes
    )
    ax.set_ylabel(clcd_label)
    ax.set_xlabel(alpha_label)
    ax.set_xlim(alpha_range[0], alpha_range[1])
    ax.set_ylim(0, 30)
    ax.grid(True)

    # Legends
    axes[0, 1].legend(loc="upper left")
    axes[1, 1].legend(loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = project_dir / "results" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aoa_correction_V3_and_Belloc.pdf"
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()

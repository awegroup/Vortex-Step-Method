"""CMz sensitivity to yaw rate and sideslip on default TUDELFT_V3_KITE geometry."""

from pathlib import Path
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Geometry and solver settings
n_panels = 30
spanwise_panel_distribution = "uniform"
ml_models_dir = PROJECT_DIR / "data" / "ml_models"
cad_derived_geometry_dir = (
    PROJECT_DIR / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
)
geometry_yaml = cad_derived_geometry_dir / "aero_geometry_CAD_CFD_polars.yaml"
reference_point = np.array([0.0, 0.0, 0.0])

Umag = 27.0  # m/s resultant speed (roughly matches 10 m/s wind + 25 m/s kite)
aoa_deg = 5.0

# Sweep ranges
yaw_rates = np.linspace(-3.0, 3.0, 13)  # rad/s
sideslip_deg = np.linspace(-10.0, 10.0, 11)  # deg


def build_body() -> BodyAerodynamics:
    body = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=geometry_yaml,
        spanwise_panel_distribution=spanwise_panel_distribution,
        ml_models_dir=ml_models_dir,
        scale=1.0,
    )
    return body


def run_sweeps():
    base_body = build_body()
    solver = Solver(
        reference_point=reference_point, gamma_initial_distribution_type="zero"
    )

    yaw_rows = []
    for yaw in yaw_rates:
        body = copy.deepcopy(base_body)
        body.va_initialize(
            Umag=Umag,
            angle_of_attack=aoa_deg,
            side_slip=0.0,
            yaw_rate=yaw,
            pitch_rate=0.0,
            roll_rate=0.0,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )
        res = solver.solve(body)
        yaw_rows.append(
            {
                "yaw_rad_s": yaw,
                "cmz": res.get("cmz", np.nan),
                "cmx": res.get("cmx", np.nan),
            }
        )

    beta_rows = []
    for beta in sideslip_deg:
        body = copy.deepcopy(base_body)
        body.va_initialize(
            Umag=Umag,
            angle_of_attack=aoa_deg,
            side_slip=beta,
            yaw_rate=0.0,
            pitch_rate=0.0,
            roll_rate=0.0,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )
        res = solver.solve(body)
        beta_rows.append(
            {
                "beta_deg": beta,
                "cmz": res.get("cmz", np.nan),
                "cmx": res.get("cmx", np.nan),
            }
        )

    yaw_df = pd.DataFrame(yaw_rows)
    beta_df = pd.DataFrame(beta_rows)
    return yaw_df, beta_df


def make_plots(yaw_df: pd.DataFrame, beta_df: pd.DataFrame):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0, 0].plot(yaw_df["yaw_rad_s"], yaw_df["cmz"], "o-")
    ax[0, 0].set_xlabel("Yaw rate [rad/s]")
    ax[0, 0].set_ylabel("cmz [-]")
    ax[0, 0].set_title("CMz vs yaw rate")
    ax[0, 0].axhline(0, color="k", linewidth=0.8)

    ax[0, 1].plot(yaw_df["yaw_rad_s"], yaw_df["cmx"], "o-", color="tab:orange")
    ax[0, 1].set_xlabel("Yaw rate [rad/s]")
    ax[0, 1].set_ylabel("cmx [-]")
    ax[0, 1].set_title("CMx vs yaw rate")
    ax[0, 1].axhline(0, color="k", linewidth=0.8)

    ax[1, 0].plot(beta_df["beta_deg"], beta_df["cmz"], "o-")
    ax[1, 0].set_xlabel("Sideslip [deg]")
    ax[1, 0].set_ylabel("cmz [-]")
    ax[1, 0].set_title("CMz vs sideslip")
    ax[1, 0].axhline(0, color="k", linewidth=0.8)

    ax[1, 1].plot(beta_df["beta_deg"], beta_df["cmx"], "o-", color="tab:orange")
    ax[1, 1].set_xlabel("Sideslip [deg]")
    ax[1, 1].set_ylabel("cmx [-]")
    ax[1, 1].set_title("CMx vs sideslip")
    ax[1, 1].axhline(0, color="k", linewidth=0.8)

    fig.tight_layout()

    save_dir = PROJECT_DIR / "results" / "TUDELFT_V3_KITE"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "cmx_cmz_sensitivity.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved sensitivity plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    yaw_df, beta_df = run_sweeps()
    make_plots(yaw_df, beta_df)

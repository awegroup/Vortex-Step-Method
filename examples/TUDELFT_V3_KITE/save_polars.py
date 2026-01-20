from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.plotting import generate_3D_polar_data

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "TUDELFT_V3_KITE" / "alpha_polar.csv"


def run_alpha_sweep(output_path: Path) -> Path:
    """
    Compute and save the alpha sweep polar for the TUDELFT_V3_KITE geometry.

    Args:
        output_path: File path where the CSV will be written.

    Returns:
        The resolved output path.
    """
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Analysis settings
    n_panels = 50
    spanwise_panel_distribution = "uniform"
    Umag = 3.15
    side_slip = 0.0
    yaw_rate = 0.0
    pitch_rate = 0.0
    roll_rate = 0.0
    angle_range = np.arange(-10.0, 31.0, 1.0)  # [-10, 30] deg inclusive

    solver = Solver(reference_point=np.array([0.0, 0.0, 0.0]))
    cad_derived_geometry_dir = (
        PROJECT_DIR / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
    )
    body_aero = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(cad_derived_geometry_dir / "aero_geometry_CAD_CFD_polars.yaml"),
        spanwise_panel_distribution=spanwise_panel_distribution,
    )

    polar_data, _ = generate_3D_polar_data(
        solver=solver,
        body_aero=body_aero,
        angle_range=angle_range,
        angle_type="angle_of_attack",
        side_slip=side_slip,
        yaw_rate=yaw_rate,
        pitch_rate=pitch_rate,
        roll_rate=roll_rate,
        Umag=Umag,
    )

    aoa, cl, cd, cs, cmx, cmy, cmz = polar_data[:7]
    cl_over_cd = np.divide(cl, cd, out=np.full_like(cl, np.nan), where=cd != 0)

    df = pd.DataFrame(
        {
            "aoa": aoa,
            "CL": cl,
            "CD": cd,
            "CL/CD": cl_over_cd,
            "CS": cs,
            "CMx": cmx,
            "CMy": cmy,
            "CMz": cmz,
        }
    )
    df.to_csv(output_path, index=False)
    return output_path


def main(save_path: str | Path | None = None) -> None:
    """
    Run the alpha sweep and save the results.

    Args:
        save_path: Optional path to override the default CSV location.
    """
    output_path = DEFAULT_OUTPUT if save_path is None else Path(save_path)
    saved_path = run_alpha_sweep(output_path)
    print(f"Alpha sweep saved to: {saved_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate alpha sweep polars for TUDELFT_V3_KITE."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    main(args.output)

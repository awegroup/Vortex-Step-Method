"""
Tutorial: Running NeuralFoil, Breukels, and Masure Analysis on V3 Kite Airfoils

This script demonstrates how to use the AirfoilAerodynamics framework to predict
aerodynamic coefficients for airfoils using multiple methods:
  - NeuralFoil: Deep learning-based aerodynamic prediction
  - Breukels: Parametric regression model for LEI kite airfoils
  - Masure: Machine learning regression for parametric shapes
  - CFD: Reference computational fluid dynamics data

The script loads airfoil parameters from YAML configuration files and generates
comparison plots showing all prediction methods against CFD polars.

Prerequisites:
    - neuralfoil: pip install neuralfoil
    - PyYAML: pip install pyyaml
    - numpy, pandas, matplotlib for data processing and visualization

See AirfoilAerodynamics.py for the core implementation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
import yaml
import sys

try:
    import neuralfoil as nf
except ImportError:
    raise ImportError("neuralfoil is required. Install it with: pip install neuralfoil")

# Add src to path for importing VSM
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from VSM.core.AirfoilAerodynamics import AirfoilAerodynamics


def load_yaml_airfoil_config(yaml_file):
    """
    Load airfoil configuration from YAML file.

    Args:
        yaml_file (Path or str): Path to YAML configuration file.

    Returns:
        tuple: (airfoil_params_dict, reynolds, alpha_range)
               where airfoil_params_dict has airfoil_id as key.
    """
    yaml_file = Path(yaml_file)
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract airfoil parameters
    wing_airfoils = config.get("wing_airfoils", {})
    alpha_range = wing_airfoils.get("alpha_range", [-10, 40, 1])
    reynolds = wing_airfoils.get("reynolds", 1e6)

    airfoil_params = {}
    for row in wing_airfoils.get("data", []):
        airfoil_id = int(row[0])
        airfoil_type = row[1]
        info_dict = row[2] if len(row) > 2 else {}

        airfoil_params[airfoil_id] = {"type": airfoil_type, "info_dict": info_dict}

    print(f"  Loaded {len(airfoil_params)} configs from {yaml_file.name}")
    return airfoil_params, reynolds, alpha_range


def load_cfd_polars(cfd_polars_dir):
    """
    Load CFD polar data from CSV files.

    Args:
        cfd_polars_dir (Path or str): Directory containing CFD polar CSV files.

    Returns:
        dict: Dictionary with airfoil number as key and DataFrame as value.
    """
    cfd_polars_dir = Path(cfd_polars_dir)
    cfd_data = {}

    for csv_file in sorted(cfd_polars_dir.glob("*.csv"), key=lambda x: int(x.stem)):
        airfoil_num = int(csv_file.stem)
        df = pd.read_csv(csv_file)
        cfd_data[airfoil_num] = df

    return cfd_data


def run_analysis(
    airfoil_num,
    airfoil_type,
    airfoil_params,
    alpha_range,
    reynolds,
    file_path,
    ml_models_dir,
):
    """
    Run aerodynamic analysis using AirfoilAerodynamics framework.

    Args:
        airfoil_num (int): Airfoil identifier.
        airfoil_type (str): Type of analysis ('neuralfoil', 'breukels_regression', 'masure_regression').
        airfoil_params (dict): Parameters for the analysis method.
        alpha_range (list): [min_alpha, max_alpha, step] in degrees.
        reynolds (float): Reynolds number.
        file_path (str): Base path for relative file references.
        ml_models_dir (str): Directory containing trained ML models.

    Returns:
        dict: Dictionary with 'alpha' (in degrees), 'CL', 'CD', 'CM' keys, or None on error.
    """
    try:
        aero = AirfoilAerodynamics.from_yaml_entry(
            airfoil_type=airfoil_type,
            airfoil_params=airfoil_params,
            alpha_range=alpha_range,
            reynolds=reynolds,
            file_path=file_path,
            ml_models_dir=ml_models_dir,
        )
        return {
            "alpha": np.rad2deg(aero.alpha),
            "CL": aero.CL,
            "CD": aero.CD,
            "CM": aero.CM,
        }
    except Exception as e:
        print(f"      ERROR in {airfoil_type}: {e}")
        return None


def plot_airfoil_results(
    airfoil_num,
    nf_data,
    breukels_data=None,
    masure_data=None,
    cfd_data=None,
    save_dir=None,
):
    """
    Create comparison plots for NeuralFoil, Breukels, Masure, and CFD data.

    Args:
        airfoil_num (int): Airfoil number for identification.
        nf_data (dict): NeuralFoil results with 'alpha', 'CL', 'CD', 'CM'.
        breukels_data (dict, optional): Breukels regression results.
        masure_data (dict, optional): Masure regression results.
        cfd_data (DataFrame, optional): CFD polar data with 'alpha', 'Cl', 'Cd', 'Cm'.
        save_dir (Path, optional): Directory to save the figure.
    """
    if nf_data is None:
        return

    breukels_data = None
    # masure_data = None

    # Filter CFD data: remove alpha > 20 deg in absolute value
    if cfd_data is not None:
        cfd_data = cfd_data[np.abs(cfd_data["alpha"]) <= 20].copy()
        if len(cfd_data) == 0:
            cfd_data = None

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)

    alpha = nf_data["alpha"]

    # Plot CL
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(alpha, nf_data["CL"], "b-", linewidth=2, label="NeuralFoil")
    if breukels_data is not None:
        ax1.plot(
            breukels_data["alpha"],
            breukels_data["CL"],
            "g--",
            linewidth=2,
            label="Breukels",
        )
    if masure_data is not None:
        ax1.plot(
            masure_data["alpha"],
            masure_data["CL"],
            "m:",
            linewidth=2.5,
            label="Masure",
        )
    if cfd_data is not None:
        ax1.plot(cfd_data["alpha"], cfd_data["Cl"], "ro", markersize=6, label="CFD")
    ax1.set_xlabel("Angle of Attack (deg)")
    ax1.set_ylabel("CL")
    ax1.set_title(f"Airfoil {airfoil_num} - Lift Coefficient")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot CD
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(alpha, nf_data["CD"], "b-", linewidth=2, label="NeuralFoil")
    if breukels_data is not None:
        ax2.plot(
            breukels_data["alpha"],
            breukels_data["CD"],
            "g--",
            linewidth=2,
            label="Breukels",
        )
    if masure_data is not None:
        ax2.plot(
            masure_data["alpha"],
            masure_data["CD"],
            "m:",
            linewidth=2.5,
            label="Masure",
        )
    if cfd_data is not None:
        ax2.plot(cfd_data["alpha"], cfd_data["Cd"], "ro", markersize=6, label="CFD")
    ax2.set_xlabel("Angle of Attack (deg)")
    ax2.set_ylabel("CD")
    ax2.set_title(f"Airfoil {airfoil_num} - Drag Coefficient")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot CM
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(alpha, nf_data["CM"], "b-", linewidth=2, label="NeuralFoil")
    if breukels_data is not None:
        ax3.plot(
            breukels_data["alpha"],
            breukels_data["CM"],
            "g--",
            linewidth=2,
            label="Breukels",
        )
    if masure_data is not None:
        ax3.plot(
            masure_data["alpha"],
            masure_data["CM"],
            "m:",
            linewidth=2.5,
            label="Masure",
        )
    if cfd_data is not None:
        ax3.plot(cfd_data["alpha"], cfd_data["Cm"], "ro", markersize=6, label="CFD")
    ax3.set_xlabel("Angle of Attack (deg)")
    ax3.set_ylabel("CM")
    ax3.set_title(f"Airfoil {airfoil_num} - Moment Coefficient")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.suptitle(
        f"Airfoil {airfoil_num}: NeuralFoil vs Breukels vs Masure vs CFD Polars",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{airfoil_num}.pdf"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"    Saved plot to {save_path}")

    plt.close(fig)


def main():
    """Main execution function."""
    print("=" * 70)
    print("NeuralFoil Tutorial: V3 Kite Airfoil Analysis")
    print("=" * 70)

    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"
    cad_geom_dir = data_dir / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
    cfd_polars_dir = cad_geom_dir / "2D_polars_CFD"
    results_dir = project_root / "results" / "TUDELFT_V3_KITE" / "2D_polars"
    ml_models_dir = data_dir / "ml_models"

    # Load YAML configurations
    print("\nStep 0: Loading YAML configurations...")
    yaml_neuralfoil = cad_geom_dir / "aero_geometry_CAD_neuralfoil.yaml"
    yaml_breukels = cad_geom_dir / "aero_geometry_CAD_breukels_regression.yaml"
    yaml_masure = cad_geom_dir / "aero_geometry_CAD_masure_regression.yaml"

    nf_params, nf_reynolds, nf_alpha_range = load_yaml_airfoil_config(yaml_neuralfoil)
    breukels_params, breukels_reynolds, breukels_alpha_range = load_yaml_airfoil_config(
        yaml_breukels
    )
    masure_params, masure_reynolds, masure_alpha_range = load_yaml_airfoil_config(
        yaml_masure
    )

    # Use alpha range from NeuralFoil config (convert to list for consistency)
    alpha_range = list(nf_alpha_range)

    print(f"\nConfiguration:")
    print(
        f"  Alpha range: {alpha_range[0]} to {alpha_range[1]} deg (step {alpha_range[2]})"
    )
    print(f"  NeuralFoil Reynolds: {nf_reynolds:.0e}")
    print(f"  Breukels Reynolds: {breukels_reynolds:.0e}")
    print(f"  Masure Reynolds: {masure_reynolds:.0e}")

    # Load CFD polars
    print("\nStep 1: Loading CFD polar data...")
    cfd_data = load_cfd_polars(cfd_polars_dir)
    print(f"  Loaded CFD data for {len(cfd_data)} airfoils")

    # Run analysis and plot results
    print("\nStep 2: Running analysis for all airfoils...")
    successful_analyses = 0
    total_airfoils = len(nf_params)

    for airfoil_num in sorted(nf_params.keys()):
        print(f"\n  Airfoil {airfoil_num}:")

        # Get NeuralFoil analysis
        nf_info = nf_params[airfoil_num]["info_dict"]
        nf_results = run_analysis(
            airfoil_num,
            "neuralfoil",
            nf_info,
            alpha_range,
            nf_reynolds,
            str(cad_geom_dir / "dummy"),  # file_path will use parent directory
            str(ml_models_dir),
        )

        if nf_results is not None:
            successful_analyses += 1

            # Get Breukels analysis
            breukels_results = None
            if airfoil_num in breukels_params:
                breukels_info = breukels_params[airfoil_num]["info_dict"]
                breukels_results = run_analysis(
                    airfoil_num,
                    "breukels_regression",
                    breukels_info,
                    alpha_range,
                    breukels_reynolds,
                    str(cad_geom_dir),
                    str(ml_models_dir),
                )

            # Get Masure analysis
            masure_results = None
            if airfoil_num in masure_params:
                masure_info = masure_params[airfoil_num]["info_dict"]
                masure_results = run_analysis(
                    airfoil_num,
                    "masure_regression",
                    masure_info,
                    alpha_range,
                    masure_reynolds,
                    str(cad_geom_dir),
                    str(ml_models_dir),
                )

            # Get CFD data
            cfd_polar = cfd_data.get(airfoil_num, None)

            # Create plot
            plot_airfoil_results(
                airfoil_num,
                nf_results,
                breukels_results,
                masure_results,
                cfd_polar,
                save_dir=results_dir,
            )

    print("\n" + "=" * 70)
    print(
        f"Analysis complete: {successful_analyses}/{total_airfoils} airfoils analyzed"
    )
    print(f"Results saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        main()

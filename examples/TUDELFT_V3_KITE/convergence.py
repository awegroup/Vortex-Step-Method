from pathlib import Path
from VSM.convergence_analysis import generate_csv_files, plot_convergence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from VSM.plot_styling import set_plot_style


def main():
    """
    Run convergence analysis for TUDELFT_V3_KITE using YAML config input.
    """

    PROJECT_DIR = Path(__file__).resolve().parents[2]
    config_path = (
        Path(PROJECT_DIR)
        / "data"
        / "TUDELFT_V3_KITE"
        / "aero_geometry_CAD_CFD_polars.yaml"
    )
    convergence_analysis_dir = (
        Path(PROJECT_DIR) / "results" / "TUDELFT_V3_KITE" / "convergence_analysis"
    )
    n_panels_list = [20, 40, 50, 100, 300, 400]
    spanwise_panel_distribution = "uniform"
    Umag = 10
    angle_of_attack = 10
    side_slip = 10
    yaw_rate = 0

    # First convergence analysis (varying n_panels and model type)
    convergence_results_dir = generate_csv_files(
        config_path,
        convergence_analysis_dir,
        spanwise_panel_distribution,
        Umag,
        angle_of_attack,
        side_slip,
        yaw_rate,
        n_panels_list,
        aerodynamic_model_type_list=["VSM"],
        allowed_error_list=None,
        core_radius_fraction_list=None,
        gamma_initial_distribution_type_list=None,
        gamma_loop_type_list=None,
        max_iterations_list=None,
        relaxation_factor_list=None,
        spanwise_panel_distribution_list=None,
    )
    plot_convergence(
        convergence_results_dir,
        name="convergence",
    )

    # Second convergence analysis (varying solver parameters)
    convergence_results_dir = generate_csv_files(
        config_path,
        convergence_analysis_dir,
        spanwise_panel_distribution,
        Umag,
        angle_of_attack,
        side_slip,
        yaw_rate,
        n_panels_list,
        aerodynamic_model_type_list=None,
        allowed_error_list=[1e-2, 1e-5],
        core_radius_fraction_list=None,
        gamma_initial_distribution_type_list=["zero", "elliptical", "previous"],
        gamma_loop_type_list=None,
        max_iterations_list=[1000, 3000],
        relaxation_factor_list=[0.01, 0.05, 0.1],
        spanwise_panel_distribution_list=None,
    )
    plot_convergence(
        convergence_results_dir,
        name="convergence_solver_parameters",
    )


def convergence_relation(
    n_panels_list=[10, 15, 20, 30, 50, 100],
    Umag=10,
    angle_of_attack=[5, 10, 15],  # Now accepts array of alpha values
    side_slip=0,
    yaw_rate=0,
    spanwise_panel_distribution="uniform",
    config_file="aero_geometry_CAD_CFD_polars.yaml",
):
    """
    Create a 2x3 subplot showing actual values and errors vs n_panels for multiple angles of attack.

    This function generates fresh convergence data by running simulations with different
    panel counts and multiple angles of attack, then creates plots showing:
    - First row: CL, CD, and CMY actual values vs n_panels (one line per alpha)
    - Second row: CL, CD, and CMY errors vs n_panels (one line per alpha)

    The last (highest) panel count in n_panels_list is used as the reference solution for each alpha.

    Args:
        n_panels_list (list): List of panel counts to test
        Umag (float): Magnitude of freestream velocity (m/s)
        angle_of_attack (list): List of angles of attack (degrees)
        side_slip (float): Side slip angle (degrees)
        yaw_rate (float): Yaw rate (rad/s)
        spanwise_panel_distribution (str): Panel distribution type
        config_file (str): Name of the YAML config file to use

    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    # Apply VSM plot styling
    set_plot_style()

    print("Generating fresh convergence data for error analysis...")

    # Define simulation parameters
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    config_path = Path(PROJECT_DIR) / "data" / "TUDELFT_V3_KITE" / config_file

    # Generate fresh convergence data using a modified function with debugging
    from VSM.core.BodyAerodynamics import BodyAerodynamics
    from VSM.core.Solver import Solver
    import time

    print("Running simulations for different panel counts with debugging...")

    # Ensure angle_of_attack is a list
    if not isinstance(angle_of_attack, list):
        angle_of_attack = [angle_of_attack]

    all_results = []
    for alpha in angle_of_attack:
        print(f"\n=== Running convergence study for α = {alpha}° ===")

        results_list = []
        for i, n_panels in enumerate(n_panels_list):
            print(
                f"\n--- Run {i+1}/{len(n_panels_list)}: {n_panels} panels, α={alpha}° ---"
            )

            # Create a fresh solver instance for each run to avoid state issues
            solver_instance = Solver()

            # Create the BodyAerodynamics instance
            print(f"Creating BodyAerodynamics with {n_panels} panels...")
            body_aero = BodyAerodynamics.instantiate(
                n_panels=n_panels,
                file_path=config_path,
                spanwise_panel_distribution=spanwise_panel_distribution,
            )

            # Check the actual number of panels created
            actual_panels = (
                len(body_aero.panels) if hasattr(body_aero, "panels") else "Unknown"
            )
            print(f"Actual panels created: {actual_panels}")

            # Initialize the aerodynamic model with the test conditions
            print(f"Initializing with α={alpha}°, β={side_slip}°, Umag={Umag}m/s")
            body_aero.va_initialize(Umag, alpha, side_slip, yaw_rate)

            # Run the solver and measure runtime
            print("Running solver...")
            start_time = time.time()
            results = solver_instance.solve(body_aero)
            runtime = time.time() - start_time

            # Debug: Print results structure
            print(f"Solver results type: {type(results)}")
            if isinstance(results, dict):
                cl_val = results.get("cl", None)
                cd_val = results.get("cd", None)
                print(f"CL = {cl_val}, CD = {cd_val}")
            else:
                print(f"Results: {results}")

            results_list.append(
                {
                    "n_panels": n_panels,
                    "actual_panels": actual_panels,
                    "va_mag": Umag,
                    "alpha": alpha,
                    "beta": side_slip,
                    "yaw_rate": yaw_rate,
                    "cl": (
                        results.get("cl", None) if isinstance(results, dict) else None
                    ),
                    "cd": (
                        results.get("cd", None) if isinstance(results, dict) else None
                    ),
                    "cs": (
                        results.get("cs", None) if isinstance(results, dict) else None
                    ),
                    "cmx": (
                        results.get("cmx", None) if isinstance(results, dict) else None
                    ),
                    "cmy": (
                        results.get("cmy", None) if isinstance(results, dict) else None
                    ),
                    "cmz": (
                        results.get("cmz", None) if isinstance(results, dict) else None
                    ),
                    "runtime": runtime,
                    "results_raw": str(results)[:100],  # First 100 chars for debugging
                }
            )

            print(f"Runtime: {runtime:.3f}s")

        # Add alpha identifier to each result
        for result in results_list:
            result["alpha_case"] = f"α={alpha}°"

        all_results.extend(results_list)

    df = pd.DataFrame(all_results)

    # Save the convergence data for future reference
    base_path = (
        Path(__file__).parent / "../../results/TUDELFT_V3_KITE/convergence_analysis"
    )
    base_path.mkdir(parents=True, exist_ok=True)
    convergence_file = base_path / "convergence_error_analysis_data.csv"
    df.to_csv(convergence_file, index=False)
    print(f"Convergence data saved to: {convergence_file}")

    # Sort by alpha and n_panels to ensure proper ordering
    df = df.sort_values(["alpha", "n_panels"])

    # Find the reference panels (highest panel count)
    reference_panels = max(n_panels_list)

    # Calculate errors for each alpha separately
    alpha_data = {}
    for alpha in angle_of_attack:
        alpha_df = df[df["alpha"] == alpha].copy()

        # Find reference case for this alpha
        ref_data = alpha_df[alpha_df["n_panels"] == reference_panels]
        if ref_data.empty:
            print(
                f"Warning: No {reference_panels}-panel reference case found for α={alpha}°"
            )
            ref_data = alpha_df[alpha_df["n_panels"] == alpha_df["n_panels"].max()]

        cl_ref = ref_data["cl"].iloc[0]
        cd_ref = ref_data["cd"].iloc[0]
        cmy_ref = ref_data["cmy"].iloc[0]
        ref_panels_actual = ref_data["n_panels"].iloc[0]

        print(
            f"Reference case for α={alpha}° ({ref_panels_actual} panels): CL={cl_ref:.6f}, CD={cd_ref:.6f}, CMY={cmy_ref:.6f}"
        )

        # Calculate relative errors for this alpha
        cl_errors = (abs(alpha_df["cl"] - cl_ref) / abs(cl_ref) * 100).tolist()
        cd_errors = (abs(alpha_df["cd"] - cd_ref) / abs(cd_ref) * 100).tolist()
        cmy_errors = (abs(alpha_df["cmy"] - cmy_ref) / abs(cmy_ref) * 100).tolist()
        panel_counts = alpha_df["n_panels"].tolist()

        alpha_data[alpha] = {
            "panel_counts": panel_counts,
            "cl_values": alpha_df["cl"].tolist(),
            "cd_values": alpha_df["cd"].tolist(),
            "cmy_values": alpha_df["cmy"].tolist(),
            "cl_errors": cl_errors,
            "cd_errors": cd_errors,
            "cmy_errors": cmy_errors,
            "cl_ref": cl_ref,
            "cd_ref": cd_ref,
            "cmy_ref": cmy_ref,
        }

    try:
        # Create 2x3 subplot (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Define colors and markers for different alpha values
        colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
        markers = ["o", "s", "^", "v", "D", "p", "*", "h"]

        # Store legend entries for the external legend
        legend_handles = []
        legend_labels = []

        # First row: Actual values
        for i, alpha in enumerate(angle_of_attack):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            data = alpha_data[alpha]

            # Plot 1: CL values vs n_panels
            (line1,) = axes[0, 0].plot(
                data["panel_counts"],
                data["cl_values"],
                f"{marker}-",
                linewidth=2,
                markersize=6,
                color=color,
            )

            # Plot 2: CD values vs n_panels
            axes[0, 1].plot(
                data["panel_counts"],
                data["cd_values"],
                f"{marker}-",
                linewidth=2,
                markersize=6,
                color=color,
            )

            # Plot 3: CMY values vs n_panels
            axes[0, 2].plot(
                data["panel_counts"],
                data["cmy_values"],
                f"{marker}-",
                linewidth=2,
                markersize=6,
                color=color,
            )

            # Store legend info (only need one entry per alpha)
            legend_handles.append(line1)
            legend_labels.append(f"α={alpha}°")

        # Set labels for first row (no titles)
        axes[0, 0].set_xlabel("Number of panels")
        axes[0, 0].set_ylabel("CL")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel("Number of panels")
        axes[0, 1].set_ylabel("CD")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].set_xlabel("Number of panels")
        axes[0, 2].set_ylabel("CMY")
        axes[0, 2].grid(True, alpha=0.3)

        # Second row: Error percentages
        for i, alpha in enumerate(angle_of_attack):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            data = alpha_data[alpha]

            # Plot 4: CL error vs n_panels
            axes[1, 0].plot(
                data["panel_counts"],
                data["cl_errors"],
                f"{marker}-",
                linewidth=2,
                markersize=6,
                color=color,
            )

            # Plot 5: CD error vs n_panels
            axes[1, 1].plot(
                data["panel_counts"],
                data["cd_errors"],
                f"{marker}-",
                linewidth=2,
                markersize=6,
                color=color,
            )

            # Plot 6: CMY error vs n_panels
            axes[1, 2].plot(
                data["panel_counts"],
                data["cmy_errors"],
                f"{marker}-",
                linewidth=2,
                markersize=6,
                color=color,
            )

        # Set labels for second row (no titles)
        axes[1, 0].set_xlabel("Number of panels")
        axes[1, 0].set_ylabel("CL error (%)")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_xlabel("Number of panels")
        axes[1, 1].set_ylabel("CD error (%)")
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].set_xlabel("Number of panels")
        axes[1, 2].set_ylabel("CMY error (%)")
        axes[1, 2].grid(True, alpha=0.3)

        # Adjust layout to make room for external legend
        plt.tight_layout()

        # Add external legend below the plot without bounding box
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(angle_of_attack),
            frameon=False,
        )

        # Save figure as PDF
        output_file = base_path / "convergence_error_analysis.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Convergence error analysis plot saved to: {output_file}")

        # Display some statistics
        print(f"\nConvergence Error Analysis Summary:")
        print(f"Reference solution: {reference_panels} panels")

        for alpha in angle_of_attack:
            data = alpha_data[alpha]
            print(f"\nFor α={alpha}°:")
            print(f"CL reference value: {data['cl_ref']:.6f}")
            print(f"CD reference value: {data['cd_ref']:.6f}")
            print(f"CMY reference value: {data['cmy_ref']:.6f}")
            print("Panel count vs CL error (%):")
            for n_panels, cl_err in zip(data["panel_counts"], data["cl_errors"]):
                print(f"  {n_panels:3.0f} panels: {cl_err:.3f}%")
            print("Panel count vs CD error (%):")
            for n_panels, cd_err in zip(data["panel_counts"], data["cd_errors"]):
                print(f"  {n_panels:3.0f} panels: {cd_err:.3f}%")
            print("Panel count vs CMY error (%):")
            for n_panels, cmy_err in zip(data["panel_counts"], data["cmy_errors"]):
                print(f"  {n_panels:3.0f} panels: {cmy_err:.3f}%")

        return fig, axes

    except Exception as e:
        print(f"Error in convergence_relation: {e}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Run convergence analysis
    # main()

    # Generate convergence error plots with custom parameters
    print("\n" + "=" * 60)
    print("Creating convergence error analysis plots...")
    print("=" * 60)
    convergence_relation(
        n_panels_list=[5, 10, 15, 20, 30, 90, 100, 150],
        Umag=10,
        angle_of_attack=[5, 10, 15],  # Array of alpha values
        side_slip=0,
        yaw_rate=0,
        spanwise_panel_distribution="uniform",
        config_file=Path("CAD_derived_geometry") / "aero_geometry_CAD_CFD_polars.yaml",
    )

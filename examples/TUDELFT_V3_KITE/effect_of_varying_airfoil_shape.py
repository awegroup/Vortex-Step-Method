#!/usr/bin/env python3
"""
Generate parameter variation YAML files for studying the effect of varying airfoil shape parameters.

This script reads the masure_regression config file and creates variations for each parameter:
t, eta, kappa, delta, lambda, phi

For each parameter, two files are created:
- {parameter}_min{percent}.yaml: with {percent}% smaller values
- {parameter}_plus{percent}.yaml: with {percent}% larger values

All other parameters remain unchanged from the original config.
"""

import yaml
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.core.AirfoilAerodynamics import AirfoilAerodynamics
from VSM.plot_styling import set_plot_style


def create_parameter_variations(percent_variation=10):
    """
    Create parameter variation files for masure_regression parameters.

    Args:
        percent_variation (float): Percentage variation to apply (e.g., 10 for ±10%)
    """
    # Define paths
    project_dir = Path(__file__).resolve().parents[2]
    original_config_path = (
        project_dir
        / "data"
        / "TUDELFT_V3_KITE"
        / "config_kite_CAD_masure_regression.yaml"
    )
    output_base_dir = (
        project_dir / "results" / "TUDELFT_V3_KITE" / "effect_of_varying_airfoil_shape"
    )

    # Load the original configuration
    print(f"Reading original config from: {original_config_path}")
    with open(original_config_path, "r", encoding="utf-8") as f:
        original_config = yaml.safe_load(f)

    # Parameters to vary
    parameters = ["t", "eta", "kappa", "delta", "lambda", "phi"]

    # Create output directories and generate files
    for param in parameters:
        print(f"\nProcessing parameter: {param}")

        # Create parameter directory
        param_dir = output_base_dir / param
        param_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {param_dir}")

        # Calculate multipliers based on percentage variation
        multiplier_min = 1 - (percent_variation / 100)
        multiplier_plus = 1 + (percent_variation / 100)

        # Generate variations
        for variation, multiplier in [
            ("min{}".format(int(percent_variation)), multiplier_min),
            ("plus{}".format(int(percent_variation)), multiplier_plus),
        ]:
            # Create a deep copy of the original config
            modified_config = copy.deepcopy(original_config)

            # Modify the specific parameter for all airfoils
            wing_airfoils_data = modified_config["wing_airfoils"]["data"]

            for airfoil_entry in wing_airfoils_data:
                airfoil_id, airfoil_type, info_dict = airfoil_entry

                # Only modify masure_regression airfoils
                if airfoil_type == "masure_regression":
                    if param in info_dict:
                        original_value = info_dict[param]
                        new_value = original_value * multiplier
                        info_dict[param] = round(
                            new_value, 6
                        )  # Round to 6 decimal places

                        print(
                            f"  Airfoil {airfoil_id}: {param} {original_value:.6f} -> {new_value:.6f}"
                        )

            # Save the modified config with proper formatting
            output_filename = f"{param}_{variation}.yaml"
            output_path = param_dir / output_filename

            # Custom YAML representers to handle formatting like the original
            def represent_none(dumper, _data):
                return dumper.represent_scalar("tag:yaml.org,2002:null", "")

            def represent_list(dumper, data):
                # Force inline (flow) style for data arrays
                if data and len(data) <= 10:  # Typical data row length
                    # Special handling for rows that contain dictionaries
                    if any(isinstance(item, dict) for item in data):
                        # For wing_airfoils data rows with info_dict
                        return dumper.represent_sequence(
                            "tag:yaml.org,2002:seq", data, flow_style=True
                        )
                    # For simple numeric/string data rows
                    elif all(isinstance(item, (int, float, str)) for item in data):
                        return dumper.represent_sequence(
                            "tag:yaml.org,2002:seq", data, flow_style=True
                        )
                return dumper.represent_sequence(
                    "tag:yaml.org,2002:seq", data, flow_style=False
                )

            # Apply custom representers
            yaml.add_representer(type(None), represent_none)
            yaml.add_representer(list, represent_list)

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    modified_config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2,
                    allow_unicode=True,
                )

            # Reset representers to avoid affecting other files
            yaml.representer.Representer.yaml_representers.pop(type(None), None)
            yaml.representer.Representer.yaml_representers.pop(list, None)

            print(f"  Saved: {output_path}")

    print("\nParameter variation files created successfully!")
    print(f"Output directory: {output_base_dir}")

    # Print summary
    print("\nSummary of generated files:")
    for param in parameters:
        param_dir = output_base_dir / param
        files = list(param_dir.glob("*.yaml"))
        print(f"  {param}: {len(files)} files ({', '.join(f.name for f in files)})")


def verify_parameter_variations(percent_variation=10):
    """
    Verify that the parameter variations were created correctly by checking a few examples.

    Args:
        percent_variation (float): Percentage variation that was applied (e.g., 10 for ±10%)
    """
    project_dir = Path(__file__).resolve().parents[2]
    original_config_path = (
        project_dir
        / "data"
        / "TUDELFT_V3_KITE"
        / "config_kite_CAD_masure_regression.yaml"
    )
    variations_base_dir = (
        project_dir / "results" / "TUDELFT_V3_KITE" / "effect_of_varying_airfoil_shape"
    )

    # Load original config
    with open(original_config_path, "r", encoding="utf-8") as f:
        original_config = yaml.safe_load(f)

    print("\nVerification of parameter variations:")
    print("=" * 50)

    # Check the first airfoil (airfoil_id = 1) for all parameters
    original_airfoil_1 = None
    for airfoil_entry in original_config["wing_airfoils"]["data"]:
        if airfoil_entry[0] == 1:  # airfoil_id = 1
            original_airfoil_1 = airfoil_entry[2]  # info_dict
            break

    if original_airfoil_1 is None:
        print("Could not find airfoil with id=1 in original config")
        return

    print(f"Original airfoil 1 parameters: {original_airfoil_1}")
    print()

    # Check each parameter variation
    parameters = ["t", "eta", "kappa", "delta", "lambda", "phi"]

    for param in parameters:
        print(f"Parameter: {param}")
        original_value = original_airfoil_1[param]

        for variation, expected_multiplier in [
            ("min{}".format(int(percent_variation)), 1 - percent_variation / 100),
            ("plus{}".format(int(percent_variation)), 1 + percent_variation / 100),
        ]:
            variation_file = variations_base_dir / param / f"{param}_{variation}.yaml"

            if variation_file.exists():
                with open(variation_file, "r", encoding="utf-8") as f:
                    variation_config = yaml.safe_load(f)

                # Find airfoil 1 in the variation config
                variation_airfoil_1 = None
                for airfoil_entry in variation_config["wing_airfoils"]["data"]:
                    if airfoil_entry[0] == 1:  # airfoil_id = 1
                        variation_airfoil_1 = airfoil_entry[2]  # info_dict
                        break

                if variation_airfoil_1:
                    variation_value = variation_airfoil_1[param]
                    expected_value = original_value * expected_multiplier

                    print(
                        f"  {variation}: {original_value:.6f} -> {variation_value:.6f} (expected: {expected_value:.6f})"
                    )

                    # Check if the change is correct (within small tolerance for floating point)
                    if abs(variation_value - expected_value) < 1e-10:
                        print("    ✓ Correct")
                    else:
                        print(
                            f"    ✗ Error: expected {expected_value:.6f}, got {variation_value:.6f}"
                        )
                else:
                    print(f"  {variation}: Could not find airfoil 1")
            else:
                print(f"  {variation}: File not found")

        print()


def generate_2D_plots(percent_variation=10, alpha_range=(-3, 15, 1), parameters=["t"]):
    """
    Generate CL vs alpha plots for each parameter variation using 2D airfoil aerodynamics,
    comparing default, min%, and plus% cases.

    Args:
        percent_variation (float): Percentage variation that was applied (e.g., 10 for ±10%)
        alpha_range (tuple): (start, stop, step) for alpha range in degrees
    """
    # Apply VSM plot styling
    set_plot_style()

    # Define paths
    project_dir = Path(__file__).resolve().parents[2]
    original_config_path = (
        project_dir
        / "data"
        / "TUDELFT_V3_KITE"
        / "config_kite_CAD_masure_regression.yaml"
    )
    variations_base_dir = (
        project_dir / "results" / "TUDELFT_V3_KITE" / "effect_of_varying_airfoil_shape"
    )

    # Parameters to analyze (just 't' for faster testing)
    parameters = ["t"]

    # Generate alpha array
    alpha_start, alpha_stop, alpha_step = alpha_range
    alpha_values = np.arange(alpha_start, alpha_stop + alpha_step, alpha_step)

    print(
        f"\nGenerating 2D CL vs alpha plots for ±{percent_variation}% parameter variations..."
    )
    print(f"Alpha range: {alpha_start}° to {alpha_stop}° (step: {alpha_step}°)")

    # Common airfoil parameters
    reynolds = 1e6  # Default Reynolds number from the config

    for param in parameters:
        print(f"\nProcessing parameter: {param}")
        param_dir = variations_base_dir / param

        # Define the three config files to compare
        config_files = {
            "Default": original_config_path,
            f"-{int(percent_variation)}%": param_dir
            / f"{param}_min{int(percent_variation)}.yaml",
            f"+{int(percent_variation)}%": param_dir
            / f"{param}_plus{int(percent_variation)}.yaml",
        }

        print(f"  Config files to compare:")
        for case_name, config_path in config_files.items():
            exists = "✓" if config_path.exists() else "✗"
            print(f"    {case_name}: {config_path} {exists}")

        # Store results for each case
        results = {}

        for case_name, config_path in config_files.items():
            if not config_path.exists():
                print(f"  Warning: Config file not found: {config_path}")
                continue

            print(f"  Computing 2D aerodynamics for {case_name} case...")

            # Load the YAML config file
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Get the first airfoil parameters as a representative case
            wing_airfoils_data = config["wing_airfoils"]["data"]
            first_airfoil = wing_airfoils_data[0]  # [airfoil_id, type, info_dict]
            airfoil_type = first_airfoil[1]
            airfoil_params = first_airfoil[2]

            # Get alpha_range and reynolds from config
            config_alpha_range = config["wing_airfoils"]["alpha_range"]
            config_reynolds = config["wing_airfoils"]["reynolds"]

            try:
                # Create AirfoilAerodynamics instance using the config parameters
                # Use the original config path for cache directory resolution to avoid
                # creating cache folders in the results directory
                cache_file_path = (
                    str(original_config_path)
                    if case_name != "Default"
                    else str(config_path)
                )
                airfoil_aero = AirfoilAerodynamics.from_yaml_entry(
                    airfoil_type=airfoil_type,
                    airfoil_params=airfoil_params,
                    alpha_range=[alpha_start, alpha_stop, alpha_step],
                    reynolds=config_reynolds,
                    file_path=cache_file_path,
                )

                # Extract CL, CD, CM values for the desired alpha range
                # Interpolate to get values at our specific alpha points
                alpha_rad = np.deg2rad(alpha_values)
                cl_values = np.interp(alpha_rad, airfoil_aero.alpha, airfoil_aero.CL)
                cd_values = np.interp(alpha_rad, airfoil_aero.alpha, airfoil_aero.CD)
                cm_values = np.interp(alpha_rad, airfoil_aero.alpha, airfoil_aero.CM)

                results[case_name] = {"cl": cl_values, "cd": cd_values, "cm": cm_values}
                print(
                    f"  ✓ Completed {case_name} case - CL: [{np.min(cl_values):.3f}, {np.max(cl_values):.3f}], "
                    f"CD: [{np.min(cd_values):.5f}, {np.max(cd_values):.5f}], "
                    f"CM: [{np.min(cm_values):.3f}, {np.max(cm_values):.3f}]"
                )

            except Exception as e:
                print(f"  ✗ Error computing {case_name} case: {e}")
                results[case_name] = {
                    "cl": np.full(len(alpha_values), np.nan),
                    "cd": np.full(len(alpha_values), np.nan),
                    "cm": np.full(len(alpha_values), np.nan),
                }

        # Create the plot
        if len(results) > 0 and any(
            not np.all(np.isnan(vals["cl"])) for vals in results.values()
        ):
            print(f"  Creating 2D airfoil multi-column plot for {param}...")
            print(f"  Results available for: {list(results.keys())}")

            # Create 1x3 subplot (1 row, 3 columns)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Define colors and line styles
            colors = {
                "Default": "black",
                f"-{int(percent_variation)}%": "blue",
                f"+{int(percent_variation)}%": "red",
            }
            line_styles = {
                "Default": "-",
                f"-{int(percent_variation)}%": "--",
                f"+{int(percent_variation)}%": "-.",
            }

            # Plot data for each coefficient
            coefficients = [
                ("cl", "2D CL [-]", 0),
                ("cd", "2D CD [-]", 1),
                ("cm", "2D CM [-]", 2),
            ]

            # Store handles and labels for the external legend
            legend_handles = []
            legend_labels = []

            for coeff_name, ylabel, col_idx in coefficients:
                ax = axes[col_idx]

                # Plot each case
                for case_name, case_data in results.items():
                    coeff_vals = case_data[coeff_name]
                    # Filter out NaN values for plotting
                    valid_mask = ~np.isnan(coeff_vals)
                    if np.any(valid_mask):
                        print(
                            f"    Plotting {case_name} {coeff_name.upper()} with {np.sum(valid_mask)} valid points"
                        )
                        line = ax.plot(
                            alpha_values[valid_mask],
                            coeff_vals[valid_mask],
                            color=colors.get(case_name, "gray"),
                            linestyle=line_styles.get(case_name, "-"),
                            linewidth=2,
                            marker="o",
                            markersize=4,
                            label=case_name,
                        )[0]

                        # Collect handles and labels for legend (only from first subplot)
                        if col_idx == 0:
                            legend_handles.append(line)
                            legend_labels.append(case_name)
                    else:
                        print(
                            f"    Warning: No valid {coeff_name.upper()} data for {case_name}"
                        )

                # Customize each subplot
                ax.set_xlabel("Angle of Attack [°]")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(alpha_start - 0.5, alpha_stop + 0.5)

            # Create external legend below the plot
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=len(legend_labels),
                frameon=False,
            )

            # Adjust layout to prevent overlap and make room for legend
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)

            # Save as PDF
            plot_path = param_dir / f"{param}_2D_coefficients_vs_alpha_comparison.pdf"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()  # Close to free memory

            print(f"  Saved plot: {plot_path}")
        else:
            print(f"  No valid results for {param}, skipping plot")

    print("\n2D CL, CD, CM vs alpha plots generation completed!")


def generate_2D_and_3D_plots(
    percent_variation=10, alpha_range=(-3, 15, 1), parameters=["t"]
):
    """
    Generate combined 2D and 3D CL, CD, CM vs alpha plots for each parameter variation.
    Top row shows 2D airfoil aerodynamics, bottom row shows 3D VSM results.

    Args:
        percent_variation (float): Percentage variation that was applied (e.g., 10 for ±10%)
        alpha_range (tuple): (start, stop, step) for alpha range in degrees
    """
    # Apply VSM plot styling
    set_plot_style()

    # Define paths
    project_dir = Path(__file__).resolve().parents[2]
    original_config_path = (
        project_dir
        / "data"
        / "TUDELFT_V3_KITE"
        / "config_kite_CAD_masure_regression.yaml"
    )
    variations_base_dir = (
        project_dir / "results" / "TUDELFT_V3_KITE" / "effect_of_varying_airfoil_shape"
    )

    # Generate alpha array
    alpha_start, alpha_stop, alpha_step = alpha_range
    alpha_values = np.arange(alpha_start, alpha_stop + alpha_step, alpha_step)

    print(
        f"\nGenerating 2D and 3D comparison plots for ±{percent_variation}% parameter variations..."
    )
    print(f"Alpha range: {alpha_start}° to {alpha_stop}° (step: {alpha_step}°)")

    # Common simulation parameters
    Umag = 10.0  # m/s
    side_slip = 0.0  # degrees
    yaw_rate = 0.0  # rad/s
    spanwise_panel_distribution = "uniform"
    n_panels = 50  # Use moderate panel count for speed

    for param in parameters:
        print(f"\nProcessing parameter: {param}")
        param_dir = variations_base_dir / param

        # Define the three config files to compare
        config_files = {
            "Default": original_config_path,
            f"-{int(percent_variation)}%": param_dir
            / f"{param}_min{int(percent_variation)}.yaml",
            f"+{int(percent_variation)}%": param_dir
            / f"{param}_plus{int(percent_variation)}.yaml",
        }

        print(f"  Config files to compare:")
        for case_name, config_path in config_files.items():
            exists = "✓" if config_path.exists() else "✗"
            print(f"    {case_name}: {config_path} {exists}")

        # Store results for each case (both 2D and 3D)
        results_2D = {}
        results_3D = {}

        for case_name, config_path in config_files.items():
            if not config_path.exists():
                print(f"  Warning: Config file not found: {config_path}")
                continue

            print(f"  Computing 2D and 3D aerodynamics for {case_name} case...")

            # Load the YAML config file
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Get the first airfoil parameters as a representative case
            wing_airfoils_data = config["wing_airfoils"]["data"]
            first_airfoil = wing_airfoils_data[0]  # [airfoil_id, type, info_dict]
            airfoil_type = first_airfoil[1]
            airfoil_params = first_airfoil[2]

            # Get alpha_range and reynolds from config
            config_reynolds = config["wing_airfoils"]["reynolds"]

            # --- 2D AIRFOIL COMPUTATION ---
            try:
                # Use the original config path for cache directory resolution
                cache_file_path = (
                    str(original_config_path)
                    if case_name != "Default"
                    else str(config_path)
                )
                airfoil_aero = AirfoilAerodynamics.from_yaml_entry(
                    airfoil_type=airfoil_type,
                    airfoil_params=airfoil_params,
                    alpha_range=[alpha_start, alpha_stop, alpha_step],
                    reynolds=config_reynolds,
                    file_path=cache_file_path,
                )

                # Extract 2D values
                alpha_rad = np.deg2rad(alpha_values)
                cl_2d = np.interp(alpha_rad, airfoil_aero.alpha, airfoil_aero.CL)
                cd_2d = np.interp(alpha_rad, airfoil_aero.alpha, airfoil_aero.CD)
                cm_2d = np.interp(alpha_rad, airfoil_aero.alpha, airfoil_aero.CM)

                results_2D[case_name] = {"cl": cl_2d, "cd": cd_2d, "cm": cm_2d}
                print(
                    f"    ✓ 2D completed - CL: [{np.min(cl_2d):.3f}, {np.max(cl_2d):.3f}]"
                )

            except Exception as e:
                print(f"    ✗ Error computing 2D {case_name} case: {e}")
                results_2D[case_name] = {
                    "cl": np.full(len(alpha_values), np.nan),
                    "cd": np.full(len(alpha_values), np.nan),
                    "cm": np.full(len(alpha_values), np.nan),
                }

            # --- 3D VSM COMPUTATION ---
            cl_3d_values = []
            cd_3d_values = []
            cm_3d_values = []

            for alpha in alpha_values:
                try:
                    # Create a fresh solver instance
                    solver_instance = Solver()

                    # Create BodyAerodynamics instance
                    body_aero = BodyAerodynamics.instantiate(
                        n_panels=n_panels,
                        file_path=config_path,
                        spanwise_panel_distribution=spanwise_panel_distribution,
                        is_with_bridles=False,
                    )

                    # Initialize aerodynamic model
                    body_aero.va_initialize(Umag, alpha, side_slip, yaw_rate)

                    # Run solver
                    result = solver_instance.solve(body_aero)

                    # Extract CL, CD, CM
                    if isinstance(result, dict):
                        cl_3d_values.append(result.get("cl", np.nan))
                        cd_3d_values.append(result.get("cd", np.nan))
                        cm_3d_values.append(result.get("cmy", np.nan))  # CMY for 3D
                    else:
                        cl_3d_values.append(np.nan)
                        cd_3d_values.append(np.nan)
                        cm_3d_values.append(np.nan)

                except Exception as e:
                    print(f"    Error in 3D computation at α={alpha}°: {e}")
                    cl_3d_values.append(np.nan)
                    cd_3d_values.append(np.nan)
                    cm_3d_values.append(np.nan)

            results_3D[case_name] = {
                "cl": np.array(cl_3d_values),
                "cd": np.array(cd_3d_values),
                "cm": np.array(cm_3d_values),
            }

            # Count valid 3D results
            valid_3d = np.sum(~np.isnan(cl_3d_values))
            print(f"    ✓ 3D completed - {valid_3d}/{len(alpha_values)} valid points")

        # Create the combined plot
        if len(results_2D) > 0 and len(results_3D) > 0:
            print(f"  Creating combined 2D/3D plot for {param}...")

            # Create 2x3 subplot (2 rows, 3 columns)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Define colors and line styles
            colors = {
                "Default": "black",
                f"-{int(percent_variation)}%": "blue",
                f"+{int(percent_variation)}%": "red",
            }
            line_styles = {
                "Default": "-",
                f"-{int(percent_variation)}%": "--",
                f"+{int(percent_variation)}%": "-.",
            }

            # Plot data for each coefficient
            coefficients = [
                ("cl", "CL [-]", 0),
                ("cd", "CD [-]", 1),
                ("cm", "CM [-]", 2),
            ]

            # Store handles and labels for the external legend
            legend_handles = []
            legend_labels = []

            for coeff_name, ylabel_base, col_idx in coefficients:
                # --- TOP ROW: 2D RESULTS ---
                ax_2d = axes[0, col_idx]

                for case_name, case_data in results_2D.items():
                    coeff_vals = case_data[coeff_name]
                    valid_mask = ~np.isnan(coeff_vals)
                    if np.any(valid_mask):
                        line = ax_2d.plot(
                            alpha_values[valid_mask],
                            coeff_vals[valid_mask],
                            color=colors.get(case_name, "gray"),
                            linestyle=line_styles.get(case_name, "-"),
                            linewidth=2,
                            marker="o",
                            markersize=4,
                            label=case_name,
                        )[0]

                        # Collect handles and labels for legend (only from first subplot)
                        if col_idx == 0:
                            legend_handles.append(line)
                            legend_labels.append(case_name)

                ax_2d.set_xlabel("Angle of Attack [°]")
                ax_2d.set_ylabel(f"2D {ylabel_base}")
                ax_2d.grid(True, alpha=0.3)
                ax_2d.set_xlim(alpha_start - 0.5, alpha_stop + 0.5)

                # --- BOTTOM ROW: 3D RESULTS ---
                ax_3d = axes[1, col_idx]

                for case_name, case_data in results_3D.items():
                    coeff_vals = case_data[coeff_name]
                    valid_mask = ~np.isnan(coeff_vals)
                    if np.any(valid_mask):
                        ax_3d.plot(
                            alpha_values[valid_mask],
                            coeff_vals[valid_mask],
                            color=colors.get(case_name, "gray"),
                            linestyle=line_styles.get(case_name, "-"),
                            linewidth=2,
                            marker="s",  # Square markers for 3D to distinguish from 2D
                            markersize=4,
                            label=case_name,
                        )

                ax_3d.set_xlabel("Angle of Attack [°]")
                ax_3d.set_ylabel(f"3D {ylabel_base}")
                ax_3d.grid(True, alpha=0.3)
                ax_3d.set_xlim(alpha_start - 0.5, alpha_stop + 0.5)

            # Create external legend below the plot
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=len(legend_labels),
                frameon=False,
            )

            # Adjust layout to prevent overlap and make room for legend
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)

            # Save as PDF
            plot_path = param_dir / f"{param}_2D_and_3D_coefficients_comparison.pdf"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()  # Close to free memory

            print(f"  Saved combined plot: {plot_path}")
        else:
            print(f"  No valid results for {param}, skipping plot")

    print("\n2D and 3D comparison plots generation completed!")


if __name__ == "__main__":
    print("Creating parameter variation files for airfoil shape study...")
    print("=" * 60)

    # User configurable percentage variation
    percentage_variation = 50  # %
    parameters = ["t", "eta", "kappa", "delta", "lambda", "phi"]
    alpha_range = (-3, 15, 1)  # Degrees

    print(f"Using ±{percentage_variation}% parameter variations")
    print(f"Parameters to vary: {', '.join(parameters)}")
    print(
        f"Alpha range for plots: {alpha_range[0]}° to {alpha_range[1]}° (step: {alpha_range[2]}°)"
    )

    # Create the parameter variations
    create_parameter_variations(percentage_variation)

    # Verify the results
    verify_parameter_variations(percentage_variation)

    # # Generate 2D-only plots (faster)
    # generate_2D_plots(
    #     percent_variation=percentage_variation,
    #     alpha_range=alpha_range,
    #     parameters=parameters,
    # )

    # Generate combined 2D and 3D plots
    generate_2D_and_3D_plots(
        percent_variation=percentage_variation,
        alpha_range=alpha_range,
        parameters=parameters,
    )

    print("\nDone!")

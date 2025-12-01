"""
Tow Angle Geometry Analysis for TUDELFT V3 Kite

This script analyzes the geometric relationship between angle of attack and tow angle,
comparing simulation results with experimental data from reel-out and reel-in phases.

Key Computations:
- Geometric tow angle: λ_b = arctan(|Δx| / L_bridle)
- Center of pressure trajectory as function of angle of attack
- Lift and drag coefficient polynomials fitted to simulation data

Outputs:
- Publication-quality plot: tow_angle_vs_aoa.pdf
- Polynomial coefficients for CL(α) and CD(α)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.fitting import fit_and_evaluate_model
from VSM.plot_styling import set_plot_style, PALETTE


def main():
    """Analyze tow angle geometry and compare with experimental data."""

    # ========================================================================
    # SETUP
    # ========================================================================
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    file_path = (
        PROJECT_DIR
        / "data"
        / "TUDELFT_V3_KITE"
        / "CAD_derived_geometry"
        / "aero_geometry_CAD_CFD_polars.yaml"
    )
    save_folder = PROJECT_DIR / "results" / "TUDELFT_V3_KITE"
    save_folder.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    # Settings
    n_panels = 40
    spanwise_panel_distribution = "uniform"
    reference_point = np.array([0.0, 0.0, 0.0])
    solver = Solver(reference_point=reference_point)

    # Load kite geometry (without bridles for cleaner analysis)
    print("Loading kite geometry...")
    body_aero = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=file_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        bridle_path=None,
    )

    # Flow conditions
    Umag = 20.0  # m/s

    # Experimental data (from flight tests)
    mean_aoa_exp_reelout = 6.37
    std_aoa_exp_reelout = 1.93
    mean_aoa_exp_reelin = 3.0  # rough estimate
    std_aoa_exp_reelin = 0.9

    # ========================================================================
    # COMPUTE TOW ANGLE VS ANGLE OF ATTACK
    # ========================================================================
    angle_of_attack_range = np.linspace(-5, 10, 61)

    # Get front bridle attachment point
    corner_points = np.array([panel.corner_points for panel in body_aero.panels])
    N = len(corner_points)
    x_corner = corner_points[N // 2, 0, :]  # Front center panel
    fbridle_length = np.linalg.norm(x_corner - reference_point)

    print(f"Front bridle length: {fbridle_length:.3f} m")
    print(f"Computing aerodynamics for {len(angle_of_attack_range)} angles...")

    # Initialize storage
    center_of_pressure = np.zeros((len(angle_of_attack_range), 3))
    tow_angle = np.zeros(len(angle_of_attack_range))
    cl = np.zeros(len(angle_of_attack_range))
    cd = np.zeros(len(angle_of_attack_range))

    # Sweep angle of attack
    gamma = None
    for i, alpha in enumerate(angle_of_attack_range):
        body_aero.va_initialize(Umag, alpha, side_slip=0.0)
        results = solver.solve(body_aero, gamma_distribution=gamma)

        center_of_pressure[i, :] = results["center_of_pressure"]

        # Compute geometric tow angle
        x_tow_point = x_corner[0] - center_of_pressure[i, 0]
        tow_angle[i] = np.arctan(abs(x_tow_point) / fbridle_length)

        # Store force coefficients
        cl[i] = np.sqrt(results["cl"] ** 2 + results["cs"] ** 2)
        cd[i] = results["cd"]

    # ========================================================================
    # FIT POLYNOMIAL MODELS
    # ========================================================================
    print("\nFitting polynomial models...")

    dependencies = ["np.ones(len(alpha))", "alpha", "alpha**2"]

    # Fit lift coefficient
    fit_cl = fit_and_evaluate_model(
        cl,
        dependencies=dependencies,
        alpha=angle_of_attack_range / 180 * np.pi,
    )
    print("Lift coefficient CL = c0 + c1*α + c2*α²")
    print(f"  Coefficients: {fit_cl['coeffs']}")

    # Fit drag coefficient
    fit_cd = fit_and_evaluate_model(
        cd,
        dependencies=dependencies,
        alpha=angle_of_attack_range / 180 * np.pi,
    )
    print("Drag coefficient CD = c0 + c1*α + c2*α²")
    print(f"  Coefficients: {fit_cd['coeffs']}")

    # ========================================================================
    # PLOT TOW ANGLE VS ANGLE OF ATTACK
    # ========================================================================
    print("\nGenerating plot...")

    fig, ax = plt.subplots(figsize=(5, 4))

    # Main curve
    ax.plot(
        angle_of_attack_range,
        tow_angle * 180 / np.pi,
        linewidth=2,
        color=PALETTE["Dark Blue"],
        label="VSM simulation",
    )

    # Experimental ranges
    ax.axvspan(
        mean_aoa_exp_reelout - std_aoa_exp_reelout,
        mean_aoa_exp_reelout + std_aoa_exp_reelout,
        alpha=0.3,
        color=PALETTE["Orange"],
        label="Reel-out range",
    )

    ax.axvspan(
        mean_aoa_exp_reelin - std_aoa_exp_reelin,
        mean_aoa_exp_reelin + std_aoa_exp_reelin,
        alpha=0.3,
        color=PALETTE["Sky Blue"],
        label="Reel-in range",
    )

    ax.set_xlabel(r"Angle of attack, $\alpha_\mathrm{w}$ [$^\circ$]")
    ax.set_ylabel(r"Tow angle, $\lambda_{\mathrm{b}}$ [$^\circ$]")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_folder / "tow_angle_vs_aoa.pdf")
    print(f"Saved: {save_folder / 'tow_angle_vs_aoa.pdf'}")
    plt.show()

    # ========================================================================
    # PLOT POLYNOMIAL FITS
    # ========================================================================
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Lift coefficient
    axs[0].plot(angle_of_attack_range, cl, "o", label="VSM data", alpha=0.6)
    axs[0].plot(
        angle_of_attack_range,
        fit_cl["data_est"],
        "-",
        label="Polynomial fit",
        linewidth=2,
    )
    axs[0].set_xlabel("Angle of Attack [°]")
    axs[0].set_ylabel("Lift Coefficient CL [-]")
    axs[0].set_title("Lift Coefficient vs Angle of Attack")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Drag coefficient
    axs[1].plot(angle_of_attack_range, cd, "o", label="VSM data", alpha=0.6)
    axs[1].plot(
        angle_of_attack_range,
        fit_cd["data_est"],
        "-",
        label="Polynomial fit",
        linewidth=2,
    )
    axs[1].set_xlabel("Angle of Attack [°]")
    axs[1].set_ylabel("Drag Coefficient CD [-]")
    axs[1].set_title("Drag Coefficient vs Angle of Attack")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Bridle length: {fbridle_length:.3f} m")
    print(
        f"Angle of attack range: {angle_of_attack_range[0]:.1f}° to "
        f"{angle_of_attack_range[-1]:.1f}°"
    )
    print(
        f"Tow angle range: {np.min(tow_angle)*180/np.pi:.2f}° to "
        f"{np.max(tow_angle)*180/np.pi:.2f}°"
    )

    # Center of pressure travel
    cp_x_range = np.max(center_of_pressure[:, 0]) - np.min(center_of_pressure[:, 0])
    print(
        f"Center of pressure X-travel: {cp_x_range:.3f} m "
        f"({cp_x_range/2.6*100:.1f}% of chord)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Tow Point Location Parametric Study for TUDELFT V3 Kite

This script performs a parametric study to understand how the location of the
tow point (reference point) affects the kite's moment coefficients and trim angles.

Two parametric sweeps are performed:
1. X-position sweep: How does fore/aft position affect moments? (with bridles)
2. Z-position sweep: How does height affect moments and bridle length? (without bridles)

Key Outputs:
- Moment coefficients (CMx, CMy, CMz) vs angle of attack for different tow point positions
- Moment coefficients vs sideslip for different tow point positions
- Trim angles as function of tow point location
- Bridle length vs trim angle relationship
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from scipy.interpolate import interp1d


def main():
    """Parametric study of tow point location effects."""

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
    bridle_path = (
        PROJECT_DIR
        / "data"
        / "TUDELFT_V3_KITE"
        / "CAD_derived_geometry"
        / "struc_geometry.yaml"
    )

    n_panels = 40
    spanwise_panel_distribution = "uniform"
    Umag = 20.0  # m/s

    print("=" * 70)
    print("TOW POINT LOCATION PARAMETRIC STUDY")
    print("=" * 70)

    # ========================================================================
    # PART 1: X-POSITION SWEEP (with bridles)
    # ========================================================================
    print("\nPART 1: X-Position Parametric Sweep")
    print("-" * 70)

    # Load geometry with bridles
    print("Loading kite geometry with bridles...")
    body_aero_with_bridles = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=file_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        bridle_path=bridle_path,
    )

    # Define x-positions to test
    x_tow_points = np.arange(-1.25, 0.5, 0.25)
    alpha_range = np.linspace(0, 20, 11)
    sideslip_range = np.linspace(0, 10, 11)

    print(f"X-positions to test: {x_tow_points}")
    print(f"Angle of attack range: {alpha_range[0]:.0f}° to {alpha_range[-1]:.0f}°")
    print(f"Sideslip range: {sideslip_range[0]:.0f}° to {sideslip_range[-1]:.0f}°")

    # Storage for results
    cmx_alpha_all = []
    cmy_alpha_all = []
    cmz_alpha_all = []
    trim_aoa_x_all = []

    cmx_ss_all = []
    cmy_ss_all = []
    cmz_ss_all = []

    # Sweep over x-positions
    for x in x_tow_points:
        solver = Solver(reference_point=[x, 0, 0])
        print(f"\n  Testing x = {x:.2f} m...")

        # Angle of attack sweep
        cmx_alpha = []
        cmy_alpha = []
        cmz_alpha = []

        for alpha in alpha_range:
            body_aero_with_bridles.va_initialize(Umag, alpha, side_slip=0.0)
            results = solver.solve(body_aero_with_bridles)
            cmx_alpha.append(results.get("cmx", np.nan))
            cmy_alpha.append(results.get("cmy", np.nan))
            cmz_alpha.append(results.get("cmz", np.nan))

        cmx_alpha_all.append(cmx_alpha)
        cmy_alpha_all.append(cmy_alpha)
        cmz_alpha_all.append(cmz_alpha)

        # Find trim angle of attack (where CMy = 0)
        cmy_array = np.array(cmy_alpha)
        if np.any(np.diff(np.sign(cmy_array))):
            f_trim = interp1d(
                cmy_array,
                alpha_range,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            trim_aoa = f_trim(0.0)
            trim_aoa_x_all.append(trim_aoa)
            print(f"    Trim AoA: {trim_aoa:.2f}°")
        else:
            trim_aoa_x_all.append(np.nan)
            print(f"    Trim AoA: Not found")

        # Sideslip sweep at trim angle
        cmx_ss = []
        cmy_ss = []
        cmz_ss = []

        for ss in sideslip_range:
            body_aero_with_bridles.va_initialize(Umag, trim_aoa_x_all[-1], side_slip=ss)
            results = solver.solve(body_aero_with_bridles)
            cmx_ss.append(results.get("cmx", np.nan))
            cmy_ss.append(results.get("cmy", np.nan))
            cmz_ss.append(results.get("cmz", np.nan))

        cmx_ss_all.append(cmx_ss)
        cmy_ss_all.append(cmy_ss)
        cmz_ss_all.append(cmz_ss)

    # Plot X-position results: Angle of Attack dependency
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for i, x in enumerate(x_tow_points):
        axs[0].plot(alpha_range, cmz_alpha_all[i], "o-", label=f"x={x:.2f} m")
        axs[1].plot(alpha_range, cmx_alpha_all[i], "o-", label=f"x={x:.2f} m")
        axs[2].plot(alpha_range, cmy_alpha_all[i], "o-", label=f"x={x:.2f} m")

    axs[0].axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axs[0].set_xlabel("Angle of Attack [°]")
    axs[0].set_ylabel("Yaw Moment Coefficient CMz [-]")
    axs[0].set_title("Yaw Moment vs Angle of Attack")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axs[1].set_xlabel("Angle of Attack [°]")
    axs[1].set_ylabel("Roll Moment Coefficient CMx [-]")
    axs[1].set_title("Roll Moment vs Angle of Attack")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    axs[2].axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axs[2].set_xlabel("Angle of Attack [°]")
    axs[2].set_ylabel("Pitch Moment Coefficient CMy [-]")
    axs[2].set_title("Pitch Moment vs Angle of Attack")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.suptitle("Effect of X-Position on Moments (with bridles)")
    plt.tight_layout()
    plt.show()

    # Plot X-position results: Sideslip dependency
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for i, x in enumerate(x_tow_points):
        axs[0].plot(sideslip_range, cmz_ss_all[i], "o-", label=f"x={x:.2f} m")
        axs[1].plot(sideslip_range, cmx_ss_all[i], "o-", label=f"x={x:.2f} m")
        axs[2].plot(sideslip_range, cmy_ss_all[i], "o-", label=f"x={x:.2f} m")

    axs[0].axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axs[0].set_xlabel("Sideslip Angle [°]")
    axs[0].set_ylabel("Yaw Moment Coefficient CMz [-]")
    axs[0].set_title("Yaw Moment vs Sideslip")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axs[1].set_xlabel("Sideslip Angle [°]")
    axs[1].set_ylabel("Roll Moment Coefficient CMx [-]")
    axs[1].set_title("Roll Moment vs Sideslip")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    axs[2].axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axs[2].set_xlabel("Sideslip Angle [°]")
    axs[2].set_ylabel("Pitch Moment Coefficient CMy [-]")
    axs[2].set_title("Pitch Moment vs Sideslip")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.suptitle("Effect of X-Position on Moments at Trim AoA (with bridles)")
    plt.tight_layout()
    plt.show()

    # ========================================================================
    # PART 2: Z-POSITION SWEEP (without bridles for cleaner geometry study)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Z-Position (Height) Parametric Sweep")
    print("-" * 70)

    # Load geometry without bridles
    print("Loading kite geometry without bridles...")
    body_aero_no_bridles = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=file_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        bridle_path=None,
    )

    # Define z-positions to test
    z_heights = np.linspace(0, 8, 9)
    alpha_range_z = np.linspace(-5, 10, 31)

    print(f"Z-heights to test: {z_heights}")
    print(f"Angle of attack range: {alpha_range_z[0]:.0f}° to {alpha_range_z[-1]:.0f}°")

    # Storage for results
    cmy_z_all = []
    trim_aoa_z_all = []
    bridle_lengths = []

    # Sweep over z-positions
    for z in z_heights:
        reference_point = np.array([0, 0, z])
        solver = Solver(reference_point=reference_point)
        print(f"\n  Testing z = {z:.2f} m...")

        # Calculate bridle length
        corner_points = np.array(
            [panel.corner_points for panel in body_aero_no_bridles.panels]
        )
        N = len(corner_points)
        x_corner = corner_points[N // 2, 0, :]
        bridle_length = np.linalg.norm(x_corner - reference_point)
        bridle_lengths.append(bridle_length)
        print(f"    Bridle length: {bridle_length:.3f} m")

        # Angle of attack sweep
        cmy_z = []
        for alpha in alpha_range_z:
            body_aero_no_bridles.va_initialize(Umag, alpha, side_slip=0.0)
            results = solver.solve(body_aero_no_bridles)
            cmy_z.append(results["cmy"])

        cmy_z_all.append(cmy_z)

        # Find trim angle
        cmy_array = np.array(cmy_z)
        sign_changes = np.where(np.diff(np.sign(cmy_array)))[0]

        if len(sign_changes) > 0:
            idx = sign_changes[0]
            f_interp = interp1d(
                cmy_array[idx : idx + 2], alpha_range_z[idx : idx + 2], kind="linear"
            )
            trim_aoa_z = f_interp(0)
            trim_aoa_z_all.append(trim_aoa_z)
            print(f"    Trim AoA: {trim_aoa_z:.2f}°")
        else:
            trim_aoa_z_all.append(np.nan)
            print(f"    Trim AoA: Not found")

    # Plot Z-position results: CMy vs AoA
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, z in enumerate(z_heights):
        ax.plot(
            alpha_range_z,
            cmy_z_all[i],
            "o-",
            label=f"z={z:.1f} m (L={bridle_lengths[i]:.2f} m)",
        )

    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Angle of Attack [°]")
    ax.set_ylabel("Pitch Moment Coefficient CMy [-]")
    ax.set_title("Effect of Z-Position on Pitch Moment")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # Plot Trim angle vs bridle length
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(bridle_lengths, trim_aoa_z_all, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Bridle Length [m]")
    ax.set_ylabel("Trim Angle of Attack [°]")
    ax.set_title("Trim Angle vs Bridle Length")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nPart 1: X-Position Effects (with bridles)")
    print(f"  X-range tested: {x_tow_points[0]:.2f} to {x_tow_points[-1]:.2f} m")
    print(
        f"  Trim AoA range: {np.nanmin(trim_aoa_x_all):.2f}° to "
        f"{np.nanmax(trim_aoa_x_all):.2f}°"
    )

    print("\nPart 2: Z-Position Effects (without bridles)")
    print(f"  Z-range tested: {z_heights[0]:.2f} to {z_heights[-1]:.2f} m")
    print(
        f"  Bridle length range: {bridle_lengths[0]:.3f} to {bridle_lengths[-1]:.3f} m"
    )
    print(
        f"  Trim AoA range: {np.nanmin(trim_aoa_z_all):.2f}° to "
        f"{np.nanmax(trim_aoa_z_all):.2f}°"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

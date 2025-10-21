"""
Kite Stability Dynamics Analysis for TUDELFT V3 Kite

This script analyzes the dynamic stability characteristics of the kite by computing:
- Restoring moment stiffness (k = -dCMy/dα)
- Natural frequency and oscillation period
- Damping characteristics

The analysis is performed for different reference point heights to understand how
bridle attachment point affects the dynamic response.

Key Outputs:
- Natural frequency ω_n [rad/s]
- Oscillation period T [s]
- Restoring stiffness k [N·m/rad]
- Stability margins

References:
- Rigid body dynamics: M_θ = I_y * θ̈ = -k*θ  →  ω_n = sqrt(k/I_y)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from scipy.interpolate import interp1d


def main():
    """Analyze kite stability dynamics for different configurations."""

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

    print("=" * 70)
    print("KITE STABILITY DYNAMICS ANALYSIS")
    print("=" * 70)

    # Settings
    n_panels = 40
    spanwise_panel_distribution = "uniform"
    Umag = 20.0  # m/s
    mass_kite = 12.0  # kg

    # Load kite geometry (without bridles for fundamental dynamics study)
    print("\nLoading kite geometry...")
    body_aero = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=file_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        bridle_path=None,
    )

    # Define reference point heights to analyze
    ref_heights = np.linspace(0, 8, 9)
    angle_of_attack_range = np.linspace(-5, 10, 31)

    print(f"Reference heights: {ref_heights}")
    print(
        f"Angle of attack range: {angle_of_attack_range[0]:.0f}° to "
        f"{angle_of_attack_range[-1]:.0f}°"
    )
    print(f"Kite mass: {mass_kite} kg")

    # ========================================================================
    # COMPUTE DYNAMICS FOR EACH CONFIGURATION
    # ========================================================================

    # Storage arrays
    trim_angles = []
    bridle_lengths = []
    restoring_stiffness = []
    natural_frequency = []
    period_full = []
    period_half = []
    time_to_zero = []
    moment_of_inertia = []

    print("\n" + "-" * 70)
    print("Computing stability parameters...")
    print("-" * 70)

    for z in ref_heights:
        reference_point = np.array([0, 0, z])
        solver = Solver(reference_point=reference_point)

        print(f"\nHeight z = {z:.2f} m:")

        # Get bridle attachment point
        corner_points = np.array([panel.corner_points for panel in body_aero.panels])
        N = len(corner_points)
        x_corner = corner_points[N // 2, 0, :]
        bridle_length = np.linalg.norm(x_corner - reference_point)
        bridle_lengths.append(bridle_length)
        print(f"  Bridle length: {bridle_length:.3f} m")

        # Compute CMy vs angle of attack
        cmy_vs_alpha = []
        for alpha in angle_of_attack_range:
            body_aero.va_initialize(Umag, alpha, side_slip=0.0)
            results = solver.solve(body_aero)
            cmy_vs_alpha.append(results["cmy"])

        cmy_array = np.array(cmy_vs_alpha)

        # Find trim angle (where CMy = 0)
        sign_changes = np.where(np.diff(np.sign(cmy_array)))[0]

        if len(sign_changes) > 0:
            idx = sign_changes[0]
            f_interp = interp1d(
                cmy_array[idx : idx + 2],
                angle_of_attack_range[idx : idx + 2],
                kind="linear",
            )
            trim_angle = f_interp(0)
            trim_angles.append(trim_angle)
            print(f"  Trim angle: {trim_angle:.2f}°")

            # Compute slope dCMy/dα at trim point using central differences
            dCMy_dalpha_field = np.gradient(
                cmy_array, np.deg2rad(angle_of_attack_range)
            )
            dCMy_dalpha = np.interp(
                trim_angle, angle_of_attack_range, dCMy_dalpha_field
            )

            # Restoring stiffness: k = -dCMy/dα (negative because positive α should give negative CMy)
            # Note: This is dimensional [N·m/rad] after multiplying by q*S*c
            # For now we use the non-dimensional value
            k = -dCMy_dalpha
            restoring_stiffness.append(k)
            print(f"  Restoring stiffness k: {k:.4f} [per rad]")

            # Check stability
            if k <= 0:
                print(f"  WARNING: Unstable! (k ≤ 0)")
                natural_frequency.append(np.nan)
                period_full.append(np.nan)
                period_half.append(np.nan)
                time_to_zero.append(np.nan)
                moment_of_inertia.append(np.nan)
                continue

            # Estimate moment of inertia (simple model: slender rod about center)
            # I_y ≈ m * L² / 12, where L is kite height
            L_kite = x_corner[2] - z  # Vertical distance from ref point to kite
            I_y = mass_kite * (L_kite**2) / 12.0
            moment_of_inertia.append(I_y)
            print(f"  Moment of inertia I_y: {I_y:.3f} kg·m²")

            if I_y <= 0:
                print(f"  WARNING: Invalid I_y!")
                natural_frequency.append(np.nan)
                period_full.append(np.nan)
                period_half.append(np.nan)
                time_to_zero.append(np.nan)
                continue

            # Natural frequency (undamped): ω_n = sqrt(k/I_y)
            # Note: k needs to be dimensional [N·m/rad]
            # For proper calculation: k_dimensional = k * q * S * c_MAC
            q_inf = 0.5 * 1.225 * Umag**2  # Dynamic pressure
            S = 21.22  # Projected area [m²] (from kite geometry)
            c_MAC = 2.565  # Mean aerodynamic chord [m]
            k_dimensional = k * q_inf * S * c_MAC

            omega_n = np.sqrt(k_dimensional / I_y)
            natural_frequency.append(omega_n)
            print(
                f"  Natural frequency ω_n: {omega_n:.3f} rad/s ({omega_n/(2*np.pi):.3f} Hz)"
            )

            # Periods
            T = 2.0 * np.pi / omega_n
            T_half = np.pi / omega_n
            t_0 = np.pi / (2.0 * omega_n)

            period_full.append(T)
            period_half.append(T_half)
            time_to_zero.append(t_0)

            print(f"  Full period T: {T:.3f} s")
            print(f"  Half period T/2: {T_half:.3f} s")
            print(f"  Time to zero: {t_0:.3f} s")

        else:
            print(f"  No trim angle found!")
            trim_angles.append(np.nan)
            restoring_stiffness.append(np.nan)
            natural_frequency.append(np.nan)
            period_full.append(np.nan)
            period_half.append(np.nan)
            time_to_zero.append(np.nan)
            moment_of_inertia.append(np.nan)

    # ========================================================================
    # PLOT RESULTS
    # ========================================================================

    # Plot 1: Trim angle vs bridle length
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bridle_lengths, trim_angles, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Bridle Length [m]")
    ax.set_ylabel("Trim Angle of Attack [°]")
    ax.set_title("Trim Angle vs Bridle Length")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Restoring stiffness vs bridle length
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bridle_lengths, restoring_stiffness, "o-", linewidth=2, markersize=8)
    ax.axhline(
        0, color="r", linestyle="--", linewidth=1, alpha=0.7, label="Stability boundary"
    )
    ax.set_xlabel("Bridle Length [m]")
    ax.set_ylabel("Restoring Stiffness k [per rad]")
    ax.set_title("Restoring Stiffness vs Bridle Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 3: Natural frequency and period
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Natural frequency
    ax1.plot(bridle_lengths, natural_frequency, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Bridle Length [m]")
    ax1.set_ylabel("Natural Frequency ω_n [rad/s]")
    ax1.set_title("Natural Frequency vs Bridle Length")
    ax1.grid(True, alpha=0.3)

    # Period
    ax2.plot(
        bridle_lengths,
        period_full,
        "o-",
        label="Full period T",
        linewidth=2,
        markersize=8,
    )
    ax2.plot(
        bridle_lengths,
        period_half,
        "s-",
        label="Half period T/2",
        linewidth=2,
        markersize=8,
    )
    ax2.set_xlabel("Bridle Length [m]")
    ax2.set_ylabel("Period [s]")
    ax2.set_title("Oscillation Period vs Bridle Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot 4: Moment of inertia vs bridle length
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bridle_lengths, moment_of_inertia, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Bridle Length [m]")
    ax.set_ylabel("Moment of Inertia I_y [kg·m²]")
    ax.set_title("Moment of Inertia vs Bridle Length")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(
        f"{'Height':<8} {'L_bridle':<10} {'Trim α':<8} {'k':<10} {'ω_n':<10} {'T':<8}"
    )
    print(
        f"{'[m]':<8} {'[m]':<10} {'[°]':<8} {'[1/rad]':<10} {'[rad/s]':<10} {'[s]':<8}"
    )
    print("-" * 70)

    for i, z in enumerate(ref_heights):
        print(
            f"{z:<8.2f} {bridle_lengths[i]:<10.3f} {trim_angles[i]:<8.2f} "
            f"{restoring_stiffness[i]:<10.4f} {natural_frequency[i]:<10.3f} "
            f"{period_full[i]:<8.3f}"
        )

    print("=" * 70)

    # Stability assessment
    print("\nSTABILITY ASSESSMENT:")
    stable_configs = np.array(restoring_stiffness) > 0
    n_stable = np.sum(~np.isnan(stable_configs) & stable_configs)
    print(f"  Stable configurations: {n_stable}/{len(ref_heights)}")

    if n_stable > 0:
        valid_idx = ~np.isnan(natural_frequency)
        print(
            f"  Natural frequency range: {np.nanmin(natural_frequency):.3f} to "
            f"{np.nanmax(natural_frequency):.3f} rad/s"
        )
        print(
            f"  Period range: {np.nanmin(period_full):.3f} to "
            f"{np.nanmax(period_full):.3f} s"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()

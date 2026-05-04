"""Solve a quasi-steady trim state and compute aerodynamic stability derivatives.

Solves the trim problem then linearises forces/moments around the trim point using
compute_stability_derivatives. Prints Jacobians, state-space matrices, eigenvalues
and timescales, and shows eigenvalue pole plots.

No inertial forces are included in the linearisation. The tether force from trim
is transferred from the reference point to the CG as a fixed force + moment.

Longitudinal states : [u (m/s, course), theta (rad, pitch), q (rad/s, pitch rate)]
Lateral states      : [phi (rad, roll), psi (rad, yaw),
                       p (rad/s, roll rate), r (rad/s, yaw rate)]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.quasi_steady_state import (
    DEFAULT_AXES,
    compute_stability_derivatives,
    solve_quasi_steady_state,
)
from awetrim.system.system_model import SystemModel
from awetrim.system.tether import RigidLumpedTether


PROJECT_DIR = Path(__file__).resolve().parents[2]

GEOMETRY_YAML = (
    PROJECT_DIR
    / "data"
    / "TUDELFT_V3_KITE"
    / "deformed_geometry"
    / "aero_geometry.yaml"
)
BRIDLE_PATH = (
    PROJECT_DIR
    / "data"
    / "TUDELFT_V3_KITE"
    / "deformed_geometry"
    / "struc_geometry.yaml"
)
REFERENCE_POINT = np.array([0.0, 0.0, 0.0], dtype=float)
CENTER_OF_GRAVITY = np.array([0.1, 0.0, 5.0], dtype=float)

# Inertial parameters (kite body)
MASS = 32  # kg
IXX = 100.0  # kg·m² (roll)
IYY = 20.0  # kg·m² (pitch)
IZZ = 100.0  # kg·m² (yaw)

MOMENT_TOLERANCE = 1e-3
N_PANELS = 18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_body(tilt_deg: float) -> BodyAerodynamics:
    body = BodyAerodynamics.instantiate(
        n_panels=N_PANELS,
        file_path=GEOMETRY_YAML,
        spanwise_panel_distribution="uniform",
        bridle_path=BRIDLE_PATH,
    )
    body.rotate(
        angle_deg=tilt_deg,
        axis=DEFAULT_AXES.course,
        point=np.array([0.5, 0.0, 7.0]),
    )
    return body


def build_system_model(case: dict) -> SystemModel:
    system = SystemModel(tether=RigidLumpedTether(diameter=0.01))
    system.mass_wing = MASS
    system.angle_elevation = np.deg2rad(case["elevation_deg"])
    system.angle_azimuth = np.deg2rad(case["azimuth_deg"])
    system.angle_course = np.deg2rad(case["course_deg"])
    system.speed_radial = case["radial_speed"]
    system.distance_radial = case["distance_radial"]
    system.wind.speed_wind_ref = case["wind_speed"]
    system.timeder_speed_tangential = 0.0
    system.timeder_speed_radial = 0.0
    return system


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def _sep(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def print_trim(result: dict) -> None:
    _sep("QUASI-STEADY TRIM SOLUTION")
    opt_x = np.asarray(result["opt_x"], dtype=float)
    cm = np.asarray(result["cm"], dtype=float)
    F_aero = np.asarray(result["total_aero_force_vec"], dtype=float)
    print(f"  success_optimizer  : {result['success']}")
    print(f"  success_physical   : {result['success_physical']}")
    print(f"  kite_speed [m/s]   : {opt_x[0]: .4f}")
    print(f"  roll [deg]         : {opt_x[1]: .4f}")
    print(f"  pitch [deg]        : {opt_x[2]: .4f}")
    print(f"  yaw [deg]          : {opt_x[3]: .4f}")
    print(f"  course_rate[rad/s] : {opt_x[4]: .5f}")
    print(f"  aoa_center [deg]   : {result['aoa_deg']: .4f}")
    print(f"  beta_center [deg]  : {result['side_slip_deg']: .4f}")
    print(f"  cl                 : {result['cl']: .4f}")
    print(f"  cd                 : {result['cd']: .4f}")
    print(f"  cm [cmx,cmy,cmz]   : [{cm[0]: .3e}  {cm[1]: .3e}  {cm[2]: .3e}]")
    print(
        f"  F_aero [N]         : [{F_aero[0]: .2f}  {F_aero[1]: .2f}  {F_aero[2]: .2f}]"
    )
    print(f"  tether_force [N]   : {result['tether_force']: .2f}")
    print(
        f"  va_trim [m/s]      : {np.array2string(result['va_vel_world'], precision=3)}"
    )


def print_stability(stab: dict) -> None:
    _sep("TETHER FORCE TRANSFERRED TO CG")
    print(f"  F_tether [N]       : {np.array2string(stab['F_tether'], precision=3)}")
    print(
        f"  M_tether_CG [N·m]  : {np.array2string(stab['M_tether_at_CG'], precision=3)}"
    )

    _sep("LONGITUDINAL JACOBIAN  J_long (3×3)")
    print("  Rows = [X (course), Z (radial), My (pitch)]")
    print("  Cols = [u (m/s),    theta (rad), q (rad/s)]")
    print()
    hdr = f"  {'':10s}  {'u':>14s}  {'theta':>14s}  {'q':>14s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, row_label in enumerate(["X [N/(m/s)]", "Z [N/(m/s)]", "My [N·m/…]"]):
        r = stab["J_long"][i]
        print(f"  {row_label:<12s}  {r[0]:+14.4e}  {r[1]:+14.4e}  {r[2]:+14.4e}")

    _sep("LATERAL JACOBIAN  J_lat (3×5)")
    print("  Rows = [Y (normal), Mx (roll), Mz (yaw)]")
    print("  Cols = [v (m/s), phi (rad), psi (rad), p (rad/s), r (rad/s)]")
    print()
    hdr = f"  {'':10s}  {'v':>14s}  {'phi':>14s}  {'psi':>14s}  {'p':>14s}  {'r':>14s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, row_label in enumerate(["Y", "Mx", "Mz"]):
        r = stab["J_lat"][i]
        print(
            f"  {row_label:<12s}  {r[0]:+14.4e}  {r[1]:+14.4e}  {r[2]:+14.4e}  {r[3]:+14.4e}  {r[4]:+14.4e}"
        )

    _sep("LONGITUDINAL STATE-SPACE  A_long (3×3)")
    print("  States = [u, theta, q]")
    print("  Row 0: m·du/dt = X   Row 1: dtheta/dt = q   Row 2: Iy·dq/dt = My")
    print()
    hdr = f"  {'':12s}  {'u':>14s}  {'theta':>14s}  {'q':>14s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, label in enumerate(["d(u)/dt", "d(theta)/dt", "d(q)/dt"]):
        r = stab["A_long"][i]
        print(f"  {label:<14s}  {r[0]:+14.4e}  {r[1]:+14.4e}  {r[2]:+14.4e}")

    _sep("LATERAL STATE-SPACE  A_lat (5×5)")
    print("  States = [v, phi, psi, p, r]")
    print("  Row 0: m·dv/dt = Y   Row 1: dphi/dt = p   Row 2: dpsi/dt = r")
    print("  Row 3: Ix·dp/dt = Mx   Row 4: Iz·dr/dt = Mz")
    print()
    hdr = f"  {'':12s}  {'v':>14s}  {'phi':>14s}  {'psi':>14s}  {'p':>14s}  {'r':>14s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, label in enumerate(
        ["d(v)/dt", "d(phi)/dt", "d(psi)/dt", "d(p)/dt", "d(r)/dt"]
    ):
        r = stab["A_lat"][i]
        print(
            f"  {label:<14s}  {r[0]:+14.4e}  {r[1]:+14.4e}  {r[2]:+14.4e}  {r[3]:+14.4e}  {r[4]:+14.4e}"
        )

    _sep("LONGITUDINAL EIGENVALUES")
    print(f"  stable_long : {stab['stable_long']}")
    print()
    print(f"  {'#':>3s}  {'Re(λ) [1/s]':>16s}  {'Im(λ) [rad/s]':>16s}  {'T [s]':>12s}")
    print("  " + "-" * 55)
    for i, (eig, T) in enumerate(zip(stab["eig_long"], stab["Tfast_long"])):
        T_str = f"{T:.4f}" if np.isfinite(T) else "    inf"
        print(f"  {i:>3d}  {eig.real:+16.6f}  {eig.imag:+16.6f}  {T_str:>12s}")

    _sep("LATERAL EIGENVALUES")
    print(f"  stable_lat  : {stab['stable_lat']}")
    print()
    print(f"  {'#':>3s}  {'Re(λ) [1/s]':>16s}  {'Im(λ) [rad/s]':>16s}  {'T [s]':>12s}")
    print("  " + "-" * 55)
    for i, (eig, T) in enumerate(zip(stab["eig_lat"], stab["Tfast_lat"])):
        T_str = f"{T:.4f}" if np.isfinite(T) else "    inf"
        print(f"  {i:>3d}  {eig.real:+16.6f}  {eig.imag:+16.6f}  {T_str:>12s}")


def _nondim_scales(V_ref: float, b_ref: float, c_ref: float) -> tuple:
    """Return scale vectors that make each state dimensionless.

    Longitudinal: [u/V,  θ,  q·c/(2V)]
    Lateral:      [v/V,  φ,  ψ,  p·b/(2V),  r·b/(2V)]
    """
    long_scales = np.array([V_ref, 1.0, 1])
    lat_scales = np.array([V_ref, 1.0, 1.0, 1, 1])
    return long_scales, lat_scales


def print_mode_decomposition(
    stab: dict, V_ref: float, b_ref: float, c_ref: float
) -> None:
    long_scales, lat_scales = _nondim_scales(V_ref, b_ref, c_ref)

    long_labels = [
        f"u/V  (={V_ref:.1f} m/s)",
        "θ (rad)",
        f"q·c/2V  (÷{2*V_ref/c_ref:.1f})",
    ]
    lat_labels = [
        f"v/V  (={V_ref:.1f} m/s)",
        "φ (rad)",
        "ψ (rad)",
        f"p·b/2V  (÷{2*V_ref/b_ref:.1f})",
        f"r·b/2V  (÷{2*V_ref/b_ref:.1f})",
    ]

    for block_label, eigs, vecs, labels, scales in [
        ("LONGITUDINAL", stab["eig_long"], stab["vec_long"], long_labels, long_scales),
        ("LATERAL", stab["eig_lat"], stab["vec_lat"], lat_labels, lat_scales),
    ]:
        _sep(f"{block_label} MODE DECOMPOSITION  (non-dimensionalised)")
        n_modes = len(eigs)
        for mi in range(n_modes):
            eig = eigs[mi]
            raw_vec = np.asarray(vecs[:, mi], dtype=complex)

            # Non-dimensionalise: divide by the reference scale for each state
            vec = raw_vec / scales

            # Normalise so the largest non-dim amplitude = 1∠0°
            amps = np.abs(vec)
            dominant = int(np.argmax(amps))
            vec_norm = vec / vec[dominant]

            freq = abs(eig.imag) / (2 * np.pi)
            T_str = f"{1.0/abs(eig.real):.3f} s" if abs(eig.real) > 1e-10 else "∞"
            stab_char = "stable" if eig.real < 0 else "UNSTABLE"
            osc_char = (
                f"  ω={eig.imag:+.3f} rad/s  f={freq:.3f} Hz"
                if abs(eig.imag) > 1e-6
                else "  (non-oscillatory)"
            )

            print(
                f"\n  Mode {mi}:  λ = {eig.real:+.4f} {eig.imag:+.4f}j"
                f"    T = {T_str}    [{stab_char}]{osc_char}"
            )
            print(f"  {'State':<22s}  {'Non-dim amp':>12s}  {'Phase [deg]':>12s}")
            print("  " + "-" * 52)
            for si, lbl in enumerate(labels):
                amp = float(np.abs(vec_norm[si]))
                phase = float(np.angle(vec_norm[si], deg=True))
                dom = " ◀" if si == dominant else ""
                print(f"  {lbl:<22s}  {amp:12.4f}  {phase:+12.2f}°{dom}")


def plot_poles(stab: dict) -> None:
    fig, axes_list = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Aerodynamic Stability Poles (no inertial forces)", fontsize=12)

    datasets = [
        (
            axes_list[0],
            stab["eig_long"],
            "Longitudinal\nstates=[u, θ, q]",
            stab["stable_long"],
        ),
        (
            axes_list[1],
            stab["eig_lat"],
            "Lateral\nstates=[v, φ, ψ, p, r]",
            stab["stable_lat"],
        ),
    ]

    for ax, eigs, title, stable in datasets:
        ax.axvline(0, color="k", linewidth=0.8, linestyle="--", zorder=0)
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--", zorder=0)
        color = "tab:green" if stable else "tab:red"
        ax.scatter(np.real(eigs), np.imag(eigs), color=color, s=90, zorder=5)
        for i, eig in enumerate(eigs):
            ax.annotate(f" λ{i}", (eig.real, eig.imag), fontsize=8)
        stability_label = "STABLE" if stable else "UNSTABLE"
        ax.set_title(f"{title}\n[{stability_label}]", color=color)
        ax.set_xlabel("Re(λ)  [1/s]")
        ax.set_ylabel("Im(λ)  [rad/s]")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()


def plot_mode_shapes(stab: dict, V_ref: float, b_ref: float, c_ref: float) -> None:
    long_scales, lat_scales = _nondim_scales(V_ref, b_ref, c_ref)
    long_labels = ["u/V", "θ", "q·c/2V"]
    lat_labels = ["v/V", "φ", "ψ", "p·b/2V", "r·b/2V"]

    for block_title, eigs, vecs, labels, scales in [
        (
            "Longitudinal mode shapes  (non-dim)",
            stab["eig_long"],
            stab["vec_long"],
            long_labels,
            long_scales,
        ),
        (
            "Lateral mode shapes  (non-dim)",
            stab["eig_lat"],
            stab["vec_lat"],
            lat_labels,
            lat_scales,
        ),
    ]:
        n_modes = len(eigs)
        n_states = len(labels)
        fig, axes_list = plt.subplots(
            1, n_modes, figsize=(3.5 * n_modes, 4), sharey=False
        )
        if n_modes == 1:
            axes_list = [axes_list]
        fig.suptitle(block_title, fontsize=11)

        for mi, ax in enumerate(axes_list):
            eig = eigs[mi]
            raw_vec = np.asarray(vecs[:, mi], dtype=complex)
            vec = raw_vec / scales  # non-dimensionalise

            amps = np.abs(vec)
            dominant = int(np.argmax(amps))
            vec_norm = vec / vec[dominant]  # largest non-dim component → 1∠0°

            amplitudes = np.abs(vec_norm)
            phases = np.angle(vec_norm, deg=True)

            colors = ["tab:blue"] * n_states
            colors[dominant] = "tab:orange"

            ax.barh(
                range(n_states), amplitudes, color=colors, edgecolor="k", linewidth=0.5
            )
            ax.set_yticks(range(n_states))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("Non-dim amplitude", fontsize=8)
            ax.set_xlim(0, 1.35)
            ax.tick_params(labelsize=8)

            for si, (amp, phi) in enumerate(zip(amplitudes, phases)):
                ax.text(
                    amp + 0.02, si, f"{phi:+.0f}°", va="center", fontsize=7, color="k"
                )

            stab_char = "stable" if eig.real < 0 else "UNSTABLE"
            T_str = f"T={1/abs(eig.real):.2f}s" if abs(eig.real) > 1e-10 else "T=∞"
            osc_str = (
                f"  f={abs(eig.imag)/(2*np.pi):.2f}Hz" if abs(eig.imag) > 1e-6 else ""
            )
            ax.set_title(
                f"λ{mi}: {eig.real:+.3f}{eig.imag:+.3f}j\n{T_str}{osc_str}  [{stab_char}]",
                fontsize=8,
            )
            ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()


# ---------------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------------


def _mode_time_response(
    eigenvalue: complex,
    eigenvector: np.ndarray,
    time_vector: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    vec = np.asarray(eigenvector, dtype=complex)
    return np.real(amplitude * np.outer(vec, np.exp(eigenvalue * time_vector)))


def _rot_rad(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    kx, ky, kz = ax
    skew = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=float)
    return (
        np.eye(3) + np.sin(angle_rad) * skew + (1 - np.cos(angle_rad)) * (skew @ skew)
    )


def plot_lateral_mode_animation(
    base_body: BodyAerodynamics,
    trim_result: dict,
    stab: dict,
    *,
    mode_index: int = 0,
    duration_s: float = 15.0,
    fps: int = 30,
    amplitude_rad: float = np.deg2rad(8.0),
    time_scale: float = 400.0,
) -> FuncAnimation:
    """Animate the kite geometry for one lateral eigenmode (roll + yaw components).

    Lateral states are [v, phi, psi, p, r]; animation uses phi (index 1) and psi (index 2).
    """
    eig_lat = np.asarray(stab["eig_lat"], dtype=complex)
    vec_lat = np.asarray(stab["vec_lat"], dtype=complex)

    if mode_index < 0 or mode_index >= eig_lat.size:
        raise ValueError(f"mode_index {mode_index} out of range.")

    eig = eig_lat[mode_index]
    mode_vec = vec_lat[:, mode_index].copy()

    # Normalise by the largest of phi/psi (indices 1, 2) so the animation amplitude is meaningful
    norm_factor = max(np.max(np.abs(mode_vec[1:3])), 1e-12)
    mode_vec = mode_vec / norm_factor

    n_frames = max(int(duration_s * fps), 2)
    t_vis = np.linspace(0.0, duration_s, n_frames)
    t_mod = t_vis / time_scale

    response = _mode_time_response(eig, mode_vec, t_mod, amplitude_rad)
    phi_perturb = response[1, :]  # roll (index 1)
    psi_perturb = response[2, :]  # yaw (index 2)

    trim_roll_rad = np.deg2rad(float(trim_result["opt_x"][1]))
    trim_yaw_rad = np.deg2rad(float(trim_result["opt_x"][3]))

    panel_corners = np.array(
        [panel.corner_points for panel in base_body.panels], dtype=float
    )
    origin = np.asarray(REFERENCE_POINT, dtype=float)

    # ---- figure layout ----
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(121, projection="3d")
    stab_char = "stable" if eig.real < 0 else "UNSTABLE"
    freq = abs(eig.imag) / (2 * np.pi)
    ax3d.set_title(
        f"Lateral mode {mode_index}  [{stab_char}]\n"
        f"λ = {eig.real:+.3f}{eig.imag:+.3f}j  f={freq:.3f} Hz",
        fontsize=9,
    )
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.grid(True, alpha=0.25)

    all_pts = panel_corners.reshape(-1, 3)
    ctr = np.mean(all_pts, axis=0)
    hr = max(0.6 * np.max(np.ptp(all_pts, axis=0)), 1.0)
    ax3d.set_xlim(ctr[0] - hr, ctr[0] + hr)
    ax3d.set_ylim(ctr[1] - hr, ctr[1] + hr)
    ax3d.set_zlim(ctr[2] - hr, ctr[2] + hr)
    ax3d.view_init(elev=20, azim=-120)

    panel_lines = [
        ax3d.plot([], [], [], color="tab:blue", linewidth=1.1)[0]
        for _ in range(panel_corners.shape[0])
    ]
    status_txt = ax3d.text2D(0.02, 0.96, "", transform=ax3d.transAxes, fontsize=8)

    # ---- signal subplots ----
    ax_phi = fig.add_subplot(222)
    ax_phi.set_title("Roll  φ(t)", fontsize=9)
    ax_phi.set_xlabel("t [s]", fontsize=8)
    ax_phi.set_ylabel("φ [deg]", fontsize=8)
    phi_total_deg = np.rad2deg(trim_roll_rad + phi_perturb)
    ax_phi.plot(t_vis, phi_total_deg, color="0.7", linewidth=1)
    (phi_line,) = ax_phi.plot([], [], color="tab:blue", linewidth=2)
    (phi_marker,) = ax_phi.plot([], [], "o", color="tab:blue")
    ax_phi.set_xlim(t_vis[0], t_vis[-1])
    pad = max(0.05 * np.ptp(phi_total_deg), 0.1)
    ax_phi.set_ylim(phi_total_deg.min() - pad, phi_total_deg.max() + pad)
    ax_phi.grid(True, alpha=0.3)

    ax_psi = fig.add_subplot(224)
    ax_psi.set_title("Yaw  ψ(t)", fontsize=9)
    ax_psi.set_xlabel("t [s]", fontsize=8)
    ax_psi.set_ylabel("ψ [deg]", fontsize=8)
    psi_total_deg = np.rad2deg(trim_yaw_rad + psi_perturb)
    ax_psi.plot(t_vis, psi_total_deg, color="0.7", linewidth=1)
    (psi_line,) = ax_psi.plot([], [], color="tab:orange", linewidth=2)
    (psi_marker,) = ax_psi.plot([], [], "o", color="tab:orange")
    ax_psi.set_xlim(t_vis[0], t_vis[-1])
    pad = max(0.05 * np.ptp(psi_total_deg), 0.1)
    ax_psi.set_ylim(psi_total_deg.min() - pad, psi_total_deg.max() + pad)
    ax_psi.grid(True, alpha=0.3)

    def _rotate_pts(pts, R):
        return origin + (pts - origin) @ R.T

    def init():
        for ln in panel_lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        phi_line.set_data([], [])
        phi_marker.set_data([], [])
        psi_line.set_data([], [])
        psi_marker.set_data([], [])
        return [*panel_lines, status_txt, phi_line, phi_marker, psi_line, psi_marker]

    def update(fi: int):
        phi_t = trim_roll_rad + float(phi_perturb[fi])
        psi_t = trim_yaw_rad + float(psi_perturb[fi])
        R = _rot_rad(DEFAULT_AXES.radial, psi_t) @ _rot_rad(DEFAULT_AXES.course, phi_t)
        rot = _rotate_pts(panel_corners, R)
        for idx, ln in enumerate(panel_lines):
            c = np.vstack([rot[idx], rot[idx][0]])
            ln.set_data(c[:, 0], c[:, 1])
            ln.set_3d_properties(c[:, 2])
        status_txt.set_text(
            f"t={t_vis[fi]:.2f} s  |  φ={np.rad2deg(phi_t):+.2f}°  ψ={np.rad2deg(psi_t):+.2f}°"
        )
        phi_line.set_data(t_vis[: fi + 1], phi_total_deg[: fi + 1])
        phi_marker.set_data([t_vis[fi]], [phi_total_deg[fi]])
        psi_line.set_data(t_vis[: fi + 1], psi_total_deg[: fi + 1])
        psi_marker.set_data([t_vis[fi]], [psi_total_deg[fi]])
        return [*panel_lines, status_txt, phi_line, phi_marker, psi_line, psi_marker]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
    )
    fig.tight_layout()
    return anim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    case = {
        "tilt_deg": 0.0,
        "course_deg": 90.0,
        "wind_speed": 8.0,
        "elevation_deg": 35.0,
        "azimuth_deg": 20.0,
        "radial_speed": 1.5,
        "distance_radial": 280.0,
    }
    x_guess = np.array([30.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    print("Solving quasi-steady trim…")
    system_model = build_system_model(case)
    body = build_body(case["tilt_deg"])

    trim_result, _ = solve_quasi_steady_state(
        body_aero=body,
        center_of_gravity=CENTER_OF_GRAVITY,
        reference_point=REFERENCE_POINT,
        system_model=system_model,
        x_guess=x_guess,
        include_gravity=True,
        axes=DEFAULT_AXES,
        moment_tolerance=MOMENT_TOLERANCE,
    )

    print_trim(trim_result)
    print("Gravity force included in trim: ")
    print(
        f"  F_gravity [N]     : {np.array2string(trim_result['gravity_force'], precision=3)}"
    )

    print("\nComputing stability derivatives (16 VSM evaluations)…")
    stab = compute_stability_derivatives(
        body_aero=build_body(case["tilt_deg"]),
        center_of_gravity=CENTER_OF_GRAVITY,
        reference_point=REFERENCE_POINT,
        x_trim=np.asarray(trim_result["opt_x"], dtype=float),
        trim_result=trim_result,
        axes=DEFAULT_AXES,
        mass=MASS,
        Ixx=IXX,
        Iyy=IYY,
        Izz=IZZ,
        distance_radial=case["distance_radial"],
    )

    # Reference values for non-dimensionalisation
    V_ref = float(trim_result["Umag"])  # trim apparent speed [m/s]
    body_ref = build_body(case["tilt_deg"])
    b_ref = float(
        max(
            np.linalg.norm(
                np.asarray(w.sections[-1].LE_point) - np.asarray(w.sections[0].LE_point)
            )
            for w in body_ref.wings
        )
    )  # wingspan estimate [m]
    c_ref = float(max(p.chord for p in body_ref.panels))  # max chord [m]

    print_stability(stab)
    print_mode_decomposition(stab, V_ref, b_ref, c_ref)
    plot_poles(stab)
    plot_mode_shapes(stab, V_ref, b_ref, c_ref)

    _ = plot_lateral_mode_animation(
        base_body=build_body(case["tilt_deg"]),
        trim_result=trim_result,
        stab=stab,
        mode_index=4,
        duration_s=20.0,
        fps=30,
        amplitude_rad=np.deg2rad(2.0),
        time_scale=1000.0,
    )
    plt.show()


if __name__ == "__main__":
    main()

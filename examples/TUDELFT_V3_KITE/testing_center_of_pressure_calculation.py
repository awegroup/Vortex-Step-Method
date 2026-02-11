from pathlib import Path
import numpy as np
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver


def main():
    """Run one TUDELFT_V3_KITE case and report global/panel center of pressure."""

    PROJECT_DIR = Path(__file__).resolve().parents[2]
    n_panels = 50
    spanwise_panel_distribution = "uniform"
    reference_point = np.array([0.0, 0.0, 0.0])
    solver_base_version = Solver(reference_point=reference_point)
    geometry_dir = Path(PROJECT_DIR) / "data" / "2D_airfoils_polars_plots_BEST"

    body_aero = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(geometry_dir / "aero_geometry_CFD_CAD_derived.yaml"),
        spanwise_panel_distribution=spanwise_panel_distribution,
    )
    Umag = 3.15
    side_slip = 0
    yaw_rate = 0

    def run_case(angle_of_attack: float) -> None:
        body_aero.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate)
        results = solver_base_version.solve(body_aero)

        x_cp_global = results["center_of_pressure"]
        panel_cp_locations = np.asarray(results["panel_cp_locations"], dtype=float)
        force_distribution = np.asarray(results["F_distribution"], dtype=float)
        moment_distribution = np.asarray(results["M_distribution"], dtype=float)

        print(f"\n\n=== Case: alpha = {angle_of_attack:.2f} deg ===")
        if x_cp_global is None:
            print(
                "No global center_of_pressure found (force-line did not intersect a panel)."
            )
        else:
            mid_idx = int(
                np.argmin([abs(panel.control_point[1]) for panel in body_aero.panels])
            )
            mid_panel = body_aero.panels[mid_idx]
            mid_le = 0.5 * (mid_panel.LE_point_1 + mid_panel.LE_point_2)
            mid_x_over_c_global = (
                np.dot(np.asarray(x_cp_global, dtype=float) - mid_le, mid_panel.y_airf)
                / mid_panel.chord
            )
            print(
                f"Global center_of_pressure = [{x_cp_global[0]: .4f}, {x_cp_global[1]: .4f}, {x_cp_global[2]: .4f}] "
                f"(x/c wrt mid-span chord = {100.0 * mid_x_over_c_global:.2f}%)"
            )

        print(
            " idx |       y [m] |      chord [m] |    x/c [%] |      |F| [N] | M_pitch_local [Nm]"
        )
        print(f"=== Per-panel CoP (x/c) === My = {results['My']:.6f} Nm")

        for i, panel in enumerate(body_aero.panels):
            cp_location = panel_cp_locations[i]
            le_mid = 0.5 * (panel.LE_point_1 + panel.LE_point_2)
            vec_le_to_cp = cp_location - le_mid
            x_over_c = np.dot(vec_le_to_cp, panel.y_airf) / panel.chord
            force_mag = np.linalg.norm(force_distribution[i])

            # Recover local panel pitching moment (about panel AC, along span axis).
            r_ac = panel.aerodynamic_center - reference_point
            m_local_vec = moment_distribution[i] - np.cross(r_ac, force_distribution[i])
            m_pitch_local = np.dot(m_local_vec, panel.z_airf)

            print(
                f"{i:4d} | {panel.control_point[1]: 11.4f} | {panel.chord: 13.4f} | {100.0 * x_over_c: 9.2f} | {force_mag: 11.4f} | {m_pitch_local: 16.6f}"
            )

        # Strict consistency checks:
        # 1) reconstruct moment from panel cp_i and F_i
        # 2) reconstruct moment from global CP and total force
        m_from_panel_cp = np.zeros(3)
        for i in range(len(body_aero.panels)):
            r_cp = panel_cp_locations[i] - reference_point
            m_from_panel_cp += np.cross(r_cp, force_distribution[i])

        f_total = np.array([results["Fx"], results["Fy"], results["Fz"]], dtype=float)
        if x_cp_global is None:
            m_from_global_cp = np.array([np.nan, np.nan, np.nan], dtype=float)
        else:
            r_global_cp = np.asarray(x_cp_global, dtype=float) - reference_point
            m_from_global_cp = np.cross(r_global_cp, f_total)

        m_total = np.array([results["Mx"], results["My"], results["Mz"]], dtype=float)
        residual_panel = m_total - m_from_panel_cp
        residual_global = m_total - m_from_global_cp

        print("=== Consistency check ===")
        print(
            f"M_total      = [{m_total[0]: .6f}, {m_total[1]: .6f}, {m_total[2]: .6f}] Nm"
        )
        print(
            f"M_from_panel_cp,F = [{m_from_panel_cp[0]: .6f}, {m_from_panel_cp[1]: .6f}, {m_from_panel_cp[2]: .6f}] Nm"
        )
        print(
            f"M_from_global_cp,F = [{m_from_global_cp[0]: .6f}, {m_from_global_cp[1]: .6f}, {m_from_global_cp[2]: .6f}] Nm"
        )
        print(
            f"Residual (M_total - M_from_panel_cp,F) = "
            f"[{residual_panel[0]: .6f}, {residual_panel[1]: .6f}, {residual_panel[2]: .6f}] Nm "
            f"({100.0 * np.linalg.norm(residual_panel) / max(np.linalg.norm(m_total), 1e-12):.3f}% error)"
        )
        print(
            f"Residual (M_total - M_from_global_cp,F) = "
            f"[{residual_global[0]: .6f}, {residual_global[1]: .6f}, {residual_global[2]: .6f}] Nm "
            f"({100.0 * np.linalg.norm(residual_global) / max(np.linalg.norm(m_total), 1e-12):.3f}% error)"
        )

    for alpha in (1, 9, 25, 50.0):
        run_case(alpha)


if __name__ == "__main__":
    main()

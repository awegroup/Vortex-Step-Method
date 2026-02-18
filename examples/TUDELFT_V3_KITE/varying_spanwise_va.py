from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.plot_geometry_plotly import interactive_plot


CASE_DEFINITIONS = {
    "uniform": {
        "vary_magnitude": False,
        "vary_direction_alpha": False,
        "label": "uniform",
    },
    "varying_magnitude": {
        "vary_magnitude": True,
        "vary_direction_alpha": False,
        "label": "varying magnitude",
    },
    "varying_direction": {
        "vary_magnitude": False,
        "vary_direction_alpha": True,
        "label": "varying direction (alpha)",
    },
    "varying_magnitude_direction": {
        "vary_magnitude": True,
        "vary_direction_alpha": True,
        "label": "varying magnitude + direction(alpha)",
    },
}


def instantiate_body_aero(
    project_dir: Path,
    n_panels: int,
    spanwise_panel_distribution: str,
) -> BodyAerodynamics:
    """Create a fresh TUDELFT V3 kite model."""
    cad_derived_geometry_dir = (
        project_dir / "data" / "TUDELFT_V3_KITE" / "CAD_derived_geometry"
    )
    return BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=(cad_derived_geometry_dir / "aero_geometry_CAD_CFD_NF_combined.yaml"),
        spanwise_panel_distribution=spanwise_panel_distribution,
    )


def build_spanwise_va_distribution(
    body_aero: BodyAerodynamics,
    Umag: float,
    angle_of_attack: float,
    side_slip: float,
    speed_variation: float = 0.5,
    direction_variation_alpha_deg: float = 4.0,
    vary_magnitude: bool = True,
    vary_direction_alpha: bool = True,
) -> np.ndarray:
    """
    Build a panelwise apparent-wind field with spanwise variation in magnitude and direction.

    Variation model:
    - speed scale: 1 + speed_variation * (eta - <eta>_area) if vary_magnitude else 1
    - local alpha: alpha + direction_variation_alpha_deg * (eta - <eta>_area)
      if vary_direction_alpha else alpha
    where eta = y / (span/2) in [-1, 1].

    This creates negative speed deltas at negative y and positive deltas at
    positive y, while preserving area-weighted mean speed.
    """
    y_coords = np.asarray(body_aero.compute_y_coordinates(), dtype=float)
    max_abs_y = float(np.max(np.abs(y_coords)))
    if max_abs_y <= 0.0:
        raise ValueError(
            "Invalid spanwise coordinates: all control-point y values are zero."
        )
    eta = y_coords / max_abs_y
    panel_areas = np.array([panel.chord * panel.width for panel in body_aero.panels])
    area_sum = float(np.sum(panel_areas))
    if area_sum <= 0.0:
        raise ValueError("Total panel area must be positive.")
    eta_area_mean = float(np.sum(panel_areas * eta) / area_sum)
    eta_centered = eta - eta_area_mean

    aoa_rad = np.deg2rad(angle_of_attack)
    beta_rad = np.deg2rad(side_slip)
    alpha_variation_rad = np.deg2rad(direction_variation_alpha_deg)

    if vary_magnitude:
        speed_scale = 1.0 + speed_variation * eta_centered
        if np.any(speed_scale <= 0.0):
            raise ValueError(
                "speed_variation too large: at least one panel speed became non-positive."
            )
    else:
        speed_scale = np.ones_like(eta)

    if vary_direction_alpha:
        local_alpha = aoa_rad + alpha_variation_rad * eta_centered
    else:
        local_alpha = np.full_like(eta, aoa_rad)

    va_distribution = np.zeros((body_aero.n_panels, 3), dtype=float)
    for i in range(body_aero.n_panels):
        direction = np.array(
            [
                np.cos(local_alpha[i]) * np.cos(beta_rad),
                np.sin(beta_rad),
                np.sin(local_alpha[i]),
            ],
            dtype=float,
        )
        va_distribution[i] = Umag * speed_scale[i] * direction

    return va_distribution


def plot_spanwise_speed_distribution_matplotlib(
    y_coords: np.ndarray,
    panel_speeds: np.ndarray,
    Umag: float,
    save_folder: Path,
    is_save: bool = True,
    is_show: bool = False,
) -> None:
    """Create and save a static matplotlib plot of panel |va| over spanwise position."""
    sort_idx = np.argsort(y_coords)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        y_coords[sort_idx],
        panel_speeds[sort_idx],
        linestyle="None",
        marker="o",
        markersize=5,
        color="C0",
        label=r"Panel $|V_a|$",
    )
    ax.axhline(
        Umag,
        color="k",
        linestyle="--",
        linewidth=1.2,
        label=f"Target mean speed: {Umag:.3f} m/s",
    )
    span_padding = 0.02 * (np.max(y_coords) - np.min(y_coords))
    ax.set_xlim(np.min(y_coords) - span_padding, np.max(y_coords) + span_padding)
    ax.set_xlabel(r"$y$ [m]")
    ax.set_ylabel(r"$|V_a|$ [m/s]")
    ax.set_title("Spanwise apparent-velocity magnitude (distributed inflow)")
    ax.grid(True)
    ax.legend()

    if is_save:
        save_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_folder / "spanwise_apparent_wind_distribution.pdf",
            bbox_inches="tight",
        )
    if is_show:
        plt.show()
    else:
        plt.close(fig)


def plot_spanwise_speed_distribution_plotly(
    y_coords: np.ndarray,
    panel_speeds: np.ndarray,
    Umag: float,
    save_folder: Path,
    is_save: bool = False,
    is_show: bool = True,
) -> None:
    """Create and save an interactive plotly plot of panel |va| over spanwise position."""
    sort_idx = np.argsort(y_coords)
    y_sorted = y_coords[sort_idx]
    speed_sorted = panel_speeds[sort_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_sorted,
            y=speed_sorted,
            mode="lines+markers",
            name="Panel |Va|",
            marker=dict(size=6),
        )
    )
    fig.add_hline(
        y=Umag,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Target mean speed: {Umag:.3f} m/s",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Spanwise apparent-velocity magnitude (distributed inflow)",
        xaxis_title="y [m]",
        yaxis_title="|Va| [m/s]",
        template="plotly_white",
    )

    if is_save:
        save_folder.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_folder / "spanwise_apparent_wind_distribution.html")
    if is_show:
        fig.show()


def run_sweep(
    project_dir: Path,
    n_panels: int,
    spanwise_panel_distribution: str,
    moment_reference_point: np.ndarray,
    angle_values: np.ndarray,
    sweep_type: str,
    Umag: float,
    fixed_alpha: float,
    fixed_beta: float,
    speed_variation: float,
    direction_variation_alpha_deg: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Run the sweep for all inflow cases and return coefficient arrays by case."""
    if sweep_type not in {"alpha", "beta"}:
        raise ValueError("sweep_type must be either 'alpha' or 'beta'")

    coeff_keys = ("cl", "cd", "cs", "cmx", "cmy", "cmz")
    results_by_case = {
        case: {k: np.zeros(len(angle_values), dtype=float) for k in coeff_keys}
        for case in CASE_DEFINITIONS
    }
    body_by_case = {
        case: instantiate_body_aero(
            project_dir=project_dir,
            n_panels=n_panels,
            spanwise_panel_distribution=spanwise_panel_distribution,
        )
        for case in CASE_DEFINITIONS
    }
    solver_by_case = {
        case: Solver(reference_point=moment_reference_point)
        for case in CASE_DEFINITIONS
    }

    for i, angle in enumerate(angle_values):
        if sweep_type == "alpha":
            alpha_i = float(angle)
            beta_i = fixed_beta
        else:
            alpha_i = fixed_alpha
            beta_i = float(angle)

        for case_name, case_cfg in CASE_DEFINITIONS.items():
            body = body_by_case[case_name]
            solver = solver_by_case[case_name]
            if case_name == "uniform":
                body.va_initialize(
                    Umag=Umag,
                    angle_of_attack=alpha_i,
                    side_slip=beta_i,
                    yaw_rate=0.0,
                )
            else:
                va_distribution = build_spanwise_va_distribution(
                    body_aero=body,
                    Umag=Umag,
                    angle_of_attack=alpha_i,
                    side_slip=beta_i,
                    speed_variation=speed_variation,
                    direction_variation_alpha_deg=direction_variation_alpha_deg,
                    vary_magnitude=case_cfg["vary_magnitude"],
                    vary_direction_alpha=case_cfg["vary_direction_alpha"],
                )
                body.va = va_distribution

            case_results = solver.solve(body)
            for key in coeff_keys:
                results_by_case[case_name][key][i] = case_results[key]

    return results_by_case


def plot_sweep_comparison(
    angle_values: np.ndarray,
    results_by_case: dict[str, dict[str, np.ndarray]],
    x_label: str,
    title: str,
    save_path: Path,
) -> None:
    """Plot all inflow-case comparisons for force and moment coefficients."""
    fig, axs = plt.subplots(2, 3, figsize=(13, 8), dpi=130)
    fields = [
        ("cl", "CL"),
        ("cd", "CD"),
        ("cs", "CS"),
        ("cmx", "CMx"),
        ("cmy", "CMy"),
        ("cmz", "CMz"),
    ]

    for ax, (field, ylabel) in zip(axs.flatten(), fields):
        for case_name, case_cfg in CASE_DEFINITIONS.items():
            ax.plot(
                angle_values,
                results_by_case[case_name][field],
                "o-",
                label=case_cfg["label"],
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Demonstrate spanwise-varying inflow by prescribing a panelwise va_distribution.
    """
    project_dir = Path(__file__).resolve().parents[2]
    save_folder = project_dir / "results" / "TUDELFT_V3_KITE" / "varying_spanwise_va"

    n_panels = 50
    spanwise_panel_distribution = "uniform"
    moment_reference_point = np.array([0.422646, 0.0, 9.3667], dtype=float)

    Umag = 8.0
    angle_of_attack = 12.5
    side_slip = 0.0
    speed_variation = 0.20
    direction_variation_alpha_deg = 4.0

    alpha_range = np.array([-3, 0, 3, 6, 9, 12, 15], dtype=float)
    beta_range = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8], dtype=float)

    body_for_distribution_plot = instantiate_body_aero(
        project_dir=project_dir,
        n_panels=n_panels,
        spanwise_panel_distribution=spanwise_panel_distribution,
    )
    solver_for_distribution_plot = Solver(reference_point=moment_reference_point)
    va_distribution = build_spanwise_va_distribution(
        body_aero=body_for_distribution_plot,
        Umag=Umag,
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        speed_variation=speed_variation,
        direction_variation_alpha_deg=direction_variation_alpha_deg,
        vary_magnitude=True,
        vary_direction_alpha=True,
    )
    body_for_distribution_plot.va = va_distribution
    results_distribution = solver_for_distribution_plot.solve(
        body_for_distribution_plot
    )

    y_coords = np.asarray(
        body_for_distribution_plot.compute_y_coordinates(), dtype=float
    )
    panel_speeds = np.array(
        [np.linalg.norm(panel.va) for panel in body_for_distribution_plot.panels]
    )
    panel_areas = np.array(
        [panel.chord * panel.width for panel in body_for_distribution_plot.panels]
    )
    area_weighted_mean_speed = float(
        np.sum(panel_areas * panel_speeds) / np.sum(panel_areas)
    )
    print(f"Area-weighted mean panel |Va|: {area_weighted_mean_speed:.3f} m/s")
    print(f"Reference Va (for normalization): {results_distribution['va_ref']}")
    print(f"Reference q (for normalization): {results_distribution['q_ref']:.3f} Pa")

    plot_spanwise_speed_distribution_matplotlib(
        y_coords=y_coords,
        panel_speeds=panel_speeds,
        Umag=Umag,
        save_folder=save_folder,
        is_save=True,
        is_show=False,
    )
    plot_spanwise_speed_distribution_plotly(
        y_coords=y_coords,
        panel_speeds=panel_speeds,
        Umag=Umag,
        save_folder=save_folder,
        is_save=False,
        is_show=True,
    )

    # Plotly geometry view, using the same helper as in tutorial.py.
    interactive_plot(
        body_for_distribution_plot,
        vel=Umag,
        angle_of_attack=angle_of_attack,
        side_slip=side_slip,
        yaw_rate=0.0,
        is_with_aerodynamic_details=True,
        title="varying_spanwise_va_geometry",
        is_with_bridles=False,
        is_save=False,
        is_show=True,
    )

    alpha_results_by_case = run_sweep(
        project_dir=project_dir,
        n_panels=n_panels,
        spanwise_panel_distribution=spanwise_panel_distribution,
        moment_reference_point=moment_reference_point,
        angle_values=alpha_range,
        sweep_type="alpha",
        Umag=Umag,
        fixed_alpha=0.0,
        fixed_beta=0.0,
        speed_variation=speed_variation,
        direction_variation_alpha_deg=direction_variation_alpha_deg,
    )
    plot_sweep_comparison(
        angle_values=alpha_range,
        results_by_case=alpha_results_by_case,
        x_label=r"$\alpha$ [deg]",
        title="Alpha sweep: inflow-case comparison",
        save_path=save_folder / "alphasweep_uniform_vs_distributed.pdf",
    )

    beta_results_by_case = run_sweep(
        project_dir=project_dir,
        n_panels=n_panels,
        spanwise_panel_distribution=spanwise_panel_distribution,
        moment_reference_point=moment_reference_point,
        angle_values=beta_range,
        sweep_type="beta",
        Umag=Umag,
        fixed_alpha=angle_of_attack,
        fixed_beta=0.0,
        speed_variation=speed_variation,
        direction_variation_alpha_deg=direction_variation_alpha_deg,
    )
    plot_sweep_comparison(
        angle_values=beta_range,
        results_by_case=beta_results_by_case,
        x_label=r"$\beta$ [deg]",
        title="Beta sweep: inflow-case comparison",
        save_path=save_folder / "betasweep_uniform_vs_distributed.pdf",
    )


if __name__ == "__main__":
    main()

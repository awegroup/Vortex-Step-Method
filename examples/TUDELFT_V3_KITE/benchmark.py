import time
from pathlib import Path
from typing import Dict, List, Tuple

from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver


def instantiate_body(
    config_path: Path,
    n_panels: int,
    spanwise_panel_distribution: str,
    ml_models_dir: Path | None = None,
) -> BodyAerodynamics:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if ml_models_dir is not None and not ml_models_dir.exists():
        raise FileNotFoundError(f"ml_models_dir does not exist: {ml_models_dir}")

    kwargs: Dict[str, Path] = {}
    if ml_models_dir is not None:
        kwargs["ml_models_dir"] = ml_models_dir

    return BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=config_path,
        spanwise_panel_distribution=spanwise_panel_distribution,
        **kwargs,
    )


def run_sweep(
    body_aero: BodyAerodynamics,
    solver: Solver,
    Umag: float,
    alpha_values: List[float],
    side_slip: float,
    yaw_rate: float,
    warm_start: bool,
) -> Tuple[List[float], List[Dict[str, float]]]:
    gamma_prev = None
    timings: List[float] = []
    results: List[Dict[str, float]] = []

    for alpha in alpha_values:
        body_aero.va_initialize(Umag, alpha, side_slip, yaw_rate)
        start_time = time.perf_counter()
        if warm_start:
            solver_results = solver.solve(body_aero, gamma_distribution=gamma_prev)
        else:
            solver_results = solver.solve(body_aero)
        elapsed = time.perf_counter() - start_time
        timings.append(elapsed)
        results.append(
            {
                "alpha": alpha,
                "cl": solver_results.get("cl"),
                "cd": solver_results.get("cd"),
                "cs": solver_results.get("cs"),
            }
        )
        gamma_prev = solver_results.get("gamma_distribution") if warm_start else None

    return timings, results


def summarise_timings(label: str, timings: List[float]) -> str:
    total = sum(timings)
    mean = total / len(timings)
    return (
        f"{label:<28} total {total:6.3f} s | mean {mean * 1000:7.2f} ms | "
        f"min {min(timings) * 1000:7.2f} ms | max {max(timings) * 1000:7.2f} ms"
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data" / "TUDELFT_V3_KITE"
    ml_models_dir = project_dir / "data" / "ml_models"

    configs = [
        (
            "Breukels regression",
            data_dir
            / "CAD_derived_geometry"
            / "aero_geometry_CAD_breukels_regression.yaml",
            None,
        ),
        (
            "Masure ML regression",
            data_dir
            / "CAD_derived_geometry"
            / "aero_geometry_CAD_masure_regression.yaml",
            ml_models_dir,
        ),
    ]

    n_panels = 25
    spanwise_distribution = "uniform"
    Umag = 10.0
    side_slip = 0.0
    yaw_rate = 0.0
    alpha_values = [0.0, 4.0, 8.0, 12.0, 14.0]

    print("TUDELFT_V3_KITE benchmark")
    print(
        f"Settings -> n_panels={n_panels}, Umag={Umag} m/s, "
        f"alpha sweep={alpha_values}, side_slip={side_slip}, yaw_rate={yaw_rate}"
    )

    for label, config_path, model_dir in configs:
        print(f"\n=== {label} ===")

        baseline_body = instantiate_body(
            config_path,
            n_panels,
            spanwise_distribution,
            model_dir,
        )
        baseline_solver = Solver(gamma_initial_distribution_type="elliptical")
        baseline_times, baseline_results = run_sweep(
            baseline_body,
            baseline_solver,
            Umag,
            alpha_values,
            side_slip,
            yaw_rate,
            warm_start=False,
        )

        iterative_body = instantiate_body(
            config_path,
            n_panels,
            spanwise_distribution,
            model_dir,
        )
        iterative_solver = Solver(gamma_initial_distribution_type="previous")
        iterative_times, iterative_results = run_sweep(
            iterative_body,
            iterative_solver,
            Umag,
            alpha_values,
            side_slip,
            yaw_rate,
            warm_start=True,
        )

        print(summarise_timings("Baseline (elliptical)", baseline_times))
        print(summarise_timings("Iterative (previous)", iterative_times))

        total_baseline = sum(baseline_times)
        total_iterative = sum(iterative_times)
        if total_baseline > 0:
            delta = total_baseline - total_iterative
            pct = delta / total_baseline * 100.0
            print(f"Speed-up from warm start: {delta:6.3f} s ({pct:4.1f}%)")

        print("\n alpha [deg] | baseline [ms] | iterative [ms] | Δ [ms]")
        for alpha, base_t, iter_t in zip(alpha_values, baseline_times, iterative_times):
            delta_ms = (base_t - iter_t) * 1000.0
            print(
                f"{alpha:10.1f} | {base_t * 1000:13.2f} | "
                f"{iter_t * 1000:13.2f} | {delta_ms:7.2f}"
            )

        final_iter = iterative_results[-1]
        print(
            f"Final coefficients at α={final_iter['alpha']:.1f}° -> "
            f"CL={final_iter['cl']:.3f}, CD={final_iter['cd']:.3f}, CS={final_iter['cs']:.3f}"
        )


if __name__ == "__main__":
    main()

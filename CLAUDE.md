# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Python implementation of the Vortex Step Method (VSM): an enhanced lifting-line aerodynamic solver coupled with 2D viscous airfoil polars, aimed at low-aspect-ratio wings and leading-edge inflatable (LEI) kites. Scientific software — physical correctness, numerical stability, and reproducibility outrank abstraction and stylistic churn. **Read `AGENTS.md` before changing solver/aerodynamic logic** (sign conventions, reference frames, and circulation definitions must never change silently).

## Commands

```bash
source venv/bin/activate          # project venv at ./venv — required; system python lacks deps
pip install -e .[dev]             # editable install (dev = pytest, pytest-cov, black)

pytest                            # full suite (~35 s), config in pytest.ini (testpaths = tests)
pytest tests/Solver/test_solver.py                 # one file
pytest tests/Solver/test_solver.py::test_name      # one test
black src tests                   # formatting (used by the project)
```

Examples run as plain scripts: `python examples/TUDELFT_V3_KITE/tutorial.py`. CI (`.github/workflows/testing.yml`) runs pytest with coverage on Python 3.10.

## Architecture (big picture)

Data flow: **geometry YAML → `BodyAerodynamics.instantiate()` → `Wing`/`Panel` mesh → `Solver.solve()` → results dict → plotting**.

- `src/VSM/core/BodyAerodynamics.py` — central orchestrator. `BodyAerodynamics.instantiate(n_panels, file_path=<aero_geometry.yaml>, spanwise_panel_distribution=..., bridle_path=...)` reads the YAML's `wing_sections` (headers `[airfoil_id, LE_x..z, TE_x..z]`) and `wing_airfoils` tables, builds polars per airfoil, adds sections to a `Wing`, and refines to `n_panels`. YAML readers index columns **by header name**, so extra columns (e.g. `VUP_x..z` from newer SurfplanAdapter exports) are ignored harmlessly.
- `src/VSM/core/AirfoilAerodynamics.py` — polar generation per airfoil type: `breukels_regression`, `masure_regression` (needs `ml_models_dir`; ~2.3 GB models downloaded from Zenodo into `data/ml_models/`, not in git), `neuralfoil`, `polars` (CSV), `inviscid`. Batch entry point: `from_yaml_entry_batch()`.
- `src/VSM/core/WingGeometry.py` (`Wing`), `Panel.py`, `Filament.py`, `Wake.py` — mesh & vortex elements. **Panels retain only LE/TE points + interpolated polar data**; airfoil `info_dict` parameters (`t`, `chord`, …) are discarded during instantiation — anything needing them later must re-read the YAML.
- `src/VSM/core/Solver.py` — iterative circulation (gamma) loop; optional Anderson acceleration and Li/Gaunaa spanwise artificial viscosity for post-stall stabilization. Returns a results dict (forces, moments, distributions).
- Top-level `src/VSM/` modules build on core: `stability_derivatives.py`, `trim_angle.py`, plus plotting: `plotting.py` (polars/distributions), `plot_geometry_matplotlib.py` (`plot_geometry`), `plot_geometry_plotly.py` (`interactive_plot` — the primary interactive 3D wing view, incl. bridle rendering), `plot_styling.py`.
- Performance-critical vector math is `numba`-jitted (`core/utils.py`).

## Data layout

`data/<KITE_NAME>/` holds per-kite inputs, e.g. `data/TUDELFT_V3_KITE/`:
- `CAD_derived_geometry/` and `Surfplan_derived_geometry/` — `aero_geometry_*.yaml` variants (one per airfoil-model type) + `struc_geometry_*.yaml` (bridle/particle data, used via `bridle_path`).
- `Surfplan_export/` — raw Surfplan export (`<name>.txt` + `profiles/*.dat`). Converted to VSM YAMLs by the sibling [SurfplanAdapter](https://github.com/awegroup/SurfplanAdapter) project, which also handles the Surfplan→VSM coordinate transform.
- Different YAML sources use different global frames (CAD-derived vs Surfplan-derived origins differ by ~10 m in z) — never mix coordinates across YAML families.

## Conventions that matter

- Body-fixed reference frame: **x rearward (LE→TE), y right wing, z up**; α positive nose-up, β positive wind from port. Aircraft-frame conversion via `stability_derivatives.map_derivatives_to_aircraft_frame()`.
- Wing sections are ordered/symmetric tip-to-tip; `airfoil_id` in `wing_sections` must match `wing_airfoils`.
- Style (`docs/style_guide.md`): typed Google docstrings, `snake_case`, path variables end `_path`, directories end `_dir`.
- Tests include verification cases against analytical solutions (`tests/verification_cases/`: elliptical, swept, curved wing, horseshoe vortices) — solver changes must keep these passing; add regression tests when numerical behaviour intentionally changes and explain what changed and why in the commit.

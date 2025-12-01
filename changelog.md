# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [2.0.1] - 21-10-2025

### Added
- `evaluate_stability_derivatives.py` example script demonstrating the use of the new
  `compute_rigid_body_stability_derivatives()` function for calculating aerodynamic stability derivatives, prints results at trim conditions in the kite reference frame (x-backward LE to TE, y-right from kite's perspective, z-up) and in the aircraft convention (x-forward, y-right, z-down).

### Fixed
- Corrected a misplaced minus sign in the sideslip handling inside `BodyAerodynamics.va_initialize`, restoring the intended sign convention for moment predictions and β-related stability derivatives.

## [2.0.0] - 08-10-2025

### ⚠️ Breaking Changes

#### API Changes in `BodyAerodynamics`

**1. `va.setter` now requires keyword-only arguments:**

The velocity setter has been completely redesigned to properly handle body angular rates and reference points. All arguments after `va` are now **keyword-only**.

**Old usage (v1.1.0):**
```python
# Setting velocity with yaw rate (tuple format)
body_aero.va = (vel_app, yaw_rate)

# Or without yaw rate
body_aero.va = vel_app
```

**New usage (v2.0.0+):**
```python
# Basic velocity setting (no body rates)
body_aero.va = vel_app

# With body rates (keyword arguments required)
body_aero.va = vel_app, roll_rate=p, pitch_rate=q, yaw_rate=r

# With custom reference point for rotational velocity
body_aero.va = vel_app, yaw_rate=r, reference_point=np.array([x, y, z])
```

**Migration guide:**
- Instead of setting it directly using `body_aero.va`, you can instead use `va_initialize()` (see below)

- If you do want to directly set it, you could should write for full control over body rates:
  ```python
  body_aero.va = vel_app, roll_rate=p, pitch_rate=q, yaw_rate=r, reference_point=ref_pt
  ```

**2. `va_initialize()` signature expanded:**

The initialization method now accepts body angular rates and reference point.

**Old signature (v1.1.0):**
```python
body_aero.va_initialize(Umag, angle_of_attack, side_slip, yaw_rate=0.0)
```

**New signature (v2.0.0+):**
```python
body_aero.va_initialize(
    Umag, 
    angle_of_attack, 
    side_slip, 
    pitch_rate=0.0,      # NEW
    roll_rate=0.0,       # NEW  
    reference_point=None # NEW
)
```

**Migration guide:**
- The `yaw_rate` parameter has been **removed**
- Body rates are now specified via `pitch_rate`, `roll_rate` keywords
- If you need yaw rate, use `pitch_rate` parameter (body-fixed z-axis)
- Most existing code will work without changes if you weren't using `yaw_rate`
- If you were using `yaw_rate`, replace with appropriate body rate:
  ```python
  # Old
  body_aero.va_initialize(Umag, alpha, beta, yaw_rate=0.5)
  
  # New (yaw is rotation about body z-axis = pitch_rate)
  body_aero.va_initialize(Umag, alpha, beta, pitch_rate=0.5)
  ```

**3. Rotational velocity calculation changed:**

The method for computing rotational velocity contributions has been fundamentally updated:

- **Old behavior:** Simple uniform yaw rate applied
- **New behavior:** Proper rotational velocity field: `v_rot = omega × (r - r_ref)`
  - Accounts for all three body angular rates (roll, pitch, yaw)
  - Reference point can be specified (defaults to panel centroids)
  - Physically accurate velocity field due to body rotation

**Impact:**
- Results involving body angular rates will differ from v1.1.0
- New results are physically more accurate
- If comparing with v1.1.0 results, expect differences in cases with non-zero angular rates

**4. New property: `_body_rates`**

Body angular rates are now stored and accessible via `_body_rates` property:
```python
p, q, r = body_aero._body_rates  # [roll_rate, pitch_rate, yaw_rate]
```

### Added

#### New Modules

**1. `stability_derivatives.py`**
- Compute rigid-body aerodynamic stability derivatives
- Function: `compute_rigid_body_stability_derivatives()`
- Supports derivatives w.r.t. angle of attack (α), sideslip (β), and body rates (p, q, r)
- Optional non-dimensionalization of rate derivatives
- Uses central finite differences for accuracy
- See `docs/StabilityDerivatives.md` for detailed documentation

**2. `trim_angle.py`**
- Find trim angle of attack where pitching moment equals zero
- Function: `compute_trim_angle()`
- Two-phase algorithm: coarse sweep + bisection refinement
- Automatic stability verification
- Configurable convergence tolerances
- See `docs/TrimAngle.md` for detailed documentation

#### New Documentation
- `docs/StabilityDerivatives.md` - Comprehensive guide to stability derivative computation
- `docs/TrimAngle.md` - Guide to trim angle finding with examples
- `docs/README.md` - Updated documentation index with quick-start guide
- Enhanced main `README.md` with Quick Start, Key Features, and Troubleshooting sections

#### New Examples
- `examples/TUDELFT_V3_KITE/tow_angle_geometry.py` - Visualize effect of tow angle on kite geometry
- `examples/TUDELFT_V3_KITE/tow_point_location_parametric_study.py` - Study tow point location effects
- `examples/TUDELFT_V3_KITE/kite_stability_dynamics.py` - Demonstrate stability derivative computation

### Fixed
- Projected area calculation now uses correct trapezoidal integration method
- Reference point handling in rotational velocity calculations


## [1.1.0] - 2025-09-26

### ⚠️ Breaking Changes
- **Class rename:** `WingAerodynamics` → **`BodyAerodynamics`**.
- **Dataset path rename:** `TUDELFT_V3_LEI_KITE` → **`TUDELFT_V3_KITE`**.
- **Polar input format update:**
  - 3D: `alpha, CL, CD, CM`
  - 2D: `alpha, Cl, Cd, Cm` (note the capitalization difference)
- **Defaults & options:**
  - Default spanwise panel distribution → **`uniform`**
  - Removed options/attributes: **`using_previous gamma`**, **`min_relaxation_error`**, **`is_new_vector_definition`**.
- **Stall model refactor:** Simonet-related stall logic moved out of `Solver` into **`solver_functions`** and a dedicated *stall* branch.

### Added
- **AirfoilAerodynamics framework**
  - New base class **`AirfoilAerodynamics`** (parent for all airfoil types).
  - **`masure_regression`** airfoil with on-disk **caching** for fast reuse.
  - **YAML** configuration input for airfoil definitions.
  - New **tests** for `AirfoilAerodynamics`.

- **LEI parametric model (masure_regression_lei_parametric)**
  - Examples showing how to reconstruct a **LEI airfoil** using 6 parameters:
    `t, eta, kappa, delta, lambda, phi`.
  - Utilities for generating **airfoil geometry** and exporting **.csv / .dat**.

- **Half-wing inputs**
  - Support for *half-wing* inputs in both **polar generation** and **geometry CSV** export.

- **Plotting & tutorials**
  - Improved **convergence plotting** utilities.
  - New **sensitivity analysis** examples & script (`sensitivity_analysis.py`).
  - Updated **tutorials** (usage, sensitivity, convergence workflows).
  - Interactive **Plotly** visualization environment.
  - **Moment** calculations + plotting (incl. user-defined reference point).
  - **Panel polar investigation** function.

- **Solver & VSM pipeline**
  - `Solver` gains **artificial viscosity** (property + setter).
  - Added **non-linear gamma loop** variants and **gamma loop type** selector.
  - New `compute_aerodynamic_quantities` helper.
  - New initial **gamma distribution** options: `elliptical`, `cosine`, `zero`.
  - `BodyAerodynamics`:
    - Added/renamed circulation helpers:
      - `calculate_circulation_distribution_elliptical_wing`
      - `calculate_circulation_distribution_cosine`
  - **Panel**:
    - New property: `panel_polar_data`.
    - New `_panel_aero_model` option: `"cl_is_pisinalpha"`.

- **Validation & examples**
  - Added **elliptical wing planform** test (Simonet ch. 4.1).
  - Added **rectangular wing** test.
  - **NeuralFoil** integration for cross-checking polars.
  - Git now **ignores** `results/`.

- **Bridle line forces**
  - Support for inputs like:
    - `bridle_line1 = [p1, p2, diameter]` where `p1 = [x, y, z]`
    - `bridle_lines = [bridle_line1, bridle_line2, ...]`
  - `BodyAerodynamics` API sugar:
    - `instantiate_body_aerodynamics` function
    - `BodyAerodynamics.from_file(...)`

### Changed
- **Polars & plotting**
  - `plotting.py::generate_3D_polar_data` now passes a **gamma distribution**
    into the solver:  
    `results = solver.solve(body_aero, gamma_distrbution=gamma)`
  - Legends for **distributions** and **polars** are displayed **below** plots.
  - Improved clarity and formatting of plot outputs.

- **Solver internals**
  - Revised `__init__`, added artificial viscosity, refactored internal state to `self.*` for broader accessibility.
  - Revised **gamma initialization** and **feedback** handling.
  - Adjusted **Simonet** model plumbing and naming (`elliptic` → `elliptical`).

- **BodyAerodynamics**
  - `calculate_panel_properties`: replaced hardcoded `0.25 / 0.75` with **`ac`** (aerodynamic center) and **`cp`** (center of pressure).

- **Docs & tutorials**
  - Updated **docstrings**, **tutorials**, and **user guides** to reflect the new APIs and behaviors.
  - Expanded convergence & sensitivity study guidance (e.g., `n_panels`).

- **Vector conventions**
  - Normalized aerodynamic vector handling for consistency across modules.

- **Polar engineering**
  - Updated to use **Surfplan** profiles; adjusted `POLAR` behavior and tests.

### Fixed
- Polar data **input robustness** and format handling.
- **Reference point** handling in moment computations.
- Multiple failing tests in **Panel**.
- Plot output issues (including prior formatting issues, e.g. #85).
- Numerous **docstring** and minor consistency fixes.

### Removed
- `min_relaxation_error` and `is_new_vector_definition` fields.
- `using_previous gamma` option (gamma initialization is now explicit/controlled).
- Consolidated/moved **stall models** into dedicated modules/branch.
- Extracted **polar_engineering** work to its own branch.

### Migration Notes
- **Imports / classes**
  - Replace:
    ```python
    from VSM.core import WingAerodynamics
    ```
    with:
    ```python
    from VSM.core import BodyAerodynamics
    ```

- **Datasets & configs**
  - Update any paths from:
    ```
    TUDELFT_V3_LEI_KITE → TUDELFT_V3_KITE
    ```

- **Polar files & arrays**
  - Ensure 3D data is in: `alpha, CL, CD, CM`
  - Ensure 2D data is in: `alpha, Cl, Cd, Cm`
  - If you relied on the old format, update your CSV writers/loaders and tests accordingly.

- **Solver usage**
  - Specify the desired **gamma initialization** and **loop type** explicitly:
    ```python
    solver.gamma_initial_distribution_type = "elliptical"  # or "cosine", "zero"
    solver.gamma_loop_type = "non_linear"                  # or "base"
    solver.artificial_viscosity =  ...                     # set as needed
    results = solver.solve(body_aero, gamma_distrbution="cosine")
    ```
  - Remove any reliance on `using_previous gamma`.

- **Panel model**
  - If you used `_panel_aero_model`, you can now set:
    ```python
    panel._panel_aero_model = "cl_is_pisinalpha"
    ```

- **Plotting**
  - Legends for certain plots are now at the **bottom**; if you have custom layout logic, adjust figure margins accordingly.

- **Half-wing workflows**
  - When generating polars or exporting geometry, adjust scripts to provide **half-wing** inputs if desired.

- **Caching (masure_regression)**
  - The regression model caches results on disk; verify your `ml_models_dir` and cache paths are writable in CI.


## [1.0.0] - 2024-09-29

### Added 
- Refactored code to object oriented. 
  - Introduced the main VSM (Vortex Step Method) parent class.
  - Added a dedicated Panel Properties class.
  - Set up an initial Horseshoe class and later introduced child classes.
  - Defined global functions supporting each of the classes.
  - Added a Filaments class and a Wake class.
  - Introduced an Abstract class for 2D infinite filaments.
- **Solver & Geometry Enhancements:**
  - Re-structured the code skeleton and overhauled the solver with all necessary methods.
  - Added new functions to support curved wing, swept wing, and elliptical wing geometries.
  - Implemented a calculate_results function based on earlier thesis code and refined it.
- **Visualization and Plotting:**
  - Added functions for plotting wing geometry, distributions, and polar plots.
  - Updated examples (e.g., rectangular wing case, V3 example folder with CFD/WindTunne interfaces).
- **Testing & Documentation:**
  - Integrated initial pytests across modules (Panel, Filaments, etc.), coverage 66%
  - Populated configuration files like `citation.cff`, `.gitignore`, and `pyproject.toml` for improved packaging.
  - Added user instructions, docstrings for each function, and updated tutorial content.
- **Performance Optimizations:**
  - Introduced significant speed improvements by refactoring loops (e.g., removing nested loops and moving functions outside loops) and optimizing AIC matrix computations (#50).
- **Additional Functionalities:**
  - Added a yaw rate implementation with an accompanying V3 kite example.
  - Implemented polar data interpolation for sections with different datasets.

### Changed
- **Code Refactoring:**
  - Renamed and refactored class names and variable identifiers across VSM, Panel, and WingAerodynamics.
  - Removed redundant properties from WingAerodynamics, now accessing values directly from the Panel.
- **Solver & Algorithm Improvements:**
  - Updated the global functions and solver structure to improve convergence and reliability.
  - Refined the handling of gamma distributions, reference system definitions, and spacing (including cosine van Garrel spacing for elliptical wings, #24).
  - Revised the induced_velocity function for better accuracy (#21).
- **Documentation & User Interface:**
  - Enhanced the README with additional images, detailed tutorial updates, and more user-friendly instructions.
  - Improved code comments and added TODO markers to guide future development.
- **Robustness & Configuration:**
  - Made the code robust against unordered or right-to-left section inputs.
  - Removed hardcoded values and replaced them with parameterized configurations.

### Fixed
- **Bug Resolutions:**
  - Fixed various issues in solver functionality, geometry processing, and test cases.
  - Corrected the induced_velocity function and core-radius handling in pytests (#21).
  - Adjusted the narrow angle-of-attack (aoa) range in aerodynamic data inputs (#42).
  - Resolved alignment issues in V3 comparisons and output formatting.
  - Fixed bugs in the stall model (ensuring the stall angle is calculated per plate) and revised related test scripts (#29, #68).
  - Addressed wake modeling issues and enhanced logging for error tracking.
- **Testing and Stability:**
  - Resolved discrepancies in AIC matrix calculations and convergence tests.
  - Fixed issues with PDF/PNG outputs in test scripts (#61).
- **Performance Corrections:**
  - Eliminated redundant calculations and optimized nested loops to achieve faster runtime.
  - Resolved pytesting issues related to numba and streamlined optimization routines.

### Removed
- **Deprecated and Redundant Code:**
  - Removed the older Horseshoe class in favor of the new implementation.
  - Eliminated unnecessary attributes from WingAerodynamics and cleaned out redundant code sections.
  - Removed legacy code and virtual environment folders from the project directory.
  - Cleared outdated TODOs and extraneous comments to streamline the codebase.

## [1.0.0] - 2024-09-29

### Added 
- Refactored code to object oriented. 
  - Introduced the main VSM (Vortex Step Method) parent class.
  - Added a dedicated Panel Properties class.
  - Set up an initial Horseshoe class and later introduced child classes.
  - Defined global functions supporting each of the classes.
  - Added a Filaments class and a Wake class.
  - Introduced an Abstract class for 2D infinite filaments.
- **Solver & Geometry Enhancements:**
  - Re-structured the code skeleton and overhauled the solver with all necessary methods.
  - Added new functions to support curved wing, swept wing, and elliptical wing geometries.
  - Implemented a calculate_results function based on earlier thesis code and refined it.
- **Visualization and Plotting:**
  - Added functions for plotting wing geometry, distributions, and polar plots.
  - Updated examples (e.g., rectangular wing case, V3 example folder with CFD/WindTunne interfaces).
- **Testing & Documentation:**
  - Integrated initial pytests across modules (Panel, Filaments, etc.), coverage 66%
  - Populated configuration files like `citation.cff`, `.gitignore`, and `pyproject.toml` for improved packaging.
  - Added user instructions, docstrings for each function, and updated tutorial content.
- **Performance Optimizations:**
  - Introduced significant speed improvements by refactoring loops (e.g., removing nested loops and moving functions outside loops) and optimizing AIC matrix computations (#50).
- **Additional Functionalities:**
  - Added a yaw rate implementation with an accompanying V3 kite example.
  - Implemented polar data interpolation for sections with different datasets.

### Changed
- **Code Refactoring:**
  - Renamed and refactored class names and variable identifiers across VSM, Panel, and WingAerodynamics.
  - Removed redundant properties from WingAerodynamics, now accessing values directly from the Panel.
- **Solver & Algorithm Improvements:**
  - Updated the global functions and solver structure to improve convergence and reliability.
  - Refined the handling of gamma distributions, reference system definitions, and spacing (including cosine van Garrel spacing for elliptical wings, #24).
  - Revised the induced_velocity function for better accuracy (#21).
- **Documentation & User Interface:**
  - Enhanced the README with additional images, detailed tutorial updates, and more user-friendly instructions.
  - Improved code comments and added TODO markers to guide future development.
- **Robustness & Configuration:**
  - Made the code robust against unordered or right-to-left section inputs.
  - Removed hardcoded values and replaced them with parameterized configurations.

### Fixed
- **Bug Resolutions:**
  - Fixed various issues in solver functionality, geometry processing, and test cases.
  - Corrected the induced_velocity function and core-radius handling in pytests (#21).
  - Adjusted the narrow angle-of-attack (aoa) range in aerodynamic data inputs (#42).
  - Resolved alignment issues in V3 comparisons and output formatting.
  - Fixed bugs in the stall model (ensuring the stall angle is calculated per plate) and revised related test scripts (#29, #68).
  - Addressed wake modeling issues and enhanced logging for error tracking.
- **Testing and Stability:**
  - Resolved discrepancies in AIC matrix calculations and convergence tests.
  - Fixed issues with PDF/PNG outputs in test scripts (#61).
- **Performance Corrections:**
  - Eliminated redundant calculations and optimized nested loops to achieve faster runtime.
  - Resolved pytesting issues related to numba and streamlined optimization routines.

### Removed
- **Deprecated and Redundant Code:**
  - Removed the older Horseshoe class in favor of the new implementation.
  - Eliminated unnecessary attributes from WingAerodynamics and cleaned out redundant code sections.
  - Removed legacy code and virtual environment folders from the project directory.
  - Cleared outdated TODOs and extraneous comments to streamline the codebase.

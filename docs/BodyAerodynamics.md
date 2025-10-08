# Vortex Step Method (VSM) Documentation

## Overview

The Vortex Step Method (VSM) is an aerodynamic analysis framework that combines vortex-based flow modeling with sectional airfoil data to compute aerodynamic forces and moments on wing geometries. The implementation supports both VSM and Lifting Line Theory (LLT) approaches.

## Core Components

### 1. WingGeometry (`WingGeometry.py`)

The `Wing` class defines wing geometry composed of multiple sections and provides methods for mesh refinement and geometric analysis.

#### Key Methods:

- **`add_section(LE_point, TE_point, polar_data)`**: Adds a wing section with leading/trailing edge points and airfoil polar data
- **`refine_aerodynamic_mesh()`**: Refines the wing mesh according to the specified spanwise panel distribution
- **`find_farthest_point_and_sort(sections)`**: Sorts sections based on proximity for proper mesh ordering
- **`compute_new_polar_data(polar_data, section_index, left_weight, right_weight)`**: Interpolates polar data between adjacent sections

#### Panel Distribution Types:
- `"uniform"`: Linear spacing along the wing
- `"cosine"`: Cosine spacing for better tip resolution  
- `"split_provided"`: Split existing sections to achieve desired panel count
- `"unchanged"`: Keep provided sections as-is

#### Properties:
- **`span`**: Calculates wing span along the specified spanwise direction
- **`compute_projected_area(z_plane_vector)`**: Computes projected area onto a specified plane

### 2. Panel (`Panel.py`)

The `Panel` class represents individual wing panels and computes local aerodynamic properties.

#### Key Methods:

- **`compute_relative_alpha_and_relative_velocity(induced_velocity)`**: Calculates effective angle of attack and relative velocity including induced effects
- **`compute_cl(alpha)`**: Computes lift coefficient from airfoil polar data at given angle of attack
- **`compute_cd_cm(alpha)`**: Computes drag and moment coefficients from polar data
- **`compute_velocity_induced_bound_2D(control_point)`**: Calculates 2D bound vortex induced velocity
- **`compute_velocity_induced_single_ring_semiinfinite(control_point, evaluation_point_on_bound, va_norm, va_unit, gamma, core_radius_fraction)`**: Computes 3D induced velocity from vortex ring with semi-infinite wake

#### Panel Properties:
- **Geometric**: `chord`, `width`, `corner_points`, `control_point`, `aerodynamic_center`
- **Reference Frame**: `x_airf` (normal), `y_airf` (chordwise), `z_airf` (spanwise)
- **Aerodynamic**: Local polar data interpolated from section data

### 3. Filament (`Filament.py`)

Implements vortex filament velocity calculations with core radius corrections.

#### BoundFilament Methods:

- **`velocity_3D_bound_vortex(XVP, gamma, core_radius_fraction)`**: Calculates velocity induced by bound vortex with Vatistas core model
- **`velocity_3D_trailing_vortex(XVP, gamma, Uinf)`**: Calculates velocity from trailing vortex with viscous core correction

#### Core Radius Models:
- **Bound Vortex**: `epsilon = core_radius_fraction * filament_length`
- **Trailing Vortex**: `epsilon = sqrt(4 * alpha0 * nu * r_perp / Uinf)` (viscous diffusion model)

### 4. AirfoilAerodynamics (`AirfoilAerodynamics.py`)

Factory class for generating airfoil polar data from various sources.

#### Supported Sources:

- **`"breukels_regression"`**: LEI kite airfoil correlation model (Breukels 2011)
- **`"neuralfoil"`**: Neural network-based airfoil analysis  
- **`"polars"`**: Direct CSV polar data import
- **`"inviscid"`**: Thin airfoil theory (2π slope, zero drag)
- **`"masure_regression"`**: Machine learning predictions using trained models

#### Usage:
```python
aero = AirfoilAerodynamics.from_yaml_entry(
    "breukels_regression", 
    {"t": 0.12, "kappa": 0.08}, 
    alpha_range=[-10, 20, 1]
)

# For masure_regression, ml_models_dir is required
aero = AirfoilAerodynamics.from_yaml_entry(
    "masure_regression",
    {"t": 0.07, "eta": 0.2, "kappa": 0.95, "delta": -2, "lambda": 0.65, "phi": 0.25},
    alpha_range=[-10, 20, 1],
    reynolds=1e6,
    ml_models_dir="/path/to/ml_models"
)

polar_data = aero.to_polar_array()  # Returns [alpha, cl, cd, cm] array
```

### 5. BodyAerodynamics (`BodyAerodynamics.py`)

Main class that orchestrates the aerodynamic analysis by combining wing geometry, panel generation, and flow calculations.

#### Initialization:

- **`__init__(wings, bridle_line_system=None)`**: Creates BodyAerodynamics from Wing objects and optional bridle lines
- **`instantiate(n_panels, file_path=None, wing_instance=None, spanwise_panel_distribution="uniform", is_with_bridles=False, ml_models_dir=None)`**: Factory method to create from YAML config or Wing instance

#### Factory Method Parameters:

- **`n_panels`** (int): Number of panels (required if wing_instance is not provided)
- **`file_path`** (str, optional): Path to the YAML config file. If None, wing_instance must be provided
- **`wing_instance`** (Wing, optional): Pre-built Wing instance. If None, file_path must be provided
- **`spanwise_panel_distribution`** (str): Panel distribution type (default: 'uniform')
- **`is_with_bridles`** (bool): Whether to include bridle lines (default: False)
- **`ml_models_dir`** (str, optional): Path to ML model files directory (**required if any airfoil uses masure_regression**)

**Important**: When using `masure_regression` airfoil type in your YAML configuration, you must provide the `ml_models_dir` parameter pointing to the directory containing the trained model files (ET_re5e6.pkl, ET_re1e6.pkl, ET_re2e7.pkl).

#### Key Analysis Methods:

- **`va_initialize(Umag, angle_of_attack, side_slip=0.0, yaw_rate=0.0, pitch_rate=0.0, roll_rate=0.0)`**: Sets flight conditions and optional body rotation rates
- **`compute_AIC_matrices(aerodynamic_model_type, core_radius_fraction, va_norm_array, va_unit_array)`**: Builds Aerodynamic Influence Coefficient matrices
- **`compute_results(gamma_new, rho, aerodynamic_model_type, ...)`**: Computes forces, moments, and coefficients from circulation distribution

#### Flow Velocity Methods:
- **`update_effective_angle_of_attack_if_VSM(...)`**: Updates effective angle of attack using induced velocities at aerodynamic centers (VSM only)
- **`compute_circulation_distribution_elliptical_wing(gamma_0)`**: Analytical elliptical circulation distribution
- **`compute_circulation_distribution_cosine(gamma_0)`**: Cosine-based circulation distribution

#### Advanced Features:
- **`viscous_drag_correction(...)`**: 3D viscous drag correction following Gaunaa et al. (2024)
- **`compute_line_aerodynamic_force(va, line)`**: Aerodynamic forces on bridle lines
- **`find_center_of_pressure(force_array, moment_array, reference_point)`**: Locates center of pressure intersection with wing surface

### 6. Solver (`Solver.py`)

Iterative solver that determines circulation distribution satisfying boundary conditions.

#### Solver Parameters:
- **`aerodynamic_model_type`**: "VSM" or "LLT"
- **`core_radius_fraction`**: Vortex core radius (typically 1e-6 to 1e-2)
- **`max_iterations`**: Maximum solver iterations (default: 5000)
- **`allowed_error`**: Convergence tolerance (default: 1e-6)
- **`relaxation_factor`**: Under-relaxation factor (default: 0.01)

#### Solution Process:
1. **Initialize**: Set up AIC matrices and boundary conditions
2. **Iterate**: Solve for circulation using iterative scheme with relaxation
3. **Converge**: Check residual against tolerance
4. **Results**: Compute forces, moments, and distributions

## Solver Output Dictionary

The `.solve` method returns a comprehensive dictionary with the following structure:

### Global Wing Aerodynamics:
- **Force Components**: `"Fx"`, `"Fy"`, `"Fz"` - Total aerodynamic forces in global coordinates
- **Moment Components**: `"Mx"`, `"My"`, `"Mz"` - Total moments about reference point
- **Aerodynamic Forces**: `"lift"`, `"drag"`, `"side"` - Forces in wind-axis coordinates
- **Force Coefficients**: `"cl"`, `"cd"`, `"cs"` - Non-dimensional force coefficients
- **Moment Coefficients**: `"cmx"`, `"cmy"`, `"cmz"` - Non-dimensional moment coefficients

### Local Panel Distributions:
- **Coefficient Distributions**: `"cl_distribution"`, `"cd_distribution"`, `"cs_distribution"` - Spanwise coefficient variations
- **Force Distributions**: `"F_distribution"`, `"M_distribution"` - 3D force and moment vectors per panel
- **Global Force Components**: `"cfx_distribution"`, `"cfy_distribution"`, `"cfz_distribution"` - Force coefficient distributions in global coordinates
- **Global Moment Components**: `"cmx_distribution"`, `"cmy_distribution"`, `"cmz_distribution"` - Moment coefficient distributions

### Flow and Geometry Information:
- **Angle of Attack Data**: 
  - `"alpha_at_ac"`: Effective angle of attack at aerodynamic centers (includes induced effects)
  - `"alpha_uncorrected"`: Geometric angle of attack at control points
  - `"alpha_geometric"`: Wing geometric angle relative to horizontal
- **Circulation**: `"gamma_distribution"` - Bound circulation strength per panel
- **Geometric Properties**:
  - `"area_all_panels"`: Sum of all panel areas
  - `"projected_area"`: Wing projected area on reference plane
  - `"wing_span"`: Total wing span
  - `"aspect_ratio_projected"`: Projected aspect ratio (span²/area)
- **Flow Conditions**: `"Rey"` - Reynolds number based on maximum chord

### Pressure and Loading:
- **Center of Pressure**: `"center_of_pressure"` - Global location where resultant force acts
- **Panel Pressure Centers**: `"panel_cp_locations"` - Center of pressure for each panel

## VSM vs LLT Comparison

| Aspect | VSM | LLT |
|--------|-----|-----|
| **Evaluation Points** | Control points (3/4 chord) | Aerodynamic centers (1/4 chord) |
| **Angle of Attack** | Corrected at AC using induced velocities | Geometric angle at control points |
| **Bound Vortex** | Excluded from self-influence | Full influence included |
| **Wake Model** | Semi-infinite straight wake | Semi-infinite straight wake |
| **Accuracy** | Higher for thick/cambered airfoils | Good for thin airfoils |
| **Computational Cost** | Slightly higher (double evaluation) | Lower |

## Usage Example

```python
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver

# Create from YAML configuration (with masure_regression airfoils)
body_aero = BodyAerodynamics.instantiate(
    n_panels=8,
    file_path="config_kite.yaml",
    spanwise_panel_distribution="cosine",
    ml_models_dir="/path/to/ml_models"  # Required for masure_regression
)

# Set flight conditions  
body_aero.va_initialize(
    Umag=15.0,           # m/s
    angle_of_attack=8.0, # degrees
    side_slip=2.0,       # degrees  
    yaw_rate=0.1,        # rad/s
    pitch_rate=0.0,      # rad/s
    roll_rate=0.05       # rad/s
)

# Solve aerodynamics
solver = Solver(
    aerodynamic_model_type="VSM",
    core_radius_fraction=1e-4,
    max_iterations=2000,
    allowed_error=1e-5
)

results = solver.solve(body_aero)

# Access results
print(f"Lift coefficient: {results['cl']:.3f}")
print(f"Drag coefficient: {results['cd']:.3f}")
print(f"Center of pressure: {results['center_of_pressure']}")
```

## YAML Configuration Format

```yaml
wing_sections:
  headers: [airfoil_id, LE_x, LE_y, LE_z, TE_x, TE_y, TE_z]
  data:
    - [tip, 0.0, -4.0, 0.0, 0.8, -4.0, 0.0]
    - [mid, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  
    - [tip, 0.0, 4.0, 0.0, 0.8, 4.0, 0.0]

wing_airfoils:
  alpha_range: [-10, 25, 1]  # [min, max, step] in degrees
  reynolds: 500000
  headers: [airfoil_id, type, info_dict]
  data:
    - [tip, breukels_regression, {t: 0.10, kappa: 0.06}]
    - [mid, breukels_regression, {t: 0.15, kappa: 0.10}]
    # For masure_regression, ensure ml_models_dir is provided to instantiate()
    - [root, masure_regression, {t: 0.07, eta: 0.2, kappa: 0.95, delta: -2, lambda: 0.65, phi: 0.25}]

# Optional bridle system
bridle_nodes:
  headers: [id, x, y, z, type]
  data:
    - [n1, 0.5, 0.0, -2.0, knot]
    - [n2, 0.5, 0.0, -5.0, knot]

bridle_lines:
  headers: [name, rest_length, diameter, material, rho]
  data:
    - [main_line, 3.0, 0.003, dyneema, 970]

bridle_connections:
  headers: [name, ci, cj, ck]  
  data:
    - [main_line, n1, n2, null]
```

**Note**: When using `masure_regression` airfoil type in your YAML configuration, you must provide the `ml_models_dir` parameter to the `BodyAerodynamics.instantiate()` method. This directory must contain the required model files:
- `ET_re5e6.pkl` (Reynolds 5×10⁶)
- `ET_re1e6.pkl` (Reynolds 1×10⁶)  
- `ET_re2e7.pkl` (Reynolds 2×10⁷)

This framework provides a complete aerodynamic analysis capability for complex wing geometries with support for various airfoil models, panel distributions, and advanced features like bridle line modeling and viscous corrections.

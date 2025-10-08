# Stability Derivatives Module Documentation

## Overview

The `stability_derivatives` module provides utilities for computing rigid-body aerodynamic stability derivatives. These derivatives quantify how aerodynamic forces and moments change with respect to kinematic angles (angle of attack, sideslip) and body rotation rates (roll, pitch, yaw).

## Module Location

```python
from VSM.stability_derivatives import compute_rigid_body_stability_derivatives
```

## Primary Function

### `compute_rigid_body_stability_derivatives()`

Computes aerodynamic derivatives using finite-difference perturbations of the flow state.

```python
derivatives = compute_rigid_body_stability_derivatives(
    body_aero,
    solver,
    angle_of_attack,
    side_slip,
    velocity_magnitude,
    roll_rate=0.0,
    pitch_rate=0.0,
    yaw_rate=0.0,
    step_sizes=None,
    reference_point=None,
    nondimensionalize_rates=True
)
```

## Parameters

### Required Parameters

- **`body_aero`** (BodyAerodynamics): Aerodynamic model instance that will be updated in-place
- **`solver`** (Solver): Solver configured for the analysis
- **`angle_of_attack`** (float): Baseline angle of attack in degrees
- **`side_slip`** (float): Baseline sideslip angle in degrees (positive for starboard-to-port flow)
- **`velocity_magnitude`** (float): Magnitude of the freestream velocity (m/s)

### Optional Parameters

- **`roll_rate`** (float): Baseline body roll rate `p` in rad/s (default: 0.0)
- **`pitch_rate`** (float): Baseline body pitch rate `q` in rad/s (default: 0.0)
- **`yaw_rate`** (float): Baseline body yaw rate `r` in rad/s (default: 0.0)
- **`step_sizes`** (dict): Optional overrides for perturbation steps. Supported keys: `{"alpha", "beta", "p", "q", "r"}`
  - Angle steps are in degrees (internally converted to radians for the derivative)
  - Rate steps are in rad/s
- **`reference_point`** (np.ndarray): Reference point for moment calculation `[x, y, z]`. If None, defaults to `solver.reference_point` if available, otherwise `[0, 0, 0]`
- **`nondimensionalize_rates`** (bool): If True (default), rate derivatives are non-dimensionalized (see below)

## Return Value

Returns a dictionary with stability derivative keys:

### Force Coefficient Derivatives

**Angle derivatives (per radian):**
- `"dCx_dalpha"`, `"dCy_dalpha"`, `"dCz_dalpha"` - Force sensitivities to angle of attack
- `"dCx_dbeta"`, `"dCy_dbeta"`, `"dCz_dbeta"` - Force sensitivities to sideslip

**Rate derivatives:**
- `"dCx_dp"`, `"dCy_dp"`, `"dCz_dp"` - Force sensitivities to roll rate
- `"dCx_dq"`, `"dCy_dq"`, `"dCz_dq"` - Force sensitivities to pitch rate
- `"dCx_dr"`, `"dCy_dr"`, `"dCz_dr"` - Force sensitivities to yaw rate

### Moment Coefficient Derivatives

**Angle derivatives (per radian):**
- `"dCMx_dalpha"`, `"dCMy_dalpha"`, `"dCMz_dalpha"` - Moment sensitivities to angle of attack
- `"dCMx_dbeta"`, `"dCMy_dbeta"`, `"dCMz_dbeta"` - Moment sensitivities to sideslip

**Rate derivatives:**
- `"dCMx_dp"`, `"dCMy_dp"`, `"dCMz_dp"` - Moment sensitivities to roll rate (roll damping)
- `"dCMx_dq"`, `"dCMy_dq"`, `"dCMz_dq"` - Moment sensitivities to pitch rate (pitch damping)
- `"dCMx_dr"`, `"dCMy_dr"`, `"dCMz_dr"` - Moment sensitivities to yaw rate (yaw damping)

## Non-dimensionalization of Rate Derivatives

When `nondimensionalize_rates=True` (default), rate derivatives are converted to dimensionless form:

```
hat_p = p * b / (2*V)         [roll rate]
hat_q = q * c_MAC / (2*V)     [pitch rate]
hat_r = r * b / (2*V)         [yaw rate]
```

Where:
- `b` is wingspan
- `c_MAC` is mean aerodynamic chord
- `V` is velocity magnitude

This converts derivatives from **per rad/s** to **per hat-rate** (dimensionless), which is the standard form in aerodynamics textbooks.

If `nondimensionalize_rates=False`, rate derivatives remain dimensional (per rad/s).

## Finite Difference Method

The function uses **central finite differences** for all derivatives:

```
dC/dx = [C(x + dx) - C(x - dx)] / (2 * dx)
```

### Default Step Sizes

- **Angle perturbations**: 
  - `alpha`: 0.5° (converted to radians for derivative)
  - `beta`: 0.5° (converted to radians for derivative)
  
- **Rate perturbations**:
  - `p`: 0.1 rad/s
  - `q`: 0.1 rad/s
  - `r`: 0.1 rad/s

These can be overridden using the `step_sizes` parameter.

## Reference Point Considerations

The `reference_point` parameter is critical for physically correct rotational velocity calculations. The rotational velocity at any point `r` is computed as:

```
v_rot(r) = omega × (r - r_ref)
```

Where:
- `omega` is the body rate vector `[p, q, r]`
- `r_ref` is the reference point

For kites and aircraft, this is typically the center of mass or aerodynamic center.

## Usage Example

### Basic Usage

```python
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.stability_derivatives import compute_rigid_body_stability_derivatives

# Setup aerodynamic model
body_aero = BodyAerodynamics.instantiate(
    n_panels=30,
    file_path="config_kite.yaml",
    spanwise_panel_distribution="cosine"
)

# Setup solver
solver = Solver(
    aerodynamic_model_type="VSM",
    core_radius_fraction=1e-4,
    reference_point=[0.5, 0.0, 0.0]  # Reference point at [x, y, z]
)

# Compute stability derivatives at trim condition
derivatives = compute_rigid_body_stability_derivatives(
    body_aero=body_aero,
    solver=solver,
    angle_of_attack=8.0,        # degrees
    side_slip=0.0,              # degrees
    velocity_magnitude=15.0,    # m/s
    nondimensionalize_rates=True
)

# Access results
print(f"Pitch stiffness dCMy/dα: {derivatives['dCMy_dalpha']:.4f} per rad")
print(f"Roll damping dCMx/dp̂: {derivatives['dCMx_dp']:.4f} per hat-p")
print(f"Pitch damping dCMy/dq̂: {derivatives['dCMy_dq']:.4f} per hat-q")
```

### Custom Step Sizes

```python
# Use custom perturbation step sizes
custom_steps = {
    "alpha": 1.0,    # degrees
    "beta": 1.0,     # degrees
    "p": 0.05,       # rad/s
    "q": 0.05,       # rad/s
    "r": 0.05        # rad/s
}

derivatives = compute_rigid_body_stability_derivatives(
    body_aero=body_aero,
    solver=solver,
    angle_of_attack=8.0,
    side_slip=0.0,
    velocity_magnitude=15.0,
    step_sizes=custom_steps
)
```

### With Body Rotation Rates

```python
# Compute derivatives at non-zero rotation rates
derivatives = compute_rigid_body_stability_derivatives(
    body_aero=body_aero,
    solver=solver,
    angle_of_attack=10.0,
    side_slip=2.0,
    velocity_magnitude=20.0,
    roll_rate=0.1,      # rad/s
    pitch_rate=0.05,    # rad/s
    yaw_rate=0.02,      # rad/s
    reference_point=[0.5, 0.0, -0.2]
)
```

## Physical Interpretation

### Longitudinal Stability

- **`dCMy_dalpha < 0`**: Pitch stiffness (restoring moment) - indicates static longitudinal stability
- **`dCMy_dq < 0`**: Pitch damping - opposes pitching motion

### Lateral-Directional Stability

- **`dCMz_dbeta > 0`**: Yaw stiffness (weather vane stability)
- **`dCMx_dp < 0`**: Roll damping - opposes rolling motion
- **`dCMx_dr > 0`**: Yaw-to-roll coupling

### Lift and Drag

- **`dCz_dalpha`**: Lift curve slope (typically positive and large)
- **`dCx_dalpha`**: Drag increase with angle of attack

## Performance Considerations

The function calls the solver **10 times** (2 evaluations for each of 5 parameters: α, β, p, q, r) using central differences. For typical configurations:

- **Computation time**: ~1-10 seconds depending on mesh resolution
- **Memory**: Minimal (only stores coefficient values)
- **Accuracy**: Second-order accurate (central differences)

## Related Functions

- **`compute_trim_angle()`**: Find angles where CMy = 0 (see TrimAngle.md)
- **`solver.solve()`**: Core aerodynamic solver (see Solver.md)
- **`body_aero.va_initialize()`**: Set flow conditions (see BodyAerodynamics.md)

## References

1. Etkin, B., & Reid, L. D. (1996). *Dynamics of Flight: Stability and Control*. Wiley.
2. Nelson, R. C. (1998). *Flight Stability and Automatic Control*. McGraw-Hill.
3. McCormick, B. W. (1995). *Aerodynamics, Aeronautics, and Flight Mechanics*. Wiley.

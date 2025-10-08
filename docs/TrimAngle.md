# Trim Angle Module Documentation

## Overview

The `trim_angle` module provides utilities for computing trim angles where the pitching moment coefficient crosses zero, and for verifying dynamic pitch stability by checking the sign of the pitch stiffness derivative.

## Module Location

```python
from VSM.trim_angle import compute_trim_angle
```

## Primary Function

### `compute_trim_angle()`

Finds trim angles (where CMy = 0) and verifies pitch stability (dCMy/dα < 0).

```python
trim_solutions = compute_trim_angle(
    body_aero,
    solver,
    side_slip=0.0,
    velocity_magnitude=10.0,
    roll_rate=0.0,
    pitch_rate=0.0,
    yaw_rate=0.0,
    alpha_min=-5.0,
    alpha_max=15.0,
    coarse_step=2.0,
    fine_tolerance=1e-3,
    derivative_step=0.1,
    max_bisection_iter=40,
    reference_point=None
)
```

## Algorithm

The function uses a two-phase approach:

1. **Coarse Sweep**: Sweeps angle of attack from `alpha_min` to `alpha_max` to detect sign changes in CMy
2. **Bisection Refinement**: Each sign change is refined using bisection until |CMy| < tolerance
3. **Stability Check**: Evaluates dCMy/dα at each solution to verify pitch stability

## Parameters

### Required Parameters

- **`body_aero`** (BodyAerodynamics): Instantiated aerodynamic model that will be updated in-place
- **`solver`** (Solver): Solver instance configured with the desired reference point

### Flight Condition Parameters

- **`side_slip`** (float): Sideslip angle in degrees (default: 0.0)
- **`velocity_magnitude`** (float): Freestream velocity magnitude in m/s (default: 10.0)
- **`roll_rate`** (float): Body roll rate in rad/s (default: 0.0)
- **`pitch_rate`** (float): Body pitch rate in rad/s (default: 0.0)
- **`yaw_rate`** (float): Body yaw rate in rad/s (default: 0.0)

### Search Parameters

- **`alpha_min`** (float): Minimum angle of attack for coarse sweep in degrees (default: -5.0)
- **`alpha_max`** (float): Maximum angle of attack for coarse sweep in degrees (default: 15.0)
- **`coarse_step`** (float): Step size for coarse sweep in degrees (default: 2.0)
- **`fine_tolerance`** (float): Angular tolerance for bisection refinement in degrees (default: 1e-3)
- **`derivative_step`** (float): Perturbation in degrees for evaluating dCMy/dα (default: 0.1)
- **`max_bisection_iter`** (int): Maximum iterations per bisection bracket (default: 40)

### Reference Point

- **`reference_point`** (np.ndarray): Reference point for moment calculation [x, y, z]. If None, defaults to `solver.reference_point`

## Return Value

Returns a **list of dictionaries**, each containing:

- **`"alpha_trim"`** (float): Trim angle of attack in degrees
- **`"dCMy_dalpha"`** (float): Pitch stiffness derivative at trim (per radian)
- **`"stable"`** (bool): True if dCMy/dα < 0 (statically stable), False otherwise
- **`"CMy"`** (float): Pitching moment coefficient at trim (should be ≈0)

### Special Cases

- **No sign change found**: Returns the angle closest to CMy = 0 from the coarse sweep with `stable=False`
- **Multiple solutions**: Returns all trim angles found (can occur for non-linear airfoils with complex moment curves)
- **Empty list**: No solutions found (should not normally occur)

## Physical Interpretation

### Trim Condition

A trim angle satisfies:
```
CMy(α_trim) = 0
```

This means the pitching moment about the reference point is zero - the aircraft/kite is in rotational equilibrium.

### Pitch Stability

For static longitudinal stability, we require:
```
dCMy/dα < 0  at α_trim
```

This means:
- If α increases (nose up), CMy becomes negative (nose-down restoring moment)
- If α decreases (nose down), CMy becomes positive (nose-up restoring moment)

This provides a restoring moment that opposes pitch disturbances.

### Dynamic Stability

The pitch stiffness `k = -dCMy/dα` can be used to estimate the natural frequency of longitudinal oscillations:

```
ω_n = √(k / I_y)
```

Where `I_y` is the pitch moment of inertia.

## Usage Examples

### Basic Trim Finding

```python
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver
from VSM.trim_angle import compute_trim_angle

# Setup aerodynamic model
body_aero = BodyAerodynamics.instantiate(
    n_panels=30,
    file_path="config_kite.yaml",
    spanwise_panel_distribution="cosine"
)

# Setup solver with reference point at center of mass
solver = Solver(
    aerodynamic_model_type="VSM",
    core_radius_fraction=1e-4,
    reference_point=[0.5, 0.0, 0.0]
)

# Find trim angles
trim_solutions = compute_trim_angle(
    body_aero=body_aero,
    solver=solver,
    velocity_magnitude=15.0,
    alpha_min=-5.0,
    alpha_max=20.0
)

# Display results
for i, solution in enumerate(trim_solutions):
    print(f"\nTrim Solution {i+1}:")
    print(f"  Trim angle: {solution['alpha_trim']:.3f}°")
    print(f"  dCMy/dα: {solution['dCMy_dalpha']:.4f} per rad")
    print(f"  Stable: {solution['stable']}")
    print(f"  CMy: {solution['CMy']:.6f}")
    
    if solution['stable']:
        # Estimate natural frequency (example)
        k = -solution['dCMy_dalpha']  # Pitch stiffness
        I_y = 100.0  # kg⋅m² (example moment of inertia)
        omega_n = np.sqrt(k / I_y)
        period = 2 * np.pi / omega_n
        print(f"  Natural frequency: {omega_n:.3f} rad/s")
        print(f"  Period: {period:.2f} s")
```

### With Sideslip

```python
# Find trim with non-zero sideslip
trim_solutions = compute_trim_angle(
    body_aero=body_aero,
    solver=solver,
    side_slip=5.0,           # 5° sideslip
    velocity_magnitude=20.0,
    alpha_min=-5.0,
    alpha_max=20.0
)
```

### With Body Rotation Rates

```python
# Find trim during turning maneuver
trim_solutions = compute_trim_angle(
    body_aero=body_aero,
    solver=solver,
    velocity_magnitude=15.0,
    yaw_rate=0.1,            # rad/s
    roll_rate=0.05,          # rad/s
    alpha_min=0.0,
    alpha_max=25.0
)
```

### Fine-Tuning Search Parameters

```python
# High-accuracy search with fine steps
trim_solutions = compute_trim_angle(
    body_aero=body_aero,
    solver=solver,
    velocity_magnitude=15.0,
    alpha_min=-10.0,
    alpha_max=30.0,
    coarse_step=1.0,         # Finer coarse step
    fine_tolerance=1e-4,     # Tighter tolerance (0.0001°)
    derivative_step=0.05,    # Smaller perturbation for derivative
    max_bisection_iter=50
)
```

## Practical Application Example

### Stability Analysis Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from VSM.trim_angle import compute_trim_angle
from VSM.stability_derivatives import compute_rigid_body_stability_derivatives

# 1. Find all trim points
trim_solutions = compute_trim_angle(
    body_aero, solver,
    velocity_magnitude=15.0,
    alpha_min=-5.0,
    alpha_max=25.0
)

# 2. Analyze stability at each trim point
for solution in trim_solutions:
    if solution['stable']:
        alpha_trim = solution['alpha_trim']
        
        # Compute full stability derivatives at trim
        derivatives = compute_rigid_body_stability_derivatives(
            body_aero=body_aero,
            solver=solver,
            angle_of_attack=alpha_trim,
            side_slip=0.0,
            velocity_magnitude=15.0
        )
        
        # Extract key stability parameters
        CL_alpha = derivatives['dCz_dalpha']  # Lift curve slope
        CM_alpha = derivatives['dCMy_dalpha']  # Pitch stiffness
        CM_q = derivatives['dCMy_dq']          # Pitch damping
        
        print(f"\nStability at α_trim = {alpha_trim:.2f}°:")
        print(f"  CL_α = {CL_alpha:.4f} per rad")
        print(f"  CM_α = {CM_alpha:.4f} per rad")
        print(f"  CM_q̂ = {CM_q:.4f} per hat-q")
```

### Moment Curve Plotting

```python
# Generate CMy vs α curve to visualize stability
alphas = np.linspace(-5, 25, 61)
CMy_values = []

for alpha in alphas:
    body_aero.va_initialize(
        Umag=15.0,
        angle_of_attack=alpha,
        side_slip=0.0
    )
    results = solver.solve(body_aero)
    CMy_values.append(results['cmy'])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(alphas, CMy_values, 'b-', linewidth=2, label='CMy')
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)

# Mark trim points
for solution in trim_solutions:
    marker = 'go' if solution['stable'] else 'ro'
    label = 'Stable trim' if solution['stable'] else 'Unstable trim'
    plt.plot(solution['alpha_trim'], solution['CMy'], marker, 
             markersize=10, label=label)

plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('Pitching Moment Coefficient, CMy')
plt.title('Pitching Moment Curve')
plt.legend()
plt.show()
```

## Error Handling

The function validates inputs and raises errors for:

```python
# ValueError cases
if derivative_step <= 0.0:
    raise ValueError("derivative_step must be positive")
    
if coarse_step <= 0.0:
    raise ValueError("coarse_step must be positive")
    
if alpha_max <= alpha_min:
    raise ValueError("alpha_max must be greater than alpha_min")
```

## Performance Considerations

### Computational Cost

For a typical search:
- Coarse sweep evaluations: `(alpha_max - alpha_min) / coarse_step`
- Bisection evaluations per bracket: ~20-30 (logarithmic convergence)
- Derivative evaluations: 2 per solution (forward difference)

**Example**: With default parameters (-5° to 15°, step 2°):
- Coarse: 11 evaluations
- Bisection: ~25 evaluations per trim point
- Total: ~40-70 solver calls for 1-2 trim points

### Optimization Tips

1. **Use coarser steps** for initial exploration
2. **Narrow the search range** if you know approximate trim angle
3. **Reduce `fine_tolerance`** only when necessary
4. **Cache results** if analyzing multiple velocities

## Related Functions

- **`compute_rigid_body_stability_derivatives()`**: Compute full stability derivative matrix (see StabilityDerivatives.md)
- **`solver.solve()`**: Core aerodynamic solver (see Solver.md)
- **`body_aero.va_initialize()`**: Set flow conditions (see BodyAerodynamics.md)

## References

1. Etkin, B., & Reid, L. D. (1996). *Dynamics of Flight: Stability and Control*. Wiley.
2. Roskam, J. (1995). *Airplane Flight Dynamics and Automatic Flight Controls*. DARcorporation.
3. Stevens, B. L., & Lewis, F. L. (2003). *Aircraft Control and Simulation*. Wiley.

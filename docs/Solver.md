# Solver Module Documentation

## Overview

The `Solver` class implements iterative algorithms to determine the circulation distribution over a wing that satisfies the boundary conditions of the aerodynamic model. It supports both VSM (Vortex Step Method) and LLT (Lifting Line Theory) approaches with various convergence strategies and stall modeling capabilities.

## Class: Solver

### Constructor

```python
Solver(
    aerodynamic_model_type="VSM",
    max_iterations=5000,
    allowed_error=1e-6,
    relaxation_factor=0.01,
    core_radius_fraction=1e-20,
    gamma_loop_type="base",
    gamma_initial_distribution_type="elliptical",
    is_only_f_and_gamma_output=False,
    is_with_viscous_drag_correction=False,
    reference_point=[0, 0, 0],
    mu=1.81e-5,
    rho=1.225,
    # Stall modeling parameters...
)
```

## Core Parameters

### Aerodynamic Model Configuration
- **`aerodynamic_model_type`** (str): "VSM" or "LLT" (default: "VSM")
- **`core_radius_fraction`** (float): Vortex core radius fraction (default: 1e-20)

### Convergence Control
- **`max_iterations`** (int): Maximum solver iterations (default: 5000)
- **`allowed_error`** (float): Convergence tolerance (default: 1e-6) 
- **`relaxation_factor`** (float): Under-relaxation factor (default: 0.01)

### Initial Conditions
- **`gamma_initial_distribution_type`** (str): Initial circulation distribution
  - `"elliptical"`: Elliptical wing theoretical distribution
  - `"cosine"`: Cosine-based distribution  
  - `"zero"`: Zero initial circulation
  - `"previous"`: Use provided distribution

### Solution Methods
- **`gamma_loop_type`** (str): Iterative algorithm type
  - `"base"`: Standard fixed-point iteration with relaxation
  - `"non_linear"`: Robust nonlinear solvers (Broyden methods)
  - `"simonet_stall"`: Stall modeling with Simonet approach
  - `"non_linear_simonet_stall"`: Combined nonlinear + stall modeling

## Primary Method: solve()

### `solve(body_aero, gamma_distribution=None)`

Main solution method that computes circulation distribution and aerodynamic forces.

**Parameters:**
- `body_aero` (BodyAerodynamics): Configured aerodynamic model
- `gamma_distribution` (np.ndarray): Initial circulation guess (optional)

**Returns:**
- `dict`: Comprehensive results dictionary with forces, moments, and distributions

### Solution Process

#### 1. Initialization
```python
# Extract panel properties
for i, panel in enumerate(body_aero.panels):
    x_airf_array[i] = panel.x_airf
    y_airf_array[i] = panel.y_airf
    va_array[i] = panel.va
    chord_array[i] = panel.chord
    # ...
```

#### 2. AIC Matrix Computation
```python
AIC_x, AIC_y, AIC_z = body_aero.compute_AIC_matrices(
    aerodynamic_model_type,
    core_radius_fraction, 
    va_norm_array,
    va_unit_array
)
```

#### 3. Initial Circulation Distribution
```python
if gamma_initial_distribution_type == "elliptical":
    gamma_initial = body_aero.compute_circulation_distribution_elliptical_wing()
elif gamma_initial_distribution_type == "cosine":
    gamma_initial = body_aero.compute_circulation_distribution_cosine()
# ...
```

#### 4. Iterative Solution
```python
converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(gamma_initial)

# Retry with reduced relaxation if not converged
if not converged:
    converged, gamma_new, alpha_array, Umag_array = self.gamma_loop(
        gamma_initial, extra_relaxation_factor=0.5
    )
```

#### 5. Results Computation
```python
results = body_aero.compute_results(
    gamma_new, rho, aerodynamic_model_type, core_radius_fraction,
    mu, alpha_array, Umag_array, chord_array, x_airf_array, y_airf_array,
    z_airf_array, va_array, va_norm_array, va_unit_array, panels,
    is_only_f_and_gamma_output, is_with_viscous_drag_correction, reference_point
)
```

## Iterative Solution Methods

### `gamma_loop(gamma_initial, extra_relaxation_factor=1.0)`

Standard fixed-point iteration with under-relaxation.

**Algorithm:**
```python
for iteration in range(max_iterations):
    # 1. Compute aerodynamic quantities from current gamma
    alpha_array, Umag_array, cl_array, Umagw_array = compute_aerodynamic_quantities(gamma)
    
    # 2. Update circulation using Kutta-Joukowski theorem
    gamma_new = 0.5 * ((Umag_array²) / Umagw_array) * cl_array * chord_array
    
    # 3. Apply under-relaxation
    gamma_new = (1 - relaxation_factor) * gamma + relaxation_factor * gamma_new
    
    # 4. Check convergence
    normalized_error = max(|gamma_new - gamma|) / max(|gamma_new|)
    if normalized_error < allowed_error:
        converged = True
        break
```

**Convergence Features:**
- Normalized error computation
- Oscillation detection and damping
- Adaptive relaxation for stability

### `gamma_loop_non_linear(gamma_initial)`

Robust nonlinear solver using SciPy optimization methods.

**Formulation:**
Solves F(γ) = γ - γ_new(γ) = 0 where γ_new(γ) is computed from:
```python
def compute_gamma_residual(gamma):
    _, Umag_array, cl_array, Umagw_array = compute_aerodynamic_quantities(gamma)
    gamma_new = 0.5 * ((Umag_array²) / Umagw_array) * cl_array * chord_array
    return gamma - gamma_new  # Residual
```

**Methods Attempted:**
1. **Broyden1**: Quasi-Newton method with rank-1 updates
2. **Broyden2**: Quasi-Newton method with rank-2 updates  
3. **Fallback**: Standard gamma_loop if nonlinear methods fail

**Advantages:**
- Superior convergence for difficult cases
- Automatic step size adaptation
- Robust handling of stiff problems

## Aerodynamic Quantity Computation

### `compute_aerodynamic_quantities(gamma)`

Computes flow variables from circulation distribution.

**Process:**
```python
# 1. Induced velocities from AIC matrices
induced_velocity = [AIC_x @ gamma, AIC_y @ gamma, AIC_z @ gamma].T

# 2. Relative velocity (apparent + induced)
relative_velocity = va_array + induced_velocity

# 3. Local angle of attack
v_normal = sum(x_airf_array * relative_velocity, axis=1)
v_tangential = sum(y_airf_array * relative_velocity, axis=1)  
alpha_array = arctan(v_normal / v_tangential)

# 4. Effective velocity magnitude
relative_velocity_crossz = cross(relative_velocity, z_airf_array)
Umag_array = norm(relative_velocity_crossz, axis=1)

# 5. Lift coefficients from polar data
cl_array = [panel.compute_cl(alpha) for panel, alpha in zip(panels, alpha_array)]
```

**Returns:**
- `alpha_array`: Effective angles of attack
- `Umag_array`: Effective velocity magnitudes  
- `cl_array`: Lift coefficients
- `Umagw_array`: Reference velocity magnitudes

## Advanced Features

### Stall Modeling

#### Smooth Circulation
- **`is_smooth_circulation`**: Apply smoothing to circulation distribution
- **`smoothness_factor`**: Smoothing strength parameter

#### Artificial Damping  
- **`is_artificial_damping`**: Enable artificial damping for stall
- **`artificial_damping`**: Damping coefficients {"k2": 0.1, "k4": 0.0}

#### Simonet Artificial Viscosity
- **`is_with_simonet_artificial_viscosity`**: Simonet stall model
- **`simonet_artificial_viscosity_fva`**: Model parameter

### Viscous Drag Correction

**`is_with_viscous_drag_correction`**: Enable 3D viscous effects following Gaunaa et al. (2024)

### Output Options

- **`is_only_f_and_gamma_output`**: Return only forces and circulation (fast mode)
- **`reference_point`**: Moment reference point for results

## Error Handling and Robustness

### Convergence Monitoring
```python
# Normalized error tracking
reference_error = max(abs(gamma_new)) if max(abs(gamma_new)) != 0 else 1e-4
normalized_error = max(abs(gamma_new - gamma)) / reference_error

# Oscillation detection
if error_history[-1] > error_history[-2] and error_history[-2] < error_history[-3]:
    # Apply additional damping
    gamma_new = 0.75 * gamma_new + 0.25 * gamma
```

### Adaptive Strategies
- Automatic relaxation factor reduction
- Method switching (linear → nonlinear)
- Graceful degradation for difficult cases

### Validation Checks
- Physical bounds on circulation values
- Angle of attack range validation
- Velocity magnitude sanity checks

## Performance Optimization

### Computational Efficiency
- JIT-compiled vector operations
- Efficient AIC matrix operations
- Minimal memory allocation in loops

### Memory Management
- Pre-allocated arrays for panel properties
- In-place updates where possible
- Efficient sparse matrix operations

### Scalability
- O(N²) AIC matrix computation (once per solution)
- O(N) per iteration for gamma updates
- Suitable for wings with 100+ panels

## Integration Examples

### Basic Usage
```python
# Create solver
solver = Solver(
    aerodynamic_model_type="VSM",
    max_iterations=2000,
    allowed_error=1e-5,
    relaxation_factor=0.02
)

# Solve aerodynamics
results = solver.solve(body_aero)

# Access results
cl = results['cl']
cd = results['cd'] 
gamma_dist = results['gamma_distribution']
```

### Advanced Configuration
```python
# High-accuracy nonlinear solver
solver = Solver(
    aerodynamic_model_type="VSM",
    gamma_loop_type="non_linear",
    allowed_error=1e-8,
    is_with_viscous_drag_correction=True,
    reference_point=[0.5, 0.0, 0.0]
)

# Stall modeling
stall_solver = Solver(
    gamma_loop_type="simonet_stall",
    is_smooth_circulation=True,
    smoothness_factor=0.1,
    is_artificial_damping=True,
    artificial_damping={"k2": 0.15, "k4": 0.05}
)
```

### Parameter Studies
```python
# Convergence study
solvers = [
    Solver(allowed_error=1e-4),
    Solver(allowed_error=1e-5),  
    Solver(allowed_error=1e-6)
]

results = [solver.solve(body_aero) for solver in solvers]
```

## Troubleshooting

### Common Convergence Issues
1. **Oscillating solutions**: Reduce relaxation_factor
2. **Slow convergence**: Try nonlinear solver
3. **Divergence**: Check flow conditions and geometry

### Performance Issues  
1. **Slow iterations**: Reduce max_iterations for initial studies
2. **Memory usage**: Use is_only_f_and_gamma_output=True
3. **Accuracy vs speed**: Balance allowed_error vs computation time

### Physical Validity
1. **Negative lift**: Check angle of attack and airfoil data
2. **Excessive circulation**: Verify geometry and flow conditions
3. **Stall behavior**: Enable appropriate stall modeling

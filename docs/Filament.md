# Filament Module Documentation

## Overview

The `Filament` module implements vortex filament velocity calculations with viscous core corrections. It provides the fundamental building blocks for computing induced velocities in the Vortex Step Method, supporting both bound and trailing vortex filaments.

## Abstract Base Class: Filament

### Properties

**Physical Constants:**
- `_alpha0 = 1.25643`: Oseen parameter for viscous diffusion
- `_nu = 1.48e-5`: Kinematic viscosity of air (m²/s)

### Constructor

```python
def __init__(self):
    self._alpha0 = 1.25643  # Oseen parameter
    self._nu = 1.48e-5      # Kinematic viscosity of air
```

## Class: BoundFilament

Represents a finite bound vortex filament between two points, typically along the quarter-chord line of a wing panel.

### Constructor

```python
BoundFilament(x1, x2)
```

**Parameters:**
- `x1` (np.ndarray): First endpoint of the filament
- `x2` (np.ndarray): Second endpoint of the filament

**Computed Properties:**
- `_length`: Filament length ||x2 - x1||
- `_r0`: Filament vector (x2 - x1)

### Properties

#### `x1`, `x2`
Returns the filament endpoints as numpy arrays.

### Methods

#### `velocity_3D_bound_vortex(XVP, gamma, core_radius_fraction)`

Calculates velocity induced by a bound vortex filament using the Vatistas core model.

**Parameters:**
- `XVP` (np.ndarray): Evaluation point coordinates
- `gamma` (float): Vortex strength (circulation)
- `core_radius_fraction` (float): Core radius as fraction of filament length

**Returns:**
- `np.ndarray`: Induced velocity vector [vx, vy, vz]

**Core Radius Model:**
```
ε = core_radius_fraction × ||filament_length||
```

**Velocity Calculation:**
- **Outside core** (r > ε): Standard Biot-Savart law
- **On filament** (r = 0): Zero velocity
- **Inside core** (r < ε): Vatistas regularization with linear ramp

**Mathematical Formulation:**
```
v_ind = (γ/4π) × (r1×r2)/(||r1×r2||²) × r0·(r1/||r1|| - r2/||r2||)
```

Where:
- r0 = filament vector
- r1, r2 = vectors from filament endpoints to evaluation point

#### `velocity_3D_trailing_vortex(XVP, gamma, Uinf)`

Calculates velocity induced by a trailing vortex filament with viscous core correction.

**Parameters:**
- `XVP` (np.ndarray): Evaluation point coordinates  
- `gamma` (float): Vortex strength
- `Uinf` (float): Inflow velocity magnitude

**Returns:**
- `np.ndarray`: Induced velocity vector

**Viscous Core Model:**
```
ε = √(4 × α₀ × ν × r_perp / U_inf)
```

Where:
- r_perp = perpendicular distance from filament
- Based on viscous diffusion in wake

**Reference:** 
Damiani et al. "A vortex step method for nonlinear airfoil polar data as implemented in KiteAeroDyn"

## Class: SemiInfiniteFilament

Represents a semi-infinite trailing vortex extending from a trailing edge point to infinity in the wake direction.

### Constructor

```python
SemiInfiniteFilament(x1, direction, vel_mag, filament_direction)
```

**Parameters:**
- `x1` (np.ndarray): Starting point (trailing edge)
- `direction` (np.ndarray): Unit vector of wake direction
- `vel_mag` (float): Wake velocity magnitude
- `filament_direction` (int): ±1 indicating filament orientation

### Properties

#### `x1`
Starting point of the semi-infinite filament.

#### `filament_direction`
Direction multiplier (±1) for filament orientation.

### Methods

#### `velocity_3D_trailing_vortex_semiinfinite(Vf, XVP, GAMMA, Uinf)`

Calculates velocity induced by a semi-infinite trailing vortex.

**Parameters:**
- `Vf` (np.ndarray): Wake velocity vector
- `XVP` (np.ndarray): Evaluation point
- `GAMMA` (float): Circulation strength
- `Uinf` (float): Inflow velocity magnitude

**Returns:**
- `np.ndarray`: Induced velocity vector

**Mathematical Formulation:**
```
K = (Γ/4π) × (1 + r1·Vf/||r1||) / ||r1×Vf||²
v_ind = K × (r1×Vf)
```

**Core Treatment:**
- **Outside core**: Standard semi-infinite vortex formula
- **Inside core**: Regularized using viscous core radius
- **On filament**: Zero velocity

## Physical Models

### Vortex Core Corrections

#### Bound Vortex Core (Vatistas Model)
- Core radius: ε = f × L (fraction of filament length)
- Prevents singularity when evaluation point approaches filament
- Linear velocity ramp inside core region

#### Trailing Vortex Core (Viscous Diffusion)
- Core radius: ε = √(4α₀νr_perp/U∞) 
- Physically based on viscous diffusion in wake
- Accounts for Reynolds number effects

### Implementation Details

#### Geometric Calculations
All methods use the fundamental vectors:
- `r0`: Filament direction vector
- `r1`: Vector from start point to evaluation point
- `r2`: Vector from end point to evaluation point
- `r1×r0`, `r2×r0`: Cross products for perpendicular distances

#### Singularity Handling
- **On filament**: Returns zero velocity
- **Inside core**: Uses regularized formulation
- **Numerical stability**: Handles edge cases and near-zero denominators

#### Performance Optimizations
- Uses JIT-compiled cross product, norm, and dot product functions
- Efficient vector operations for repeated calculations
- Minimal memory allocation in hot paths

## Usage in VSM Framework

### Panel Integration

```python
# Bound filament (quarter-chord line)
bound_filament = BoundFilament(
    x1=panel.bound_point_1,
    x2=panel.bound_point_2
)

# Trailing filaments from trailing edge
trailing_1 = BoundFilament(
    x1=panel.bound_point_1,
    x2=panel.TE_point_1
)

# Semi-infinite wake filaments
wake_1 = SemiInfiniteFilament(
    x1=panel.TE_point_1,
    direction=va_unit,
    vel_mag=va_magnitude,
    filament_direction=1
)
```

### Velocity Computation

```python
# Compute induced velocity at control point
v_bound = bound_filament.velocity_3D_bound_vortex(
    control_point, gamma, core_radius_fraction
)

v_trailing = trailing_filament.velocity_3D_trailing_vortex(
    control_point, gamma, va_magnitude
)

v_wake = wake_filament.velocity_3D_trailing_vortex_semiinfinite(
    va_unit, control_point, gamma, va_magnitude
)

# Total induced velocity
v_total = v_bound + v_trailing + v_wake
```

## Error Handling and Edge Cases

### Zero Division Protection
- Checks for zero-length filaments
- Handles coincident evaluation points
- Robust handling of parallel vectors

### Numerical Stability
- Minimum threshold for vector norms
- Graceful degradation for extreme aspect ratios
- Consistent behavior near singular points

### Physical Validity
- Core radius always positive
- Velocity magnitude bounded by physical limits
- Proper handling of negative circulations

## Performance Considerations

### Computational Complexity
- O(1) per filament-point evaluation
- Efficient for large numbers of panels
- JIT compilation reduces function call overhead

### Memory Usage
- Minimal temporary array allocation
- In-place vector operations where possible
- Suitable for real-time applications

### Accuracy vs Speed
- Core models balance accuracy and computational cost
- User-configurable core radius fractions
- Adaptive schemes possible for varying flow conditions

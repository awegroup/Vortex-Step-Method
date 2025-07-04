# Panel Module Documentation

## Overview

The `Panel` class represents a discrete wing panel bounded by two wing sections. It encapsulates the geometric properties, aerodynamic characteristics, and vortex filament system needed for computing induced velocities and aerodynamic forces in the Vortex Step Method.

## Class: Panel

### Constructor

```python
Panel(
    section_1,          # First wing section
    section_2,          # Second wing section  
    aerodynamic_center, # Quarter-chord point
    control_point,      # Three-quarter-chord point
    bound_point_1,      # First bound vortex point
    bound_point_2,      # Second bound vortex point
    x_airf,             # Normal unit vector
    y_airf,             # Chordwise unit vector
    z_airf              # Spanwise unit vector
)
```

**Parameters:**
- `section_1/2`: Section objects defining panel boundaries
- `aerodynamic_center`: Panel aerodynamic center (typically 1/4 chord)
- `control_point`: Panel control point (typically 3/4 chord)
- `bound_point_1/2`: Endpoints of bound vortex filament
- `x_airf/y_airf/z_airf`: Local reference frame unit vectors

## Geometric Properties

### Reference Frame Definition

#### `x_airf` - Normal Vector
- Points upward from chord line
- Perpendicular to panel surface
- Used for computing angle of attack

#### `y_airf` - Chordwise Vector  
- Points from leading edge to trailing edge
- Parallel to local chord line
- Defines drag direction

#### `z_airf` - Spanwise Vector
- Points toward left wing tip
- In airfoil plane, perpendicular to chord
- Defines side force direction

### Geometric Calculations

#### Corner Points
Quadrilateral defined by:
- `LE_point_1`: Leading edge of section 1
- `TE_point_1`: Trailing edge of section 1  
- `TE_point_2`: Trailing edge of section 2
- `LE_point_2`: Leading edge of section 2

#### Dimensional Properties
- **Chord**: Average chord length of bounding sections
- **Width**: Distance between bound vortex endpoints
- **Area**: Chord × Width (for force calculations)

## Aerodynamic Properties

### Polar Data Interpolation

Panel aerodynamic characteristics are interpolated from bounding sections:

```python
self._panel_polar_data = np.array([
    0.5 * (a1 + a2) for a1, a2 in zip(section_1.polar_data, section_2.polar_data)
])
```

**Format**: (N, 4) array with columns [α, CL, CD, CM]

### Coefficient Lookup Methods

#### `compute_cl(alpha)`
Returns lift coefficient for given angle of attack.

**Parameters:**
- `alpha` (float): Angle of attack in radians

**Returns:**
- `float`: Lift coefficient

**Implementation:**
```python
return np.interp(alpha, self._panel_polar_data[:, 0], self._panel_polar_data[:, 1])
```

#### `compute_cd_cm(alpha)`
Returns drag and moment coefficients.

**Parameters:**
- `alpha` (float): Angle of attack in radians

**Returns:**
- `tuple`: (CD, CM) drag and moment coefficients

### Relative Velocity Analysis

#### `compute_relative_alpha_and_relative_velocity(induced_velocity)`

Computes effective angle of attack and relative velocity including induced effects.

**Parameters:**
- `induced_velocity` (np.ndarray): Induced velocity at evaluation point

**Returns:**
- `tuple`: (alpha, relative_velocity)
  - `alpha`: Effective angle of attack (radians)
  - `relative_velocity`: Total velocity vector

**Mathematical Formulation:**
```python
relative_velocity = self.va + induced_velocity
v_normal = dot(x_airf, relative_velocity)
v_tangential = dot(y_airf, relative_velocity)  
alpha = arctan(v_normal / v_tangential)
```

## Vortex Filament System

### Filament Configuration

Each panel contains a system of vortex filaments:

1. **Bound Filament**: Along quarter-chord line
2. **Trailing Filament 1**: From bound to trailing edge 1
3. **Trailing Filament 2**: From bound to trailing edge 2
4. **Semi-infinite Wake 1**: From TE1 to infinity
5. **Semi-infinite Wake 2**: From TE2 to infinity

### Filament Initialization

```python
self._filaments = [
    BoundFilament(x1=bound_point_2, x2=bound_point_1),
    BoundFilament(x1=bound_point_1, x2=TE_point_1),
    BoundFilament(x1=TE_point_2, x2=bound_point_2)
]
# Semi-infinite filaments added by Wake.frozen_wake()
```

### Velocity Computation Methods

#### `compute_velocity_induced_bound_2D(evaluation_point)`

Computes 2D bound vortex induced velocity (VSM correction).

**Purpose:** Remove self-influence of bound vortex in VSM method

**Parameters:**
- `evaluation_point` (np.ndarray): Point where velocity is evaluated

**Returns:**
- `np.ndarray`: 2D induced velocity vector

**Mathematical Formulation:**
```python
r3 = evaluation_point - (bound_point_1 + bound_point_2) / 2
r0 = bound_point_1 - bound_point_2
cross = cross_product(r0, r3)
return cross / ||cross||² / (2π) * ||r0||
```

#### `compute_velocity_induced_single_ring_semiinfinite(...)`

Computes total induced velocity from complete vortex ring system.

**Parameters:**
- `evaluation_point`: Point where velocity is computed
- `evaluation_point_on_bound`: Boolean for LLT vs VSM treatment
- `va_norm`: Apparent velocity magnitude
- `va_unit`: Apparent velocity unit vector
- `gamma`: Circulation strength
- `core_radius_fraction`: Vortex core radius parameter

**Returns:**
- `np.ndarray`: Total induced velocity from all filaments

**Implementation Logic:**
```python
for i, filament in enumerate(self.filaments):
    if i == 0:  # Bound filament
        if evaluation_point_on_bound:
            tempvel = [0, 0, 0]  # No self-influence in LLT
        else:
            tempvel = filament.velocity_3D_bound_vortex(...)
    elif i == 1 or i == 2:  # Trailing filaments
        tempvel = filament.velocity_3D_trailing_vortex(...)
    elif i == 3 or i == 4:  # Semi-infinite wake
        tempvel = filament.velocity_3D_trailing_vortex_semiinfinite(...)
    
    velind += tempvel
```

## Visualization Support

### `compute_filaments_for_plotting()`

Prepares filament data for 3D visualization.

**Returns:**
- `list`: List of [start_point, end_point, color] for each filament

**Color Coding:**
- **Magenta**: Bound vortex filaments
- **Green**: Trailing vortex filaments  
- **Orange**: Forward semi-infinite wake
- **Red**: Backward semi-infinite wake

**Usage:**
```python
filaments = panel.compute_filaments_for_plotting()
for start, end, color in filaments:
    plot_line_3d(start, end, color=color)
```

## VSM vs LLT Differences

### Evaluation Points

**VSM (Vortex Step Method):**
- Boundary conditions enforced at control points (3/4 chord)
- Induced velocities computed at aerodynamic centers (1/4 chord)
- Bound vortex self-influence removed at control points

**LLT (Lifting Line Theory):**
- Boundary conditions enforced at aerodynamic centers (1/4 chord)
- No bound vortex self-influence correction needed

### Implementation Differences

```python
# VSM: evaluation_point_on_bound = False
if not evaluation_point_on_bound:
    # Include bound vortex influence
    v_bound = filament.velocity_3D_bound_vortex(...)
    # Later subtract 2D self-influence
    v_total -= panel.compute_velocity_induced_bound_2D(control_point)

# LLT: evaluation_point_on_bound = True  
if evaluation_point_on_bound:
    # Exclude bound vortex self-influence
    v_bound = [0, 0, 0]
```

## Error Handling and Edge Cases

### Geometric Validation
- Ensures non-zero chord and width
- Validates proper section ordering
- Checks for degenerate quadrilaterals

### Aerodynamic Data
- Handles extrapolation beyond polar data range
- Graceful degradation for missing coefficients
- Interpolation stability for sparse data

### Numerical Stability
- Avoids division by zero in angle calculations
- Robust handling of very small velocities
- Consistent behavior at stall angles

## Performance Considerations

### Memory Efficiency
- Minimal data storage per panel
- Efficient polar data interpolation
- In-place velocity calculations

### Computational Optimization
- JIT-compiled vector operations
- Cached geometric properties
- Optimized filament ordering for cache locality

### Scalability
- O(1) coefficient lookup per panel
- O(N) velocity influence per panel pair
- Suitable for wings with hundreds of panels

## Integration with Framework

### Panel Creation
```python
# From BodyAerodynamics._build_panels()
panel = Panel(
    section_list[j],           # Current section
    section_list[j + 1],       # Next section
    aerodynamic_center_list[j],
    control_point_list[j],
    bound_point_1_list[j],
    bound_point_2_list[j],
    x_airf_list[j],
    y_airf_list[j],
    z_airf_list[j]
)
```

### Force Calculation Pipeline
```python
# 1. Solve for circulation distribution
gamma = solver.solve(body_aero)

# 2. Compute induced velocities
for panel in panels:
    v_ind = panel.compute_velocity_induced_single_ring_semiinfinite(...)
    
# 3. Update effective angle of attack
alpha, v_rel = panel.compute_relative_alpha_and_relative_velocity(v_ind)

# 4. Lookup aerodynamic coefficients  
cl = panel.compute_cl(alpha)
cd, cm = panel.compute_cd_cm(alpha)

# 5. Compute dimensional forces
force = 0.5 * rho * ||v_rel||² * area * [cd, cl, cs]
```
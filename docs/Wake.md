# Wake Module Documentation

## Overview

The `Wake` class manages the trailing vortex wake system behind wing panels. It handles the creation, update, and configuration of semi-infinite vortex filaments that extend from the wing trailing edge into the wake region, representing the shed vorticity from bound circulation.

## Class: Wake

### Design Pattern

**Important**: The Wake class uses a static factory pattern and should not be instantiated directly.

```python
# Correct usage
panels = Wake.frozen_wake(va_distribution, panels)

# Incorrect usage - will raise RuntimeError
wake = Wake()  # RuntimeError: Use static methods
```

## Primary Method: frozen_wake()

### `frozen_wake(va_distribution, panels)`

Updates panel filament systems with semi-infinite wake elements based on the local flow conditions.

**Parameters:**
- `va_distribution` (np.ndarray): Array of velocity vectors at each panel
- `panels` (List[Panel]): List of panel objects to update

**Returns:**
- `List[Panel]`: Updated panel list with wake filaments

### Wake Model Implementation

#### Physical Basis

The frozen wake assumption treats the wake as:
- **Straight lines**: Wake filaments aligned with local velocity
- **Semi-infinite**: Extending to infinity downstream
- **Fixed geometry**: Wake shape doesn't change during iteration

#### Mathematical Formulation

For each panel, wake filaments are created at trailing edge points:

```python
for i, panel in enumerate(panels):
    va_i = va_distribution[i]
    vel_mag = ||va_i||
    direction = va_i / ||va_i||
    
    # Create wake filaments
    wake_fil_1 = SemiInfiniteFilament(TE_point_1, direction, vel_mag, +1)
    wake_fil_2 = SemiInfiniteFilament(TE_point_2, direction, vel_mag, -1)
```

### Filament System Management

#### Initial Configuration (3 filaments)
When panels are first created, they contain:
1. **Bound filament**: Quarter-chord bound vortex
2. **Trailing filament 1**: From bound to TE_point_1  
3. **Trailing filament 2**: From bound to TE_point_2

#### Updated Configuration (5 filaments)
After `frozen_wake()` is called:
1. **Bound filament**: (unchanged)
2. **Trailing filament 1**: (unchanged)
3. **Trailing filament 2**: (unchanged)
4. **Semi-infinite wake 1**: From TE_point_1 to infinity
5. **Semi-infinite wake 2**: From TE_point_2 to infinity

#### Filament Direction Convention

**Filament Direction = +1 (Forward wake)**
- Extends in direction of local velocity
- Represents vorticity convected downstream

**Filament Direction = -1 (Backward wake)**  
- Extends opposite to local velocity direction
- Maintains circulation conservation around panel

```python
# Forward wake filament
SemiInfiniteFilament(TE_point_1, direction, vel_mag, filament_direction=+1)

# Backward wake filament  
SemiInfiniteFilament(TE_point_2, direction, vel_mag, filament_direction=-1)
```

### Update Logic and State Management

#### Smart Update Strategy

The method handles both initial creation and subsequent updates:

```python
if len(panel.filaments) == 3:
    # Initial wake creation
    panel.filaments.append(
        SemiInfiniteFilament(TE_point_1, direction, vel_mag, +1)
    )
    panel.filaments.append(
        SemiInfiniteFilament(TE_point_2, direction, vel_mag, -1)
    )
    
elif len(panel.filaments) == 5:
    # Update existing wake filaments
    panel.filaments[3] = SemiInfiniteFilament(TE_point_1, direction, vel_mag, +1)
    panel.filaments[4] = SemiInfiniteFilament(TE_point_2, direction, vel_mag, -1)
    
else:
    raise ValueError("Unexpected number of filaments")
```

#### Advantages of Update Strategy
- **Preserves bound circulation**: Doesn't modify core vortex system
- **Efficient memory usage**: Reuses filament objects when possible
- **Consistent indexing**: Maintains predictable filament order
- **Error detection**: Catches unexpected filament configurations

## Wake Physics and Assumptions

### Frozen Wake Model

#### Assumptions
1. **Steady flow**: Wake geometry doesn't change with time
2. **Inviscid flow**: No viscous diffusion of wake vortices
3. **Straight wake**: Local velocity determines wake direction
4. **Semi-infinite extent**: Wake extends infinitely downstream

#### Validity Range
- **Low angles of attack**: Wake stays attached and straight
- **Moderate sweep**: Local flow approximation remains valid
- **Steady conditions**: No significant unsteady effects

#### Limitations
- **High angles of attack**: Wake may separate and roll up
- **Highly swept wings**: 3D effects become important
- **Unsteady maneuvers**: Wake history effects neglected

### Physical Significance

#### Circulation Conservation
The wake system ensures that circulation is conserved around each panel:
```
Γ_bound + Γ_trailing_1 + Γ_trailing_2 + Γ_wake_1 + Γ_wake_2 = 0
```

#### Kutta Condition
Wake filaments enforce the Kutta condition at the trailing edge by:
- Extending circulation smoothly into wake
- Preventing flow around sharp trailing edge
- Maintaining finite velocities at TE

#### Induced Velocity Field
Wake filaments contribute to induced velocities throughout the flow field:
- **Near field**: Significant influence on wing panels
- **Far field**: Determines downwash and induced drag
- **Ground effect**: Modified by ground proximity

## Integration with VSM Framework

### Panel Velocity Updates

```python
# In BodyAerodynamics.va setter
def va(self, va_value, yaw_rate=0.0):
    # Update panel velocities
    for i, panel in enumerate(self.panels):
        panel.va = va_distribution[i]
    
    # Update wake filaments based on new velocities
    self.panels = Wake.frozen_wake(va_distribution, self.panels)
```

### Solver Integration

```python
# In Solver.solve()
# AIC matrices include wake filament contributions
AIC_x, AIC_y, AIC_z = body_aero.compute_AIC_matrices(
    aerodynamic_model_type,
    core_radius_fraction,
    va_norm_array, 
    va_unit_array
)

# Wake filaments affect induced velocity calculations
for panel in panels:
    v_induced = panel.compute_velocity_induced_single_ring_semiinfinite(...)
```

### Visualization Support

Wake filaments are included in plotting routines:

```python
# From Panel.compute_filaments_for_plotting()
for i, filament in enumerate(self.filaments):
    if i >= 3:  # Semi-infinite wake filaments
        x2 = x1 + chord_length * (va / ||va||)
        color = "orange" if filament_direction == 1 else "red"
        filaments.append([x1, x2, color])
```

## Performance Considerations

### Computational Efficiency
- **O(N) update complexity**: Linear in number of panels
- **Minimal memory allocation**: Reuses existing filament objects
- **JIT compatibility**: Uses optimized vector operations

### Memory Management
- **In-place updates**: Modifies existing panel list
- **Object reuse**: Avoids creating new filament objects when possible
- **Garbage collection**: Old filaments automatically cleaned up

### Numerical Stability
- **Direction normalization**: Ensures unit wake direction vectors
- **Velocity magnitude**: Handles zero and very small velocities
- **Filament ordering**: Maintains consistent indexing scheme

## Error Handling and Validation

### Input Validation
```python
# Velocity distribution validation
if len(va_distribution) != len(panels):
    raise ValueError("Velocity distribution length mismatch")

# Panel filament count validation  
if len(panel.filaments) not in [3, 5]:
    raise ValueError("Unexpected number of filaments")
```

### Robustness Features
- **Zero velocity handling**: Graceful degradation for zero flow
- **Direction computation**: Robust normalization for small velocities
- **State consistency**: Maintains valid filament configurations

## Advanced Wake Models (Future Extensions)

### Potential Enhancements

#### Relaxed Wake Model
- Time-dependent wake evolution
- Wake rollup and contraction effects
- History-dependent wake shapes

#### Viscous Wake Model  
- Viscous diffusion of wake vortices
- Core radius growth with downstream distance
- Turbulent mixing effects

#### Ground Effect Wake
- Wake interaction with ground plane
- Image vortex systems
- Modified induced velocity field

### Implementation Considerations
- **Computational cost**: More complex models require additional computation
- **Memory requirements**: Time-dependent wakes need history storage
- **Numerical stability**: Advanced models may need special treatment

## Usage Examples

### Basic Wake Update
```python
# Create velocity distribution
va_distribution = np.array([
    [10.0, 0.0, 1.0],  # Panel 1 velocity
    [10.0, 0.0, 1.2],  # Panel 2 velocity  
    [10.0, 0.0, 1.4],  # Panel 3 velocity
])

# Update wake system
panels = Wake.frozen_wake(va_distribution, panels)

# Verify wake filament creation
for panel in panels:
    assert len(panel.filaments) == 5
    assert panel.filaments[3].filament_direction == 1
    assert panel.filaments[4].filament_direction == -1
```

### Integration with Flow Solution
```python
# In iterative solver loop
for iteration in range(max_iterations):
    # Solve for new circulation distribution
    gamma_new = solve_circulation(panels)
    
    # Update induced velocities (includes wake effects)
    for i, panel in enumerate(panels):
        v_induced[i] = compute_induced_velocity(panel, gamma_new)
    
    # Update apparent velocities  
    va_new = va_freestream + v_induced
    
    # Update wake filaments based on new flow field
    panels = Wake.frozen_wake(va_new, panels)
```

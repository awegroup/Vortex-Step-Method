# WingGeometry Module Documentation

## Overview

The `WingGeometry` module defines the geometric representation of wing structures through the `Wing` and `Section` classes. It provides comprehensive functionality for creating, modifying, and refining wing geometries for aerodynamic analysis, including advanced mesh generation and geometric property calculations.

## Class: Section

A dataclass representing a single wing section with leading edge, trailing edge, and aerodynamic properties.

### Attributes
- **`LE_point`** (np.ndarray): Leading edge coordinates [x, y, z]
- **`TE_point`** (np.ndarray): Trailing edge coordinates [x, y, z]
- **`polar_data`** (np.ndarray): Airfoil polar data (N×4: [α, CL, CD, CM])

### Properties

#### `chord_vector`
Returns the chord vector from leading edge to trailing edge.
```python
@property
def chord_vector(self):
    return self.TE_point - self.LE_point
```

#### `chord_length`
Returns the magnitude of the chord vector.
```python
@property  
def chord_length(self):
    return np.linalg.norm(self.chord_vector)
```

## Class: Wing

A dataclass representing a complete wing geometry composed of multiple sections with configurable panel distributions and aerodynamic mesh refinement capabilities.

### Attributes

#### Core Configuration
- **`n_panels`** (int): Number of panels in the aerodynamic mesh
- **`spanwise_panel_distribution`** (str): Panel distribution strategy
- **`spanwise_direction`** (np.ndarray): Wing spanwise unit vector [0,1,0]
- **`sections`** (List[Section]): Ordered list of wing sections

#### Panel Distribution Types

**`"uniform"`** - Linear Spacing
- Equal spacing along wing span
- Simple and robust for most applications
- Good for preliminary analysis

**`"cosine"`** - Cosine Spacing  
- Higher density near wing tips
- Better resolution of tip effects
- Improved accuracy for induced drag calculations

**`"cosine_van_Garrel"`** - Van Garrel Cosine Method
- Specialized cosine distribution
- Optimized for lifting line applications
- Currently not implemented

**`"split_provided"`** - Section Splitting
- Subdivides existing sections uniformly
- Maintains original geometry definition points
- Requires n_panels to be multiple of existing panels

**`"unchanged"`** - Keep Original Sections
- Uses sections exactly as provided
- No mesh refinement applied
- Number of panels = number of sections - 1

## Wing Construction and Modification

### `add_section(LE_point, TE_point, polar_data)`

Adds a new section to the wing definition.

**Parameters:**
- `LE_point` (np.ndarray): Leading edge coordinates [x, y, z]
- `TE_point` (np.ndarray): Trailing edge coordinates [x, y, z]  
- `polar_data` (np.ndarray): Airfoil polar data (N×4: [α, CL, CD, CM])

**Usage:**
```python
wing = Wing(n_panels=8, spanwise_panel_distribution="cosine")

# Add wing sections from root to tip
sections = [
    ([0.0, -4.0, 0.0], [1.0, -4.0, 0.0]),  # Left tip
    ([0.0, -2.0, 0.0], [1.2, -2.0, 0.0]),  # Left mid
    ([0.0,  0.0, 0.0], [1.5,  0.0, 0.0]),  # Root
    ([0.0,  2.0, 0.0], [1.2,  2.0, 0.0]),  # Right mid
    ([0.0,  4.0, 0.0], [1.0,  4.0, 0.0]),  # Right tip
]

for le, te in sections:
    wing.add_section(np.array(le), np.array(te), polar_data)
```

### `update_wing_from_points(le_arr, te_arr, aero_input_type, polar_data_arr)`

Updates wing geometry from coordinate arrays.

**Parameters:**
- `le_arr` (np.ndarray): Array of leading edge points
- `te_arr` (np.ndarray): Array of trailing edge points
- `aero_input_type` (str): Must be "reuse_initial_polar_data"
- `polar_data_arr` (list): List of polar data arrays for each section

## Mesh Refinement System

### `refine_aerodynamic_mesh()`

Main method that refines the wing mesh according to the specified distribution type.

**Process:**
1. **Section Sorting**: Orders sections using proximity-based algorithm
2. **Distribution Selection**: Applies chosen panel distribution method
3. **Interpolation**: Creates new sections with interpolated properties
4. **Validation**: Ensures correct number of output sections

**Returns:**
- `List[Section]`: Refined section list with (n_panels + 1) sections

### `find_farthest_point_and_sort(sections)`

Intelligent section sorting algorithm for proper mesh ordering.

**Algorithm:**
```python
def find_farthest_point_and_sort(sections):
    # 1. Find section with positive y-coordinate that is farthest from all others
    farthest_point = max(sections with y > 0, key=total_distance_to_others)
    
    # 2. Start sorted list with farthest point  
    sorted_sections = [farthest_point]
    remaining = sections - farthest_point
    
    # 3. Iteratively add closest remaining section
    while remaining:
        last_point = sorted_sections[-1].LE_point
        closest = min(remaining, key=distance_to_last_point)
        sorted_sections.append(closest)
        remaining.remove(closest)
    
    return sorted_sections
```

**Purpose:**
- Ensures consistent section ordering from tip to tip
- Prevents mesh folding and geometric artifacts
- Required for proper interpolation

### Advanced Mesh Refinement Methods

#### `refine_mesh_for_uniform_or_cosine_distribution()`

Sophisticated interpolation method for uniform and cosine distributions.

**Process:**

1. **Quarter-Chord Line Construction**
   ```python
   quarter_chord = LE + 0.25 * (TE - LE)
   ```

2. **Arc Length Parameterization**
   ```python
   qc_lengths = ||quarter_chord[i+1] - quarter_chord[i]||
   qc_cum_length = cumsum([0, qc_lengths])
   ```

3. **Target Position Calculation**
   ```python
   # Uniform distribution
   target_lengths = linspace(0, total_length, n_sections)
   
   # Cosine distribution  
   theta = linspace(0, π, n_sections)
   target_lengths = total_length * (1 - cos(theta)) / 2
   ```

4. **Geometric Interpolation**
   ```python
   # Find which segment contains target position
   section_index = searchsorted(qc_cum_length, target_length) - 1
   t = (target_length - qc_cum_length[section_index]) / segment_length
   
   # Interpolate quarter-chord point
   new_qc = qc[section_index] + t * (qc[section_index+1] - qc[section_index])
   ```

5. **Chord Vector Interpolation**
   ```python
   # Normalize chord directions
   left_chord_norm = left_chord / ||left_chord||
   right_chord_norm = right_chord / ||right_chord||
   
   # Interpolate direction and length separately
   avg_direction = normalize(left_weight * left_chord_norm + right_weight * right_chord_norm)
   avg_length = left_weight * left_length + right_weight * right_length
   
   # Reconstruct chord vector
   avg_chord = avg_direction * avg_length
   ```

6. **Section Reconstruction**
   ```python
   new_LE = new_qc - 0.25 * avg_chord
   new_TE = new_qc + 0.75 * avg_chord
   ```

#### `compute_new_polar_data(polar_data, section_index, left_weight, right_weight)`

Advanced polar data interpolation between adjacent sections.

**Process:**
```python
# 1. Extract polar data from bounding sections
alpha_left, CL_left, CD_left, CM_left = polar_left.T
alpha_right, CL_right, CD_right, CM_right = polar_right.T

# 2. Create union of alpha ranges
alpha_common = union(alpha_left, alpha_right)

# 3. Interpolate to common alpha array
CL_left_common = interp(alpha_common, alpha_left, CL_left)
CL_right_common = interp(alpha_common, alpha_right, CL_right)
# ... similar for CD, CM

# 4. Weighted interpolation
CL_interp = CL_left_common * left_weight + CL_right_common * right_weight
CD_interp = CD_left_common * left_weight + CD_right_common * right_weight
CM_interp = CM_left_common * left_weight + CM_right_common * right_weight

# 5. Combine into new polar array
new_polar = column_stack([alpha_common, CL_interp, CD_interp, CM_interp])
```

#### `refine_mesh_by_splitting_provided_sections()`

Splits existing sections to achieve desired panel count.

**Requirements:**
- `n_panels_desired` must be multiple of `n_panels_provided`
- Maintains original section definition points
- Uniform subdivision between section pairs

**Algorithm:**
```python
n_new_sections = n_panels_desired + 1 - n_sections_provided
n_section_pairs = n_sections_provided - 1
new_sections_per_pair, remaining = divmod(n_new_sections, n_section_pairs)

for pair_index in range(n_section_pairs):
    # Add original section
    new_sections.append(sections[pair_index])
    
    # Calculate subdivisions for this pair
    num_subdivisions = new_sections_per_pair + (1 if pair_index < remaining else 0)
    
    # Create subdivisions using uniform interpolation
    if num_subdivisions > 0:
        subdivisions = refine_mesh_for_uniform_distribution(
            "uniform", num_subdivisions + 2, LE_pair, TE_pair, polar_pair
        )
        new_sections.extend(subdivisions[1:-1])  # Exclude endpoints

# Add final section
new_sections.append(sections[-1])
```

## Geometric Properties

### `span` (Property)

Computes wing span along the specified spanwise direction.

**Implementation:**
```python
@property
def span(self):
    y_coords = [section.LE_point[1] for section in self.sections]
    return max(y_coords) - min(y_coords)
```

**Returns:**
- `float`: Wing span in spanwise direction units

### `compute_projected_area(z_plane_vector=None)`

Calculates the projected area of the wing onto a specified plane.

**Parameters:**
- `z_plane_vector` (np.ndarray, optional): Normal vector of projection plane (default: [0,0,1])

**Returns:**
- `float`: Projected wing area

**Mathematical Formulation:**
```python
# For each panel between adjacent sections
panel_area = 0.5 * ||cross_product(diagonal1, diagonal2)||
projected_area = sum(panel_areas)
```

**Applications:**
- Reference area for force coefficients
- Planform area calculations
- Ground effect analysis

## Usage Examples

### Basic Wing Creation
```python
# Create wing with cosine panel distribution
wing = Wing(n_panels=12, spanwise_panel_distribution="cosine")

# Define airfoil polar data
alpha = np.deg2rad(np.arange(-10, 21, 1))
cl = 2 * np.pi * alpha  # Linear lift slope
cd = 0.01 * np.ones_like(alpha)  # Constant drag
cm = np.zeros_like(alpha)  # No pitching moment
polar_data = np.column_stack([alpha, cl, cd, cm])

# Add sections (tip to tip)
sections = [
    ([0.0, -5.0, 0.0], [0.8, -5.0, 0.0]),  # Left tip
    ([0.0, -2.5, 0.0], [1.0, -2.5, 0.0]),  # Left mid
    ([0.0,  0.0, 0.0], [1.2,  0.0, 0.0]),  # Root
    ([0.0,  2.5, 0.0], [1.0,  2.5, 0.0]),  # Right mid  
    ([0.0,  5.0, 0.0], [0.8,  5.0, 0.0]),  # Right tip
]

for le, te in sections:
    wing.add_section(np.array(le), np.array(te), polar_data)
```

### Mesh Refinement
```python
# Refine mesh according to distribution type
refined_sections = wing.refine_aerodynamic_mesh()

print(f"Original sections: {len(wing.sections)}")
print(f"Refined sections: {len(refined_sections)}")
print(f"Number of panels: {len(refined_sections) - 1}")

# Verify panel count matches specification
assert len(refined_sections) - 1 == wing.n_panels
```

### Geometric Analysis
```python
# Calculate wing properties
span = wing.span
area = wing.compute_projected_area()
aspect_ratio = span**2 / area

print(f"Wing span: {span:.2f}")
print(f"Projected area: {area:.2f}")  
print(f"Aspect ratio: {aspect_ratio:.2f}")

# Analyze section properties
for i, section in enumerate(wing.sections):
    chord = section.chord_length
    y_pos = section.LE_point[1]
    print(f"Section {i}: y={y_pos:.2f}, chord={chord:.3f}")
```

### Integration with VSM Framework
```python
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver

# Create aerodynamic model
body_aero = BodyAerodynamics([wing])

# Set flight conditions
body_aero.va_initialize(
    Umag=15.0,
    angle_of_attack=8.0,
    side_slip=0.0
)

# Solve aerodynamics
solver = Solver(aerodynamic_model_type="VSM")
results = solver.solve(body_aero)

print(f"Lift coefficient: {results['cl']:.3f}")
print(f"Drag coefficient: {results['cd']:.4f}")
```

## Advanced Features

### Custom Panel Distributions

#### Implementing Custom Distributions
```python
def custom_distribution(n_sections):
    """Custom panel distribution function"""
    # Example: Higher density at mid-span
    positions = np.linspace(0, 1, n_sections)
    weights = 1 + 2 * np.exp(-((positions - 0.5) / 0.2)**2)
    return positions, weights

# Modify Wing class to support custom distributions
wing.spanwise_panel_distribution = "custom"
wing.custom_distribution_func = custom_distribution
```

#### Adaptive Mesh Refinement
```python
def adaptive_refinement(wing, criterion_func, max_panels=50):
    """Adaptively refine mesh based on geometric criterion"""
    current_panels = len(wing.sections) - 1
    
    while current_panels < max_panels:
        sections = wing.refine_aerodynamic_mesh()
        
        # Evaluate refinement criterion
        if criterion_func(sections):
            break
            
        # Increase panel count
        wing.n_panels = min(wing.n_panels * 2, max_panels)
        current_panels = wing.n_panels
    
    return wing.refine_aerodynamic_mesh()
```

### Geometric Validation

#### Section Ordering Validation
```python
def validate_section_ordering(sections):
    """Validate that sections are properly ordered"""
    y_coords = [s.LE_point[1] for s in sections]
    
    # Check for monotonic ordering
    is_monotonic = all(y_coords[i] <= y_coords[i+1] for i in range(len(y_coords)-1))
    
    if not is_monotonic:
        raise ValueError("Sections are not properly ordered spanwise")
    
    return True
```

#### Geometric Consistency Checks
```python
def check_geometric_consistency(wing):
    """Perform geometric consistency checks"""
    issues = []
    
    for i, section in enumerate(wing.sections):
        # Check for zero-chord sections
        if section.chord_length < 1e-6:
            issues.append(f"Section {i} has near-zero chord length")
        
        # Check for inverted airfoils
        chord_vec = section.chord_vector
        if chord_vec[0] < 0:  # Negative x-component
            issues.append(f"Section {i} may have inverted chord direction")
    
    return issues
```

## Performance Considerations

### Computational Complexity
- **Section sorting**: O(N²) for N sections
- **Mesh refinement**: O(N × M) for N sections and M panels
- **Polar interpolation**: O(P) for P polar data points per section

### Memory Usage
- **Section storage**: O(N × P) for N sections with P polar points each
- **Temporary arrays**: O(M) during refinement for M target sections
- **Garbage collection**: Automatic cleanup of intermediate objects

### Optimization Strategies
- **Pre-sorted sections**: Skip sorting if sections already ordered
- **Cached calculations**: Store computed quarter-chord lines
- **Vectorized operations**: Use NumPy for bulk geometric calculations

## Error Handling and Robustness

### Input Validation
```python
# Validate panel distribution type
valid_distributions = ["uniform", "cosine", "split_provided", "unchanged"]
if spanwise_panel_distribution not in valid_distributions:
    raise ValueError(f"Invalid distribution type: {spanwise_panel_distribution}")

# Validate panel count
if n_panels < 1:
    raise ValueError("Number of panels must be positive")
```

### Geometric Robustness
- **Degenerate sections**: Handle zero-chord and coincident points
- **Numerical precision**: Use appropriate tolerances for floating-point comparisons
- **Interpolation bounds**: Ensure interpolation weights sum to unity

### Recovery Strategies
- **Fallback distributions**: Use uniform distribution if custom method fails
- **Section validation**: Automatically fix minor geometric inconsistencies
- **Error reporting**: Provide detailed error messages with suggested fixes

## Integration with Other Modules

### BodyAerodynamics Integration
```python
# Wing geometry flows into panel generation
body_aero = BodyAerodynamics([wing])
panels = body_aero.panels  # Generated from refined wing sections
```

### AirfoilAerodynamics Integration
```python
# Polar data from AirfoilAerodynamics
from VSM.core.AirfoilAerodynamics import AirfoilAerodynamics

aero = AirfoilAerodynamics.from_yaml_entry(
    "breukels_regression",
    {"t": 0.12, "kappa": 0.08}
)
polar_data = aero.to_polar_array()

# Use in wing sections
wing.add_section(LE_point, TE_point, polar_data)
```

### Visualization Integration
```python
# Wing geometry visualization
from VSM.plot_geometry_plotly import interactive_plot

interactive_plot(
    body_aero,
    vel=15.0,
    angle_of_attack=8.0,
    title="Wing Geometry",
    is_show=True
)
```

## References

1. **Mesh Generation**: Thompson, J.F. et al. "Handbook of Grid Generation" (1999)
2. **Geometric Modeling**: Farin, G. "Curves and Surfaces for CAGD" (2002)
3. **Aerodynamic Applications**: Katz, J. & Plotkin, A. "Low-Speed Aerodynamics" (2001)
4. **Computational Geometry**: de Berg, M. et al. "Computational Geometry" (2008)
5. **Panel Method Theory**: Hess, J.L. & Smith, A.M.O. "Calculation of potential flow about arbitrary bodies" (1967)

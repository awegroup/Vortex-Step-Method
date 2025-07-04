# Style Guide for Python Code

## Documentation Standards

### Docstrings
- Use typed Google syntax for all docstrings
- Include comprehensive Args, Returns, and Raises sections
- Provide clear descriptions with mathematical formulations where applicable
- Example:
```python
def compute_velocity_induced_bound_vortex(
    self, XVP: np.ndarray, gamma: float, core_radius_fraction: float
) -> np.ndarray:
    """Calculate velocity induced by bound vortex filament using Vatistas core model.

    Vortex core correction from: Rick Damiani et al. "A vortex step method for nonlinear
    airfoil polar data as implemented in KiteAeroDyn".

    Args:
        XVP (np.ndarray): Evaluation point coordinates.
        gamma (float): Vortex strength (circulation).
        core_radius_fraction (float): Core radius as fraction of filament length.

    Returns:
        np.ndarray: Induced velocity vector [vx, vy, vz].

    Raises:
        ValueError: If core_radius_fraction is negative.
    """
```

### Type Hints
- Use type hints for all function parameters and return values
- Import from `typing` module when needed: `List`, `Tuple`, `Dict`, `Optional`, `Union`
- Use `np.ndarray` for NumPy arrays
- Example: `def solve(self, body_aero: BodyAerodynamics, gamma_distribution: np.ndarray = None) -> dict:`

## Naming Conventions

### Variables and Functions
- Use `snake_case` for variable and function names
- Variable names should be descriptive and self-documenting
- Use full words rather than abbreviations when possible
- Examples: `gamma_distribution`, `compute_aerodynamic_quantities`, `alpha_array`

### File and Directory Naming
- Variable name of path to a file ends with `_path`
- Variable name of a directory ends with `_dir`
- Examples: `polar_file_path`, `airfoil_data_dir`, `nf_airfoil_data_dir`

### Classes
- Use `CamelCase` for class names
- Class names should be nouns that clearly describe the entity
- Examples: `BodyAerodynamics`, `AirfoilAerodynamics`, `SemiInfiniteFilament`

### Constants
- Use `UPPER_SNAKE_CASE` for module-level constants
- Examples: `MAX_ITERATIONS`, `DEFAULT_TOLERANCE`

### Private Attributes
- Use leading underscore `_` for private/internal class attributes
- Examples: `_alpha0`, `_nu`, `_panel_polar_data`

## Code Structure

### Class Design
- Use `@dataclass` for simple data containers with automatic `__init__`
- Use `@property` for computed attributes and getters
- Use `@classmethod` for alternative constructors
- Use `@staticmethod` for utility functions that don't need class state
- Example:
```python
@dataclass
class Wing:
    n_panels: int
    spanwise_panel_distribution: str = "uniform"
    sections: List[Section] = field(default_factory=list)

    @classmethod
    def from_yaml_entry(cls, config: dict) -> 'Wing':
        """Alternative constructor from configuration."""
        ...

    @property
    def span(self) -> float:
        """Calculate wing span."""
        ...

    @staticmethod
    def compute_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """Utility function for distance calculation."""
        ...
```

### Method Organization
- Group methods logically within classes:
  1. `__init__` and class methods
  2. Properties (getters/setters)
  3. Public methods
  4. Private methods (prefixed with `_`)

### Factory Pattern
- Use factory methods for complex object creation
- Prevent direct instantiation when appropriate
- Example:
```python
class AirfoilAerodynamics:
    def __init__(self):
        raise RuntimeError("Use AirfoilAerodynamics.from_yaml_entry(...) to instantiate.")

    @classmethod
    def from_yaml_entry(cls, airfoil_type: str, airfoil_params: dict, ...) -> 'AirfoilAerodynamics':
        """Create instance from configuration parameters."""
        ...
```

## String Formatting
- Use f-strings for string formatting
- Examples: `f"Normalized error at iteration {i}: {normalized_error}"`
- For logging: `logging.info(f"Converged after {iterations} iterations")`

## Error Handling

### Exception Types
- Use specific exception types: `ValueError`, `TypeError`, `FileNotFoundError`
- Provide descriptive error messages
- Example:
```python
if n_panels < 1:
    raise ValueError("Number of panels must be positive")

if not file_path.exists():
    raise FileNotFoundError(f"Airfoil file not found: {file_path}")
```

### Validation
- Validate inputs early in functions
- Use clear error messages that guide the user
- Example:
```python
if aerodynamic_model_type not in ["VSM", "LLT"]:
    raise ValueError(f"Invalid model type: {aerodynamic_model_type}. Use 'VSM' or 'LLT'.")
```

## Imports and Dependencies

### Import Organization
- Standard library imports first
- Third-party imports second
- Local imports last
- Use absolute imports for internal modules
- Example:
```python
from abc import ABC, abstractmethod
import numpy as np
import logging
from pathlib import Path

from VSM.core.Filament import BoundFilament
from . import jit_cross, jit_norm, jit_dot
```

### JIT Compilation
- Use JIT-compiled utility functions for performance-critical operations
- Import from utils module: `jit_cross`, `jit_norm`, `jit_dot`
- Example usage: `distance = jit_norm(vector)`

## Mathematical and Scientific Conventions

### Physical Units
- Document units in docstrings and variable names where appropriate
- Examples: `velocity_ms` (m/s), `angle_rad` (radians), `length_m` (meters)

### Array Conventions
- Use descriptive suffixes for arrays:
  - `_array` for vector/array of values (one per panel)
  - `_distribution` for spanwise distributions
  - `_norm` for magnitude/norm of vector quantity
  - `_unit` for unit vectors (normalized direction)

### Coordinate Systems
- Document coordinate system conventions clearly
- Use consistent naming: `x_airf`, `y_airf`, `z_airf` for local reference frames
- Examples: `LE_point`, `TE_point`, `control_point`, `aerodynamic_center`

## Performance Optimization

### NumPy Usage
- Use vectorized operations instead of loops where possible
- Pre-allocate arrays with known sizes
- Use in-place operations when memory is a concern
- Example:
```python
# Good: vectorized operation
alpha_array = np.arctan(v_normal_array / v_tangential_array)

# Avoid: element-wise loop
alpha_array = [np.arctan(v_n / v_t) for v_n, v_t in zip(v_normal_array, v_tangential_array)]
```

### Memory Management
- Use `np.zeros()` or `np.empty()` for pre-allocation
- Minimize temporary array creation in loops
- Reuse objects when possible (e.g., filament updates in Wake class)

## Logging and Debugging

### Logging Levels
- Use appropriate logging levels:
  - `logging.debug()` for detailed diagnostic information
  - `logging.info()` for general information
  - `logging.warning()` for warnings that don't stop execution
  - `logging.error()` for errors that might cause problems

### Debugging Information
- Include iteration numbers and convergence metrics
- Log important state changes and method calls
- Example:
```python
logging.debug(f"Normalized error at iteration {i}: {normalized_error}")
logging.info(f"Converged (non_linear: broyden1)")
logging.warning(f"NOT Converged after {self.max_iterations} iterations")
```

## Testing and Validation

### Input Validation
- Check array shapes and dimensions
- Validate ranges for physical parameters
- Example:
```python
if len(va_distribution) != len(panels):
    raise ValueError("Velocity distribution length mismatch")

if alpha_range[0] >= alpha_range[1]:
    raise ValueError("Invalid alpha range: min must be less than max")
```

### Robustness
- Handle edge cases gracefully
- Provide fallback options when possible
- Check for division by zero and other numerical issues
- Example:
```python
reference_error = max(abs(gamma_new)) if max(abs(gamma_new)) != 0 else 1e-4
direction = direction / max(jit_norm(direction), 1e-12)  # Avoid division by zero
```

## Configuration and Parameters

### Default Values
- Provide sensible defaults for optional parameters
- Document the reasoning behind default choices
- Example:
```python
def __init__(
    self,
    aerodynamic_model_type: str = "VSM",  # Most accurate method
    max_iterations: int = 5000,           # Conservative limit
    allowed_error: float = 1e-6,          # Engineering accuracy
    relaxation_factor: float = 0.01,      # Stable convergence
):
```

### Configuration Patterns
- Use dictionaries for grouped parameters
- Support both direct instantiation and configuration-based creation
- Example:
```python
artificial_damping: dict = {"k2": 0.1, "k4": 0.0}
stall_params = {
    "is_smooth_circulation": False,
    "smoothness_factor": 0.08,
    "is_artificial_damping": False
}
```

## Code Comments

### Inline Comments
- Use sparingly for complex algorithms or non-obvious code
- Explain the "why" not the "what"
- Example:
```python
# Apply under-relaxation for stability
gamma_new = (1 - relaxation_factor) * gamma + relaxation_factor * gamma_new

# Oseen parameter for viscous diffusion
self._alpha0 = 1.25643
```

### Section Comments
- Use clear section headers for major code blocks
- Example:
```python
###########################
## GETTER FUNCTIONS
###########################

###########################
## SETTER FUNCTIONS  
###########################

###########################
## CALCULATE FUNCTIONS      # All these return something
###########################
```

## File Organization

### Module Structure
- Keep related functionality together
- Separate abstract base classes from concrete implementations
- Use clear inheritance hierarchies
- Example: `Filament` (ABC) → `BoundFilament`, `SemiInfiniteFilament`

### Dependencies
- Minimize circular imports
- Use relative imports within packages: `from . import jit_cross`
- Keep external dependencies to minimum and document requirements

This style guide ensures consistency, readability, and maintainability across the VSM codebase while following Python best practices and scientific computing conventions.


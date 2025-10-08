# VSM Documentation Index

## Core Module Documentation

### Aerodynamic Components
1. **[AirfoilAerodynamics.md](AirfoilAerodynamics.md)** - Factory interface for 2D airfoil polar data
   - Breukels regression for LEI kites
   - NeuralFoil integration
   - Polar data import from CSV
   - Inviscid thin airfoil theory
   - Masure regression (ML-based predictions)

2. **[WingGeometry.md](WingGeometry.md)** - Wing geometry definition and mesh refinement
   - Section-based wing construction
   - Panel distribution strategies (uniform, cosine, split_provided)
   - Geometric property calculations

3. **[Panel.md](Panel.md)** - Individual wing panel aerodynamics
   - Local aerodynamic property calculations
   - Induced velocity computations
   - Reference frame transformations

4. **[Filament.md](Filament.md)** - Vortex filament velocity calculations
   - Bound vortex with Vatistas core model
   - Trailing vortex with viscous diffusion
   - Semi-infinite filament models

5. **[BodyAerodynamics.md](BodyAerodynamics.md)** - Main orchestration class
   - Combines wing geometry with aerodynamic analysis
   - VSM and LLT implementations
   - Flight condition initialization
   - Bridle line modeling

6. **[Solver.md](Solver.md)** - Iterative circulation solver
   - Fixed-point iteration with relaxation
   - Nonlinear solvers (Broyden methods)
   - Stall modeling capabilities
   - Comprehensive results output

7. **[Wake.md](Wake.md)** - Wake modeling and evolution
   - Semi-infinite wake structures
   - Wake panel definitions

## Analysis Utilities

8. **[StabilityDerivatives.md](StabilityDerivatives.md)** - Stability derivative computation
   - Rigid-body aerodynamic derivatives
   - Angle derivatives (dC*/dα, dC*/dβ)
   - Rate derivatives (dC*/dp, dC*/dq, dC*/dr)
   - Non-dimensionalization options
   - Finite-difference evaluation

9. **[TrimAngle.md](TrimAngle.md)** - Trim angle finding
   - Zero-moment angle of attack search
   - Pitch stability verification
   - Bisection refinement algorithm
   - Multiple trim point detection

## Additional Resources

10. **[nomenclature.md](nomenclature.md)** - Symbol definitions and conventions

11. **[style_guide.md](style_guide.md)** - Python coding standards
    - Docstring conventions (typed Google syntax)
    - Type hints usage
    - Naming conventions
    - File and directory naming

12. **[Aerodynamic_model.pdf](Aerodynamic_model.pdf)** - Theoretical background and mathematical formulations

## Quick Start Examples

### Basic Aerodynamic Analysis

```python
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver

# Create from YAML configuration
body_aero = BodyAerodynamics.instantiate(
    n_panels=30,
    file_path="config.yaml",
    spanwise_panel_distribution="cosine"
)

# Set flight conditions
body_aero.va_initialize(
    Umag=15.0,              # m/s
    angle_of_attack=8.0,    # degrees
    side_slip=0.0           # degrees
)

# Solve
solver = Solver(aerodynamic_model_type="VSM")
results = solver.solve(body_aero)

print(f"CL = {results['cl']:.3f}")
print(f"CD = {results['cd']:.3f}")
```

### Stability Analysis

```python
from VSM.stability_derivatives import compute_rigid_body_stability_derivatives
from VSM.trim_angle import compute_trim_angle

# Find trim angle
trim_solutions = compute_trim_angle(
    body_aero, solver,
    velocity_magnitude=15.0,
    alpha_min=-5.0,
    alpha_max=20.0
)

# Compute stability derivatives at trim
for solution in trim_solutions:
    if solution['stable']:
        derivatives = compute_rigid_body_stability_derivatives(
            body_aero, solver,
            angle_of_attack=solution['alpha_trim'],
            side_slip=0.0,
            velocity_magnitude=15.0
        )
        
        print(f"Trim angle: {solution['alpha_trim']:.2f}°")
        print(f"Pitch stiffness: {derivatives['dCMy_dalpha']:.4f}")
        print(f"Pitch damping: {derivatives['dCMy_dq']:.4f}")
```

## YAML Configuration Format

### Wing Definition

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
    - [mid, neuralfoil, {dat_file_path: "airfoils/custom.dat"}]
```

See individual documentation files for detailed YAML configuration options.

## Module Dependencies

### External Dependencies
- **NumPy**: Array operations and linear algebra
- **SciPy**: Optimization and root finding
- **NeuralFoil** (optional): Neural network airfoil analysis
- **scikit-learn** (optional): Machine learning models for Masure regression

### Internal Structure
```
VSM/
├── core/
│   ├── AirfoilAerodynamics.py
│   ├── BodyAerodynamics.py
│   ├── Filament.py
│   ├── Panel.py
│   ├── Solver.py
│   ├── Wake.py
│   ├── WingGeometry.py
│   └── utils.py
├── stability_derivatives.py
├── trim_angle.py
├── plotting.py
├── convergence_analysis.py
├── sensitivity_analysis.py
└── fitting.py
```

## Common Workflows

### 1. Parametric Study (Angle of Attack Sweep)

```python
alphas = np.linspace(-5, 20, 26)
results_list = []

for alpha in alphas:
    body_aero.va_initialize(Umag=15.0, angle_of_attack=alpha)
    results = solver.solve(body_aero)
    results_list.append(results)

# Extract CL vs alpha
CL_array = [r['cl'] for r in results_list]
```

### 2. Convergence Study (Panel Count)

```python
from VSM.convergence_analysis import run_convergence_analysis

convergence_results = run_convergence_analysis(
    config_file="config.yaml",
    panel_counts=[10, 20, 30, 40, 50],
    angle_of_attack=8.0,
    velocity=15.0
)
```

### 3. Sensitivity Analysis

```python
from VSM.sensitivity_analysis import analyze_parameter_sensitivity

sensitivity_results = analyze_parameter_sensitivity(
    config_file="config.yaml",
    parameters_to_vary={
        'angle_of_attack': [6, 8, 10],
        'velocity': [10, 15, 20]
    }
)
```

## Coordinate System Conventions

### Global Reference Frame
- **x-axis**: Forward (aligned with velocity in symmetric flight)
- **y-axis**: Spanwise (right wing positive)
- **z-axis**: Upward (vertical)

### Aerodynamic Angles
- **Angle of Attack (α)**: Positive nose-up
- **Sideslip (β)**: Positive for starboard-to-port flow (right-to-left)

### Body Rotation Rates
- **Roll rate (p)**: About x-axis, positive right wing down
- **Pitch rate (q)**: About y-axis, positive nose-up
- **Yaw rate (r)**: About z-axis, positive nose-right

### Sign Conventions
- **Lift**: Positive upward (in -z direction in body frame)
- **Drag**: Positive opposing motion
- **Side force**: Positive to starboard (right)
- **Moments**: Right-hand rule about respective axes

## Performance Optimization Tips

1. **Panel Count**: Start with 20-30 panels, increase only if needed
2. **Panel Distribution**: Use "cosine" for better tip resolution
3. **Convergence**: Typical relaxation_factor: 0.01-0.05
4. **Cache ML Models**: Masure regression caches models automatically
5. **Batch Processing**: Use `from_yaml_entry_batch()` for multiple airfoils

## Troubleshooting

### Common Issues

**Solver not converging:**
- Reduce `relaxation_factor` (try 0.001-0.01)
- Increase `max_iterations`
- Check for numerical issues in geometry
- Try different `gamma_initial_distribution_type`

**Unrealistic forces:**
- Verify YAML geometry (check LE/TE coordinates)
- Check airfoil polar data quality
- Ensure appropriate `core_radius_fraction` (1e-6 to 1e-2)
- Verify reference_point location

**ImportError for optional dependencies:**
- NeuralFoil: `pip install neuralfoil`
- Masure regression: `pip install scikit-learn`
- Ensure model files exist in `ml_models_dir`

## Version Information

This documentation corresponds to the current development version of VSM.
For specific version information, check the repository tags and release notes.

## Contributing

When adding new features:
1. Follow the style guide (style_guide.md)
2. Add comprehensive docstrings
3. Update relevant documentation files
4. Add examples to the examples/ directory
5. Include unit tests in tests/

## References

1. Katz, J., & Plotkin, A. (2001). *Low-Speed Aerodynamics*. Cambridge University Press.
2. Anderson, J. D. (2011). *Fundamentals of Aerodynamics*. McGraw-Hill.
3. Damiani, R., et al. (2015). *A vortex step method for nonlinear airfoil polar data*.
4. Gaunaa, M., et al. (2024). *3D viscous drag correction methods*.

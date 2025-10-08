# Vortex Step Method
The Vortex Step Method (VSM) is an enhanced lifting line method that improves upon the classic approach by solving the circulation system at the three-quarter chord position. This adjustment allows for more accurate calculations of lift and drag forces, particularly addressing the shortcomings in induced drag prediction. VSM is further refined by coupling it with 2D viscous airfoil polars, making it well-suited for complex geometries, including low aspect ratio wings, as well as configurations with sweep, dihedral, and anhedral angles, typical for leading-edge inflatable (LEI) kites, that are used in airborne wind energy production, boat-towing and kite-surfing. An open-source example kite is the [TU Delft V3 Kite](https://awegroup.github.io/TUDELFT_V3_KITE/), of which a video is shown below, made using the internal plotting library.

![](docs/TUDELFT_V3_KITE_plotly.gif)

The software presented here includes examples for: a rectangular wing and a leading-edge inflatable kite.

A Julia version of this project is available at [VortexStepMethod.jl](https://github.com/Albatross-Kite-Transport/VortexStepMethod.jl)

## Key Features

- **Accurate low-aspect-ratio wing modeling** with enhanced lifting line theory
- **Viscous-inviscid coupling** using 2D airfoil polars (inviscid, CFD, ML-based)
- **Complex geometry support**: sweep, dihedral, anhedral, leading-edge inflatable (LEI) kites
- **Rigid-body stability derivatives**: automatic computation of dCx/dα, dCMy/dq, etc.
- **Trim angle solver**: automatic determination of trimmed angle of attack
- **Non-dimensional rate derivatives**: controls-friendly output (hat_p, hat_q, hat_r)
- **Reference point flexibility**: correctly handles moment reference point for rotational velocities
- **Interactive visualization**: Plotly and Matplotlib geometry and results plotting

## Documentation
For detailed documentation, please refer to the following resources.

**Explanatory Notes**
- [Aerodynamic Model](docs/Aerodynamic_model.pdf)
- [Paper: Fast Aero-Structural Model of a Leading-Edge Inflatable Kite](https://doi.org/10.3390/en16073061)

**Code Core**
- [Airfoil Aerodynamics](docs/AirfoilAerodynamics.md)
- [Body Aerodynamics](docs/BodyAerodynamics.md)
- [Filament](docs/Filament.md)
- [Panel](docs/Panel.md)
- [Solver](docs/Solver.md)
- [Wake](docs/Wake.md)
- [Wing Geometry](docs/WingGeometry.md)
- [Stability Derivatives](docs/StabilityDerivatives.md)
- [Trim Angle](docs/TrimAngle.md)

**Other**
- [Nomenclature](docs/nomenclature.md)
- [Style Guide](docs/style_guide.md)

## Installation Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/ocayon/Vortex-Step-Method
    ```

2. Navigate to the repository folder:
    ```bash
    cd Vortex-Step-Method
    ```
    
3. Create a virtual environment:
   
   Linux or Mac:
    ```bash
    python3 -m venv venv
    ```
    
    Windows:
    ```bash
    python -m venv venv
    ```
    
4. Activate the virtual environment:

   Linux or Mac:
    ```bash
    source venv/bin/activate
    ```

    Windows
    ```bash
    .\venv\Scripts\activate
    ```

5. Install the required dependencies:

   For users:
    ```bash
    pip install .
    ```
        
   For developers:
    ```bash
    pip install -e .[dev]
    ```
    
    For ubuntu add:
    ```
    pip install pyqt5
    sudo apt install cm-super
    sudo apt install dvipng
   ```

6. To deactivate the virtual environment:
    ```bash
    deactivate
    ```

## Quick Start

Here's a minimal example to get started:

```python
from pathlib import Path
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver

# Load kite geometry from YAML configuration
config_path = "data/TUDELFT_V3_KITE/CAD_derived_geometry/config_kite_CAD_CFD_polars.yaml"
body_aero = BodyAerodynamics.instantiate(
    n_panels=30,
    file_path=config_path,
    spanwise_panel_distribution="uniform"
)

# Set flow conditions
body_aero.va_initialize(
    Umag=10.0,              # Velocity magnitude [m/s]
    angle_of_attack=6.0,    # Angle of attack [deg]
    side_slip=0.0,          # Sideslip angle [deg]
)

# Solve and get results
solver = Solver()
results = solver.solve(body_aero)

print(f"CL = {results['cl']:.3f}")
print(f"CD = {results['cd']:.3f}")
print(f"L/D = {results['cl']/results['cd']:.2f}")
```

For more examples, see the `examples/` directory.

## Dependencies

- numpy - Numerical computing
- matplotlib - 2D plotting
- scipy - Scientific computing and optimization
- plotly - Interactive 3D visualization
- pandas - Data manipulation
- neuralfoil - Neural network airfoil predictions
- PyYAML - Configuration file parsing
- scikit-learn - Machine learning utilities
- numba - Just-in-time compilation for performance-critical loops
- screeninfo - Display information for plotting

See also [pyproject.toml](pyproject.toml) for complete dependency list and version requirements


**Machine Learning**

The code base is adapted to work with a machine learning model trained on more than a hundred thousands Reynolds-average Navier Stokes (RANS) Computational Fluid Dynamics (CFD) simulations made for leading-edge inflatable airfoils, documented in the MSc. thesis of [K.R.G. Masure](https://resolver.tudelft.nl/uuid:865d59fc-ccff-462e-9bac-e81725f1c0c9), the [code base is also open-source accessible](https://github.com/awegroup/Pointwise-Openfoam-toolchain).

As the three trained models, for Reynolds number = 1e6, 5e6 and 1e7 are too large (~2.3GB) for GitHub, they have to be downloaded separately, and added to the `data/ml_models` folder. They are accessible through [Zenodo](https://doi.org/10.5281/zenodo.16925758), and so is the [CFD data](https://doi.org/10.5281/zenodo.16925833) on which the models are trained. More description on its usage is found in [Airfoil Aerodynamics](docs/AirfoilAerodynamics.md).

## Usage Examples

The `examples/` folder contains comprehensive tutorials:

### **Rectangular Wing**
- `rectangular_wing/tutorial.py` - Basic wing analysis workflow

### **TU Delft V3 Kite**
- `tutorial.py` - Complete kite aerodynamic analysis
- `evaluate_stability_derivatives.py` - Stability and control derivatives computation
- `tow_angle_geometry.py` - Geometric tow angle and center of pressure analysis
- `tow_point_location_parametric_study.py` - Design space exploration for tow point
- `kite_stability_dynamics.py` - Natural frequency and oscillation period calculation
- `convergence.py` - Panel count convergence study
- `benchmark.py` - Performance benchmarking

### **Machine Learning for LEI Airfoils**
- `machine_learning_for_lei_airfoils/tutorial.py` - Using ML models for airfoil aerodynamics

See individual files for detailed documentation and usage instructions.

## Performance Optimization

For large-scale parametric studies:

1. **Use fewer panels during exploration** (`n_panels=20-30`), then increase for final results
2. **Numba JIT compilation** automatically accelerates vortex calculations (first run slower)
3. **Parallel studies**: Run multiple configurations using `multiprocessing` or `joblib`
4. **Cache geometries**: Instantiate `BodyAerodynamics` once, reuse with different flow conditions

Example:
```python
# Fast exploration
body_aero = BodyAerodynamics.instantiate(n_panels=20, ...)  # ~0.1s per solve

# High accuracy
body_aero = BodyAerodynamics.instantiate(n_panels=100, ...) # ~2s per solve
```

## Troubleshooting

### Common Issues

**Import errors with numba:**
```bash
pip install --upgrade numba
```

**Matplotlib backend issues on Linux:**
```bash
export MPLBACKEND=TkAgg  # Or 'Qt5Agg' if PyQt5 installed
```

**Missing ML models:**
Download from [Zenodo](https://doi.org/10.5281/zenodo.16925758) and place in `data/ml_models/`

**Plotly not showing in Jupyter:**
```bash
pip install jupyterlab "ipywidgets>=7.5"
```

For more issues, check the [GitHub Issues](https://github.com/ocayon/Vortex-Step-Method/issues) page.

## Contributing Guide
Please report issues and create pull requests using the URL:
```
https://github.com/ocayon/Vortex-Step-Method/
```

This is required because you cannot/should not do it using the URL
```
https://github.com/awegroup/Vortex-Step-Method
```

We welcome contributions to this project! Whether you're reporting a bug, suggesting a feature, or writing code, here’s how you can contribute:

1. **Create an issue** on GitHub
2. **Create a branch** from this issue
   ```bash
   git checkout -b issue_number-new-feature
   ```
3. Implement your new feature
4. Verify nothing broke using **pytest**
```
  pytest
```
5. **Commit your changes** with a descriptive message
```
  git commit -m "#<number> <message>"
```
6. **Push your changes** to the github repo:
   git push origin branch-name
   
7. **Create a pull-request**, with `base:develop`, to merge this feature branch
8. Once the pull request has been accepted, **close the issue**

## Citation
If you use this project in your research, please consider citing it. 
Citation details can be found in the [CITATION.cff](CITATION.cff) file included in this repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## WAIVER
Technische Universiteit Delft hereby disclaims all copyright interest in the package written by the Author(s).
Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering

### Copyright
Copyright (c) 2022 Oriol Cayon

Copyright (c) 2024 Oriol Cayon, Jelle Poland, TU Delft

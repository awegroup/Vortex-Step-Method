# Vortex Step Method
The Vortex Step Method (VSM) is an enhanced lifting line method that improves upon the classic approach by solving the circulation system at the three-quarter chord position, among the most important details. This adjustment allows for more accurate calculations of lift and drag forces, particularly addressing the shortcomings in induced drag prediction. VSM is further refined by coupling it with 2D viscous airfoil polars, making it well-suited for complex geometries, including low aspect ratio wings, as well as configurations with sweep, dihedral, and anhedral angles.

The software presented here includes a couple of examples: a rectangular wing and a leading-edge inflatable kite.

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
    
5. Activate the virtual environment:

   Linux or Mac:
    ```bash
    source venv/bin/activate
    ```

    Windows
    ```bash
    .\venv\Scripts\activate
    ```

6. Install the required dependencies:

   For users:
    ```bash
    pip install .
    ```
        
   For developers:
    ```bash
    pip install -e .[dev]
    ```

7. To deactivate the virtual environment:
    ```bash
    deactivate
    ```
### Dependencies
- numpy
- matplotlib>=3.7.1
- seaborn
- scipy
- numba
- ipykernel
- screeninfo

## Usages
Please look at the tutorial on a rectangular wing, where the code usage and settings are fully detailed.
You can find it in `examples/rectangular_wing/tutorial_rectangular_wing.ipynb`

Another tutorial is present under `examples/TUDELFT_V3_LEI_KITE/tutorial_testing_stall_model.py` where a geometry is loaded from .csv, plotted, distributions are plotted and polars are created to demonstrate the effect of the stall model.

For more detailed information one is referred to:
- [Aerodynamic model](docs/Aerodynamic_model.pdf)
- [Paper: Fast Aero-Structural Model of a Leading-Edge Inflatable Kite](https://doi.org/10.3390/en16073061) 

## Contributing Guide
We welcome contributions to this project! Whether you're reporting a bug, suggesting a feature, or writing code, here’s how you can contribute:

1. **Create an issue** on GitHub
2. **Create a branch** from this issue
   ```bash
   git checkout -b issue_number-new-feature
   ```
3. --- Implement your new feature---
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

### Copyright
Copyright (c) 2022 Oriol Cayon

Copyright (c) 2024 Oriol Cayon, Jelle Poland, TU Delft

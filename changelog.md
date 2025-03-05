# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] - 2025-03-03

# Unstructured

# dealing with Simonet Stall Model
- [x] plotting.py: inside the generate_3D_polar_data, 
feeding in gamma_distribution `results = solver.solve(body_aero,gamma_distrbution=gamma)`, also changed the way the legends are plotted: is now below the graphs in distribution and polars.
- [x] BodyAerodynamics, changed `calculate_circulation_distribution_elliptical_wing` and added `calculate_circulation_distribution_cosine`
- [x] Solver, changed _init_ statement, added artificial viscosity as property and setter, change gamma_initialisation inside solver function, added the gamma_loop types, and added the non_linear gamma loop functions. Also updated the artificial viscosity stall model, changed elliptic to elliptical, rewriting some of the variables inside solver, to become self. to make them callable from anywhere within the class, added min_relaxation_error, adding a compute_aerodynamic_quantities function

-Panel: added panel_polar_data property, added "cl_is_pisinalpha" to the _panel_aero_model options

Added a test function for the elliptical_wing_planform, from ch 4.1 of Simonet




### Added
- **Interactive Visualization:** Introduced an interactive Plotly environment for data visualization.
- **Moment Calculations:** Added moment calculations and plotting functionality (#89).
- **Panel Polar Investigation:** Implemented a panel polar investigation function (#92).
- **User Configuration:** Enabled support for a user-defined reference point in moment calculations.
- **NeuralFoil Integration:** Integrated NeuralFoil to verify and improve polar predictions.
- **Plot Enhancements:** Enhanced the interactive plot with additional parameters (span, height, chord) and decimal support.
- **Bridle Line Forces:** Added aerodynamic force functions for bridle lines (#105), supporting input in the form:
  - `bridle_line1 = [p1, p2, diameter]` where `p1 = [x, y, z]`
  - `bridle_lines = [bridle_line1, bridle_line2, ...]`
- **BodyAerodynamics Enhancements:** 
  - Renamed `WingAerodynamics` to `BodyAerodynamics`.
  - Added the `instantiate_body_aerodynamics` function.
  - Introduced a `from_file` class method for easier input handling (#115).

### Changed
- **Initialization Improvements:** Improved gamma initialization by removing redundant logic (#81).
- **Force Calculation:** Updated the side force calculation to use the cross product of lift and drag (#83).
- **Plot Output:** Modified plot outputs to address issues (e.g., issue #85) and improve clarity.
- **Interactive Environment:** Upgraded the interactive plotting environment.
- **Moment Calculation:** Revised the moment calculation method and updated tutorial documentation accordingly.
- **Polar Handling:** Enhanced polar data handling by:
  - Incorporating a new `va_initialize` function.
  - Modifying the Panel class to support an (N,4) polar input format.
- **Aerodynamic Vectors:** Normalized aerodynamic vectors for improved consistency (#91).
- **Convergence & Sensitivity:** Refined convergence studies on the number of panels (n_panels) and added sensitivity plots.
- **Damping & Smoothing:** Improved artificial damping and further enhanced the smoothing algorithm.
- **Polar Engineering:** Updated the polar_engineering module to use Surfplan profiles and adjusted POLAR functionality along with its pytests.
- **Documentation:** Applied several minor documentation and tutorial updates to reflect these changes.

### Fixed
- **Polar Data Input:** Resolved issues related to polar-data input.
- **Reference Input:** Corrected the reference point input handling.
- **Testing:** Fixed failing tests within the Panel module.

### Removed
- **Stall Models:** Moved stall models to a dedicated stall branch.
- **Polar Engineering Branch:** Relocated polar_engineering efforts to its own branch.
- **Redundant Variables:** Removed `tol_reference_error` as it was no longer necessary.


## [1.0.0] - 2024-09-29

### Added 
- Refactored code to object oriented. 
  - Introduced the main VSM (Vortex Step Method) parent class.
  - Added a dedicated Panel Properties class.
  - Set up an initial Horseshoe class and later introduced child classes.
  - Defined global functions supporting each of the classes.
  - Added a Filaments class and a Wake class.
  - Introduced an Abstract class for 2D infinite filaments.
- **Solver & Geometry Enhancements:**
  - Re-structured the code skeleton and overhauled the solver with all necessary methods.
  - Added new functions to support curved wing, swept wing, and elliptical wing geometries.
  - Implemented a calculate_results function based on earlier thesis code and refined it.
- **Visualization and Plotting:**
  - Added functions for plotting wing geometry, distributions, and polar plots.
  - Updated examples (e.g., rectangular wing case, V3 example folder with CFD/WindTunne interfaces).
- **Testing & Documentation:**
  - Integrated initial pytests across modules (Panel, Filaments, etc.), coverage 66%
  - Populated configuration files like `citation.cff`, `.gitignore`, and `pyproject.toml` for improved packaging.
  - Added user instructions, docstrings for each function, and updated tutorial content.
- **Performance Optimizations:**
  - Introduced significant speed improvements by refactoring loops (e.g., removing nested loops and moving functions outside loops) and optimizing AIC matrix computations (#50).
- **Additional Functionalities:**
  - Added a yaw rate implementation with an accompanying V3 kite example.
  - Implemented polar data interpolation for sections with different datasets.

### Changed
- **Code Refactoring:**
  - Renamed and refactored class names and variable identifiers across VSM, Panel, and WingAerodynamics.
  - Removed redundant properties from WingAerodynamics, now accessing values directly from the Panel.
- **Solver & Algorithm Improvements:**
  - Updated the global functions and solver structure to improve convergence and reliability.
  - Refined the handling of gamma distributions, reference system definitions, and spacing (including cosine van Garrel spacing for elliptical wings, #24).
  - Revised the induced_velocity function for better accuracy (#21).
- **Documentation & User Interface:**
  - Enhanced the README with additional images, detailed tutorial updates, and more user-friendly instructions.
  - Improved code comments and added TODO markers to guide future development.
- **Robustness & Configuration:**
  - Made the code robust against unordered or right-to-left section inputs.
  - Removed hardcoded values and replaced them with parameterized configurations.

### Fixed
- **Bug Resolutions:**
  - Fixed various issues in solver functionality, geometry processing, and test cases.
  - Corrected the induced_velocity function and core-radius handling in pytests (#21).
  - Adjusted the narrow angle-of-attack (aoa) range in aerodynamic data inputs (#42).
  - Resolved alignment issues in V3 comparisons and output formatting.
  - Fixed bugs in the stall model (ensuring the stall angle is calculated per plate) and revised related test scripts (#29, #68).
  - Addressed wake modeling issues and enhanced logging for error tracking.
- **Testing and Stability:**
  - Resolved discrepancies in AIC matrix calculations and convergence tests.
  - Fixed issues with PDF/PNG outputs in test scripts (#61).
- **Performance Corrections:**
  - Eliminated redundant calculations and optimized nested loops to achieve faster runtime.
  - Resolved pytesting issues related to numba and streamlined optimization routines.

### Removed
- **Deprecated and Redundant Code:**
  - Removed the older Horseshoe class in favor of the new implementation.
  - Eliminated unnecessary attributes from WingAerodynamics and cleaned out redundant code sections.
  - Removed legacy code and virtual environment folders from the project directory.
  - Cleared outdated TODOs and extraneous comments to streamline the codebase.

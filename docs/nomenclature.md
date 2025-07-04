# Nomenclature

## Core Classes

### Primary Components
- **`Panel`**: Represents a discrete wing panel bounded by two wing sections. Contains geometric properties, aerodynamic characteristics, and vortex filament system for computing induced velocities and forces.
- **`Filament`**: Abstract base class for vortex filaments. Includes `BoundFilament` (finite vortex along quarter-chord) and `SemiInfiniteFilament` (trailing vortex extending to infinity).
- **`Wake`**: Static factory class managing trailing vortex wake systems. Handles creation and updates of semi-infinite vortex filaments behind wing panels.
- **`AirfoilAerodynamics`**: Factory class for generating 2D airfoil polar data from various sources (Breukels regression, NeuralFoil, CSV files, inviscid theory).
- **`BodyAerodynamics`**: Main orchestrating class combining wing geometry, panel generation, and flow calculations for complete aerodynamic analysis.
- **`Solver`**: Implements iterative algorithms to determine circulation distribution satisfying boundary conditions for VSM and LLT models.
- **`Wing`**: Represents complete wing geometry composed of multiple sections with configurable panel distributions and mesh refinement.
- **`Section`**: Individual wing section with leading edge, trailing edge coordinates and aerodynamic polar data.

## Geometric Properties

### Coordinate Systems
- **`x_airf`**: Normal unit vector pointing upward from chord line, perpendicular to panel surface (used for angle of attack calculations)
- **`y_airf`**: Chordwise unit vector pointing from leading edge to trailing edge, parallel to local chord line (defines drag direction)
- **`z_airf`**: Spanwise unit vector pointing toward left wing tip, in airfoil plane perpendicular to chord (defines side force direction)

### Reference Points
- **`aerodynamic_center`**: Panel aerodynamic center, typically at 1/4 chord (quarter-chord point)
- **`control_point`**: Panel control point, typically at 3/4 chord (three-quarter-chord point)
- **`bound_point_1/2`**: Endpoints of bound vortex filament along quarter-chord line
- **`LE_point`**: Leading edge coordinates [x, y, z]
- **`TE_point`**: Trailing edge coordinates [x, y, z]
- **`corner_points`**: Quadrilateral panel vertices [LE_point_1, TE_point_1, TE_point_2, LE_point_2]

### Dimensional Properties
- **`chord`**: Average chord length of panel (distance from leading to trailing edge)
- **`chord_vector`**: Vector from leading edge to trailing edge (TE - LE)
- **`chord_length`**: Magnitude of chord vector
- **`width`**: Panel width at bound vortex location (spanwise distance)
- **`span`**: Wing span along specified spanwise direction
- **`projected_area`**: Wing area projected onto specified plane

## Aerodynamic Variables

### Flow Conditions
- **`va`**: Apparent velocity vector (relative wind velocity including induced effects)
- **`va_norm`**: Magnitude of apparent velocity vector
- **`va_unit`**: Unit vector in apparent velocity direction
- **`Umag`**: Effective velocity magnitude (cross product with spanwise direction)
- **`Uinf`**: Inflow velocity magnitude (freestream conditions)
- **`alpha`**: Angle of attack in radians (effective local angle including induced effects)
- **`angle_of_attack`**: Geometric angle of attack of wing relative to freestream
- **`side_slip`**: Sideslip angle in aerodynamic analysis
- **`yaw_rate`**: Angular velocity about vertical axis

### Circulation and Forces
- **`gamma`**: Circulation strength around vortex filament (bound circulation)
- **`gamma_distribution`**: Array of circulation values along wing span
- **`gamma_initial`**: Initial guess for circulation distribution in iterative solver
- **`gamma_new`**: Updated circulation distribution during iteration

### Force Coefficients
- **`cl`**: Lift coefficient from airfoil polar data
- **`cd`**: Drag coefficient from airfoil polar data  
- **`cm`**: Pitching moment coefficient from airfoil polar data
- **`polar_data`**: Airfoil aerodynamic data array with columns [α, CL, CD, CM]

### Induced Velocities
- **`induced_velocity`**: Velocity induced by vortex system at evaluation point
- **`relative_velocity`**: Total velocity (apparent + induced) at panel
- **`v_normal`**: Velocity component normal to chord line (for angle of attack)
- **`v_tangential`**: Velocity component tangential to chord line

## Vortex System Components

### Filament Types
- **`BoundFilament`**: Finite vortex filament between two points (quarter-chord bound vortex)
- **`SemiInfiniteFilament`**: Semi-infinite vortex extending from trailing edge to infinity
- **`filaments`**: List of vortex filament objects comprising panel's vortex system

### Filament Properties
- **`x1, x2`**: Start and end points of finite filament
- **`filament_direction`**: Direction multiplier (±1) for semi-infinite filaments
- **`direction`**: Unit vector of wake/filament direction
- **`vel_mag`**: Velocity magnitude associated with filament

### Core Radius Parameters
- **`core_radius_fraction`**: Vortex core radius as fraction of filament length
- **`epsilon`**: Vortex core radius for regularization (prevents singularities)
- **`_alpha0`**: Oseen parameter for viscous diffusion (1.25643)
- **`_nu`**: Kinematic viscosity of air (1.48e-5 m²/s)

## Solver Parameters

### Convergence Control
- **`max_iterations`**: Maximum number of solver iterations
- **`allowed_error`**: Convergence tolerance for normalized error
- **`relaxation_factor`**: Under-relaxation factor for iterative stability
- **`converged`**: Boolean indicating successful convergence

### Algorithm Types
- **`aerodynamic_model_type`**: Analysis method ('VSM' for Vortex Step Method, 'LLT' for Lifting Line Theory)
- **`gamma_loop_type`**: Iterative algorithm ('base', 'non_linear', 'simonet_stall')
- **`gamma_initial_distribution_type`**: Initial circulation distribution ('elliptical', 'cosine', 'zero', 'previous')

### Stall Modeling
- **`is_smooth_circulation`**: Enable circulation distribution smoothing
- **`smoothness_factor`**: Smoothing strength parameter
- **`is_artificial_damping`**: Enable artificial damping for stall conditions
- **`artificial_damping`**: Dictionary with damping coefficients {'k2': value, 'k4': value}
- **`is_with_simonet_artificial_viscosity`**: Enable Simonet stall model

## Matrix Systems

### Aerodynamic Influence Coefficients
- **`AIC_x, AIC_y, AIC_z`**: Aerodynamic Influence Coefficient matrices for x, y, z velocity components
- **`evaluation_point`**: Point where induced velocity is computed
- **`evaluation_point_on_bound`**: Boolean flag (True for LLT, False for VSM treatment)

## Mesh Properties

### Panel Distribution
- **`n_panels`**: Number of discretized panels in aerodynamic mesh
- **`spanwise_panel_distribution`**: Panel spacing strategy ('uniform', 'cosine', 'split_provided', 'unchanged')
- **`sections`**: List of wing section objects defining geometry
- **`n_sections`**: Number of wing sections (n_panels + 1)

### Interpolation Parameters
- **`left_weight, right_weight`**: Interpolation weights between adjacent sections
- **`section_index`**: Index for section-based interpolation
- **`target_length`**: Target position along quarter-chord line for mesh refinement

## Physical Constants

### Fluid Properties
- **`rho`**: Fluid density (kg/m³)
- **`mu`**: Dynamic viscosity (Pa·s)
- **`reynolds`**: Reynolds number for aerodynamic analysis

### Reference Values
- **`reference_point`**: Reference point for moment calculations [x, y, z]
- **`spanwise_direction`**: Wing spanwise unit vector (default [0, 1, 0])

## Airfoil Data Sources

### Data Types
- **`"breukels_regression"`**: LEI kite airfoil correlation model with thickness and camber parameters
- **`"neuralfoil"`**: Neural network-based airfoil analysis using .dat geometry files
- **`"polars"`**: Direct import of pre-computed CSV polar data
- **`"inviscid"`**: Theoretical thin airfoil approximation (2π slope, zero drag)

### Parameters
- **`t`**: Airfoil thickness ratio for Breukels model
- **`kappa`**: Camber parameter for Breukels model
- **`dat_file_path`**: Path to airfoil geometry file for NeuralFoil
- **`polar_file_path`**: Path to CSV polar data file
- **`alpha_range`**: Angle of attack range [min, max, step] in degrees

## Output Variables

### Force Results
- **`F_x, F_y, F_z`**: Total force components in global coordinate system
- **`M_x, M_y, M_z`**: Total moment components about reference point
- **`CL, CD, CM`**: Total lift, drag, and moment coefficients

### Distributions
- **`force_distribution`**: Array of force vectors per panel
- **`moment_distribution`**: Array of moment vectors per panel
- **`alpha_array`**: Array of effective angles of attack per panel
- **`cl_array, cd_array`**: Arrays of section coefficients per panel

## Utility Functions

### Vector Operations (JIT-compiled)
- **`jit_cross(a, b)`**: Optimized cross product computation
- **`jit_norm(value)`**: Optimized vector norm calculation
- **`jit_dot(a, b)`**: Optimized dot product computation

### Geometric Functions
- **`intersect_line_with_plane()`**: Line-plane intersection calculation
- **`point_in_triangle()`**: Point-in-triangle test with barycentric coordinates
- **`point_in_quad()`**: Point-in-quadrilateral containment test

## Acronyms and Abbreviations

### Methods
- **VSM**: Vortex Step Method (enhanced lifting line with 3/4 chord boundary conditions)
- **LLT**: Lifting Line Theory (classical Prandtl approach with 1/4 chord conditions)
- **AIC**: Aerodynamic Influence Coefficient (matrix relating circulations to induced velocities)

### Components
- **LE**: Leading Edge
- **TE**: Trailing Edge  
- **AC**: Aerodynamic Center (1/4 chord)
- **CP**: Control Point (3/4 chord for VSM)
- **QC**: Quarter Chord

### Physical Models
- **LEI**: Leading Edge Inflatable (kite airfoil type)
- **2D**: Two-dimensional (airfoil section analysis)
- **3D**: Three-dimensional (finite wing analysis)

## Variable Naming Conventions

### Suffixes
- **`_array`**: Vector/array of values (one per panel)
- **`_distribution`**: Spanwise distribution of scalar or vector quantities
- **`_norm`**: Magnitude/norm of vector quantity
- **`_unit`**: Unit vector (normalized direction)
- **`_initial`**: Initial value for iterative process
- **`_new`**: Updated value during iteration

### Prefixes
- **`is_`**: Boolean flag parameters
- **`n_`**: Count/number of items
- **`_`**: Private/internal class attributes


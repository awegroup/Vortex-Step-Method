# AirfoilAerodynamics Module Documentation

## Overview

The `AirfoilAerodynamics` class provides a factory interface for generating 2D airfoil aerodynamic polar data from various sources. It encapsulates airfoil coefficients (CL, CD, CM) as functions of angle of attack and supports multiple data generation methods.

**Supported Methods:**
- `breukels_regression`: LEI kite airfoil correlation model
- `neuralfoil`: Neural network-based analysis from geometry files
- `polars`: Direct import from CSV polar data
- `inviscid`: Theoretical thin airfoil approximation
- `masure_regression`: Machine learning predictions using trained models

## Class: AirfoilAerodynamics

### Instantiation

**Important**: Do not use the constructor directly. Always use the factory method:

```python
aero = AirfoilAerodynamics.from_yaml_entry(airfoil_type, airfoil_params, ...)
```

### Factory Method

#### `from_yaml_entry(airfoil_type, airfoil_params, alpha_range=None, reynolds=None, file_path=None)`

Creates an AirfoilAerodynamics instance from configuration parameters.

**Parameters:**
- `airfoil_type` (str): Type of airfoil model to use
- `airfoil_params` (dict): Parameters specific to the airfoil type
- `alpha_range` (list): [min_alpha, max_alpha, step] in degrees (optional)
- `reynolds` (float): Reynolds number for analysis (optional)
- `file_path` (str): Base path for relative file references (optional)

**Returns:** AirfoilAerodynamics instance with populated polar data

## Supported Airfoil Types

### 1. Breukels Regression (`"breukels_regression"`)

LEI (Leading Edge Inflatable) kite airfoil correlation model based on Breukels (2011).

**Required Parameters:**
- `t` (float): Thickness ratio (typically 0.08-0.20)
- `kappa` (float): Camber parameter (typically 0.04-0.12)

**Example:**
```python
aero = AirfoilAerodynamics.from_yaml_entry(
    "breukels_regression",
    {"t": 0.12, "kappa": 0.08},
    alpha_range=[-10, 20, 1]
)
```

**Features:**
- Polynomial coefficients for CL, CD, CM
- Built-in stall logic for |α| > 20°
- Optimized for LEI kite airfoils

### 2. NeuralFoil (`"neuralfoil"`)

Neural network-based airfoil analysis using external .dat geometry files.

**Required Parameters:**
- `dat_file_path` (str): Path to airfoil geometry file (relative to file_path)

**Optional Parameters:**
- `model_size` (str): Neural network size ("xxxlarge" default)
- `xtr_lower` (float): Lower transition location (0.01 default)
- `xtr_upper` (float): Upper transition location (0.01 default) 
- `n_crit` (float): Critical amplification factor (9 default)

**Example:**
```python
aero = AirfoilAerodynamics.from_yaml_entry(
    "neuralfoil",
    {
        "dat_file_path": "airfoils/naca0012.dat",
        "model_size": "xxxlarge",
        "n_crit": 9
    },
    alpha_range=[-15, 25, 1],
    reynolds=500000,
    file_path="/path/to/config"
)
```

### 3. Polar Data (`"polars"`)

Direct import of pre-computed polar data from CSV files.

**Required Parameters:**
- `polar_file_path` (str): Path to CSV file with polar data

**CSV Format:**
- Columns: `alpha` (radians), `CL`, `CD`, `CM`
- No header row required

**Example:**
```python
aero = AirfoilAerodynamics.from_yaml_entry(
    "polars",
    {"polar_file_path": "polars/custom_airfoil.csv"},
    file_path="/path/to/config"
)
```

### 4. Inviscid (`"inviscid"`)

Theoretical thin airfoil approximation for preliminary analysis.

**No Parameters Required**

**Characteristics:**
- CL = 2π × α (linear lift slope)
- CD = 0 (no viscous drag)
- CM = 0 (no pitching moment)

**Example:**
```python
aero = AirfoilAerodynamics.from_yaml_entry(
    "inviscid",
    {},
    alpha_range=[-10, 15, 1]
)
```

### 5. Masure Regression (`"masure_regression"`)

Machine learning-based airfoil prediction using trained scikit-learn models. This method uses Extra Trees regression models trained on a comprehensive dataset of airfoil geometries and their aerodynamic coefficients.

**Required Parameters:**
- `t` (float): Thickness parameter
- `eta` (float): Geometric parameter (cx)
- `kappa` (float): Geometric parameter (cy)
- `delta` (float): Geometric parameter (r)
- `lambda` (float): Leading edge parameter (LE)
- `phi` (float): Camber/trailing edge parameter (camTE)

**Dependencies:**
- Requires scikit-learn
- Requires trained model files (.pkl) in `data/models/` directory

**Model Files:**
- `ET_re5e6.pkl`: Model for Reynolds number 5×10⁶
- `ET_re1e6.pkl`: Model for Reynolds number 1×10⁶
- `ET_re2e7.pkl`: Model for Reynolds number 2×10⁷

**Example:**
```python
aero = AirfoilAerodynamics.from_yaml_entry(
    "masure_regression",
    {
        "t": 0.07,
        "eta": 0.2,
        "kappa": 0.95,
        "delta": -2,
        "lambda": 0.65,
        "phi": 0.25
    },
    alpha_range=[-10, 25, 1],
    reynolds=1e6,
    file_path="/path/to/config"
)
```

**Features:**
- High accuracy predictions based on geometric parameters
- Supports multiple Reynolds numbers
- Includes compatibility patches for different scikit-learn versions
- Fast prediction once models are loaded

**Model Installation:**
The required .pkl model files are not included in the repository due to size constraints. They must be installed separately in the `data/models/` directory. See `data/models/README.md` for installation instructions.

## Key Methods

### `to_polar_array()`

Converts the airfoil data to a standardized numpy array format.

**Returns:** 
- `np.ndarray`: Shape (N, 4) with columns [alpha, CL, CD, CM]
- Alpha values in radians
- Suitable for direct use in Panel objects

### Private Methods

#### `_from_breukels_regression(airfoil_params, alpha_range)`
Implements the Breukels correlation model with polynomial coefficients.

#### `_from_neuralfoil(airfoil_params, alpha_range, reynolds, file_path)`
Interfaces with NeuralFoil for high-fidelity predictions.

#### `_from_polars(airfoil_params, alpha_range, file_path)`
Loads and processes CSV polar data files.

#### `_from_inviscid(alpha_range)`
Generates theoretical inviscid polar data.

#### `_from_masure_regression(airfoil_params, alpha_range, reynolds, file_path)`
Implements the Masure regression model using trained machine learning models.

#### `_load_regression_model(reynolds, file_path)`
Loads the appropriate scikit-learn model based on Reynolds number.

#### `_patch_sklearn_compatibility(model)`
Applies compatibility patches to handle different scikit-learn versions.

#### `_predict_aerodynamics(X_input, reynolds, file_path)`
Predicts aerodynamic coefficients using the loaded regression model.

#### `_instantiate_lei_airfoil_breukels_cl_cd_cm_coefficients(t, kappa)`
Computes polynomial coefficients for the Breukels model.

## Usage in VSM Framework

### Integration with Wing Sections

```python
# In YAML configuration
wing_airfoils:
  alpha_range: [-10, 25, 1]
  reynolds: 500000
  headers: [airfoil_id, type, info_dict]
  data:
    - [root, breukels_regression, {t: 0.15, kappa: 0.10}]
    - [tip, breukels_regression, {t: 0.10, kappa: 0.06}]
    - [mid, masure_regression, {t: 0.07, eta: 0.2, kappa: 0.95, delta: -2, lambda: 0.65, phi: 0.25}]
```

### Direct Usage

```python
# Create airfoil data with Breukels regression
aero = AirfoilAerodynamics.from_yaml_entry(
    "breukels_regression",
    {"t": 0.12, "kappa": 0.08},
    alpha_range=[-10, 20, 1]
)

# Or create with Masure regression  
aero = AirfoilAerodynamics.from_yaml_entry(
    "masure_regression",
    {"t": 0.07, "eta": 0.2, "kappa": 0.95, "delta": -2, "lambda": 0.65, "phi": 0.25},
    alpha_range=[-10, 20, 1],
    reynolds=1e6,
    file_path="/path/to/config"
)

# Convert to array format
polar_data = aero.to_polar_array()

# Use in wing section
wing.add_section(LE_point, TE_point, polar_data)
```

## Data Format Standards

All airfoil data is standardized to the following format:

- **Alpha**: Angle of attack in radians
- **CL**: Lift coefficient (dimensionless)
- **CD**: Drag coefficient (dimensionless) 
- **CM**: Pitching moment coefficient (dimensionless, about quarter-chord)

## Error Handling

- **Missing files**: FileNotFoundError for invalid paths
- **Invalid parameters**: ValueError for out-of-range values
- **Format errors**: ValueError for malformed CSV data
- **Import errors**: ImportError for missing dependencies (NeuralFoil)
- **Model files**: FileNotFoundError for missing .pkl model files (masure_regression)

## Performance Considerations

- **Breukels**: Fastest, analytical polynomials
- **Inviscid**: Very fast, simple theory
- **Polars**: Fast, pre-computed data
- **Masure Regression**: Fast (after initial model loading), high accuracy ML predictions
- **NeuralFoil**: Fast, high accuracy neural network predictions

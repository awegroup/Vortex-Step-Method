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

#### `from_yaml_entry(airfoil_type, airfoil_params, alpha_range=None, reynolds=None, file_path=None, ml_models_dir=None)`

Creates an AirfoilAerodynamics instance from configuration parameters.

**Parameters:**
- `airfoil_type` (str): Type of airfoil model to use
- `airfoil_params` (dict): Parameters specific to the airfoil type
- `alpha_range` (list): [min_alpha, max_alpha, step] in degrees (optional)
- `reynolds` (float): Reynolds number for analysis (optional)
- `file_path` (str): Base path for relative file references (optional, required for neuralfoil and polars)
- `ml_models_dir` (str): Base path for ML model files (optional, **required for masure_regression**)

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
- `file_path` (str): Base path for resolving relative file paths

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

**Example .dat file:**
If structure deviates from expected, an error will be raised.
```
0.0000, 0.0000
0.0010, 0.0015
0.0020, 0.0030
```


### 3. Polar Data (`"polars"`)

Direct import of pre-computed polar data from CSV files.

**Required Parameters:**
- `csv_file_path` (str): Path to CSV file with polar data
- `file_path` (str): Base path for resolving relative file paths

**CSV Format:**
- Columns: `alpha` (radians or degrees), `CL`, `CD`, `CM`
- No header row required

**Example:**
```python
aero = AirfoilAerodynamics.from_yaml_entry(
    "polars",
    {"csv_file_path": "polars/custom_airfoil.csv"},
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
- `ml_models_dir` (str): **Required** - Path to directory containing trained model files

**Dependencies:**
- Requires scikit-learn
- Requires trained model files (.pkl) in the specified `ml_models_dir`

**Model Files (must be present in `ml_models_dir`):**
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
    ml_models_dir="/path/to/ml_models"
)
```

**Features:**
- High accuracy predictions based on geometric parameters
- Supports multiple Reynolds numbers (5×10⁶, 1×10⁶, 2×10⁷)
- Includes compatibility patches for different scikit-learn versions
- Fast prediction once models are loaded
- Model caching for improved performance across multiple airfoils
- Batch processing optimization when multiple masure_regression airfoils are used

**Model Installation:**
The required .pkl model files must be installed in the directory specified by `ml_models_dir`. The directory structure should be:
```
ml_models_dir/
├── ET_re5e6.pkl
├── ET_re1e6.pkl
├── ET_re2e7.pkl
└── cache/  (created automatically for performance optimization)
```

**Error Handling:**
- Raises `ValueError` if `ml_models_dir` is not provided
- Raises `FileNotFoundError` if required model files are missing
- Raises `ValueError` if Reynolds number is not supported (must be 5e6, 1e6, or 2e7)
- Includes scikit-learn version compatibility handling

## Batch Processing

### `from_yaml_entry_batch(airfoil_ids, airfoil_types, airfoil_params_list, alpha_range=None, reynolds=None, file_path=None, ml_models_dir=None)`

Creates multiple AirfoilAerodynamics instances with batch optimization and caching for expensive computations (masure_regression and neuralfoil).

**Parameters:**
- `airfoil_ids` (list): List of airfoil identifiers
- `airfoil_types` (list): List of airfoil types for each ID
- `airfoil_params_list` (list): List of parameter dictionaries for each airfoil
- `alpha_range` (list): [min_alpha, max_alpha, step] in degrees (optional)
- `reynolds` (float): Reynolds number for analysis (optional)
- `file_path` (str): Base path for relative file references (optional)
- `ml_models_dir` (str): Base path for ML model files (optional, **required if any airfoil uses masure_regression**)

**Returns:** Dictionary mapping airfoil_id to polar data arrays

**Features:**
- Automatic caching of expensive computations
- Batch processing optimization for masure_regression and neuralfoil types
- Cache cleanup of old files
- Significant performance improvements for multiple airfoils

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

#### `_from_masure_regression(airfoil_params, alpha_range, reynolds, ml_models_dir)`
Implements the Masure regression model using trained machine learning models.

#### `_load_masure_regression_model(reynolds, ml_models_dir)`
Loads the appropriate scikit-learn model based on Reynolds number with caching.

#### `_patch_sklearn_compatibility(model)`
Applies compatibility patches to handle different scikit-learn versions.

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

# Create with Masure regression (requires ml_models_dir)
aero = AirfoilAerodynamics.from_yaml_entry(
    "masure_regression",
    {"t": 0.07, "eta": 0.2, "kappa": 0.95, "delta": -2, "lambda": 0.65, "phi": 0.25},
    alpha_range=[-10, 20, 1],
    reynolds=1e6,
    ml_models_dir="/path/to/ml_models"
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
- **Missing ml_models_dir**: ValueError when ml_models_dir is not provided for masure_regression

## Performance Considerations

- **Breukels**: Fastest, analytical polynomials
- **Inviscid**: Very fast, simple theory
- **Polars**: Fast, pre-computed data
- **Masure Regression**: Fast (after initial model loading), high accuracy ML predictions with caching and batch processing
- **NeuralFoil**: Fast, high accuracy neural network predictions with batch processing optimization

**Caching Benefits:**
- Automatic caching for expensive computations (masure_regression, neuralfoil)
- Significant performance improvements when processing multiple similar configurations
- Automatic cache cleanup of old files
- Cache directory created automatically in `ml_models_dir/cache/`

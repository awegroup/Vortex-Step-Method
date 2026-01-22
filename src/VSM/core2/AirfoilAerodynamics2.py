import numpy as np
import pandas as pd
from pathlib import Path
import neuralfoil as nf
import pickle
import warnings
import hashlib
import json
from datetime import datetime
import glob
import os


class AirfoilAerodynamics:
    """Class to encapsulate 2D airfoil aerodynamic data and interpolation.

    Use AirfoilAerodynamics.from_yaml_entry(...) to instantiate.
    """

    # Class-level cache for masure regression models
    _masure_model_cache = {}

    # Cache settings
    _cache_enabled = True
    _cacheable_types = {
        "masure_regression",
        "neuralfoil",
    }  # Only cache expensive computations

    def __init__(self):
        """Initialize AirfoilAerodynamics instance.

        Note:
            Do not use this constructor directly. Use from_yaml_entry() instead.

        Raises:
            RuntimeError: Always raised to prevent direct instantiation.
        """
        raise RuntimeError(
            "Use AirfoilAerodynamics.from_yaml_entry(...) to instantiate."
        )

    @classmethod
    def from_yaml_entry(
        cls,
        airfoil_type: str,
        airfoil_params: dict,
        alpha_range: list = None,
        reynolds: float = None,
        file_path: str = None,
        ml_models_dir: str = None,
    ):
        """Create AirfoilAerodynamics instance from configuration parameters.

        Args:
            airfoil_type (str): Type of airfoil model ('breukels_regression', 'neuralfoil', 'polars', 'inviscid', 'masure_regression').
            airfoil_params (dict): Parameters specific to the airfoil type.
            alpha_range (list, optional): [min_alpha, max_alpha, step] in degrees. Defaults to None.
            reynolds (float, optional): Reynolds number for analysis. Defaults to None.
            file_path (str, optional): Base path for relative file references. Defaults to None.
            ml_models_dir (str, optional): Base path for ML model files (required for masure_regression). Defaults to None.

        Returns:
            AirfoilAerodynamics: Instance with populated polar data.

        Raises:
            ValueError: If airfoil_type is not supported or required parameters are missing.
            Error: If .dat file not structured correctly: should contain "x, y" coordinates ONLY.
        """
        obj = object.__new__(AirfoilAerodynamics)
        obj.source = airfoil_type.lower()
        obj.alpha = None
        obj.CL = None
        obj.CD = None
        obj.CM = None

        if obj.source == "neuralfoil":
            obj._from_neuralfoil(airfoil_params, alpha_range, reynolds, file_path)
        elif obj.source == "breukels_regression":
            obj._from_breukels_regression(airfoil_params, alpha_range)
        elif obj.source == "polars":
            obj._from_polars(airfoil_params, alpha_range, file_path)
        elif obj.source == "inviscid":
            obj._from_inviscid(alpha_range)
        elif obj.source == "masure_regression":
            obj._from_masure_regression(
                airfoil_params, alpha_range, reynolds, ml_models_dir
            )
        else:
            raise ValueError(f"Unknown airfoil type: {airfoil_type}")

        return obj

    def _from_neuralfoil(self, airfoil_params, alpha_range, reynolds, file_path):
        """Generate polar data using NeuralFoil analysis.

        Args:
            airfoil_params (dict): Dictionary containing 'dat_file_path' and optional NeuralFoil parameters.
            alpha_range (list): [min_alpha, max_alpha, step] in degrees.
            reynolds (float): Reynolds number for analysis.
            file_path (str): Base path for resolving relative file paths.

        Returns:
            None: Populates self._polar_data.

        Raises:
            ImportError: If NeuralFoil is not installed.
            FileNotFoundError: If airfoil .dat file is not found.
        """
        if file_path is None:
            raise ValueError("file_path must be provided for airfoil type 'polars'.")
        file_path = Path(file_path)
        airfoil_params["dat_file_path"] = (
            file_path.parent / airfoil_params["dat_file_path"]
        )
        filename = airfoil_params["dat_file_path"]
        alpha = np.arange(
            alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]
        )
        aero = nf.get_aero_from_dat_file(
            filename=filename,
            alpha=alpha,
            Re=reynolds,
            model_size=airfoil_params.get("model_size", "xxxlarge"),
            xtr_lower=airfoil_params.get("xtr_lower", 0.01),
            xtr_upper=airfoil_params.get("xtr_upper", 0.01),
            n_crit=airfoil_params.get("n_crit", 9),
        )
        self.alpha = np.deg2rad(alpha)
        self.CL = aero["CL"]
        self.CD = aero["CD"]
        self.CM = aero["CM"]

    def _from_breukels_regression(self, airfoil_params: dict, alpha_range: list):
        """Generate polar data using Breukels regression model for LEI kite airfoils.

        Args:
            airfoil_params (dict): Dictionary containing 't' (thickness ratio) and 'kappa' (camber).
            alpha_range (list): [min_alpha, max_alpha, step] in degrees.

        Returns:
            None: Populates self._polar_data.
        """
        t = airfoil_params["t"]
        kappa = airfoil_params["kappa"]
        alpha_deg = np.arange(
            alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]
        )
        alpha_rad = np.deg2rad(alpha_deg)
        self._instantiate_lei_airfoil_breukels_cl_cd_cm_coefficients(t, kappa)
        self.alpha = alpha_rad
        CL = np.polyval(self._cl_coefficients, alpha_deg)
        CD = np.polyval(self._cd_coefficients, alpha_deg)
        CM = np.polyval(self._cm_coefficients, alpha_deg)
        # Stall logic to match LEI_airf_coeff
        stall_mask = (alpha_deg > 20) | (alpha_deg < -20)
        CL[stall_mask] = (
            2
            * np.cos(np.deg2rad(alpha_deg[stall_mask]))
            * np.sin(np.deg2rad(alpha_deg[stall_mask])) ** 2
        )
        CD[stall_mask] = 2 * np.sin(np.deg2rad(alpha_deg[stall_mask])) ** 3
        self.CL = CL
        self.CD = CD
        self.CM = CM

    def _from_polars(self, airfoil_params: dict, alpha_range: list, file_path: str):
        """Load polar data from CSV file.

        Args:
            airfoil_params (dict): Dictionary containing 'polar_file_path'.
            file_path (str): Base path for resolving relative file paths.

        Returns:
            None: Populates self._polar_data.

        Raises:
            FileNotFoundError: If polar CSV file is not found.
            ValueError: If CSV format is invalid.
        """
        if file_path is None:
            raise ValueError("file_path must be provided for airfoil type 'polars'.")
        file_path = Path(file_path)
        airfoil_params["csv_file_path"] = (
            file_path.parent / airfoil_params["csv_file_path"]
        )
        df = pd.read_csv(airfoil_params["csv_file_path"])
        if np.max(np.abs(df["alpha"])) > 2 * np.pi:
            alpha_orig = np.deg2rad(df["alpha"].values)
        else:
            alpha_orig = df["alpha"].values
        CL_orig = df[df.columns[df.columns.str.lower().str.contains("cl")][0]].values
        CD_orig = df[df.columns[df.columns.str.lower().str.contains("cd")][0]].values
        CM_orig = df[df.columns[df.columns.str.lower().str.contains("cm")][0]].values

        if alpha_range is not None:
            alpha_new = np.arange(
                alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]
            )
            alpha_new = np.deg2rad(alpha_new)
            self.alpha = alpha_new
            self.CL = np.interp(alpha_new, alpha_orig, CL_orig)
            self.CD = np.interp(alpha_new, alpha_orig, CD_orig)
            self.CM = np.interp(alpha_new, alpha_orig, CM_orig)
        else:
            self.alpha = alpha_orig
            self.CL = CL_orig
            self.CD = CD_orig
            self.CM = CM_orig

    def _from_inviscid(self, alpha_range: list):
        """Generate inviscid polar data using thin airfoil theory.

        Args:
            alpha_range (list): [min_alpha, max_alpha, step] in degrees.

        Returns:
            None: Populates self._polar_data with theoretical values.
        """
        alpha = np.arange(
            alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]
        )
        self.alpha = alpha
        self.CL = 2 * np.pi * alpha
        self.CD = np.zeros_like(alpha)
        self.CM = np.zeros_like(alpha)

    def _instantiate_lei_airfoil_breukels_cl_cd_cm_coefficients(self, t, kappa):
        """
        Instantiate the coefficients for the LEI airfoil Breukels regression model.
        This method computes the coefficients for lift (CL), drag (CD), and moment (CM)
        based on the given thickness (t) and camber (kappa) parameters.
        """

        # cl_coefficients
        C20 = -0.008011
        C21 = -0.000336
        C22 = 0.000992
        C23 = 0.013936
        C24 = -0.003838
        C25 = -0.000161
        C26 = 0.001243
        C27 = -0.009288
        C28 = -0.002124
        C29 = 0.012267
        C30 = -0.002398
        C31 = -0.000274
        C32 = 0
        C33 = 0
        C34 = 0
        C35 = -3.371000
        C36 = 0.858039
        C37 = 0.141600
        C38 = 7.201140
        C39 = -0.676007
        C40 = 0.806629
        C41 = 0.170454
        C42 = -0.390563
        C43 = 0.101966

        S9 = C20 * t**2 + C21 * t + C22
        S10 = C23 * t**2 + C24 * t + C25
        S11 = C26 * t**2 + C27 * t + C28
        S12 = C29 * t**2 + C30 * t + C31
        S13 = C32 * t**2 + C33 * t + C34
        S14 = C35 * t**2 + C36 * t + C37
        S15 = C38 * t**2 + C39 * t + C40
        S16 = C41 * t**2 + C42 * t + C43

        lambda5 = S9 * kappa + S10
        lambda6 = S11 * kappa + S12
        lambda7 = S13 * kappa + S14
        lambda8 = S15 * kappa + S16

        self._cl_coefficients = [lambda5, lambda6, lambda7, lambda8]

        # cd_coefficients
        C44 = 0.546094
        C45 = 0.022247
        C46 = -0.071462
        C47 = -0.006527
        C48 = 0.002733
        C49 = 0.000686
        C50 = 0.123685
        C51 = 0.143755
        C52 = 0.495159
        C53 = -0.105362
        C54 = 0.033468

        cd_2_deg = (
            (C44 * t + C45) * kappa**2 + (C46 * t + C47) * kappa + (C48 * t + C49)
        )
        cd_1_deg = 0
        cd_0_deg = (C50 * t + C51) * kappa + (C52 * t**2 + C53 * t + C54)

        self._cd_coefficients = [cd_2_deg, cd_1_deg, cd_0_deg]

        # cm_coefficients
        C55 = -0.284793
        C56 = -0.026199
        C57 = -0.024060
        C58 = 0.000559
        C59 = -1.787703
        C60 = 0.352443
        C61 = -0.839323
        C62 = 0.137932

        cm_2_deg = (C55 * t + C56) * kappa + (C57 * t + C58)
        cm_1_deg = 0
        cm_0_deg = (C59 * t + C60) * kappa + (C61 * t + C62)

        self._cm_coefficients = [cm_2_deg, cm_1_deg, cm_0_deg]

    def _from_masure_regression(
        self, airfoil_params, alpha_range, reynolds, ml_models_dir
    ):
        """Generate polar data using Masure regression model.

        Args:
            airfoil_params (dict): Dictionary containing regression parameters
                                   't', 'eta', 'kappa', 'delta', 'lambda', 'phi'.
            alpha_range (list): [min_alpha, max_alpha, step] in degrees.
            reynolds (float): Reynolds number for analysis.
            ml_models_dir (str): Base path for resolving relative model file paths.

        Returns:
            None: Populates self.alpha, self.CL, self.CD, self.CM.
        """
        # Extract parameters
        t = airfoil_params["t"]
        eta = airfoil_params["eta"]
        kappa = airfoil_params["kappa"]
        delta = airfoil_params["delta"]
        lambda_param = airfoil_params["lambda"]
        phi = airfoil_params["phi"]

        # Generate alpha range
        alpha_deg = np.arange(
            alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]
        )

        # Prepare input matrix for regression model
        # Input format: [t, cx, cy, r, LE, camTE, a]
        # Map parameters to the expected format
        n_alpha = len(alpha_deg)
        X_input = np.zeros((n_alpha, 7))

        for i, alpha in enumerate(alpha_deg):
            X_input[i, :] = [t, eta, kappa, delta, lambda_param, phi, alpha]

        # Load the trained model
        model = self._load_masure_regression_model(reynolds, ml_models_dir)

        # Make predictions
        predictions = model.predict(X_input)

        # Store results
        self.alpha = np.deg2rad(alpha_deg)
        self.CD = predictions[:, 0]  # Cd
        self.CL = predictions[:, 1]  # Cl
        self.CM = predictions[:, 2]  # Cm

    def _load_masure_regression_model(self, reynolds, ml_models_dir):
        """Load the trained regression model for a given Reynolds number.

        Args:
            reynolds (float): Reynolds number (5e6, 1e6, or 2e7)
            ml_models_dir (str): Directory containing the model files.

        Returns:
            sklearn model: Trained regression model
        """

        # Check if the model is already cached
        if reynolds in self._masure_model_cache:
            return self._masure_model_cache[reynolds]

        # Determine model file based on Reynolds number
        if reynolds == 5e6:
            model_name = "ET_re5e6.pkl"
        elif reynolds == 1e6:
            model_name = "ET_re1e6.pkl"
        elif reynolds == 2e7:
            model_name = "ET_re2e7.pkl"
        else:
            raise ValueError(
                f"No model available for Re={reynolds}. Available: 5e6, 1e6, 2e7"
            )

        # Construct path
        if ml_models_dir is None:
            raise ValueError("ml_models_dir must be provided for masure_regression.")

        model_path = Path(ml_models_dir) / model_name

        try:
            # Suppress sklearn version warnings during unpickling
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                with open(model_path, "rb") as f:
                    model = pickle.load(f)

            # Apply compatibility patches
            model = self._patch_sklearn_compatibility(model)

            # Test if the model can make predictions (compatibility check)
            test_input = np.array([[0.07, 0.2, 0.95, -2, 0.65, 0.25, 10]])
            _ = model.predict(test_input)

            # Cache the loaded model
            self._masure_model_cache[reynolds] = model

            return model

        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Model file {model_path} not found. Please ensure the model files are "
                f"installed in the data/models/ directory."
            ) from exc
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("This is likely due to scikit-learn version differences.")
            print("The model was trained with an older version of scikit-learn.")
            print(
                "For best results, consider retraining the model with the current sklearn version."
            )
            raise RuntimeError(
                f"Cannot load model {model_name} due to version incompatibility. "
                f"Original error: {e}"
            ) from e

    def _patch_sklearn_compatibility(self, model):
        """Patch sklearn model for version compatibility issues."""

        def patch_estimator(estimator):
            # Add missing monotonic_cst attribute for ExtraTreeRegressor
            if hasattr(estimator, "estimators_"):
                for tree in estimator.estimators_:
                    if not hasattr(tree, "monotonic_cst"):
                        tree.monotonic_cst = None
            elif not hasattr(estimator, "monotonic_cst"):
                estimator.monotonic_cst = None

            # Add other potential missing attributes
            if not hasattr(estimator, "_support_missing_values"):
                estimator._support_missing_values = lambda X: False

        # Handle different model structures
        if hasattr(model, "named_steps"):
            # Pipeline structure
            for _, step in model.named_steps.items():
                if hasattr(step, "estimators_"):
                    # MultiOutputRegressor
                    for estimator in step.estimators_:
                        patch_estimator(estimator)
                else:
                    patch_estimator(step)
        elif hasattr(model, "estimators_"):
            # Direct MultiOutputRegressor
            for estimator in model.estimators_:
                patch_estimator(estimator)
        else:
            # Single estimator
            patch_estimator(model)

        return model

    def to_polar_array(self):
        """Convert airfoil data to standardized numpy array format.

        Returns:
            np.ndarray: Array of shape (N, 4) with columns [alpha, CL, CD, CM].
                Alpha values are in radians.
        """
        return np.column_stack([self.alpha, self.CL, self.CD, self.CM])

    @classmethod
    def from_yaml_entry_batch(
        cls,
        airfoil_ids: list,
        airfoil_types: list,
        airfoil_params_list: list,
        alpha_range: list = None,
        reynolds: float = None,
        file_path: str = None,
        ml_models_dir: str = None,
    ):
        """Create multiple AirfoilAerodynamics instances with batch optimization and caching.

        This method optimizes the creation of multiple airfoil instances by:
        1. Checking for cached results first
        2. Grouping compatible types (masure_regression, neuralfoil) for batch processing
        3. Caching expensive computations for future use
        4. Falling back to individual processing for non-cacheable types

        Args:
            airfoil_ids (list): List of airfoil identifiers.
            airfoil_types (list): List of airfoil types for each ID.
            airfoil_params_list (list): List of parameter dictionaries for each airfoil.
            alpha_range (list, optional): [min_alpha, max_alpha, step] in degrees.
            reynolds (float, optional): Reynolds number for analysis.
            file_path (str, optional): Base path for resolving relative file references.
            ml_models_dir (str, optional): Base path for ML model files (required for masure_regression).

        Returns:
            dict: Dictionary mapping airfoil_id to polar data arrays.

        Raises:
            ValueError: If input lists have different lengths or invalid parameters.
        """
        if not (len(airfoil_ids) == len(airfoil_types) == len(airfoil_params_list)):
            raise ValueError("All input lists must have the same length")

        # Check cache first
        if cls._cache_enabled and ml_models_dir is not None:
            config_hash = cls._get_cache_config_hash(
                airfoil_ids, airfoil_types, airfoil_params_list, alpha_range, reynolds
            )
            cache_dir = cls._get_cache_dir(ml_models_dir)
            cache_file = cls._get_cache_filename(config_hash, cache_dir)

            # Try to load from cache
            if cache_file.exists():
                cached_data = cls._load_cache(cache_file)
                if cached_data is not None:
                    return cached_data["airfoil_polars"]

        # Cache miss or caching disabled - compute fresh results
        airfoil_polar_map = {}

        # Group airfoils by type for potential batch processing
        type_groups = {}
        for i, (airfoil_id, airfoil_type, airfoil_params) in enumerate(
            zip(airfoil_ids, airfoil_types, airfoil_params_list)
        ):
            airfoil_type_lower = airfoil_type.lower()
            if airfoil_type_lower not in type_groups:
                type_groups[airfoil_type_lower] = []
            type_groups[airfoil_type_lower].append((i, airfoil_id, airfoil_params))

        # Process each type group
        for airfoil_type, group_items in type_groups.items():
            if airfoil_type == "masure_regression":
                # Batch process masure_regression airfoils
                batch_results = cls._batch_process_masure_regression(
                    group_items, alpha_range, reynolds, ml_models_dir
                )
                for airfoil_id, polar_data in batch_results.items():
                    airfoil_polar_map[airfoil_id] = polar_data

            elif airfoil_type == "neuralfoil":
                # Batch process neuralfoil airfoils
                batch_results = cls._batch_process_neuralfoil(
                    group_items, alpha_range, reynolds, file_path
                )
                for airfoil_id, polar_data in batch_results.items():
                    airfoil_polar_map[airfoil_id] = polar_data

            else:
                # Process individually for other types (these are fast, no caching needed)
                for _, airfoil_id, airfoil_params in group_items:
                    aero = cls.from_yaml_entry(
                        airfoil_type,
                        airfoil_params,
                        alpha_range=alpha_range,
                        reynolds=reynolds,
                        file_path=file_path,
                        ml_models_dir=ml_models_dir,
                    )
                    airfoil_polar_map[airfoil_id] = aero.to_polar_array()

        # Save to cache if enabled and we have cacheable types
        if cls._cache_enabled and ml_models_dir is not None:
            cacheable_types_present = any(
                airfoil_type.lower() in cls._cacheable_types
                for airfoil_type in airfoil_types
            )

            if cacheable_types_present:
                # Clean up old cache files first
                cls._cleanup_old_cache_files(cache_dir)

                # Prepare cache data
                cache_data = {
                    "metadata": {
                        "creation_time": datetime.now().isoformat(),
                        "config_hash": config_hash,
                        "alpha_range": alpha_range,
                        "reynolds": reynolds,
                        "airfoil_types": airfoil_types,
                        "cacheable_types": sorted(cls._cacheable_types),
                    },
                    "airfoil_polars": airfoil_polar_map,
                }

                # Save to cache
                cls._save_cache(cache_data, cache_file)

        return airfoil_polar_map

    @classmethod
    def _batch_process_masure_regression(
        cls,
        group_items: list,
        alpha_range: list,
        reynolds: float,
        ml_models_dir: str,
    ):
        """Batch process masure_regression airfoils for efficiency.

        Args:
            group_items (list): List of (index, airfoil_id, airfoil_params) tuples.
            alpha_range (list): [min_alpha, max_alpha, step] in degrees.
            reynolds (float): Reynolds number for analysis.
            ml_models_dir (str): Base path for resolving relative model file paths.

        Returns:
            dict: Dictionary mapping airfoil_id to polar data arrays.
        """
        if not group_items:
            return {}

        # Generate alpha range once
        alpha_deg = np.arange(
            alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]
        )
        n_alpha = len(alpha_deg)
        n_airfoils = len(group_items)

        # Prepare batch input matrix
        # Shape: (n_airfoils * n_alpha, 7) where 7 is [t, eta, kappa, delta, lambda, phi, alpha]
        X_batch = np.zeros((n_airfoils * n_alpha, 7))

        # Fill the batch input matrix
        airfoil_ids = []
        for i, (_, airfoil_id, airfoil_params) in enumerate(group_items):
            airfoil_ids.append(airfoil_id)
            t = airfoil_params["t"]
            eta = airfoil_params["eta"]
            kappa = airfoil_params["kappa"]
            delta = airfoil_params["delta"]
            lambda_param = airfoil_params["lambda"]
            phi = airfoil_params["phi"]

            start_idx = i * n_alpha
            end_idx = (i + 1) * n_alpha

            for j, alpha in enumerate(alpha_deg):
                X_batch[start_idx + j, :] = [
                    t,
                    eta,
                    kappa,
                    delta,
                    lambda_param,
                    phi,
                    alpha,
                ]

        # Load model once (uses caching)
        obj = object.__new__(cls)
        model = obj._load_masure_regression_model(reynolds, ml_models_dir)

        # Make batch prediction
        y_batch = model.predict(X_batch)

        # Split results back into individual airfoils
        results = {}
        alpha_rad = np.deg2rad(alpha_deg)

        for i, airfoil_id in enumerate(airfoil_ids):
            start_idx = i * n_alpha
            end_idx = (i + 1) * n_alpha

            predictions = y_batch[start_idx:end_idx]
            CD = predictions[:, 0]  # Cd
            CL = predictions[:, 1]  # Cl
            CM = predictions[:, 2]  # Cm

            # Store as polar array [alpha, CL, CD, CM]
            polar_data = np.column_stack([alpha_rad, CL, CD, CM])
            results[airfoil_id] = polar_data

        return results

    @classmethod
    def _batch_process_neuralfoil(
        cls,
        group_items: list,
        alpha_range: list,
        reynolds: float,
        file_path: str,
    ):
        """Batch process neuralfoil airfoils for efficiency.

        Args:
            group_items (list): List of (index, airfoil_id, airfoil_params) tuples.
            alpha_range (list): [min_alpha, max_alpha, step] in degrees.
            reynolds (float): Reynolds number for analysis.
            file_path (str): Base path for resolving relative file paths.

        Returns:
            dict: Dictionary mapping airfoil_id to polar data arrays.
        """
        if not group_items:
            return {}

        # Generate alpha range once
        alpha_deg = np.arange(
            alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]
        )

        results = {}
        file_path = Path(file_path)

        # Process all neuralfoil airfoils
        # Note: NeuralFoil might not support true batching, so we process individually
        # but with shared alpha range for consistency
        for _, airfoil_id, airfoil_params in group_items:
            airfoil_params["dat_file_path"] = (
                file_path.parent / airfoil_params["dat_file_path"]
            )
            filename = airfoil_params["dat_file_path"]

            aero = nf.get_aero_from_dat_file(
                filename=filename,
                alpha=alpha_deg,
                Re=reynolds,
                model_size=airfoil_params.get("model_size", "xxxlarge"),
                xtr_lower=airfoil_params.get("xtr_lower", 0.01),
                xtr_upper=airfoil_params.get("xtr_upper", 0.01),
                n_crit=airfoil_params.get("n_crit", 9),
            )

            alpha_rad = np.deg2rad(alpha_deg)
            CL = aero["CL"]
            CD = aero["CD"]
            CM = aero["CM"]

            # Store as polar array [alpha, CL, CD, CM]
            polar_data = np.column_stack([alpha_rad, CL, CD, CM])
            results[airfoil_id] = polar_data

        return results

    @classmethod
    def _get_cache_config_hash(
        cls, airfoil_ids, airfoil_types, airfoil_params_list, alpha_range, reynolds
    ):
        """Generate a hash for the cache configuration.

        Args:
            airfoil_ids (list): List of airfoil IDs
            airfoil_types (list): List of airfoil types
            airfoil_params_list (list): List of airfoil parameters
            alpha_range (list): Alpha range [min, max, step]
            reynolds (float): Reynolds number

        Returns:
            str: Hash string for cache identification
        """
        # Create a deterministic representation of the configuration
        cache_data = {
            "airfoil_ids": airfoil_ids,
            "airfoil_types": airfoil_types,
            "airfoil_params": airfoil_params_list,
            "alpha_range": alpha_range,
            "reynolds": reynolds,
            "cacheable_types": sorted(
                cls._cacheable_types
            ),  # Include cacheable types in hash
        }

        # Convert to JSON string with sorted keys for deterministic hashing
        config_str = json.dumps(cache_data, sort_keys=True)

        # Generate SHA256 hash
        return hashlib.sha256(config_str.encode()).hexdigest()[
            :16
        ]  # Use first 16 chars

    @classmethod
    def _get_cache_dir(cls, ml_models_dir):
        """Get the cache directory path.

        Args:
            ml_models_dir (str): Base directory for resolving model paths

        Returns:
            Path: Cache directory path
        """
        if ml_models_dir is None:
            raise ValueError("ml_models_dir is required for cache operations")

        # Create cache directory inside ml_models_dir
        cache_dir = Path(ml_models_dir) / "cache"

        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir

    @classmethod
    def _cleanup_old_cache_files(cls, cache_dir):
        """Remove cache files from previous dates.

        Args:
            cache_dir (Path): Cache directory path
        """
        if not cache_dir.exists():
            return

        current_date = datetime.now().strftime("%Y%m%d")

        # Find all cache files
        cache_files = glob.glob(str(cache_dir / "aerodynamic_cache_*.pkl"))

        for cache_file in cache_files:
            filename = os.path.basename(cache_file)
            # Extract date from filename: aerodynamic_cache_YYYYMMDD_hash.pkl
            try:
                parts = filename.split("_")
                if len(parts) >= 3:
                    file_date = parts[2]  # YYYYMMDD part
                    if file_date != current_date:
                        os.remove(cache_file)
                        print(f"Removed old cache file: {filename}")
            except (IndexError, ValueError):
                # Skip files that don't match expected pattern
                continue

    @classmethod
    def _get_cache_filename(cls, config_hash, cache_dir):
        """Get the cache filename for a given configuration hash.

        Args:
            config_hash (str): Configuration hash
            cache_dir (Path): Cache directory path

        Returns:
            Path: Full path to cache file
        """
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"aerodynamic_cache_{current_date}_{config_hash}.pkl"
        return cache_dir / filename

    @classmethod
    def _save_cache(cls, cache_data, cache_file):
        """Save aerodynamic data to cache file.

        Args:
            cache_data (dict): Cache data to save
            cache_file (Path): Cache file path
        """
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cached aerodynamic data to: {cache_file.name}")
        except Exception as e:
            print(f"Warning: Failed to save cache file {cache_file}: {e}")

    @classmethod
    def _load_cache(cls, cache_file):
        """Load aerodynamic data from cache file.

        Args:
            cache_file (Path): Cache file path

        Returns:
            dict or None: Cached data or None if loading failed
        """
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            print(f"Loaded cached aerodynamic data from: {cache_file.name}")
            return cache_data
        except Exception as e:
            print(f"Warning: Failed to load cache file {cache_file}: {e}")
            return None

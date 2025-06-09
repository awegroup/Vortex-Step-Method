import numpy as np
import pandas as pd
from pathlib import Path
import neuralfoil as nf


class AirfoilAerodynamics:
    """Class to encapsulate 2D airfoil aerodynamic data and interpolation.

    Use AirfoilAerodynamics.from_yaml_entry(...) to instantiate.
    """

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
    ):
        """Create AirfoilAerodynamics instance from configuration parameters.

        Args:
            airfoil_type (str): Type of airfoil model ('breukels_regression', 'neuralfoil', 'polars', 'inviscid').
            airfoil_params (dict): Parameters specific to the airfoil type.
            alpha_range (list, optional): [min_alpha, max_alpha, step] in degrees. Defaults to None.
            reynolds (float, optional): Reynolds number for analysis. Defaults to None.
            file_path (str, optional): Base path for relative file references. Defaults to None.

        Returns:
            AirfoilAerodynamics: Instance with populated polar data.

        Raises:
            ValueError: If airfoil_type is not supported or required parameters are missing.
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

    def _from_polars(self, airfoil_params: dict, file_path: str):
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

    def to_polar_array(self):
        """Convert airfoil data to standardized numpy array format.

        Returns:
            np.ndarray: Array of shape (N, 4) with columns [alpha, CL, CD, CM].
                Alpha values are in radians.
        """
        return np.column_stack([self.alpha, self.CL, self.CD, self.CM])

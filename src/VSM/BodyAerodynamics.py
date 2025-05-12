import numpy as np
import logging
import pandas as pd
from pathlib import Path

from VSM.WingGeometry import Wing
from VSM.Panel import Panel
import VSM.Wake as Wake
from . import jit_cross, jit_norm, jit_dot


class BodyAerodynamics:
    """BodyAerodynamics class

    This class calculates the aerodynamic properties of a wing system, including the generation
    of panels, evaluation of circulatory distributions, and computation of aerodynamic forces,
    moments, and induced velocities. It supports both standard wing and bridled configurations.

    Args:
        wings (list): List of Wing object instances.
        bridle_line_system (list, optional): List of bridles, each defined as [p1, p2, diameter]. Defaults to None.
        aerodynamic_center_location (float, optional): The location factor for the aerodynamic center (default is 0.25).
        control_point_location (float, optional): The location factor for the control point (default is 0.75).

    Properties:
        panels (list): List of Panel object instances constructed from the wing geometry.
        n_panels (int): Number of panels.
        va (np.ndarray): The apparent velocity vector used in calculations.
        gamma_distribution (np.ndarray): Distribution of circulation along the panels.
        wings (list): List of Wing object instances.

    Methods:
        __init__: Initializes the BodyAerodynamics instance and builds the panel list.
        _build_panels: Constructs panels from the current wing geometry.
        from_file: Class method to instantiate an object by reading wing geometry and optional polar/bridle data from files.
        update_from_points: Updates the wing geometry from new LE, TE, tube diameter, and camber points.
        calculate_panel_properties: Computes the aerodynamic and geometric properties for each panel.
        calculate_AIC_matrices: Calculates the Aerodynamic Influence Coefficient matrices based on the specified aerodynamic model.
        calculate_circulation_distribution_elliptical_wing: Returns the circulation distribution for an elliptical wing.
        calculate_circulation_distribution_cosine: Returns the circulation distribution based on a cosine profile.
        calculate_results: Computes aerodynamic forces, moments, and other metrics based on the current state.
        va_initialize: Initializes the apparent velocity vector (va) and yaw rate.
        update_effective_angle_of_attack_if_VSM: Updates the effective angle of attack for VSM using induced velocities.
        calculate_line_aerodynamic_force: Computes the aerodynamic force on a line element (used for bridles).

    """

    def __init__(
        self,
        wings: list,  # List of Wing object instances
        bridle_line_system: list = None,
        aerodynamic_center_location: float = 0.25,
        control_point_location: float = 0.75,
    ):
        self._wings = wings
        self._aerodynamic_center_location = aerodynamic_center_location
        self._control_point_location = control_point_location

        ##TODO:
        self._bridle_line_system = bridle_line_system
        self.cd_cable = 1.1
        self.cf_cable = 0.01

        # Build the initial panel list from the wing geometry.
        self._build_panels()

        # Other initializations
        self._va = None
        self._gamma_distribution = None
        self._alpha_uncorrected = None
        self._alpha_corrected = None

    def _build_panels(self):
        """Helper method to build the panel list from the current wing geometry."""
        panels = []
        for wing in self.wings:
            section_list = wing.refine_aerodynamic_mesh()
            n_panels_per_wing = len(section_list) - 1

            # Calculate the panel properties
            (
                aerodynamic_center_list,
                control_point_list,
                bound_point_1_list,
                bound_point_2_list,
                x_airf_list,
                y_airf_list,
                z_airf_list,
            ) = self.calculate_panel_properties(
                section_list,
                n_panels_per_wing,
                self._aerodynamic_center_location,
                self._control_point_location,
            )
            for j in range(n_panels_per_wing):
                panels.append(
                    Panel(
                        section_list[j],
                        section_list[j + 1],
                        aerodynamic_center_list[j],
                        control_point_list[j],
                        bound_point_1_list[j],
                        bound_point_2_list[j],
                        x_airf_list[j],
                        y_airf_list[j],
                        z_airf_list[j],
                    )
                )
        self._panels = panels
        self._n_panels = len(panels)

    ####################
    ## CLASS METHODS ###
    ####################
    @classmethod
    def from_file(
        cls,
        file_path: str,
        n_panels: int,
        spanwise_panel_distribution: str,
        is_with_corrected_polar: bool = False,
        polar_data_dir: str = None,
        is_with_bridles: bool = False,
        bridle_data_path: str = None,
        is_half_wing: bool = False,
        is_neuralfoil: bool = False,
        nf_airfoil_data_dir: str = None,
        nf_reynolds_number: float = None,
        nf_alpha_range: np.ndarray = [-10, 25, 26],
        nf_xtr_lower: float = 0.01,
        nf_xtr_upper: float = 0.01,
        nf_n_crit: float = 9,
        nf_is_with_save_polar: bool = False,
    ):
        """
        Instantiate a BodyAerodynamics object from wing geometry data and optional aerodynamic polar or bridle data.

        This class method constructs a Wing instance and populates it with sections using data from a CSV
        file that defines wing geometry. Optionally, it can load pre-computed corrected polars or run NeuralFoil
        to generate polar data. Bridles can also be incorporated if required.

        ## Args:
            file_path (str): Path to the wing geometry CSV file. Required columns:
            • LE_x, LE_y, LE_z: Leading Edge coordinates.
            • TE_x, TE_y, TE_z: Trailing Edge coordinates.

            ### --- Breukels Correlation (if used) ---
            - d_tube: Tube diameter (required).
            - camber (or y_camber): Camber information (required).

            ### --- Wing Configuration ---
            - is_half_wing (bool, optional): If True, the input represents half a wing. The data will be mirrored
              (excluding the mid-span slice) to form a full wing.

            ### --- Panel Configuration ---
            - n_panels (int): Number of panels to discretize the wing.
            - spanwise_panel_distribution (str): Method for distributing panels along the wing span
              (e.g., uniform, cosine, etc.).

            ### --- Corrected Polar Data ---
            - is_with_corrected_polar (bool): If True, uses pre-computed corrected polar data from CSV files.
            - polar_data_dir (str): Directory containing corrected polar CSV files, each with columns:
              alpha, cl, cd, cm (with alpha in radians).

            ### --- NeuralFoil Polar Data ---
            - is_neuralfoil (bool): If True, computes airfoil polar data using NeuralFoil.
            - nf_airfoil_data_dir (str): Directory containing airfoil .dat files with (x, y) columns, orderd like
                • 1.dat: mid-span slice
                • 2.dat: first slice from the root
                • ...
                • n.dat: last 'tip' slice

            - nf_reynolds_number (float): Reynolds Number at which NeuralFoil should run.
            - nf_alpha_range (np.ndarray): Array with the minimum, maximum, and number of alpha values (in degrees).
            - nf_xtr_lower (float): Lower bound for transition location (0 means forced, 1 is fully free).
            - nf_xtr_upper (float): Upper bound for transition location.
            - nf_n_crit (float): Critical amplification factor for turbulent transition.
            Guidelines:
                • Sailplane:           12–14
                • Motorglider:         11–13
                • Clean wind tunnel:   10–12
                • Average wind tunnel: 9   (standard "e^9 method")
                • Dirty wind tunnel:   4–8
            - nf_is_with_save_polar (bool): If True, saves the generated polar data to a CSV file in the specified directory.

            ### --- Bridle Data ---
            - is_with_bridles (bool, optional): If True, reads an additional CSV file for bridle information.
            - bridle_data_path (str): Path to the bridle data CSV file. Required columns:
            • p1_x, p1_y, p1_z: First bridle point.
            • p2_x, p2_y, p2_z: Second bridle point.
            • diameter: Cable diameter.

        ## Returns:
            BodyAerodynamics: An instance built using the provided wing geometry and (if applicable) the aerodynamic
            polar or bridle data.

        ## Raises:
            ValueError: If required data for any enabled configuration (corrected polar, NeuralFoil, or Breukels) is missing.
        """
        # Initialize wing
        wing_instance = Wing(
            n_panels=n_panels, spanwise_panel_distribution=spanwise_panel_distribution
        )
        # Read wing geometry and airfoil parameters from file
        df = pd.read_csv(file_path)

        if is_half_wing:
            # make a copy of df
            df_orderded_opposite = df.copy()
            # Remove the last item, as this should be the mid-span slice
            df_orderded_opposite = df_orderded_opposite.drop(
                df_orderded_opposite.index[-1]
            )
            # reverse the order of the rows
            df_orderded_opposite = df_orderded_opposite[::-1]
            # change sign of y-coordinates
            df_orderded_opposite["LE_y"] = -df_orderded_opposite["LE_y"]
            df_orderded_opposite["TE_y"] = -df_orderded_opposite["TE_y"]
            # add rows to df, mirroring the existing rows to create a full wing
            df = pd.concat([df, df_orderded_opposite])

        for i, row in df.iterrows():
            # 1) extract leading/trailing‐edge coords
            LE = np.array([row["LE_x"], row["LE_y"], row["LE_z"]])
            TE = np.array([row["TE_x"], row["TE_y"], row["TE_z"]])

            ##########################
            ### PreComputed Polars ####
            ##########################
            if is_with_corrected_polar:
                print(f"corrected_polar index: {i}")
                # 2a) corrected‐polar branch
                df_polar = pd.read_csv(Path(polar_data_dir) / f"{i}.csv")
                polar_data = [
                    "polar_data",
                    np.column_stack(
                        (
                            np.deg2rad(df_polar["alpha"].values),
                            df_polar["Cl"].values,
                            df_polar["Cd"].values,
                            df_polar["Cm"].values,
                        )
                    ),
                ]
                wing_instance.add_section(LE, TE, polar_data)

            ##########################
            ### NeuralFoil polars ####
            ##########################
            elif is_neuralfoil:

                if nf_airfoil_data_dir is None:
                    raise ValueError(
                        "airfoil_data_dir must be set if is_neuralfoil is True"
                    )
                if nf_reynolds_number is None:
                    raise ValueError(
                        "Re_for_neuralfoil must be set if is_neuralfoil is True"
                    )
                n_files_in_airfoil_data_dir = len(
                    [
                        f
                        for f in Path(nf_airfoil_data_dir).iterdir()
                        if f.is_file() and f.suffix == ".dat"
                    ]
                )
                idx = n_files_in_airfoil_data_dir - i
                # if not is_half_wing and nf_is_airfoils_of_half_wing:
                #     # deal with the index change
                #     if idx < 0:
                #         idx = n_files_in_airfoil_data_dir + idx
                # read the airfoil data
                airfoil_dat_file_path = Path(nf_airfoil_data_dir) / f"{idx}.dat"

                # Run neuralfoil
                import neuralfoil as nf

                alpha_range = np.linspace(
                    nf_alpha_range[0],
                    nf_alpha_range[1],
                    nf_alpha_range[2],
                )
                aero = nf.get_aero_from_dat_file(
                    filename=airfoil_dat_file_path,
                    alpha=alpha_range,
                    Re=nf_reynolds_number,
                    model_size="xxxlarge",
                    xtr_lower=nf_xtr_lower,
                    xtr_upper=nf_xtr_upper,
                    n_crit=nf_n_crit,
                )
                polar_data = [
                    "polar_data",
                    np.column_stack(
                        (np.deg2rad(alpha_range), aero["CL"], aero["CD"], aero["CM"])
                    ),
                ]
                wing_instance.add_section(LE, TE, polar_data)

                if nf_is_with_save_polar:
                    # Save the polar data
                    save_path = (
                        Path(nf_airfoil_data_dir)
                        / "neuralfoil_calculated_2D_polars"
                        / f"{idx}.csv"
                    )
                    df_to_save = pd.DataFrame(
                        {
                            "alpha": alpha_range,
                            "CL": aero["CL"],
                            "CD": aero["CD"],
                            "CM": aero["CM"],
                        }
                    )
                    if not save_path.parent.exists():
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                    df_to_save.to_csv(save_path, index=False)

            #############################
            ### Breukels Correlation ####
            #############################
            else:
                if row["d_tube"] is None:
                    raise ValueError(
                        "d_tube must be provided as column in wing_geometry if using Breukels Correlation"
                    )
                if row["y_camber"] is None:
                    raise ValueError(
                        "y_camber must be provided as column in wing_geometry if using Breukels Correlation"
                    )
                # 2b) always fall back to breukels
                airfoil_data = [
                    "lei_airfoil_breukels",
                    [row["d_tube"], row["y_camber"]],
                ]
                wing_instance.add_section(LE, TE, airfoil_data)

        # Instantiate BodyAerodynamics with or without bridles
        if is_with_bridles:
            df_bridle = pd.read_csv(bridle_data_path)
            bridle_lines = [
                [
                    np.array([row["p1_x"], row["p1_y"], row["p1_z"]]),
                    np.array([row["p2_x"], row["p2_y"], row["p2_z"]]),
                    row["diameter"],
                ]
                for _, row in df_bridle.iterrows()
            ]
            return cls([wing_instance], bridle_lines)
        else:
            return cls([wing_instance])

    ###########################
    ## GETTER FUNCTIONS
    ###########################

    @property
    def panels(self):
        return self._panels

    @property
    def n_panels(self):
        return self._n_panels

    @property
    def va(self):
        return self._va

    @property
    def gamma_distribution(self):
        return self._gamma_distribution

    @property
    def wings(self):
        return self._wings

    # @property
    # def stall_angle_list(self):
    #     return self._stall_angle_list

    ###########################
    ## SETTER FUNCTIONS
    ###########################

    @gamma_distribution.setter
    def gamma_distribution(self, value):
        self._gamma_distribution = value

    @panels.setter
    def panels(self, value):
        self._panels = value

    @va.setter
    def va(self, va, yaw_rate: float = 0.0):

        # # Removing old wake filaments
        # self.panels = Wake.remove_frozen_wake(self.panels)
        if isinstance(va, tuple) and len(va) == 2:
            va, yaw_rate = va

        self._va = np.array(va)

        if len(va) == 3 and yaw_rate == 0.0:
            va_distribution = np.repeat([va], len(self.panels), axis=0)
        elif len(va) == len(self.panels):
            va_distribution = va
        elif yaw_rate != 0.0 and len(va) == 3:
            va_distribution = []

            for wing in self.wings:
                # Create the spanwise positions array
                spanwise_positions = np.array(
                    [panel.control_point[1] for panel in self.panels]
                )

                for i in range(wing.n_panels):
                    yaw_rate_apparent_velocity = np.array(
                        [-yaw_rate * spanwise_positions[i], 0, 0]
                    )

                    # Append the current wing's velocities to the overall distribution
                    va_distribution.append(yaw_rate_apparent_velocity + va)

            # Concatenate all wings' distributions into a single array
            va_distribution = np.vstack(va_distribution)

        else:
            raise ValueError(
                f"Invalid va distribution, len(va) :{len(va)} != len(self.panels):{len(self.panels)}"
            )
        # Update the va attribute of each panel
        for i, panel in enumerate(self.panels):
            panel.va = va_distribution[i]

        # Add the frozen wake elements based on the va distribution
        self.panels = Wake.frozen_wake(va_distribution, self.panels)

    ###########################
    ## CALCULATE FUNCTIONS
    ###########################
    def calculate_panel_properties(
        self,
        section_list,
        n_panels,
        aerodynamic_center_location,
        control_point_location,
    ):
        ac = aerodynamic_center_location
        cp = control_point_location

        # Initialize lists
        aerodynamic_center_list = []
        control_point_list = []
        bound_point_1_list = []
        bound_point_2_list = []
        x_airf_list = []
        y_airf_list = []
        z_airf_list = []

        # defining coordinates
        coordinates = np.zeros((2 * (n_panels + 1), 3))
        logging.debug(f"shape of coordinates: {coordinates.shape}")
        for i in range(n_panels):
            coordinates[2 * i] = section_list[i].LE_point
            coordinates[2 * i + 1] = section_list[i].TE_point
            coordinates[2 * i + 2] = section_list[i + 1].LE_point
            coordinates[2 * i + 3] = section_list[i + 1].TE_point

        logging.debug(f"coordinates: {coordinates}")

        for i in range(n_panels):
            # Identify points defining the panel
            section = {
                "p1": coordinates[2 * i, :],  # p1 = LE_1
                "p2": coordinates[2 * i + 2, :],  # p2 = LE_2
                "p3": coordinates[2 * i + 3, :],  # p3 = TE_2
                "p4": coordinates[2 * i + 1, :],  # p4 = TE_1
            }

            di = jit_norm(
                coordinates[2 * i, :] * cp
                + coordinates[2 * i + 1, :] * ac
                - (coordinates[2 * i + 2, :] * cp + coordinates[2 * i + 3, :] * ac)
            )
            if i == 0:
                diplus = jit_norm(
                    coordinates[2 * (i + 1), :] * cp
                    + coordinates[2 * (i + 1) + 1, :] * ac
                    - (
                        coordinates[2 * (i + 1) + 2, :] * cp
                        + coordinates[2 * (i + 1) + 3, :] * ac
                    )
                )
                ncp = di / (di + diplus)
            elif i == n_panels - 1:
                dimin = jit_norm(
                    coordinates[2 * (i - 1), :] * cp
                    + coordinates[2 * (i - 1) + 1, :] * ac
                    - (
                        coordinates[2 * (i - 1) + 2, :] * cp
                        + coordinates[2 * (i - 1) + 3, :] * ac
                    )
                )
                ncp = dimin / (dimin + di)
            else:
                dimin = jit_norm(
                    coordinates[2 * (i - 1), :] * cp
                    + coordinates[2 * (i - 1) + 1, :] * ac
                    - (
                        coordinates[2 * (i - 1) + 2, :] * cp
                        + coordinates[2 * (i - 1) + 3, :] * ac
                    )
                )
                diplus = jit_norm(
                    coordinates[2 * (i + 1), :] * cp
                    + coordinates[2 * (i + 1) + 1, :] * ac
                    - (
                        coordinates[2 * (i + 1) + 2, :] * cp
                        + coordinates[2 * (i + 1) + 3, :] * ac
                    )
                )
                ncp = ac * (dimin / (dimin + di) + di / (di + diplus) + 1)

            ncp = 1 - ncp

            # aerodynamic center at 1/4c
            LLpoint = (section["p2"] * (1 - ncp) + section["p1"] * ncp) * cp + (
                section["p3"] * (1 - ncp) + section["p4"] * ncp
            ) * ac
            # control point at 3/4c
            VSMpoint = (section["p2"] * (1 - ncp) + section["p1"] * ncp) * ac + (
                section["p3"] * (1 - ncp) + section["p4"] * ncp
            ) * cp

            # Calculating the bound
            bound_1 = section["p1"] * cp + section["p4"] * ac
            bound_2 = section["p2"] * cp + section["p3"] * ac

            ### Calculate the local reference frame, below are all unit_vectors
            # NORMAL x_airf defined upwards from the chord-line, perpendicular to the panel
            # used to be: p2 - p1
            x_airf = jit_cross(VSMpoint - LLpoint, section["p1"] - section["p2"])
            x_airf = x_airf / jit_norm(x_airf)

            # TANGENTIAL y_airf defined parallel to the chord-line, from LE-to-TE
            y_airf = VSMpoint - LLpoint
            y_airf = y_airf / jit_norm(y_airf)

            # SPAN z_airf along the LE, in plane (towards left tip, along span) from the airfoil perspective
            # used to be bound_2 - bound_1
            z_airf = bound_1 - bound_2
            z_airf = z_airf / jit_norm(z_airf)

            # Appending
            aerodynamic_center_list.append(LLpoint)
            control_point_list.append(VSMpoint)
            bound_point_1_list.append(bound_1)
            bound_point_2_list.append(bound_2)
            x_airf_list.append(x_airf)
            y_airf_list.append(y_airf)
            z_airf_list.append(z_airf)

        return (
            aerodynamic_center_list,
            control_point_list,
            bound_point_1_list,
            bound_point_2_list,
            x_airf_list,
            y_airf_list,
            z_airf_list,
        )

    def calculate_AIC_matrices(
        self, aerodynamic_model_type, core_radius_fraction, va_norm_array, va_unit_array
    ):
        """Calculates the AIC matrices for the given aerodynamic model

        Args:
            aerodynamic_model_type (str): The aerodynamic model to be used, either VSM or LLT
            core_radius_fraction (float): The core radius fraction for the vortex model

        Returns:
            Tuple[np.array, np.array, np.array]: The x, y, and z components of the AIC matrix
        """
        if aerodynamic_model_type not in ["VSM", "LLT"]:
            raise ValueError("Invalid aerodynamic model type, should be VSM or LLT")

        evaluation_point = (
            "control_point" if aerodynamic_model_type == "VSM" else "aerodynamic_center"
        )
        evaluation_point_on_bound = aerodynamic_model_type == "LLT"

        AIC = np.empty((3, self.n_panels, self.n_panels))

        for icp, panel_icp in enumerate(self.panels):
            ep = getattr(panel_icp, evaluation_point)
            for jring, panel_jring in enumerate(self.panels):
                velocity_induced = (
                    panel_jring.calculate_velocity_induced_single_ring_semiinfinite(
                        ep,
                        evaluation_point_on_bound,
                        va_norm_array[jring],
                        va_unit_array[jring],
                        gamma=1,
                        core_radius_fraction=core_radius_fraction,
                    )
                )
                AIC[:, icp, jring] = velocity_induced

                if icp == jring and aerodynamic_model_type == "VSM":
                    U_2D = panel_jring.calculate_velocity_induced_bound_2D(ep)
                    AIC[:, icp, jring] -= U_2D

        return AIC[0], AIC[1], AIC[2]

    def calculate_circulation_distribution_elliptical_wing(self, gamma_0=1):
        """
        Calculates the circulation distribution for an elliptical wing.

        Assumes that the wing's span is defined in self.wings[0].span and that the
        y-coordinates of the control points (from self.panels) are measured relative
        to the wing center, ranging from -wing_span/2 to wing_span/2.

        Args:
            gamma_0 (float): The circulation at the wing root.

        Returns:
            np.array: The circulation distribution following an elliptical profile.
        """
        if len(self.wings) > 1:
            raise NotImplementedError("Multiple wings not yet implemented")

        wing_span = self.wings[0].span
        logging.debug(f"wing_span: {wing_span}")

        # Get the y-coordinate for each panel's control point.
        y = np.array([panel.control_point[1] for panel in self.panels])

        # Compute the elliptical circulation distribution.
        gamma_i = gamma_0 * np.sqrt(1 - (2 * y / wing_span) ** 2)

        logging.debug(f"Calculated elliptical gamma distribution: {gamma_i}")
        return gamma_i

    def calculate_circulation_distribution_cosine(self, gamma_0=1):
        """
        Calculates the circulation distribution based on a cosine profile,
        i.e. f(x) = 1 - cos(x), where x is remapped over [0, π].

        This function assumes that the number of panels defines the resolution
        of the distribution. The distribution is scaled such that its maximum
        (at x = π) is gamma_0.

        Args:
            gamma_0 (float): Scaling factor (or the circulation value at x = π).

        Returns:
            np.array: The circulation distribution based on the cosine function.
        """
        import numpy as np

        # Create a set of x-values uniformly distributed between 0 and π.
        x = np.linspace(0, np.pi, len(self.panels))

        # Compute the distribution: f(x) = 1 - cos(x).
        gamma_i = gamma_0 * (1 - np.cos(x))

        logging.debug(f"Calculated cosine gamma distribution: {gamma_i}")
        return gamma_i

    ##TODO: add this
    # def calculate_kcu(model_type: cylinder, area, location, va)

    def calculate_results(
        self,
        gamma_new,
        density,
        aerodynamic_model_type,
        core_radius_fraction,
        mu,
        alpha_array,
        Umag_array,
        chord_array,
        x_airf_array,
        y_airf_array,
        z_airf_array,
        va_array,
        va_norm_array,
        va_unit_array,
        panels,
        is_only_f_and_gamma_output,
        reference_point,
    ):

        cl_array, cd_array, cm_array = (
            np.zeros(len(panels)),
            np.zeros(len(panels)),
            np.zeros(len(panels)),
        )
        panel_width_array = np.zeros(len(panels))
        for icp, panel_i in enumerate(panels):
            cl_array[icp] = panel_i.calculate_cl(alpha_array[icp])
            cd_array[icp], cm_array[icp] = panel_i.calculate_cd_cm(alpha_array[icp])
            panel_width_array[icp] = panel_i.width
        lift = (cl_array * 0.5 * density * Umag_array**2 * chord_array)[:, np.newaxis]
        drag = (cd_array * 0.5 * density * Umag_array**2 * chord_array)[:, np.newaxis]
        moment = (cm_array * 0.5 * density * Umag_array**2 * chord_array**2)[
            :, np.newaxis
        ]

        if aerodynamic_model_type == "VSM":
            alpha_corrected = self.update_effective_angle_of_attack_if_VSM(
                gamma_new,
                core_radius_fraction,
                x_airf_array,
                y_airf_array,
                va_array,
                va_norm_array,
                va_unit_array,
            )
            alpha_uncorrected = alpha_array[:, np.newaxis]

        elif aerodynamic_model_type == "LLT":
            alpha_corrected = alpha_array[:, np.newaxis]
            alpha_uncorrected = alpha_array[:, np.newaxis]
        else:
            raise ValueError("Unknown aerodynamic model type, should be LLT or VSM")
        # Checking that va is not distributed input
        if len(self._va) != 3:
            raise ValueError("Calc.results not ready for va_distributed input")

        # Initializing variables
        cl_prescribed_va_list = []
        cd_prescribed_va_list = []
        cs_prescribed_va_list = []
        f_global_3D_list = []
        fx_global_3D_list = []
        fy_global_3D_list = []
        fz_global_3D_list = []
        area_all_panels = 0
        lift_wing_3D_sum = 0
        drag_wing_3D_sum = 0
        side_wing_3D_sum = 0
        fx_global_3D_sum = 0
        fy_global_3D_sum = 0
        fz_global_3D_sum = 0

        ### Moments
        m_global_3D_list = []
        mx_global_3D_list = []
        my_global_3D_list = []
        mz_global_3D_list = []
        mx_global_3D_sum = 0
        my_global_3D_sum = 0
        mz_global_3D_sum = 0

        spanwise_direction = self.wings[0].spanwise_direction
        va_mag = jit_norm(self._va)
        va = self._va
        va_unit = va / va_mag
        q_inf = 0.5 * density * va_mag**2
        for i, panel_i in enumerate(self.panels):

            ### Defining panel_variables
            # Defining directions of airfoil that defines current panel_i
            z_airf_span = panel_i.z_airf  # along the span
            y_airf_chord = panel_i.y_airf  # along the chord
            x_airf_normal_to_chord = panel_i.x_airf  # normal to the chord
            # TODO: implement these
            alpha_corrected_i = alpha_corrected[i]
            panel_chord = panel_i.chord
            panel_width = panel_i.width
            panel_area = panel_chord * panel_width
            area_all_panels += panel_area

            ### Calculate the direction of the induced apparent wind speed to the airfoil orientation
            # this is done using the CORRECTED CALCULATED (comes from gamma distribution) angle of attack
            # For VSM the correction is applied, and it is the angle of attack, from calculating induced velocities at the 1/4c aerodynamic center location
            # For LTT the correction is NOT applied, and it is the angle of attack, from calculating induced velocities at the 3/4c control point
            induced_va_airfoil = (
                np.cos(alpha_corrected_i) * y_airf_chord
                + np.sin(alpha_corrected_i) * x_airf_normal_to_chord
            )
            dir_induced_va_airfoil = induced_va_airfoil / jit_norm(induced_va_airfoil)

            ### Calculate the direction of the lift and drag vectors
            # lift is perpendical/normal to induced apparent wind speed
            dir_lift_induced_va = jit_cross(dir_induced_va_airfoil, z_airf_span)
            dir_lift_induced_va = dir_lift_induced_va / jit_norm(dir_lift_induced_va)
            # drag is parallel/tangential to induced apparent wind speed
            dir_drag_induced_va = jit_cross(spanwise_direction, dir_lift_induced_va)
            dir_drag_induced_va = dir_drag_induced_va / jit_norm(dir_drag_induced_va)

            ### Calculating the MAGNITUDE of the lift and drag
            # The VSM and LTT methods do NOT differ here, both use the uncorrected angle of attack
            # i.e. evaluate the magnitude at the (3/4c) control point
            # 2D AIRFOIL aerodynamic forces, so multiplied by chord
            lift_induced_va_mag = lift[i]
            drag_induced_va_mag = drag[i]

            # panel force VECTOR NORMAL to CALCULATED induced velocity
            lift_induced_va = lift_induced_va_mag * dir_lift_induced_va
            # panel force VECTOR TANGENTIAL to CALCULATED induced velocity
            drag_induced_va = drag_induced_va_mag * dir_drag_induced_va
            ftotal_induced_va = lift_induced_va + drag_induced_va
            logging.debug(f" OLD ftotal_induced_va: {ftotal_induced_va}")

            ### Converting forces to prescribed wing va
            dir_lift_prescribed_va = jit_cross(va, spanwise_direction)
            dir_lift_prescribed_va = dir_lift_prescribed_va / jit_norm(
                dir_lift_prescribed_va
            )
            lift_prescribed_va = jit_dot(
                lift_induced_va, dir_lift_prescribed_va
            ) + jit_dot(drag_induced_va, dir_lift_prescribed_va)
            drag_prescribed_va = jit_dot(lift_induced_va, va_unit) + jit_dot(
                drag_induced_va, va_unit
            )

            dir_side = jit_cross(dir_lift_prescribed_va, va_unit)
            side_prescribed_va = jit_dot(lift_induced_va, dir_side) + jit_dot(
                drag_induced_va, dir_side
            )
            # else:
            #     side_prescribed_va = jit_dot(
            #         lift_induced_va, spanwise_direction
            #     ) + jit_dot(drag_induced_va, spanwise_direction)

            side_prescribed_va = side_prescribed_va

            ### Converting forces to the global reference frame
            fx_global_2D = jit_dot(ftotal_induced_va, np.array([1, 0, 0]))
            fy_global_2D = jit_dot(ftotal_induced_va, np.array([0, 1, 0]))
            fz_global_2D = jit_dot(ftotal_induced_va, np.array([0, 0, 1]))

            # 3D, by multiplying with the panel width
            lift_wing_3D = lift_prescribed_va * panel_width
            drag_wing_3D = drag_prescribed_va * panel_width
            side_wing_3D = side_prescribed_va * panel_width
            fx_global_3D = fx_global_2D * panel_width
            fy_global_3D = fy_global_2D * panel_width
            fz_global_3D = fz_global_2D * panel_width

            # summing it up for totals
            lift_wing_3D_sum += lift_wing_3D
            drag_wing_3D_sum += drag_wing_3D
            side_wing_3D_sum += side_wing_3D
            fx_global_3D_sum += fx_global_3D
            fy_global_3D_sum += fy_global_3D
            fz_global_3D_sum += fz_global_3D

            # Storing results that are useful
            cl_prescribed_va_list.append(lift_prescribed_va / (q_inf * panel_chord))
            cd_prescribed_va_list.append(drag_prescribed_va / (q_inf * panel_chord))
            cs_prescribed_va_list.append(side_prescribed_va / (q_inf * panel_chord))
            fx_global_3D_list.append(fx_global_3D)
            fy_global_3D_list.append(fy_global_3D)
            fz_global_3D_list.append(fz_global_3D)
            f_global_3D_list.append(
                np.array([fx_global_3D, fy_global_3D, fz_global_3D])
            )

            ####################
            ##### MOMENTS ######
            ####################

            # # Get the moment magnitude
            # moment_induced_va_mag = moment[i]
            # # moment_lever_arm
            # # The moment is defined, and represents the value at 1/4c
            # # The force is however computed at 3/4c control point
            # # The moment lever arm is then defined as the distance between these two points
            # moment_lever_arm_dir = y_airf_chord

            # # Moment direction computation
            # # Use cross product to define moment vector direction,
            # #   using the total force direction and the lever arm direction
            # ftotal_induced_va_unit = ftotal_induced_va / jit_norm(ftotal_induced_va)
            # dir_moment_induced_va = jit_cross(
            #     ftotal_induced_va_unit, moment_lever_arm_dir
            # )
            # dir_moment_induced_va = dir_moment_induced_va / jit_norm(
            #     dir_moment_induced_va
            # )

            # # Moment vector computation
            # moment_induced_va = moment_induced_va_mag * dir_moment_induced_va

            # ### Converting moments to the global reference frame
            # mx_global_2D = jit_dot(moment_induced_va, np.array([1, 0, 0]))
            # my_global_2D = jit_dot(moment_induced_va, np.array([0, 1, 0]))
            # mz_global_2D = jit_dot(moment_induced_va, np.array([0, 0, 1]))

            # # 3D, by multiplying with the panel width
            # mx_global_3D = mx_global_2D * panel_width
            # my_global_3D = my_global_2D * panel_width
            # mz_global_3D = mz_global_2D * panel_width

            # # Summing up totals
            # mx_global_3D_sum += mx_global_3D
            # my_global_3D_sum += my_global_3D
            # mz_global_3D_sum += mz_global_3D

            # # Storing results
            # mx_global_3D_list.append(mx_global_3D)
            # my_global_3D_list.append(my_global_3D)
            # mz_global_3D_list.append(mz_global_3D)
            # m_global_3D_list.append(
            #     np.array([mx_global_3D, my_global_3D, mz_global_3D])
            # )

            # (1) Panel aerodynamic center in global frame:
            panel_ac_global = panel_i.aerodynamic_center  # 3D [x, y, z]

            # (2) Convert local (2D) pitching moment to a 3D vector in global coords.
            #     Use the axis around which the moment is defined,
            #     which is the z-axis pointing "spanwise"
            moment_axis_global = panel_i.z_airf

            # Scale by panel width if your 'moment[i]' is 2D moment-per-unit-span:
            M_local_3D = moment[i] * moment_axis_global * panel_i.width

            # (3) Vector from panel AC to the chosen reference point:
            r_vector = panel_ac_global - reference_point  # e.g. CG, wing root, etc.

            # (4) Cross product to shift the force from panel AC to ref. point:
            force_global_3D = np.array([fx_global_3D, fy_global_3D, fz_global_3D])
            M_shift = np.cross(r_vector, force_global_3D)

            # (5) Total panel moment about the reference point:
            M_ref_panel = M_local_3D + M_shift

            # (6) Accumulate or store:
            mx_global_3D_sum += M_ref_panel[0]
            my_global_3D_sum += M_ref_panel[1]
            mz_global_3D_sum += M_ref_panel[2]
            mx_global_3D_list.append(M_ref_panel[0])
            my_global_3D_list.append(M_ref_panel[1])
            mz_global_3D_list.append(M_ref_panel[2])
            m_global_3D_list.append(M_ref_panel)

        if is_only_f_and_gamma_output:
            return {
                "F_distribution": f_global_3D_list,
                "gamma_distribution": gamma_new,
            }

        # Calculating projected_area, wing_span, aspect_ratio
        projected_area = 0
        if len(self.wings) > 1:
            raise ValueError("more than 1 wing functions have not been implemented yet")

        wing = self.wings[0]
        projected_area = wing.calculate_projected_area()
        wing_span = wing.span
        aspect_ratio_projected = wing_span**2 / projected_area

        # Calculate geometric angle of attack wrt horizontal at mid-span
        horizontal_direction = np.array([1, 0, 0])
        alpha_geometric = np.array(
            [
                np.rad2deg(
                    np.arccos(jit_dot(panel_i.y_airf, horizontal_direction))
                    / (jit_norm(panel_i.y_airf) * jit_norm(horizontal_direction))
                )
                for panel_i in self.panels
            ]
        )
        # Calculating Reynolds Number
        max_chord = max(np.array([panel.chord for panel in self.panels]))
        reynolds_number = density * va_mag * max_chord / mu

        ##TODO: if bridle not none add lift, drag...
        if self._bridle_line_system is not None:
            fa_bridle = 0
            # TODO: Calculate induced va at each point of the bridle
            # TODO: Calculate moments at each point of the bridle
            for bridle_line in self._bridle_line_system:
                fa_bridle += self.calculate_line_aerodynamic_force(va, bridle_line)

            fx_global_3D_sum += fa_bridle[0]
            fy_global_3D_sum += fa_bridle[1]
            fz_global_3D_sum += fa_bridle[2]
            lift_wing_3D_sum += jit_dot(fa_bridle, dir_lift_prescribed_va)
            drag_wing_3D_sum += jit_dot(fa_bridle, va_unit)
            side_wing_3D_sum += jit_dot(fa_bridle, dir_side)

            # sum up

            # add to the total lift, drag, not the distributions of the wing

        ##TODO: if KCU, use Va, find a force.

        ### Storing results in a dictionary
        results_dict = {}
        # Global wing aerodynamics
        results_dict.update([("Fx", fx_global_3D_sum)])
        results_dict.update([("Fy", fy_global_3D_sum)])
        results_dict.update([("Fz", fz_global_3D_sum)])
        results_dict.update([("Mx", mx_global_3D_sum)])
        results_dict.update([("My", my_global_3D_sum)])
        results_dict.update([("Mz", mz_global_3D_sum)])
        results_dict.update([("lift", lift_wing_3D_sum)])
        results_dict.update([("drag", drag_wing_3D_sum)])
        results_dict.update([("side", side_wing_3D_sum)])
        results_dict.update([("cl", lift_wing_3D_sum / (q_inf * projected_area))])
        results_dict.update([("cd", drag_wing_3D_sum / (q_inf * projected_area))])
        results_dict.update([("cs", side_wing_3D_sum / (q_inf * projected_area))])
        results_dict.update(
            [("cmx", mx_global_3D_sum / (q_inf * projected_area * max_chord))]
        )
        results_dict.update(
            [("cmy", my_global_3D_sum / (q_inf * projected_area * max_chord))]
        )
        results_dict.update(
            [("cmz", mz_global_3D_sum / (q_inf * projected_area * max_chord))]
        )
        # Local panel aerodynamics
        results_dict.update([("cl_distribution", cl_prescribed_va_list)])
        results_dict.update([("cd_distribution", cd_prescribed_va_list)])
        results_dict.update([("cs_distribution", cs_prescribed_va_list)])
        results_dict.update([("F_distribution", f_global_3D_list)])
        results_dict.update([("M_distribution", m_global_3D_list)])

        # Additional info
        results_dict.update(
            [
                (
                    "cfx_distribution",
                    np.array(fx_global_3D_list) / (q_inf * projected_area),
                )
            ]
        )
        results_dict.update(
            [
                (
                    "cfy_distribution",
                    np.array(fy_global_3D_list) / (q_inf * projected_area),
                )
            ]
        )
        results_dict.update(
            [
                (
                    "cfz_distribution",
                    np.array(fz_global_3D_list) / (q_inf * projected_area),
                )
            ]
        )
        results_dict.update(
            [
                (
                    "cmx_distribution",
                    np.array(mx_global_3D_list)
                    / (q_inf * projected_area * chord_array),
                )
            ]
        )
        results_dict.update(
            [
                (
                    "cmy_distribution",
                    np.array(my_global_3D_list)
                    / (q_inf * projected_area * chord_array),
                )
            ]
        )
        results_dict.update(
            [
                (
                    "cmz_distribution",
                    np.array(mz_global_3D_list)
                    / (q_inf * projected_area * chord_array),
                )
            ]
        )
        results_dict.update([("alpha_at_ac", alpha_corrected)])
        results_dict.update([("alpha_uncorrected", alpha_uncorrected)])
        results_dict.update([("alpha_geometric", alpha_geometric)])
        results_dict.update([("gamma_distribution", gamma_new)])
        results_dict.update([("area_all_panels", area_all_panels)])
        results_dict.update([("projected_area", projected_area)])
        results_dict.update([("wing_span", wing_span)])
        results_dict.update([("aspect_ratio_projected", aspect_ratio_projected)])
        results_dict.update([("Rey", reynolds_number)])

        ### Logging
        logging.debug(f"cl:{results_dict['cl']}")
        logging.debug(f"cd:{results_dict['cd']}")
        logging.debug(f"cs:{results_dict['cs']}")
        logging.debug(f"lift:{lift_wing_3D_sum}")
        logging.debug(f"drag:{drag_wing_3D_sum}")
        logging.debug(f"side:{side_wing_3D_sum}")
        logging.debug(f"area: {area_all_panels}")
        logging.debug(f"Projected Area: {projected_area}")
        logging.debug(f"Aspect Ratio Projected: {aspect_ratio_projected}")

        return results_dict

    def calculate_y_coordinates(self):
        """
        Calculates the y-coordinates of the control points for each panel in the WingAero object.

        Returns:
            np.ndarray: An array of y-coordinates for each panel's control point.
        """
        return np.array([panel.control_point[1] for panel in self.panels])

    ###########################
    ## UPDATE FUNCTIONS
    ###########################

    def va_initialize(
        self,
        Umag: float = 3.15,
        angle_of_attack: float = 6.8,
        side_slip: float = 0.0,
        yaw_rate: float = 0.0,
    ):
        """
        Initializes the apparent velocity (va) and yaw rate for the WingAero object.

        Parameters:
        Umag (float): Magnitude of the velocity.
        angle_of_attack (float): Angle of attack in degrees.
        side_slip (float): Sideslip angle in degrees, a minus is added because its defined counter-clockwise.
        yaw_rate (float): Yaw rate, default is 0.0.
        """
        # Convert angles to radians
        aoa_rad = np.deg2rad(angle_of_attack)
        # a (-) is added because its defined counter-clockwise
        side_slip_rad = -np.deg2rad(side_slip)

        # Calculate apparent velocity vector
        vel_app = (
            np.array(
                [
                    np.cos(aoa_rad) * np.cos(side_slip_rad),
                    np.sin(side_slip_rad),
                    np.sin(aoa_rad),
                ]
            )
            * Umag
        )

        # Set the va attribute using the setter
        self.va = (vel_app, yaw_rate)

    def update_effective_angle_of_attack_if_VSM(
        self,
        gamma,
        core_radius_fraction,
        x_airf_array,
        y_airf_array,
        va_array,
        va_norm_array,
        va_unit_array,
    ):
        """Updates the angle of attack at the aerodynamic center of each panel,
            Calculated at the AERODYNAMIC CENTER, which needs an update for VSM
            And can just use the old value for the LLT

        Args:
            None

        Returns:
            None
        """
        # The correction is done by calculating the alpha at the aerodynamic center,
        # where as before the control_point was used in the VSM method
        aerodynamic_model_type = "LLT"
        AIC_x, AIC_y, AIC_z = self.calculate_AIC_matrices(
            aerodynamic_model_type, core_radius_fraction, va_norm_array, va_unit_array
        )
        induced_velocity_all = np.array(
            [
                np.matmul(AIC_x, gamma),
                np.matmul(AIC_y, gamma),
                np.matmul(AIC_z, gamma),
            ]
        ).T
        relative_velocity_array = va_array + induced_velocity_all
        v_normal_array = np.sum(x_airf_array * relative_velocity_array, axis=1)
        v_tangential_array = np.sum(y_airf_array * relative_velocity_array, axis=1)
        alpha_array = np.arctan(v_normal_array / v_tangential_array)

        return alpha_array[:, np.newaxis]

    def calculate_line_aerodynamic_force(
        self, va, line, cd_cable=1.1, cf_cable=0.01, density=1.225
    ):
        # TODO: test this function
        p1 = line[0]
        p2 = line[1]
        d = line[2]

        if p1[2] > p2[2]:
            p1, p2 = p2, p1

        length = np.linalg.norm(p2 - p1)
        ej = (p2 - p1) / length
        theta = np.arccos(np.dot(va, ej) / (np.linalg.norm(va) * np.linalg.norm(ej)))

        cd_t = cd_cable * np.sin(theta) ** 3 + np.pi * cf_cable * np.cos(theta) ** 3
        cl_t = (
            cd_cable * np.sin(theta) ** 2 * np.cos(theta)
            - np.pi * cf_cable * np.sin(theta) * np.cos(theta) ** 2
        )
        dir_D = va / np.linalg.norm(va)  # Drag direction
        dir_L = -(ej - np.dot(ej, dir_D) * dir_D)  # Lift direction
        dynamic_pressure_area = 0.5 * density * np.linalg.norm(va) ** 2 * length * d

        # Calculate lift and drag using the common factor
        lift_j = dynamic_pressure_area * cl_t * dir_L
        drag_j = dynamic_pressure_area * cd_t * dir_D

        return lift_j + drag_j

    def update_from_points(
        self,
        le_arr: np.ndarray,
        te_arr: np.ndarray,
        d_tube_arr: np.ndarray,
        y_camber_arr: np.ndarray,
        aero_input_type: str = "lei_airfoil_breukels",
    ):
        # Update each wing with the new points.
        for wing in self.wings:
            wing.update_wing_from_points(
                le_arr, te_arr, d_tube_arr, y_camber_arr, aero_input_type
            )
        # Rebuild the panels based on the updated geometry.
        self._build_panels()

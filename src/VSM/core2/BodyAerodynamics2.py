import numpy as np
import logging
import pandas as pd
from pathlib import Path
import yaml
from .WingGeometry2 import Wing
from .Panel2 import Panel
from .Wake2 import Wake
from .AirfoilAerodynamics2 import AirfoilAerodynamics
from .utils2 import (
    intersect_line_with_plane,
    point_in_quad,
    compute_effective_section_axes,  #TODO: NEW
)
from . import jit_cross, jit_norm, jit_dot


class BodyAerodynamics:
    """BodyAerodynamics class

    This class computes the aerodynamic properties of a wing system, including the generation
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
        compute_panel_properties: Computes the aerodynamic and geometric properties for each panel.
        compute_AIC_matrices: Calculates the Aerodynamic Influence Coefficient matrices based on the specified aerodynamic model.
        compute_circulation_distribution_elliptical_wing: Returns the circulation distribution for an elliptical wing.
        compute_circulation_distribution_cosine: Returns the circulation distribution based on a cosine profile.
        compute_results: Computes aerodynamic forces, moments, and other metrics based on the current state.
        va_initialize: Initializes the apparent velocity vector (va) and body rotation rates.
        update_effective_angle_of_attack_if_VSM: Updates the effective angle of attack for VSM using induced velocities.
        compute_line_aerodynamic_force: Computes the aerodynamic force on a line element (used for bridles).

    """

    def __init__(
        self,
        wings: list,  # List of Wing object instances
        bridle_line_system: list = None,
        aerodynamic_center_location: float = 0.25,
        control_point_location: float = 0.75,
        use_jointed_wake: bool = False,  #TODO: NEW
    ):
        self._wings = wings
        self._aerodynamic_center_location = aerodynamic_center_location
        self._control_point_location = control_point_location
        self._use_jointed_wake = use_jointed_wake  #TODO: NEW

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
        self._body_rates = np.zeros(3)

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
            ) = self.compute_panel_properties(
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
    def instantiate(
        cls,
        n_panels: int,
        file_path=None,
        wing_instance=None,
        spanwise_panel_distribution="uniform",
        bridle_path=None,
        ml_models_dir=None,
        use_jointed_wake: bool = False,  #TODO: NEW
    ):
        """
        Instantiate a BodyAerodynamics object from either a provided wing_instance or a YAML config file.

        Args:
            n_panels (int): Number of panels (required if wing_instance is not provided).
            file_path (str, optional): Path to the YAML config file. If None, wing_instance must be provided.
            wing_instance (Wing, optional): Pre-built Wing instance. If None, file_path must be provided.
            spanwise_panel_distribution (str): Panel distribution type (default: 'linear').
            is_with_bridles (bool): Whether to include bridle lines (default: False).

        Returns:
            BodyAerodynamics instance.

        YAML file structure expected (see aero_geometry_surfplan_inviscid.yaml):

        - wing_sections:
            headers: [airfoil_id, LE_x, LE_y, LE_z, TE_x, TE_y, TE_z]
            data:
              - [airfoil_id, LE_x, LE_y, LE_z, TE_x, TE_y, TE_z]
              - ...

        - wing_airfoils:
            alpha_range: [min, max, step]  # [deg], range for polar calculation
            reynolds: <float>              # Reynolds number
            headers: [airfoil_id, type, info_dict]
            data:
              - [airfoil_id, type, {parameters...}]
              - ...
            # type: one of [neuralfoil, breukels_regression, masure_regression, polars]
            # info_dict fields depend on type, e.g. for breukels_regression: t, kappa

        - bridle_nodes: (optional)
            headers: [id, x, y, z, type]
            data:
              - [id, x, y, z, type]
              - ...

        - bridle_lines: (optional)
            headers: [name, rest_length, diameter, material, rho]
            data:
              - [name, rest_length, diameter, material, rho]
              - ...

        - bridle_connections: (optional)
            headers: [name, ci, cj, ck]
            data:
              - [name, ci, cj, ck]
              - ...

        Notes:
            - The 'bridle_nodes', 'bridle_lines', and 'bridle_connections' sections are optional and only needed if is_with_bridles is True.
            - The airfoil_id in wing_sections must match the airfoil_id in wing_airfoils.
            - Each airfoil section in wing_airfoils must provide the required parameters for its type.

        """
        if wing_instance is None and n_panels is None:
            raise ValueError("Without a wing_instance, n_panels must be provided.")

        if wing_instance is None:
            if file_path is None:
                raise ValueError("Either file_path or wing_instance must be provided.")

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            section_headers = config["wing_sections"]["headers"]
            section_data = config["wing_sections"]["data"]
            idx_airfoil = section_headers.index("airfoil_id")
            idx_LE = [section_headers.index(f"LE_{ax}") for ax in ("x", "y", "z")]
            idx_TE = [section_headers.index(f"TE_{ax}") for ax in ("x", "y", "z")]

            airfoil_headers = config["wing_airfoils"]["headers"]
            airfoil_data = config["wing_airfoils"]["data"]
            try:
                alpha_range = config["wing_airfoils"]["alpha_range"]
            except KeyError:
                raise ValueError(
                    "Missing required 'alpha_range' in 'wing_airfoils' section of YAML config."
                )
            try:
                reynolds = config["wing_airfoils"]["reynolds"]
            except KeyError:
                raise ValueError(
                    "Missing required 'reynolds' in 'wing_airfoils' section of YAML config."
                )

            # --- Collect all airfoil data for batch processing ---
            airfoil_ids = []
            airfoil_types = []
            airfoil_params_list = []

            for airfoil_row in airfoil_data:
                airfoil_id = airfoil_row[airfoil_headers.index("airfoil_id")]
                airfoil_type = airfoil_row[airfoil_headers.index("type")]
                airfoil_params = airfoil_row[airfoil_headers.index("info_dict")]

                airfoil_ids.append(airfoil_id)
                airfoil_types.append(airfoil_type)
                airfoil_params_list.append(airfoil_params)

            # Check if a masure_regression model is specified
            if "masure_regression" in airfoil_types:
                if ml_models_dir is None:
                    raise ValueError(
                        "ml_models_dir must be provided for masure_regression."
                    )

            # --- Batch process all airfoils using optimized method ---
            airfoil_polar_map = AirfoilAerodynamics.from_yaml_entry_batch(
                airfoil_ids=airfoil_ids,
                airfoil_types=airfoil_types,
                airfoil_params_list=airfoil_params_list,
                alpha_range=alpha_range,
                reynolds=reynolds,
                file_path=file_path,
                ml_models_dir=ml_models_dir,
            )

            wing_instance = Wing(
                n_panels=n_panels,
                spanwise_panel_distribution=spanwise_panel_distribution,
            )

            for row in section_data:
                airfoil_id = row[idx_airfoil]
                LE = np.array([row[idx_LE[0]], row[idx_LE[1]], row[idx_LE[2]]])
                TE = np.array([row[idx_TE[0]], row[idx_TE[1]], row[idx_TE[2]]])
                polar_data = airfoil_polar_map[airfoil_id]
                wing_instance.add_section(LE, TE, polar_data)

        # --- Add bridle lines if requested ---
        if bridle_path is not None:
            with open(bridle_path, "r") as f:
                struc_geometry = yaml.safe_load(f)

            # ---- Particles as one big array, index = id ----
            # KCU (id = 0) from "bridle_point_node" if present, else [0,0,0]
            kcu_xyz = struc_geometry.get("bridle_point_node", [0.0, 0.0, 0.0])

            # Collect all [id, x, y, z] rows
            wing_rows = struc_geometry["wing_particles"]["data"]
            bridle_rows = struc_geometry["bridle_particles"]["data"]

            # Determine array size from max id
            all_ids = [0] + [r[0] for r in wing_rows] + [r[0] for r in bridle_rows]
            max_id = int(max(all_ids))

            # Initialize with NaNs and fill by id
            particles = np.full((max_id + 1, 3), np.nan, dtype=float)
            particles[0] = np.asarray(kcu_xyz, dtype=float)

            for row in wing_rows:
                pid, x, y, z = row[0], row[1], row[2], row[3]
                particles[int(pid)] = [x, y, z]

            for row in bridle_rows:
                pid, x, y, z = row[0], row[1], row[2], row[3]
                particles[int(pid)] = [x, y, z]

            # Optional: sanity check for any missing ids (still NaN)
            if np.isnan(particles).any():
                missing = np.where(np.isnan(particles).any(axis=1))[0]
                raise ValueError(
                    f"Particles missing coordinates for ids: {missing.tolist()}"
                )

            # ---- Bridle elements (nested dict) ----
            be_hdr = struc_geometry["bridle_lines"]["headers"][
                1:
            ]  # [l0, d, material, linktype]
            bridle_lines_dict = {
                row[0]: dict(zip(be_hdr, row[1:]))
                for row in struc_geometry["bridle_lines"]["data"]
            }

            # ---- Build bridle line segments ----
            bridle_lines = (
                []
            )  # each item: [p_start_xyz (np.array), p_end_xyz (np.array), diameter]
            for row in struc_geometry["bridle_connections"]["data"]:
                name = row[0]
                if name not in bridle_lines_dict:
                    raise KeyError(
                        f"Connection '{name}' not found in bridle_lines. "
                        "Add it there (with diameter 'd') or rename to match."
                    )

                d = bridle_lines_dict[name]["diameter"]

                # First segment (ci -> cj)
                ci = int(row[1])
                cj = int(row[2])
                p1 = particles[ci]
                p2 = particles[cj]
                bridle_lines.append([p1, p2, d])

                # Pulley segment (cj -> ck), if a 3rd node is present
                if len(row) == 4:
                    ck = int(row[3])
                    p3 = particles[ck]
                    bridle_lines.append([p2, p3, d])
            return cls([wing_instance], bridle_lines, use_jointed_wake=use_jointed_wake)  #TODO: NEW
        else:
            return cls([wing_instance], use_jointed_wake=use_jointed_wake)  #TODO: NEW

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
    def body_rates(self):
        return self._body_rates

    @property
    def gamma_distribution(self):
        return self._gamma_distribution

    @property
    def wings(self):
        return self._wings

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
    def va(
        self,
        va: np.ndarray,
        *,  # the asterisk forces the following args to be keyword-only
        roll_rate: float = 0.0,
        pitch_rate: float = 0.0,
        yaw_rate: float = 0.0,
        reference_point: np.ndarray | None = None,
    ) -> None:
        """
        Set the apparent velocity distribution and optional rigid-body rates.

        Parameters
        ----------
        va : array-like
            - shape (3,) for a uniform freestream vector, or
            - shape (n_panels, 3) for a per-panel apparent velocity.
        roll_rate, pitch_rate, yaw_rate : float
            Body rates in rad/s (p, q, r).
        reference_point : array-like, optional
            3-vector r0 for rotational inflow. Default is (0,0,0).
            v_rot(r) = omega x (r - r0).
        """
        # normalize and store the canonical 'va' (see note below)
        va = np.asarray(va, dtype=float)
        self._va = va

        n = self.n_panels
        if va.shape == (3,):
            va_distribution = np.tile(va, (n, 1))
        elif va.shape == (n, 3):
            va_distribution = va.copy()
        else:
            raise ValueError(f"'va' must be shape (3,) or ({n}, 3); got {va.shape}")

        # store body rates for introspection
        self._body_rates = np.array([roll_rate, pitch_rate, yaw_rate], dtype=float)

        # add rotational inflow only if any rate is nonzero
        if (roll_rate != 0.0) or (pitch_rate != 0.0) or (yaw_rate != 0.0):
            r0 = (
                np.zeros(3, dtype=float)
                if reference_point is None
                else np.asarray(reference_point, dtype=float)
            )
            if r0.shape != (3,):
                raise ValueError(f"reference_point must be shape (3,), got {r0.shape}")

            control_points = np.array(
                [p.control_point for p in self.panels], dtype=float
            )
            # v_rot = [p,q,r] x (r - r0)
            va_distribution += np.cross(self._body_rates, control_points - r0)

        # push to panels
        for i, panel in enumerate(self.panels):
            panel.va = va_distribution[i]

        # update wake using the new distribution
        self.panels = Wake.frozen_wake(  #TODO: NEW
            va_distribution, self.panels, use_jointed_wake=self._use_jointed_wake  #TODO: NEW
        )  #TODO: NEW

    ###########################
    ## CALCULATE FUNCTIONS
    ###########################
    def compute_panel_properties(
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

    def compute_AIC_matrices(
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
                    panel_jring.compute_velocity_induced_single_ring_semiinfinite(
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
                    U_2D = panel_jring.compute_velocity_induced_bound_2D(ep)

                    AIC[:, icp, jring] -= U_2D

        return AIC[0], AIC[1], AIC[2]

    def compute_circulation_distribution_elliptical_wing(self, gamma_0=1):
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

    def compute_circulation_distribution_cosine(self, gamma_0=1):
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
    # def compute_kcu(model_type: cylinder, area, location, va)

    def find_center_of_pressure(self, force_array, moment_array, reference_point):
        """
        Finds the intersection of the force line with all panels and returns the first valid one.
        """

        F = force_array
        M0 = moment_array
        r0 = np.array(reference_point)

        F_norm_sq = np.dot(F, F)

        if F_norm_sq == 0:
            raise ValueError("Force vector must not be zero.")

        r0_moment = r0 + np.cross(F, M0) / F_norm_sq
        F_unit = F / np.linalg.norm(F)

        for panel_idx, panel in enumerate(self.panels):
            corner_points = panel.corner_points  # shape (4, 3)

            v1 = corner_points[1] - corner_points[0]
            v2 = corner_points[2] - corner_points[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            intersection = intersect_line_with_plane(
                r0_moment, F_unit, corner_points[0], normal
            )

            if intersection is not None:
                if point_in_quad(intersection, corner_points):
                    return intersection  # Found the intersection!

        logging.warning(
            "No intersection found with any panel, in the center_of_pressure calculation."
        )
        return None

    def viscous_drag_correction(
        self,
        Umag,
        chord,
        dir_induced_va,
        panel,  # your panel object
        rho,
        mu,
        q_inf,
    ):
        """
        Returns two 3D force vectors: (f_corr_drag, f_corr_span)
        in the panel's true local drag- and spanwise- directions.
        # this is following:
        "A correction model for the effect of spanwise flow on the
        viscous force contribution in BEM and Lifting Line methods"
        Mac Gaunaa et al 2024 J. Phys.: Conf. Ser. 2767 022068
        DOI: 10.1088/1742-6596/2767/2/022068
        """
        # 1) decompose into spanwise vs. normal components
        v_par = Umag * np.dot(dir_induced_va, panel.z_airf)
        v_perp = np.sqrt(max(0.0, Umag**2 - v_par**2))

        # 2) angle & Re
        β = np.arctan2(v_par, v_perp)
        Re_ref = rho * v_perp * chord / mu

        # 3) nondim corrections (Eqns 10 & 11)
        f0 = 0.062 * Re_ref ** (-1 / 7)
        ΔCd = f0 * ((np.cos(β)) ** (-5 / 7) - 1.0)
        C_para = f0 * np.tan(β) * (np.cos(β)) ** (-5 / 7)

        # 4) dimensional magnitudes
        extra_D = ΔCd * q_inf * chord
        extra_S = C_para * q_inf * chord

        # 5) build true‐direction vectors
        #    — drag is _tangent_ to the panel, i.e. in the direction of the induced‐wind drag
        dir_drag = np.cross(panel.z_airf, np.cross(panel.z_airf, dir_induced_va))
        dir_drag = dir_drag / np.linalg.norm(dir_drag)

        #    — spanwise is simply panel.z_airf
        dir_span = panel.z_airf

        return extra_D * dir_drag, extra_S * dir_span

    def compute_panel_center_of_pressures(
        self, results_dict, reference_point=[0, 0, 0]
    ):
        """
        Compute the center of pressure for each panel by using the full 3D
        force and moment vectors and projecting them onto the panel's local axes.

        Parameters
        ----------
        results_dict : dict
            Must contain keys
            - "F_distribution": list of (3,) force vectors in global coords
            - "M_distribution": list of (3,) moment vectors about reference_point
        reference_point : array_like (3,)
            The point about which M_distribution is measured.

        Returns
        -------
        panel_cp_locations : list of (3,) floats
            Global XYZ location of CP for each panel.
        """
        panel_cp_locations = []
        # sum of these is equal to entire force and moment on the body
        F_dist = results_dict["F_distribution"]
        M_dist = results_dict["M_distribution"]

        for i, (F_glob, M_ref) in enumerate(zip(F_dist, M_dist)):
            panel = self.panels[i]
            ac = panel.aerodynamic_center  # (3,)
            y_airf = panel.y_airf  # chord direction
            z_airf = panel.z_airf  # span direction
            c = panel.chord

            F = np.array(F_glob)
            M_ref = np.array(M_ref)
            # 1) shift moment from 'reference_point' back to the AC:
            r = ac - np.array(reference_point)
            M_local = M_ref - np.cross(r, F)

            # 2) pitching moment about span axis:
            m_pitch = np.dot(M_local, z_airf)

            # 3) force in the chord plane (perpendicular to span)
            F_perp = F - np.dot(F, z_airf) * z_airf
            F_perp_mag = np.linalg.norm(F_perp)

            # 4) if there's no pitching‐plane force, fallback to AC
            if F_perp_mag < 1e-12:
                panel_cp_locations.append(ac)
                continue

            # 5) lever arm along chord = M / F
            lever = m_pitch / F_perp_mag

            # 6) clamp lever arm so CP stays on [LE, TE]:
            #    at quarter chord AC sits at +0.25c from LE,
            #    so lever ∈ [−0.25c, +0.75c]
            lever = np.clip(lever, -0.25 * c, 0.75 * c)

            # 7) build CP in global coords
            cp = ac + lever * y_airf
            panel_cp_locations.append(cp)

        return panel_cp_locations

    def compute_results(
        self,
        gamma_new,
        rho,
        aerodynamic_model_type,
        core_radius_fraction,
        mu,
        alpha_array,
        Umag_array,
        chord_array,
        chord_eff_array,  #TODO: NEW
        x_airf_array,
        y_airf_array,
        z_airf_array,
        va_array,
        va_norm_array,
        va_unit_array,
        panels,
        is_only_f_and_gamma_output,
        is_with_viscous_drag_correction,
        reference_point,
        is_aoa_corrected,
    ):

        cl_array, cd_array, cm_array = (
            np.zeros(len(panels)),
            np.zeros(len(panels)),
            np.zeros(len(panels)),
        )
        panel_width_array = np.zeros(len(panels))
        for icp, panel_i in enumerate(panels):
            cl_array[icp] = panel_i.compute_cl(alpha_array[icp])
            cd_array[icp], cm_array[icp] = panel_i.compute_cd_cm(alpha_array[icp])
            panel_width_array[icp] = panel_i.width
        x_eff_array, y_eff_array, _ = compute_effective_section_axes(  #TODO: NEW
            y_airf_array, z_airf_array  #TODO: NEW
        )  #TODO: NEW
        lift = (cl_array * 0.5 * rho * Umag_array**2 * chord_array)[:, np.newaxis]  #TODO: NEW
        drag = (cd_array * 0.5 * rho * Umag_array**2 * chord_array)[:, np.newaxis]  #TODO: NEW
        moment = (cm_array * 0.5 * rho * Umag_array**2 * chord_array**2)[  #TODO: NEW
            :, np.newaxis  #TODO: NEW
        ]  #TODO: NEW

        if is_aoa_corrected:
            alpha_corrected = self.update_effective_angle_of_attack_if_VSM(
                gamma_new,
                core_radius_fraction,
                x_airf_array,
                y_airf_array,
                z_airf_array,  #TODO: NEW
                va_array,
                va_norm_array,
                va_unit_array,
            )
            alpha_uncorrected = alpha_array[:, np.newaxis]

        else:
            alpha_corrected = alpha_array[:, np.newaxis]
            alpha_uncorrected = alpha_array[:, np.newaxis]
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
        q_inf = 0.5 * rho * va_mag**2
        for i, panel_i in enumerate(self.panels):

            ### Defining panel_variables
            # Defining directions of airfoil that defines current panel_i
            z_airf_span = panel_i.z_airf  # along the span
            y_airf_chord = y_eff_array[i]  # along the effective chord  #TODO: NEW
            x_airf_normal_to_chord = x_eff_array[i]  # normal to the effective chord  #TODO: NEW
            # TODO: implement these
            alpha_corrected_i = alpha_corrected[i]
            panel_chord = panel_i.chord  #TODO: NEW
            panel_chord_safe = panel_chord if panel_chord > 1e-12 else 1e-12  #TODO: NEW
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

            ##################################
            if is_with_viscous_drag_correction:
                f_corr_drag, f_corr_span = self.viscous_drag_correction(
                    Umag=Umag_array[i],
                    chord=panel_chord,
                    dir_induced_va=dir_induced_va_airfoil,  # needed to compute β
                    panel=panel_i,  # needed for true span & drag dirs
                    rho=rho,
                    mu=mu,
                    q_inf=q_inf,
                )
                ftotal_induced_va += f_corr_drag + f_corr_span

                # Decompose corrections into the (D, L, S) basis
                e_D = va_unit
                e_L = dir_lift_prescribed_va
                e_S = dir_side

                # project both correction vectors
                dD = np.dot(f_corr_drag, e_D) + np.dot(f_corr_span, e_D)
                dL = np.dot(f_corr_drag, e_L) + np.dot(f_corr_span, e_L)
                dS = np.dot(f_corr_drag, e_S) + np.dot(f_corr_span, e_S)

                # printing the delta's
                print(f"\nPanel {i}")
                print(
                    f"Drag: {drag_prescribed_va:.3f}, Lift: {lift_prescribed_va:.3f}, Side: {side_prescribed_va:.3f}"
                )
                print(f"+Drag: {dD:.3f}, +Lift: {dL:.3f}, +Side: {dS:.3f}")

                # add into your existing scalars
                drag_prescribed_va += dD
                lift_prescribed_va += dL
                side_prescribed_va += dS

            # ----------------------------------
            ####################################

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
            cl_prescribed_va_list.append(lift_prescribed_va / (q_inf * panel_chord_safe))  #TODO: NEW
            cd_prescribed_va_list.append(drag_prescribed_va / (q_inf * panel_chord_safe))  #TODO: NEW
            cs_prescribed_va_list.append(side_prescribed_va / (q_inf * panel_chord_safe))  #TODO: NEW
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
        projected_area = wing.compute_projected_area()
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
        reynolds_number = rho * va_mag * max_chord / mu

        if self._bridle_line_system is not None:
            # Calculate forces and moments for each bridle line individually
            for bridle_line in self._bridle_line_system:
                # Calculate force for this individual bridle line
                fa_bridle_line = self.compute_line_aerodynamic_force(va, bridle_line)

                # Add bridle forces to global force totals
                fx_global_3D_sum += fa_bridle_line[0]
                fy_global_3D_sum += fa_bridle_line[1]
                fz_global_3D_sum += fa_bridle_line[2]
                lift_wing_3D_sum += jit_dot(fa_bridle_line, dir_lift_prescribed_va)
                drag_wing_3D_sum += jit_dot(fa_bridle_line, va_unit)
                side_wing_3D_sum += jit_dot(fa_bridle_line, dir_side)

                # Calculate moment for this bridle line
                # Bridle line midpoint as moment application point
                bridle_midpoint = 0.5 * (bridle_line[0] + bridle_line[1])

                # Vector from reference point to bridle midpoint
                r_bridle = bridle_midpoint - np.array(reference_point)

                # Moment contribution from this bridle line
                M_bridle_line = np.cross(r_bridle, fa_bridle_line)

                # Add to global moment totals
                mx_global_3D_sum += M_bridle_line[0]
                my_global_3D_sum += M_bridle_line[1]
                mz_global_3D_sum += M_bridle_line[2]

        # Find center of pressure (now uses consistent force and moment data)
        x_cp = self.find_center_of_pressure(
            [fx_global_3D_sum, fy_global_3D_sum, fz_global_3D_sum],
            [mx_global_3D_sum, my_global_3D_sum, mz_global_3D_sum],
            reference_point,
        )

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
        results_dict.update([("center_of_pressure", x_cp)])

        panel_cp_locations = self.compute_panel_center_of_pressures(results_dict)
        results_dict.update([("panel_cp_locations", panel_cp_locations)])

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

    def compute_y_coordinates(self):
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
        pitch_rate: float = 0.0,
        roll_rate: float = 0.0,
        reference_point: np.ndarray = None,
    ):
        """
        Initializes the apparent velocity (va) and body rates for the WingAero object.

        Parameters:
        Umag (float): Magnitude of the velocity.
        angle_of_attack (float): Angle of attack in degrees.
        side_slip (float): Sideslip angle in degrees.
        yaw_rate (float): Yaw rate about the body z-axis, default is 0.0.
        pitch_rate (float): Pitch rate about the body y-axis, default is 0.0.
        roll_rate (float): Roll rate about the body x-axis, default is 0.0.
        reference_point (np.ndarray): Reference point for moment calculation [x, y, z], default is None (uses [0, 0, 0]).
        """
        # Convert angles to radians
        aoa_rad = np.deg2rad(angle_of_attack)
        side_slip_rad = np.deg2rad(side_slip)

        # Set the va attribute using the setter with keyword arguments
        # The va setter requires keyword-only arguments after the va parameter
        type(self).va.fset(
            self,
            va=(
                np.array(
                    [
                        np.cos(aoa_rad) * np.cos(side_slip_rad),
                        np.sin(side_slip_rad),
                        np.sin(aoa_rad),
                    ]
                )
                * Umag
            ),
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
            reference_point=reference_point,
        )

    def update_effective_angle_of_attack_if_VSM(
        self,
        gamma,
        core_radius_fraction,
        x_airf_array,
        y_airf_array,
        z_airf_array,  #TODO: NEW
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
        AIC_x, AIC_y, AIC_z = self.compute_AIC_matrices(
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
        x_eff_array, y_eff_array, _ = compute_effective_section_axes(  #TODO: NEW
            y_airf_array, z_airf_array  #TODO: NEW
        )  #TODO: NEW
        u_dot_z = np.sum(relative_velocity_array * z_airf_array, axis=1)  #TODO: NEW
        u_eff_array = relative_velocity_array - u_dot_z[:, None] * z_airf_array  #TODO: NEW
        v_normal_array = np.sum(x_eff_array * u_eff_array, axis=1)  #TODO: NEW
        v_tangential_array = np.sum(y_eff_array * u_eff_array, axis=1)  #TODO: NEW
        alpha_array = np.arctan2(v_normal_array, v_tangential_array)  #TODO: NEW

        return alpha_array[:, np.newaxis]

    def compute_line_aerodynamic_force(
        self, va, line, cd_cable=1.1, cf_cable=0.01, rho=1.225
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
        dynamic_pressure_area = 0.5 * rho * np.linalg.norm(va) ** 2 * length * d

        # Calculate lift and drag using the common factor
        lift_j = dynamic_pressure_area * cl_t * dir_L
        drag_j = dynamic_pressure_area * cd_t * dir_D

        return lift_j + drag_j

    def update_from_points(self, le_arr, te_arr, aero_input_type, initial_polar_data):
        # Update each wing with the new points.
        for wing in self.wings:
            wing.update_wing_from_points(
                le_arr, te_arr, aero_input_type, initial_polar_data
            )
        # Rebuild the panels based on the updated geometry.
        self._build_panels()

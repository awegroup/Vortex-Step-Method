import numpy as np
import logging
import pandas as pd
from pathlib import Path
import yaml
from VSM.core.WingGeometry import Wing
from VSM.core.Panel import Panel
from VSM.core.Wake import Wake
from VSM.core.AirfoilAerodynamics import AirfoilAerodynamics
from VSM.core.utils import (
    intersect_line_with_plane,
    point_in_quad,
    assemble_AIC_matrices,
    induced_velocity_at_points,
    assemble_bound_vortex_AIC,
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
    ):
        self._wings = wings
        self._aerodynamic_center_location = aerodynamic_center_location
        self._control_point_location = control_point_location

        self._bridle_line_system = bridle_line_system
        self._reference_point = np.zeros(3, dtype=float)
        self.cd_cable = 1.1
        self.cf_cable = 0.01

        # Track the cumulative geometry rotation applied to the body.
        self.geometry_rotation = np.eye(3)

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
        aerodynamic_center_location: float = 0.25,
        control_point_location: float = 0.75,
    ):
        """
        Instantiate a BodyAerodynamics object from either a provided wing_instance or a YAML config file.

        Args:
            n_panels (int): Number of panels (required if wing_instance is not provided).
            file_path (str, optional): Path to the YAML config file. If None, wing_instance must be provided.
            wing_instance (Wing, optional): Pre-built Wing instance. If None, file_path must be provided.
            spanwise_panel_distribution (str): Panel distribution type (default: 'linear').
            is_with_bridles (bool): Whether to include bridle lines (default: False).
            aerodynamic_center_location (float): Chord fraction (from LE) of the aerodynamic
                center / bound vortex (default: 0.25).
            control_point_location (float): Chord fraction (from LE) of the control point
                (default: 0.75).

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

                d = bridle_lines_dict[name]["d"]

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
            return cls(
                [wing_instance],
                bridle_lines,
                aerodynamic_center_location=aerodynamic_center_location,
                control_point_location=control_point_location,
            )
        else:
            return cls(
                [wing_instance],
                aerodynamic_center_location=aerodynamic_center_location,
                control_point_location=control_point_location,
            )

    def rotate(
        self,
        *,
        angle_deg: float | None = None,
        angle_rad: float | None = None,
        axis: np.ndarray | list = np.array([0.0, 0.0, 1.0]),
        point: np.ndarray | list = np.zeros(3),
    ) -> None:
        """Rotate the full geometry about an axis passing through a point."""

        if (angle_deg is None) == (angle_rad is None):
            raise ValueError("Provide exactly one of angle_deg or angle_rad.")

        theta = np.deg2rad(angle_deg) if angle_rad is None else float(angle_rad)

        axis_vec = np.asarray(axis, dtype=float)
        if axis_vec.shape != (3,):
            raise ValueError(f"axis must be a 3-vector, got shape {axis_vec.shape}")
        norm = np.linalg.norm(axis_vec)
        if norm == 0:
            raise ValueError("axis must be non-zero.")
        axis_vec /= norm

        origin = np.asarray(point, dtype=float)
        if origin.shape != (3,):
            raise ValueError(
                f"point must be a 3-vector specifying the rotation center, got {origin.shape}"
            )

        # Rodrigues' rotation formula
        kx, ky, kz = axis_vec
        K = np.array(
            [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]],
            dtype=float,
        )
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        def rotate_point(p: np.ndarray) -> np.ndarray:
            return origin + R @ (np.asarray(p, dtype=float) - origin)

        for wing in self.wings:
            for section in wing.sections:
                section.LE_point = rotate_point(section.LE_point)
                section.TE_point = rotate_point(section.TE_point)

            # Keep spanwise direction aligned with the rotated geometry
            rotated_span = R @ wing.spanwise_direction
            norm_rotated_span = np.linalg.norm(rotated_span)
            if norm_rotated_span == 0:
                raise ValueError("Rotation produced zero spanwise direction vector.")
            wing.spanwise_direction = rotated_span / norm_rotated_span

        self.geometry_rotation = R @ self.geometry_rotation

        self._build_panels()

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
    def reference_point(self):
        """``r0`` of the rotational inflow, as last given to the ``va`` setter.

        Paired with :attr:`body_rates` and :attr:`va` (the freestream) this is
        everything a consumer needs to evaluate the inflow at its own station,
        ``va(r) = va_free - omega x (r - r0)``, instead of borrowing the wing's.
        """
        return self._reference_point

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
        body_rates: np.ndarray | float = 0.0,
        body_axis: np.ndarray | float | None = None,
        reference_point: np.ndarray | None = None,
        rates_in_body_frame: bool = False,
    ) -> None:
        """
        Set the apparent velocity distribution and optional rigid-body rates.

        Parameters
        ----------
        va : array-like
            - shape (3,) for a uniform freestream vector, or
            - shape (n_panels, 3) for a per-panel apparent velocity.
        body_rates : float or array-like
            Single angular rate or a sequence of rates (rad/s). When multiple rates are
            provided they are paired with the corresponding axes in body_axis.
        body_axis : array-like, optional
            Single 3-vector or an array of shape (n_rates, 3) for the body rotation
            axes (need not be unit length). If None, defaults to +z for all rates.
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

        rates_array = np.atleast_1d(np.asarray(body_rates, dtype=float)).ravel()
        if rates_array.ndim != 1:
            raise ValueError(
                f"body_rates must be a scalar or 1D array, got shape {rates_array.shape}"
            )

        if body_axis is None:
            axes_array = np.tile(
                np.array([0.0, 0.0, 1.0], dtype=float), (len(rates_array), 1)
            )
        else:
            axes_array = np.asarray(body_axis, dtype=float)
            if axes_array.shape == (3,):
                axes_array = np.tile(axes_array, (len(rates_array), 1))
            elif axes_array.shape != (len(rates_array), 3):
                raise ValueError(
                    "body_axis must be shape (3,) or (n_rates, 3); "
                    f"got {axes_array.shape} for {len(rates_array)} rates"
                )

        axis_units = []
        for axis_vec in axes_array:
            norm = np.linalg.norm(axis_vec)
            if norm < 1e-12:
                raise ValueError("body_axis must be nonzero")
            axis_units.append(axis_vec / norm)

        omega_vec = np.zeros(3, dtype=float)
        for rate, axis_unit in zip(rates_array, axis_units):
            omega_vec += rate * axis_unit  # combine multiple rate/axis contributions

        # store body rates for introspection
        self._body_rates = omega_vec
        if not rates_in_body_frame:
            self._body_rates = self.geometry_rotation @ self._body_rates

        # r0 is stored unconditionally, not just on the rotating branch: the
        # pair (body_rates, reference_point) is what any consumer needs to
        # evaluate va(r) = va_free - omega x (r - r0) at a station of its own
        # -- a bridle segment, a KCU, a tether attachment. Leaving it on the
        # stack meant such a consumer could only guess the origin, and guessing
        # right is indistinguishable from guessing wrong until r0 moves.
        r0 = (
            np.zeros(3, dtype=float)
            if reference_point is None
            else np.asarray(reference_point, dtype=float)
        )
        if r0.shape != (3,):
            raise ValueError(f"reference_point must be shape (3,), got {r0.shape}")
        self._reference_point = r0

        # add rotational inflow only if any rate is nonzero
        if np.any(self._body_rates != 0.0):
            control_points = np.array(
                [p.control_point for p in self.panels], dtype=float
            )
            # v_rot = [p,q,r] x (r - r0)
            va_distribution += -np.cross(self._body_rates, control_points - r0)

        # push to panels
        for i, panel in enumerate(self.panels):
            panel.va = va_distribution[i]

        # update wake using the new distribution
        self.panels = Wake.frozen_wake(va_distribution, self.panels)

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
                coordinates[2 * i, :] * (1 - ac)
                + coordinates[2 * i + 1, :] * ac
                - (
                    coordinates[2 * i + 2, :] * (1 - ac)
                    + coordinates[2 * i + 3, :] * ac
                )
            )
            if i == 0:
                diplus = jit_norm(
                    coordinates[2 * (i + 1), :] * (1 - ac)
                    + coordinates[2 * (i + 1) + 1, :] * ac
                    - (
                        coordinates[2 * (i + 1) + 2, :] * (1 - ac)
                        + coordinates[2 * (i + 1) + 3, :] * ac
                    )
                )
                ncp = di / (di + diplus)
            elif i == n_panels - 1:
                dimin = jit_norm(
                    coordinates[2 * (i - 1), :] * (1 - ac)
                    + coordinates[2 * (i - 1) + 1, :] * ac
                    - (
                        coordinates[2 * (i - 1) + 2, :] * (1 - ac)
                        + coordinates[2 * (i - 1) + 3, :] * ac
                    )
                )
                ncp = dimin / (dimin + di)
            else:
                dimin = jit_norm(
                    coordinates[2 * (i - 1), :] * (1 - ac)
                    + coordinates[2 * (i - 1) + 1, :] * ac
                    - (
                        coordinates[2 * (i - 1) + 2, :] * (1 - ac)
                        + coordinates[2 * (i - 1) + 3, :] * ac
                    )
                )
                diplus = jit_norm(
                    coordinates[2 * (i + 1), :] * (1 - ac)
                    + coordinates[2 * (i + 1) + 1, :] * ac
                    - (
                        coordinates[2 * (i + 1) + 2, :] * (1 - ac)
                        + coordinates[2 * (i + 1) + 3, :] * ac
                    )
                )
                ncp = ac * (dimin / (dimin + di) + di / (di + diplus) + 1)

            ncp = 1 - ncp

            # aerodynamic center at the aerodynamic_center_location (default 1/4c)
            # LE/TE blend weights must sum to 1 to stay on the chord line, so the
            # LE weight is (1 - ac), independent of the control_point_location.
            LLpoint = (section["p2"] * (1 - ncp) + section["p1"] * ncp) * (1 - ac) + (
                section["p3"] * (1 - ncp) + section["p4"] * ncp
            ) * ac
            # control point at the control_point_location (default 3/4c)
            VSMpoint = (section["p2"] * (1 - ncp) + section["p1"] * ncp) * (1 - cp) + (
                section["p3"] * (1 - ncp) + section["p4"] * ncp
            ) * cp

            # Calculating the bound (located at the aerodynamic center)
            bound_1 = section["p1"] * (1 - ac) + section["p4"] * ac
            bound_2 = section["p2"] * (1 - ac) + section["p3"] * ac

            ### Calculate the local reference frame, below are all unit_vectors.
            # The frame is orthonormal and built on the bound-vortex (quarter
            # chord) axis, so the inner 2D section is the one perpendicular to
            # the local span (Crossflow Principle, CP1 in Gaunaa, Li & Pirrung,
            # TORQUE 2026). On swept panels the raw LE-to-TE chord direction
            # is not perpendicular to the bound vortex; its spanwise component
            # is removed here.
            # SPAN z_airf along the bound vortex (towards left tip, along span)
            z_airf = bound_1 - bound_2
            z_airf = z_airf / jit_norm(z_airf)

            # TANGENTIAL y_airf: chord-line direction LE-to-TE, projected
            # perpendicular to the span
            y_airf = VSMpoint - LLpoint
            y_airf = y_airf - jit_dot(y_airf, z_airf) * z_airf
            y_airf = y_airf / jit_norm(y_airf)

            # NORMAL x_airf upwards from the chord-line, perpendicular to both
            x_airf = jit_cross(y_airf, z_airf)
            x_airf = x_airf / jit_norm(x_airf)

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

    @staticmethod
    def _compute_reference_velocity_from_distribution(
        va_input: np.ndarray,
        n_panels: int,
        panel_areas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a single reference velocity vector from uniform or distributed inflow.

        For distributed inflow, the speed is area-weighted RMS and the direction is
        area-weighted mean direction.
        """
        va_input = np.asarray(va_input, dtype=float)
        if va_input.shape == (3,):
            return va_input
        if va_input.shape != (n_panels, 3):
            raise ValueError(
                f"'va' must be shape (3,) or ({n_panels}, 3); got {va_input.shape}"
            )

        if panel_areas is None:
            panel_areas = np.ones(n_panels, dtype=float)
        else:
            panel_areas = np.asarray(panel_areas, dtype=float)
            if panel_areas.shape != (n_panels,):
                raise ValueError(
                    f"panel_areas must be shape ({n_panels},), got {panel_areas.shape}"
                )

        total_area = float(np.sum(panel_areas))
        if total_area <= 0.0:
            raise ValueError("Total panel area must be positive.")

        speeds = np.linalg.norm(va_input, axis=1)
        reference_speed = float(np.sqrt(np.sum(panel_areas * speeds**2) / total_area))

        direction = np.sum(panel_areas[:, None] * va_input, axis=0)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 0.0:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)
            direction_norm = 1.0

        return direction / direction_norm * reference_speed

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
        wake_units, wake_speeds = self._wake_directions(va_norm_array, va_unit_array)

        # Assembled in one numba-compiled double loop (utils.assemble_AIC_matrices)
        # over packed filament geometry; per-filament kernels are jit ports of
        # Filament.py, matching it to floating-point round-off. The per-panel
        # Python loop is kept verbatim in _compute_AIC_matrices_reference as the
        # readable reference and regression oracle.
        eval_points = np.ascontiguousarray(
            [getattr(panel, evaluation_point) for panel in self.panels], dtype=float
        )
        bound_point_1 = np.ascontiguousarray(
            [panel.bound_point_1 for panel in self.panels], dtype=float
        )
        bound_point_2 = np.ascontiguousarray(
            [panel.bound_point_2 for panel in self.panels], dtype=float
        )
        TE_point_1 = np.ascontiguousarray(
            [panel.TE_point_1 for panel in self.panels], dtype=float
        )
        TE_point_2 = np.ascontiguousarray(
            [panel.TE_point_2 for panel in self.panels], dtype=float
        )
        AIC = assemble_AIC_matrices(
            eval_points,
            bound_point_1,
            bound_point_2,
            TE_point_1,
            TE_point_2,
            wake_units,
            wake_speeds,
            float(core_radius_fraction),
            evaluation_point_on_bound,
            aerodynamic_model_type == "VSM",
        )
        return AIC[0], AIC[1], AIC[2]

    def _wake_directions(self, va_norm_array, va_unit_array):
        """Per-panel direction and speed of the frozen wake: each ring's
        semi-infinite filaments follow that panel's own apparent velocity
        (freestream plus the body-rate term, or the distributed inflow), so a
        rotating body or a spanwise-varying inflow gets a locally aligned wake.
        In a uniform inflow every panel gets the same direction, the classical
        straight wake along the freestream.
        """
        wake_speeds = np.ascontiguousarray(va_norm_array, dtype=float).ravel()
        wake_units = np.ascontiguousarray(va_unit_array, dtype=float)
        # a single speed and direction (the classical shared frozen wake) is
        # broadcast to every ring
        wake_units = wake_units.reshape(-1, 3)
        if wake_speeds.size != self.n_panels and np.allclose(wake_speeds, wake_speeds[0]):
            wake_speeds = np.full(self.n_panels, float(wake_speeds[0]))
        if wake_units.shape[0] != self.n_panels and np.allclose(wake_units, wake_units[0]):
            wake_units = np.ascontiguousarray(np.tile(wake_units[0], (self.n_panels, 1)))
        if wake_speeds.shape != (self.n_panels,) or wake_units.shape != (
            self.n_panels,
            3,
        ):
            raise ValueError(
                "va_norm_array must be (n_panels,) and va_unit_array (n_panels, 3)."
            )
        if np.any(wake_speeds <= 0.0):
            raise ValueError("Wake reference speed must be positive on every panel.")
        return wake_units, wake_speeds

    def _compute_AIC_matrices_reference(
        self, aerodynamic_model_type, core_radius_fraction, va_norm_array, va_unit_array
    ):
        """Reference (pure-Python) AIC assembly: the original per-panel loop over
        Panel.compute_velocity_induced_single_ring_semiinfinite. Kept as the
        readable formulation and as the oracle for regression-testing the
        numba-compiled fast path in :meth:`compute_AIC_matrices`.
        """
        if aerodynamic_model_type not in ["VSM", "LLT"]:
            raise ValueError("Invalid aerodynamic model type, should be VSM or LLT")

        evaluation_point = (
            "control_point" if aerodynamic_model_type == "VSM" else "aerodynamic_center"
        )
        evaluation_point_on_bound = aerodynamic_model_type == "LLT"
        wake_units, wake_speeds = self._wake_directions(va_norm_array, va_unit_array)

        AIC = np.empty((3, self.n_panels, self.n_panels))

        for icp, panel_icp in enumerate(self.panels):
            ep = getattr(panel_icp, evaluation_point)
            for jring, panel_jring in enumerate(self.panels):
                velocity_induced = (
                    panel_jring.compute_velocity_induced_single_ring_semiinfinite(
                        ep,
                        evaluation_point_on_bound,
                        wake_speeds[jring],
                        wake_units[jring],
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

        The spanwise control-point coordinate is measured relative to the wing
        center along self.wings[0].spanwise_direction, so the result is independent
        of the global origin.

        Args:
            gamma_0 (float): The circulation at the wing root.

        Returns:
            np.array: The circulation distribution following an elliptical profile.
        """
        if len(self.wings) > 1:
            raise NotImplementedError("Multiple wings not yet implemented")

        wing = self.wings[0]
        spanwise_axis_norm = np.linalg.norm(wing.spanwise_direction)
        if spanwise_axis_norm == 0:
            raise ValueError("Wing spanwise_direction must be non-zero.")

        spanwise_axis = wing.spanwise_direction / spanwise_axis_norm
        wing_points = np.concatenate(
            [[section.LE_point, section.TE_point] for section in wing.sections]
        )
        spanwise_projections = np.dot(wing_points, spanwise_axis)
        spanwise_min = np.min(spanwise_projections)
        spanwise_max = np.max(spanwise_projections)
        wing_span = spanwise_max - spanwise_min
        if wing_span <= 0:
            raise ValueError("Wing span must be positive for elliptical circulation.")

        spanwise_center = 0.5 * (spanwise_min + spanwise_max)
        logging.debug(f"wing_span: {wing_span}")

        # Spanwise coordinate relative to the wing center.
        spanwise_coordinate = np.array(
            [
                np.dot(panel.control_point, spanwise_axis) - spanwise_center
                for panel in self.panels
            ]
        )

        # Compute the elliptical circulation distribution.
        radicand = 1 - (2 * spanwise_coordinate / wing_span) ** 2
        if np.any(radicand < -1e-12):
            raise ValueError(
                "Panel control points extend outside the wing span used for "
                "elliptical circulation initialization."
            )
        gamma_i = gamma_0 * np.sqrt(np.maximum(radicand, 0.0))

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
            logging.warning("Force vector is zero, cannot compute center of pressure.")
            return None

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

        logging.debug(
            "No intersection found with any panel, in the center_of_pressure calculation."
        )
        return None

    @staticmethod
    def viscous_drag_correction(v_rel, chord, dir_induced_va, z_airf, rho, mu):
        """Spanwise-flow correction of the viscous (friction) force, per unit
        span, following Gaunaa, Sorensen & Li 2024, J. Phys.: Conf. Ser. 2767
        022068 (doi:10.1088/1742-6596/2767/2/022068), Eqs. 4, 10 and 11.

        Args:
            v_rel: full 3D relative velocity at the section (freestream plus
                induced), including its spanwise component.
            chord: section chord (perpendicular to the span).
            dir_induced_va: unit direction of the span-perpendicular inner
                flow, i.e. the direction the profile drag acts along.
            z_airf: unit spanwise axis of the panel.

        Returns:
            (f_corr_drag, f_corr_span): the drag increment along
            ``dir_induced_va`` and the spanwise friction force along
            ``z_airf`` (signed with the spanwise flow), both per unit span.
        """
        # beta: angle between the relative flow and the span-normal plane
        v_par = np.dot(v_rel, z_airf)
        v_perp = np.linalg.norm(v_rel - v_par * z_airf)
        if v_perp <= 0.0:
            return np.zeros(3), np.zeros(3)
        beta = np.arctan2(v_par, v_perp)
        cos_beta = np.cos(beta)

        # reference Reynolds number and dynamic pressure on V_perp (Eq. 4)
        Re_ref = rho * v_perp * chord / mu
        q_perp = 0.5 * rho * v_perp**2
        f0 = 0.062 * Re_ref ** (-1.0 / 7.0)

        # Eq. 10: increment of the profile drag coefficient
        delta_cd = f0 * (cos_beta ** (-5.0 / 7.0) - 1.0)
        # Eq. 11: spanwise friction force coefficient (sign of beta)
        c_par = f0 * np.tan(beta) * cos_beta ** (-5.0 / 7.0)

        f_corr_drag = delta_cd * q_perp * chord * dir_induced_va
        f_corr_span = c_par * q_perp * chord * z_airf
        return f_corr_drag, f_corr_span

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

            ###  Commented this on 11/02/2026, and replaced with improved version that DOES preserve moments
            # # 3) force in the chord plane (perpendicular to span)
            # F_perp = F - np.dot(F, z_airf) * z_airf
            # F_perp_mag = np.linalg.norm(F_perp)

            # # 4) if there's no pitching‐plane force, fallback to AC
            # if F_perp_mag < 1e-12:
            #     panel_cp_locations.append(ac)
            #     continue

            F_perp_mag = np.dot(np.cross(y_airf, F), z_airf)

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

    def compute_trefftz_plane_induced_drag(self, gamma, rho):
        """Induced drag from the Trefftz plane, for a steady translating wing in
        a uniform inflow (Katz & Plotkin, Sec. 8.2). The wake filaments are
        projected onto the plane perpendicular to the freestream, where each is
        a 2D vortex of strength equal to the jump in bound circulation it
        trails, and the drag is the Kutta-Joukowski force of half the far-wake
        velocity on the wake trace carrying the bound circulation:

            D = 1/2 rho sum_i Gamma_i ((V_T,i x l_i) . e_inf),

        with l_i the trace segment of panel i between its two wake origins
        (projected) and V_T,i the 2D velocity of all wake vortices at its
        midpoint. This does not depend on where the forces are evaluated on
        the blade, so it is the reference the on-blade induced drag
        (quarter-chord directions plus the attached-trailed force) is checked
        against. Returns None when the inflow is not one uniform vector
        (per-panel distribution or body rates), where the argument does not
        apply.
        """
        va = np.asarray(self._va, dtype=float)
        if va.shape != (3,) or np.any(self._body_rates != 0.0):
            return None
        va_norm = np.linalg.norm(va)
        if va_norm <= 0.0:
            return None
        e_inf = va / va_norm
        panels = self.panels
        gamma = np.asarray(gamma, dtype=float).ravel()
        n = len(panels)
        # wake origins, one per section boundary, with the circulation jump each
        # trails (bound -> TE -> downstream sense, as the semi-infinite filaments)
        origins = np.vstack(
            [[p.TE_point_1 for p in panels], panels[-1].TE_point_2[None, :]]
        ).astype(float)
        d_gamma = np.concatenate(([gamma[0]], gamma[1:] - gamma[:-1], [-gamma[-1]]))
        q = origins - np.outer(origins @ e_inf, e_inf)  # projected onto the plane
        drag = 0.0
        for i in range(n):
            mid = 0.5 * (q[i] + q[i + 1])
            d = mid - q
            r2 = np.sum(d * d, axis=1)
            v_t = np.sum(
                (d_gamma / (2.0 * np.pi * r2))[:, None] * np.cross(e_inf, d), axis=0
            )
            # trace segment in the sense of the bound vortex (section i+1 -> i)
            drag += gamma[i] * np.dot(np.cross(v_t, q[i] - q[i + 1]), e_inf)
        return 0.5 * rho * drag

    def compute_attached_trailed_vortex_forces(
        self, gamma, rho, core_radius_fraction, va_array
    ):
        """Kutta-Joukowski force on the attached trailed (AT) vortex segments,
        the chordwise legs running from the bound vortex to the trailing edge
        (Gaunaa, Li & Pirrung, TORQUE 2026, Sec. 3).

        The two legs of neighbouring panels coincide on their shared section
        boundary with opposite sense, so boundary ``j`` (``j = 0..n``) carries
        the net circulation ``gamma_j - gamma_{j-1}`` in the bound-to-TE
        direction (``gamma_{-1} = gamma_n = 0``). The force on boundary ``j``
        is ``rho * Gamma_net_j * (V x l_j)`` with ``l_j`` the leg vector and
        ``V`` the full 3D relative velocity at the section's 3/4-chord point
        on that leg (the paper's choice: the point where the flow has no
        component through the wing). For an unswept wing these forces are
        spanwise and cancel; for swept wings they change lift and induced drag.

        Returns:
            (forces, points): ``(n+1, 3)`` force vectors and the points they act at.
        """
        panels = self.panels
        n = len(panels)
        gamma = np.asarray(gamma, dtype=float).ravel()
        gamma_net = np.zeros(n + 1)
        gamma_net[:n] += gamma  # leg bound_1 -> TE_1 of panel j sits on boundary j
        gamma_net[1:] -= gamma  # leg TE_2 -> bound_2 of panel j sits on boundary j+1

        bound_point_1 = np.ascontiguousarray(
            [p.bound_point_1 for p in panels], dtype=float
        )
        bound_point_2 = np.ascontiguousarray(
            [p.bound_point_2 for p in panels], dtype=float
        )
        TE_point_1 = np.ascontiguousarray([p.TE_point_1 for p in panels], dtype=float)
        TE_point_2 = np.ascontiguousarray([p.TE_point_2 for p in panels], dtype=float)

        bound = np.vstack([bound_point_1, bound_point_2[-1:]])
        te = np.vstack([TE_point_1, TE_point_2[-1:]])
        legs = te - bound
        # 3/4-chord point of the section, on the leg (bound at ac, TE at 1)
        ac = self._aerodynamic_center_location
        cp = self._control_point_location
        t = (cp - ac) / (1.0 - ac)
        points = np.ascontiguousarray(bound + t * legs)

        # inflow at a boundary: mean of the neighbouring panels' inflow
        va_array = np.asarray(va_array, dtype=float)
        va_boundary = np.zeros((n + 1, 3))
        va_boundary[:n] += va_array
        va_boundary[1:] += va_array
        va_boundary[1:n] *= 0.5

        va_speeds = np.linalg.norm(va_array, axis=1)
        wake_units, wake_speeds = self._wake_directions(
            va_speeds, va_array / va_speeds[:, None]
        )
        v_ind = induced_velocity_at_points(
            points,
            bound_point_1,
            bound_point_2,
            TE_point_1,
            TE_point_2,
            gamma,
            wake_units,
            wake_speeds,
            float(core_radius_fraction),
        )
        v_rel = va_boundary + v_ind
        forces = rho * gamma_net[:, None] * np.cross(v_rel, legs)
        return forces, points

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
        relative_velocity_array=None,
        is_with_attached_trailed_vortex_force=False,
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
        lift = (cl_array * 0.5 * rho * Umag_array**2 * chord_array)[:, np.newaxis]
        drag = (cd_array * 0.5 * rho * Umag_array**2 * chord_array)[:, np.newaxis]
        moment = (cm_array * 0.5 * rho * Umag_array**2 * chord_array**2)[:, np.newaxis]

        if is_aoa_corrected:
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

        else:
            alpha_corrected = alpha_array[:, np.newaxis]
            alpha_uncorrected = alpha_array[:, np.newaxis]

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
        # Reference inflow used for global decomposition and global coefficient
        # normalization.
        panel_areas = chord_array * panel_width_array
        va_ref_vector = self._compute_reference_velocity_from_distribution(
            self._va, len(panels), panel_areas
        )

        va_ref_mag = jit_norm(va_ref_vector)
        if va_ref_mag <= 0.0:
            raise ValueError("Reference freestream magnitude must be positive.")
        va_ref_unit = va_ref_vector / va_ref_mag
        dir_lift_ref = jit_cross(va_ref_vector, spanwise_direction)
        dir_lift_ref_norm = jit_norm(dir_lift_ref)
        if dir_lift_ref_norm <= 0.0:
            raise ValueError(
                "Reference lift direction is undefined because reference flow is "
                "parallel to spanwise direction."
            )
        dir_lift_ref = dir_lift_ref / dir_lift_ref_norm
        dir_side_ref = jit_cross(dir_lift_ref, va_ref_unit)

        q_ref = 0.5 * rho * va_ref_mag**2
        if reference_point is None:
            reference_point = np.zeros(3)
        reference_point = np.asarray(reference_point, dtype=float)
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
            # (TAT3: the local inner flow direction, which lies in the panel's
            # span-perpendicular plane; the global span axis is NOT used here)
            dir_drag_induced_va = dir_induced_va_airfoil

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
            va_panel = va_array[i]
            va_panel_mag = va_norm_array[i]
            if va_panel_mag <= 0.0:
                raise ValueError(
                    f"Panel {i} has non-positive apparent velocity magnitude."
                )
            va_panel_unit = va_unit_array[i]
            q_panel = 0.5 * rho * va_panel_mag**2

            dir_lift_prescribed_va = jit_cross(va_panel, spanwise_direction)
            dir_lift_prescribed_va = dir_lift_prescribed_va / jit_norm(
                dir_lift_prescribed_va
            )
            lift_prescribed_va = jit_dot(
                lift_induced_va, dir_lift_prescribed_va
            ) + jit_dot(drag_induced_va, dir_lift_prescribed_va)
            drag_prescribed_va = jit_dot(lift_induced_va, va_panel_unit) + jit_dot(
                drag_induced_va, va_panel_unit
            )

            dir_side = jit_cross(dir_lift_prescribed_va, va_panel_unit)
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
                # Full 3D relative velocity at the control point (the spanwise
                # component is what the correction is about). Without it the
                # correction degenerates to the freestream.
                v_rel_i = (
                    relative_velocity_array[i]
                    if relative_velocity_array is not None
                    else va_panel
                )
                f_corr_drag, f_corr_span = self.viscous_drag_correction(
                    v_rel=v_rel_i,
                    chord=panel_chord,
                    dir_induced_va=dir_induced_va_airfoil,
                    z_airf=z_airf_span,
                    rho=rho,
                    mu=mu,
                )
                f_corr = f_corr_drag + f_corr_span
                ftotal_induced_va += f_corr

                # Decompose the correction into the prescribed (D, L, S) basis
                drag_prescribed_va += np.dot(f_corr, va_panel_unit)
                lift_prescribed_va += np.dot(f_corr, dir_lift_prescribed_va)
                side_prescribed_va += np.dot(f_corr, dir_side)

            # ----------------------------------
            ####################################

            ### Converting forces to the global reference frame
            fx_global_2D = jit_dot(ftotal_induced_va, np.array([1, 0, 0]))
            fy_global_2D = jit_dot(ftotal_induced_va, np.array([0, 1, 0]))
            fz_global_2D = jit_dot(ftotal_induced_va, np.array([0, 0, 1]))

            # 3D, by multiplying with the panel width
            lift_wing_3D_local = lift_prescribed_va * panel_width
            drag_wing_3D_local = drag_prescribed_va * panel_width
            side_wing_3D_local = side_prescribed_va * panel_width
            # Build panel force components in local (L, D, S) directions first,
            # then project each on the single global reference axis.
            lift_wing_3D = jit_dot(
                lift_wing_3D_local * dir_lift_prescribed_va, dir_lift_ref
            )
            drag_wing_3D = jit_dot(drag_wing_3D_local * va_panel_unit, va_ref_unit)
            side_wing_3D = jit_dot(side_wing_3D_local * dir_side, dir_side_ref)
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
            cl_prescribed_va_list.append(lift_prescribed_va / (q_panel * panel_chord))
            cd_prescribed_va_list.append(drag_prescribed_va / (q_panel * panel_chord))
            cs_prescribed_va_list.append(side_prescribed_va / (q_panel * panel_chord))
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

            # (3) Force application point used for r x F:
            #     use aerodynamic center (1/4c) for all models.
            panel_force_point_global = panel_i.aerodynamic_center

            # (4) Vector from panel force point to the chosen reference point:
            r_vector = panel_force_point_global - reference_point

            # (5) Cross product to shift the force to the reference point:
            force_global_3D = np.array([fx_global_3D, fy_global_3D, fz_global_3D])
            M_shift = np.cross(r_vector, force_global_3D)

            # (6) Total panel moment about the reference point:
            M_ref_panel = M_local_3D + M_shift

            # (7) Accumulate or store:
            mx_global_3D_sum += M_ref_panel[0]
            my_global_3D_sum += M_ref_panel[1]
            mz_global_3D_sum += M_ref_panel[2]
            mx_global_3D_list.append(M_ref_panel[0])
            my_global_3D_list.append(M_ref_panel[1])
            mz_global_3D_list.append(M_ref_panel[2])
            m_global_3D_list.append(M_ref_panel)

        ### Attached trailed (AT) vortex forces (Gaunaa, Li & Pirrung 2026, Sec. 3)
        f_at_list = []
        if is_with_attached_trailed_vortex_force:
            f_at, p_at = self.compute_attached_trailed_vortex_forces(
                gamma_new, rho, core_radius_fraction, va_array
            )
            n_p = len(panels)
            for j in range(n_p + 1):
                force_j = f_at[j]
                moment_j = np.cross(p_at[j] - reference_point, force_j)
                # split each boundary force over its two neighbouring panels
                if j == 0:
                    shares = [(0, 1.0)]
                elif j == n_p:
                    shares = [(n_p - 1, 1.0)]
                else:
                    shares = [(j - 1, 0.5), (j, 0.5)]
                for i_p, w in shares:
                    f_global_3D_list[i_p] = f_global_3D_list[i_p] + w * force_j
                    fx_global_3D_list[i_p] += w * force_j[0]
                    fy_global_3D_list[i_p] += w * force_j[1]
                    fz_global_3D_list[i_p] += w * force_j[2]
                    m_global_3D_list[i_p] = m_global_3D_list[i_p] + w * moment_j
                    mx_global_3D_list[i_p] += w * moment_j[0]
                    my_global_3D_list[i_p] += w * moment_j[1]
                    mz_global_3D_list[i_p] += w * moment_j[2]
                fx_global_3D_sum += force_j[0]
                fy_global_3D_sum += force_j[1]
                fz_global_3D_sum += force_j[2]
                lift_wing_3D_sum += jit_dot(force_j, dir_lift_ref)
                drag_wing_3D_sum += jit_dot(force_j, va_ref_unit)
                side_wing_3D_sum += jit_dot(force_j, dir_side_ref)
                mx_global_3D_sum += moment_j[0]
                my_global_3D_sum += moment_j[1]
                mz_global_3D_sum += moment_j[2]
                f_at_list.append(force_j)

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
        reynolds_number = rho * va_ref_mag * max_chord / mu

        bridle_line_force_list = []
        bridle_line_midpoint_list = []
        if self._bridle_line_system is not None:
            # Each segment is charged at its OWN inflow, rather than one
            # vector standing in for the whole bridle. The panel inflow this
            # class builds is
            #     va(r) = va_free - omega x (r - r0)        (see the va setter)
            # but ``va_ref_vector`` carries NO rotational term at all: it is
            # ``_compute_reference_velocity_from_distribution(self._va, ...)``
            # and ``self._va`` is the inflow exactly as HANDED to the setter,
            # before v_rot is added. For the usual uniform case that IS the
            # freestream. So every bridle segment used to be charged va_free --
            # the inflow at the reference point -- wherever the segment
            # actually sits. On a rotating body that is the wrong dynamic
            # pressure on every segment except those at r0, in the force and in
            # its moment about the reference point, which is the balance a
            # caller's trim is solved against. With no body rates the two are
            # identical, so only turning/steered cases move.
            #
            # ``self._va`` is the freestream exactly as handed to the setter. A
            # caller who supplied a full per-panel distribution has no single
            # freestream to extrapolate from, so that case keeps the reference
            # vector.
            omega_body = np.asarray(self._body_rates, dtype=float).ravel()
            va_freestream = np.asarray(self._va, dtype=float)
            is_per_segment_inflow = (
                va_freestream.shape == (3,)
                and omega_body.shape == (3,)
                and np.any(omega_body)
            )

            # Calculate forces and moments for each bridle line individually
            for bridle_line in self._bridle_line_system:
                # Bridle line midpoint: both the inflow station and the moment
                # application point.
                bridle_midpoint = 0.5 * (bridle_line[0] + bridle_line[1])

                # Calculate force for this individual bridle line
                va_bridle_line = (
                    va_freestream
                    - np.cross(omega_body, bridle_midpoint - reference_point)
                    if is_per_segment_inflow
                    else va_ref_vector
                )
                fa_bridle_line = self.compute_line_aerodynamic_force(
                    va_bridle_line, bridle_line, rho=rho
                )
                bridle_line_force_list.append(fa_bridle_line)
                bridle_line_midpoint_list.append(bridle_midpoint)

                # Add bridle forces to global force totals
                fx_global_3D_sum += fa_bridle_line[0]
                fy_global_3D_sum += fa_bridle_line[1]
                fz_global_3D_sum += fa_bridle_line[2]
                lift_wing_3D_sum += jit_dot(fa_bridle_line, dir_lift_ref)
                drag_wing_3D_sum += jit_dot(fa_bridle_line, va_ref_unit)
                side_wing_3D_sum += jit_dot(fa_bridle_line, dir_side_ref)

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
        results_dict.update(
            [("drag_induced_trefftz", self.compute_trefftz_plane_induced_drag(gamma_new, rho))]
        )
        results_dict.update([("side", side_wing_3D_sum)])
        results_dict.update([("cl", lift_wing_3D_sum / (q_ref * projected_area))])
        results_dict.update([("cd", drag_wing_3D_sum / (q_ref * projected_area))])
        results_dict.update([("cs", side_wing_3D_sum / (q_ref * projected_area))])
        results_dict.update(
            [("cmx", mx_global_3D_sum / (q_ref * projected_area * max_chord))]
        )
        results_dict.update(
            [("cmy", my_global_3D_sum / (q_ref * projected_area * max_chord))]
        )
        results_dict.update(
            [("cmz", mz_global_3D_sum / (q_ref * projected_area * max_chord))]
        )
        # Local panel aerodynamics
        results_dict.update([("cl_distribution", cl_prescribed_va_list)])
        results_dict.update([("cd_distribution", cd_prescribed_va_list)])
        results_dict.update([("cs_distribution", cs_prescribed_va_list)])
        results_dict.update([("F_distribution", f_global_3D_list)])
        results_dict.update([("F_attached_trailed_distribution", f_at_list)])
        results_dict.update([("M_distribution", m_global_3D_list)])
        # Bridle-line loads exactly as charged above: one row per segment of
        # ``bridle_line_system``, each force paired with the midpoint it acts
        # at. Published so a coupled solver can hand the STRUCTURE the same
        # load the trim balanced, rather than recomputing it and drifting.
        results_dict.update([("bridle_line_forces", np.array(bridle_line_force_list))])
        results_dict.update(
            [("bridle_line_midpoints", np.array(bridle_line_midpoint_list))]
        )

        # Additional info
        results_dict.update(
            [
                (
                    "cfx_distribution",
                    np.array(fx_global_3D_list) / (q_ref * projected_area),
                )
            ]
        )
        results_dict.update(
            [
                (
                    "cfy_distribution",
                    np.array(fy_global_3D_list) / (q_ref * projected_area),
                )
            ]
        )
        results_dict.update(
            [
                (
                    "cfz_distribution",
                    np.array(fz_global_3D_list) / (q_ref * projected_area),
                )
            ]
        )
        results_dict.update(
            [
                (
                    "cm_panel_dist",
                    np.divide(
                        moment[:, 0],
                        0.5 * rho * Umag_array**2 * chord_array**2,
                        out=np.zeros_like(chord_array),
                        where=(Umag_array > 0.0) & (chord_array > 0.0),
                    ),
                )
            ]
        )
        # cm_panel_dist:
        #   sectional airfoil moment coefficient (about local span axis), based on
        #   intrinsic panel moment m' and local normalization q(y) * c(y)^2.
        results_dict.update(
            [
                (
                    "cmx_distribution",
                    np.array(mx_global_3D_list) / (q_ref * projected_area * max_chord),
                )
            ]
        )
        # cmx_distribution:
        #   per-panel contribution to global CMx, using the same normalization
        #   as the global cmx coefficient (q_ref * S_ref * c_ref).
        results_dict.update(
            [
                (
                    "cmy_distribution",
                    np.array(my_global_3D_list)
                    / (q_ref * projected_area * chord_array),
                )
            ]
        )
        results_dict.update(
            [
                (
                    "cmz_distribution",
                    np.array(mz_global_3D_list)
                    / (q_ref * projected_area * chord_array),
                )
            ]
        )
        results_dict.update([("alpha_at_ac", alpha_corrected)])
        results_dict.update([("alpha_uncorrected", alpha_uncorrected)])
        results_dict.update([("alpha_geometric", alpha_geometric)])

        # AoA and sideslip referenced to the center chord local frame.
        center_idx_left = (len(self.panels) - 1) // 2
        center_idx_right = len(self.panels) // 2
        center_panels = [self.panels[center_idx_left], self.panels[center_idx_right]]

        y_center = np.mean([panel.y_airf for panel in center_panels], axis=0)
        x_center = np.mean([panel.x_airf for panel in center_panels], axis=0)
        z_center = np.mean([panel.z_airf for panel in center_panels], axis=0)

        y_center = y_center / np.linalg.norm(y_center)
        x_center = x_center / np.linalg.norm(x_center)
        z_center = z_center / np.linalg.norm(z_center)

        va_input = np.asarray(self._va, dtype=float)
        if va_input.shape == (3,):
            va_center = va_input
        elif va_input.shape == (len(self.panels), 3):
            if center_idx_left == center_idx_right:
                va_center = va_input[center_idx_left]
            else:
                va_center = 0.5 * (
                    va_input[center_idx_left] + va_input[center_idx_right]
                )
        else:
            raise ValueError(
                f"'_va' must be shape (3,) or ({len(self.panels)}, 3); got {va_input.shape}"
            )
        va_center_mag = np.linalg.norm(va_center)
        if va_center_mag < 1e-12:
            alpha_center_chord_deg = np.nan
            beta_center_chord_deg = np.nan
        else:
            v_chord = np.dot(va_center, y_center)
            v_normal = np.dot(va_center, x_center)
            v_span = np.dot(va_center, z_center)
            alpha_center_chord_deg = np.rad2deg(np.arctan2(v_normal, v_chord))
            beta_center_chord_deg = np.rad2deg(
                np.arctan2(v_span, np.linalg.norm([v_chord, v_normal]))
            )

        results_dict.update([("alpha_center_chord_deg", alpha_center_chord_deg)])
        results_dict.update([("beta_center_chord_deg", beta_center_chord_deg)])
        results_dict.update([("gamma_distribution", gamma_new)])
        results_dict.update([("area_all_panels", area_all_panels)])
        results_dict.update([("projected_area", projected_area)])
        results_dict.update([("va_distribution", va_array)])
        results_dict.update([("wing_span", wing_span)])
        results_dict.update([("aspect_ratio_projected", aspect_ratio_projected)])
        results_dict.update([("Rey", reynolds_number)])
        results_dict.update([("q_ref", q_ref)])
        results_dict.update([("va_ref", np.asarray(va_ref_vector, dtype=float))])
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
        body_rates: np.ndarray = 0,
        body_axis: np.ndarray = None,
        reference_point: np.ndarray = None,
        rates_in_body_frame: bool = False,
    ):
        """
        Initializes the apparent velocity (va) and body rates for the WingAero object.

        Parameters:
        Umag (float): Magnitude of the velocity.
        angle_of_attack (float): Angle of attack in degrees.
        side_slip (float): Sideslip angle in degrees.
        body_rate (np.ndarray): Body rates [p, q, r], default is None.
        body_axis (np.ndarray): Axis for body rotation (same frame as other rates), default is None.
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
            body_rates=body_rates,
            body_axis=body_axis,
            reference_point=reference_point,
            rates_in_body_frame=rates_in_body_frame,
        )

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
        # The direction-defining flow is the undisturbed relative flow at the
        # quarter chord (TAT3 in Gaunaa, Li & Pirrung, TORQUE 2026): freestream
        # plus every trailed vortex plus the bound vortices of the OTHER
        # panels (the curved/swept bound-vortex influence of Li et al. 2020).
        # The LLT AIC drops all bound vortices, which is exact only for a
        # straight lifting line; the other panels' bound segments are added
        # back here. The panel's own straight bound segment induces nothing on
        # its own quarter-chord point.
        aerodynamic_model_type = "LLT"
        AIC_x, AIC_y, AIC_z = self.compute_AIC_matrices(
            aerodynamic_model_type, core_radius_fraction, va_norm_array, va_unit_array
        )
        eval_points = np.ascontiguousarray(
            [panel.aerodynamic_center for panel in self.panels], dtype=float
        )
        bound_point_1 = np.ascontiguousarray(
            [panel.bound_point_1 for panel in self.panels], dtype=float
        )
        bound_point_2 = np.ascontiguousarray(
            [panel.bound_point_2 for panel in self.panels], dtype=float
        )
        AIC_bound = assemble_bound_vortex_AIC(
            eval_points, bound_point_1, bound_point_2, float(core_radius_fraction)
        )
        AIC_x = AIC_x + AIC_bound[0]
        AIC_y = AIC_y + AIC_bound[1]
        AIC_z = AIC_z + AIC_bound[2]
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
        alpha_array = np.arctan2(v_normal_array, v_tangential_array)

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

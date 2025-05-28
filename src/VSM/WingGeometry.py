from dataclasses import dataclass, field
import numpy as np
from typing import List
import logging
from . import jit_norm

logging.basicConfig(level=logging.INFO)


@dataclass
class Wing:
    """Class to define a wing object for aerodynamic analysis.

    The Wing class represents a wing geometry composed of multiple sections and provides methods
    to update, refine, and analyze the aerodynamic mesh as well as compute geometric properties.

    Attributes:
        n_panels (int): Number of panels used in the aerodynamic mesh (the number of sections is n_panels + 1).
        spanwise_panel_distribution (str): Spanwise panel distribution type. Options include:
            - "uniform": Linear spacing.
            - "cosine": Cosine spacing.
            - "cosine_van_Garrel": Cosine spacing based on the van Garrel method.
            - "split_provided": Split provided sections to achieve the desired number of panels.
            - "unchanged": Keep the provided sections unchanged.
        spanwise_direction (np.ndarray): Unit vector representing the wing's spanwise direction.
        sections (List[Section]): List of Section objects that define the wing geometry.

    Methods:
        update_wing_from_points(le_arr, te_arr, d_tube_arr, y_camber_arr, aero_input_type):
            Updates the wing geometry using arrays of leading edge points, trailing edge points,
            tube diameters, and camber heights.
        add_section(LE_point, TE_point, aero_input):
            Adds a new section to the wing with the given leading edge, trailing edge, and aerodynamic input.
        find_farthest_point_and_sort(sections):
            Determines a starting section and sorts all sections based on proximity for mesh refinement.
        refine_aerodynamic_mesh():
            Refines the wing's aerodynamic mesh according to the selected spanwise panel distribution.
        refine_mesh_for_uniform_or_cosine_distribution(spanwise_panel_distribution, n_sections, LE, TE, aero_input):
            Refines the mesh using linear or cosine spacing by interpolating leading/trailing edge points
            and aerodynamic input.
        calculate_new_aero_input(aero_input, section_index, left_weight, right_weight):
            Interpolates aerodynamic input data between adjacent sections.
        refine_mesh_by_splitting_provided_sections():
            Splits the provided sections into additional sections to match the desired number of panels.
        calculate_cosine_van_Garrel(new_sections):
            Applies the van Garrel cosine distribution correction to the sections.
        span (property):
            Computes the wing span along the specified spanwise direction.
        calculate_projected_area(z_plane_vector):
            Calculates the projected area of the wing onto a plane defined by the given normal vector.
    """

    # this creates self.n_panels and so on
    n_panels: int
    spanwise_panel_distribution: str = "uniform"
    spanwise_direction: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    sections: List["Section"] = field(default_factory=list)  # child-class

    def update_wing_from_points(
        self,
        le_arr,
        te_arr,
        aero_input_type,
        initial_polar_data,
    ):
        """Update the wing geometry from points
        Args:
            le_arr (np.ndarray): Array of leading edge points.
            te_arr (np.ndarray): Array of trailing edge points.
            aero_input_type (str): Type of aerodynamic input. Options include:
                - "reuse_initial_polar_data": Reuse initial polar data.
        Returns:
            None
        """
        if aero_input_type == "reuse_initial_polar_data":
            # 1. clear
            self.sections.clear()
            # 2. add sections
            for le, te, polar_data in zip(le_arr, te_arr, initial_polar_data):
                self.add_section(le, te, polar_data)
        else:
            raise ValueError(
                f"Unsupported aero model: {aero_input_type}. Supported: lei_airfoil_breukels"
            )

    def add_section(self, LE_point: np.array, TE_point: np.array, aero_input: List):
        """Add a section to the wing
        Args:
            LE_point (np.array): Leading edge point of the section
            TE_point (np.array): Trailing edge point of the section
            aero_input (list): Aerodynamic input for the section
        Returns:
            None
        Example:
            wing.add_section(
                LE_point=np.array([0, 0, 0]),
                TE_point=np.array([1, 0, 0]),
                aero_input=["lei_airfoil_breukels", [0.1, 0.05]],
            )
        """
        self.sections.append(Section(LE_point, TE_point, aero_input))

    def find_farthest_point_and_sort(self, sections):
        """Sorts the sections based on their proximity to each other
        Args:
            sections (list): List of Section objects to be sorted
        Returns:
            sorted_sections (list): Sorted list of Section objects
        Example:
            sorted_sections = wing.find_farthest_point_and_sort(sections)
        """

        # Helper function to calculate radial distance
        def radial_distance(point1, point2):
            return np.linalg.norm(np.array(point1) - np.array(point2))

        # Find the point with positive y that is furthest from all others
        farthest_point = None
        max_distance = -1
        for section in sections:
            if section.LE_point[1] > 0:  # Ensure the y-coordinate is positive
                total_distance = sum(
                    radial_distance(section.LE_point, other.LE_point)
                    for other in sections
                )
                if total_distance > max_distance:
                    max_distance = total_distance
                    farthest_point = section

        if not farthest_point:
            raise ValueError("No section has a positive y-coordinate.")

        # Remove the farthest point from the list and use it as the starting point
        sorted_sections = [farthest_point]
        remaining_sections = [
            section
            for section in sections
            if not np.allclose(section.LE_point, farthest_point.LE_point)
        ]

        # Iteratively sort the remaining sections based on proximity
        while remaining_sections:
            last_point = sorted_sections[-1].LE_point
            # Find the closest point to the last sorted point
            closest_index = min(
                range(len(remaining_sections)),
                key=lambda i: radial_distance(
                    last_point, remaining_sections[i].LE_point
                ),
            )
            # Add the closest section to the sorted list
            closest_section = remaining_sections.pop(closest_index)
            sorted_sections.append(closest_section)

        return sorted_sections

    def refine_aerodynamic_mesh(self):
        """Refine the aerodynamic mesh of the wing based on the specified spanwise panel distribution
        Args:
            None
        Returns:
            new_sections (list): List of Section objects with refined aerodynamic mesh
        Example:
            new_sections = wing.refine_aerodynamic_mesh()
        """
        if self.spanwise_panel_distribution not in [
            "uniform",
            "cosine",
            # "cosine_van_Garrel",
            "split_provided",
            "unchanged",
        ]:
            raise ValueError(
                f"Unsupported spanwise panel distribution: {self.spanwise_panel_distribution}, choose: uniform, unchanged, cosine, cosine_van_Garrel"
            )
        # # Ensure that the sections are declared from left to right
        # self.sections = sorted(
        #     self.sections, key=lambda section: section.LE_point[1], reverse=True
        # )
        # Perform additional sorting
        self.sections = self.find_farthest_point_and_sort(self.sections)

        # Ensure we get 1 section more than the desired number of panels
        n_sections = self.n_panels + 1
        logging.debug(f"n_panels: {self.n_panels}")
        logging.debug(f"n_sections: {n_sections}")

        ### Defining variables extracting from the object
        # Extract LE, TE, and aero_input from the sections
        LE, TE, aero_input = (
            np.zeros((len(self.sections), 3)),
            np.zeros((len(self.sections), 3)),
            [],
        )
        for i, section in enumerate(self.sections):
            LE[i] = section.LE_point
            TE[i] = section.TE_point
            aero_input.append(section.aero_input)

        # refine the mesh
        if self.spanwise_panel_distribution in [
            "uniform",
            "cosine",
            "cosine_van_Garrel",
        ]:
            return self.refine_mesh_for_uniform_or_cosine_distribution(
                self.spanwise_panel_distribution, n_sections, LE, TE, aero_input
            )
        elif (self.spanwise_panel_distribution == "unchanged") or (
            len(self.sections) == n_sections
        ):
            return self.sections
        elif self.spanwise_panel_distribution == "split_provided":
            return self.refine_mesh_by_splitting_provided_sections()
        else:
            raise ValueError(
                f"Unsupported spanwise panel distribution: {self.spanwise_panel_distribution}, choose: uniform, unchanged, cosine, cosine_van_Garrel"
            )

    def refine_mesh_for_uniform_or_cosine_distribution(
        self, spanwise_panel_distribution, n_sections, LE, TE, aero_input
    ):
        """Refine the aerodynamic mesh of the wing based on uniform or cosine spacing
        Args:
            spanwise_panel_distribution (str): Spanwise panel distribution type
            n_sections (int): Number of sections to create
            LE (np.ndarray): Leading edge points
            TE (np.ndarray): Trailing edge points
            aero_input (list): Aerodynamic input for each section
        Returns:
            new_sections (list): List of Section objects with refined aerodynamic mesh
        Example:
            new_sections = wing.refine_mesh_for_uniform_or_cosine_distribution(
                "uniform", 5, LE, TE, aero_input
            )
        """
        if n_sections == 2:
            return [
                Section(LE[0], TE[0], aero_input[0]),
                Section(LE[-1], TE[-1], aero_input[-1]),
            ]

        # 1. Compute the 1/4 chord line
        quarter_chord = LE + 0.25 * (TE - LE)

        # Calculate the length of each segment for the quarter chord line
        qc_lengths = np.linalg.norm(quarter_chord[1:] - quarter_chord[:-1], axis=1)
        qc_total_length = np.sum(qc_lengths)

        # Make cumulative array from 0 to the total length
        qc_cum_length = np.concatenate(([0], np.cumsum(qc_lengths)))

        # 2. Define target lengths based on desired spacing
        if spanwise_panel_distribution == "uniform":
            target_lengths = np.linspace(0, qc_total_length, n_sections)
        elif spanwise_panel_distribution in ["cosine", "cosine_van_Garrel"]:
            theta = np.linspace(0, np.pi, n_sections)
            target_lengths = qc_total_length * (1 - np.cos(theta)) / 2
        else:
            raise ValueError("Unsupported spanwise panel distribution")

        new_quarter_chord = np.zeros((n_sections, 3))
        new_LE = np.zeros((n_sections, 3))
        new_TE = np.zeros((n_sections, 3))
        new_aero_input = np.empty((n_sections,), dtype=object)
        new_sections = []

        # 3. Calculate new quarter chord points and interpolate aero inputs
        for i in range(n_sections):
            target_length = target_lengths[i]

            # Find which segment the target length falls into
            section_index = np.searchsorted(qc_cum_length, target_length) - 1
            section_index = min(max(section_index, 0), len(qc_cum_length) - 2)

            # 4. Determine weights
            segment_start_length = qc_cum_length[section_index]
            segment_end_length = qc_cum_length[section_index + 1]
            t = (target_length - segment_start_length) / (
                segment_end_length - segment_start_length
            )
            left_weight = 1 - t
            right_weight = t

            # 3. Calculate new quarter chord point
            new_quarter_chord[i] = quarter_chord[section_index] + t * (
                quarter_chord[section_index + 1] - quarter_chord[section_index]
            )

            # 5. Compute average chord vector (corrected method)
            left_chord = TE[section_index] - LE[section_index]
            right_chord = TE[section_index + 1] - LE[section_index + 1]

            # Normalize the chord vectors
            left_chord_norm = left_chord / max(jit_norm(left_chord), 1e-12)
            right_chord_norm = right_chord / max(jit_norm(right_chord), 1e-12)

            # Interpolate the direction
            avg_direction = (
                left_weight * left_chord_norm + right_weight * right_chord_norm
            )
            avg_direction = avg_direction / max(jit_norm(avg_direction), 1e-12)

            # Interpolate the length
            left_length = jit_norm(left_chord)
            right_length = jit_norm(right_chord)
            avg_length = left_weight * left_length + right_weight * right_length

            # Compute the final average chord vector
            avg_chord = avg_direction * avg_length

            # 6. Calculate new LE and TE points
            new_LE[i] = new_quarter_chord[i] - 0.25 * avg_chord
            new_TE[i] = new_quarter_chord[i] + 0.75 * avg_chord

            # Interpolate aero_input
            new_aero_input[i] = self.calculate_new_aero_input(
                aero_input, section_index, left_weight, right_weight
            )

            new_sections.append(Section(new_LE[i], new_TE[i], new_aero_input[i]))

            if self.spanwise_panel_distribution == "cosine_van_Garrel":
                raise NotImplementedError(
                    "Cosine van Garrel distribution is not yet implemented"
                )
                # new_sections = self.calculate_cosine_van_Garrel(new_sections)

        return new_sections

    def interpolate_to_common_alpha(
        self, alpha_common, alpha_orig, CL_orig, CD_orig, CM_orig
    ):
        CL_common = np.interp(alpha_common, alpha_orig, CL_orig)
        CD_common = np.interp(alpha_common, alpha_orig, CD_orig)
        CM_common = np.interp(alpha_common, alpha_orig, CM_orig)
        return CL_common, CD_common, CM_common

    def calculate_new_aero_input(
        self, aero_input, section_index, left_weight, right_weight
    ):
        """Interpolates the aero_input of two sections
        Args:
            aero_input (list): List of aerodynamic input data
            section_index (int): Index of the section to interpolate
            left_weight (float): Weight for the left section
            right_weight (float): Weight for the right section
        Returns:
            new_aero_input (list): Interpolated aerodynamic input data
        Example:
            new_aero_input = wing.calculate_new_aero_input(
                aero_input, section_index, left_weight, right_weight
            )
        """
        if aero_input[section_index][0] != aero_input[section_index + 1][0]:
            raise NotImplementedError(
                "Different aero models over the span are not supported"
            )
        if aero_input[section_index][0] == "inviscid":
            return ["inviscid"]
        elif aero_input[section_index][0] == "polar_data":
            polar_left = aero_input[section_index][1]
            polar_right = aero_input[section_index + 1][1]

            # Unpack polar data for (N,4) arrays:
            # Each row is [alpha, cl, cd, cm]
            alpha_left = polar_left[:, 0]
            CL_left = polar_left[:, 1]
            CD_left = polar_left[:, 2]
            CM_left = polar_left[:, 3]

            alpha_right = polar_right[:, 0]
            CL_right = polar_right[:, 1]
            CD_right = polar_right[:, 2]
            CM_right = polar_right[:, 3]

            # Create a common alpha array spanning the union of both alpha arrays
            alpha_common = np.union1d(alpha_left, alpha_right)

            # Interpolate both sets to the common alpha array.
            # Assume interpolate_to_common_alpha returns arrays for Cl, Cd, and Cm.
            CL_left_common, CD_left_common, CM_left_common = (
                self.interpolate_to_common_alpha(
                    alpha_common, alpha_left, CL_left, CD_left, CM_left
                )
            )
            CL_right_common, CD_right_common, CM_right_common = (
                self.interpolate_to_common_alpha(
                    alpha_common, alpha_right, CL_right, CD_right, CM_right
                )
            )

            # Interpolate using the given weights.
            CL_interp = CL_left_common * left_weight + CL_right_common * right_weight
            CD_interp = CD_left_common * left_weight + CD_right_common * right_weight
            CM_interp = CM_left_common * left_weight + CM_right_common * right_weight

            # Return the interpolated polar data in an (N,4) array.
            new_polar = np.column_stack((alpha_common, CL_interp, CD_interp, CM_interp))
            return ["polar_data", new_polar]
        elif aero_input[section_index][0] == "lei_airfoil_breukels":
            tube_diameter_left = aero_input[section_index][1][0]
            tube_diameter_right = aero_input[section_index + 1][1][0]
            tube_diameter_i = np.array(
                tube_diameter_left * left_weight + tube_diameter_right * right_weight
            )

            chamber_height_left = aero_input[section_index][1][1]
            chamber_height_right = aero_input[section_index + 1][1][1]
            chamber_height_i = np.array(
                chamber_height_left * left_weight + chamber_height_right * right_weight
            )
            logging.debug(f"left_weight: {left_weight}")
            logging.debug(f"right_weight: {right_weight}")
            logging.debug(f"tube_diameter_i: {tube_diameter_i}")
            logging.debug(f"chamber_height_i: {chamber_height_i}")
            return [
                "lei_airfoil_breukels",
                np.array([tube_diameter_i, chamber_height_i]),
            ]

        else:
            raise NotImplementedError(
                f"Unsupported aero model: {aero_input[section_index][0]}"
            )

    def refine_mesh_by_splitting_provided_sections(self):
        """Refine the aerodynamic mesh of the wing by splitting the provided sections
        Args:
            None
        Returns:
            new_sections (list): List of Section objects with refined aerodynamic mesh
        Example:
            new_sections = wing.refine_mesh_by_splitting_provided_sections()
        """

        n_sections_provided = len(self.sections)
        n_panels_provided = n_sections_provided - 1
        n_panels_desired = self.n_panels
        logging.debug(f"n_panels_provided: {n_panels_provided}")
        logging.debug(f"n_panels_desired: {n_panels_desired}")
        logging.debug(f"n_sections_provided: {n_sections_provided}")
        logging.debug(
            f"n_panels_provided % n_panels_desired: {n_panels_desired % n_panels_provided}"
        )
        if n_panels_provided == n_panels_desired:
            return self.sections
        if n_panels_desired % n_panels_provided != 0:
            raise ValueError(
                f"Desired n_panels: {n_panels_desired} is not a multiple of the {n_panels_provided} n_panels provided, choose: {n_panels_provided*2}, {n_panels_provided*3}, {n_panels_provided*4},{n_panels_provided*5},..."
            )

        n_new_sections = self.n_panels + 1 - n_sections_provided
        n_section_pairs = n_sections_provided - 1
        new_sections_per_pair, remaining = divmod(n_new_sections, n_section_pairs)

        # Lists to track the final sections
        new_sections = []

        # Extract provided LE, TE, and aero_inputs for interpolation
        LE = [np.array(section.LE_point) for section in self.sections]
        TE = [np.array(section.TE_point) for section in self.sections]
        aero_input = [section.aero_input for section in self.sections]

        for left_section_index in range(n_section_pairs):
            # Add the provided section at the start of this pair
            new_sections.append(self.sections[left_section_index])

            # Calculate the number of new sections for this pair
            num_new_sections_this_pair = new_sections_per_pair + (
                1 if left_section_index < remaining else 0
            )

            # Prepare inputs for the interpolation function
            LE_pair_list = np.array(
                [LE[left_section_index], LE[left_section_index + 1]]
            )
            TE_pair_list = np.array(
                [TE[left_section_index], TE[left_section_index + 1]]
            )
            aero_input_pair_list = [
                aero_input[left_section_index],
                aero_input[left_section_index + 1],
            ]

            # Generate the new sections for this pair
            if num_new_sections_this_pair > 0:
                new_splitted_sections = self.refine_mesh_for_uniform_or_cosine_distribution(
                    "uniform",
                    num_new_sections_this_pair
                    + 2,  # +2 because refine_mesh expects total sections, including endpoints
                    LE_pair_list,
                    TE_pair_list,
                    aero_input_pair_list,
                )
                # Append only the new sections (excluding the endpoints, which are already in `new_sections`)
                new_sections.extend(new_splitted_sections[1:-1])

        # Finally, add the last provided section
        new_sections.append(self.sections[-1])

        # Debug logging
        logging.debug(f"n_sections_provided: {n_sections_provided}")
        logging.debug(f"n_new_sections: {n_new_sections}")
        logging.debug(f"n_section_pairs: {n_section_pairs}")
        logging.debug(f"new_sections_per_pair: {new_sections_per_pair}")
        logging.debug(f"new_sections length: {len(new_sections)}")

        if len(new_sections) != self.n_panels + 1:
            logging.info(
                f" Number of new panels {len(new_sections) - 1} is NOT equal to the {self.n_panels} desired number of panels."
                f" This could be due to rounding or even splitting issues."
            )

        return new_sections

    # TODO: add test here, assessing for example the types of the inputs
    @property
    def span(self):
        """Calculates the span of the wing along the specified spanwise direction.
        The span is defined as the distance between the leading edge points of the first and last sections
        projected onto the spanwise direction.
        Args:
            None
        Returns:
            - span (float): The span of the wing along the specified spanwise direction.
        Example:
            span = wing.span
        """
        # Normalize the vector_axis to ensure it's a unit vector
        vector_axis = self.spanwise_direction / np.linalg.norm(self.spanwise_direction)

        # Concatenate the leading and trailing edge points for all sections
        all_points = np.concatenate(
            [[section.LE_point, section.TE_point] for section in self.sections]
        )

        # Project all points onto the vector axis
        projections = np.dot(all_points, vector_axis)

        # Calculate the span of the wing along the given vector axis
        span = np.max(projections) - np.min(projections)
        return span

    def calculate_projected_area(self, z_plane_vector=np.array([0, 0, 1])):
        """Calculates the projected area of the wing onto a specified plane.
        The plane is defined by a normal vector (z_plane_vector).
        Args:
            z_plane_vector (np.ndarray): Normal vector defining the plane onto which the area is projected.
                Default is the z-axis [0, 0, 1].
        Returns:
            - projected_area (float): The projected area of the wing onto the specified plane.
        Example:
            projected_area = wing.calculate_projected_area()
        """
        # Normalize the z_plane_vector
        z_plane_vector = z_plane_vector / jit_norm(z_plane_vector)

        # Helper function to project a point onto the plane
        def project_onto_plane(point, normal):
            return point - np.dot(point, normal) * normal

        projected_area = 0.0
        for i in range(len(self.sections) - 1):
            # Get the points for the current and next section
            LE_current = self.sections[i].LE_point
            TE_current = self.sections[i].TE_point
            LE_next = self.sections[i + 1].LE_point
            TE_next = self.sections[i + 1].TE_point

            # Project the points onto the plane
            LE_current_proj = project_onto_plane(LE_current, z_plane_vector)
            TE_current_proj = project_onto_plane(TE_current, z_plane_vector)
            LE_next_proj = project_onto_plane(LE_next, z_plane_vector)
            TE_next_proj = project_onto_plane(TE_next, z_plane_vector)

            # Calculate the lengths of the projected edges
            chord_current_proj = jit_norm(TE_current_proj - LE_current_proj)
            chord_next_proj = jit_norm(TE_next_proj - LE_next_proj)

            # Calculate the spanwise distance between the projected sections
            spanwise_distance_proj = jit_norm(LE_next_proj - LE_current_proj)

            # Calculate the projected area of the trapezoid formed by these points
            area = 0.5 * (chord_current_proj + chord_next_proj) * spanwise_distance_proj
            projected_area += area

        return projected_area


@dataclass
class Section:
    """Section class representing the geometry of a wing section.

    Parameters:
        LE_point (np.ndarray): The leading edge coordinate.
        TE_point (np.ndarray): The trailing edge coordinate.
        aero_input (list): The aerodynamic input data, which can be one of:
            - ["inviscid"]: for inviscid aerodynamics.
            - ["polar_data", [alpha, CL, CD, CM]]:
                * alpha: Array of angles of attack (in radians).
                * CL: Array of lift coefficients.
                * CD: Array of drag coefficients.
                * CM: Array of moment coefficients.
            - ["lei_airfoil_breukels", [d_tube, camber]]:
                * d_tube: Non-dimensional tube diameter.
                * camber: Non-dimensional camber height.

    Returns:
        Section: An instance representing the wing section.
    """

    LE_point: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    TE_point: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    aero_input: list = field(default_factory=list)

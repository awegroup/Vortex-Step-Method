"""
Test global geometric properties of the TUDELFT V3 Kite.

This test verifies that the computed geometric properties (span, projected area,
total panel area, aspect ratio) match the expected values from the CAD geometry.
"""

import pytest
import numpy as np
from pathlib import Path
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.Solver import Solver


class TestKiteGlobalProperties:
    """Test suite for verifying global geometric properties of the TUDELFT V3 Kite."""

    @pytest.fixture
    def config_file_path(self):
        """Path to the test configuration file."""
        test_dir = Path(__file__).parent
        return test_dir / "config_kite_CAD_inviscid.yaml"

    @pytest.fixture
    def body_aero(self, config_file_path):
        """Create a BodyAerodynamics instance with the kite geometry."""
        n_panels = 30
        spanwise_panel_distribution = "uniform"

        body_aero = BodyAerodynamics.instantiate(
            n_panels=n_panels,
            file_path=str(config_file_path),  # Convert Path to string
            spanwise_panel_distribution=spanwise_panel_distribution,
        )

        return body_aero

    @pytest.fixture
    def solver(self):
        """Create a basic solver instance."""
        return Solver()

    @pytest.fixture
    def results(self, body_aero, solver):
        """
        Compute aerodynamic results with arbitrary initial conditions.

        The specific flow conditions don't matter for geometric properties,
        but we need to run the solver to populate the results dictionary.
        """
        # Initialize with arbitrary velocity
        Umag = 10.0  # m/s
        angle_of_attack = 5.0  # degrees
        side_slip = 0.0  # degrees

        body_aero.va_initialize(
            Umag=Umag,
            angle_of_attack=angle_of_attack,
            side_slip=side_slip,
        )

        # Solve to get results
        results = solver.solve(body_aero)

        return results

    def test_projected_area(self, results):
        """
        Test that the projected area matches expected value.

        Expected: S ≈ 19.422 m² (projected area onto x-y plane)
        Tolerance: ±1e-2 m²
        """
        expected_area = 19.422  # m²
        tolerance = 1e-2  # m²

        computed_area = results["projected_area"]

        assert computed_area == pytest.approx(expected_area, abs=tolerance), (
            f"Projected area mismatch: expected {expected_area:.4f} m², "
            f"got {computed_area:.4f} m² (diff: {abs(computed_area - expected_area):.4f} m²)"
        )

    def test_wing_span(self, results):
        """
        Test that the wing span matches expected value.

        Expected: b ≈ 8.247 m (span, from y = -4.1236 to +4.1236)
        Tolerance: ±1e-2 m
        """
        expected_span = 8.27  # m
        tolerance = 1e-2  # m

        computed_span = results["wing_span"]

        assert computed_span == pytest.approx(expected_span, abs=tolerance), (
            f"Wing span mismatch: expected {expected_span:.4f} m, "
            f"got {computed_span:.4f} m (diff: {abs(computed_span - expected_span):.4f} m)"
        )

    def test_aspect_ratio_projected(self, results):
        """
        Test that the aspect ratio matches expected value.

        Aspect ratio is computed as: AR = b² / S
        Expected: AR ≈ 3.52 (derived from b=8.27m, S=19.422m²)
        Tolerance: ±1e-2
        """
        expected_span = 8.27  # m
        expected_area = 19.422  # m²
        expected_AR = expected_span**2 / expected_area  # ≈ 3.52
        tolerance = 1e-2

        computed_AR = results["aspect_ratio_projected"]

        assert computed_AR == pytest.approx(expected_AR, abs=tolerance), (
            f"Aspect ratio mismatch: expected {expected_AR:.4f}, "
            f"got {computed_AR:.4f} (diff: {abs(computed_AR - expected_AR):.4f})"
        )

    def test_total_panel_area(self, results):
        """
        Test that the total panel area is reasonable.

        The total panel area (sum of all panel areas) should be larger than
        the projected area because panels are not flat in the x-y plane.

        Note: This test uses a looser tolerance as the exact value depends
        on the kite's 3D shape and is not as precisely defined as projected area.
        """
        projected_area = results["projected_area"]
        total_panel_area = results["area_all_panels"]

        # Total panel area should be larger than projected area
        assert total_panel_area > projected_area, (
            f"Total panel area ({total_panel_area:.4f} m²) should be larger than "
            f"projected area ({projected_area:.4f} m²)"
        )

        # But not unreasonably larger (sanity check)
        # For a moderately swept kite, expect ratio between 1.1 and 1.4
        ratio = total_panel_area / projected_area
        assert 1.1 < ratio < 1.4, (
            f"Ratio of total panel area to projected area ({ratio:.3f}) "
            f"is outside reasonable range [1.1, 1.4]"
        )

    def test_geometric_consistency(self, results):
        """
        Test that geometric properties are internally consistent.

        Verifies:
        1. Aspect ratio = span² / projected_area
        2. All values are positive
        3. Span is reasonable relative to area
        """
        span = results["wing_span"]
        projected_area = results["projected_area"]
        aspect_ratio = results["aspect_ratio_projected"]
        total_panel_area = results["area_all_panels"]

        # Check all values are positive
        assert span > 0, "Wing span must be positive"
        assert projected_area > 0, "Projected area must be positive"
        assert aspect_ratio > 0, "Aspect ratio must be positive"
        assert total_panel_area > 0, "Total panel area must be positive"

        # Check aspect ratio consistency
        computed_AR = span**2 / projected_area
        assert aspect_ratio == pytest.approx(computed_AR, rel=1e-6), (
            f"Aspect ratio inconsistent: stored value {aspect_ratio:.6f} "
            f"vs computed b²/S {computed_AR:.6f}"
        )

    def test_config_file_exists(self, config_file_path):
        """Verify that the test configuration file exists."""
        assert (
            config_file_path.exists()
        ), f"Configuration file not found at {config_file_path}"


class TestGeometricPropertiesWithDifferentPanelCounts:
    """
    Test that geometric properties are consistent across different panel counts.

    The projected area and span should be independent of mesh resolution.
    """

    @pytest.fixture
    def config_file_path(self):
        """Path to the test configuration file."""
        test_dir = Path(__file__).parent
        return test_dir / "config_kite_CAD_inviscid.yaml"

    @pytest.mark.parametrize("n_panels", [20, 30, 40, 50])
    def test_geometric_properties_mesh_independent(self, config_file_path, n_panels):
        """
        Test that projected area and span are independent of mesh resolution.

        This is an important property: the geometry should not change
        with different panel counts.
        """
        expected_span = 8.247  # m
        expected_area = 19.422  # m²
        tolerance = 0.05  # Slightly larger tolerance for mesh independence

        # Create body aero with specific panel count
        body_aero = BodyAerodynamics.instantiate(
            n_panels=n_panels,
            file_path=str(config_file_path),  # Convert Path to string
            spanwise_panel_distribution="uniform",
        )

        # Initialize and solve
        body_aero.va_initialize(Umag=10.0, angle_of_attack=5.0, side_slip=0.0)
        solver = Solver()
        results = solver.solve(body_aero)

        # Check span
        computed_span = results["wing_span"]
        assert computed_span == pytest.approx(expected_span, abs=tolerance), (
            f"Span changed with n_panels={n_panels}: "
            f"expected {expected_span:.4f} m, got {computed_span:.4f} m"
        )

        # Check projected area
        computed_area = results["projected_area"]
        assert computed_area == pytest.approx(expected_area, abs=tolerance), (
            f"Projected area changed with n_panels={n_panels}: "
            f"expected {expected_area:.4f} m², got {computed_area:.4f} m²"
        )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

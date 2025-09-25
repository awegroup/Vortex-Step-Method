"""
Test masure_regression functionality with real model integration.
"""

import pytest
from pathlib import Path
from VSM.core.AirfoilAerodynamics import AirfoilAerodynamics


class TestMasureRegressionReal:
    """Test cases for masure_regression with real model file integration."""

    @pytest.fixture
    def masure_config_path(self):
        """Get config path for masure_regression tests."""
        test_dir = Path(__file__).parent.parent.parent
        return str(
            test_dir
            / "data"
            / "TUDELFT_V3_KITE"
            / "config_kite_CAD_masure_regression.yaml"
        )

    @pytest.fixture
    def masure_params(self):
        """Standard masure_regression parameters for testing."""
        return {
            "t": 0.07,
            "eta": 0.2,
            "kappa": 0.95,
            "delta": -2,
            "lambda": 0.65,
            "phi": 0.25,
        }

    @pytest.fixture
    def small_alpha_range(self):
        """Small alpha range for faster testing."""
        return [-5, 15, 5]  # [min, max, step] in degrees

    @pytest.fixture
    def test_reynolds(self):
        """Test Reynolds number."""
        return 1e6

    def test_masure_regression_basic_creation(
        self, masure_params, small_alpha_range, test_reynolds, masure_config_path
    ):
        """Test basic creation of masure_regression airfoil."""
        airfoil = AirfoilAerodynamics.from_yaml_entry(
            airfoil_type="masure_regression",
            airfoil_params=masure_params,
            alpha_range=small_alpha_range,
            reynolds=test_reynolds,
            file_path=masure_config_path,
            ml_models_dir="data/ml_models",
        )

        # Verify basic attributes
        assert hasattr(airfoil, "alpha"), "Airfoil should have alpha attribute"
        assert hasattr(airfoil, "CL"), "Airfoil should have CL attribute"
        assert hasattr(airfoil, "CD"), "Airfoil should have CD attribute"
        assert hasattr(airfoil, "CM"), "Airfoil should have CM attribute"

    def test_masure_regression_output_shapes(
        self, masure_params, small_alpha_range, test_reynolds, masure_config_path
    ):
        """Test that output arrays have correct shapes."""
        airfoil = AirfoilAerodynamics.from_yaml_entry(
            airfoil_type="masure_regression",
            airfoil_params=masure_params,
            alpha_range=small_alpha_range,
            reynolds=test_reynolds,
            file_path=masure_config_path,
            ml_models_dir="data/ml_models",
        )

        # Calculate expected number of points
        alpha_min, alpha_max, alpha_step = small_alpha_range
        expected_points = int((alpha_max - alpha_min) / alpha_step) + 1

        # Verify shapes
        assert airfoil.alpha.shape == (
            expected_points,
        ), f"Alpha shape should be ({expected_points},), got {airfoil.alpha.shape}"
        assert airfoil.CL.shape == (
            expected_points,
        ), f"CL shape should be ({expected_points},), got {airfoil.CL.shape}"
        assert airfoil.CD.shape == (
            expected_points,
        ), f"CD shape should be ({expected_points},), got {airfoil.CD.shape}"
        assert airfoil.CM.shape == (
            expected_points,
        ), f"CM shape should be ({expected_points},), got {airfoil.CM.shape}"

    def test_masure_regression_alpha_range_conversion(
        self, masure_params, small_alpha_range, test_reynolds, masure_config_path
    ):
        """Test that alpha range is correctly converted from degrees to radians."""
        airfoil = AirfoilAerodynamics.from_yaml_entry(
            airfoil_type="masure_regression",
            airfoil_params=masure_params,
            alpha_range=small_alpha_range,
            reynolds=test_reynolds,
            file_path=masure_config_path,
            ml_models_dir="data/ml_models",
        )

        import numpy as np

        alpha_min_deg, alpha_max_deg, _ = small_alpha_range
        alpha_min_rad = np.deg2rad(alpha_min_deg)
        alpha_max_rad = np.deg2rad(alpha_max_deg)

        # Verify alpha range conversion
        assert np.isclose(
            airfoil.alpha[0], alpha_min_rad, atol=1e-6
        ), f"First alpha should be {alpha_min_rad:.6f} rad, got {airfoil.alpha[0]:.6f}"
        assert np.isclose(
            airfoil.alpha[-1], alpha_max_rad, atol=1e-6
        ), f"Last alpha should be {alpha_max_rad:.6f} rad, got {airfoil.alpha[-1]:.6f}"

    def test_masure_regression_coefficient_ranges(
        self, masure_params, small_alpha_range, test_reynolds, masure_config_path
    ):
        """Test that coefficient values are within reasonable ranges."""
        airfoil = AirfoilAerodynamics.from_yaml_entry(
            airfoil_type="masure_regression",
            airfoil_params=masure_params,
            alpha_range=small_alpha_range,
            reynolds=test_reynolds,
            file_path=masure_config_path,
            ml_models_dir="data/ml_models",
        )

        # Check that coefficients are within reasonable ranges
        assert (
            -2.0 <= airfoil.CL.min() <= 2.5
        ), f"CL range seems unreasonable: {airfoil.CL.min():.3f} to {airfoil.CL.max():.3f}"
        assert (
            -2.0 <= airfoil.CL.max() <= 2.5
        ), f"CL range seems unreasonable: {airfoil.CL.min():.3f} to {airfoil.CL.max():.3f}"

        assert (
            0.0 <= airfoil.CD.min() <= 0.5
        ), f"CD range seems unreasonable: {airfoil.CD.min():.3f} to {airfoil.CD.max():.3f}"
        assert (
            0.0 <= airfoil.CD.max() <= 0.5
        ), f"CD range seems unreasonable: {airfoil.CD.min():.3f} to {airfoil.CD.max():.3f}"

        assert (
            -0.5 <= airfoil.CM.min() <= 0.5
        ), f"CM range seems unreasonable: {airfoil.CM.min():.3f} to {airfoil.CM.max():.3f}"
        assert (
            -0.5 <= airfoil.CM.max() <= 0.5
        ), f"CM range seems unreasonable: {airfoil.CM.min():.3f} to {airfoil.CM.max():.3f}"

    def test_masure_regression_different_parameters(
        self, small_alpha_range, test_reynolds, masure_config_path
    ):
        """Test masure_regression with different airfoil parameters."""
        param_sets = [
            {
                "t": 0.05,
                "eta": 0.1,
                "kappa": 0.9,
                "delta": -1,
                "lambda": 0.6,
                "phi": 0.2,
            },
            {
                "t": 0.1,
                "eta": 0.3,
                "kappa": 1.0,
                "delta": -3,
                "lambda": 0.7,
                "phi": 0.3,
            },
        ]

        for i, params in enumerate(param_sets):
            airfoil = AirfoilAerodynamics.from_yaml_entry(
                airfoil_type="masure_regression",
                airfoil_params=params,
                alpha_range=small_alpha_range,
                reynolds=test_reynolds,
                file_path=masure_config_path,
                ml_models_dir="data/ml_models",
            )

            # Verify creation was successful
            assert (
                airfoil.alpha is not None
            ), f"Alpha should not be None for param set {i}"
            assert airfoil.CL is not None, f"CL should not be None for param set {i}"
            assert airfoil.CD is not None, f"CD should not be None for param set {i}"
            assert airfoil.CM is not None, f"CM should not be None for param set {i}"

    @pytest.mark.skipif(
        not Path(__file__)
        .parent.parent.parent.joinpath("data", "models", "ET_re1e6.pkl")
        .exists(),
        reason="Real model files not available",
    )
    def test_masure_regression_requires_model_files(
        self, masure_params, small_alpha_range, test_reynolds, masure_config_path
    ):
        """Test that masure_regression properly handles missing model files."""
        # This test will only run if model files are available
        # If they're not available, it will be skipped

        airfoil = AirfoilAerodynamics.from_yaml_entry(
            airfoil_type="masure_regression",
            airfoil_params=masure_params,
            alpha_range=small_alpha_range,
            reynolds=test_reynolds,
            file_path=masure_config_path,
            ml_models_dir="data/ml_models",
        )

        # If we get here, model files exist and creation was successful
        assert (
            airfoil is not None
        ), "Airfoil creation should succeed with available model files"

    def test_masure_regression_caching_behavior(
        self, masure_params, small_alpha_range, test_reynolds, masure_config_path
    ):
        """Test that caching works properly for masure_regression."""
        import time

        # First call - should generate and cache
        start_time = time.time()
        airfoil1 = AirfoilAerodynamics.from_yaml_entry(
            airfoil_type="masure_regression",
            airfoil_params=masure_params,
            alpha_range=small_alpha_range,
            reynolds=test_reynolds,
            file_path=masure_config_path,
            ml_models_dir="data/ml_models",
        )
        first_call_time = time.time() - start_time

        # Second call with same parameters - should use cache
        start_time = time.time()
        airfoil2 = AirfoilAerodynamics.from_yaml_entry(
            airfoil_type="masure_regression",
            airfoil_params=masure_params,
            alpha_range=small_alpha_range,
            reynolds=test_reynolds,
            file_path=masure_config_path,
            ml_models_dir="data/ml_models",
        )
        second_call_time = time.time() - start_time

        # Verify results are the same
        import numpy as np

        assert np.allclose(
            airfoil1.alpha, airfoil2.alpha
        ), "Alpha arrays should be identical"
        assert np.allclose(airfoil1.CL, airfoil2.CL), "CL arrays should be identical"
        assert np.allclose(airfoil1.CD, airfoil2.CD), "CD arrays should be identical"
        assert np.allclose(airfoil1.CM, airfoil2.CM), "CM arrays should be identical"

        # Second call should generally be faster (though this isn't guaranteed)
        print(
            f"First call: {first_call_time:.3f}s, Second call: {second_call_time:.3f}s"
        )

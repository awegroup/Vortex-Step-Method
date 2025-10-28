import numpy as np
import pytest
import time
from pathlib import Path
from VSM.core.AirfoilAerodynamics import AirfoilAerodynamics


@pytest.fixture
def breukels_params():
    return {"t": 0.1, "kappa": 0.05}


@pytest.fixture
def alpha_range():
    return [-10, 30, 1]


@pytest.fixture
def small_alpha_range():
    """Smaller alpha range for faster testing."""
    return [-10, 25, 2]


@pytest.fixture
def reynolds():
    return 1e6


@pytest.fixture
def masure_test_config():
    """Configuration path for masure_regression tests."""
    test_dir = Path(__file__).parent.parent.parent
    return str(
        test_dir / "data" / "TUDELFT_V3_KITE" / "aero_geometry_CAD_masure_regression.yaml"
    )


@pytest.fixture
def masure_params_list():
    """Sample masure_regression parameters for testing."""
    return [
        {
            "t": 0.07,
            "eta": 0.2,
            "kappa": 0.95,
            "delta": -2,
            "lambda": 0.65,
            "phi": 0.25,
        },
        {
            "t": 0.08,
            "eta": 0.3,
            "kappa": 0.90,
            "delta": -1,
            "lambda": 0.70,
            "phi": 0.30,
        },
        {
            "t": 0.06,
            "eta": 0.15,
            "kappa": 1.00,
            "delta": -3,
            "lambda": 0.60,
            "phi": 0.20,
        },
    ]


@pytest.fixture
def tmp_csv(tmp_path):
    # Create a temporary CSV file with polar data
    alpha = np.linspace(-10, 30, 41)
    cl = 2 * np.pi * np.deg2rad(alpha)
    cd = np.zeros_like(alpha)
    cm = np.zeros_like(alpha)
    arr = np.column_stack([alpha, cl, cd, cm])
    csv_path = tmp_path / "test_polar.csv"
    np.savetxt(csv_path, arr, delimiter=",", header="alpha,cl,cd,cm", comments="")
    return csv_path


def test_breukels_regression(breukels_params, alpha_range):
    """Test basic breukels_regression functionality."""
    aero = AirfoilAerodynamics.from_yaml_entry(
        "breukels_regression", breukels_params, alpha_range=alpha_range
    )
    assert aero.CL is not None
    assert aero.CD is not None
    assert aero.CM is not None
    assert len(aero.alpha) == len(aero.CL)
    assert np.all(np.isfinite(aero.CL))
    assert np.all(np.isfinite(aero.CD))
    assert np.all(np.isfinite(aero.CM))


def test_inviscid(alpha_range):
    """Test inviscid airfoil theory implementation."""
    aero = AirfoilAerodynamics.from_yaml_entry("inviscid", {}, alpha_range=alpha_range)
    assert np.allclose(aero.CL, 2 * np.pi * aero.alpha)
    assert np.allclose(aero.CD, 0)
    assert np.allclose(aero.CM, 0)


def test_polars(tmp_csv, alpha_range):
    """Test loading polars from CSV file."""
    # Use the polars type with a CSV file
    params = {"csv_file_path": tmp_csv.name}
    aero = AirfoilAerodynamics.from_yaml_entry(
        "polars", params, alpha_range=alpha_range, file_path=tmp_csv
    )
    assert aero.CL.shape == aero.alpha.shape
    assert np.all(np.isfinite(aero.CL))
    assert np.all(np.isfinite(aero.CD))
    assert np.all(np.isfinite(aero.CM))


def test_polars_no_alpha_range(tmp_csv):
    """Test loading polars without alpha_range resampling."""
    # Use the polars type with a CSV file and no alpha_range
    params = {"csv_file_path": tmp_csv.name}
    aero = AirfoilAerodynamics.from_yaml_entry(
        "polars", params, alpha_range=None, file_path=tmp_csv
    )
    assert aero.CL.shape == aero.alpha.shape
    assert np.all(np.isfinite(aero.CL))


def test_invalid_type_raises(alpha_range):
    """Test that invalid airfoil types raise ValueError."""
    with pytest.raises(ValueError):
        AirfoilAerodynamics.from_yaml_entry("not_a_type", {}, alpha_range=alpha_range)


# NeuralFoil test is optional, as it requires the neuralfoil package and .dat files.
# Here is a placeholder for completeness.
@pytest.mark.skip(reason="Requires neuralfoil package and .dat file")
def test_neuralfoil(alpha_range, reynolds):
    params = {
        "dat_file_path": "airfoil.dat",  # Should exist in the test directory
        "model_size": "xxxlarge",
        "xtr_lower": 0.01,
        "xtr_upper": 0.01,
        "n_crit": 9,
    }
    # This will fail unless a valid .dat file is present
    aero = AirfoilAerodynamics.from_yaml_entry(
        "neuralfoil",
        params,
        alpha_range=alpha_range,
        reynolds=reynolds,
        file_path=Path("."),
    )
    assert aero.CL is not None
    assert aero.CD is not None
    assert aero.CM is not None


# Test masure_regression if model files are available
@pytest.mark.skip(reason="Requires masure_regression model files in data/models/")
def test_masure_regression(masure_test_config, small_alpha_range):
    """Test masure_regression airfoil type."""
    params = {
        "t": 0.07,
        "eta": 0.2,
        "kappa": 0.95,
        "delta": -2,
        "lambda": 0.65,
        "phi": 0.25,
    }
    aero = AirfoilAerodynamics.from_yaml_entry(
        "masure_regression",
        params,
        alpha_range=small_alpha_range,
        reynolds=5e6,
        file_path=masure_test_config,
    )
    assert aero.CL is not None
    assert aero.CD is not None
    assert aero.CM is not None
    assert len(aero.alpha) == len(aero.CL)
    assert np.all(np.isfinite(aero.CL))
    assert np.all(np.isfinite(aero.CD))
    assert np.all(np.isfinite(aero.CM))


# ========================
# BATCH PROCESSING TESTS
# ========================


def test_batch_processing_input_validation():
    """Test that batch processing validates input properly."""
    with pytest.raises(ValueError, match="All input lists must have the same length"):
        AirfoilAerodynamics.from_yaml_entry_batch(
            airfoil_ids=["id1", "id2"],
            airfoil_types=["inviscid"],  # Wrong length
            airfoil_params_list=[{}, {}],
            alpha_range=[-10, 10, 1],
        )


def test_batch_processing_empty_input():
    """Test batch processing with empty input."""
    result = AirfoilAerodynamics.from_yaml_entry_batch(
        airfoil_ids=[],
        airfoil_types=[],
        airfoil_params_list=[],
        alpha_range=[-10, 10, 1],
    )
    assert result == {}


def test_batch_processing_inviscid(small_alpha_range):
    """Test batch processing with inviscid airfoils."""
    airfoil_ids = ["inv1", "inv2", "inv3"]
    airfoil_types = ["inviscid"] * 3
    airfoil_params_list = [{}, {}, {}]

    result = AirfoilAerodynamics.from_yaml_entry_batch(
        airfoil_ids=airfoil_ids,
        airfoil_types=airfoil_types,
        airfoil_params_list=airfoil_params_list,
        alpha_range=small_alpha_range,
    )

    assert len(result) == 3
    for airfoil_id in airfoil_ids:
        assert airfoil_id in result
        polar_data = result[airfoil_id]
        assert polar_data.shape[1] == 4  # [alpha, CL, CD, CM]
        # Check inviscid theory: CL = 2*pi*alpha, CD = 0, CM = 0
        alpha_col = polar_data[:, 0]
        cl_col = polar_data[:, 1]
        cd_col = polar_data[:, 2]
        cm_col = polar_data[:, 3]
        assert np.allclose(cl_col, 2 * np.pi * alpha_col)
        assert np.allclose(cd_col, 0)
        assert np.allclose(cm_col, 0)


def test_batch_processing_breukels(small_alpha_range):
    """Test batch processing with breukels_regression airfoils."""
    airfoil_ids = ["b1", "b2"]
    airfoil_types = ["breukels_regression"] * 2
    airfoil_params_list = [
        {"t": 0.1, "kappa": 0.05},
        {"t": 0.12, "kappa": 0.08},
    ]

    result = AirfoilAerodynamics.from_yaml_entry_batch(
        airfoil_ids=airfoil_ids,
        airfoil_types=airfoil_types,
        airfoil_params_list=airfoil_params_list,
        alpha_range=small_alpha_range,
    )

    assert len(result) == 2
    for airfoil_id in airfoil_ids:
        assert airfoil_id in result
        polar_data = result[airfoil_id]
        assert polar_data.shape[1] == 4
        assert np.all(np.isfinite(polar_data))


def test_batch_processing_mixed_types(small_alpha_range):
    """Test batch processing with mixed airfoil types."""
    airfoil_ids = ["inv1", "b1", "inv2"]
    airfoil_types = ["inviscid", "breukels_regression", "inviscid"]
    airfoil_params_list = [
        {},
        {"t": 0.1, "kappa": 0.05},
        {},
    ]

    result = AirfoilAerodynamics.from_yaml_entry_batch(
        airfoil_ids=airfoil_ids,
        airfoil_types=airfoil_types,
        airfoil_params_list=airfoil_params_list,
        alpha_range=small_alpha_range,
    )

    assert len(result) == 3
    for airfoil_id in airfoil_ids:
        assert airfoil_id in result
        polar_data = result[airfoil_id]
        assert polar_data.shape[1] == 4
        assert np.all(np.isfinite(polar_data))


@pytest.mark.skip(reason="Requires masure_regression model files")
def test_batch_vs_individual_masure_regression(
    masure_test_config, masure_params_list, small_alpha_range
):
    """Test that batch processing produces identical results to individual processing for masure_regression."""
    reynolds = 5e6

    # Individual processing
    individual_results = {}
    for i, params in enumerate(masure_params_list):
        airfoil_id = f"test{i+1}"
        aero = AirfoilAerodynamics.from_yaml_entry(
            "masure_regression",
            params,
            alpha_range=small_alpha_range,
            reynolds=reynolds,
            file_path=masure_test_config,
        )
        individual_results[airfoil_id] = aero.to_polar_array()

    # Batch processing
    airfoil_ids = [f"test{i+1}" for i in range(len(masure_params_list))]
    airfoil_types = ["masure_regression"] * len(masure_params_list)

    batch_results = AirfoilAerodynamics.from_yaml_entry_batch(
        airfoil_ids=airfoil_ids,
        airfoil_types=airfoil_types,
        airfoil_params_list=masure_params_list,
        alpha_range=small_alpha_range,
        reynolds=reynolds,
        file_path=masure_test_config,
    )

    # Compare results
    tolerance = 1e-10
    for airfoil_id in airfoil_ids:
        individual_data = individual_results[airfoil_id]
        batch_data = batch_results[airfoil_id]

        assert individual_data.shape == batch_data.shape
        max_diff = np.max(np.abs(individual_data - batch_data))
        assert max_diff < tolerance, f"Results differ by {max_diff} for {airfoil_id}"


# ========================
# CACHING TESTS
# ========================


@pytest.mark.skip(reason="Requires masure_regression model files")
def test_masure_regression_caching(masure_test_config, small_alpha_range):
    """Test that masure_regression model caching improves performance."""
    params = {
        "t": 0.07,
        "eta": 0.2,
        "kappa": 0.95,
        "delta": -2,
        "lambda": 0.65,
        "phi": 0.25,
    }
    reynolds = 5e6

    # Clear cache
    AirfoilAerodynamics._masure_model_cache.clear()

    # First call (should load model)
    start_time = time.time()
    aero1 = AirfoilAerodynamics.from_yaml_entry(
        "masure_regression",
        params,
        alpha_range=small_alpha_range,
        reynolds=reynolds,
        file_path=masure_test_config,
    )
    first_time = time.time() - start_time

    # Second call (should use cached model)
    start_time = time.time()
    aero2 = AirfoilAerodynamics.from_yaml_entry(
        "masure_regression",
        params,
        alpha_range=small_alpha_range,
        reynolds=reynolds,
        file_path=masure_test_config,
    )
    second_time = time.time() - start_time

    # Verify results are identical
    assert np.allclose(aero1.to_polar_array(), aero2.to_polar_array())

    # Verify caching improved performance (second call should be at least 2x faster)
    assert (
        second_time < first_time / 2
    ), f"Caching didn't improve performance: {first_time:.3f}s vs {second_time:.3f}s"

    # Verify cache contains the model
    assert reynolds in AirfoilAerodynamics._masure_model_cache


# ========================
# ADDITIONAL TESTS
# ========================


def test_breukels_regression_stall_logic(breukels_params):
    """Test that Breukels regression implements stall logic correctly."""
    # Use a range that includes stall angles (> 20 deg or < -20 deg)
    alpha_range = [-25, 25, 5]

    aero = AirfoilAerodynamics.from_yaml_entry(
        "breukels_regression", breukels_params, alpha_range=alpha_range
    )

    alpha_deg = np.arange(
        alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]
    )
    stall_mask = (alpha_deg > 20) | (alpha_deg < -20)

    if np.any(stall_mask):
        # Check that stall logic is applied
        expected_cl_stall = (
            2
            * np.cos(np.deg2rad(alpha_deg[stall_mask]))
            * np.sin(np.deg2rad(alpha_deg[stall_mask])) ** 2
        )
        expected_cd_stall = 2 * np.sin(np.deg2rad(alpha_deg[stall_mask])) ** 3

        actual_cl_stall = aero.CL[stall_mask]
        actual_cd_stall = aero.CD[stall_mask]

        assert np.allclose(
            actual_cl_stall, expected_cl_stall
        ), "Stall CL logic not applied correctly"
        assert np.allclose(
            actual_cd_stall, expected_cd_stall
        ), "Stall CD logic not applied correctly"


def test_alpha_range_validation():
    """Test that alpha_range parameter is handled correctly."""
    params = {"t": 0.1, "kappa": 0.05}

    # Test with different step sizes
    for step in [0.5, 1, 2, 5]:
        alpha_range = [-10, 10, step]
        aero = AirfoilAerodynamics.from_yaml_entry(
            "breukels_regression", params, alpha_range=alpha_range
        )
        expected_n_points = int((alpha_range[1] - alpha_range[0]) / alpha_range[2]) + 1
        assert len(aero.alpha) == expected_n_points


def test_source_attribute_assignment():
    """Test that the source attribute is correctly assigned."""
    test_cases = [
        ("inviscid", {}),
        ("breukels_regression", {"t": 0.1, "kappa": 0.05}),
    ]

    for airfoil_type, params in test_cases:
        aero = AirfoilAerodynamics.from_yaml_entry(
            airfoil_type, params, alpha_range=[-10, 10, 1]
        )
        assert aero.source == airfoil_type.lower()


def test_to_polar_array_format():
    """Test that to_polar_array returns the correct format."""
    params = {"t": 0.1, "kappa": 0.05}
    alpha_range = [-10, 10, 2]

    aero = AirfoilAerodynamics.from_yaml_entry(
        "breukels_regression", params, alpha_range=alpha_range
    )

    polar_array = aero.to_polar_array()

    # Check shape
    expected_n_points = int((alpha_range[1] - alpha_range[0]) / alpha_range[2]) + 1
    assert polar_array.shape == (expected_n_points, 4)

    # Check that columns correspond to the right values
    assert np.allclose(polar_array[:, 0], aero.alpha)  # alpha
    assert np.allclose(polar_array[:, 1], aero.CL)  # CL
    assert np.allclose(polar_array[:, 2], aero.CD)  # CD
    assert np.allclose(polar_array[:, 3], aero.CM)  # CM


def test_direct_instantiation_blocked():
    """Test that direct instantiation is properly blocked."""
    with pytest.raises(RuntimeError, match="Use AirfoilAerodynamics.from_yaml_entry"):
        AirfoilAerodynamics()


def test_polars_angle_unit_detection(tmp_path):
    """Test that polars correctly detect angle units (degrees vs radians)."""
    # Create CSV with angles in degrees
    alpha_deg = np.linspace(-10, 30, 41)
    cl = 2 * np.pi * np.deg2rad(alpha_deg)  # Theoretical values
    cd = np.zeros_like(alpha_deg)
    cm = np.zeros_like(alpha_deg)

    # Test with degrees
    arr_deg = np.column_stack([alpha_deg, cl, cd, cm])
    csv_path_deg = tmp_path / "test_polar_deg.csv"
    np.savetxt(
        csv_path_deg, arr_deg, delimiter=",", header="alpha,cl,cd,cm", comments=""
    )

    params_deg = {"csv_file_path": csv_path_deg.name}
    aero_deg = AirfoilAerodynamics.from_yaml_entry(
        "polars", params_deg, alpha_range=None, file_path=csv_path_deg
    )

    # Test with radians
    alpha_rad = np.deg2rad(alpha_deg)
    arr_rad = np.column_stack([alpha_rad, cl, cd, cm])
    csv_path_rad = tmp_path / "test_polar_rad.csv"
    np.savetxt(
        csv_path_rad, arr_rad, delimiter=",", header="alpha,cl,cd,cm", comments=""
    )

    params_rad = {"csv_file_path": csv_path_rad.name}
    aero_rad = AirfoilAerodynamics.from_yaml_entry(
        "polars", params_rad, alpha_range=None, file_path=csv_path_rad
    )

    # Both should produce similar results (alpha should be in radians internally)
    assert np.allclose(aero_deg.alpha, aero_rad.alpha, rtol=1e-10)
    assert np.allclose(aero_deg.CL, aero_rad.CL)


def test_missing_required_parameters():
    """Test that missing required parameters raise appropriate errors."""
    # Test missing file_path for polars
    with pytest.raises(ValueError, match="file_path must be provided"):
        AirfoilAerodynamics.from_yaml_entry(
            "polars", {"csv_file_path": "test.csv"}, alpha_range=[-10, 10, 1]
        )

    # Test missing file_path for neuralfoil
    with pytest.raises(ValueError, match="file_path must be provided"):
        AirfoilAerodynamics.from_yaml_entry(
            "neuralfoil",
            {"dat_file_path": "test.dat"},
            alpha_range=[-10, 10, 1],
            reynolds=1e6,
        )

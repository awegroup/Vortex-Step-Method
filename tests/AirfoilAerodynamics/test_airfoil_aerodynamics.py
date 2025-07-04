import numpy as np
import pytest
from pathlib import Path
from VSM.core.AirfoilAerodynamics import AirfoilAerodynamics


@pytest.fixture
def breukels_params():
    return {"t": 0.1, "kappa": 0.05}


@pytest.fixture
def alpha_range():
    return [-10, 30, 1]


@pytest.fixture
def reynolds():
    return 1e6


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
    aero = AirfoilAerodynamics.from_yaml_entry("inviscid", {}, alpha_range=alpha_range)
    assert np.allclose(aero.CL, 2 * np.pi * aero.alpha)
    assert np.allclose(aero.CD, 0)
    assert np.allclose(aero.CM, 0)


def test_polars(tmp_csv, alpha_range):
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
    # Use the polars type with a CSV file and no alpha_range
    params = {"csv_file_path": tmp_csv.name}
    aero = AirfoilAerodynamics.from_yaml_entry(
        "polars", params, alpha_range=None, file_path=tmp_csv
    )
    assert aero.CL.shape == aero.alpha.shape
    assert np.all(np.isfinite(aero.CL))


def test_to_polar_array_shape(breukels_params, alpha_range):
    aero = AirfoilAerodynamics.from_yaml_entry(
        "breukels_regression", breukels_params, alpha_range=alpha_range
    )
    polar = aero.to_polar_array()
    assert polar.shape[1] == 4
    assert polar.shape[0] == len(aero.alpha)


def test_invalid_type_raises(alpha_range):
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

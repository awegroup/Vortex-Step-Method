"""
Test suite to verify that all example scripts run without errors.

This test module runs all Python example files in the examples/ directory,
excluding those in the machine_learning_for_lei_airfoils subdirectory.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# Get the project root directory (parent of tests/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def find_example_files():
    """
    Find all .py files in the examples directory, excluding the
    machine_learning_for_lei_airfoils subdirectory.

    Returns:
        list: List of Path objects for each example file
    """
    example_files = []

    for py_file in EXAMPLES_DIR.rglob("*.py"):
        # Skip files in machine_learning_for_lei_airfoils directory
        if "machine_learning_for_lei_airfoils" in str(py_file):
            continue

        # Skip __pycache__ directories
        if "__pycache__" in str(py_file):
            continue

        example_files.append(py_file)

    return sorted(example_files)


# Generate test parameters: (file_path, test_id)
example_files = find_example_files()
test_params = [(f, f.relative_to(EXAMPLES_DIR)) for f in example_files]

RUNNER_CODE = textwrap.dedent(
    """
    import runpy
    import sys
    from pathlib import Path


    def suppress_matplotlib_show():
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception:
            return

        def _suppress_show(*args, **kwargs):
            return None

        plt.show = _suppress_show


    def suppress_plotly_show():
        try:
            from plotly.graph_objects import Figure
        except Exception:
            return

        def _suppress_show(self, *args, **kwargs):
            return None

        Figure.show = _suppress_show

        try:
            import plotly.io as pio
        except Exception:
            return

        def _suppress_io_show(*args, **kwargs):
            return None

        pio.show = _suppress_io_show


    suppress_matplotlib_show()
    suppress_plotly_show()

    script_path = Path(sys.argv[1]).resolve()
    runpy.run_path(str(script_path), run_name="__main__")
    """
)


@pytest.mark.parametrize(
    "example_file,relative_path", test_params, ids=[str(p[1]) for p in test_params]
)
def test_example_runs_without_error(example_file, relative_path):
    """
    Test that an example file runs without errors.

    Parameters:
        example_file: Path object pointing to the example file
        relative_path: Relative path from examples/ directory (used for test ID)
    """
    # Run the example file as a subprocess
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    result = subprocess.run(
        [sys.executable, "-c", RUNNER_CODE, str(example_file)],
        cwd=example_file.parent,  # Run from the file's directory
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout per example
        env=env,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    has_traceback = "Traceback (most recent call last)" in stdout or "Traceback (most recent call last)" in stderr
    breaking_error = result.returncode != 0 or has_traceback

    # Check if the process completed successfully
    if breaking_error:
        error_msg = (
            f"\n{'='*70}\n"
            f"Example file failed: {relative_path}\n"
            f"{'='*70}\n"
            f"STDOUT:\n{stdout}\n"
            f"{'-'*70}\n"
            f"STDERR:\n{stderr}\n"
            f"{'='*70}\n"
        )
        pytest.fail(error_msg)

    # Report non-breaking anomalies without failing the test
    if result.returncode != 0:
        print(
            f"Non-breaking return code ({result.returncode}) while running {relative_path}"
        )


def test_examples_directory_exists():
    """Verify that the examples directory exists."""
    assert EXAMPLES_DIR.exists(), f"Examples directory not found at {EXAMPLES_DIR}"
    assert (
        EXAMPLES_DIR.is_dir()
    ), f"Examples path exists but is not a directory: {EXAMPLES_DIR}"


def test_found_example_files():
    """Verify that we found at least one example file to test."""
    assert len(example_files) > 0, "No example files found to test"
    print(f"\nFound {len(example_files)} example files to test:")
    for f in example_files:
        print(f"  - {f.relative_to(EXAMPLES_DIR)}")

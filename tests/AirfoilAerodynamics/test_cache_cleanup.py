"""
Test cache cleanup functionality for AirfoilAerodynamics.
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
from VSM.core.AirfoilAerodynamics import AirfoilAerodynamics


class TestCacheCleanup:
    """Test cases for cache cleanup functionality."""

    @pytest.fixture
    def config_path(self):
        """Get config path for masure_regression tests."""
        test_dir = Path(__file__).parent.parent.parent
        return str(
            test_dir
            / "data"
            / "TUDELFT_V3_KITE"
            / "config_kite_CAD_masure_regression.yaml"
        )

    @pytest.fixture
    def cache_dir(self, config_path):
        """Get cache directory and ensure it exists."""
        cache_dir = AirfoilAerodynamics._get_cache_dir(config_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def test_cache_cleanup_removes_old_files(self, cache_dir):
        """Test that old cache files are properly cleaned up."""
        # Create a fake old cache file (yesterday's date)
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        fake_old_file = cache_dir / f"aerodynamic_cache_{yesterday}_fakehash.pkl"

        # Create the fake old file
        with open(fake_old_file, "w") as f:
            f.write("fake cache data")

        # Verify the file was created
        assert fake_old_file.exists(), "Fake old cache file should exist before cleanup"

        # Trigger cleanup
        AirfoilAerodynamics._cleanup_old_cache_files(cache_dir)

        # Verify the old file was removed
        assert (
            not fake_old_file.exists()
        ), "Old cache file should be removed after cleanup"

    def test_cache_cleanup_preserves_current_files(self, cache_dir):
        """Test that current cache files are preserved during cleanup."""
        # Create a fake current cache file (today's date)
        today = datetime.now().strftime("%Y%m%d")
        fake_current_file = cache_dir / f"aerodynamic_cache_{today}_fakehash.pkl"

        # Create the fake current file
        with open(fake_current_file, "w") as f:
            f.write("fake current cache data")

        # Verify the file was created
        assert (
            fake_current_file.exists()
        ), "Fake current cache file should exist before cleanup"

        # Trigger cleanup
        AirfoilAerodynamics._cleanup_old_cache_files(cache_dir)

        # Verify the current file was preserved
        assert (
            fake_current_file.exists()
        ), "Current cache file should be preserved after cleanup"

        # Clean up the test file
        fake_current_file.unlink()

    def test_cache_cleanup_handles_empty_directory(self, cache_dir):
        """Test that cleanup works when cache directory is empty."""
        # Remove all cache files
        for cache_file in cache_dir.glob("aerodynamic_cache_*.pkl"):
            cache_file.unlink()

        # Trigger cleanup (should not raise an error)
        AirfoilAerodynamics._cleanup_old_cache_files(cache_dir)

        # Test passes if no exception is raised

    def test_cache_cleanup_handles_non_cache_files(self, cache_dir):
        """Test that cleanup ignores non-cache files."""
        # Create a non-cache file
        non_cache_file = cache_dir / "some_other_file.txt"
        with open(non_cache_file, "w") as f:
            f.write("not a cache file")

        # Trigger cleanup
        AirfoilAerodynamics._cleanup_old_cache_files(cache_dir)

        # Verify the non-cache file was preserved
        assert non_cache_file.exists(), "Non-cache files should be preserved"

        # Clean up the test file
        non_cache_file.unlink()

    def test_get_cache_dir_creates_directory(self, config_path):
        """Test that _get_cache_dir creates the cache directory if it doesn't exist."""
        cache_dir = AirfoilAerodynamics._get_cache_dir(config_path)

        # The cache directory should exist after calling _get_cache_dir
        assert (
            cache_dir.exists()
        ), "Cache directory should be created if it doesn't exist"
        assert cache_dir.is_dir(), "Cache path should be a directory"

    def test_get_cache_dir_path_construction(self, config_path):
        """Test that cache directory path is constructed correctly."""
        cache_dir = AirfoilAerodynamics._get_cache_dir(config_path)

        # Verify the path structure
        assert cache_dir.name == "cache", "Cache directory should be named 'cache'"
        assert cache_dir.parent.name == "data", "Cache should be under data directory"

        # Verify it's an absolute path
        assert cache_dir.is_absolute(), "Cache directory path should be absolute"

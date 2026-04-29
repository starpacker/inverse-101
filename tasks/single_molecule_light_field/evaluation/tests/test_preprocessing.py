"""Tests for preprocessing module."""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing import load_localizations, center_localizations, _pixel_size_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_standardized_locs(n=16, rng=None):
    """Create a synthetic (N, 8) standardized localisation array."""
    if rng is None:
        rng = np.random.default_rng(42)
    locs = np.zeros((n, 8))
    locs[:, 0] = rng.integers(1, 10, n)       # frame
    locs[:, 1] = rng.uniform(-10, 10, n)      # X um
    locs[:, 2] = rng.uniform(-10, 10, n)      # Y um
    locs[:, 3] = rng.uniform(0.05, 0.2, n)    # sigma_X
    locs[:, 4] = rng.uniform(0.05, 0.2, n)    # sigma_Y
    locs[:, 5] = rng.uniform(100, 5000, n)    # intensity (photons)
    locs[:, 6] = rng.uniform(10, 100, n)      # background (photons)
    locs[:, 7] = rng.uniform(0.01, 0.1, n)    # precision um
    return locs


def _default_meta():
    """Minimal meta_data dict matching the preprocessing pipeline."""
    return {
        "num_aperture": 1.27,
        "mla_lens_pitch": 222.0,
        "focal_length_mla": 5.556,
        "focal_length_obj_lens": 3.333,
        "focal_length_tube_lens": 200.0,
        "focal_length_fourier_lens": 150.0,
        "pixel_size_camera": 6.5,
        "ref_idx_immersion": 1.406,
        "ref_idx_medium": 1.33,
        "csv_format": "PEAKFIT",
    }


# ---------------------------------------------------------------------------
# Tests: load_localizations
# ---------------------------------------------------------------------------

class TestLoadLocalizations:
    """Tests for load_localizations with both standardized and PEAKFIT layouts."""

    def test_standardized_passthrough(self, tmp_path):
        """An (N, 8) array should be returned unchanged."""
        locs = _make_standardized_locs(32)
        npz_path = tmp_path / "raw_data.npz"
        np.savez(npz_path, localizations_2d=locs)

        result = load_localizations(npz_path, _default_meta())
        np.testing.assert_array_equal(result, locs)
        assert result.shape == (32, 8)
        assert result.dtype == float

    def test_standardized_shape_and_dtype(self, tmp_path):
        """Output must always be float64 (N, 8)."""
        locs = _make_standardized_locs(8).astype(np.float32)
        npz_path = tmp_path / "raw_data.npz"
        np.savez(npz_path, localizations_2d=locs)

        result = load_localizations(npz_path, _default_meta())
        assert result.shape == (8, 8)
        assert result.dtype == float

    def test_peakfit_conversion(self, tmp_path):
        """PEAKFIT raw columns should be converted to the standardized 8-column layout."""
        meta = _default_meta()
        rng = np.random.default_rng(99)
        n = 16
        # Build a raw array with >= 14 columns as PEAKFIT expects
        raw = rng.uniform(1, 100, (n, 14))
        raw[:, 0] = rng.integers(1, 10, n)  # frame

        npz_path = tmp_path / "raw_data.npz"
        np.savez(npz_path, localizations_2d=raw)

        result = load_localizations(npz_path, meta)
        assert result.shape == (n, 8)

        # Verify specific column mappings
        pixel_size = _pixel_size_sample(meta)
        np.testing.assert_allclose(result[:, 0], raw[:, 0])                     # frame
        np.testing.assert_allclose(result[:, 1], raw[:, 9] * pixel_size)        # X
        np.testing.assert_allclose(result[:, 2], raw[:, 10] * pixel_size)       # Y
        np.testing.assert_allclose(result[:, 3], raw[:, 12] * pixel_size)       # sigma_X
        np.testing.assert_allclose(result[:, 4], raw[:, 12] * pixel_size)       # sigma_Y
        np.testing.assert_allclose(result[:, 5], raw[:, 8])                     # intensity
        np.testing.assert_allclose(result[:, 6], raw[:, 7])                     # background
        np.testing.assert_allclose(result[:, 7], raw[:, 13] / 1000.0)          # precision

    def test_1d_array_raises(self, tmp_path):
        """A 1-D localisation array should raise ValueError."""
        npz_path = tmp_path / "raw_data.npz"
        np.savez(npz_path, localizations_2d=np.zeros(10))
        with pytest.raises(ValueError, match="2D localisation"):
            load_localizations(npz_path, _default_meta())

    def test_unsupported_columns_raises(self, tmp_path):
        """A 2-D array that is neither 8-col nor PEAKFIT-compatible should raise."""
        npz_path = tmp_path / "raw_data.npz"
        np.savez(npz_path, localizations_2d=np.zeros((10, 5)))
        with pytest.raises(ValueError, match="Unsupported"):
            load_localizations(npz_path, _default_meta())


# ---------------------------------------------------------------------------
# Tests: center_localizations
# ---------------------------------------------------------------------------

class TestCenterLocalizations:
    """Tests for center_localizations mean-subtraction."""

    def test_output_is_copy(self):
        """center_localizations must return a copy, not modify in-place."""
        locs = _make_standardized_locs(16)
        original = locs.copy()
        _ = center_localizations(locs)
        np.testing.assert_array_equal(locs, original)

    def test_mean_xy_is_zero(self):
        """After centering, mean(X) and mean(Y) should be ~0."""
        locs = _make_standardized_locs(32)
        centred = center_localizations(locs)
        np.testing.assert_allclose(centred[:, 1].mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(centred[:, 2].mean(), 0.0, atol=1e-12)

    def test_non_xy_columns_unchanged(self):
        """Columns other than X (1) and Y (2) must be preserved exactly."""
        locs = _make_standardized_locs(16)
        centred = center_localizations(locs)
        for col in [0, 3, 4, 5, 6, 7]:
            np.testing.assert_array_equal(centred[:, col], locs[:, col])

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        locs = _make_standardized_locs(8)
        centred = center_localizations(locs)
        assert centred.shape == locs.shape


# ---------------------------------------------------------------------------
# Tests: _pixel_size_sample
# ---------------------------------------------------------------------------

class TestPixelSizeSample:
    """Tests for the internal _pixel_size_sample helper."""

    def test_positive_result(self):
        """Pixel size in sample space must be positive."""
        meta = _default_meta()
        ps = _pixel_size_sample(meta)
        assert ps > 0

    def test_formula(self):
        """pixel_size_sample = pixel_size_camera / magnification."""
        meta = _default_meta()
        mag = (meta["focal_length_tube_lens"] / meta["focal_length_obj_lens"]
               * meta["focal_length_mla"] / meta["focal_length_fourier_lens"])
        expected = meta["pixel_size_camera"] / mag
        np.testing.assert_allclose(_pixel_size_sample(meta), expected, rtol=1e-12)

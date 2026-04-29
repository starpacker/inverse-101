"""Tests for the lensless imaging preprocessing module."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing import preprocess_psf, preprocess_measurement, load_npz


# ---------------------------------------------------------------------------
# Tests for preprocess_psf
# ---------------------------------------------------------------------------

class TestPreprocessPsf:
    def test_output_range_zero_to_one(self):
        """Output must be in [0, 1] after background-subtract + normalise."""
        rng = np.random.default_rng(0)
        psf_raw = rng.uniform(0.2, 0.8, (16, 16, 3)).astype(np.float32)
        result = preprocess_psf(psf_raw)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-7

    def test_max_is_one(self):
        """After normalisation the peak value should be exactly 1."""
        rng = np.random.default_rng(1)
        psf_raw = rng.uniform(0.1, 0.9, (8, 8, 3)).astype(np.float32)
        result = preprocess_psf(psf_raw)
        np.testing.assert_allclose(result.max(), 1.0, atol=1e-6)

    def test_min_is_zero(self):
        """After subtracting the minimum, the minimum value should be 0."""
        rng = np.random.default_rng(2)
        psf_raw = rng.uniform(0.3, 0.7, (8, 8, 3)).astype(np.float32)
        result = preprocess_psf(psf_raw)
        np.testing.assert_allclose(result.min(), 0.0, atol=1e-6)

    def test_output_dtype(self):
        psf_raw = np.ones((8, 8, 3), dtype=np.float64)
        result = preprocess_psf(psf_raw)
        assert result.dtype == np.float32

    def test_output_shape_unchanged(self):
        psf_raw = np.ones((12, 10, 3), dtype=np.float32) * 0.5
        result = preprocess_psf(psf_raw)
        assert result.shape == (12, 10, 3)

    def test_constant_input(self):
        """Constant input: all values equal -> subtraction gives 0, no division error."""
        psf_raw = np.full((8, 8, 3), 0.5, dtype=np.float32)
        result = preprocess_psf(psf_raw)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Tests for preprocess_measurement
# ---------------------------------------------------------------------------

class TestPreprocessMeasurement:
    def test_output_nonnegative(self):
        """Preprocessed measurement should have no negative values."""
        rng = np.random.default_rng(10)
        psf_raw = rng.uniform(0.1, 0.9, (16, 16, 3)).astype(np.float32)
        data_raw = rng.uniform(0.0, 1.0, (16, 16, 3)).astype(np.float32)
        result = preprocess_measurement(data_raw, psf_raw)
        assert np.all(result >= 0.0)

    def test_output_dtype(self):
        psf_raw = np.ones((8, 8, 3), dtype=np.float64)
        data_raw = np.ones((8, 8, 3), dtype=np.float64)
        result = preprocess_measurement(data_raw, psf_raw)
        assert result.dtype == np.float32

    def test_output_shape_unchanged(self):
        psf_raw = np.ones((10, 14, 3), dtype=np.float32) * 0.5
        data_raw = np.ones((10, 14, 3), dtype=np.float32) * 0.8
        result = preprocess_measurement(data_raw, psf_raw)
        assert result.shape == (10, 14, 3)

    def test_dark_subtraction(self):
        """When PSF min is subtracted, measurement values shift down accordingly."""
        psf_raw = np.full((8, 8, 3), 0.2, dtype=np.float32)
        psf_raw[0, 0, 0] = 0.8  # max = 0.8, min = 0.2
        data_raw = np.full((8, 8, 3), 0.5, dtype=np.float32)
        result = preprocess_measurement(data_raw, psf_raw)
        # data - psf.min() = 0.5 - 0.2 = 0.3, then / psf.max() = 0.3 / 0.8 = 0.375
        np.testing.assert_allclose(result, 0.375, atol=1e-5)

    def test_clipping_removes_negatives(self):
        """When data < psf.min(), result should be clipped to 0."""
        psf_raw = np.full((8, 8, 3), 0.5, dtype=np.float32)
        psf_raw[0, 0, 0] = 1.0  # max = 1.0, min = 0.5
        data_raw = np.full((8, 8, 3), 0.3, dtype=np.float32)  # 0.3 < 0.5
        result = preprocess_measurement(data_raw, psf_raw)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Tests for load_npz
# ---------------------------------------------------------------------------

class TestLoadNpz:
    def test_loads_and_removes_batch_dim(self, tmp_path):
        """load_npz should squeeze the leading batch dimension."""
        H, W, C = 8, 10, 3
        psf = np.random.rand(1, H, W, C).astype(np.float32)
        meas = np.random.rand(1, H, W, C).astype(np.float32)
        path = str(tmp_path / "test.npz")
        np.savez(path, psf=psf, measurement=meas)

        psf_out, data_out = load_npz(path)
        assert psf_out.shape == (H, W, C)
        assert data_out.shape == (H, W, C)

    def test_output_dtype_float32(self, tmp_path):
        H, W, C = 8, 8, 3
        psf = np.random.rand(1, H, W, C).astype(np.float64)
        meas = np.random.rand(1, H, W, C).astype(np.float64)
        path = str(tmp_path / "test.npz")
        np.savez(path, psf=psf, measurement=meas)

        psf_out, data_out = load_npz(path)
        assert psf_out.dtype == np.float32
        assert data_out.dtype == np.float32

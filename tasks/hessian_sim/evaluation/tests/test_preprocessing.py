"""Tests for src.preprocessing module."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing import (
    estimate_sim_parameters,
    estimate_modulation_and_phase,
    wiener_sim_reconstruct,
    running_average,
)


class TestRunningAverage:
    """running_average is a self-contained utility that can be tested independently."""

    def test_constant_stack_unchanged(self):
        """A constant stack should be unchanged after running average."""
        stack = np.ones((5, 8, 8), dtype=np.float32) * 3.0
        result = running_average(stack, window=3)
        # Interior frames (indices 1..3) should equal 3.0
        np.testing.assert_allclose(result[1:-1], 3.0, atol=1e-6)

    def test_output_shape(self):
        """Output shape should match input shape."""
        stack = np.random.default_rng(42).random((6, 8, 8)).astype(np.float32)
        result = running_average(stack, window=3)
        assert result.shape == stack.shape

    def test_output_dtype(self):
        """Output dtype should match input dtype."""
        stack = np.random.default_rng(42).random((5, 8, 8)).astype(np.float32)
        result = running_average(stack, window=3)
        assert result.dtype == stack.dtype

    def test_boundary_frames_zero(self):
        """Boundary frames that fall outside the window should remain zero."""
        stack = np.ones((5, 4, 4), dtype=np.float32)
        result = running_average(stack, window=3)
        # First and last frames are not filled by window=3
        np.testing.assert_array_equal(result[0], 0.0)
        np.testing.assert_array_equal(result[-1], 0.0)

    def test_averaging_correctness(self):
        """Interior frames should be the mean of the window."""
        stack = np.zeros((5, 4, 4), dtype=np.float32)
        stack[0] = 1.0
        stack[1] = 2.0
        stack[2] = 3.0
        stack[3] = 4.0
        stack[4] = 5.0
        result = running_average(stack, window=3)
        # Frame 1: mean of frames 0,1,2 = 2.0
        np.testing.assert_allclose(result[1], 2.0, atol=1e-6)
        # Frame 2: mean of frames 1,2,3 = 3.0
        np.testing.assert_allclose(result[2], 3.0, atol=1e-6)
        # Frame 3: mean of frames 2,3,4 = 4.0
        np.testing.assert_allclose(result[3], 4.0, atol=1e-6)

    def test_short_stack_returns_as_is(self):
        """A stack with fewer than 3 frames should be returned unchanged."""
        stack = np.ones((2, 4, 4), dtype=np.float32)
        result = running_average(stack, window=3)
        np.testing.assert_array_equal(result, stack)

    def test_2d_input_returns_as_is(self):
        """A 2D input should be returned unchanged."""
        arr = np.ones((8, 8), dtype=np.float32)
        result = running_average(arr, window=3)
        np.testing.assert_array_equal(result, arr)


class TestEstimateSimParameters:
    """Smoke tests for estimate_sim_parameters with tiny synthetic data."""

    def test_output_shapes(self):
        """zuobiaox, zuobiaoy should have shape (nframes, 1)."""
        n = 32
        nangles, nphases = 2, 3
        nframes = nangles * nphases
        rng = np.random.default_rng(42)
        raw = rng.random((nframes, 16, 16)).astype(np.float64) * 100
        otf = np.ones((n, n), dtype=np.float64)
        pg = 8
        fanwei = 10
        regul = 2 * np.pi
        spjg = [4, 4, 3]

        zuobiaox, zuobiaoy = estimate_sim_parameters(
            raw, otf, nangles, nphases, n, pg, fanwei, regul, spjg, beishu_an=1
        )
        assert zuobiaox.shape == (nframes, 1)
        assert zuobiaoy.shape == (nframes, 1)

    def test_output_dtype(self):
        """Output coordinates should be float."""
        n = 32
        nangles, nphases = 2, 3
        nframes = nangles * nphases
        rng = np.random.default_rng(42)
        raw = rng.random((nframes, 16, 16)).astype(np.float64) * 100
        otf = np.ones((n, n), dtype=np.float64)
        pg = 8
        fanwei = 10
        regul = 2 * np.pi
        spjg = [4, 4, 3]

        zuobiaox, zuobiaoy = estimate_sim_parameters(
            raw, otf, nangles, nphases, n, pg, fanwei, regul, spjg, beishu_an=1
        )
        assert np.issubdtype(zuobiaox.dtype, np.floating)
        assert np.issubdtype(zuobiaoy.dtype, np.floating)

    def test_no_nan_values(self):
        """Output should not contain NaN."""
        n = 32
        nangles, nphases = 2, 3
        nframes = nangles * nphases
        rng = np.random.default_rng(42)
        raw = rng.random((nframes, 16, 16)).astype(np.float64) * 100
        otf = np.ones((n, n), dtype=np.float64)
        pg = 8
        fanwei = 10
        regul = 2 * np.pi
        spjg = [4, 4, 3]

        zuobiaox, zuobiaoy = estimate_sim_parameters(
            raw, otf, nangles, nphases, n, pg, fanwei, regul, spjg, beishu_an=1
        )
        assert not np.any(np.isnan(zuobiaox))
        assert not np.any(np.isnan(zuobiaoy))

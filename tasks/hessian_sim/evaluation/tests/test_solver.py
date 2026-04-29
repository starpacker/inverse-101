"""Tests for src.solver module."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.solver import hessian_denoise, tv_denoise, _fdiff, _bdiff


class TestFdiff:
    def test_constant_array_zero(self):
        """Forward difference of a constant array should be zero."""
        x = np.ones((3, 8, 8))
        result = _fdiff(x, dim=2)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_output_shape(self):
        """Forward difference should preserve shape."""
        x = np.random.default_rng(42).random((3, 8, 8))
        for dim in [0, 1, 2]:
            result = _fdiff(x, dim)
            assert result.shape == x.shape

    def test_periodic_boundary(self):
        """Forward diff wraps around: fdiff[last] = x[0] - x[last]."""
        x = np.zeros((1, 1, 4))
        x[0, 0, :] = [1.0, 3.0, 6.0, 10.0]
        result = _fdiff(x, dim=2)
        expected = [2.0, 3.0, 4.0, -9.0]  # wraps: 1 - 10 = -9
        np.testing.assert_allclose(result[0, 0, :], expected, atol=1e-14)


class TestBdiff:
    def test_constant_array_zero(self):
        """Backward difference of a constant array should be zero."""
        x = np.ones((3, 8, 8))
        result = _bdiff(x, dim=1)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_adjoint_relationship(self):
        """_bdiff should be the negative adjoint of _fdiff (with periodic BC).
        That is: <fdiff(x), y> = -<x, bdiff(y)> for periodic BC."""
        rng = np.random.default_rng(42)
        x = rng.random((3, 8, 8))
        y = rng.random((3, 8, 8))
        for dim in [0, 1, 2]:
            lhs = np.sum(_fdiff(x, dim) * y)
            rhs = -np.sum(x * _bdiff(y, dim))
            np.testing.assert_allclose(lhs, rhs, atol=1e-10)


class TestHessianDenoise:
    def test_output_shape_3d(self):
        """Output should have same shape as input for 3D input."""
        rng = np.random.default_rng(42)
        stack = rng.random((3, 16, 16)).astype(np.float64) * 100
        result = hessian_denoise(stack, mu=50.0, sigma_z=1.0, n_iter=5, lamda=1.0)
        assert result.shape == stack.shape

    def test_output_shape_2d(self):
        """A 2D input should produce a 3D output with nz=1."""
        rng = np.random.default_rng(42)
        img = rng.random((16, 16)).astype(np.float64) * 100
        result = hessian_denoise(img, mu=50.0, sigma_z=1.0, n_iter=5, lamda=1.0)
        assert result.shape == (1, 16, 16)

    def test_nonnegative_output(self):
        """Output should be non-negative (clipping is applied)."""
        rng = np.random.default_rng(42)
        stack = rng.random((3, 16, 16)).astype(np.float64) * 100
        result = hessian_denoise(stack, mu=50.0, sigma_z=1.0, n_iter=10, lamda=1.0)
        assert np.all(result >= 0)

    def test_zero_input_returns_zero(self):
        """Zero input should yield zero output."""
        stack = np.zeros((3, 8, 8), dtype=np.float64)
        result = hessian_denoise(stack, mu=50.0, sigma_z=1.0, n_iter=5, lamda=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_constant_input_preserved(self):
        """A constant input should be approximately preserved (no gradients to penalize)."""
        stack = np.ones((3, 16, 16), dtype=np.float64) * 50.0
        result = hessian_denoise(stack, mu=200.0, sigma_z=1.0, n_iter=20, lamda=1.0)
        # With high mu (data fidelity), constant should be nearly preserved
        np.testing.assert_allclose(result, 50.0, rtol=0.1)

    def test_denoising_reduces_variance(self):
        """Denoising a noisy constant should reduce pixel variance."""
        rng = np.random.default_rng(42)
        clean = np.ones((3, 16, 16), dtype=np.float64) * 100.0
        noisy = clean + rng.normal(0, 10, clean.shape)
        noisy[noisy < 0] = 0
        result = hessian_denoise(noisy, mu=50.0, sigma_z=1.0, n_iter=30, lamda=0.5)
        assert np.var(result) < np.var(noisy)

    def test_single_frame_no_crash(self):
        """A single-frame input (nz=1) should not crash (z-padding is applied)."""
        rng = np.random.default_rng(42)
        stack = rng.random((1, 16, 16)).astype(np.float64) * 100
        result = hessian_denoise(stack, mu=50.0, sigma_z=1.0, n_iter=5, lamda=1.0)
        assert result.shape == (1, 16, 16)


class TestTvDenoise:
    def test_output_shape_3d(self):
        """Output should match input shape for 3D input."""
        rng = np.random.default_rng(42)
        stack = rng.random((3, 16, 16)).astype(np.float64) * 100
        result = tv_denoise(stack, mu=50.0, n_iter=5)
        assert result.shape == stack.shape

    def test_output_shape_2d(self):
        """A 2D input should produce a 3D output with nz=1."""
        rng = np.random.default_rng(42)
        img = rng.random((16, 16)).astype(np.float64) * 100
        result = tv_denoise(img, mu=50.0, n_iter=5)
        assert result.shape == (1, 16, 16)

    def test_output_dtype_float32(self):
        """TV denoise should return float32."""
        rng = np.random.default_rng(42)
        stack = rng.random((3, 8, 8)).astype(np.float64) * 100
        result = tv_denoise(stack, mu=50.0, n_iter=5)
        assert result.dtype == np.float32

    def test_nonnegative_output(self):
        """Output should be non-negative."""
        rng = np.random.default_rng(42)
        stack = rng.random((3, 16, 16)).astype(np.float64) * 100
        result = tv_denoise(stack, mu=50.0, n_iter=10)
        assert np.all(result >= 0)

    def test_zero_input_returns_zero(self):
        """Zero input should yield zero output."""
        stack = np.zeros((3, 8, 8), dtype=np.float64)
        result = tv_denoise(stack, mu=50.0, n_iter=5)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_constant_input_preserved(self):
        """A constant input should be approximately preserved."""
        stack = np.ones((3, 16, 16), dtype=np.float64) * 50.0
        result = tv_denoise(stack, mu=200.0, n_iter=20)
        np.testing.assert_allclose(result, 50.0, rtol=0.1)

    def test_denoising_reduces_total_variation(self):
        """TV denoising should reduce total variation of a noisy signal."""
        rng = np.random.default_rng(42)
        clean = np.ones((3, 16, 16), dtype=np.float64) * 100.0
        noisy = clean + rng.normal(0, 15, clean.shape)
        noisy[noisy < 0] = 0

        def total_variation(x):
            dx = np.abs(np.diff(x, axis=2)).sum()
            dy = np.abs(np.diff(x, axis=1)).sum()
            return dx + dy

        result = tv_denoise(noisy, mu=50.0, n_iter=30)
        assert total_variation(result) < total_variation(noisy)

    def test_high_mu_stays_close_to_input(self):
        """With very high mu (data fidelity), output should closely match input."""
        rng = np.random.default_rng(42)
        stack = rng.random((3, 8, 8)).astype(np.float64) * 100
        stack[stack < 0] = 0
        result = tv_denoise(stack, mu=1e6, n_iter=20)
        np.testing.assert_allclose(result, stack, rtol=0.05, atol=1.0)

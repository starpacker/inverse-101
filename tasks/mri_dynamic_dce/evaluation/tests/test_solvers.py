"""Tests for src/solvers.py"""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.solvers import (
    zero_filled_recon, temporal_tv_pgd, temporal_tv_admm,
    _temporal_diff, _temporal_diff_adjoint, _soft_threshold_complex,
)
from src.physics_model import fft2c, ifft2c


@pytest.fixture
def solver_fixtures():
    return np.load(os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'solvers.npz'))


@pytest.fixture
def parity_fixtures():
    return np.load(os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'solver_parity.npz'))


class TestTemporalDiff:
    def test_values(self, solver_fixtures):
        x = solver_fixtures['input_x_test']
        expected = solver_fixtures['output_temporal_diff']
        np.testing.assert_allclose(_temporal_diff(x), expected, rtol=1e-10)

    def test_shape(self):
        x = np.random.randn(5, 8, 8)
        dx = _temporal_diff(x)
        assert dx.shape == (4, 8, 8)

    def test_adjoint_values(self, solver_fixtures):
        x = solver_fixtures['input_x_test']
        dx = _temporal_diff(x)
        T = x.shape[0]
        expected = solver_fixtures['output_temporal_diff_adjoint']
        np.testing.assert_allclose(
            _temporal_diff_adjoint(dx, T), expected, rtol=1e-10)

    def test_adjoint_property(self):
        """<Dx, y> == <x, D^H y> for random vectors."""
        rng = np.random.RandomState(42)
        T, N = 6, 8
        x = rng.randn(T, N, N)
        y = rng.randn(T - 1, N, N)

        Dx = _temporal_diff(x)
        DHy = _temporal_diff_adjoint(y, T)

        lhs = np.sum(Dx * y)
        rhs = np.sum(x * DHy)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


class TestSoftThreshold:
    def test_zero_threshold(self):
        x = np.array([1.0 + 2j, -3.0, 0.5j])
        result = _soft_threshold_complex(x, 0.0)
        np.testing.assert_allclose(result, x, rtol=1e-10)

    def test_large_threshold(self):
        x = np.array([1.0, -0.5, 0.3])
        result = _soft_threshold_complex(x, 10.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_partial_shrinkage(self):
        x = np.array([3.0, -2.0, 0.5])
        result = _soft_threshold_complex(x, 1.0)
        expected = np.array([2.0, -1.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestZeroFilled:
    def test_shape(self):
        rng = np.random.RandomState(0)
        ksp = rng.randn(5, 16, 16) + 1j * rng.randn(5, 16, 16)
        recon = zero_filled_recon(ksp)
        assert recon.shape == (5, 16, 16)

    def test_non_negative(self):
        rng = np.random.RandomState(0)
        ksp = rng.randn(5, 16, 16) + 1j * rng.randn(5, 16, 16)
        recon = zero_filled_recon(ksp)
        assert np.all(recon >= 0)

    def test_dtype(self):
        rng = np.random.RandomState(0)
        ksp = rng.randn(3, 8, 8) + 1j * rng.randn(3, 8, 8)
        recon = zero_filled_recon(ksp)
        assert recon.dtype == np.float64

    def test_parity(self, parity_fixtures):
        ksp = parity_fixtures['input_kspace']
        expected = parity_fixtures['output_zero_fill']
        np.testing.assert_allclose(zero_filled_recon(ksp), expected, rtol=1e-10)


class TestTemporalTVPGD:
    def test_output_shape(self):
        rng = np.random.RandomState(0)
        T, N = 4, 16
        images = rng.randn(T, N, N)
        masks = (rng.rand(T, N, N) > 0.5).astype(float)
        ksp = fft2c(images) * masks

        recon, info = temporal_tv_pgd(ksp, masks, lamda=0.01, max_iter=5)
        assert recon.shape == (T, N, N)
        assert 'loss_history' in info
        assert 'num_iter' in info

    def test_non_negative_output(self):
        rng = np.random.RandomState(0)
        T, N = 4, 16
        images = np.abs(rng.randn(T, N, N))
        masks = (rng.rand(T, N, N) > 0.5).astype(float)
        ksp = fft2c(images) * masks

        recon, _ = temporal_tv_pgd(ksp, masks, lamda=0.01, max_iter=10)
        assert np.all(recon >= 0)

    def test_loss_decreasing(self):
        """Loss should generally decrease over iterations."""
        rng = np.random.RandomState(0)
        T, N = 4, 16
        images = np.abs(rng.randn(T, N, N))
        masks = (rng.rand(T, N, N) > 0.5).astype(float)
        ksp = fft2c(images) * masks

        _, info = temporal_tv_pgd(ksp, masks, lamda=0.001, max_iter=50)
        losses = info['loss_history']
        # Allow some fluctuation but overall trend should be decreasing
        assert losses[-1] < losses[0]

    def test_parity(self, parity_fixtures):
        """Check that solver reproduces fixture output."""
        ksp = parity_fixtures['input_kspace']
        masks = parity_fixtures['input_masks']
        expected = parity_fixtures['output_tv_recon']
        lamda = float(parity_fixtures['config_lamda'])
        max_iter = int(parity_fixtures['config_max_iter'])

        recon, _ = temporal_tv_pgd(ksp, masks, lamda=lamda,
                                   max_iter=max_iter, tol=1e-8)
        np.testing.assert_allclose(recon, expected, rtol=1e-6)


class TestTemporalTVADMM:
    def test_output_shape(self):
        rng = np.random.RandomState(0)
        T, N = 4, 16
        images = rng.randn(T, N, N)
        masks = (rng.rand(T, N, N) > 0.5).astype(float)
        ksp = fft2c(images) * masks

        recon, info = temporal_tv_admm(ksp, masks, lamda=0.01, max_iter=5)
        assert recon.shape == (T, N, N)
        assert 'loss_history' in info

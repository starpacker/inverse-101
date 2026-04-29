"""Tests for the iterative solvers (SART, TV-PDHG)."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.solvers import (
    sart_reconstruction, tv_pdhg_reconstruction,
    _gradient_2d, _divergence_2d,
)
from src.physics_model import radon_forward
from src.visualization import compute_ncc


@pytest.fixture
def small_problem():
    """Create a small test problem (32x32)."""
    N = 32
    y, x = np.ogrid[:N, :N]
    cx, cy = N / 2, N / 2
    phantom = np.zeros((N, N))
    phantom[(x - cx)**2 + (y - cy)**2 < (N * 0.25)**2] = 1e-5

    angles = np.linspace(0, 180, 30, endpoint=False)
    sino = radon_forward(phantom, angles)
    return phantom, sino, angles, N


def test_gradient_2d_shape():
    x = np.random.default_rng(42).random((16, 16))
    grad = _gradient_2d(x)
    assert grad.shape == (2, 16, 16)


def test_gradient_divergence_adjoint():
    """Gradient and divergence should satisfy <Dx, p> = <x, -div(p)>."""
    rng = np.random.default_rng(42)
    x = rng.random((16, 16))
    p = rng.random((2, 16, 16))

    Dx = _gradient_2d(x)
    div_p = _divergence_2d(p)

    lhs = np.sum(Dx * p)
    rhs = -np.sum(x * div_p)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_sart_output_shape(small_problem):
    phantom, sino, angles, N = small_problem
    recon, loss = sart_reconstruction(sino, angles, N, n_iter=5)
    assert recon.shape == (N, N)
    assert len(loss) == 5


def test_sart_loss_decreasing(small_problem):
    """SART loss should generally decrease."""
    phantom, sino, angles, N = small_problem
    _, loss = sart_reconstruction(sino, angles, N, n_iter=20)
    # Allow some non-monotonicity but overall trend should be down
    assert loss[-1] < loss[0], "SART loss did not decrease overall"


def test_sart_improves_over_zero(small_problem):
    """SART reconstruction should be better than all-zeros."""
    phantom, sino, angles, N = small_problem
    recon, _ = sart_reconstruction(sino, angles, N, n_iter=10)
    ncc = compute_ncc(recon, phantom)
    assert ncc > 0.5, f"SART NCC too low: {ncc}"


def test_tv_pdhg_output_shape(small_problem):
    phantom, sino, angles, N = small_problem
    recon, loss = tv_pdhg_reconstruction(sino, angles, N, n_iter=5)
    assert recon.shape == (N, N)
    assert len(loss) == 5


def test_tv_pdhg_loss_decreasing(small_problem):
    """TV-PDHG loss should generally decrease."""
    phantom, sino, angles, N = small_problem
    _, loss = tv_pdhg_reconstruction(sino, angles, N, lam=1e-7, n_iter=50)
    assert loss[-1] < loss[0], "TV-PDHG loss did not decrease overall"


def test_tv_pdhg_improves_over_fbp(small_problem):
    """TV-PDHG should achieve reasonable NCC."""
    phantom, sino, angles, N = small_problem
    recon, _ = tv_pdhg_reconstruction(sino, angles, N, lam=1e-7, n_iter=50)
    ncc = compute_ncc(recon, phantom)
    assert ncc > 0.5, f"TV-PDHG NCC too low: {ncc}"

"""Tests for the TV reconstruction solver."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.solvers import gradient_2d, divergence_2d, tv_reconstruction
from src.physics_model import radon_transform


def test_gradient_shape():
    x = np.random.rand(16, 16)
    grad = gradient_2d(x)
    assert grad.shape == (2, 16, 16)


def test_gradient_constant_image():
    """Gradient of constant image should be zero."""
    x = np.ones((16, 16)) * 5.0
    grad = gradient_2d(x)
    np.testing.assert_allclose(grad, 0.0, atol=1e-15)


def test_gradient_divergence_adjointness():
    """<grad(x), p> should equal -<x, div(p)> (adjointness)."""
    rng = np.random.default_rng(42)
    x = rng.random((32, 32))
    p = rng.random((2, 32, 32))

    grad_x = gradient_2d(x)
    div_p = divergence_2d(p)

    lhs = np.sum(grad_x * p)
    rhs = -np.sum(x * div_p)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_gradient_deterministic():
    x = np.random.rand(16, 16)
    g1 = gradient_2d(x)
    g2 = gradient_2d(x)
    np.testing.assert_array_equal(g1, g2)


def test_divergence_shape():
    p = np.random.rand(2, 16, 16)
    div = divergence_2d(p)
    assert div.shape == (16, 16)


def test_tv_reconstruction_output_shape():
    """TV reconstruction should return correct shape."""
    N = 32
    angles = np.linspace(0, 180, 10, endpoint=False)
    x = np.zeros((N, N))
    x[10:22, 10:22] = 1.0  # simple square
    sino = radon_transform(x, angles)

    recon, loss = tv_reconstruction(sino, angles, N, lam=0.01, n_iter=5)
    assert recon.shape == (N, N)
    assert len(loss) == 5


def test_tv_reconstruction_positivity():
    """With positivity=True, reconstruction should be non-negative."""
    N = 32
    angles = np.linspace(0, 180, 10, endpoint=False)
    x = np.zeros((N, N))
    x[10:22, 10:22] = 1.0
    sino = radon_transform(x, angles)

    recon, _ = tv_reconstruction(sino, angles, N, lam=0.01, n_iter=20,
                                  positivity=True)
    assert recon.min() >= -1e-10, f"Min value {recon.min()} violates positivity"


def test_tv_reconstruction_loss_decreasing():
    """Loss should generally decrease over iterations."""
    N = 32
    angles = np.linspace(0, 180, 10, endpoint=False)
    x = np.zeros((N, N))
    x[10:22, 10:22] = 1.0
    sino = radon_transform(x, angles)

    _, loss = tv_reconstruction(sino, angles, N, lam=0.01, n_iter=50)
    # Check that final loss is less than initial (not necessarily monotonic for PDHG)
    assert loss[-1] < loss[0], "Loss did not decrease overall"


def test_tv_reconstruction_reference_parity():
    """TV reconstruction on task data must match reference outputs."""
    ref_path = os.path.join(TASK_DIR, "evaluation", "reference_outputs",
                            "reconstructions.npz")
    if not os.path.exists(ref_path):
        pytest.skip("Reference outputs not found")

    ref = np.load(ref_path)
    tv_ref = ref["tv_recon"].squeeze(0)

    data_dir = os.path.join(TASK_DIR, "data")
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))
    sino = raw["sinogram_sparse"].squeeze(0)
    angles = raw["angles_sparse"].squeeze(0)
    meta_path = os.path.join(data_dir, "meta_data.json")
    import json
    with open(meta_path) as f:
        meta = json.load(f)

    recon, _ = tv_reconstruction(sino, angles, meta["image_size"],
                                  lam=0.01, n_iter=300, positivity=True)

    np.testing.assert_allclose(recon, tv_ref, rtol=1e-6, atol=1e-8)

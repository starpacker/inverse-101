"""Tests for the TV-PDHG reconstruction solver."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.solvers import solve_tv_pdhg, _gradient_2d, _divergence_2d, _prox_l1_norm
from src.physics_model import fan_beam_geometry, fan_beam_forward_vectorized


# --- Gradient / Divergence operators ---

def test_gradient_shape():
    x = np.random.rand(16, 16)
    grad = _gradient_2d(x)
    assert grad.shape == (2, 16, 16)


def test_gradient_constant_image():
    """Gradient of constant image should be zero."""
    x = np.ones((16, 16)) * 5.0
    grad = _gradient_2d(x)
    np.testing.assert_allclose(grad, 0.0, atol=1e-15)


def test_gradient_divergence_adjointness():
    """<grad(x), p> should equal -<x, div(p)> (adjointness)."""
    rng = np.random.default_rng(42)
    x = rng.random((32, 32))
    p = rng.random((2, 32, 32))

    grad_x = _gradient_2d(x)
    div_p = _divergence_2d(p)

    lhs = np.sum(grad_x * p)
    rhs = -np.sum(x * div_p)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_gradient_deterministic():
    x = np.random.rand(16, 16)
    g1 = _gradient_2d(x)
    g2 = _gradient_2d(x)
    np.testing.assert_array_equal(g1, g2)


def test_divergence_shape():
    p = np.random.rand(2, 16, 16)
    div = _divergence_2d(p)
    assert div.shape == (16, 16)


# --- Proximal operator ---

def test_prox_l1_norm_shrinks():
    """Proximal should shrink gradient magnitudes."""
    rng = np.random.default_rng(42)
    p = rng.random((2, 8, 8)) * 10  # large values
    result = _prox_l1_norm(p, sigma=1.0)
    # Result magnitudes should be <= original magnitudes
    orig_mag = np.sqrt(p[0]**2 + p[1]**2)
    result_mag = np.sqrt(result[0]**2 + result[1]**2)
    assert np.all(result_mag <= orig_mag + 1e-10)


# --- TV-PDHG solver ---

def test_tv_output_shape():
    geo = fan_beam_geometry(32, 48, 18, 128, 128, angle_range=2*np.pi)
    image = np.zeros((32, 32))
    image[10:22, 10:22] = 1.0
    sino = fan_beam_forward_vectorized(image, geo)

    recon, loss = solve_tv_pdhg(sino, geo, lam=0.01, n_iter=5, verbose=False)
    assert recon.shape == (32, 32)
    assert len(loss) == 5


def test_tv_positivity():
    """With positivity=True, reconstruction should be non-negative."""
    geo = fan_beam_geometry(32, 48, 18, 128, 128, angle_range=2*np.pi)
    sino = np.ones((18, 48)) * 0.1
    recon, _ = solve_tv_pdhg(sino, geo, lam=0.01, n_iter=10,
                              positivity=True, verbose=False)
    assert recon.min() >= -1e-10, f"Min value {recon.min()} violates positivity"


def test_tv_loss_decreasing():
    """Loss should generally decrease over iterations."""
    geo = fan_beam_geometry(32, 48, 18, 128, 128, angle_range=2*np.pi)
    image = np.zeros((32, 32))
    image[10:22, 10:22] = 1.0
    sino = fan_beam_forward_vectorized(image, geo)

    _, loss = solve_tv_pdhg(sino, geo, lam=0.01, n_iter=30, verbose=False)
    assert loss[-1] < loss[0], "Loss did not decrease overall"


def test_tv_reference_parity():
    """TV reconstruction on task data must match reference outputs."""
    ref_path = os.path.join(TASK_DIR, "evaluation", "reference_outputs",
                            "recon_tv_short.npz")
    if not os.path.exists(ref_path):
        pytest.skip("Reference outputs not found")

    ref = np.load(ref_path)
    tv_ref = ref["reconstruction"].squeeze(0)

    data_dir = os.path.join(TASK_DIR, "data")
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))
    import json
    with open(os.path.join(data_dir, "meta_data.json")) as f:
        meta = json.load(f)

    sino_short = raw["sino_short"].squeeze(0).astype(np.float64)
    angles_short = raw["angles_short"].astype(np.float64)
    N = meta["image_size"]
    D_sd = meta["source_to_isocenter_pixels"]
    D_dd = meta["isocenter_to_detector_pixels"]
    short_range = meta["short_scan_range_deg"] * np.pi / 180

    geo_short = fan_beam_geometry(N, meta["n_det"], len(angles_short),
                                   D_sd, D_dd, angle_range=short_range)

    recon, _ = solve_tv_pdhg(sino_short, geo_short, lam=0.005, n_iter=150,
                              positivity=True, verbose=False)

    from src.visualization import compute_ncc
    ncc = compute_ncc(recon.ravel(), tv_ref.ravel())
    assert ncc > 0.999, f"Reference parity NCC too low: {ncc}"

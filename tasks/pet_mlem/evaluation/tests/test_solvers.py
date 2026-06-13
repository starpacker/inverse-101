"""Tests for src/solvers.py."""
import numpy as np
import pytest
import sys, os

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

from src.physics_model import pet_forward_project
from src.solvers import solve_mlem, solve_osem
from src.visualization import compute_ncc


def _make_test_sinogram(N=32, n_angles=30):
    """Create a test sinogram with correct shape for image size N."""
    theta = np.linspace(0, 180, n_angles, endpoint=False)
    image = np.zeros((N, N))
    image[N//4:3*N//4, N//4:3*N//4] = 1.0
    sino = pet_forward_project(image, theta)
    sino = np.maximum(sino, 0.1)
    return sino, theta


class TestMLEM:
    def test_runs(self):
        sino, theta = _make_test_sinogram()
        recon, ll = solve_mlem(sino, theta, 32, n_iter=5, verbose=False)
        assert recon.shape == (32, 32)
        assert len(ll) == 5

    def test_non_negative(self):
        sino, theta = _make_test_sinogram()
        recon, _ = solve_mlem(sino, theta, 32, n_iter=5, verbose=False)
        assert np.all(recon >= 0)

    def test_likelihood_increases(self):
        """Log-likelihood should generally increase with MLEM iterations."""
        sino, theta = _make_test_sinogram()
        _, ll = solve_mlem(sino, theta, 32, n_iter=20, verbose=False)
        assert ll[-1] > ll[0]


class TestOSEM:
    def test_runs(self):
        sino, theta = _make_test_sinogram()
        recon, ll = solve_osem(sino, theta, 32, n_iter=3, n_subsets=3,
                                verbose=False)
        assert recon.shape == (32, 32)
        assert len(ll) == 3

    def test_non_negative(self):
        sino, theta = _make_test_sinogram()
        recon, _ = solve_osem(sino, theta, 32, n_iter=3, n_subsets=3,
                               verbose=False)
        assert np.all(recon >= 0)


class TestReferenceParity:
    def test_mlem_reference_parity(self):
        """MLEM on task data must match reference output."""
        ref_path = os.path.join(TASK_DIR, "evaluation", "reference_outputs",
                                "recon_mlem.npz")
        if not os.path.exists(ref_path):
            pytest.skip("Reference outputs not found")

        ref = np.load(ref_path)["reconstruction"].squeeze(0)

        from src.preprocessing import load_sinogram_data, load_ground_truth, load_metadata, preprocess_sinogram
        sino, bg, theta = load_sinogram_data(TASK_DIR)
        meta = load_metadata(TASK_DIR)
        sino_2d = preprocess_sinogram(sino)
        bg_2d = preprocess_sinogram(bg)
        theta_1d = theta.squeeze(0)

        recon, _ = solve_mlem(sino_2d, theta_1d, meta['image_size'],
                               n_iter=50, background=bg_2d, verbose=False)

        ncc = compute_ncc(recon.ravel(), ref.ravel())
        assert ncc > 0.999, f"MLEM reference parity NCC too low: {ncc}"

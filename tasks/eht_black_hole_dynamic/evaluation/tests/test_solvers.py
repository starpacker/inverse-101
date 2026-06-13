"""Tests for solvers module."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'solvers')

import sys
sys.path.insert(0, TASK_DIR)

from src.physics_model import DFTForwardModel, gauss_image_covariance
from src.solvers import solve_single_image, StaticPerFrameSolver


class TestSolveSingleImage:
    """Tests for single-image Gaussian MAP solver."""

    def test_solve_single_image_parity(self):
        """Test solve_single_image output matches fixture."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'solve_single_image.npz'))
        N = int(fix['param_N'])
        psize = float(fix['param_psize'])

        model = DFTForwardModel(fix['input_uv'], N, psize)
        prior_cov = gauss_image_covariance(N, psize, fix['input_prior_mean'])

        z_vec, P, z_lin = solve_single_image(
            fix['input_prior_mean'], prior_cov,
            model, fix['input_vis'], fix['input_sigma'])

        np.testing.assert_allclose(z_vec, fix['output_z_vec'], rtol=1e-10)

    def test_solve_single_image_shape(self):
        """Test output shapes are correct."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'solve_single_image.npz'))
        N = int(fix['param_N'])
        psize = float(fix['param_psize'])
        npixels = N * N

        model = DFTForwardModel(fix['input_uv'], N, psize)
        prior_cov = gauss_image_covariance(N, psize, fix['input_prior_mean'])

        z_vec, P, z_lin = solve_single_image(
            fix['input_prior_mean'], prior_cov,
            model, fix['input_vis'], fix['input_sigma'])

        assert z_vec.shape == (npixels,)
        assert P.shape == (npixels, npixels)
        assert z_lin.shape == (npixels,)


class TestStaticPerFrameSolver:
    """Tests for static per-frame solver."""

    def test_static_solver_output_shapes(self):
        """Test that StaticPerFrameSolver returns correct frame shapes."""
        N = 8
        psize = 1e-11
        rng = np.random.default_rng(999)

        uv = rng.standard_normal((10, 2)) * 1e9
        model = DFTForwardModel(uv, N, psize)

        prior_mean = np.ones(N * N) / (N * N)
        prior_cov = gauss_image_covariance(N, psize, prior_mean)

        vis = rng.standard_normal(10) + 1j * rng.standard_normal(10)
        sigma = np.ones(10) * 0.1

        obs = {
            'vis': [vis, vis],
            'sigma': [sigma, sigma],
        }

        solver = StaticPerFrameSolver(prior_mean, prior_cov)
        frames = solver.reconstruct([model, model], obs, N)

        assert len(frames) == 2
        for f in frames:
            assert f.shape == (N, N)
            assert np.all(f >= 0)  # clipped non-negative

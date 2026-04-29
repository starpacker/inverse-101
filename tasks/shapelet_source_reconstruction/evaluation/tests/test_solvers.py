"""Unit tests for solvers.py"""

import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.solvers import build_response_matrix, linear_solve, reduced_residuals
from src.physics_model import array2image

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures')


class TestLinearSolve:
    def test_recovers_clean_signal(self):
        """With zero noise, WLS should recover exact parameters."""
        np.random.seed(789)
        A = np.random.randn(3, 16)
        true_params = np.array([1.0, -0.5, 0.3])
        data_1d = A.T.dot(true_params)
        data_2d = array2image(data_1d)
        params, model = linear_solve(A, data_2d, 1e10, 1e10)
        np.testing.assert_allclose(params, true_params, rtol=1e-5)

    def test_fixture(self):
        inp = np.load(os.path.join(FIXTURE_DIR, 'input_linear_solve.npz'))
        out = np.load(os.path.join(FIXTURE_DIR, 'output_linear_solve.npz'))
        par = np.load(os.path.join(FIXTURE_DIR, 'param_linear_solve.npz'))
        params, model = linear_solve(inp['A'], inp['data_noisy'],
                                      float(par['background_rms']), float(par['exp_time']))
        np.testing.assert_allclose(params, out['param_solved'], rtol=1e-8)
        np.testing.assert_allclose(model, out['model_solved'], rtol=1e-8)


class TestReducedResiduals:
    def test_zero_residual(self):
        model = np.ones((4, 4)) * 10
        res = reduced_residuals(model, model, 5.0, 100.0)
        np.testing.assert_array_equal(res, 0.0)

    def test_fixture(self):
        inp = np.load(os.path.join(FIXTURE_DIR, 'input_reduced_residuals.npz'))
        par = np.load(os.path.join(FIXTURE_DIR, 'param_reduced_residuals.npz'))
        out = np.load(os.path.join(FIXTURE_DIR, 'output_reduced_residuals.npz'))
        res = reduced_residuals(inp['model'], inp['data'],
                                 float(par['background_rms']), float(par['exp_time']))
        np.testing.assert_allclose(res, out['residuals'], rtol=1e-10)

    def test_normalization(self):
        """For pure background noise, reduced residuals should have std ~ 1."""
        np.random.seed(0)
        model = np.ones((50, 50)) * 100.0
        noise = np.random.randn(50, 50) * 5.0
        data = model + noise
        res = reduced_residuals(model, data, 5.0, 1e10)
        assert np.isclose(np.std(res), 1.0, rtol=0.15)


class TestResponseMatrix:
    def test_shape(self):
        n_max = 3
        numPix = 8
        A = build_response_matrix(numPix, 0.1, 1, 0.0,
                                   n_max, 0.5, 0.0, 0.0, apply_lens=False)
        expected_rows = (n_max + 1) * (n_max + 2) // 2
        assert A.shape == (expected_rows, numPix**2)

    def test_nonzero(self):
        A = build_response_matrix(8, 0.1, 1, 0.0,
                                   2, 0.5, 0.0, 0.0, apply_lens=False)
        assert np.any(A != 0)

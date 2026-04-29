"""Tests for physics_model module."""

import pathlib
import sys

import numpy as np
import pytest

TASK_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

from src.physics_model import forward, residual, mse


class TestForward:
    def test_shape(self):
        C = np.random.rand(100, 3)
        ST = np.random.rand(3, 50)
        D = forward(C, ST)
        assert D.shape == (100, 50)

    def test_identity(self):
        C = np.eye(3)
        ST = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        D = forward(C, ST)
        np.testing.assert_array_equal(D, ST)

    def test_single_component(self):
        C = np.ones((10, 1)) * 0.5
        ST = np.ones((1, 20)) * 4.0
        D = forward(C, ST)
        np.testing.assert_allclose(D, 2.0)


class TestResidual:
    def test_zero_residual(self):
        C = np.random.rand(50, 3)
        ST = np.random.rand(3, 30)
        D = C @ ST
        R = residual(C, ST, D)
        np.testing.assert_allclose(R, 0.0, atol=1e-12)

    def test_nonzero_residual(self):
        C = np.random.rand(50, 3)
        ST = np.random.rand(3, 30)
        D = C @ ST + 1.0
        R = residual(C, ST, D)
        np.testing.assert_allclose(R, 1.0, atol=1e-12)


class TestMSE:
    def test_perfect_fit(self):
        C = np.random.rand(50, 3)
        ST = np.random.rand(3, 30)
        D = C @ ST
        assert mse(C, ST, D) < 1e-24

    def test_known_error(self):
        C = np.zeros((10, 1))
        ST = np.zeros((1, 10))
        D = np.ones((10, 10))
        # MSE = mean(1^2) = 1.0
        np.testing.assert_allclose(mse(C, ST, D), 1.0)

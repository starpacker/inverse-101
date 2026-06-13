"""Unit tests for the linear spectral mixing forward model."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.physics_model import forward, residual, reconstruction_error

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures")


class TestForward:
    def setup_method(self):
        f = np.load(f"{FIXTURE_DIR}/physics_model.npz")
        self.endmembers = f["endmembers"]
        self.abundances = f["abundances"]
        self.expected = f["output_forward"]

    def test_shape(self):
        result = forward(self.endmembers, self.abundances)
        assert result.shape == (20, 50)

    def test_values(self):
        result = forward(self.endmembers, self.abundances)
        np.testing.assert_allclose(result, self.expected, rtol=1e-10)


class TestResidual:
    def setup_method(self):
        f = np.load(f"{FIXTURE_DIR}/physics_model.npz")
        self.endmembers = f["endmembers"]
        self.abundances = f["abundances"]
        self.observed = f["input_observed"]

    def test_shape(self):
        r = residual(self.observed, self.endmembers, self.abundances)
        assert r.shape == self.observed.shape

    def test_zero_for_perfect_prediction(self):
        predicted = forward(self.endmembers, self.abundances)
        r = residual(predicted, self.endmembers, self.abundances)
        np.testing.assert_allclose(r, 0, atol=1e-12)


class TestReconstructionError:
    def setup_method(self):
        f = np.load(f"{FIXTURE_DIR}/physics_model.npz")
        self.endmembers = f["endmembers"]
        self.abundances = f["abundances"]
        self.observed = f["input_observed"]
        self.expected_rmse = float(f["output_rmse"])

    def test_value(self):
        rmse = reconstruction_error(self.observed, self.endmembers,
                                    self.abundances)
        np.testing.assert_allclose(rmse, self.expected_rmse, rtol=1e-10)

    def test_zero_for_perfect(self):
        predicted = forward(self.endmembers, self.abundances)
        rmse = reconstruction_error(predicted, self.endmembers,
                                    self.abundances)
        assert rmse < 1e-12

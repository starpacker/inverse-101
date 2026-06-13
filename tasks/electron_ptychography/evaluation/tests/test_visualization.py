"""Tests for visualization module."""

import os
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.visualization import compute_metrics


class TestComputeMetrics:
    def test_self_comparison(self):
        x = np.random.randn(64, 64)
        m = compute_metrics(x, x)
        np.testing.assert_allclose(m["ncc"], 1.0, atol=1e-10)
        np.testing.assert_allclose(m["nrmse"], 0.0, atol=1e-10)

    def test_scaled_comparison(self):
        x = np.random.randn(64, 64)
        m = compute_metrics(2 * x, x)
        # NCC should be 1.0 for positive scaling
        np.testing.assert_allclose(m["ncc"], 1.0, atol=1e-10)
        # NRMSE should be > 0 for different amplitudes
        assert m["nrmse"] > 0

    def test_orthogonal(self):
        x = np.zeros((4, 4))
        x[0, 0] = 1.0
        y = np.zeros((4, 4))
        y[1, 1] = 1.0
        m = compute_metrics(x, y)
        np.testing.assert_allclose(m["ncc"], 0.0, atol=1e-10)

    def test_returns_dict(self):
        x = np.ones((10, 10))
        m = compute_metrics(x, x)
        assert "ncc" in m
        assert "nrmse" in m

"""Tests for visualization module."""

import pathlib
import sys

import numpy as np
import pytest

TASK_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

from src.visualization import compute_metrics


class TestComputeMetrics:
    def test_perfect_match(self):
        x = np.random.rand(100)
        m = compute_metrics(x, x)
        np.testing.assert_allclose(m["nrmse"], 0.0, atol=1e-12)
        np.testing.assert_allclose(m["ncc"], 1.0, atol=1e-12)

    def test_dict_keys(self):
        x = np.random.rand(10)
        y = np.random.rand(10)
        m = compute_metrics(x, y)
        assert "nrmse" in m
        assert "ncc" in m

    def test_nrmse_positive(self):
        x = np.random.rand(50)
        y = np.random.rand(50)
        m = compute_metrics(x, y)
        assert m["nrmse"] >= 0

    def test_ncc_range(self):
        x = np.random.rand(50) + 0.1
        y = np.random.rand(50) + 0.1
        m = compute_metrics(x, y)
        assert -1.0 <= m["ncc"] <= 1.0

    def test_scaled_copy(self):
        x = np.random.rand(100) + 1.0
        m = compute_metrics(2 * x, x)
        # NCC should be 1.0 for scaled version
        np.testing.assert_allclose(m["ncc"], 1.0, atol=1e-12)

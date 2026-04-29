"""Tests for src/visualization.py."""

import numpy as np
import pytest
import sys, os

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

from src.visualization import compute_ncc, compute_nrmse


class TestNCC:
    def test_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        assert abs(compute_ncc(x, x) - 1.0) < 1e-10

    def test_scaled(self):
        """NCC is 1 for scaled versions (cosine similarity)."""
        x = np.array([1.0, 2.0, 3.0])
        assert abs(compute_ncc(x, 2 * x) - 1.0) < 1e-10

    def test_orthogonal(self):
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        assert abs(compute_ncc(x, y)) < 1e-10

    def test_with_mask(self):
        x = np.array([1.0, 2.0, 999.0])
        y = np.array([1.0, 2.0, -999.0])
        mask = np.array([True, True, False])
        assert abs(compute_ncc(x, y, mask=mask) - 1.0) < 1e-10

    def test_zero_signal(self):
        x = np.zeros(5)
        assert compute_ncc(x, np.ones(5)) == 0.0


class TestNRMSE:
    def test_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        assert compute_nrmse(x, x) == 0.0

    def test_known_value(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 3.0])
        # RMSE = sqrt(mean([1, 1])) = 1.0
        # dynamic_range = 3 - 2 = 1
        # NRMSE = 1.0 / 1.0 = 1.0
        assert abs(compute_nrmse(x, y) - 1.0) < 1e-10

    def test_with_mask(self):
        x = np.array([1.0, 2.0, 999.0])
        y = np.array([1.0, 2.0, 0.0])
        mask = np.array([True, True, False])
        assert compute_nrmse(x, y, mask=mask) == 0.0

    def test_constant_reference(self):
        """Constant reference has zero dynamic range -> inf."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 1.0])
        assert compute_nrmse(x, y) == float('inf')

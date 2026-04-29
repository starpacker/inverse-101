"""Tests for src/visualization.py."""
import numpy as np
import sys, os

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

from src.visualization import compute_ncc, compute_nrmse


class TestNCC:
    def test_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        assert abs(compute_ncc(x, x) - 1.0) < 1e-10

    def test_with_mask(self):
        x = np.array([1.0, 2.0, 999.0])
        y = np.array([1.0, 2.0, -999.0])
        mask = np.array([True, True, False])
        assert abs(compute_ncc(x, y, mask=mask) - 1.0) < 1e-10


class TestNRMSE:
    def test_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        assert compute_nrmse(x, x) == 0.0

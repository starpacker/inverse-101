"""Unit tests for visualisation metrics."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.visualization import compute_ncc, compute_nrmse, compute_metrics

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures")


class TestNCC:
    def setup_method(self):
        f = np.load(f"{FIXTURE_DIR}/visualization_metrics.npz")
        self.est = f["input_estimate"]
        self.ref = f["input_reference"]
        self.expected = float(f["output_ncc"])

    def test_value(self):
        ncc = compute_ncc(self.est, self.ref)
        np.testing.assert_allclose(ncc, self.expected, rtol=1e-10)

    def test_self_correlation(self):
        ncc = compute_ncc(self.ref, self.ref)
        np.testing.assert_allclose(ncc, 1.0, atol=1e-10)

    def test_range(self):
        ncc = compute_ncc(self.est, self.ref)
        assert -1.0 <= ncc <= 1.0


class TestNRMSE:
    def setup_method(self):
        f = np.load(f"{FIXTURE_DIR}/visualization_metrics.npz")
        self.est = f["input_estimate"]
        self.ref = f["input_reference"]
        self.expected = float(f["output_nrmse"])

    def test_value(self):
        nrmse = compute_nrmse(self.est, self.ref)
        np.testing.assert_allclose(nrmse, self.expected, rtol=1e-10)

    def test_zero_for_identical(self):
        nrmse = compute_nrmse(self.ref, self.ref)
        assert nrmse < 1e-12

    def test_nonnegative(self):
        nrmse = compute_nrmse(self.est, self.ref)
        assert nrmse >= 0


class TestComputeMetrics:
    def test_keys(self):
        m = compute_metrics(np.ones((5, 5)), np.ones((5, 5)))
        assert "ncc" in m
        assert "nrmse" in m

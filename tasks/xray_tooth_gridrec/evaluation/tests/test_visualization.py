"""Tests for visualization module."""

import os
import numpy as np
import pytest
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "visualization")

from src.visualization import compute_metrics


class TestComputeMetrics:
    @pytest.fixture
    def fixture(self):
        return np.load(os.path.join(FIXTURE_DIR, "metrics.npz"))

    def test_nrmse(self, fixture):
        metrics = compute_metrics(fixture["input_estimate"], fixture["input_reference"])
        np.testing.assert_allclose(
            metrics["nrmse"], float(fixture["output_nrmse"]), rtol=1e-10
        )

    def test_ncc(self, fixture):
        metrics = compute_metrics(fixture["input_estimate"], fixture["input_reference"])
        np.testing.assert_allclose(
            metrics["ncc"], float(fixture["output_ncc"]), rtol=1e-10
        )

    def test_perfect_match(self):
        x = np.random.RandomState(0).randn(10, 10)
        metrics = compute_metrics(x, x)
        assert metrics["nrmse"] == pytest.approx(0.0, abs=1e-14)
        assert metrics["ncc"] == pytest.approx(1.0, abs=1e-14)

    def test_ncc_range(self):
        est = np.random.RandomState(1).randn(8, 8)
        ref = np.random.RandomState(2).randn(8, 8)
        metrics = compute_metrics(est, ref)
        assert -1.0 <= metrics["ncc"] <= 1.0
        assert metrics["nrmse"] >= 0.0

import os
import pytest
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

import sys
sys.path.insert(0, TASK_DIR)
from src.visualization import compute_snr, compute_metrics


class TestComputeSNR:
    def test_identical_images(self):
        x = np.random.randn(64, 64)
        snr = compute_snr(x, x)
        assert snr == np.inf or snr > 100

    def test_positive_snr(self):
        np.random.seed(0)
        x = np.random.randn(64, 64)
        x_noisy = x + 0.1 * np.random.randn(64, 64)
        snr = compute_snr(x, x_noisy)
        assert snr > 0

    def test_symmetry(self):
        """SNR is not symmetric but should be well-defined both ways."""
        np.random.seed(0)
        x = np.ones((64, 64))
        y = x + 0.1 * np.random.randn(64, 64)
        snr = compute_snr(x, y)
        assert np.isfinite(snr)


class TestComputeMetrics:
    def test_returns_required_keys(self):
        np.random.seed(0)
        x = np.random.randn(64, 64)
        y = x + 0.1 * np.random.randn(64, 64)
        m = compute_metrics(x, y)
        assert "snr_db" in m
        assert "nrmse" in m
        assert "ncc" in m

    def test_perfect_reconstruction(self):
        x = np.random.randn(64, 64)
        m = compute_metrics(x, x)
        np.testing.assert_allclose(m["nrmse"], 0.0, atol=1e-10)
        np.testing.assert_allclose(m["ncc"], 1.0, atol=1e-10)

    def test_nrmse_range(self):
        np.random.seed(0)
        x = np.random.randn(64, 64)
        y = x + 0.5 * np.random.randn(64, 64)
        m = compute_metrics(x, y)
        assert 0 < m["nrmse"] < 2
        assert 0 < m["ncc"] <= 1

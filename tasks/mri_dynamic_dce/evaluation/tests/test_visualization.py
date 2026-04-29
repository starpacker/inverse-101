"""Tests for src/visualization.py"""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.visualization import compute_nrmse, compute_ncc, compute_psnr, compute_frame_metrics


class TestNRMSE:
    def test_identical(self):
        x = np.random.randn(10, 10)
        assert compute_nrmse(x, x) == 0.0

    def test_positive(self):
        x = np.random.randn(10, 10)
        y = np.random.randn(10, 10)
        assert compute_nrmse(x, y) > 0.0

    def test_known_value(self):
        ref = np.array([0.0, 1.0, 0.0, 1.0])
        est = np.array([0.0, 0.0, 0.0, 1.0])
        # RMSE = sqrt(1/4 * 1) = 0.5, range = 1.0, NRMSE = 0.5
        np.testing.assert_allclose(compute_nrmse(est, ref), 0.5, rtol=1e-10)


class TestNCC:
    def test_identical(self):
        x = np.random.randn(10, 10)
        np.testing.assert_allclose(compute_ncc(x, x), 1.0, rtol=1e-10)

    def test_orthogonal(self):
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        np.testing.assert_allclose(compute_ncc(x, y), 0.0, atol=1e-10)

    def test_range(self):
        x = np.random.randn(20)
        y = np.random.randn(20)
        ncc = compute_ncc(x, y)
        assert -1.0 <= ncc <= 1.0


class TestPSNR:
    def test_identical(self):
        x = np.random.rand(10, 10)
        assert compute_psnr(x, x) >= 90.0  # should be very high

    def test_known_value(self):
        ref = np.ones((2, 2))  # peak = 1
        est = np.ones((2, 2)) * 0.9  # MSE = 0.01
        expected_psnr = 10 * np.log10(1.0 / 0.01)  # 20 dB
        np.testing.assert_allclose(compute_psnr(est, ref), expected_psnr, rtol=1e-6)


class TestFrameMetrics:
    def test_keys(self):
        T, N = 5, 8
        recon = np.random.rand(T, N, N)
        ref = np.random.rand(T, N, N)
        m = compute_frame_metrics(recon, ref)
        assert 'per_frame' in m
        assert 'avg_nrmse' in m
        assert 'avg_ncc' in m
        assert 'avg_psnr' in m
        assert 'overall_nrmse' in m
        assert 'overall_ncc' in m

    def test_per_frame_length(self):
        T, N = 5, 8
        recon = np.random.rand(T, N, N)
        ref = np.random.rand(T, N, N)
        m = compute_frame_metrics(recon, ref)
        assert len(m['per_frame']) == T

    def test_perfect_reconstruction(self):
        T, N = 3, 8
        ref = np.random.rand(T, N, N)
        m = compute_frame_metrics(ref, ref)
        np.testing.assert_allclose(m['avg_nrmse'], 0.0, atol=1e-10)
        np.testing.assert_allclose(m['avg_ncc'], 1.0, rtol=1e-10)
        np.testing.assert_allclose(m['overall_nrmse'], 0.0, atol=1e-10)
        np.testing.assert_allclose(m['overall_ncc'], 1.0, rtol=1e-10)

"""Tests for visualization module."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'visualization')

import sys
sys.path.insert(0, TASK_DIR)

from src.visualization import compute_metrics, compute_video_metrics


class TestComputeMetrics:
    """Tests for image quality metrics."""

    def test_compute_metrics_parity(self):
        """Test metrics match fixture."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'compute_metrics.npz'))
        m = compute_metrics(fix['input_est'], fix['input_ref'])
        np.testing.assert_allclose(m['nrmse'], float(fix['output_nrmse']), rtol=1e-10)
        np.testing.assert_allclose(m['ncc'], float(fix['output_ncc']), rtol=1e-10)

    def test_identical_images(self):
        """Perfect reconstruction should give NRMSE=0, NCC=1."""
        img = np.random.default_rng(42).standard_normal((8, 8))
        m = compute_metrics(img, img)
        np.testing.assert_allclose(m['nrmse'], 0.0, atol=1e-15)
        np.testing.assert_allclose(m['ncc'], 1.0, atol=1e-15)

    def test_ncc_range(self):
        """NCC should be in [-1, 1]."""
        rng = np.random.default_rng(42)
        est = rng.standard_normal((8, 8))
        ref = rng.standard_normal((8, 8))
        m = compute_metrics(est, ref)
        assert -1.0 <= m['ncc'] <= 1.0

    def test_nrmse_nonnegative(self):
        """NRMSE should be non-negative."""
        rng = np.random.default_rng(42)
        est = rng.standard_normal((8, 8))
        ref = rng.standard_normal((8, 8))
        m = compute_metrics(est, ref)
        assert m['nrmse'] >= 0.0


class TestComputeVideoMetrics:
    """Tests for video-level metrics."""

    def test_video_metrics_structure(self):
        """Test output structure of compute_video_metrics."""
        rng = np.random.default_rng(42)
        est = [rng.standard_normal((8, 8)) for _ in range(3)]
        ref = [rng.standard_normal((8, 8)) for _ in range(3)]
        vm = compute_video_metrics(est, ref)

        assert 'per_frame' in vm
        assert 'average' in vm
        assert len(vm['per_frame']) == 3
        assert 'nrmse' in vm['average']
        assert 'ncc' in vm['average']

    def test_video_metrics_average(self):
        """Average should be mean of per-frame values."""
        rng = np.random.default_rng(42)
        est = [rng.standard_normal((8, 8)) for _ in range(3)]
        ref = [rng.standard_normal((8, 8)) for _ in range(3)]
        vm = compute_video_metrics(est, ref)

        avg_nrmse = np.mean([m['nrmse'] for m in vm['per_frame']])
        avg_ncc = np.mean([m['ncc'] for m in vm['per_frame']])
        np.testing.assert_allclose(vm['average']['nrmse'], avg_nrmse, rtol=1e-10)
        np.testing.assert_allclose(vm['average']['ncc'], avg_ncc, rtol=1e-10)

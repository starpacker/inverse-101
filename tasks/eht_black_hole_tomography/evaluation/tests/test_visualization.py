"""
Tests for src/visualization.py
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.visualization import (
    compute_metrics, compute_image_metrics,
    plot_emission_slices, plot_movie_comparison,
    plot_lightcurve, plot_loss_curves,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'visualization')


class TestComputeMetrics(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'compute_metrics.npz'))

    def test_output_nrmse(self):
        metrics = compute_metrics(self.f['input_estimate'],
                                  self.f['input_ground_truth'])
        np.testing.assert_allclose(metrics['nrmse'],
                                   float(self.f['output_nrmse']),
                                   rtol=1e-10)

    def test_output_ncc(self):
        metrics = compute_metrics(self.f['input_estimate'],
                                  self.f['input_ground_truth'])
        np.testing.assert_allclose(metrics['ncc'],
                                   float(self.f['output_ncc']),
                                   rtol=1e-10)

    def test_output_psnr(self):
        metrics = compute_metrics(self.f['input_estimate'],
                                  self.f['input_ground_truth'])
        np.testing.assert_allclose(metrics['psnr'],
                                   float(self.f['output_psnr']),
                                   rtol=1e-10)

    def test_perfect_match(self):
        x = np.random.rand(10, 10)
        metrics = compute_metrics(x, x)
        self.assertAlmostEqual(metrics['nrmse'], 0.0, places=10)
        self.assertAlmostEqual(metrics['ncc'], 1.0, places=10)


class TestComputeImageMetrics(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'compute_image_metrics.npz'))

    def test_output_nrmse_image(self):
        metrics = compute_image_metrics(self.f['input_pred_movie'],
                                        self.f['input_true_movie'])
        np.testing.assert_allclose(metrics['nrmse_image'],
                                   float(self.f['output_nrmse_image']),
                                   rtol=1e-10)

    def test_output_ncc_image(self):
        metrics = compute_image_metrics(self.f['input_pred_movie'],
                                        self.f['input_true_movie'])
        np.testing.assert_allclose(metrics['ncc_image'],
                                   float(self.f['output_ncc_image']),
                                   rtol=1e-10)

    def test_output_lightcurve_mse(self):
        metrics = compute_image_metrics(self.f['input_pred_movie'],
                                        self.f['input_true_movie'])
        np.testing.assert_allclose(metrics['lightcurve_mse'],
                                   float(self.f['output_lightcurve_mse']),
                                   rtol=1e-10)


class TestPlotFunctions(unittest.TestCase):
    """Smoke tests: ensure plot functions run without error."""

    def test_plot_emission_slices(self):
        emission = np.random.rand(8, 8, 8).astype(np.float32)
        plot_emission_slices(emission, fov_M=24.0)

    def test_plot_emission_slices_with_gt(self):
        emission = np.random.rand(8, 8, 8).astype(np.float32)
        gt = np.random.rand(8, 8, 8).astype(np.float32)
        plot_emission_slices(emission, fov_M=24.0, ground_truth=gt)

    def test_plot_movie_comparison(self):
        pred = np.random.rand(5, 8, 8).astype(np.float32)
        true = np.random.rand(5, 8, 8).astype(np.float32)
        t = np.linspace(0, 100, 5)
        plot_movie_comparison(pred, true, t, n_show=3)

    def test_plot_lightcurve(self):
        pred = np.random.rand(5, 8, 8).astype(np.float32)
        true = np.random.rand(5, 8, 8).astype(np.float32)
        t = np.linspace(0, 100, 5)
        plot_lightcurve(pred, true, t)

    def test_plot_loss_curves(self):
        loss = np.logspace(0, -3, 100)
        plot_loss_curves(loss)


if __name__ == '__main__':
    unittest.main()

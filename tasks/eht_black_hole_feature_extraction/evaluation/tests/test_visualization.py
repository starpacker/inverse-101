"""Tests for visualization module."""
import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'visualization')
sys.path.insert(0, TASK_DIR)

import matplotlib
matplotlib.use('Agg')

from src.visualization import (
    compute_feature_metrics, plot_corner, plot_elbo_comparison,
    plot_posterior_images, plot_loss_curves,
)


class TestComputeFeatureMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(os.path.join(FIX_DIR, 'feature_metrics.npz'))

    def test_returns_dict(self):
        metrics = compute_feature_metrics(
            self.data['input_params'], self.data['input_gt'],
            self.data['input_weights'],
            ['diameter', 'width', 'asymmetry', 'PA']
        )
        self.assertIn('bias', metrics)
        self.assertIn('std', metrics)
        self.assertIn('coverage', metrics)
        self.assertIn('n_params', metrics)

    def test_n_params(self):
        metrics = compute_feature_metrics(
            self.data['input_params'], self.data['input_gt'],
            self.data['input_weights'],
            ['diameter', 'width', 'asymmetry', 'PA']
        )
        self.assertEqual(metrics['n_params'], int(self.data['output_n_params']))

    def test_coverage_bounded(self):
        metrics = compute_feature_metrics(
            self.data['input_params'], self.data['input_gt'],
            self.data['input_weights'],
            ['diameter', 'width', 'asymmetry', 'PA']
        )
        for name, cov in metrics['coverage'].items():
            self.assertGreaterEqual(cov, 0.0)
            self.assertLessEqual(cov, 1.0)

    def test_std_positive(self):
        metrics = compute_feature_metrics(
            self.data['input_params'], self.data['input_gt'],
            self.data['input_weights'],
            ['diameter', 'width', 'asymmetry', 'PA']
        )
        for name, s in metrics['std'].items():
            self.assertGreater(s, 0.0)

    def test_without_weights(self):
        metrics = compute_feature_metrics(
            self.data['input_params'], self.data['input_gt'],
            None, ['diameter', 'width', 'asymmetry', 'PA']
        )
        self.assertIn('bias', metrics)
        self.assertIn('std', metrics)


class TestPlotCorner(unittest.TestCase):
    def test_smoke(self):
        np.random.seed(0)
        params = np.random.randn(100, 4)
        gt = np.array([0, 0, 0, 0])
        fig = plot_corner(params, ['a', 'b', 'c', 'd'], ground_truth=gt)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotElboComparison(unittest.TestCase):
    def test_smoke_dict(self):
        elbos = {'0 Gauss': -100, '1 Gauss': -80, '2 Gauss': -70}
        fig = plot_elbo_comparison(elbos)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_smoke_list(self):
        fig = plot_elbo_comparison([-100, -80, -70],
                                    model_names=['A', 'B', 'C'])
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotPosteriorImages(unittest.TestCase):
    def test_smoke(self):
        images = np.random.rand(10, 64, 64)
        fig = plot_posterior_images(images, n_show=4)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotLossCurves(unittest.TestCase):
    def test_smoke(self):
        loss_history = {
            'total': np.random.randn(100),
            'cphase': np.random.rand(100),
            'logca': np.random.rand(100),
            'logdet': np.random.randn(100),
        }
        fig = plot_loss_curves(loss_history)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()

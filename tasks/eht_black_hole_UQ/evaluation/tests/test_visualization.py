"""
Unit tests for visualization.py
"""

import os
import unittest
import numpy as np

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "visualization")


class TestComputeMetrics(unittest.TestCase):
    """Test reconstruction quality metrics."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.visualization import compute_metrics
        self.compute_metrics = compute_metrics
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "compute_metrics.npz"),
                               allow_pickle=False)

    def test_nrmse(self):
        metrics = self.compute_metrics(self.fixture['input_estimate'],
                                        self.fixture['input_ground_truth'])
        np.testing.assert_allclose(
            metrics['nrmse'], float(self.fixture['output_nrmse']), rtol=1e-5)

    def test_ncc(self):
        metrics = self.compute_metrics(self.fixture['input_estimate'],
                                        self.fixture['input_ground_truth'])
        np.testing.assert_allclose(
            metrics['ncc'], float(self.fixture['output_ncc']), rtol=1e-5)

    def test_dynamic_range(self):
        metrics = self.compute_metrics(self.fixture['input_estimate'],
                                        self.fixture['input_ground_truth'])
        np.testing.assert_allclose(
            metrics['dynamic_range'], float(self.fixture['output_dynamic_range']),
            rtol=1e-4)


class TestComputeUqMetrics(unittest.TestCase):
    """Test uncertainty quantification metrics."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.visualization import compute_uq_metrics
        self.compute_uq_metrics = compute_uq_metrics
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "compute_uq_metrics.npz"),
                               allow_pickle=False)

    def test_calibration(self):
        metrics = self.compute_uq_metrics(
            self.fixture['input_mean'],
            self.fixture['input_std'],
            self.fixture['input_gt'])
        np.testing.assert_allclose(
            metrics['calibration'], float(self.fixture['output_calibration']),
            rtol=1e-5)

    def test_mean_uncertainty(self):
        metrics = self.compute_uq_metrics(
            self.fixture['input_mean'],
            self.fixture['input_std'],
            self.fixture['input_gt'])
        np.testing.assert_allclose(
            metrics['mean_uncertainty'], float(self.fixture['output_mean_uncertainty']),
            rtol=1e-5)


class TestPrintMetricsTable(unittest.TestCase):
    """Smoke test for print_metrics_table."""

    def test_runs_without_error(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.visualization import print_metrics_table
        import io
        from contextlib import redirect_stdout
        metrics = {'nrmse': 0.123, 'ncc': 0.987, 'dynamic_range': 25.0}
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_metrics_table(metrics)
        output = buf.getvalue()
        self.assertIn('nrmse', output)
        self.assertIn('0.1230', output)


class TestGenerateCrescentImage(unittest.TestCase):
    """Test synthetic crescent image generation."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.generate_data import generate_crescent_image
        self.generate = generate_crescent_image

    def test_default_shape(self):
        img = self.generate()
        self.assertEqual(img.shape, (32, 32))

    def test_custom_shape(self):
        img = self.generate(npix=64)
        self.assertEqual(img.shape, (64, 64))

    def test_positive(self):
        img = self.generate()
        self.assertTrue(np.all(img >= 0))

    def test_flux_normalization(self):
        img = self.generate(total_flux=2.5)
        np.testing.assert_allclose(img.sum(), 2.5, rtol=1e-6)

    def test_ring_structure(self):
        """Center should be dimmer than the ring region."""
        img = self.generate(npix=64, radius_uas=20.0)
        center = img[31:33, 31:33].mean()
        ring_region = img[20:25, 31:33].mean()
        self.assertGreater(ring_region, center)


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

"""Unit tests for visualization module."""
import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.visualization import compute_metrics

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "visualization")


class TestComputeMetrics(unittest.TestCase):
    def test_basic_metrics(self):
        gt = np.array([1.0, 2.0, 3.0, 4.0])
        recon = np.array([1.1, 1.9, 3.1, 3.9])
        m = compute_metrics(recon, gt)
        self.assertIn("nrmse", m)
        self.assertIn("ncc", m)
        self.assertGreater(m["ncc"], 0.9)

    def test_perfect_match(self):
        gt = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(gt, gt)
        self.assertAlmostEqual(m["nrmse"], 0.0, places=5)
        self.assertAlmostEqual(m["ncc"], 1.0, places=5)

    def test_fixture_bp_metrics(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "output_metrics_bp.npz"))
        # Just verify the fixture values are reasonable
        self.assertGreater(float(fix["ncc"]), 0.0)
        self.assertGreater(float(fix["nrmse"]), 0.0)


if __name__ == "__main__":
    unittest.main()

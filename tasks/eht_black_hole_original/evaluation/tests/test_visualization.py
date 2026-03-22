"""Unit tests for visualization module (metrics computation)."""
import os
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "visualization")


class TestComputeMetrics(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "compute_metrics.npz"))
        from src.visualization import compute_metrics
        self.result = compute_metrics(
            self.f["input_estimate"], self.f["input_ground_truth"],
        )

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_required_keys(self):
        for key in ["nrmse", "ncc", "dynamic_range"]:
            self.assertIn(key, self.result)

    def test_types(self):
        for key in ["nrmse", "ncc", "dynamic_range"]:
            self.assertIsInstance(self.result[key], float)

    def test_nrmse_value(self):
        assert_allclose(self.result["nrmse"],
                        float(self.f["output_nrmse"]), rtol=1e-10)

    def test_ncc_value(self):
        assert_allclose(self.result["ncc"],
                        float(self.f["output_ncc"]), rtol=1e-10)

    def test_dynamic_range_value(self):
        assert_allclose(self.result["dynamic_range"],
                        float(self.f["output_dynamic_range"]), rtol=1e-10)

    def test_perfect_reconstruction(self):
        """Perfect reconstruction should give NRMSE~0, NCC~1."""
        from src.visualization import compute_metrics
        gt = np.random.default_rng(0).uniform(0, 1, (8, 8))
        gt /= gt.sum()
        m = compute_metrics(gt, gt)
        self.assertAlmostEqual(m["nrmse"], 0.0, places=10)
        self.assertAlmostEqual(m["ncc"], 1.0, places=10)


if __name__ == "__main__":
    unittest.main()

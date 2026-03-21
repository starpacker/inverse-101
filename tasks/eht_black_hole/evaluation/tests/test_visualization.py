"""
Tests for visualization.py — Decoupled per-function tests
===========================================================

Each test loads its own fixture from evaluation/fixtures/visualization/
and tests exactly one function.

Tested functions:
  - compute_metrics(estimate, ground_truth) → dict
"""

import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "visualization")


# ═══════════════════════════════════════════════════════════════════════════
# compute_metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeMetrics(unittest.TestCase):
    """
    Fixture: compute_metrics.npz
      input_estimate       : (N, N) reconstructed image
      input_ground_truth   : (N, N) reference image
      output_nrmse         : scalar
      output_ncc           : scalar
      output_dynamic_range : scalar
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "compute_metrics.npz"))
        from src.visualization import compute_metrics
        self.result = compute_metrics(self.f["input_estimate"], self.f["input_ground_truth"])

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_required_keys(self):
        for key in ["nrmse", "ncc", "dynamic_range"]:
            self.assertIn(key, self.result)

    def test_nrmse_value(self):
        np.testing.assert_allclose(
            self.result["nrmse"], float(self.f["output_nrmse"]), rtol=1e-10
        )

    def test_ncc_value(self):
        np.testing.assert_allclose(
            self.result["ncc"], float(self.f["output_ncc"]), rtol=1e-10
        )

    def test_dynamic_range_value(self):
        np.testing.assert_allclose(
            self.result["dynamic_range"], float(self.f["output_dynamic_range"]), rtol=1e-10
        )


if __name__ == "__main__":
    unittest.main()

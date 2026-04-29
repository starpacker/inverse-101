"""Unit tests for src/visualization.py (metrics-only; plotting is smoke-tested)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

from src.visualization import (
    compute_metrics_per_channel,
    compute_weighted_nrmse_per_channel,
    metrics_to_jsonable,
    plot_all_channels,
)


class TestPerfectMatch(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.x = rng.rand(3, 5, 8, 4).astype(np.float32)
        self.weight = rng.rand(5, 8, 4).astype(np.float32)

    def test_zero_weighted_nrmse(self):
        per_ch = compute_weighted_nrmse_per_channel(self.x, self.x, self.weight)
        np.testing.assert_allclose(per_ch, np.zeros(5), atol=1e-12)

    def test_metrics_perfect(self):
        m = compute_metrics_per_channel(self.x, self.x, self.weight)
        np.testing.assert_allclose(m["ncc"], np.ones(5), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(m["nrmse"], np.zeros(5), atol=1e-6)
        np.testing.assert_allclose(m["weighted_nrmse"], np.zeros(5), atol=1e-6)


class TestMetricsShape(unittest.TestCase):
    def test_shapes(self):
        rng = np.random.RandomState(1)
        est = rng.rand(2, 5, 6, 4).astype(np.float32)
        ref = rng.rand(2, 5, 6, 4).astype(np.float32)
        weight = np.ones((5, 6, 4), dtype=np.float32)
        m = compute_metrics_per_channel(est, ref, weight)
        self.assertEqual(m["ncc"].shape, (5,))
        self.assertEqual(m["nrmse"].shape, (5,))
        self.assertEqual(m["weighted_nrmse"].shape, (5,))
        self.assertIsInstance(m["ncc_mean"], float)


class TestMetricsToJsonable(unittest.TestCase):
    def test_round_trip(self):
        m = {
            "ncc": np.array([0.123456789, 0.987654321]),
            "ncc_mean": 0.5555555555,
            "channels": ["a", "b"],
        }
        out = metrics_to_jsonable(m)
        self.assertEqual(out["ncc"], [0.123457, 0.987654])
        self.assertEqual(out["ncc_mean"], 0.555556)
        self.assertEqual(out["channels"], ["a", "b"])


class TestPlotSmoke(unittest.TestCase):
    def test_plot_runs(self):
        import matplotlib

        matplotlib.use("Agg")
        rng = np.random.RandomState(2)
        est = rng.rand(2, 5, 6, 4).astype(np.float32)
        ref = rng.rand(2, 5, 6, 4).astype(np.float32)
        weight = np.ones((5, 6, 4), dtype=np.float32)
        m = compute_metrics_per_channel(est, ref, weight)
        fig = plot_all_channels(est, ref, m)
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()

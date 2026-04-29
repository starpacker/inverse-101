"""Unit tests for visualization module."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from src.visualization import compute_metrics


def test_compute_metrics():
    """Metrics computation on identical inputs."""
    a = np.random.randn(32, 32).astype(np.float32)
    metrics = compute_metrics(a, a)
    assert metrics["mean_abs_diff_rad"] == 0.0
    assert metrics["rmse_rad"] == 0.0
    assert metrics["fraction_within_pi"] == 1.0
    assert metrics["fraction_within_2pi"] == 1.0


def test_compute_metrics_with_offset():
    """Metrics should handle constant offset (mean-removed comparison)."""
    a = np.ones((16, 16), dtype=np.float32)
    b = a + 100.0  # large constant offset
    metrics = compute_metrics(a, b)
    # After mean removal, they're identical
    assert metrics["rmse_rad"] < 1e-5
    assert metrics["fraction_within_pi"] == 1.0

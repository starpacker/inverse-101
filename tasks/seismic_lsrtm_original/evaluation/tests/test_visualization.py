"""Unit tests for src/visualization.py."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from src.visualization import (
    plot_scatter_image,
    plot_velocity_models,
    plot_scattered_data,
    plot_data_comparison,
    plot_loss_curve,
    compute_data_metrics,
)


class TestComputeDataMetrics:
    def test_perfect_match(self):
        data = np.random.randn(3, 10, 50).astype(np.float32)
        m = compute_data_metrics(data, data, shot_idx=0)
        assert m["mse"] == pytest.approx(0.0, abs=1e-10)
        assert m["rel_l2"] == pytest.approx(0.0, abs=1e-10)

    def test_keys(self):
        data = np.random.randn(2, 5, 20).astype(np.float32)
        m = compute_data_metrics(data, data + 0.1, shot_idx=0)
        assert set(m.keys()) == {"mse", "rms", "rel_l2"}

    def test_nonzero_error(self):
        data = np.random.randn(2, 5, 20).astype(np.float32)
        m = compute_data_metrics(data, data + 1.0, shot_idx=0)
        assert m["mse"] > 0


class TestPlotFunctions:
    def test_plot_scatter_image(self):
        fig = plot_scatter_image(np.random.randn(50, 30).astype(np.float32), 4.0)
        assert fig is not None
        import matplotlib.pyplot as plt; plt.close(fig)

    def test_plot_velocity_models(self):
        v = np.ones((50, 30), dtype=np.float32) * 2000
        fig = plot_velocity_models(v, v * 0.9, 4.0)
        assert fig is not None
        import matplotlib.pyplot as plt; plt.close(fig)

    def test_plot_scattered_data(self):
        d = np.random.randn(3, 10, 50).astype(np.float32)
        fig = plot_scattered_data(d, d * 0.5, d * 0.5, shot_idx=0)
        assert fig is not None
        import matplotlib.pyplot as plt; plt.close(fig)

    def test_plot_data_comparison(self):
        d = np.random.randn(2, 10, 50).astype(np.float32)
        fig = plot_data_comparison(d, d + 0.1, shot_idx=0)
        assert fig is not None
        import matplotlib.pyplot as plt; plt.close(fig)

    def test_plot_loss_curve(self):
        fig = plot_loss_curve([1.0, 0.5, 0.3])
        assert fig is not None
        import matplotlib.pyplot as plt; plt.close(fig)

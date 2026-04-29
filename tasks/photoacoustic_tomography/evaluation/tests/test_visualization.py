"""Tests for src/visualization.py."""

import numpy as np
import os
import sys
import pytest
import matplotlib
matplotlib.use("Agg")

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


class TestComputeNCC:
    """Tests for NCC computation."""

    def test_fixture_value(self):
        from src.visualization import compute_ncc
        fix = np.load(os.path.join(FIXTURES_DIR, "visualization_metrics.npz"))
        ncc = compute_ncc(fix["input_a"], fix["input_b"])
        np.testing.assert_allclose(ncc, float(fix["output_ncc"]), rtol=1e-10)

    def test_identical(self):
        from src.visualization import compute_ncc
        a = np.array([1.0, 2.0, 3.0])
        assert abs(compute_ncc(a, a) - 1.0) < 1e-10

    def test_range(self):
        from src.visualization import compute_ncc
        a = np.random.rand(100)
        b = np.random.rand(100)
        ncc = compute_ncc(a, b)
        assert -1.0 <= ncc <= 1.0


class TestComputeNRMSE:
    """Tests for NRMSE computation."""

    def test_fixture_value(self):
        from src.visualization import compute_nrmse
        fix = np.load(os.path.join(FIXTURES_DIR, "visualization_metrics.npz"))
        nrmse = compute_nrmse(fix["input_a"], fix["input_b"])
        np.testing.assert_allclose(nrmse, float(fix["output_nrmse"]), rtol=1e-10)

    def test_identical(self):
        from src.visualization import compute_nrmse
        a = np.array([1.0, 2.0, 3.0])
        assert compute_nrmse(a, a) == 0.0

    def test_nonnegative(self):
        from src.visualization import compute_nrmse
        a = np.random.rand(100)
        b = np.random.rand(100)
        assert compute_nrmse(a, b) >= 0.0


class TestCentreCrop:
    """Tests for centre crop utility."""

    def test_output_size(self):
        from src.visualization import centre_crop
        img = np.random.rand(100, 100)
        cropped = centre_crop(img, fraction=0.5)
        assert cropped.shape[0] == 50
        assert cropped.shape[1] == 50

    def test_3d_squeeze(self):
        from src.visualization import centre_crop
        img = np.random.rand(100, 100, 1)
        cropped = centre_crop(img, fraction=0.8)
        assert cropped.ndim == 2


class TestPlotFunctions:
    """Smoke tests for plotting functions (verify they don't crash)."""

    def test_plot_reconstruction(self):
        import matplotlib.pyplot as plt
        from src.visualization import plot_reconstruction
        recon = np.random.rand(20, 20)
        xf = np.linspace(-0.01, 0.01, 20)
        yf = np.linspace(-0.01, 0.01, 20)
        ax = plot_reconstruction(recon, xf, yf)
        assert ax is not None
        plt.close("all")

    def test_plot_cross_sections(self):
        import matplotlib.pyplot as plt
        from src.visualization import plot_cross_sections
        recon = np.random.rand(20, 20)
        xf = np.linspace(-0.01, 0.01, 20)
        yf = np.linspace(-0.01, 0.01, 20)
        axes = plot_cross_sections(recon, xf, yf)
        assert axes is not None
        plt.close("all")

    def test_plot_signals(self):
        import matplotlib.pyplot as plt
        from src.visualization import plot_signals
        signals = np.random.rand(100, 5, 5)
        t = np.linspace(0, 1e-5, 100)
        xd = np.linspace(-0.01, 0.01, 5)
        yd = np.linspace(-0.01, 0.01, 5)
        ax = plot_signals(signals, t, xd, yd)
        assert ax is not None
        plt.close("all")

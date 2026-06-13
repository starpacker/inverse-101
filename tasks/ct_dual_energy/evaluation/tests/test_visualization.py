"""Tests for visualization.py."""

import os
import sys
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.visualization import (
    compute_ncc, compute_nrmse, compute_metrics,
    plot_material_maps, plot_sinograms, plot_spectra_and_mac,
)
import matplotlib
matplotlib.use("Agg")

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")


class TestComputeNCC:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert compute_ncc(a, a) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert compute_ncc(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "visualization_metrics.npz"))
        ncc = compute_ncc(fix["input_a"], fix["input_b"])
        np.testing.assert_allclose(ncc, float(fix["output_ncc"]), rtol=1e-10)


class TestComputeNRMSE:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert compute_nrmse(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        ref = np.array([0.0, 1.0])
        est = np.array([0.0, 0.5])
        # RMSE = sqrt(0.25/2) = 0.3536, range = 1.0
        assert compute_nrmse(est, ref) == pytest.approx(0.3535533906, rel=1e-6)

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "visualization_metrics.npz"))
        nrmse = compute_nrmse(fix["input_a"], fix["input_b"])
        np.testing.assert_allclose(nrmse, float(fix["output_nrmse"]), rtol=1e-10)


class TestComputeMetrics:
    def test_perfect_match(self):
        # Use a tissue map with varying values so dynamic range > 0
        tissue = np.zeros((32, 32))
        tissue[4:28, 4:28] = 0.8
        tissue[10:20, 10:20] = 1.0
        bone = np.zeros((32, 32))
        bone[10:20, 10:20] = 1.0
        metrics = compute_metrics(tissue, tissue, bone, bone)
        assert metrics["tissue_ncc"] == pytest.approx(1.0)
        assert metrics["tissue_nrmse"] == pytest.approx(0.0)
        assert metrics["bone_ncc"] == pytest.approx(1.0)
        assert metrics["bone_nrmse"] == pytest.approx(0.0)

    def test_returns_expected_keys(self):
        t = np.ones((10, 10))
        b = np.zeros((10, 10))
        b[3:7, 3:7] = 1.0
        metrics = compute_metrics(t, t, b, b)
        expected = {"tissue_ncc", "tissue_nrmse", "bone_ncc", "bone_nrmse",
                    "mean_ncc", "mean_nrmse"}
        assert expected == set(metrics.keys())

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "visualization_compute_metrics.npz"))
        m = compute_metrics(fix["input_tissue_est"], fix["input_tissue_ref"],
                            fix["input_bone_est"], fix["input_bone_ref"])
        np.testing.assert_allclose(m["tissue_ncc"], float(fix["output_tissue_ncc"]), rtol=1e-10)
        np.testing.assert_allclose(m["bone_ncc"], float(fix["output_bone_ncc"]), rtol=1e-10)
        np.testing.assert_allclose(m["tissue_nrmse"], float(fix["output_tissue_nrmse"]), rtol=1e-10)
        np.testing.assert_allclose(m["bone_nrmse"], float(fix["output_bone_nrmse"]), rtol=1e-10)


class TestPlotFunctions:
    """Smoke tests for plotting functions (verify they don't crash)."""

    def test_plot_material_maps(self, tmp_path):
        t = np.random.rand(32, 32)
        b = np.random.rand(32, 32)
        plot_material_maps(t, b, t, b, save_path=str(tmp_path / "test.png"))
        assert (tmp_path / "test.png").exists()

    def test_plot_sinograms(self, tmp_path):
        s1 = np.random.rand(32, 30)
        s2 = np.random.rand(32, 30)
        plot_sinograms(s1, s2, save_path=str(tmp_path / "test.png"))
        assert (tmp_path / "test.png").exists()

    def test_plot_spectra_and_mac(self, tmp_path):
        energies = np.arange(20, 151, dtype=np.float64)
        spectra = np.random.rand(2, 131)
        mus = np.random.rand(2, 131)
        plot_spectra_and_mac(energies, spectra, mus,
                             save_path=str(tmp_path / "test.png"))
        assert (tmp_path / "test.png").exists()

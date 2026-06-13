"""Unit tests for src/visualization.py."""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.visualization import compute_psnr, compute_metrics, print_metrics

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/visualization")


class TestComputePsnr:
    def test_identical(self):
        a = np.random.rand(16, 16)
        p = compute_psnr(a, a)
        assert p > 100  # effectively infinite

    def test_positive_for_similar(self):
        a = np.random.rand(16, 16)
        b = a + 0.01 * np.random.randn(16, 16)
        assert compute_psnr(b, a) > 0


class TestComputeMetrics:
    def test_perfect_match(self):
        a = np.random.rand(16, 16)
        m = compute_metrics(a, a)
        assert abs(m["ncc"] - 1.0) < 1e-10
        assert abs(m["nrmse"]) < 1e-10

    def test_ncc_range(self):
        a = np.random.rand(16, 16)
        b = np.random.rand(16, 16)
        m = compute_metrics(a, b)
        assert -1 <= m["ncc"] <= 1

    def test_fixture_parity(self):
        f = np.load(os.path.join(FIXTURES_DIR, "metrics_test.npz"))
        m = compute_metrics(f["input_a"], f["input_b"])
        np.testing.assert_allclose(m["ncc"], float(f["output_ncc"]), rtol=1e-10)
        np.testing.assert_allclose(m["nrmse"], float(f["output_nrmse"]), rtol=1e-10)
        np.testing.assert_allclose(m["psnr"], float(f["output_psnr"]), rtol=1e-10)


class TestPrintMetrics:
    def test_smoke(self, capsys):
        print_metrics({"psnr": 20.0, "ncc": 0.99, "nrmse": 0.03}, "test")
        captured = capsys.readouterr()
        assert "20.00" in captured.out

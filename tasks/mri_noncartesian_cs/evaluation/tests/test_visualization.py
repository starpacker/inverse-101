"""Unit tests for visualization.py."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.visualization import compute_metrics, compute_batch_metrics

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/visualization")


class TestComputeMetrics:
    def setup_method(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_metrics.npz"))
        out = np.load(os.path.join(FIXTURE_DIR, "output_metrics.npz"))
        self.estimate = inp["estimate"]
        self.reference = inp["reference"]
        self.expected_nrmse = float(out["nrmse"])
        self.expected_ncc = float(out["ncc"])
        self.expected_psnr = float(out["psnr"])

    def test_nrmse(self):
        m = compute_metrics(self.estimate, self.reference)
        np.testing.assert_allclose(m["nrmse"], self.expected_nrmse, rtol=1e-5)

    def test_ncc(self):
        m = compute_metrics(self.estimate, self.reference)
        np.testing.assert_allclose(m["ncc"], self.expected_ncc, rtol=1e-5)

    def test_psnr(self):
        m = compute_metrics(self.estimate, self.reference)
        np.testing.assert_allclose(m["psnr"], self.expected_psnr, rtol=1e-5)

    def test_perfect_match(self):
        m = compute_metrics(self.reference, self.reference)
        assert m["nrmse"] == 0.0
        assert m["ncc"] == pytest.approx(1.0, abs=1e-10)

    def test_ncc_range(self):
        m = compute_metrics(self.estimate, self.reference)
        assert -1.0 <= m["ncc"] <= 1.0

    def test_nrmse_nonnegative(self):
        m = compute_metrics(self.estimate, self.reference)
        assert m["nrmse"] >= 0.0


class TestComputeBatchMetrics:
    def test_batch(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_metrics.npz"))
        est = inp["estimate"][np.newaxis]
        ref = inp["reference"][np.newaxis]
        bm = compute_batch_metrics(est, ref)
        assert "per_sample" in bm
        assert "avg_nrmse" in bm
        assert "avg_ncc" in bm
        assert len(bm["per_sample"]) == 1

"""Unit tests for visualization.py."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.visualization import compute_metrics, compute_batch_metrics, print_metrics_table

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/visualization")
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
REF_DIR = os.path.join(os.path.dirname(__file__), "../reference_outputs")


class TestComputeMetrics:
    def setup_method(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "output_metrics_sample0.npz"))
        self.expected_nrmse = float(fix["nrmse"])
        self.expected_ncc = float(fix["ncc"])
        self.expected_psnr = float(fix["psnr"])

        gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))
        ref = np.load(os.path.join(REF_DIR, "tv_reconstruction.npz"))
        self.gt_mag = np.abs(gt["mvue"][0, 0])
        self.recon_mag = np.abs(ref["reconstruction"][0])

    def test_nrmse(self):
        m = compute_metrics(self.recon_mag, self.gt_mag)
        np.testing.assert_allclose(m["nrmse"], self.expected_nrmse, rtol=1e-4)

    def test_ncc(self):
        m = compute_metrics(self.recon_mag, self.gt_mag)
        np.testing.assert_allclose(m["ncc"], self.expected_ncc, rtol=1e-4)

    def test_psnr(self):
        m = compute_metrics(self.recon_mag, self.gt_mag)
        np.testing.assert_allclose(m["psnr"], self.expected_psnr, rtol=1e-3)

    def test_perfect_reconstruction(self):
        m = compute_metrics(self.gt_mag, self.gt_mag)
        assert m["ncc"] > 0.9999
        assert m["nrmse"] < 1e-6

    def test_ncc_range(self):
        m = compute_metrics(self.recon_mag, self.gt_mag)
        assert -1 <= m["ncc"] <= 1

    def test_nrmse_positive(self):
        m = compute_metrics(self.recon_mag, self.gt_mag)
        assert m["nrmse"] >= 0


class TestComputeBatchMetrics:
    def setup_method(self):
        gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))
        ref = np.load(os.path.join(REF_DIR, "tv_reconstruction.npz"))
        self.gt_mag = np.abs(gt["mvue"][:, 0])
        self.recon_mag = np.abs(ref["reconstruction"])

    def test_per_sample_count(self):
        m = compute_batch_metrics(self.recon_mag, self.gt_mag)
        assert len(m["per_sample"]) == 1

    def test_avg_keys(self):
        m = compute_batch_metrics(self.recon_mag, self.gt_mag)
        assert "avg_nrmse" in m
        assert "avg_ncc" in m
        assert "avg_psnr" in m

    def test_avg_ncc_reasonable(self):
        m = compute_batch_metrics(self.recon_mag, self.gt_mag)
        assert m["avg_ncc"] > 0.9

    def test_avg_nrmse_reasonable(self):
        m = compute_batch_metrics(self.recon_mag, self.gt_mag)
        assert m["avg_nrmse"] < 0.1


class TestPrintMetricsTable:
    def test_smoke(self):
        """Ensure print_metrics_table runs without errors."""
        metrics = {
            "per_sample": [{"psnr": 25.0, "ncc": 0.95, "nrmse": 0.05}],
            "avg_psnr": 25.0,
            "avg_ncc": 0.95,
            "avg_nrmse": 0.05,
        }
        print_metrics_table(metrics)

"""End-to-end integration test for non-Cartesian MRI reconstruction."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
REF_DIR = os.path.join(os.path.dirname(__file__), "../reference_outputs")


class TestReferenceOutputsExist:
    def test_gridding_exists(self):
        path = os.path.join(REF_DIR, "gridding_reconstruction.npz")
        assert os.path.exists(path)

    def test_l1wav_exists(self):
        path = os.path.join(REF_DIR, "l1wav_reconstruction.npz")
        assert os.path.exists(path)

    def test_metrics_detail_exists(self):
        path = os.path.join(REF_DIR, "metrics_detail.json")
        assert os.path.exists(path)


class TestL1WavQuality:
    """Verify L1-wavelet reconstruction quality against ground truth."""

    def setup_method(self):
        from src.visualization import compute_metrics

        gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))
        ref = np.load(os.path.join(REF_DIR, "l1wav_reconstruction.npz"))
        gt_mag = np.abs(gt["phantom"][0].astype(np.complex128))
        recon_mag = np.abs(ref["reconstruction"][0].astype(np.complex128))
        self.metrics = compute_metrics(recon_mag, gt_mag)

    def test_ncc_above_threshold(self):
        """L1-wavelet should achieve NCC > 0.95."""
        assert self.metrics["ncc"] > 0.95

    def test_nrmse_below_threshold(self):
        """L1-wavelet should achieve NRMSE < 0.1."""
        assert self.metrics["nrmse"] < 0.1

    def test_psnr_above_threshold(self):
        """L1-wavelet should achieve PSNR > 20 dB."""
        assert self.metrics["psnr"] > 20.0


class TestL1WavBetterThanGridding:
    """L1-wavelet should outperform gridding."""

    def setup_method(self):
        from src.visualization import compute_metrics

        gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))
        gt_mag = np.abs(gt["phantom"][0].astype(np.complex128))

        grid = np.load(os.path.join(REF_DIR, "gridding_reconstruction.npz"))
        grid_mag = np.abs(grid["reconstruction"][0].astype(np.complex128))

        l1wav = np.load(os.path.join(REF_DIR, "l1wav_reconstruction.npz"))
        l1wav_mag = np.abs(l1wav["reconstruction"][0].astype(np.complex128))

        self.grid_metrics = compute_metrics(grid_mag, gt_mag)
        self.l1wav_metrics = compute_metrics(l1wav_mag, gt_mag)

    def test_ncc_improvement(self):
        assert self.l1wav_metrics["ncc"] > self.grid_metrics["ncc"]

    def test_nrmse_improvement(self):
        assert self.l1wav_metrics["nrmse"] < self.grid_metrics["nrmse"]

    def test_psnr_improvement(self):
        assert self.l1wav_metrics["psnr"] > self.grid_metrics["psnr"]

"""End-to-end integration test for MRI TV reconstruction pipeline."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing import prepare_data
from src.solvers import tv_reconstruct_single
from src.visualization import compute_metrics

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")


class TestEndToEnd:
    """Test the full pipeline on sample 0."""

    def setup_method(self):
        self.obs_data, self.ground_truth, self.metadata = prepare_data(DATA_DIR)

    def test_single_sample_pipeline(self):
        masked_kspace = self.obs_data["masked_kspace"][0]
        smaps = self.obs_data["sensitivity_maps"][0]
        gt_mag = np.abs(self.ground_truth[0, 0])

        recon = tv_reconstruct_single(masked_kspace, smaps, lamda=1e-4)
        recon_mag = np.abs(recon)

        metrics = compute_metrics(recon_mag, gt_mag)

        assert metrics["ncc"] > 0.9, f"NCC too low: {metrics['ncc']}"
        assert metrics["nrmse"] < 0.1, f"NRMSE too high: {metrics['nrmse']}"
        assert metrics["psnr"] > 20, f"PSNR too low: {metrics['psnr']}"

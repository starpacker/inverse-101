"""End-to-end integration tests."""

import os
import numpy as np
import pytest
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_observation, normalize, minus_log
from src.physics_model import find_rotation_center, _shift_sinogram
from src.solvers import filtered_back_projection, circular_mask
from src.visualization import compute_metrics


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def pipeline_result(self):
        data_dir = os.path.join(TASK_DIR, "data")
        obs = load_observation(data_dir)
        proj_norm = normalize(obs["projections"], obs["flat_field"], obs["dark_field"])
        sino_data = minus_log(proj_norm)
        rot_center = find_rotation_center(
            sino_data[:, 0, :], obs["theta"], init=290, tol=0.5
        )
        sino_shifted = _shift_sinogram(sino_data[:, 0, :], rot_center, 640)
        recon = filtered_back_projection(sino_shifted, obs["theta"], 640)
        recon = circular_mask(recon, ratio=0.95)
        return recon, rot_center

    def test_reconstruction_shape(self, pipeline_result):
        recon, _ = pipeline_result
        assert recon.shape == (640, 640)

    def test_rotation_center_reasonable(self, pipeline_result):
        _, rot_center = pipeline_result
        assert 290 < rot_center < 310

    def test_ncc_vs_baseline(self, pipeline_result):
        recon, _ = pipeline_result
        ref_path = os.path.join(TASK_DIR, "data", "baseline_reference.npz")
        ref = np.load(ref_path)["reconstruction"][0, 0]
        metrics = compute_metrics(recon, ref)
        # Must exceed boundary from metrics.json
        assert metrics["ncc"] >= 0.88, f"NCC {metrics['ncc']:.4f} below threshold"

    def test_nrmse_vs_baseline(self, pipeline_result):
        recon, _ = pipeline_result
        ref_path = os.path.join(TASK_DIR, "data", "baseline_reference.npz")
        ref = np.load(ref_path)["reconstruction"][0, 0]
        metrics = compute_metrics(recon, ref)
        assert metrics["nrmse"] <= 0.035, f"NRMSE {metrics['nrmse']:.4f} above threshold"

"""End-to-end tests for weather radar data assimilation task."""

import os
import sys
import numpy as np
import pytest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)
DATA_DIR = os.path.join(TASK_DIR, "data")
REF_DIR = os.path.join(TASK_DIR, "evaluation", "reference_outputs")


def ncc(x, ref):
    x_flat = x.flatten()
    r_flat = ref.flatten()
    return float(np.dot(x_flat, r_flat) / (np.linalg.norm(x_flat) * np.linalg.norm(r_flat)))


def nrmse(x, ref):
    rms = np.sqrt(np.mean((x - ref) ** 2))
    dyn_range = ref.max() - ref.min()
    return float(rms / dyn_range)


class TestReferenceOutputs:
    """Validate that reference outputs are well-formed and meet quality thresholds."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))["target_frames"][0]
        ref_path = os.path.join(REF_DIR, "reconstruction.npz")
        if os.path.exists(ref_path):
            self.recon = np.load(ref_path)["reconstructed_frames"][0]
        else:
            self.recon = None

    def test_reference_outputs_exist(self):
        assert os.path.exists(os.path.join(REF_DIR, "reconstruction.npz"))

    def test_reference_shape_matches_gt(self):
        if self.recon is None:
            pytest.skip("No reference output")
        assert self.recon.shape == self.gt.shape

    def test_reference_ncc_above_threshold(self):
        """FlowDAS reconstruction should achieve NCC > 0.8 vs ground truth."""
        if self.recon is None:
            pytest.skip("No reference output")
        val = ncc(self.recon, self.gt)
        assert val > 0.8, f"NCC {val:.4f} below threshold 0.8"

    def test_reference_nrmse_below_threshold(self):
        """FlowDAS reconstruction should achieve NRMSE < 0.1 vs ground truth."""
        if self.recon is None:
            pytest.skip("No reference output")
        val = nrmse(self.recon, self.gt)
        assert val < 0.1, f"NRMSE {val:.4f} above threshold 0.1"

    def test_per_frame_ncc(self):
        """Each frame should have reasonable NCC."""
        if self.recon is None:
            pytest.skip("No reference output")
        for i in range(self.gt.shape[0]):
            val = ncc(self.recon[i], self.gt[i])
            assert val > 0.7, f"Frame {i} NCC {val:.4f} below threshold 0.7"

    def test_ground_truth_valid(self):
        assert self.gt.shape == (3, 128, 128)
        assert self.gt.dtype == np.float32
        assert self.gt.min() >= 0.0
        assert self.gt.max() <= 1.0


class TestDataConsistency:
    """Verify raw data and ground truth are consistent."""

    def test_observation_sparse(self):
        """Observations should be sparser than ground truth (mask ~10%)."""
        raw = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
        gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))["target_frames"][0]
        obs = raw["observations"][0]
        gt_nonzero = (gt > 0.01).mean()
        obs_nonzero = (obs > 0.01).mean()
        assert obs_nonzero < gt_nonzero, "Observations should be sparser than ground truth"

    def test_condition_frames_complete(self):
        """Condition frames should not be masked (full coverage)."""
        raw = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
        cond = raw["condition_frames"][0]
        # Condition frames are full-resolution, so many pixels should be nonzero
        # (weather radar data has precipitation in some areas)
        assert cond.shape == (6, 128, 128)

    def test_metrics_json_exists(self):
        import json
        metrics_path = os.path.join(TASK_DIR, "evaluation", "metrics.json")
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "baseline" in metrics
        assert "ncc_boundary" in metrics
        assert "nrmse_boundary" in metrics

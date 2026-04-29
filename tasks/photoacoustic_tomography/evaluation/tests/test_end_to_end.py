"""End-to-end integration tests for PAT reconstruction."""

import numpy as np
import json
import os
import pytest


class TestEndToEnd:
    """Full pipeline quality tests."""

    def test_reconstruction_matches_reference(self):
        """Reconstruction should match saved reference outputs exactly."""
        from src.preprocessing import load_raw_data, load_metadata
        from src.solvers import universal_back_projection
        from src.visualization import compute_ncc, compute_nrmse, centre_crop

        signals, xd, yd, t = load_raw_data("data")
        meta = load_metadata("data")
        c = meta["sound_speed_m_per_s"]
        z_target = meta["target_plane_z_m"]

        recon, xf, yf, zf = universal_back_projection(
            signals, xd, yd, t, z_target, c=c,
            resolution=500e-6,
            det_area=(meta["detector_size_m"])**2,
        )

        ref = np.load("evaluation/reference_outputs/reconstruction.npz")
        ref_recon = ref["reconstruction"][0]

        np.testing.assert_allclose(
            np.squeeze(recon), ref_recon, rtol=1e-10)

    def test_metrics_above_boundary(self):
        """Reconstruction quality should meet metrics.json boundaries."""
        with open("evaluation/metrics.json", "r") as f:
            metrics = json.load(f)

        from src.preprocessing import load_raw_data, load_ground_truth, load_metadata
        from src.solvers import universal_back_projection
        from src.visualization import compute_ncc, compute_nrmse, centre_crop

        signals, xd, yd, t = load_raw_data("data")
        gt_image, _, _ = load_ground_truth("data")
        meta = load_metadata("data")

        recon, xf, yf, zf = universal_back_projection(
            signals, xd, yd, t, meta["target_plane_z_m"],
            c=meta["sound_speed_m_per_s"],
            resolution=500e-6,
            det_area=(meta["detector_size_m"])**2,
        )

        recon_crop = centre_crop(np.squeeze(recon))
        gt_crop = centre_crop(gt_image)

        ncc = compute_ncc(recon_crop, gt_crop)
        nrmse = compute_nrmse(recon_crop, gt_crop)

        assert ncc >= metrics["ncc_boundary"], \
            f"NCC {ncc:.4f} below boundary {metrics['ncc_boundary']}"
        assert nrmse <= metrics["nrmse_boundary"], \
            f"NRMSE {nrmse:.4f} above boundary {metrics['nrmse_boundary']}"

    def test_data_files_exist(self):
        """All required data files should exist."""
        assert os.path.exists("data/raw_data.npz")
        assert os.path.exists("data/ground_truth.npz")
        assert os.path.exists("data/meta_data.json")

    def test_reference_outputs_exist(self):
        """Reference outputs should exist."""
        assert os.path.exists("evaluation/reference_outputs/reconstruction.npz")
        assert os.path.exists("evaluation/metrics.json")

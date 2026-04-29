"""End-to-end tests for the DCE-MRI reconstruction pipeline."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.preprocessing import prepare_data
from src.solvers import zero_filled_recon, temporal_tv_pgd
from src.visualization import compute_frame_metrics, compute_ncc, compute_nrmse


class TestEndToEnd:
    @pytest.fixture(scope='class')
    def data(self):
        data_dir = os.path.join(TASK_DIR, 'data')
        obs, gt, meta = prepare_data(data_dir)
        return obs, gt, meta

    def test_zero_fill_quality(self, data):
        """Zero-fill should achieve reasonable quality."""
        obs, gt, _ = data
        zf = zero_filled_recon(obs['undersampled_kspace'])
        ncc = compute_ncc(zf, gt['dynamic_images'])
        nrmse = compute_nrmse(zf, gt['dynamic_images'])
        assert ncc > 0.90, f"Zero-fill NCC {ncc:.4f} too low"
        assert nrmse < 0.15, f"Zero-fill NRMSE {nrmse:.4f} too high"

    def test_tv_improves_over_zero_fill(self, data):
        """Temporal TV should improve over zero-fill."""
        obs, gt, _ = data
        kspace = obs['undersampled_kspace']
        masks = obs['undersampling_masks']
        gt_images = gt['dynamic_images']

        zf = zero_filled_recon(kspace)
        tv_recon, _ = temporal_tv_pgd(kspace, masks, lamda=0.001,
                                       max_iter=100, tol=1e-6)

        zf_nrmse = compute_nrmse(zf, gt_images)
        tv_nrmse = compute_nrmse(tv_recon, gt_images)
        assert tv_nrmse < zf_nrmse, (
            f"TV NRMSE {tv_nrmse:.4f} should be less than ZF {zf_nrmse:.4f}")

    def test_tv_meets_boundary(self, data):
        """Temporal TV should meet the metrics boundary."""
        obs, gt, _ = data
        kspace = obs['undersampled_kspace']
        masks = obs['undersampling_masks']
        gt_images = gt['dynamic_images']

        tv_recon, _ = temporal_tv_pgd(kspace, masks, lamda=0.001,
                                       max_iter=200, tol=1e-6)

        ncc = compute_ncc(tv_recon, gt_images)
        nrmse = compute_nrmse(tv_recon, gt_images)

        # From metrics.json boundaries
        assert ncc >= 0.8777, f"NCC {ncc:.4f} below boundary 0.8777"
        assert nrmse <= 0.0667, f"NRMSE {nrmse:.4f} above boundary 0.0667"

    def test_reference_outputs_exist(self):
        """Reference outputs should be present."""
        ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
        assert os.path.exists(os.path.join(ref_dir, 'tv_reconstruction.npz'))
        assert os.path.exists(os.path.join(ref_dir, 'zero_filled.npz'))
        assert os.path.exists(os.path.join(ref_dir, 'convergence.npz'))

    def test_reference_output_shape(self):
        """Reference outputs should have correct shapes."""
        ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
        tv = np.load(os.path.join(ref_dir, 'tv_reconstruction.npz'))
        assert tv['reconstruction'].shape == (20, 128, 128)

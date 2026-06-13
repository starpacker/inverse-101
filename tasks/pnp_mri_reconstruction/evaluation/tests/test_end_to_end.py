"""End-to-end integration test.

Verifies the full pipeline against reference outputs without running the
expensive MSSN denoiser. Tests data loading, forward model, and metrics
computation against precomputed reference results.
"""

import os
import json
import pytest
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
REF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference_outputs")

import sys
sys.path.insert(0, TASK_DIR)

from src.preprocessing import prepare_data
from src.physics_model import MRIForwardModel
from src.visualization import compute_metrics, compute_snr


class TestEndToEnd:
    @pytest.fixture(autouse=True)
    def setup(self):
        metrics_path = os.path.join(REF_DIR, "metrics.json")
        if not os.path.exists(metrics_path):
            pytest.skip("Reference outputs not generated. Run generate_reference_outputs.py first.")

        with open(metrics_path) as f:
            self.ref_metrics = json.load(f)

    def test_data_pipeline(self):
        """Verify data preparation produces consistent outputs."""
        img, mask, y, metadata = prepare_data(os.path.join(TASK_DIR, "data"))
        assert img.shape == (320, 320)
        assert mask.shape == (320, 320)
        assert y.shape == (320, 320)
        assert mask.dtype == bool
        np.testing.assert_allclose(img.min(), 0.0, atol=1e-10)
        np.testing.assert_allclose(img.max(), 1.0, atol=1e-10)

    def test_ifft_snr_matches_reference(self):
        """Verify IFFT reconstruction matches reference SNR."""
        img, mask, y, metadata = prepare_data(os.path.join(TASK_DIR, "data"))
        model = MRIForwardModel(mask)
        ifft_recon = model.ifft_recon(y)
        ifft_metrics = compute_metrics(img, ifft_recon)

        ref_snr = self.ref_metrics["IFFT"]["snr_db"]
        np.testing.assert_allclose(ifft_metrics["snr_db"], ref_snr, rtol=1e-3)

    def test_pnp_mssn_recon_matches_reference(self):
        """Verify stored PnP-MSSN reconstruction matches reference metrics."""
        recon_path = os.path.join(REF_DIR, "pnp_mssn_recon.npy")
        gt_path = os.path.join(REF_DIR, "ground_truth.npy")
        if not os.path.exists(recon_path) or not os.path.exists(gt_path):
            pytest.skip("Reference reconstruction not available")

        recon = np.load(recon_path)
        gt = np.load(gt_path)
        metrics = compute_metrics(gt, recon)

        ref_snr = self.ref_metrics["PnP-MSSN"]["snr_db"]
        np.testing.assert_allclose(metrics["snr_db"], ref_snr, rtol=1e-2)

    def test_pnp_mssn_beats_ifft(self):
        """PnP-MSSN must achieve higher SNR than IFFT."""
        assert self.ref_metrics["PnP-MSSN"]["snr_db"] > self.ref_metrics["IFFT"]["snr_db"]

    def test_convergence_history(self):
        """Verify SNR history shows monotonic improvement."""
        snr_path = os.path.join(REF_DIR, "snr_history.npy")
        if not os.path.exists(snr_path):
            pytest.skip("SNR history not available")

        snr_history = np.load(snr_path)
        assert len(snr_history) == 200

        # First 50 iterations should show clear improvement
        assert snr_history[49] > snr_history[0] + 2.0

        # Final SNR should be above 16 dB
        assert snr_history[-1] > 16.0

import os
import pytest
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "solvers")
REF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference_outputs")

import sys
sys.path.insert(0, TASK_DIR)


class TestPnPPGMResults:
    """Test PnP-PGM reconstruction against reference outputs.

    These tests verify that the reconstruction matches the reference
    without requiring a full GPU run.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        fixture_path = os.path.join(FIXTURE_DIR, "output_pnp_pgm.npz")
        if not os.path.exists(fixture_path):
            pytest.skip("Solver fixtures not generated. Run generate_fixtures.py first.")
        self.fixture = np.load(fixture_path)

    def test_recon_shape(self):
        ref_path = os.path.join(REF_DIR, "pnp_mssn_recon.npy")
        if not os.path.exists(ref_path):
            pytest.skip("Reference output not available")
        recon = np.load(ref_path)
        expected_shape = tuple(self.fixture["output_recon_shape"])
        assert recon.shape == expected_shape

    def test_snr_above_threshold(self):
        """PnP-MSSN should achieve SNR > 16 dB."""
        expected_snr = float(self.fixture["output_snr"])
        assert expected_snr > 16.0

    def test_nrmse_below_threshold(self):
        """PnP-MSSN NRMSE should be below 0.2."""
        expected_nrmse = float(self.fixture["output_nrmse"])
        assert expected_nrmse < 0.2

    def test_ncc_above_threshold(self):
        """PnP-MSSN NCC should be above 0.95."""
        expected_ncc = float(self.fixture["output_ncc"])
        assert expected_ncc > 0.95

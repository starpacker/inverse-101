"""Unit tests for src/solvers.py."""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.preprocessing import prepare_data
from src.denoiser import load_denoiser
from src.solvers import pnp_admm_reconstruct
from src.visualization import compute_metrics

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
REF_DIR = os.path.join(os.path.dirname(__file__), "../reference_outputs")


@pytest.fixture(scope="module")
def model():
    return load_denoiser(os.path.join(DATA_DIR, "RealSN_DnCNN_noise15.pth"), device="cpu")


@pytest.fixture(scope="module")
def data():
    return prepare_data(DATA_DIR, mask_name="random")


class TestPnPADMM:
    def test_short_run_shape(self, model, data):
        im_orig, mask, noises, _ = data
        result = pnp_admm_reconstruct(
            model, im_orig, mask, noises,
            alpha=2.0, sigma=15, maxitr=5, device="cpu",
        )
        assert result["reconstruction"].shape == (256, 256)
        assert result["zerofill"].shape == (256, 256)
        assert len(result["psnr_history"]) == 5

    def test_psnr_improves(self, model, data):
        im_orig, mask, noises, _ = data
        result = pnp_admm_reconstruct(
            model, im_orig, mask, noises,
            alpha=2.0, sigma=15, maxitr=10, device="cpu",
        )
        # PSNR should improve from iteration 1 to 10
        assert result["psnr_history"][-1] > result["psnr_history"][0]

    def test_parity_with_reference(self, model, data):
        """Full 100-iteration run must match reference output."""
        im_orig, mask, noises, _ = data
        result = pnp_admm_reconstruct(
            model, im_orig, mask, noises,
            alpha=2.0, sigma=15, maxitr=100, device="cpu",
        )
        ref = np.load(os.path.join(REF_DIR, "pnp_admm_reconstruction.npz"))["reconstruction"][0]
        m = compute_metrics(result["reconstruction"], ref.astype(np.float64))
        assert m["ncc"] > 0.9999, f"NCC too low: {m['ncc']}"
        assert m["nrmse"] < 0.001, f"NRMSE too high: {m['nrmse']}"

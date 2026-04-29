"""End-to-end integration tests for PnP-ADMM CS-MRI task."""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
REF_DIR = os.path.join(os.path.dirname(__file__), "../reference_outputs")


class TestEndToEnd:
    def test_short_pipeline(self):
        from src.preprocessing import prepare_data
        from src.denoiser import load_denoiser
        from src.solvers import pnp_admm_reconstruct
        from src.visualization import compute_metrics

        im_orig, mask, noises, _ = prepare_data(DATA_DIR)
        model = load_denoiser(os.path.join(DATA_DIR, "RealSN_DnCNN_noise15.pth"))

        result = pnp_admm_reconstruct(
            model, im_orig, mask, noises,
            alpha=2.0, sigma=15, maxitr=5, device="cpu",
        )
        m = compute_metrics(result["reconstruction"], im_orig)
        assert m["ncc"] > 0.9  # even 5 iters should improve on zerofill
        assert m["psnr"] > 15


class TestReferenceQuality:
    def test_pnp_vs_ground_truth(self):
        from src.visualization import compute_metrics

        recon = np.load(os.path.join(REF_DIR, "pnp_admm_reconstruction.npz"))["reconstruction"][0]
        gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))["image"][0]

        m = compute_metrics(recon.astype(np.float64), gt.astype(np.float64))
        assert m["ncc"] > 0.99, f"NCC too low: {m['ncc']}"
        assert m["nrmse"] < 0.05, f"NRMSE too high: {m['nrmse']}"
        assert m["psnr"] > 18, f"PSNR too low: {m['psnr']}"

    def test_pnp_better_than_zerofill(self):
        from src.visualization import compute_metrics

        recon = np.load(os.path.join(REF_DIR, "pnp_admm_reconstruction.npz"))["reconstruction"][0]
        zf = np.load(os.path.join(REF_DIR, "zerofill.npz"))["zerofill"][0]
        gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))["image"][0]

        m_pnp = compute_metrics(recon.astype(np.float64), gt.astype(np.float64))
        m_zf = compute_metrics(zf.astype(np.float64), gt.astype(np.float64))

        assert m_pnp["psnr"] > m_zf["psnr"], "PnP should have higher PSNR than zero-fill"
        assert m_pnp["ncc"] > m_zf["ncc"], "PnP should have higher NCC than zero-fill"
        assert m_pnp["nrmse"] < m_zf["nrmse"], "PnP should have lower NRMSE than zero-fill"

    def test_convergence(self):
        psnr_hist = np.load(os.path.join(REF_DIR, "psnr_history.npy"))
        # Should converge (last value close to plateau)
        assert psnr_hist[-1] >= psnr_hist[0]
        # PSNR should plateau (last 50 iters within 0.1 dB)
        assert abs(psnr_hist[-1] - psnr_hist[50]) < 0.1

    def test_reconstruction_non_negative_approx(self):
        recon = np.load(os.path.join(REF_DIR, "pnp_admm_reconstruction.npz"))["reconstruction"][0]
        # PnP doesn't enforce strict positivity, but most values should be non-negative
        frac_negative = (recon < 0).sum() / recon.size
        assert frac_negative < 0.15, f"Too many negative pixels: {frac_negative*100:.1f}%"

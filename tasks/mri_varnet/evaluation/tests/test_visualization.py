"""Unit tests for src/visualization.py."""
import os, sys, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.visualization import compute_metrics

class TestComputeMetrics:
    def test_perfect_match(self):
        a = np.random.rand(32, 32).astype(np.float32)
        m = compute_metrics(a, a)
        assert abs(m["ncc"] - 1.0) < 1e-6
        assert abs(m["nrmse"]) < 1e-6
        assert abs(m["ssim"] - 1.0) < 1e-4

    def test_ncc_range(self):
        a, b = np.random.rand(32, 32), np.random.rand(32, 32)
        m = compute_metrics(a, b)
        assert -1 <= m["ncc"] <= 1

class TestReferenceQuality:
    def test_varnet_vs_gt(self):
        REF_DIR = os.path.join(os.path.dirname(__file__), "../reference_outputs")
        DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
        recon = np.load(os.path.join(REF_DIR, "varnet_reconstruction.npz"))["reconstruction"]
        gt = np.load(os.path.join(DATA_DIR, "ground_truth.npz"))["image"]
        m = compute_metrics(recon[0], gt[0])
        assert m["ssim"] > 0.90, f"SSIM too low: {m['ssim']}"
        assert m["ncc"] > 0.99, f"NCC too low: {m['ncc']}"

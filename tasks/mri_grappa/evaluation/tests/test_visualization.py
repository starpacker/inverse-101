"""Unit tests for src/visualization.py."""
import os, sys, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.visualization import compute_metrics, print_metrics

class TestComputeMetrics:
    def test_perfect_match(self):
        a = np.random.rand(32, 32)
        m = compute_metrics(a, a)
        assert abs(m["ncc"] - 1.0) < 1e-10
        assert abs(m["nrmse"]) < 1e-10
        assert abs(m["ssim"] - 1.0) < 1e-6

    def test_ncc_range(self):
        a, b = np.random.rand(32, 32), np.random.rand(32, 32)
        m = compute_metrics(a, b)
        assert -1 <= m["ncc"] <= 1

class TestEndToEnd:
    def test_grappa_better_than_zerofill(self):
        REF_DIR = os.path.join(os.path.dirname(__file__), "../reference_outputs")
        DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
        recon = np.load(os.path.join(REF_DIR, "grappa_reconstruction.npz"))["reconstruction"][0]
        zf = np.load(os.path.join(REF_DIR, "zerofill.npz"))["reconstruction"][0]
        ref = np.load(os.path.join(REF_DIR, "ground_truth.npy"))
        m_g = compute_metrics(recon.astype(np.float64), ref.astype(np.float64))
        m_zf = compute_metrics(zf.astype(np.float64), ref.astype(np.float64))
        assert m_g["ssim"] > m_zf["ssim"]
        assert m_g["ncc"] > m_zf["ncc"]
        assert m_g["nrmse"] < m_zf["nrmse"]

"""Unit tests for src/visualization.py."""
import os, sys, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.visualization import compute_metrics

class TestMetrics:
    def test_perfect_match(self):
        a = np.random.rand(32, 32).astype(np.float64)
        m = compute_metrics(a, a)
        assert abs(m["ncc"] - 1.0) < 1e-10
        assert abs(m["nrmse"]) < 1e-10

    def test_ncc_range(self):
        a, b = np.random.rand(32, 32), np.random.rand(32, 32)
        m = compute_metrics(a, b)
        assert -1 <= m["ncc"] <= 1

class TestReferenceQuality:
    def test_sense_vs_gt(self):
        REF = os.path.join(os.path.dirname(__file__), "../reference_outputs")
        recon = np.load(os.path.join(REF, "sense_reconstruction.npz"))["reconstruction"][0]
        gt = np.load(os.path.join(REF, "fully_sampled.npy"))
        m = compute_metrics(recon, gt)
        assert m["ncc"] > 0.95
        assert m["nrmse"] < 0.10

    def test_sense_better_than_zerofill(self):
        REF = os.path.join(os.path.dirname(__file__), "../reference_outputs")
        recon = np.load(os.path.join(REF, "sense_reconstruction.npz"))["reconstruction"][0]
        zf = np.load(os.path.join(REF, "zerofill.npz"))["reconstruction"][0]
        gt = np.load(os.path.join(REF, "fully_sampled.npy"))
        m_s = compute_metrics(recon, gt)
        m_zf = compute_metrics(zf, gt)
        assert m_s["ncc"] > m_zf["ncc"]
        assert m_s["nrmse"] < m_zf["nrmse"]

"""End-to-end integration tests."""

import os
import sys
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.visualization import compute_ncc, compute_nrmse


class TestMetrics:
    def test_ncc_identity(self):
        a = np.random.randn(10, 10)
        assert abs(compute_ncc(a, a) - 1.0) < 1e-10

    def test_nrmse_zero(self):
        a = np.random.randn(10, 10)
        assert compute_nrmse(a, a) < 1e-10

    def test_nrmse_positive(self):
        a = np.random.randn(10, 10)
        b = np.random.randn(10, 10)
        assert compute_nrmse(a, b) > 0


class TestReferenceOutputs:
    def test_reconstruction_exists(self):
        path = os.path.join(TASK_DIR, "evaluation", "reference_outputs", "reconstruction.npy")
        assert os.path.exists(path)

    def test_velocity_in_bounds(self):
        path = os.path.join(TASK_DIR, "evaluation", "reference_outputs", "reconstruction.npy")
        if not os.path.exists(path):
            pytest.skip("Reconstruction not available")
        vp = np.load(path)
        assert vp.min() >= 1300, f"Vp min={vp.min():.1f} < 1300"
        assert vp.max() <= 1700, f"Vp max={vp.max():.1f} > 1700"

    def test_baseline_reference_batch_first(self):
        """Verify baseline_reference.npz uses batch-first convention."""
        ref = np.load(os.path.join(TASK_DIR, "data", "baseline_reference.npz"))
        assert ref["vp_reconstructed"].ndim == 3
        assert ref["vp_reconstructed"].shape[0] == 1

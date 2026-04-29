"""End-to-end integration test for X-ray ptychography reconstruction."""

import os
import sys

import numpy as np
import pytest

TASK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

try:
    import cupy
    cupy.array([1])
    HAS_GPU = True
except Exception:
    HAS_GPU = False

DATA_DIR = os.path.join(TASK_DIR, "data")


@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
class TestEndToEnd:
    """Integration test: load real data, run short reconstruction, verify."""

    def test_short_reconstruction(self):
        """Run 2 iterations on real data and verify outputs are sensible."""
        from src.preprocessing import (
            load_raw_data,
            shift_scan_positions,
            initialize_psi,
            add_probe_modes,
        )
        from src.solvers import reconstruct

        # Load
        raw = load_raw_data(DATA_DIR)
        data = raw["diffraction_patterns"]
        scan = raw["scan_positions"]
        probe = raw["probe_guess"]

        # Preprocess
        scan = shift_scan_positions(scan, offset=20.0)
        probe = add_probe_modes(probe, n_modes=1)
        psi = initialize_psi(scan, probe_shape=probe.shape)

        # Verify preprocessing shapes
        assert data.shape == (516, 128, 128)
        assert scan.shape == (516, 2)
        assert probe.ndim == 5
        assert psi.ndim == 3
        assert psi.shape[0] == 1

        # Short reconstruction (2 iterations, 7 batches)
        result = reconstruct(
            data=data,
            scan=scan,
            probe=probe,
            psi=psi,
            num_iter=2,
            num_batch=7,
        )

        # Verify output shapes
        assert result['psi'].ndim == 3
        assert result['psi'].shape[0] == 1
        assert result['probe'].ndim == 5
        assert result['probe'].shape[:2] == (1, 1)

        # Verify cost decreased
        costs = np.array(result['costs'])
        assert costs.shape[0] >= 2, "Expected at least 2 cost entries"
        mean_costs = costs.mean(axis=1) if costs.ndim == 2 else costs
        assert mean_costs[-1] < mean_costs[0], (
            f"Cost should decrease: first={mean_costs[0]:.4f}, "
            f"last={mean_costs[-1]:.4f}"
        )

        # Verify output values are finite
        assert np.all(np.isfinite(result['psi']))
        assert np.all(np.isfinite(result['probe']))


class TestMetrics:
    """Metric computation tests (no GPU required)."""

    def test_metrics_computation(self):
        """Verify metric computation works on synthetic arrays."""
        from src.visualization import compute_metrics

        # Create two similar arrays
        ref = np.random.rand(1, 100, 100).astype(np.float32)
        est = ref + np.random.randn(*ref.shape).astype(np.float32) * 0.01

        metrics = compute_metrics(est, ref)
        assert "ncc" in metrics
        assert "nrmse" in metrics
        assert 0 < metrics["ncc"] <= 1.0
        assert metrics["nrmse"] >= 0
        # For nearly identical arrays, NCC should be very high
        assert metrics["ncc"] > 0.99

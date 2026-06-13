"""Tests for src/solvers.py."""

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


@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
class TestReconstruct:
    """Tests for the reconstruct function."""

    def _make_synthetic_data(self):
        """Create a small synthetic dataset for testing."""
        from src.physics_model import simulate_diffraction

        N, W, H = 30, 32, 32
        probe = np.random.rand(1, 1, 1, W, H).astype(np.complex64) * 0.5
        scan = np.random.rand(N, 2).astype(np.float32) * 40 + 20
        psi_true = np.full((1, 130, 130), 0.5 + 0j, dtype=np.complex64)
        data = simulate_diffraction(probe, psi_true, scan)
        psi_init = np.full((1, 130, 130), 0.5 + 0j, dtype=np.complex64)
        return data, scan, probe, psi_init

    def test_result_has_expected_keys(self):
        from src.solvers import reconstruct

        data, scan, probe, psi = self._make_synthetic_data()
        result = reconstruct(data, scan, probe, psi,
                             num_iter=2, num_batch=2)

        assert 'psi' in result
        assert 'probe' in result
        assert 'costs' in result

    def test_output_shapes(self):
        from src.solvers import reconstruct

        data, scan, probe, psi = self._make_synthetic_data()
        result = reconstruct(data, scan, probe, psi,
                             num_iter=2, num_batch=2)

        assert result['psi'].ndim == 3
        assert result['probe'].ndim == 5
        assert result['probe'].shape[:2] == (1, 1)

    def test_costs_populated(self):
        from src.solvers import reconstruct

        data, scan, probe, psi = self._make_synthetic_data()
        result = reconstruct(data, scan, probe, psi,
                             num_iter=2, num_batch=2)

        costs = result['costs']
        assert len(costs) > 0

    def test_output_dtypes(self):
        from src.solvers import reconstruct

        data, scan, probe, psi = self._make_synthetic_data()
        result = reconstruct(data, scan, probe, psi,
                             num_iter=2, num_batch=2)

        assert np.iscomplexobj(result['psi'])
        assert np.iscomplexobj(result['probe'])

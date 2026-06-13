"""Tests for src/solvers.py."""

import os
import sys

import numpy as np
import pytest

try:
    import cupy
    cupy.array([1])
    HAS_GPU = True
except Exception:
    HAS_GPU = False

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)


@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
class TestReconstruct:
    """Tests for reconstruct."""

    def test_output_shape(self):
        """Test that reconstruction has correct volume shape."""
        from src.solvers import reconstruct

        n = 32
        n_angles = 8
        data = np.zeros((n_angles, n, n), dtype=np.complex64)
        theta = np.linspace(0, np.pi, n_angles, endpoint=False, dtype=np.float32)
        tilt = np.float32(np.pi / 2)

        result = reconstruct(
            data=data,
            theta=theta,
            tilt=tilt,
            volume_shape=(n, n, n),
            n_rounds=1,
            n_iter_per_round=2,
        )
        assert result['obj'].shape == (n, n, n)

    def test_output_dtype(self):
        """Test that reconstruction is complex64."""
        from src.solvers import reconstruct

        n = 32
        n_angles = 8
        data = np.zeros((n_angles, n, n), dtype=np.complex64)
        theta = np.linspace(0, np.pi, n_angles, endpoint=False, dtype=np.float32)
        tilt = np.float32(np.pi / 2)

        result = reconstruct(
            data=data,
            theta=theta,
            tilt=tilt,
            volume_shape=(n, n, n),
            n_rounds=1,
            n_iter_per_round=2,
        )
        assert result['obj'].dtype == np.complex64

    def test_costs_returned(self):
        """Test that costs list has one entry per round."""
        from src.solvers import reconstruct

        n = 32
        n_angles = 8
        data = np.zeros((n_angles, n, n), dtype=np.complex64)
        theta = np.linspace(0, np.pi, n_angles, endpoint=False, dtype=np.float32)
        tilt = np.float32(np.pi / 2)

        n_rounds = 3
        result = reconstruct(
            data=data,
            theta=theta,
            tilt=tilt,
            volume_shape=(n, n, n),
            n_rounds=n_rounds,
            n_iter_per_round=2,
        )
        assert len(result['costs']) == n_rounds

    def test_assertion_on_bad_data_ndim(self):
        """Test that non-3D data raises assertion."""
        from src.solvers import reconstruct

        data = np.zeros((1, 8, 32, 32), dtype=np.complex64)
        theta = np.array([0.0], dtype=np.float32)
        with pytest.raises(AssertionError):
            reconstruct(data, theta, np.pi / 2, (32, 32, 32))

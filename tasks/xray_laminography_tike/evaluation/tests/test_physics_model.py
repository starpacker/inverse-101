"""Tests for src/physics_model.py."""

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
class TestForwardProject:
    """Tests for forward_project."""

    def test_output_shape(self):
        """Test that output has shape (n_angles, n, n)."""
        from src.physics_model import forward_project

        n = 32
        n_angles = 8
        obj = np.zeros((n, n, n), dtype=np.complex64)
        theta = np.linspace(0, np.pi, n_angles, endpoint=False, dtype=np.float32)
        tilt = np.float32(np.pi / 2)

        result = forward_project(obj, theta, tilt)
        assert result.shape == (n_angles, n, n)

    def test_output_dtype(self):
        """Test that output is complex64."""
        from src.physics_model import forward_project

        n = 32
        n_angles = 8
        obj = np.zeros((n, n, n), dtype=np.complex64)
        theta = np.linspace(0, np.pi, n_angles, endpoint=False, dtype=np.float32)
        tilt = np.float32(np.pi / 2)

        result = forward_project(obj, theta, tilt)
        assert result.dtype == np.complex64

    def test_zero_volume_gives_zero_projections(self):
        """Test that projecting a zero volume gives near-zero projections."""
        from src.physics_model import forward_project

        n = 32
        n_angles = 8
        obj = np.zeros((n, n, n), dtype=np.complex64)
        theta = np.linspace(0, np.pi, n_angles, endpoint=False, dtype=np.float32)
        tilt = np.float32(np.pi / 2)

        result = forward_project(obj, theta, tilt)
        assert np.allclose(result, 0, atol=1e-6)

    def test_nonzero_volume_gives_nonzero_projections(self):
        """Test that projecting a nonzero volume gives nonzero projections."""
        from src.physics_model import forward_project

        n = 32
        n_angles = 8
        obj = np.ones((n, n, n), dtype=np.complex64)
        theta = np.linspace(0, np.pi, n_angles, endpoint=False, dtype=np.float32)
        tilt = np.float32(np.pi / 2)

        result = forward_project(obj, theta, tilt)
        assert np.max(np.abs(result)) > 0

    def test_assertion_on_bad_obj_ndim(self):
        """Test that a non-3D obj raises an assertion error."""
        from src.physics_model import forward_project

        obj = np.zeros((1, 32, 32, 32), dtype=np.complex64)
        theta = np.array([0.0], dtype=np.float32)
        with pytest.raises(AssertionError):
            forward_project(obj, theta, np.pi / 2)

    def test_assertion_on_bad_theta_ndim(self):
        """Test that a non-1D theta raises an assertion error."""
        from src.physics_model import forward_project

        obj = np.zeros((32, 32, 32), dtype=np.complex64)
        theta = np.array([[0.0]], dtype=np.float32)
        with pytest.raises(AssertionError):
            forward_project(obj, theta, np.pi / 2)

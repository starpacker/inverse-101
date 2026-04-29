"""Tests for src/physics_model.py."""

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

from src.physics_model import validate_inputs


class TestValidateInputs:
    """Tests for validate_inputs."""

    def _make_valid_inputs(self):
        """Create a minimal set of valid inputs."""
        N, W, H, S, D = 10, 32, 32, 1, 1
        data = np.random.rand(N, W, H).astype(np.float32)
        scan = np.random.rand(N, 2).astype(np.float32) * 10 + 20
        probe = np.random.rand(1, 1, S, W, H).astype(np.complex64)
        psi = np.full((D, 100, 100), 0.5 + 0j, dtype=np.complex64)
        return data, scan, probe, psi

    def test_valid_inputs_pass(self):
        data, scan, probe, psi = self._make_valid_inputs()
        validate_inputs(data, scan, probe, psi)

    def test_wrong_data_ndim(self):
        data, scan, probe, psi = self._make_valid_inputs()
        with pytest.raises(ValueError, match="data must be 3D"):
            validate_inputs(data[0], scan, probe, psi)

    def test_wrong_scan_shape(self):
        data, scan, probe, psi = self._make_valid_inputs()
        with pytest.raises(ValueError, match="scan must be"):
            validate_inputs(data, scan[:, :1], probe, psi)

    def test_mismatched_n_positions(self):
        data, scan, probe, psi = self._make_valid_inputs()
        with pytest.raises(ValueError, match="position counts differ"):
            validate_inputs(data[:5], scan, probe, psi)

    def test_wrong_probe_ndim(self):
        data, scan, probe, psi = self._make_valid_inputs()
        with pytest.raises(ValueError, match="probe must be"):
            validate_inputs(data, scan, probe[0], psi)

    def test_wrong_psi_ndim(self):
        data, scan, probe, psi = self._make_valid_inputs()
        with pytest.raises(ValueError, match="psi must be 3D"):
            validate_inputs(data, scan, probe, psi[0])

    def test_real_probe_rejected(self):
        data, scan, probe, psi = self._make_valid_inputs()
        with pytest.raises(ValueError, match="complex-valued"):
            validate_inputs(data, scan, np.abs(probe), psi)

    def test_real_psi_rejected(self):
        data, scan, probe, psi = self._make_valid_inputs()
        with pytest.raises(ValueError, match="complex-valued"):
            validate_inputs(data, scan, probe, np.abs(psi))


class TestExtractPatches:
    """Tests for extract_patches."""

    def test_basic_extraction(self):
        from src.physics_model import extract_patches

        psi = np.ones((1, 100, 100), dtype=np.complex64) * (0.5 + 0.1j)
        scan = np.array([[10, 20], [30, 40]], dtype=np.float32)
        patches = extract_patches(psi, scan, (32, 32))
        assert patches.shape == (2, 32, 32)
        np.testing.assert_allclose(patches[0], 0.5 + 0.1j)

    def test_patches_differ_for_varying_object(self):
        from src.physics_model import extract_patches

        psi = np.zeros((1, 100, 100), dtype=np.complex64)
        psi[0, 10:42, 20:52] = 1.0 + 0j
        psi[0, 30:62, 40:72] = 2.0 + 0j
        scan = np.array([[10, 20], [30, 40]], dtype=np.float32)
        patches = extract_patches(psi, scan, (32, 32))
        assert not np.allclose(patches[0], patches[1])


class TestForward:
    """Tests for the forward model."""

    def test_output_shape(self):
        from src.physics_model import forward

        N, W, H = 5, 16, 16
        psi = np.full((1, 80, 80), 0.5 + 0j, dtype=np.complex64)
        probe = np.random.rand(1, 1, 1, W, H).astype(np.complex64)
        scan = np.array([[10 + i * 5, 10 + i * 5] for i in range(N)],
                        dtype=np.float32)
        intensity = forward(psi, probe, scan)
        assert intensity.shape == (N, W, H)
        assert intensity.dtype == np.float32

    def test_output_non_negative(self):
        from src.physics_model import forward

        N, W, H = 5, 16, 16
        psi = np.full((1, 80, 80), 0.5 + 0j, dtype=np.complex64)
        probe = np.random.rand(1, 1, 1, W, H).astype(np.complex64)
        scan = np.array([[10 + i * 5, 10 + i * 5] for i in range(N)],
                        dtype=np.float32)
        intensity = forward(psi, probe, scan)
        assert np.all(intensity >= 0)


@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
class TestSimulateDiffractionGPU:
    """Tests for simulate_diffraction on GPU."""

    def test_gpu_output_shape(self):
        from src.physics_model import simulate_diffraction

        N, W, H = 20, 32, 32
        probe = np.random.rand(1, 1, 1, W, H).astype(np.complex64)
        scan = np.random.rand(N, 2).astype(np.float32) * 50 + 20
        psi = np.full((1, 150, 150), 0.5 + 0j, dtype=np.complex64)

        data = simulate_diffraction(probe, psi, scan)
        assert data.shape == (N, W, H)

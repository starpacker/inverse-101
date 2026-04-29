"""Tests for solvers module."""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.solvers import tv_3d, SSNPReconstructor
from src.physics_model import SSNPConfig, SSNPForwardModel


def _small_config():
    """Create a small SSNPConfig for testing."""
    return SSNPConfig(
        volume_shape=(4, 8, 8),
        res=(0.3, 0.3, 0.5),
        n0=1.0,
        NA=0.55,
        wavelength_um=0.5,
        res_um=(0.1, 0.1, 0.2),
        n_angles=2,
    )


# ── tv_3d ────────────────────────────────────────────────────────────────

class TestTV3D:
    def test_constant_volume_minimal_tv(self):
        """A constant volume should have near-zero TV (only epsilon contribution)."""
        vol = torch.ones(4, 8, 8, dtype=torch.float64) * 5.0
        tv_val = tv_3d(vol, epsilon=1e-8)
        # For a constant volume, all finite differences are 0,
        # TV = sum(sqrt(0 + 0 + 0 + eps)) = N * sqrt(eps)
        n_voxels = 4 * 8 * 8
        expected_max = n_voxels * np.sqrt(1e-8)
        assert tv_val.item() < expected_max * 1.1

    def test_tv_nonnegative(self):
        """TV should always be non-negative."""
        rng = np.random.default_rng(42)
        vol = torch.tensor(rng.random((4, 8, 8)), dtype=torch.float64)
        tv_val = tv_3d(vol)
        assert tv_val.item() >= 0

    def test_tv_increases_with_variation(self):
        """A volume with larger gradients should have larger TV."""
        vol_smooth = torch.ones(4, 8, 8, dtype=torch.float64)
        vol_rough = torch.tensor(np.random.default_rng(42).random((4, 8, 8)),
                                 dtype=torch.float64)

        tv_smooth = tv_3d(vol_smooth).item()
        tv_rough = tv_3d(vol_rough).item()
        assert tv_rough > tv_smooth

    def test_tv_scaling(self):
        """TV should scale linearly with the magnitude of gradients (for large gradients)."""
        rng = np.random.default_rng(42)
        vol = torch.tensor(rng.random((4, 8, 8)), dtype=torch.float64)
        scale = 10.0

        tv1 = tv_3d(vol, epsilon=1e-12).item()
        tv_scaled = tv_3d(vol * scale, epsilon=1e-12).item()

        # For large gradients (>> epsilon), TV(scale * x) ~ scale * TV(x)
        np.testing.assert_allclose(tv_scaled / tv1, scale, rtol=0.01)

    def test_tv_differentiable(self):
        """TV should support autograd."""
        vol = torch.ones(4, 8, 8, dtype=torch.float64, requires_grad=True)
        tv_val = tv_3d(vol)
        tv_val.backward()
        assert vol.grad is not None
        assert vol.grad.shape == (4, 8, 8)

    def test_tv_output_scalar(self):
        """TV should return a scalar tensor."""
        vol = torch.ones(4, 8, 8, dtype=torch.float64)
        tv_val = tv_3d(vol)
        assert tv_val.dim() == 0


# ── SSNPReconstructor ────────────────────────────────────────────────────

class TestSSNPReconstructor:
    def test_reconstruct_output_shape(self):
        """Reconstructed volume should have shape (Nz, Ny, Nx)."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        # Generate synthetic measurements from a known phantom
        rng = np.random.default_rng(42)
        phantom = rng.uniform(0, 0.005, (nz, ny, nx))
        dn_tensor = torch.tensor(phantom, dtype=torch.float64)

        with torch.no_grad():
            intensities = model.forward(dn_tensor)
        meas_amp = torch.sqrt(intensities)

        reconstructor = SSNPReconstructor(n_iter=2, lr=10.0, tv_weight=0.0,
                                          positivity=True, device="cpu")
        dn_recon, loss_history = reconstructor.reconstruct(meas_amp, model)

        assert dn_recon.shape == (nz, ny, nx)
        assert isinstance(dn_recon, np.ndarray)

    def test_reconstruct_returns_loss_history(self):
        """Loss history should have length equal to n_iter."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        dn_tensor = torch.zeros(nz, ny, nx, dtype=torch.float64)
        with torch.no_grad():
            intensities = model.forward(dn_tensor)
        meas_amp = torch.sqrt(intensities)

        n_iter = 3
        reconstructor = SSNPReconstructor(n_iter=n_iter, lr=10.0, tv_weight=0.0,
                                          positivity=True, device="cpu")
        _, loss_history = reconstructor.reconstruct(meas_amp, model)

        assert len(loss_history) == n_iter
        assert all(isinstance(l, float) for l in loss_history)

    def test_loss_decreases(self):
        """Loss should decrease during reconstruction (final < initial)."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        # Use a non-trivial phantom so there is something to reconstruct
        rng = np.random.default_rng(42)
        phantom = rng.uniform(0, 0.01, (nz, ny, nx))
        dn_tensor = torch.tensor(phantom, dtype=torch.float64)

        with torch.no_grad():
            intensities = model.forward(dn_tensor)
        meas_amp = torch.sqrt(intensities)

        reconstructor = SSNPReconstructor(n_iter=10, lr=50.0, tv_weight=0.0,
                                          positivity=True, device="cpu")
        _, loss_history = reconstructor.reconstruct(meas_amp, model)

        assert loss_history[-1] < loss_history[0], (
            f"Loss did not decrease: {loss_history[0]:.6f} -> {loss_history[-1]:.6f}"
        )

    def test_positivity_constraint(self):
        """With positivity=True, reconstruction should be non-negative."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        rng = np.random.default_rng(42)
        phantom = rng.uniform(0, 0.01, (nz, ny, nx))
        dn_tensor = torch.tensor(phantom, dtype=torch.float64)

        with torch.no_grad():
            intensities = model.forward(dn_tensor)
        meas_amp = torch.sqrt(intensities)

        reconstructor = SSNPReconstructor(n_iter=5, lr=50.0, tv_weight=0.0,
                                          positivity=True, device="cpu")
        dn_recon, _ = reconstructor.reconstruct(meas_amp, model)

        assert dn_recon.min() >= -1e-10, (
            f"Positivity violated: min = {dn_recon.min()}"
        )

    def test_output_dtype_float64(self):
        """Reconstruction should return float64 numpy array."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        dn_tensor = torch.zeros(nz, ny, nx, dtype=torch.float64)
        with torch.no_grad():
            intensities = model.forward(dn_tensor)
        meas_amp = torch.sqrt(intensities)

        reconstructor = SSNPReconstructor(n_iter=2, lr=10.0, tv_weight=0.0,
                                          positivity=True, device="cpu")
        dn_recon, _ = reconstructor.reconstruct(meas_amp, model)

        assert dn_recon.dtype == np.float64

    def test_tv_regularization_effect(self):
        """TV regularization should produce smoother reconstructions."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        rng = np.random.default_rng(42)
        phantom = rng.uniform(0, 0.01, (nz, ny, nx))
        dn_tensor = torch.tensor(phantom, dtype=torch.float64)

        with torch.no_grad():
            intensities = model.forward(dn_tensor)
        meas_amp = torch.sqrt(intensities)

        # Reconstruct without TV
        recon_no_tv = SSNPReconstructor(n_iter=5, lr=50.0, tv_weight=0.0,
                                         positivity=True, device="cpu")
        dn_no_tv, _ = recon_no_tv.reconstruct(meas_amp, model)

        # Reconstruct with heavy TV
        recon_tv = SSNPReconstructor(n_iter=5, lr=50.0, tv_weight=10.0,
                                      positivity=True, device="cpu")
        dn_tv, _ = recon_tv.reconstruct(meas_amp, model)

        # TV-regularized reconstruction should have smaller total variation
        tv_no_reg = tv_3d(torch.tensor(dn_no_tv)).item()
        tv_reg = tv_3d(torch.tensor(dn_tv)).item()

        assert tv_reg <= tv_no_reg, (
            f"TV regularization did not reduce TV: {tv_no_reg:.6f} vs {tv_reg:.6f}"
        )

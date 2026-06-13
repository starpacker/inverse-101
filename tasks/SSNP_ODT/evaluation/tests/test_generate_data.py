"""Tests for generate_data module."""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics_model import SSNPConfig, SSNPForwardModel
from src.generate_data import generate_measurements


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


class TestGenerateMeasurementsIntegration:
    """Integration-level tests for generate_measurements.

    Since generate_measurements requires disk files (sample.tiff, meta_data.json),
    we test the core logic it wraps: forward model on a phantom volume.
    """

    def test_forward_model_output_shape(self):
        """Simulated measurements should have shape (n_angles, Ny, Nx)."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        rng = np.random.default_rng(42)
        phantom_dn = rng.uniform(-0.005, 0.005, (nz, ny, nx))
        dn_tensor = torch.tensor(phantom_dn, dtype=torch.float64)

        with torch.no_grad():
            intensities = model.forward(dn_tensor)

        result = intensities.cpu().numpy()
        assert result.shape == (config.n_angles, ny, nx)

    def test_measurements_are_nonnegative(self):
        """Intensity measurements must be non-negative."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        rng = np.random.default_rng(7)
        phantom_dn = rng.uniform(0.0, 0.01, (nz, ny, nx))
        dn_tensor = torch.tensor(phantom_dn, dtype=torch.float64)

        with torch.no_grad():
            intensities = model.forward(dn_tensor)

        result = intensities.cpu().numpy()
        assert np.all(result >= 0), f"Negative intensity found: min={result.min()}"

    def test_measurements_dtype_float64(self):
        """Forward model should produce float64 outputs."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        dn_tensor = torch.zeros(nz, ny, nx, dtype=torch.float64)

        with torch.no_grad():
            intensities = model.forward(dn_tensor)

        assert intensities.dtype == torch.float64

    def test_zero_phantom_produces_nonzero_intensity(self):
        """Even a zero-contrast phantom transmits the incident field, giving nonzero intensity."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        dn_tensor = torch.zeros(nz, ny, nx, dtype=torch.float64)

        with torch.no_grad():
            intensities = model.forward(dn_tensor)

        result = intensities.cpu().numpy()
        assert result.sum() > 0, "Zero phantom should still produce transmitted intensity"

    def test_different_phantoms_produce_different_measurements(self):
        """Different RI distributions should yield different measurements."""
        config = _small_config()
        model = SSNPForwardModel(config, device="cpu")
        nz, ny, nx = config.volume_shape

        rng = np.random.default_rng(10)
        phantom1 = torch.tensor(rng.uniform(0, 0.01, (nz, ny, nx)), dtype=torch.float64)
        phantom2 = torch.tensor(rng.uniform(0, 0.02, (nz, ny, nx)), dtype=torch.float64)

        with torch.no_grad():
            I1 = model.forward(phantom1).cpu().numpy()
            I2 = model.forward(phantom2).cpu().numpy()

        assert not np.allclose(I1, I2), "Different phantoms should give different measurements"

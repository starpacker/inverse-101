"""Tests for the generate_data module (measurement simulation)."""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.generate_data import generate_measurements
from src.physics_model import ReflectionBPMConfig, ReflectionBPMForwardModel
from src.preprocessing import generate_phantom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_metadata():
    """Return a minimal metadata dict for a tiny phantom."""
    return {
        "volume_shape": [2, 16, 16],
        "n0": 1.5,
        "NA_obj": 0.55,
        "wavelength_um": 0.532,
        "res_um": [0.1, 0.1, 0.5],
        "ri_contrast": 0.02,
        "illumination_rings": [
            {"NA": 0.3, "n_angles": 2, "type": "BF"},
        ],
        "dz_layer": 0.5,
        "dz_gap": 10.0,
    }


def _run_small_forward():
    """Build a small model and simulate measurements directly."""
    metadata = _small_metadata()
    config = ReflectionBPMConfig.from_metadata(metadata)
    model = ReflectionBPMForwardModel(config, device="cpu")

    phantom = generate_phantom(metadata)
    dn = torch.tensor(phantom, dtype=torch.float64)

    with torch.no_grad():
        intensities = model.forward(dn)
    return intensities.cpu().numpy(), phantom, config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateMeasurementsSmall:
    """Smoke tests using a small forward pass (avoids disk I/O)."""

    def test_output_shapes(self):
        """Measurements shape must be (n_angles, Ny, Nx)."""
        meas, phantom, config = _run_small_forward()
        n_angles = config.n_angles
        nz, ny, nx = config.volume_shape
        assert meas.shape == (n_angles, ny, nx)

    def test_measurement_dtype(self):
        """Intensities should be real-valued float64."""
        meas, _, _ = _run_small_forward()
        assert meas.dtype == np.float64

    def test_intensities_nonnegative(self):
        """Intensity images must be >= 0 (|u|^2)."""
        meas, _, _ = _run_small_forward()
        assert np.all(meas >= 0)

    def test_intensities_finite(self):
        """No NaN or Inf in measurements."""
        meas, _, _ = _run_small_forward()
        assert np.all(np.isfinite(meas))

    def test_nonzero_signal(self):
        """At least some angles should produce nonzero intensity."""
        meas, _, _ = _run_small_forward()
        assert meas.max() > 0.0

    def test_phantom_matches_metadata_shape(self):
        """Phantom produced by generate_phantom should match volume_shape."""
        metadata = _small_metadata()
        phantom = generate_phantom(metadata)
        assert phantom.shape == tuple(metadata["volume_shape"])

    def test_multiple_rings(self):
        """Measurements with two rings should have correct total angle count."""
        metadata = _small_metadata()
        metadata["illumination_rings"] = [
            {"NA": 0.2, "n_angles": 2, "type": "BF"},
            {"NA": 0.6, "n_angles": 3, "type": "DF"},
        ]
        config = ReflectionBPMConfig.from_metadata(metadata)
        model = ReflectionBPMForwardModel(config, device="cpu")

        phantom = generate_phantom(metadata)
        dn = torch.tensor(phantom, dtype=torch.float64)

        with torch.no_grad():
            meas = model.forward(dn)
        assert meas.shape[0] == 5  # 2 + 3

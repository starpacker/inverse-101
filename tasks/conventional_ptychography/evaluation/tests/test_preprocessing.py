"""Tests for preprocessing module."""

import os
import sys
import json
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing import (
    PtyData,
    PtyState,
    load_experimental_data,
    setup_reconstruction,
    setup_params,
    setup_monitor,
)


def _make_synthetic_data_dir(tmp_path):
    """Create a minimal synthetic data directory with raw_data.npz and meta_data.json."""
    Nd = 16
    J = 5
    wavelength = 632.8e-9
    zo = 5e-2
    dxd = 7.2e-5
    dxp = wavelength * zo / (Nd * dxd)
    No = 64

    rng = np.random.default_rng(42)
    ptychogram = rng.uniform(0, 100, (J, Nd, Nd)).astype(np.float32)
    encoder = (rng.uniform(-5, 5, (J, 2)) * dxp).astype(np.float32)

    np.savez(os.path.join(tmp_path, "raw_data.npz"), ptychogram=ptychogram, encoder=encoder)

    meta = {
        "wavelength_m": wavelength,
        "zo_m": zo,
        "dxd_m": dxd,
        "No": No,
        "entrance_pupil_diameter_m": 2.5e-4,
        "Nd": Nd,
    }
    with open(os.path.join(tmp_path, "meta_data.json"), "w") as f:
        json.dump(meta, f)

    return Nd, J, No


class TestLoadExperimentalData:
    """Tests for loading CP datasets."""

    def test_loads_successfully(self, tmp_path):
        Nd, J, No = _make_synthetic_data_dir(str(tmp_path))
        data = load_experimental_data(str(tmp_path))
        assert isinstance(data, PtyData)
        assert data.ptychogram.shape == (J, Nd, Nd)
        assert data.encoder.shape == (J, 2)

    def test_wavelength_and_zo(self, tmp_path):
        _make_synthetic_data_dir(str(tmp_path))
        data = load_experimental_data(str(tmp_path))
        np.testing.assert_allclose(data.wavelength, 632.8e-9, rtol=1e-6)
        np.testing.assert_allclose(data.zo, 5e-2, rtol=1e-6)

    def test_energy_at_pos(self, tmp_path):
        Nd, J, _ = _make_synthetic_data_dir(str(tmp_path))
        data = load_experimental_data(str(tmp_path))
        assert data.energy_at_pos.shape == (J,)
        # energy_at_pos should be sum of each diffraction frame
        expected = np.sum(data.ptychogram, axis=(-1, -2))
        np.testing.assert_allclose(data.energy_at_pos, expected, rtol=1e-5)

    def test_max_probe_power(self, tmp_path):
        _make_synthetic_data_dir(str(tmp_path))
        data = load_experimental_data(str(tmp_path))
        expected = float(np.sqrt(np.max(data.energy_at_pos)))
        np.testing.assert_allclose(data.max_probe_power, expected, rtol=1e-6)


class TestSetupReconstruction:
    """Tests for reconstruction state initialization."""

    def test_state_shapes(self, tmp_path):
        Nd, J, No = _make_synthetic_data_dir(str(tmp_path))
        data = load_experimental_data(str(tmp_path))
        state = setup_reconstruction(data, seed=0)
        assert state.object.shape == (state.No, state.No)
        assert state.probe.shape == (state.Np, state.Np)
        assert state.positions.shape == (J, 2)

    def test_probe_is_complex(self, tmp_path):
        _make_synthetic_data_dir(str(tmp_path))
        data = load_experimental_data(str(tmp_path))
        state = setup_reconstruction(data, seed=0)
        assert np.iscomplexobj(state.probe)

    def test_object_is_complex(self, tmp_path):
        _make_synthetic_data_dir(str(tmp_path))
        data = load_experimental_data(str(tmp_path))
        state = setup_reconstruction(data, seed=0)
        assert np.iscomplexobj(state.object)

    def test_Np_equals_Nd(self, tmp_path):
        """In CP, probe size equals detector size."""
        Nd, _, _ = _make_synthetic_data_dir(str(tmp_path))
        data = load_experimental_data(str(tmp_path))
        state = setup_reconstruction(data, seed=0)
        assert state.Np == Nd

    def test_reproducible_with_seed(self, tmp_path):
        _make_synthetic_data_dir(str(tmp_path))
        data = load_experimental_data(str(tmp_path))
        s1 = setup_reconstruction(data, seed=42)
        s2 = setup_reconstruction(data, seed=42)
        np.testing.assert_allclose(s1.object, s2.object, rtol=1e-14)
        np.testing.assert_allclose(s1.probe, s2.probe, rtol=1e-14)


class TestSetupParams:
    """Tests for default parameter setup."""

    def test_returns_namespace(self):
        p = setup_params()
        assert hasattr(p, "propagatorType")
        assert p.propagatorType == "Fraunhofer"

    def test_default_switches(self):
        p = setup_params()
        assert p.probeSmoothenessSwitch == True
        assert p.probePowerCorrectionSwitch == True
        assert p.gpuSwitch == False


class TestSetupMonitor:
    """Tests for monitor namespace."""

    def test_returns_namespace(self):
        m = setup_monitor(figure_update_freq=25)
        assert m.figureUpdateFrequency == 25
        assert hasattr(m, "verboseLevel")

"""Unit tests for src/preprocessing.py."""
import numpy as np
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.preprocessing import (
    load_experimental_data, setup_reconstruction,
    setup_params, setup_monitor, save_results,
)

DATA_DIR = Path(__file__).parents[2] / "data"


# ---------------------------------------------------------------------------
# load_experimental_data
# ---------------------------------------------------------------------------

def test_load_experimental_data_shapes(fpm_data):
    assert fpm_data.ptychogram.ndim == 3
    assert fpm_data.ptychogram.shape[1] == fpm_data.Nd
    assert fpm_data.ptychogram.shape[2] == fpm_data.Nd
    assert fpm_data.encoder.shape == (fpm_data.ptychogram.shape[0], 2)


def test_load_experimental_data_dtypes(fpm_data):
    assert fpm_data.ptychogram.dtype == np.float32
    assert fpm_data.encoder.dtype == np.float64


def test_load_experimental_data_params(fpm_data):
    assert fpm_data.wavelength == pytest.approx(6.25e-7)
    assert fpm_data.NA == pytest.approx(0.1)
    assert fpm_data.magnification == pytest.approx(4.0)
    assert fpm_data.Nd == 256
    assert fpm_data.No == 730


def test_load_experimental_data_positive(fpm_data):
    """All diffraction intensities must be non-negative."""
    assert np.all(fpm_data.ptychogram >= 0)


def test_load_experimental_data_energy(fpm_data):
    """energy_at_pos must equal row-sums of ptychogram."""
    expected = fpm_data.ptychogram.sum(axis=(-1, -2))
    np.testing.assert_allclose(fpm_data.energy_at_pos, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# setup_reconstruction
# ---------------------------------------------------------------------------

def test_setup_reconstruction_shapes(fpm_data):
    state = setup_reconstruction(fpm_data, seed=42)
    assert state.object.shape == (state.No, state.No)
    assert state.probe.shape == (state.Np, state.Np)
    assert state.positions.shape == (fpm_data.ptychogram.shape[0], 2)


def test_setup_reconstruction_complex(fpm_data):
    state = setup_reconstruction(fpm_data, seed=42)
    assert np.iscomplexobj(state.probe)
    assert np.iscomplexobj(state.object)


def test_setup_reconstruction_deterministic(fpm_data):
    """Same seed must produce identical probe and object."""
    s1 = setup_reconstruction(fpm_data, seed=42)
    s2 = setup_reconstruction(fpm_data, seed=42)
    np.testing.assert_array_equal(s1.probe, s2.probe)
    np.testing.assert_array_equal(s1.object, s2.object)


def test_setup_reconstruction_different_seeds(fpm_data):
    """Different seeds must produce different probe initializations."""
    s1 = setup_reconstruction(fpm_data, seed=0)
    s2 = setup_reconstruction(fpm_data, seed=1)
    assert not np.array_equal(s1.probe, s2.probe)


def test_setup_reconstruction_probe_window(fpm_data):
    """probeWindow must be a 2D float array of the same size as the probe."""
    state = setup_reconstruction(fpm_data, seed=42)
    assert state.probeWindow.shape == (state.Np, state.Np)
    assert state.probeWindow.dtype == np.float32


# ---------------------------------------------------------------------------
# setup_params / setup_monitor
# ---------------------------------------------------------------------------

def test_setup_params_fields():
    p = setup_params()
    assert hasattr(p, "probeBoundary")
    assert p.probeBoundary is True
    assert hasattr(p, "adaptiveDenoisingSwitch")
    assert hasattr(p, "positionOrder")
    assert p.positionOrder == "NA"


def test_setup_monitor_fields():
    m = setup_monitor(figure_update_freq=25)
    assert m.figureUpdateFrequency == 25


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

def test_save_results_hdf5_keys(fpm_data):
    """save_results must write object, probe and error to HDF5."""
    import h5py
    state = setup_reconstruction(fpm_data, seed=0)
    state.error = [1.0, 0.8]
    with tempfile.NamedTemporaryFile(suffix=".hdf5") as tmp:
        save_results(state, tmp.name)
        with h5py.File(tmp.name, "r") as f:
            assert "object" in f
            assert "probe" in f
            assert "error" in f
            np.testing.assert_allclose(f["error"][:], [1.0, 0.8])


def test_save_results_shapes(fpm_data):
    """Saved arrays must have 6D shape (PtyLab-compatible)."""
    import h5py
    state = setup_reconstruction(fpm_data, seed=0)
    state.error = []
    with tempfile.NamedTemporaryFile(suffix=".hdf5") as tmp:
        save_results(state, tmp.name)
        with h5py.File(tmp.name, "r") as f:
            assert f["object"].ndim == 6
            assert f["probe"].ndim == 6

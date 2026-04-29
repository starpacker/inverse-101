"""
Unit tests for confocal-nlos-fk: src/physics_model.py and src/solvers.py
"""
import numpy as np
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / 'fixtures'

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.physics_model import nlos_forward_model
from src.preprocessing import preprocess_measurements
from src.solvers import fk_reconstruction


# ---------------------------------------------------------------------------
# nlos_forward_model
# ---------------------------------------------------------------------------

class TestForwardModel:
    def setup_method(self):
        f = np.load(FIXTURES / 'forward_model.npz')
        self.rho       = f['input_rho']
        self.wall_size = float(f['param_wall_size'])
        self.bin_res   = float(f['param_bin_res'])
        self.n_time    = int(f['param_n_time'])
        self.ref_meas  = f['output_meas']

    def test_shape(self):
        meas = nlos_forward_model(self.rho, self.wall_size, self.bin_res,
                                   n_time_bins=self.n_time)
        assert meas.shape == (self.n_time, *self.rho.shape[1:])

    def test_deterministic(self):
        meas = nlos_forward_model(self.rho, self.wall_size, self.bin_res,
                                   n_time_bins=self.n_time)
        np.testing.assert_allclose(meas, self.ref_meas, rtol=1e-10)

    def test_non_negative(self):
        meas = nlos_forward_model(self.rho, self.wall_size, self.bin_res,
                                   n_time_bins=self.n_time)
        assert (meas >= 0).all()


# ---------------------------------------------------------------------------
# preprocess_measurements
# ---------------------------------------------------------------------------

class TestPreprocess:
    def setup_method(self):
        f = np.load(FIXTURES / 'preprocess.npz')
        self.meas_store  = f['input_meas_store']
        self.bin_res     = float(f['param_bin_res'])
        self.crop        = int(f['param_crop'])
        self.ref         = f['output_processed']

    def test_shape(self):
        out = preprocess_measurements(self.meas_store, tofgrid=None,
                                       bin_resolution=self.bin_res,
                                       crop=self.crop)
        Ny, Nx, Nt = self.meas_store.shape
        assert out.shape == (min(self.crop, Nt), Ny, Nx)

    def test_deterministic(self):
        out = preprocess_measurements(self.meas_store, tofgrid=None,
                                       bin_resolution=self.bin_res,
                                       crop=self.crop)
        np.testing.assert_allclose(out, self.ref, rtol=1e-10)

    def test_no_tofgrid_preserves_values(self):
        Ny, Nx, Nt = self.meas_store.shape
        crop = min(self.crop, Nt)
        expected = self.meas_store[:, :, :crop].transpose(2, 0, 1).astype(np.float64)
        out = preprocess_measurements(self.meas_store, tofgrid=None,
                                       bin_resolution=self.bin_res, crop=self.crop)
        np.testing.assert_allclose(out, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# f-k reconstruction
# ---------------------------------------------------------------------------

class TestFK:
    def setup_method(self):
        f = np.load(FIXTURES / 'solvers.npz')
        self.meas      = f['input_meas']
        self.wall_size = float(f['param_wall_size'])
        self.bin_res   = float(f['param_bin_res'])
        self.ref       = f['output_fk']

    def test_shape(self):
        vol = fk_reconstruction(self.meas, self.wall_size, self.bin_res)
        assert vol.shape == self.meas.shape

    def test_deterministic(self):
        vol = fk_reconstruction(self.meas, self.wall_size, self.bin_res)
        np.testing.assert_allclose(vol, self.ref, rtol=1e-8)

    def test_non_negative(self):
        vol = fk_reconstruction(self.meas, self.wall_size, self.bin_res)
        assert vol.min() >= 0.0

    def test_point_scatterer_depth_recovery(self):
        """f-k should recover a single point scatterer at the correct depth."""
        Nz, N = 32, 8
        rho = np.zeros((Nz, N, N))
        depth_bin = 16
        rho[depth_bin, N // 2, N // 2] = 1.0
        meas_clean = nlos_forward_model(rho, wall_size=1.0,
                                         bin_resolution=32e-12, n_time_bins=Nz)
        vol = fk_reconstruction(meas_clean, wall_size=1.0, bin_resolution=32e-12)
        peak_z = vol.max(axis=(1, 2)).argmax()
        assert abs(peak_z - depth_bin) <= 3, \
            f"Expected peak near bin {depth_bin}, got {peak_z}"

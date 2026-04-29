"""Unit tests for src/physics_model.py."""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.physics_model import (
    forward_model, add_noise, simulate_observation,
    zero_filled_recon, data_fidelity_proximal,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/physics_model")


class TestForwardModel:
    def test_output_shape(self):
        img = np.random.rand(32, 32)
        mask = np.ones((32, 32))
        kspace = forward_model(img, mask)
        assert kspace.shape == (32, 32)
        assert np.iscomplexobj(kspace)

    def test_masking(self):
        img = np.random.rand(32, 32)
        mask = np.zeros((32, 32))
        mask[0, 0] = 1
        kspace = forward_model(img, mask)
        assert kspace[0, 0] != 0
        assert kspace[1, 1] == 0

    def test_fixture_parity(self):
        f = np.load(os.path.join(FIXTURES_DIR, "forward_model.npz"))
        kspace = forward_model(f["input_image"], f["input_mask"])
        np.testing.assert_allclose(kspace.real, f["output_kspace_real"], rtol=1e-10)
        np.testing.assert_allclose(kspace.imag, f["output_kspace_imag"], rtol=1e-10)


class TestZeroFilledRecon:
    def test_shape(self):
        y = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
        zf = zero_filled_recon(y)
        assert zf.shape == (32, 32)
        assert not np.iscomplexobj(zf)

    def test_fixture_parity(self):
        f = np.load(os.path.join(FIXTURES_DIR, "forward_model.npz"))
        kspace = f["output_kspace_real"] + 1j * f["output_kspace_imag"]
        zf = zero_filled_recon(kspace)
        np.testing.assert_allclose(zf, f["output_zerofill"], rtol=1e-10)


class TestDataFidelityProximal:
    def test_output_shape(self):
        vtilde = np.random.rand(32, 32)
        y = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
        mask = np.ones((32, 32))
        v = data_fidelity_proximal(vtilde, y, mask, alpha=2.0)
        assert v.shape == (32, 32)
        assert not np.iscomplexobj(v)

    def test_fixture_parity(self):
        f = np.load(os.path.join(FIXTURES_DIR, "proximal.npz"))
        y = f["input_y_real"] + 1j * f["input_y_imag"]
        v = data_fidelity_proximal(f["input_vtilde"], y, f["input_mask"], float(f["input_alpha"]))
        np.testing.assert_allclose(v, f["output_v"], rtol=1e-10)


class TestSimulateObservation:
    def test_noise_added(self):
        img = np.random.rand(32, 32)
        mask = np.ones((32, 32))
        noises = 0.1 * (np.random.randn(32, 32) + 1j * np.random.randn(32, 32))
        y_clean = forward_model(img, mask)
        y_noisy = simulate_observation(img, mask, noises)
        assert not np.allclose(y_clean, y_noisy)
        np.testing.assert_allclose(y_noisy, y_clean + noises, rtol=1e-10)

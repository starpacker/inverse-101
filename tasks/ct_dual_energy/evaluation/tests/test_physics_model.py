"""Tests for physics_model.py."""

import os
import sys
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src import physics_model as pm
from src.generate_data import get_attenuation_coefficients

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
DATA_DIR = os.path.join(TASK_DIR, "data")


class TestAttenuationCoefficients:
    """Tests for get_attenuation_coefficients (in generate_data.py)."""

    def test_shape_and_dtype(self):
        energies = np.arange(20, 151, dtype=np.float64)
        mus = get_attenuation_coefficients(energies)
        assert mus.shape == (2, 131)
        assert mus.dtype == np.float64

    def test_positive_values(self):
        energies = np.arange(20, 151, dtype=np.float64)
        mus = get_attenuation_coefficients(energies)
        assert np.all(mus > 0)

    def test_bone_higher_at_low_energy(self):
        """Bone has much higher MAC at low energies due to calcium."""
        energies = np.array([30.0, 40.0])
        mus = get_attenuation_coefficients(energies)
        assert np.all(mus[1] > mus[0])  # bone > tissue at low E

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "physics_model_attenuation.npz"))
        energies = fix["input_energies"]
        mus = get_attenuation_coefficients(energies)
        np.testing.assert_allclose(mus, fix["output_mus"], rtol=1e-10)

    def test_matches_raw_data(self):
        """Generated mus should match what's stored in raw_data.npz."""
        if not os.path.exists(os.path.join(DATA_DIR, "raw_data.npz")):
            pytest.skip("raw_data.npz not found")
        raw = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
        mus_stored = raw["mus"].squeeze(0)  # (2, nE)
        energies = raw["energies"].squeeze(0)  # (nE,)
        mus_computed = get_attenuation_coefficients(energies)
        np.testing.assert_allclose(mus_computed, mus_stored, rtol=1e-10)


class TestSpectra:
    """Tests for get_spectra."""

    def test_shape(self):
        energies = np.arange(20, 151, dtype=np.float64)
        spectra = pm.get_spectra(energies)
        assert spectra.shape == (2, 131)

    def test_non_negative(self):
        energies = np.arange(20, 151, dtype=np.float64)
        spectra = pm.get_spectra(energies)
        assert np.all(spectra >= 0)

    def test_low_energy_cutoff(self):
        """Low-energy spectrum should be zero above 80 keV."""
        energies = np.arange(20, 151, dtype=np.float64)
        spectra = pm.get_spectra(energies)
        high_idx = energies > 80
        assert np.all(spectra[0, high_idx] == 0.0)

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "physics_model_spectra.npz"))
        energies = fix["input_energies"]
        spectra = pm.get_spectra(energies)
        np.testing.assert_allclose(spectra, fix["output_spectra"], rtol=1e-10)


class TestRadonTransform:
    """Tests for radon_transform."""

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "physics_model_radon.npz"))
        sino = pm.radon_transform(fix["input_image"], fix["input_theta"])
        np.testing.assert_allclose(sino, fix["output_sinogram"], rtol=1e-10)

    def test_zero_image(self):
        img = np.zeros((32, 32))
        theta = np.array([0.0, 90.0])
        sino = pm.radon_transform(img, theta)
        np.testing.assert_allclose(sino, 0.0, atol=1e-15)


class TestPolychromaticForward:
    """Tests for polychromatic_forward."""

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "physics_model_forward.npz"))
        counts = pm.polychromatic_forward(
            fix["input_material_sinograms"],
            fix["input_spectra"],
            fix["input_mus"],
            dE=1.0,
        )
        np.testing.assert_allclose(counts, fix["output_counts"], rtol=1e-10)

    def test_zero_material_gives_unattenuated(self):
        """Zero material densities -> counts = sum of spectra."""
        energies = np.arange(20, 151, dtype=np.float64)
        mus = get_attenuation_coefficients(energies)
        spectra = pm.get_spectra(energies)
        zero_sinos = np.zeros((2, 3, 2))  # 2 materials, 3 bins, 2 angles
        counts = pm.polychromatic_forward(zero_sinos, spectra, mus, dE=1.0)
        expected_low = spectra[0].sum()
        expected_high = spectra[1].sum()
        np.testing.assert_allclose(counts[0], expected_low, rtol=1e-10)
        np.testing.assert_allclose(counts[1], expected_high, rtol=1e-10)

    def test_positive_counts(self):
        energies = np.arange(20, 151, dtype=np.float64)
        mus = get_attenuation_coefficients(energies)
        spectra = pm.get_spectra(energies)
        sinos = np.ones((2, 5, 3)) * 2.0
        counts = pm.polychromatic_forward(sinos, spectra, mus, dE=1.0)
        assert np.all(counts > 0)


class TestFBPReconstruct:
    """Tests for fbp_reconstruct."""

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "physics_model_fbp.npz"))
        recon = pm.fbp_reconstruct(fix["input_sinogram"], fix["input_theta"],
                                    output_size=int(fix["param_output_size"]))
        np.testing.assert_allclose(recon, fix["output_fbp"], rtol=1e-10)

    def test_round_trip(self):
        """Radon + FBP should approximately recover the original image."""
        img = np.zeros((64, 64))
        img[16:48, 16:48] = 1.0
        theta = np.linspace(0, 180, 180, endpoint=False)
        sino = pm.radon_transform(img, theta)
        recon = pm.fbp_reconstruct(sino, theta, output_size=64)
        ncc = np.dot(img.ravel(), recon.ravel()) / (
            np.linalg.norm(img.ravel()) * np.linalg.norm(recon.ravel()))
        assert ncc > 0.9

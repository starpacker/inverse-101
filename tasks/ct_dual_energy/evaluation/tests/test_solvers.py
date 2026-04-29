"""Tests for solvers.py."""

import os
import sys
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.solvers import gauss_newton_decompose, reconstruct_material_maps
from src import physics_model as pm
from src.generate_data import get_attenuation_coefficients

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")


class TestGaussNewtonDecompose:
    """Tests for gauss_newton_decompose."""

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "solvers_decompose.npz"))
        result = gauss_newton_decompose(
            fix["input_sinograms"],
            fix["input_spectra"],
            fix["input_mus"],
            n_iters=int(fix["param_n_iters"]),
            dE=1.0,
            eps=1e-6,
            verbose=False,
        )
        np.testing.assert_allclose(result, fix["output_material_sinograms"],
                                   rtol=1e-10)

    def test_output_shape(self):
        energies = np.arange(20, 151, dtype=np.float64)
        mus = get_attenuation_coefficients(energies)
        spectra = pm.get_spectra(energies)
        sinograms = np.ones((2, 5, 3)) * 500000.0  # (nMeas, nBins, nAngles)
        result = gauss_newton_decompose(sinograms, spectra, mus,
                                         n_iters=5, verbose=False)
        assert result.shape == (2, 5, 3)

    def test_non_negative(self):
        energies = np.arange(20, 151, dtype=np.float64)
        mus = get_attenuation_coefficients(energies)
        spectra = pm.get_spectra(energies)
        sinograms = np.ones((2, 5, 3)) * 500000.0
        result = gauss_newton_decompose(sinograms, spectra, mus,
                                         n_iters=10, verbose=False)
        assert np.all(result >= 0)

    def test_recovery_from_clean_data(self):
        """With noiseless data, decomposition should recover true sinograms."""
        energies = np.arange(20, 151, dtype=np.float64)
        mus = get_attenuation_coefficients(energies)
        spectra = pm.get_spectra(energies)

        # True material sinograms
        true_sinos = np.array([[[3.0, 1.0], [5.0, 2.0]],
                               [[1.0, 0.5], [2.0, 1.5]]])  # (2, 2, 2)
        counts = pm.polychromatic_forward(true_sinos, spectra, mus, dE=1.0)

        result = gauss_newton_decompose(counts, spectra, mus,
                                         n_iters=50, verbose=False)
        np.testing.assert_allclose(result, true_sinos, rtol=1e-3, atol=1e-4)


class TestReconstructMaterialMaps:
    """Tests for reconstruct_material_maps."""

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "solvers_reconstruct.npz"))
        maps = reconstruct_material_maps(
            fix["input_material_sinograms"], fix["input_theta"],
            image_size=int(fix["param_image_size"]),
            pixel_size=float(fix["param_pixel_size"]))
        np.testing.assert_allclose(maps, fix["output_maps"], rtol=1e-10)

    def test_output_shape(self):
        sino = np.random.default_rng(42).standard_normal((2, 64, 90))
        theta = np.linspace(0, 180, 90, endpoint=False)
        maps = reconstruct_material_maps(sino, theta, image_size=64,
                                          pixel_size=0.1)
        assert maps.shape == (2, 64, 64)

    def test_zero_sinogram(self):
        sino = np.zeros((2, 64, 90))
        theta = np.linspace(0, 180, 90, endpoint=False)
        maps = reconstruct_material_maps(sino, theta, image_size=64,
                                          pixel_size=0.1)
        np.testing.assert_allclose(maps, 0.0, atol=1e-10)

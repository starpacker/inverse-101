"""Tests for generate_data module."""

import pathlib
import sys

import numpy as np
import pytest

TASK_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

from src.generate_data import make_spectral_components, make_concentration_maps, generate_hsi


class TestMakeSpectralComponents:
    def test_shape(self):
        wn = np.linspace(400, 2800, 200)
        sp = make_spectral_components(wn, [1200, 1600], [300, 500])
        assert sp.shape == (2, 200)

    def test_positive(self):
        wn = np.linspace(400, 2800, 200)
        sp = make_spectral_components(wn, [1200], [300], baseline=1000)
        assert np.all(sp > 0)

    def test_peak_locations(self):
        wn = np.linspace(400, 2800, 200)
        sp = make_spectral_components(wn, [1200, 2000], [300, 300])
        # Peak of first component should be near wn=1200
        peak0 = wn[np.argmax(sp[0])]
        peak1 = wn[np.argmax(sp[1])]
        assert abs(peak0 - 1200) < 50
        assert abs(peak1 - 2000) < 50


class TestMakeConcentrationMaps:
    def test_shape(self):
        rng = np.random.RandomState(42)
        conc = make_concentration_maps(50, 100, 3, rng)
        assert conc.shape == (50, 100, 3)

    def test_sum_to_one(self):
        rng = np.random.RandomState(42)
        conc = make_concentration_maps(50, 100, 3, rng)
        np.testing.assert_allclose(conc.sum(axis=-1), 1.0, atol=1e-12)

    def test_nonnegative(self):
        rng = np.random.RandomState(42)
        conc = make_concentration_maps(50, 100, 3, rng)
        assert np.all(conc >= 0)


class TestGenerateHSI:
    def test_shapes(self):
        conc = np.random.rand(10, 20, 3)
        conc /= conc.sum(axis=-1, keepdims=True)
        sp = np.random.rand(3, 50)
        rng = np.random.RandomState(0)
        clean, noisy = generate_hsi(conc, sp, 0.1, rng)
        assert clean.shape == (200, 50)
        assert noisy.shape == (200, 50)

    def test_clean_is_bilinear(self):
        conc = np.random.rand(5, 10, 2)
        conc /= conc.sum(axis=-1, keepdims=True)
        sp = np.random.rand(2, 30)
        rng = np.random.RandomState(0)
        clean, _ = generate_hsi(conc, sp, 1.0, rng)
        expected = conc.reshape(-1, 2) @ sp
        np.testing.assert_allclose(clean, expected, atol=1e-12)

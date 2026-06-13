"""Unit tests for solvers.py."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.solvers import (
    get_wavelet_filters, _dwt1d, _idwt1d,
    wavelet_forward, wavelet_inverse, soft_thresh,
    sense_nufft_forward, sense_nufft_adjoint, sense_nufft_normal,
    estimate_max_eigenvalue, fista_l1_wavelet_nufft,
    l1wav_reconstruct_single, l1wav_reconstruct_batch,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/solvers")


# ---------------------------------------------------------------------------
# Wavelet tests
# ---------------------------------------------------------------------------

class TestWaveletFilters:
    def test_db4_length(self):
        dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters("db4")
        assert len(dec_lo) == 8

    def test_haar_length(self):
        dec_lo, _, _, _ = get_wavelet_filters("haar")
        assert len(dec_lo) == 2


class TestDWT1D:
    def test_roundtrip_db4(self):
        x = np.random.randn(32)
        dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters("db4")
        cA, cD = _dwt1d(x, dec_lo, dec_hi)
        rec = _idwt1d(cA, cD, rec_lo, rec_hi, len(x))
        np.testing.assert_allclose(rec, x, atol=1e-12)

    def test_roundtrip_complex(self):
        x = np.random.randn(32) + 1j * np.random.randn(32)
        dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters("db4")
        cA, cD = _dwt1d(x, dec_lo, dec_hi)
        rec = _idwt1d(cA, cD, rec_lo, rec_hi, len(x))
        np.testing.assert_allclose(rec, x, atol=1e-12)


class TestWaveletTransform:
    def test_roundtrip_real(self):
        x = np.random.randn(32, 32)
        coeffs, info, shape = wavelet_forward(x, "db4")
        rec = wavelet_inverse(coeffs, info, shape, "db4")
        np.testing.assert_allclose(rec, x, atol=1e-10)

    def test_roundtrip_complex(self):
        x = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
        coeffs, info, shape = wavelet_forward(x, "db4")
        rec = wavelet_inverse(coeffs, info, shape, "db4")
        np.testing.assert_allclose(rec, x, atol=1e-10)


# ---------------------------------------------------------------------------
# Soft thresholding tests
# ---------------------------------------------------------------------------

class TestSoftThresh:
    def test_shrinkage(self):
        x = np.array([1.0, -1.0])
        result = soft_thresh(0.3, x)
        np.testing.assert_allclose(result, [0.7, -0.7], atol=1e-15)

    def test_zeros_below_threshold(self):
        x = np.array([0.1, -0.05])
        result = soft_thresh(0.2, x)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-15)

    def test_complex(self):
        x = np.array([1 + 1j])
        result = soft_thresh(0.5, x)
        assert np.abs(result[0]) < np.abs(x[0])
        assert np.abs(result[0]) > 0


# ---------------------------------------------------------------------------
# SENSE-NUFFT operator tests
# ---------------------------------------------------------------------------

class TestSenseNUFFTOperators:
    def setup_method(self):
        np.random.seed(42)
        self.H, self.W, self.C = 16, 16, 4
        self.coil_maps = (np.random.randn(self.C, self.H, self.W)
                          + 1j * np.random.randn(self.C, self.H, self.W))
        n_pts = 100
        angles = np.linspace(0, np.pi, 10, endpoint=False)
        r = np.linspace(-self.H // 2, self.H // 2, n_pts // 10)
        coords = []
        for a in angles:
            for ri in r:
                coords.append([ri * np.cos(a), ri * np.sin(a)])
        self.coord = np.array(coords)

    def test_forward_shape(self):
        x = np.random.randn(self.H, self.W) + 0j
        y = sense_nufft_forward(x, self.coil_maps, self.coord)
        assert y.shape == (self.C, self.coord.shape[0])

    def test_adjoint_shape(self):
        y = np.random.randn(self.C, self.coord.shape[0]) + 0j
        x = sense_nufft_adjoint(y, self.coil_maps, self.coord, (self.H, self.W))
        assert x.shape == (self.H, self.W)

    def test_normal_shape(self):
        x = np.random.randn(self.H, self.W) + 0j
        AHAx = sense_nufft_normal(x, self.coil_maps, self.coord)
        assert AHAx.shape == (self.H, self.W)


# ---------------------------------------------------------------------------
# Power iteration test
# ---------------------------------------------------------------------------

class TestPowerIteration:
    def test_positive_eigenvalue(self):
        np.random.seed(0)
        C, H, W = 2, 8, 8
        coil_maps = np.random.randn(C, H, W) + 1j * np.random.randn(C, H, W)
        n_pts = 50
        coord = np.random.randn(n_pts, 2) * (H // 2)
        eig = estimate_max_eigenvalue(coil_maps, coord, max_iter=5)
        assert eig > 0


# ---------------------------------------------------------------------------
# Full reconstruction tests (using fixtures)
# ---------------------------------------------------------------------------

class TestL1WavReconstructSingle:
    def setup_method(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_l1wav.npz"))
        self.kdata = inp["kdata"]
        self.coord = inp["coord"]
        self.coil_maps = inp["coil_maps"]

    def test_output_shape(self):
        recon = l1wav_reconstruct_single(
            self.kdata, self.coord, self.coil_maps,
            lamda=1e-4, max_iter=5,
        )
        expected_shape = self.coil_maps.shape[1:]
        assert recon.shape == expected_shape

    def test_output_dtype(self):
        recon = l1wav_reconstruct_single(
            self.kdata, self.coord, self.coil_maps,
            lamda=1e-4, max_iter=5,
        )
        assert np.iscomplexobj(recon)

    def test_nonzero(self):
        recon = l1wav_reconstruct_single(
            self.kdata, self.coord, self.coil_maps,
            lamda=1e-4, max_iter=5,
        )
        assert np.max(np.abs(recon)) > 0


class TestL1WavReconstructBatch:
    def setup_method(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_l1wav.npz"))
        self.kdata = inp["kdata"][np.newaxis]
        self.coord = inp["coord"][np.newaxis]
        self.coil_maps = inp["coil_maps"][np.newaxis]

    def test_batch_output_shape(self):
        recons = l1wav_reconstruct_batch(
            self.kdata, self.coord, self.coil_maps,
            lamda=1e-4, max_iter=5,
        )
        assert recons.shape[0] == 1
        assert recons.shape[1:] == self.coil_maps.shape[2:]

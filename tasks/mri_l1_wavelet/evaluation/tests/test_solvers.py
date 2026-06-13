"""Unit tests for solvers.py."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.solvers import (
    get_wavelet_filters, _dwt1d, _idwt1d,
    wavelet_forward, wavelet_inverse, soft_thresh,
    sense_forward, sense_adjoint, sense_normal,
    estimate_max_eigenvalue, fista_l1_wavelet,
    l1_wavelet_reconstruct_single, l1_wavelet_reconstruct_batch,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/solvers")


# ---------------------------------------------------------------------------
# Wavelet transform tests
# ---------------------------------------------------------------------------

class TestWaveletFilters:
    def test_db4_filter_length(self):
        dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters("db4")
        assert len(dec_lo) == 8
        assert len(dec_hi) == 8
        assert len(rec_lo) == 8
        assert len(rec_hi) == 8

    def test_haar_filter_length(self):
        dec_lo, _, _, _ = get_wavelet_filters("haar")
        assert len(dec_lo) == 2

    def test_unknown_wavelet(self):
        with pytest.raises(ValueError):
            get_wavelet_filters("unknown")


class TestDWT1D:
    def test_roundtrip_db4(self):
        x = np.random.randn(32)
        dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters("db4")
        cA, cD = _dwt1d(x, dec_lo, dec_hi)
        rec = _idwt1d(cA, cD, rec_lo, rec_hi, len(x))
        np.testing.assert_allclose(rec, x, atol=1e-12)

    def test_roundtrip_haar(self):
        x = np.random.randn(16)
        dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters("haar")
        cA, cD = _dwt1d(x, dec_lo, dec_hi)
        rec = _idwt1d(cA, cD, rec_lo, rec_hi, len(x))
        np.testing.assert_allclose(rec, x, atol=1e-12)

    def test_complex_roundtrip(self):
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

    def test_odd_shape(self):
        x = np.random.randn(31, 33)
        coeffs, info, shape = wavelet_forward(x, "haar")
        rec = wavelet_inverse(coeffs, info, shape, "haar")
        np.testing.assert_allclose(rec, x, atol=1e-10)

    def test_coeffs_larger_than_input(self):
        x = np.random.randn(32, 32)
        coeffs, _, _ = wavelet_forward(x, "db4")
        assert coeffs.size >= x.size


# ---------------------------------------------------------------------------
# Soft thresholding tests
# ---------------------------------------------------------------------------

class TestSoftThresh:
    def test_zeros_below_threshold(self):
        x = np.array([0.5, -0.3, 0.1, -0.05])
        result = soft_thresh(0.4, x)
        np.testing.assert_allclose(result[2], 0.0, atol=1e-15)
        np.testing.assert_allclose(result[3], 0.0, atol=1e-15)

    def test_shrinkage(self):
        x = np.array([1.0, -1.0])
        result = soft_thresh(0.3, x)
        np.testing.assert_allclose(result, [0.7, -0.7], atol=1e-15)

    def test_complex(self):
        x = np.array([1 + 1j])
        result = soft_thresh(0.5, x)
        # |x| = sqrt(2) ~ 1.414, shrunk to ~0.914, same direction
        assert np.abs(result[0]) < np.abs(x[0])
        assert np.abs(result[0]) > 0

    def test_zero_input(self):
        x = np.array([0.0 + 0j])
        result = soft_thresh(0.5, x)
        assert result[0] == 0


# ---------------------------------------------------------------------------
# SENSE operator tests
# ---------------------------------------------------------------------------

class TestSenseOperators:
    def setup_method(self):
        np.random.seed(42)
        self.H, self.W, self.C = 16, 16, 4
        self.sens = np.random.randn(self.C, self.H, self.W) + \
                    1j * np.random.randn(self.C, self.H, self.W)
        self.mask = np.ones(self.W, dtype=np.float32)
        self.mask[::2] = 0

    def test_forward_shape(self):
        x = np.random.randn(self.H, self.W) + 0j
        y = sense_forward(x, self.sens, self.mask)
        assert y.shape == (self.C, self.H, self.W)

    def test_adjoint_shape(self):
        y = np.random.randn(self.C, self.H, self.W) + 0j
        x = sense_adjoint(y, self.sens)
        assert x.shape == (self.H, self.W)

    def test_normal_shape(self):
        x = np.random.randn(self.H, self.W) + 0j
        AHAx = sense_normal(x, self.sens, self.mask)
        assert AHAx.shape == (self.H, self.W)

    def test_masking_zeros_out(self):
        x = np.random.randn(self.H, self.W) + 0j
        y = sense_forward(x, self.sens, self.mask)
        # Unsampled columns should be zero
        assert np.allclose(y[:, :, self.mask == 0], 0)


# ---------------------------------------------------------------------------
# Power iteration test
# ---------------------------------------------------------------------------

class TestPowerIteration:
    def test_positive_eigenvalue(self):
        np.random.seed(0)
        C, H, W = 4, 16, 16
        sens = np.random.randn(C, H, W) + 1j * np.random.randn(C, H, W)
        mask = np.ones(W, dtype=np.float32)
        eig = estimate_max_eigenvalue(sens, mask, max_iter=20)
        assert eig > 0

    def test_eigenvalue_decreases_with_less_sampling(self):
        np.random.seed(0)
        C, H, W = 4, 16, 16
        sens = np.random.randn(C, H, W) + 1j * np.random.randn(C, H, W)
        mask_full = np.ones(W, dtype=np.float32)
        mask_half = np.zeros(W, dtype=np.float32)
        mask_half[::2] = 1
        eig_full = estimate_max_eigenvalue(sens, mask_full, max_iter=20)
        eig_half = estimate_max_eigenvalue(sens, mask_half, max_iter=20)
        assert eig_half < eig_full


# ---------------------------------------------------------------------------
# Full reconstruction tests
# ---------------------------------------------------------------------------

class TestL1WaveletReconstructSingle:
    def setup_method(self):
        data = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
        self.masked_kspace_0 = data["masked_kspace"][0]
        self.smaps_0 = data["sensitivity_maps"][0]
        fix = np.load(os.path.join(FIXTURE_DIR, "output_l1wav_recon_sample0.npz"))
        self.expected = fix["reconstruction"]

    def test_output_shape(self):
        recon = l1_wavelet_reconstruct_single(
            self.masked_kspace_0, self.smaps_0, lamda=1e-3, max_iter=30,
        )
        assert recon.shape == (128, 128)

    def test_output_complex(self):
        recon = l1_wavelet_reconstruct_single(
            self.masked_kspace_0, self.smaps_0, lamda=1e-3, max_iter=30,
        )
        assert np.iscomplexobj(recon)

    def test_parity_with_reference(self):
        """Our FISTA should produce results comparable to SigPy's (NCC > 0.95)."""
        recon = l1_wavelet_reconstruct_single(
            self.masked_kspace_0, self.smaps_0, lamda=1e-3, max_iter=100,
        )
        recon_mag = np.abs(recon).flatten()
        ref_mag = np.abs(self.expected).flatten()
        ncc = np.dot(recon_mag, ref_mag) / (
            np.linalg.norm(recon_mag) * np.linalg.norm(ref_mag)
        )
        assert ncc > 0.95, f"NCC too low: {ncc}"


class TestL1WaveletReconstructBatch:
    def setup_method(self):
        data = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
        self.masked_kspace = data["masked_kspace"][:1]
        self.smaps = data["sensitivity_maps"][:1]

    def test_output_shape(self):
        recons = l1_wavelet_reconstruct_batch(
            self.masked_kspace, self.smaps, lamda=1e-3, max_iter=30,
        )
        assert recons.shape == (1, 128, 128)

    def test_single_batch_consistency(self):
        single = l1_wavelet_reconstruct_single(
            self.masked_kspace[0], self.smaps[0], lamda=1e-3, max_iter=30,
        )
        batch = l1_wavelet_reconstruct_batch(
            self.masked_kspace[:1], self.smaps[:1], lamda=1e-3, max_iter=30,
        )
        s = np.abs(single).flatten()
        b = np.abs(batch[0]).flatten()
        ncc = np.dot(s, b) / (np.linalg.norm(s) * np.linalg.norm(b))
        assert ncc > 0.999, f"NCC too low: {ncc}"

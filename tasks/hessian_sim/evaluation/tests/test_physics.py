"""Tests for src.physics module."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics import generate_otf, pad_to_size, dft_conv, shift_otf, emd_decompose, compute_merit


class TestGenerateOtf:
    def test_output_shape(self):
        """OTF should have shape (n, n)."""
        otf = generate_otf(32, na=1.2, wavelength=525.0, pixel_size=65.0)
        assert otf.shape == (32, 32)

    def test_dtype_float64(self):
        """OTF should be float64."""
        otf = generate_otf(16, na=1.0, wavelength=488.0, pixel_size=65.0)
        assert otf.dtype == np.float64

    def test_normalized_to_unit_max(self):
        """OTF should be normalized so max == 1."""
        otf = generate_otf(32, na=1.2, wavelength=525.0, pixel_size=65.0)
        np.testing.assert_allclose(otf.max(), 1.0, atol=1e-12)

    def test_nonnegative(self):
        """OTF values should be >= 0."""
        otf = generate_otf(32, na=1.2, wavelength=525.0, pixel_size=65.0)
        assert np.all(otf >= 0)

    def test_center_is_max(self):
        """OTF should peak at DC (center of array)."""
        n = 32
        otf = generate_otf(n, na=1.2, wavelength=525.0, pixel_size=65.0)
        peak = np.unravel_index(np.argmax(otf), otf.shape)
        # Center should be near (n//2, n//2); the peak region includes center
        assert abs(peak[0] - n // 2) <= 1
        assert abs(peak[1] - n // 2) <= 1

    def test_higher_na_wider_support(self):
        """Higher NA should produce wider OTF support (more non-zero pixels)."""
        otf_low = generate_otf(64, na=0.5, wavelength=525.0, pixel_size=65.0)
        otf_high = generate_otf(64, na=1.4, wavelength=525.0, pixel_size=65.0)
        assert np.sum(otf_high > 0) > np.sum(otf_low > 0)


class TestPadToSize:
    def test_output_shape(self):
        """Padded array should have the target shape."""
        arr = np.ones((8, 8))
        padded = pad_to_size(arr, (16, 16))
        assert padded.shape == (16, 16)

    def test_preserves_dtype(self):
        """Padding should preserve the input dtype."""
        arr = np.ones((8, 8), dtype=np.float32)
        padded = pad_to_size(arr, (16, 16))
        assert padded.dtype == np.float32

    def test_centered_content(self):
        """Original content should be centered in the padded array."""
        arr = np.ones((4, 4)) * 5.0
        padded = pad_to_size(arr, (8, 8))
        # content at rows 2:6, cols 2:6
        np.testing.assert_array_equal(padded[2:6, 2:6], arr)
        # corners should be zero
        assert padded[0, 0] == 0.0
        assert padded[7, 7] == 0.0

    def test_total_energy_preserved(self):
        """Sum of padded array should equal sum of original."""
        rng = np.random.default_rng(99)
        arr = rng.random((10, 10))
        padded = pad_to_size(arr, (20, 20))
        np.testing.assert_allclose(padded.sum(), arr.sum(), atol=1e-12)


class TestDftConv:
    def test_output_shape(self):
        """DFT convolution output should have shape K + L - 1."""
        h = np.ones((8, 8))
        g = np.ones((6, 6))
        result = dft_conv(h, g)
        assert result.shape == (13, 13)

    def test_delta_convolution_identity(self):
        """Convolving with a centered delta should approximately reproduce input."""
        rng = np.random.default_rng(42)
        h = rng.random((8, 8))
        delta = np.zeros((1, 1))
        delta[0, 0] = 1.0
        result = dft_conv(h, delta)
        np.testing.assert_allclose(np.real(result), h, atol=1e-10)

    def test_commutativity(self):
        """Convolution should be commutative: h*g == g*h."""
        rng = np.random.default_rng(42)
        h = rng.random((8, 8))
        g = rng.random((6, 6))
        r1 = dft_conv(h, g)
        r2 = dft_conv(g, h)
        np.testing.assert_allclose(r1, r2, atol=1e-10)


class TestShiftOtf:
    def test_output_shape(self):
        """Shifted OTF should have shape (2n, 2n)."""
        n = 16
        H_2n = np.zeros((2 * n, 2 * n), dtype=complex)
        H_2n[n, n] = 1.0
        result = shift_otf(H_2n, 0.0, 0.0, n)
        assert result.shape == (2 * n, 2 * n)

    def test_zero_shift_preserves_energy(self):
        """A zero-frequency shift should preserve total spectral energy."""
        n = 16
        rng = np.random.default_rng(42)
        H_2n = rng.random((2 * n, 2 * n)).astype(complex)
        original_energy = np.sum(np.abs(H_2n) ** 2)
        shifted = shift_otf(H_2n, 0.0, 0.0, n)
        # Energy should be preserved (Parseval's theorem applied twice gives same)
        shifted_energy = np.sum(np.abs(shifted) ** 2)
        np.testing.assert_allclose(shifted_energy, original_energy, rtol=1e-6)

    def test_complex_output(self):
        """Shifted OTF should be complex."""
        n = 16
        H_2n = np.ones((2 * n, 2 * n), dtype=complex)
        result = shift_otf(H_2n, 0.1, 0.2, n)
        assert np.iscomplexobj(result)


class TestEmdDecompose:
    def test_output_covers_signal(self):
        """Sum of IMFs should approximate the original signal."""
        rng = np.random.default_rng(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 64)) + 0.5 * rng.random(64)
        imfs = emd_decompose(signal)
        reconstructed = np.sum(imfs, axis=0)
        np.testing.assert_allclose(reconstructed, signal, atol=1e-6)

    def test_imfs_shape(self):
        """Each IMF row should have same length as input."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 32))
        imfs = emd_decompose(signal)
        assert imfs.ndim == 2
        assert imfs.shape[1] == 32


class TestComputeMerit:
    def test_nonnegative(self):
        """Merit function should return a non-negative value."""
        n = 8
        H = np.ones((2 * n, 2 * n), dtype=np.float64) * 0.5
        H1 = np.ones((2 * n, 2 * n), dtype=np.float64)
        sp_center = np.ones((2 * n, 2 * n), dtype=np.complex128)
        sp_shifted = np.ones((2 * n, 2 * n), dtype=np.complex128)
        merit = compute_merit(H, H1, sp_center, sp_shifted, 0.0, 0.0, n)
        assert merit >= 0

    def test_scalar_output(self):
        """Merit function should return a scalar."""
        n = 8
        H = np.ones((2 * n, 2 * n), dtype=np.float64) * 0.5
        H1 = np.ones((2 * n, 2 * n), dtype=np.float64)
        sp_center = np.ones((2 * n, 2 * n), dtype=np.complex128)
        sp_shifted = np.ones((2 * n, 2 * n), dtype=np.complex128)
        merit = compute_merit(H, H1, sp_center, sp_shifted, 0.1, 0.1, n)
        assert np.isscalar(merit) or merit.ndim == 0

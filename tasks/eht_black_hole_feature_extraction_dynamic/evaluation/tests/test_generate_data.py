"""Unit tests for generate_data module."""

import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

from src.generate_data import (
    generate_simple_crescent_image,
    build_dft_matrix,
    compute_sefd_noise,
)


class TestGenerateSimpleCrescentImage(unittest.TestCase):
    """Tests for generate_simple_crescent_image."""

    def test_output_shape(self):
        """Output should be (npix, npix)."""
        npix = 16
        img = generate_simple_crescent_image(npix, 120.0, 44.0, 8.0, 0.5, 0.0)
        self.assertEqual(img.shape, (npix, npix))

    def test_output_dtype(self):
        """Output should be float64."""
        img = generate_simple_crescent_image(16, 120.0, 44.0, 8.0, 0.5, 0.0)
        self.assertEqual(img.dtype, np.float64)

    def test_non_negative(self):
        """All pixel values should be non-negative."""
        img = generate_simple_crescent_image(32, 120.0, 44.0, 8.0, 0.5, 45.0)
        self.assertTrue(np.all(img >= 0))

    def test_unit_sum_normalization(self):
        """Image should be approximately normalized to unit sum."""
        img = generate_simple_crescent_image(32, 200.0, 50.0, 10.0, 0.3, 90.0)
        np.testing.assert_allclose(img.sum(), 1.0, atol=1e-3)

    def test_asymmetry_effect(self):
        """Nonzero asymmetry should produce a non-symmetric brightness pattern."""
        npix = 32
        img_sym = generate_simple_crescent_image(npix, 120.0, 44.0, 8.0, 0.0, 0.0)
        img_asym = generate_simple_crescent_image(npix, 120.0, 44.0, 8.0, 0.8, 0.0)
        # Asymmetric image should differ from symmetric one
        self.assertFalse(np.allclose(img_sym, img_asym, atol=1e-6))

    def test_position_angle_rotation(self):
        """Different position angles should produce different images."""
        npix = 32
        img_pa0 = generate_simple_crescent_image(npix, 120.0, 44.0, 8.0, 0.5, 0.0)
        img_pa90 = generate_simple_crescent_image(npix, 120.0, 44.0, 8.0, 0.5, 90.0)
        self.assertFalse(np.allclose(img_pa0, img_pa90, atol=1e-6))

    def test_peak_on_ring(self):
        """The brightest pixel should lie near the expected ring radius."""
        npix = 64
        fov_uas = 200.0
        diameter_uas = 50.0
        img = generate_simple_crescent_image(
            npix, fov_uas, diameter_uas, 5.0, 0.0, 0.0)
        half_fov = 0.5 * fov_uas
        gap = 1.0 / npix
        xs = np.arange(-1 + gap, 1, 2 * gap)
        grid_y, grid_x = np.meshgrid(-xs, xs, indexing='ij')
        grid_r = np.sqrt(grid_x ** 2 + grid_y ** 2)
        peak_idx = np.unravel_index(img.argmax(), img.shape)
        peak_r_uas = grid_r[peak_idx] * half_fov
        # Peak radius should be within 2 * width of the ring radius
        np.testing.assert_allclose(peak_r_uas, diameter_uas / 2, atol=15.0)


class TestBuildDftMatrix(unittest.TestCase):
    """Tests for build_dft_matrix."""

    def test_output_shape(self):
        """DFT matrix shape should be (M, N*N)."""
        N = 8
        M = 5
        uv = np.random.randn(M, 2) * 1e9
        pixel_size_rad = 1e-10
        A = build_dft_matrix(uv, N, pixel_size_rad)
        self.assertEqual(A.shape, (M, N * N))

    def test_output_dtype(self):
        """DFT matrix should be complex."""
        uv = np.random.randn(3, 2) * 1e9
        A = build_dft_matrix(uv, 8, 1e-10)
        self.assertTrue(np.iscomplexobj(A))

    def test_unit_modulus(self):
        """Each element should have unit modulus (exp(i*phase))."""
        uv = np.random.randn(4, 2) * 1e9
        A = build_dft_matrix(uv, 8, 1e-10)
        np.testing.assert_allclose(np.abs(A), 1.0, rtol=1e-12)

    def test_zero_baseline_all_ones(self):
        """A zero-spacing baseline should give all ones (zero phase)."""
        uv = np.array([[0.0, 0.0]])
        N = 8
        A = build_dft_matrix(uv, N, 1e-10)
        np.testing.assert_allclose(np.abs(A), 1.0, rtol=1e-12)


class TestComputeSefdNoise(unittest.TestCase):
    """Tests for compute_sefd_noise."""

    def test_output_shape(self):
        """Output shape should match number of baselines."""
        station_ids = np.array([[0, 1], [0, 2], [1, 2]])
        sefds = np.array([100.0, 200.0, 300.0])
        sigma = compute_sefd_noise(station_ids, sefds)
        self.assertEqual(sigma.shape, (3,))

    def test_positive_values(self):
        """Noise values should be strictly positive."""
        station_ids = np.array([[0, 1], [0, 2]])
        sefds = np.array([100.0, 200.0, 300.0])
        sigma = compute_sefd_noise(station_ids, sefds)
        self.assertTrue(np.all(sigma > 0))

    def test_symmetric_sefd(self):
        """Equal SEFDs should give equal noise for all baselines."""
        station_ids = np.array([[0, 1], [0, 2], [1, 2]])
        sefds = np.array([1000.0, 1000.0, 1000.0])
        sigma = compute_sefd_noise(station_ids, sefds)
        np.testing.assert_allclose(sigma[0], sigma[1], rtol=1e-12)
        np.testing.assert_allclose(sigma[1], sigma[2], rtol=1e-12)

    def test_sefd_formula(self):
        """Check against the expected SEFD noise formula."""
        station_ids = np.array([[0, 1]])
        sefds = np.array([90.0, 3500.0])
        bw = 2e9
        tau = 10.0
        eta = 0.88
        expected = (1.0 / eta) * np.sqrt(90.0 * 3500.0) / np.sqrt(2.0 * bw * tau)
        sigma = compute_sefd_noise(station_ids, sefds, bandwidth_hz=bw,
                                   tau_int=tau, eta=eta)
        np.testing.assert_allclose(sigma[0], expected, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()

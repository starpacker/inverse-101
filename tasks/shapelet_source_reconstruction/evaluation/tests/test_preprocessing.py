"""Unit tests for preprocessing.py"""

import numpy as np
import pytest
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing import load_and_prepare_galaxy, decompose_shapelets, reconstruct_from_shapelets

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
NGC_PATH = os.path.join(REPO_ROOT, 'Data', 'Galaxies', 'ngc1300.jpg')
REF_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_outputs')


class TestLoadAndPrepareGalaxy:
    @pytest.fixture(autouse=True)
    def _skip_if_no_image(self):
        if not os.path.exists(NGC_PATH):
            pytest.skip("NGC1300 image not found")

    def test_output_types(self):
        ngc_square, ngc_conv, ngc_resized, numPix_resized = load_and_prepare_galaxy(NGC_PATH)
        assert isinstance(ngc_square, np.ndarray)
        assert isinstance(ngc_conv, np.ndarray)
        assert isinstance(ngc_resized, np.ndarray)
        assert isinstance(numPix_resized, int)

    def test_square_output(self):
        ngc_square, _, _, _ = load_and_prepare_galaxy(NGC_PATH)
        assert ngc_square.shape[0] == ngc_square.shape[1]

    def test_resized_is_smaller(self):
        ngc_square, _, ngc_resized, _ = load_and_prepare_galaxy(NGC_PATH)
        assert ngc_resized.shape[0] < ngc_square.shape[0]
        assert ngc_resized.shape[1] < ngc_square.shape[1]

    def test_downsample_factor(self):
        _, _, ngc_resized, numPix_resized = load_and_prepare_galaxy(NGC_PATH, downsample_factor=25)
        assert ngc_resized.shape == (numPix_resized, numPix_resized)

    def test_smoothing_reduces_variance(self):
        ngc_square, ngc_conv, _, _ = load_and_prepare_galaxy(NGC_PATH)
        assert np.std(ngc_conv) < np.std(ngc_square)

    def test_background_subtracted(self):
        """After background subtraction, the corner median should be near zero."""
        ngc_square, _, _, _ = load_and_prepare_galaxy(NGC_PATH)
        corner_median = np.median(ngc_square[:50, :50])
        assert abs(corner_median) < 5.0

    def test_matches_reference(self):
        """Output should match the saved reference data."""
        ref_path = os.path.join(REF_DIR, 'raw_data.npz')
        if not os.path.exists(ref_path):
            pytest.skip("Reference outputs not generated yet")
        ref = np.load(ref_path)
        ngc_square, ngc_conv, ngc_resized, _ = load_and_prepare_galaxy(NGC_PATH)
        np.testing.assert_allclose(ngc_square, ref['ngc_square'], rtol=1e-10)
        np.testing.assert_allclose(ngc_conv, ref['ngc_conv'], rtol=1e-10)
        np.testing.assert_allclose(ngc_resized, ref['ngc_resized'], rtol=1e-10)


class TestDecomposeShapelets:
    def test_coefficient_count(self):
        img = np.random.rand(16, 16)
        n_max = 3
        coeff = decompose_shapelets(img, n_max, beta=3.0)
        expected = (n_max + 1) * (n_max + 2) // 2
        assert len(coeff) == expected

    def test_zero_image_gives_zero_coefficients(self):
        img = np.zeros((16, 16))
        coeff = decompose_shapelets(img, n_max=3, beta=3.0)
        np.testing.assert_array_equal(coeff, 0.0)

    def test_deterministic(self):
        img = np.random.rand(16, 16)
        c1 = decompose_shapelets(img, n_max=3, beta=3.0)
        c2 = decompose_shapelets(img, n_max=3, beta=3.0)
        np.testing.assert_array_equal(c1, c2)


class TestReconstructFromShapelets:
    def test_output_shape(self):
        n_max = 3
        num_param = (n_max + 1) * (n_max + 2) // 2
        coeff = np.random.randn(num_param)
        img = reconstruct_from_shapelets(coeff, n_max, beta=3.0, numPix=16)
        assert img.shape == (16, 16)

    def test_zero_coefficients_give_zero_image(self):
        n_max = 3
        num_param = (n_max + 1) * (n_max + 2) // 2
        coeff = np.zeros(num_param)
        img = reconstruct_from_shapelets(coeff, n_max, beta=3.0, numPix=16)
        np.testing.assert_array_equal(img, 0.0)

    def test_roundtrip_with_decompose(self):
        """Decompose then reconstruct should approximate the original (large grid, low order)."""
        n_max = 2
        beta = 5.0
        numPix = 64
        num_param = (n_max + 1) * (n_max + 2) // 2
        coeff_orig = np.random.randn(num_param)
        img = reconstruct_from_shapelets(coeff_orig, n_max, beta, numPix, deltaPix=0.5)
        coeff_recov = decompose_shapelets(img, n_max, beta, deltaPix=0.5)
        np.testing.assert_allclose(coeff_recov, coeff_orig, rtol=0.02, atol=1e-3)

    def test_matches_reference(self):
        """Reconstruction from reference coefficients should match saved output."""
        ref_path = os.path.join(REF_DIR, 'raw_data.npz')
        if not os.path.exists(ref_path):
            pytest.skip("Reference outputs not generated yet")
        ref = np.load(ref_path)
        numPix = ref['ngc_resized'].shape[0]
        img = reconstruct_from_shapelets(ref['coeff_ngc'], n_max=150, beta=10, numPix=numPix)
        np.testing.assert_allclose(img, ref['image_reconstructed'], rtol=1e-10)

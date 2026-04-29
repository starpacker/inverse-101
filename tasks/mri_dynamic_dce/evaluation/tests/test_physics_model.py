"""Tests for src/physics_model.py"""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.physics_model import (
    fft2c, ifft2c, forward_single, adjoint_single,
    forward_dynamic, adjoint_dynamic, normal_operator_dynamic,
)


@pytest.fixture
def fixtures():
    return np.load(os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'physics_model.npz'))


class TestFFT:
    def test_fft2c_roundtrip(self, fixtures):
        """FFT followed by IFFT should recover the original."""
        image = fixtures['input_image']
        result = ifft2c(fft2c(image))
        np.testing.assert_allclose(result.real, image, atol=1e-10)
        np.testing.assert_allclose(result.imag, 0.0, atol=1e-10)

    def test_fft2c_values(self, fixtures):
        """FFT should match saved fixture values."""
        image = fixtures['input_image']
        expected = fixtures['output_fft2c']
        np.testing.assert_allclose(fft2c(image), expected, rtol=1e-10)

    def test_ifft2c_values(self, fixtures):
        """IFFT should match saved fixture values."""
        ksp = fixtures['output_fft2c']
        expected = fixtures['output_ifft2c']
        np.testing.assert_allclose(ifft2c(ksp), expected, rtol=1e-10)

    def test_parseval_theorem(self, fixtures):
        """Energy in image and k-space should be equal (Parseval's theorem)."""
        image = fixtures['input_image']
        ksp = fft2c(image)
        image_energy = np.sum(np.abs(image) ** 2)
        kspace_energy = np.sum(np.abs(ksp) ** 2)
        np.testing.assert_allclose(image_energy, kspace_energy, rtol=1e-10)


class TestForwardAdjoint:
    def test_forward_single_values(self, fixtures):
        image = fixtures['input_image']
        mask = fixtures['input_mask_single']
        expected = fixtures['output_forward_single']
        np.testing.assert_allclose(forward_single(image, mask), expected, rtol=1e-10)

    def test_adjoint_single_values(self, fixtures):
        ksp = fixtures['output_forward_single']
        expected = fixtures['output_adjoint_single']
        np.testing.assert_allclose(adjoint_single(ksp), expected, rtol=1e-10)

    def test_forward_adjoint_consistency(self, fixtures):
        """<Ax, y> == <x, A^H y> for random vectors."""
        rng = np.random.RandomState(42)
        N = 16
        mask = fixtures['input_mask_single']
        x = rng.randn(N, N) + 1j * rng.randn(N, N)
        y = rng.randn(N, N) + 1j * rng.randn(N, N)

        Ax = forward_single(x, mask)
        AHy = adjoint_single(y * mask)  # adjoint of masked FFT

        lhs = np.sum(Ax * np.conj(y))
        rhs = np.sum(x * np.conj(AHy))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    def test_forward_dynamic_values(self, fixtures):
        images = fixtures['input_images']
        masks = fixtures['input_masks']
        expected = fixtures['output_forward_dynamic']
        np.testing.assert_allclose(forward_dynamic(images, masks), expected, rtol=1e-10)

    def test_adjoint_dynamic_values(self, fixtures):
        ksp = fixtures['output_forward_dynamic']
        expected = fixtures['output_adjoint_dynamic']
        np.testing.assert_allclose(adjoint_dynamic(ksp), expected, rtol=1e-10)

    def test_normal_operator_values(self, fixtures):
        images = fixtures['input_images']
        masks = fixtures['input_masks']
        expected = fixtures['output_normal_dynamic']
        np.testing.assert_allclose(
            normal_operator_dynamic(images, masks), expected, rtol=1e-10)

    def test_masking_zeros_unsampled(self, fixtures):
        """Forward model should produce zeros at unsampled locations."""
        image = fixtures['input_image']
        mask = fixtures['input_mask_single']
        result = forward_single(image, mask)
        assert np.all(result[mask == 0] == 0)

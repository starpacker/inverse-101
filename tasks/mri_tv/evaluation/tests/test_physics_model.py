"""Unit tests for physics_model.py."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.physics_model import fft2c, ifft2c, forward_operator, adjoint_operator, generate_undersampling_mask

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/physics_model")


class TestFFT2c:
    def setup_method(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_fft2c.npz"))
        out = np.load(os.path.join(FIXTURE_DIR, "output_fft2c.npz"))
        self.image = inp["image"]
        self.expected_kspace = out["kspace"]

    def test_forward(self):
        result = fft2c(self.image)
        np.testing.assert_allclose(result, self.expected_kspace, rtol=1e-5)

    def test_round_trip(self):
        result = ifft2c(fft2c(self.image))
        np.testing.assert_allclose(result, self.image, rtol=1e-5)

    def test_inverse_round_trip(self):
        result = fft2c(ifft2c(self.expected_kspace))
        np.testing.assert_allclose(result, self.expected_kspace, rtol=1e-4)

    def test_parseval(self):
        """FFT with ortho norm preserves energy."""
        ksp = fft2c(self.image)
        energy_img = np.sum(np.abs(self.image) ** 2)
        energy_ksp = np.sum(np.abs(ksp) ** 2)
        np.testing.assert_allclose(energy_img, energy_ksp, rtol=1e-5)


class TestForwardOperator:
    def setup_method(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_forward.npz"))
        out = np.load(os.path.join(FIXTURE_DIR, "output_forward.npz"))
        self.image = inp["image"]
        self.smaps = inp["sensitivity_maps"]
        self.mask = inp["mask"]
        self.expected = out["masked_kspace"]

    def test_output_shape(self):
        result = forward_operator(self.image, self.smaps, self.mask)
        assert result.shape == self.expected.shape

    def test_output_values(self):
        result = forward_operator(self.image, self.smaps, self.mask)
        np.testing.assert_allclose(result, self.expected, rtol=1e-5)

    def test_mask_zeros(self):
        """Unsampled lines should be zero."""
        result = forward_operator(self.image, self.smaps, self.mask)
        zero_lines = np.where(self.mask == 0)[0]
        for line_idx in zero_lines[:5]:
            assert np.allclose(result[:, :, line_idx], 0)


class TestAdjointOperator:
    def setup_method(self):
        data = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
        self.masked_kspace = data["masked_kspace"][0]
        self.smaps = data["sensitivity_maps"][0]
        out = np.load(os.path.join(FIXTURE_DIR, "output_adjoint.npz"))
        self.expected = out["image"]

    def test_output_shape(self):
        result = adjoint_operator(self.masked_kspace, self.smaps)
        assert result.shape == (320, 320)

    def test_output_values(self):
        result = adjoint_operator(self.masked_kspace, self.smaps)
        np.testing.assert_allclose(result, self.expected, rtol=1e-5)


class TestGenerateUndersampling:
    def setup_method(self):
        out = np.load(os.path.join(FIXTURE_DIR, "output_mask.npz"))
        self.expected_mask = out["mask"]

    def test_mask_shape(self):
        mask = generate_undersampling_mask(320, 8, 0.04, "random", seed=0)
        assert mask.shape == (320,)

    def test_mask_values(self):
        mask = generate_undersampling_mask(320, 8, 0.04, "random", seed=0)
        np.testing.assert_array_equal(mask, self.expected_mask)

    def test_mask_binary(self):
        mask = generate_undersampling_mask(320, 8, 0.04, "random", seed=0)
        assert np.all((mask == 0) | (mask == 1))

    def test_n_sampled(self):
        mask = generate_undersampling_mask(320, 8, 0.04, "random", seed=0)
        assert int(mask.sum()) == 40

    def test_acs_center(self):
        """ACS lines at center should always be sampled."""
        mask = generate_undersampling_mask(320, 8, 0.04, "random", seed=0)
        acs_lines = int(np.floor(0.04 * 320))
        center_start = (320 - acs_lines) // 2
        center_end = center_start + acs_lines
        assert np.all(mask[center_start:center_end] == 1)

    def test_reproducibility(self):
        mask1 = generate_undersampling_mask(320, 8, 0.04, "random", seed=42)
        mask2 = generate_undersampling_mask(320, 8, 0.04, "random", seed=42)
        np.testing.assert_array_equal(mask1, mask2)

    def test_different_seeds(self):
        mask1 = generate_undersampling_mask(320, 8, 0.04, "random", seed=0)
        mask2 = generate_undersampling_mask(320, 8, 0.04, "random", seed=1)
        assert not np.array_equal(mask1, mask2)

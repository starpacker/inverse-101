"""Tests for solvers module."""

import os
import numpy as np
import pytest
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "solvers")

from src.solvers import ramp_filter, filter_sinogram, back_project, filtered_back_projection, circular_mask


class TestRampFilter:
    def test_shape(self):
        filt = ramp_filter(16)
        f = np.load(os.path.join(FIXTURE_DIR, "ramp_filter.npz"))
        assert filt.shape == f["output_filter"].shape

    def test_values(self):
        filt = ramp_filter(16)
        f = np.load(os.path.join(FIXTURE_DIR, "ramp_filter.npz"))
        np.testing.assert_allclose(filt, f["output_filter"], rtol=1e-10)

    def test_dc_zero(self):
        filt = ramp_filter(32)
        assert filt[0] == 0.0

    def test_non_negative(self):
        filt = ramp_filter(64)
        assert np.all(filt >= 0)


class TestFilteredBackProjection:
    @pytest.fixture
    def fixture(self):
        return np.load(os.path.join(FIXTURE_DIR, "fbp.npz"))

    def test_output_shape(self, fixture):
        result = filtered_back_projection(
            fixture["input_sinogram"],
            fixture["input_theta"],
            int(fixture["param_n_pixels"]),
        )
        assert result.shape == fixture["output_reconstruction"].shape

    def test_output_values(self, fixture):
        result = filtered_back_projection(
            fixture["input_sinogram"],
            fixture["input_theta"],
            int(fixture["param_n_pixels"]),
        )
        np.testing.assert_allclose(
            result, fixture["output_reconstruction"], rtol=1e-10
        )


class TestCircularMask:
    @pytest.fixture
    def fixture(self):
        return np.load(os.path.join(FIXTURE_DIR, "circular_mask.npz"))

    def test_output_values(self, fixture):
        result = circular_mask(
            fixture["input_image"], ratio=float(fixture["config_ratio"])
        )
        np.testing.assert_allclose(result, fixture["output_masked"], rtol=1e-10)

    def test_corners_zeroed(self):
        img = np.ones((100, 100))
        masked = circular_mask(img, ratio=0.8)
        assert masked[0, 0] == 0.0
        assert masked[0, -1] == 0.0
        assert masked[-1, 0] == 0.0
        assert masked[-1, -1] == 0.0

    def test_center_preserved(self):
        img = np.ones((100, 100))
        masked = circular_mask(img, ratio=0.95)
        assert masked[50, 50] == 1.0

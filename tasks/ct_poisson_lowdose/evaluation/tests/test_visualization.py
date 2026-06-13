"""Unit tests for visualization / metrics module."""

import os
import sys

import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.visualization import compute_ncc, compute_nrmse, centre_crop

FIXTURES_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")


@pytest.fixture(scope="module")
def fixtures():
    return dict(np.load(os.path.join(FIXTURES_DIR, "metrics_fixtures.npz")))


class TestNCC:
    def test_identical(self):
        a = np.random.RandomState(0).randn(10, 10)
        assert abs(compute_ncc(a, a) - 1.0) < 1e-10

    def test_negated(self):
        a = np.random.RandomState(0).randn(10, 10)
        assert abs(compute_ncc(a, -a) + 1.0) < 1e-10

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(compute_ncc(a, b)) < 1e-10

    def test_reference_value_uw(self, fixtures):
        ncc = compute_ncc(fixtures["input_estimate_uw"], fixtures["input_reference"])
        expected = float(fixtures["output_ncc_uw"])
        np.testing.assert_allclose(ncc, expected, rtol=1e-10)

    def test_reference_value_pwls(self, fixtures):
        ncc = compute_ncc(fixtures["input_estimate_pwls"], fixtures["input_reference"])
        expected = float(fixtures["output_ncc_pwls"])
        np.testing.assert_allclose(ncc, expected, rtol=1e-10)


class TestNRMSE:
    def test_identical(self):
        a = np.random.RandomState(0).randn(10, 10)
        assert compute_nrmse(a, a) == 0.0

    def test_positive(self):
        a = np.random.RandomState(0).randn(10, 10)
        b = a + 0.1
        assert compute_nrmse(a, b) > 0

    def test_reference_value_uw(self, fixtures):
        nrmse = compute_nrmse(fixtures["input_estimate_uw"], fixtures["input_reference"])
        expected = float(fixtures["output_nrmse_uw"])
        np.testing.assert_allclose(nrmse, expected, rtol=1e-10)

    def test_reference_value_pwls(self, fixtures):
        nrmse = compute_nrmse(fixtures["input_estimate_pwls"], fixtures["input_reference"])
        expected = float(fixtures["output_nrmse_pwls"])
        np.testing.assert_allclose(nrmse, expected, rtol=1e-10)


class TestCentreCrop:
    def test_shape(self):
        img = np.zeros((100, 100))
        cropped = centre_crop(img, 0.5)
        assert cropped.shape == (50, 50)

    def test_full_crop(self):
        img = np.random.RandomState(0).randn(100, 100)
        cropped = centre_crop(img, 1.0)
        np.testing.assert_array_equal(cropped, img)

    def test_center_values(self):
        img = np.arange(100).reshape(10, 10).astype(float)
        cropped = centre_crop(img, 0.6)
        assert cropped.shape == (6, 6)
        # Should be centered: rows 2:8, cols 2:8
        np.testing.assert_array_equal(cropped, img[2:8, 2:8])

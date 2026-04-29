"""Tests for visualization and metrics functions."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")

import sys
sys.path.insert(0, TASK_DIR)

from src.visualization import compute_ncc, compute_nrmse, compute_ssim, centre_crop


@pytest.fixture
def fixtures():
    return np.load(os.path.join(FIXTURE_DIR, "visualization_fixtures.npz"))


def test_ncc_values(fixtures):
    a = fixtures["input_a"]
    b = fixtures["input_b"]
    ncc = compute_ncc(a, b)
    np.testing.assert_allclose(ncc, float(fixtures["output_ncc"]), rtol=1e-10)


def test_ncc_identical():
    """NCC of identical arrays should be 1."""
    a = np.random.default_rng(42).random((16, 16))
    assert abs(compute_ncc(a, a) - 1.0) < 1e-10


def test_ncc_orthogonal():
    """NCC of orthogonal vectors should be 0."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(compute_ncc(a, b)) < 1e-10


def test_nrmse_values(fixtures):
    a = fixtures["input_a"]
    b = fixtures["input_b"]
    nrmse = compute_nrmse(a, b)
    np.testing.assert_allclose(nrmse, float(fixtures["output_nrmse"]), rtol=1e-10)


def test_nrmse_identical():
    """NRMSE of identical arrays should be 0."""
    a = np.random.default_rng(42).random((16, 16))
    assert compute_nrmse(a, a) == 0.0


def test_ssim_identical():
    """SSIM of identical arrays should be 1."""
    a = np.random.default_rng(42).random((32, 32))
    ssim = compute_ssim(a, a)
    assert abs(ssim - 1.0) < 1e-6


def test_centre_crop_shape(fixtures):
    a = fixtures["input_a"]
    cropped = centre_crop(a, 0.5)
    np.testing.assert_array_equal(cropped, fixtures["output_crop"])


def test_centre_crop_preserves_center():
    """Centre crop should keep the center pixel."""
    a = np.zeros((32, 32))
    a[16, 16] = 1.0
    cropped = centre_crop(a, 0.5)
    assert cropped[8, 8] == 1.0

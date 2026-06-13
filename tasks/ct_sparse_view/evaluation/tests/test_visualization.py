"""Tests for visualization and metrics module."""

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


def test_ncc_identical():
    """NCC of identical images should be 1."""
    x = np.random.rand(32, 32)
    assert abs(compute_ncc(x, x) - 1.0) < 1e-10


def test_ncc_scaled():
    """NCC should be 1 for positively scaled copies."""
    x = np.random.rand(32, 32) + 0.1
    assert abs(compute_ncc(2.0 * x, x) - 1.0) < 1e-10


def test_ncc_range():
    """NCC should be in [-1, 1]."""
    rng = np.random.default_rng(42)
    a = rng.random((32, 32))
    b = rng.random((32, 32))
    ncc = compute_ncc(a, b)
    assert -1.0 <= ncc <= 1.0


def test_ncc_fixture(fixtures):
    """NCC must match fixture value."""
    est = fixtures["input_estimate"]
    ref = fixtures["input_reference"]
    expected = float(fixtures["output_ncc"])
    actual = compute_ncc(est, ref)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_nrmse_identical():
    """NRMSE of identical images should be 0."""
    x = np.random.rand(32, 32)
    assert compute_nrmse(x, x) == 0.0


def test_nrmse_positive():
    """NRMSE should be non-negative."""
    rng = np.random.default_rng(42)
    a = rng.random((32, 32))
    b = rng.random((32, 32))
    assert compute_nrmse(a, b) >= 0.0


def test_nrmse_fixture(fixtures):
    """NRMSE must match fixture value."""
    est = fixtures["input_estimate"]
    ref = fixtures["input_reference"]
    expected = float(fixtures["output_nrmse"])
    actual = compute_nrmse(est, ref)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_ssim_identical():
    """SSIM of identical images should be ~1."""
    x = np.random.rand(32, 32)
    ssim = compute_ssim(x, x)
    assert ssim > 0.99


def test_centre_crop_shape():
    x = np.random.rand(100, 100)
    cropped = centre_crop(x, 0.8)
    assert cropped.shape == (80, 80)


def test_centre_crop_identity():
    """Crop fraction 1.0 should return the full image."""
    x = np.random.rand(100, 100)
    cropped = centre_crop(x, 1.0)
    assert cropped.shape == (100, 100)

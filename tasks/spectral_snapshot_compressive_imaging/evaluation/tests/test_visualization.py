"""Unit tests for src/visualization.py."""

import numpy as np
import pytest

from src.visualization import psnr, ssim, calculate_ssim


def test_psnr_identical():
    """PSNR of identical images should be 100."""
    x = np.random.rand(32, 32)
    assert psnr(x, x) == 100


def test_psnr_positive():
    """PSNR should be positive for similar images."""
    x = np.random.rand(32, 32)
    y = x + np.random.randn(32, 32) * 0.01
    assert psnr(x, y) > 0


def test_psnr_noise_monotonic():
    """More noise should give lower PSNR."""
    x = np.random.rand(32, 32)
    y_low = x + np.random.randn(32, 32) * 0.01
    y_high = x + np.random.randn(32, 32) * 0.1
    assert psnr(x, y_low) > psnr(x, y_high)


def test_ssim_identical():
    """SSIM of identical images should be ~1."""
    x = np.random.rand(32, 32).astype(np.float64)
    result = ssim(x, x)
    np.testing.assert_allclose(result, 1.0, atol=1e-6)


def test_calculate_ssim_multichannel():
    """calculate_ssim should work on 3D arrays."""
    x = np.random.rand(32, 32, 5)
    result = calculate_ssim(x, x)
    np.testing.assert_allclose(result, 1.0, atol=1e-6)


def test_calculate_ssim_shape_mismatch():
    """Should raise ValueError for mismatched shapes."""
    x = np.random.rand(32, 32, 3)
    y = np.random.rand(32, 32, 5)
    with pytest.raises(ValueError):
        calculate_ssim(x, y)

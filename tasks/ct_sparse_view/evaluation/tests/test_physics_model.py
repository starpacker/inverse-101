"""Tests for the CT physics model (Radon transform, FBP, noise)."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")

import sys
sys.path.insert(0, TASK_DIR)

from src.physics_model import radon_transform, filtered_back_projection, add_gaussian_noise


@pytest.fixture
def fixtures():
    return np.load(os.path.join(FIXTURE_DIR, "physics_model_fixtures.npz"))


def test_radon_transform_shape(fixtures):
    image = fixtures["input_image"]
    angles = fixtures["input_angles"]
    sino = radon_transform(image, angles)
    assert sino.shape == (32, 10)


def test_radon_transform_deterministic(fixtures):
    """Radon transform must be deterministic."""
    image = fixtures["input_image"]
    angles = fixtures["input_angles"]
    sino1 = radon_transform(image, angles)
    sino2 = radon_transform(image, angles)
    np.testing.assert_array_equal(sino1, sino2)


def test_radon_transform_values(fixtures):
    """Radon transform must match fixture values exactly."""
    image = fixtures["input_image"]
    angles = fixtures["input_angles"]
    sino = radon_transform(image, angles)
    np.testing.assert_allclose(sino, fixtures["output_sinogram"], rtol=1e-10)


def test_fbp_shape(fixtures):
    sino = fixtures["output_sinogram"]
    angles = fixtures["input_angles"]
    fbp = filtered_back_projection(sino, angles, output_size=32)
    assert fbp.shape == (32, 32)


def test_fbp_values(fixtures):
    """FBP must match fixture values exactly."""
    sino = fixtures["output_sinogram"]
    angles = fixtures["input_angles"]
    fbp = filtered_back_projection(sino, angles, output_size=32)
    np.testing.assert_allclose(fbp, fixtures["output_fbp"], rtol=1e-10)


def test_radon_fbp_roundtrip():
    """Full-angle Radon+FBP should approximately recover a simple phantom."""
    # Create a simple disc phantom
    N = 64
    x, y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    phantom = ((x**2 + y**2) < 0.5**2).astype(float)

    angles = np.linspace(0, 180, 180, endpoint=False)
    sino = radon_transform(phantom, angles)
    recon = filtered_back_projection(sino, angles, output_size=N)

    # With 180 full angles, FBP should be reasonable
    from src.visualization import compute_ncc
    ncc = compute_ncc(recon, phantom)
    assert ncc > 0.9, f"FBP roundtrip NCC too low: {ncc}"


def test_add_noise_reproducible():
    """Noise with same RNG seed should be identical."""
    sino = np.ones((32, 10))
    n1 = add_gaussian_noise(sino, 0.05, rng=np.random.default_rng(123))
    n2 = add_gaussian_noise(sino, 0.05, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(n1, n2)


def test_add_noise_changes_sinogram():
    """Noise should actually change the sinogram."""
    sino = np.ones((32, 10))
    noisy = add_gaussian_noise(sino, 0.05, rng=np.random.default_rng(0))
    assert not np.allclose(sino, noisy)


def test_add_noise_statistics():
    """Noise should have approximately correct standard deviation."""
    sino = np.ones((256, 100))
    rng = np.random.default_rng(42)
    noisy = add_gaussian_noise(sino, 0.05, rng=rng)
    noise = noisy - sino
    expected_std = 0.05 * np.max(np.abs(sino))
    actual_std = np.std(noise)
    np.testing.assert_allclose(actual_std, expected_std, rtol=0.05)

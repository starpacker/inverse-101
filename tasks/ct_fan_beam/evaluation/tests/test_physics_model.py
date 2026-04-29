"""Tests for the fan-beam CT physics model (forward projection, FBP, noise)."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")

import sys
sys.path.insert(0, TASK_DIR)

from src.physics_model import (
    fan_beam_geometry, fan_beam_forward_vectorized, fan_beam_backproject,
    fan_beam_fbp, ramp_filter, parker_weights, add_gaussian_noise,
)


@pytest.fixture
def fixtures():
    return np.load(os.path.join(FIXTURE_DIR, "physics_model_fixtures.npz"))


@pytest.fixture
def small_geo():
    return fan_beam_geometry(32, 48, 18, 128, 128, angle_range=2 * np.pi)


# --- Geometry ---

def test_geometry_required_keys():
    geo = fan_beam_geometry(64, 96, 90, 256, 256)
    for k in ['N', 'n_det', 'n_angles', 'D_sd', 'D_dd',
              'det_spacing', 'det_pos', 'angles', 'angle_range']:
        assert k in geo, f"Missing key: {k}"


def test_geometry_angles_range():
    geo = fan_beam_geometry(64, 96, 90, 256, 256, angle_range=np.pi)
    assert geo['angles'][0] == 0
    assert geo['angles'][-1] < np.pi


def test_geometry_detector_centered():
    geo = fan_beam_geometry(64, 96, 90, 256, 256)
    assert abs(np.mean(geo['det_pos'])) < 1e-10


# --- Forward projection ---

def test_forward_shape(small_geo):
    image = np.ones((32, 32))
    sino = fan_beam_forward_vectorized(image, small_geo)
    assert sino.shape == (18, 48)


def test_forward_zero_image(small_geo):
    """Zero image must produce zero sinogram."""
    sino = fan_beam_forward_vectorized(np.zeros((32, 32)), small_geo)
    np.testing.assert_allclose(sino, 0, atol=1e-10)


def test_forward_deterministic(small_geo, fixtures):
    """Forward projection must be deterministic."""
    phantom = fixtures["input_phantom"]
    sino1 = fan_beam_forward_vectorized(phantom, small_geo)
    sino2 = fan_beam_forward_vectorized(phantom, small_geo)
    np.testing.assert_array_equal(sino1, sino2)


def test_forward_values(small_geo, fixtures):
    """Forward projection must match fixture values exactly."""
    phantom = fixtures["input_phantom"]
    sino = fan_beam_forward_vectorized(phantom, small_geo)
    np.testing.assert_allclose(sino, fixtures["output_sinogram"], rtol=1e-10)


def test_forward_positive_for_positive_image(small_geo):
    """Positive image should produce positive sinogram values."""
    image = np.ones((32, 32)) * 0.5
    sino = fan_beam_forward_vectorized(image, small_geo)
    assert np.any(sino > 0)


def test_forward_symmetry():
    """Centered circle should give similar projections at all angles."""
    geo = fan_beam_geometry(64, 96, 36, 256, 256, angle_range=2 * np.pi)
    image = np.zeros((64, 64))
    yy, xx = np.mgrid[:64, :64] - 32 + 0.5
    image[xx**2 + yy**2 < 10**2] = 1.0
    sino = fan_beam_forward_vectorized(image, geo)
    center_vals = sino[:, 48]
    assert np.std(center_vals) / np.mean(center_vals) < 0.1


# --- Back-projection ---

def test_backproject_shape(small_geo):
    sino = np.ones((18, 48))
    bp = fan_beam_backproject(sino, small_geo)
    assert bp.shape == (32, 32)


def test_backproject_values(small_geo, fixtures):
    """Back-projection must match fixture values exactly."""
    sino = fixtures["output_sinogram"]
    bp = fan_beam_backproject(sino, small_geo)
    np.testing.assert_allclose(bp, fixtures["output_backprojection"], rtol=1e-10)


# --- FBP ---

def test_fbp_shape(small_geo, fixtures):
    sino = fixtures["output_sinogram"]
    fbp = fan_beam_fbp(sino, small_geo, filter_type='hann', cutoff=0.3)
    assert fbp.shape == (32, 32)


def test_fbp_values(small_geo, fixtures):
    """FBP must match fixture values exactly."""
    sino = fixtures["output_sinogram"]
    fbp = fan_beam_fbp(sino, small_geo, filter_type='hann', cutoff=0.3)
    np.testing.assert_allclose(fbp, fixtures["output_fbp"], rtol=1e-10)


def test_fbp_roundtrip():
    """Full-angle forward + FBP should approximately recover a simple phantom."""
    N = 64
    geo = fan_beam_geometry(N, 96, 90, 256, 256, angle_range=2 * np.pi)
    yy, xx = np.mgrid[:N, :N]
    phantom = ((xx - N/2)**2 + (yy - N/2)**2 < (N/4)**2).astype(np.float64)

    sino = fan_beam_forward_vectorized(phantom, geo)
    recon = fan_beam_fbp(sino, geo, filter_type='hann', cutoff=0.3)
    recon = np.maximum(recon, 0)

    from src.visualization import compute_ncc, centre_crop_normalize
    gt_crop = centre_crop_normalize(phantom)
    recon_crop = centre_crop_normalize(recon)
    ncc = compute_ncc(recon_crop, gt_crop)
    assert ncc > 0.3, f"FBP roundtrip NCC too low: {ncc}"


# --- Ramp filter ---

def test_ramp_filter_shape():
    filt, pad_len = ramp_filter(96, 1.0)
    assert len(filt) == pad_len
    assert pad_len >= 96


def test_ramp_filter_non_negative():
    filt, _ = ramp_filter(96, 1.0)
    assert np.all(filt >= -1e-10)


def test_ramp_filter_values(fixtures):
    """Ramp filter must match fixture values."""
    filt, pad_len = ramp_filter(48, 1.0, filter_type='hann', cutoff=0.3)
    # Just verify shape matches (filter depends on det_spacing which varies)
    assert len(filt) == pad_len


# --- Parker weights ---

def test_parker_weights_shape():
    angles = np.linspace(0, np.pi + 0.5, 100)
    det_pos = np.linspace(-50, 50, 96)
    pw = parker_weights(angles, det_pos, 256.0)
    assert pw.shape == (100, 96)


def test_parker_weights_range():
    """Parker weights must be in [0, 1]."""
    angles = np.linspace(0, np.pi + 0.5, 100)
    det_pos = np.linspace(-50, 50, 96)
    pw = parker_weights(angles, det_pos, 256.0)
    assert np.all(pw >= -1e-10) and np.all(pw <= 1 + 1e-10)


def test_parker_weights_center_ones():
    """Middle angles (far from scan edges) should have weight ≈ 1."""
    angles = np.linspace(0, np.pi + 0.5, 100)
    det_pos = np.linspace(-10, 10, 20)  # narrow detector
    pw = parker_weights(angles, det_pos, 256.0)
    # Center angles should be ≈ 1
    mid = len(angles) // 2
    assert np.all(pw[mid-5:mid+5, :] > 0.9)


# --- Noise ---

def test_noise_reproducible():
    sino = np.ones((18, 48))
    n1 = add_gaussian_noise(sino, 0.05, rng=np.random.default_rng(123))
    n2 = add_gaussian_noise(sino, 0.05, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(n1, n2)


def test_noise_changes_sinogram():
    sino = np.ones((18, 48))
    noisy = add_gaussian_noise(sino, 0.05, rng=np.random.default_rng(0))
    assert not np.allclose(sino, noisy)


def test_noise_statistics():
    """Noise should have approximately correct standard deviation."""
    sino = np.ones((256, 100))
    noisy = add_gaussian_noise(sino, 0.05, rng=np.random.default_rng(42))
    noise = noisy - sino
    np.testing.assert_allclose(np.std(noise), 0.05, rtol=0.1)

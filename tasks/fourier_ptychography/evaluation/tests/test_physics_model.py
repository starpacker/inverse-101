"""Unit tests for src/physics_model.py."""
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.physics_model import (
    compute_pupil_mask, compute_kspace_shift,
    fpm_forward_single, forward_model_stack,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# compute_pupil_mask
# ---------------------------------------------------------------------------

def test_compute_pupil_mask_deterministic():
    """compute_pupil_mask must return the saved fixture value."""
    fix = np.load(FIXTURES / "compute_pupil_mask.npz")
    result = compute_pupil_mask(
        int(fix["Nd"]), float(fix["dxp"]), float(fix["wavelength"]), float(fix["NA"])
    )
    np.testing.assert_array_equal(result, fix["output"])


def test_compute_pupil_mask_shape():
    pupil = compute_pupil_mask(Nd=64, dxp=1.625e-6, wavelength=625e-9, NA=0.1)
    assert pupil.shape == (64, 64)
    assert pupil.dtype == np.float32


def test_compute_pupil_mask_binary():
    """Pupil must be binary (only 0.0 and 1.0)."""
    pupil = compute_pupil_mask(64, 1.625e-6, 625e-9, 0.1)
    unique = np.unique(pupil)
    assert set(unique).issubset({0.0, 1.0})


def test_compute_pupil_mask_dc_inside():
    """DC (center pixel) must always be inside the pupil for NA > 0."""
    pupil = compute_pupil_mask(64, 1.625e-6, 625e-9, NA=0.1)
    assert pupil[32, 32] == 1.0


def test_compute_pupil_mask_larger_na_more_pixels():
    """A larger NA must pass more pixels."""
    p_small = compute_pupil_mask(64, 1.625e-6, 625e-9, NA=0.05)
    p_large = compute_pupil_mask(64, 1.625e-6, 625e-9, NA=0.2)
    assert p_large.sum() > p_small.sum()


# ---------------------------------------------------------------------------
# compute_kspace_shift
# ---------------------------------------------------------------------------

def test_compute_kspace_shift_deterministic():
    """compute_kspace_shift must return the saved fixture value."""
    fix = np.load(FIXTURES / "compute_kspace_shift.npz")
    result = compute_kspace_shift(
        fix["led_pos"], float(fix["z_led"]),
        float(fix["wavelength"]), int(fix["Nd"]), float(fix["dxp"])
    )
    np.testing.assert_allclose(result, fix["output"], rtol=1e-10)


def test_compute_kspace_shift_onaxis_is_zero():
    """On-axis LED must produce zero k-space shift."""
    shift = compute_kspace_shift(
        np.array([0.0, 0.0]), z_led=60e-3,
        wavelength=625e-9, Nd=256, dxp=1.625e-6
    )
    np.testing.assert_allclose(shift, [0.0, 0.0], atol=1e-12)


def test_compute_kspace_shift_formula():
    """Verify shift = encoder / z_led * dxp / wavelength * Nd (paraxial)."""
    led = np.array([4e-3, -2e-3])
    z, wl, Nd, dxp = 60e-3, 625e-9, 256, 1.625e-6
    expected = led / z * dxp / wl * Nd
    result = compute_kspace_shift(led, z, wl, Nd, dxp)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# fpm_forward_single
# ---------------------------------------------------------------------------

def test_fpm_forward_single_deterministic():
    """fpm_forward_single must return the saved fixture image."""
    fix = np.load(FIXTURES / "fpm_forward_single.npz")
    img, _ = fpm_forward_single(fix["obj_spectrum"], fix["pupil"],
                                 np.array([0.0, 0.0]), 32)
    np.testing.assert_allclose(img, fix["output_image"], rtol=1e-10)


def test_fpm_forward_single_shape():
    Nd = 32
    rng = np.random.default_rng(0)
    obj = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
    pupil = compute_pupil_mask(Nd, 1.625e-6, 625e-9, 0.1)
    img, field = fpm_forward_single(obj, pupil, np.array([0.0, 0.0]), Nd)
    assert img.shape == (Nd, Nd)
    assert field.shape == (Nd, Nd)


def test_fpm_forward_single_nonnegative():
    """Intensity image must be non-negative."""
    Nd = 32
    rng = np.random.default_rng(0)
    obj = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
    pupil = compute_pupil_mask(Nd, 1.625e-6, 625e-9, 0.1)
    img, _ = fpm_forward_single(obj, pupil, np.array([0.0, 0.0]), Nd)
    assert np.all(img >= 0)


# ---------------------------------------------------------------------------
# forward_model_stack
# ---------------------------------------------------------------------------

def test_forward_model_stack_shape():
    """forward_model_stack must return (J, Nd, Nd) float32."""
    No, Nd, J = 128, 32, 5
    rng = np.random.default_rng(0)
    obj = rng.standard_normal((No, No)) + 1j * rng.standard_normal((No, No))
    pupil = compute_pupil_mask(Nd, 1.625e-6, 625e-9, 0.1)
    encoder = np.zeros((J, 2))
    stack = forward_model_stack(obj, pupil, encoder, 60e-3, 625e-9, 1.625e-6, Nd)
    assert stack.shape == (J, Nd, Nd)
    assert stack.dtype == np.float32


def test_forward_model_stack_nonnegative():
    rng = np.random.default_rng(0)
    obj = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
    pupil = compute_pupil_mask(32, 1.625e-6, 625e-9, 0.1)
    encoder = np.zeros((3, 2))
    stack = forward_model_stack(obj, pupil, encoder, 60e-3, 625e-9, 1.625e-6, 32)
    assert np.all(stack >= 0)

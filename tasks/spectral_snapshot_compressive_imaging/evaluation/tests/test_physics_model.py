"""Unit tests for src/physics_model.py."""

import os
import numpy as np
import pytest

from src.physics_model import A, At, shift, shift_back

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'physics_model_fixtures.npz')


@pytest.fixture
def fixtures():
    return dict(np.load(FIXTURES_PATH, allow_pickle=True))


def test_A_output_shape(fixtures):
    """Forward model should reduce 3D cube to 2D measurement."""
    mask_patch = fixtures['mask_patch']
    truth_patch = fixtures['truth_patch']
    step = int(fixtures['step'])
    nC = int(fixtures['nC'])
    shifted = shift(truth_patch, step=step)
    result = A(shifted, mask_patch)
    assert result.shape == (truth_patch.shape[0], truth_patch.shape[1] + step * (nC - 1))


def test_At_output_shape(fixtures):
    """Transpose should produce 3D cube from 2D measurement."""
    meas_patch = fixtures['meas_patch']
    mask_patch = fixtures['mask_patch']
    nC = int(fixtures['nC'])
    result = At(meas_patch, mask_patch)
    assert result.shape == mask_patch.shape
    assert result.shape[2] == nC


def test_A_At_adjoint(fixtures):
    """<A(x), y> should equal <x, At(y)> (adjoint property)."""
    mask_patch = fixtures['mask_patch']
    nC = int(fixtures['nC'])
    step = int(fixtures['step'])
    patch_size = int(fixtures['patch_size'])
    np.random.seed(42)
    x = np.random.randn(patch_size, patch_size + step * (nC - 1), nC)
    y = np.random.randn(patch_size, patch_size + step * (nC - 1))
    lhs = np.sum(A(x, mask_patch) * y)
    rhs = np.sum(x * At(y, mask_patch))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_shift_back_inverts_shift(fixtures):
    """shift_back(shift(x)) should recover x."""
    truth_patch = fixtures['truth_patch']
    step = int(fixtures['step'])
    shifted = shift(truth_patch, step=step)
    recovered = shift_back(shifted.copy(), step=step)
    np.testing.assert_allclose(recovered, truth_patch, rtol=1e-10)


def test_shift_output_shape(fixtures):
    """shift should expand spatial dimension."""
    truth_patch = fixtures['truth_patch']
    step = int(fixtures['step'])
    nC = int(fixtures['nC'])
    shifted = shift(truth_patch, step=step)
    assert shifted.shape == (truth_patch.shape[0],
                              truth_patch.shape[1] + (nC - 1) * step,
                              nC)


def test_A_exact_values(fixtures):
    """Forward model should match pre-computed fixture values."""
    meas_patch = fixtures['meas_patch']
    mask_patch = fixtures['mask_patch']
    truth_patch = fixtures['truth_patch']
    step = int(fixtures['step'])
    shifted = shift(truth_patch, step=step)
    result = A(shifted, mask_patch)
    np.testing.assert_allclose(result, meas_patch, rtol=1e-10)


def test_At_exact_values(fixtures):
    """Transpose should match pre-computed fixture values."""
    meas_patch = fixtures['meas_patch']
    mask_patch = fixtures['mask_patch']
    back_proj = fixtures['back_proj']
    result = At(meas_patch, mask_patch)
    np.testing.assert_allclose(result, back_proj, rtol=1e-10)

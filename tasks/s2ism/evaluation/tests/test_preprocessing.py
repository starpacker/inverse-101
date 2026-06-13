"""Unit tests for src/preprocessing.py"""

import numpy as np
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / 'fixtures'


def test_rotation_matrix():
    from src.preprocessing import rotation_matrix
    fix = np.load(FIXTURES / 'rotation_matrix.npz')
    theta = fix['input_theta'][0]
    expected = fix['output_matrix']
    result = rotation_matrix(theta)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_mirror_matrix():
    from src.preprocessing import mirror_matrix
    fix = np.load(FIXTURES / 'mirror_matrix.npz')
    alpha = fix['input_alpha'][0]
    expected = fix['output_matrix']
    result = mirror_matrix(alpha)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_shift_matrix_rect():
    from src.preprocessing import shift_matrix
    fix = np.load(FIXTURES / 'shift_matrix.npz')
    expected = fix['output_rect']
    result = shift_matrix('rect')
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_transform_shift_vectors():
    from src.preprocessing import transform_shift_vectors
    fix = np.load(FIXTURES / 'transform_shift_vectors.npz')
    param = fix['input_param'].tolist()
    shift_in = fix['input_shift']
    expected = fix['output_shift']
    result = transform_shift_vectors(param, shift_in)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_loss_shifts():
    from src.preprocessing import loss_shifts
    fix = np.load(FIXTURES / 'loss_shifts.npz')
    x0 = fix['input_x0'].tolist()
    shift_exp = fix['input_shift_exp']
    shift_theor = fix['input_shift_theor']
    mirror = fix['input_mirror'][0]
    expected = fix['output_loss'][0]
    result = loss_shifts(x0, shift_exp, shift_theor, mirror)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_scalar_psf():
    from src.preprocessing import scalar_psf
    fix = np.load(FIXTURES / 'scalar_psf.npz')
    r = fix['input_r']
    wl = fix['input_wl'][0]
    na = fix['input_na'][0]
    expected = fix['output_psf']
    result = scalar_psf(r, wl, na)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_rect_function():
    from src.preprocessing import rect
    fix = np.load(FIXTURES / 'rect_func.npz')
    r = fix['input_r']
    d = fix['input_d'][0]
    expected = fix['output_rect']
    result = rect(r, d)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_gaussian_2d():
    from src.preprocessing import gaussian_2d
    fix = np.load(FIXTURES / 'gaussian_2d.npz')
    params = fix['input_params'].tolist()
    x = fix['input_x']
    y = fix['input_y']
    expected = fix['output_gaussian']
    result = gaussian_2d(params, x, y)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_crop_shift_rect_shape():
    from src.preprocessing import crop_shift
    shift = np.random.randn(25, 2)
    result = crop_shift(shift, 'rect')
    assert result.shape == (9, 2)


def test_rotation_matrix_orthogonal():
    from src.preprocessing import rotation_matrix
    R = rotation_matrix(0.7)
    np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

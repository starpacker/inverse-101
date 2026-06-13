"""Unit tests for src/preprocessing.py."""

import os
import numpy as np
import pytest

from src.preprocessing import load_meta_data, load_mask, load_ground_truth, generate_measurement

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
FIXTURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'preprocessing_fixtures.npz')


@pytest.fixture
def fixtures():
    return dict(np.load(FIXTURES_PATH, allow_pickle=True))


@pytest.fixture
def meta():
    return load_meta_data(os.path.join(DATA_DIR, 'meta_data.json'))


def test_load_meta_data(meta):
    assert meta['r'] == 256
    assert meta['c'] == 256
    assert meta['nC'] == 31
    assert meta['step'] == 1
    assert meta['wavelength_start_nm'] == 400
    assert meta['wavelength_end_nm'] == 700


def test_load_mask_shape(meta):
    r, c, nC, step = meta['r'], meta['c'], meta['nC'], meta['step']
    mask_3d = load_mask(os.path.join(DATA_DIR, 'mask256.mat'), r, c, nC, step)
    assert mask_3d.shape == (r, c + step * (nC - 1), nC)


def test_load_ground_truth_shape():
    truth = load_ground_truth(os.path.join(DATA_DIR, 'kaist_crop256_01.mat'))
    assert truth.shape == (256, 256, 31)


def test_generate_measurement_shape(fixtures, meta):
    r, c, nC, step = meta['r'], meta['c'], meta['nC'], meta['step']
    truth = load_ground_truth(os.path.join(DATA_DIR, 'kaist_crop256_01.mat'))
    mask_3d = load_mask(os.path.join(DATA_DIR, 'mask256.mat'), r, c, nC, step)
    meas = generate_measurement(truth, mask_3d, step)
    expected_shape = tuple(fixtures['meas_shape'])
    assert meas.shape == expected_shape


def test_generate_measurement_values(fixtures, meta):
    r, c, nC, step = meta['r'], meta['c'], meta['nC'], meta['step']
    truth = load_ground_truth(os.path.join(DATA_DIR, 'kaist_crop256_01.mat'))
    mask_3d = load_mask(os.path.join(DATA_DIR, 'mask256.mat'), r, c, nC, step)
    meas = generate_measurement(truth, mask_3d, step)
    np.testing.assert_allclose(meas.sum(), float(fixtures['meas_sum']), rtol=1e-6)

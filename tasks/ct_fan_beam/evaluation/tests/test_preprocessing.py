"""Tests for src/preprocessing.py."""
import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.preprocessing import (
    load_sinogram_data, load_ground_truth, load_metadata, preprocess_sinogram,
)


@pytest.fixture
def data_dir():
    return TASK_DIR


def test_sinogram_shapes(data_dir):
    sino_full, sino_short, af, ash, dp = load_sinogram_data(TASK_DIR)
    assert sino_full.ndim == 3
    assert sino_full.shape[0] == 1
    assert sino_short.shape[0] == 1
    assert len(af) == sino_full.shape[1]
    assert len(ash) == sino_short.shape[1]
    assert len(dp) == sino_full.shape[2]


def test_sinogram_dtype(data_dir):
    sino_full, _, _, _, _ = load_sinogram_data(TASK_DIR)
    assert sino_full.dtype == np.float64


def test_ground_truth_shape(data_dir):
    phantom = load_ground_truth(TASK_DIR)
    assert phantom.ndim == 3
    assert phantom.shape[0] == 1


def test_ground_truth_range(data_dir):
    phantom = load_ground_truth(TASK_DIR)
    assert phantom.min() >= 0
    assert phantom.max() > 0


def test_metadata_keys(data_dir):
    meta = load_metadata(TASK_DIR)
    required = ['image_size', 'n_det', 'source_to_isocenter_pixels',
                 'isocenter_to_detector_pixels', 'fan_half_angle_deg',
                 'short_scan_range_deg']
    for k in required:
        assert k in meta, f"Missing key: {k}"


def test_preprocess_removes_batch():
    sino = np.ones((1, 10, 48))
    result = preprocess_sinogram(sino)
    assert result.shape == (10, 48)


def test_preprocess_preserves_values():
    sino = np.array([[[1.0, 2.0, 3.0]]])
    result = preprocess_sinogram(sino)
    np.testing.assert_array_equal(result, [[1.0, 2.0, 3.0]])

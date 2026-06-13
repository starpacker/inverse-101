"""Tests for src/preprocessing.py."""
import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")

import sys
sys.path.insert(0, TASK_DIR)

from src.preprocessing import (
    load_sinogram_data, load_ground_truth, load_metadata, preprocess_sinogram,
)


class TestLoadData:
    def test_sinogram_shapes(self):
        sino, bg, theta = load_sinogram_data(TASK_DIR)
        assert sino.ndim == 3
        assert sino.shape[0] == 1  # batch dim
        assert bg.shape == sino.shape
        assert theta.ndim == 2
        assert theta.shape[0] == 1  # batch dim
        assert theta.shape[1] == sino.shape[2]  # n_angles

    def test_sinogram_dtype(self):
        sino, _, _ = load_sinogram_data(TASK_DIR)
        assert sino.dtype == np.float64

    def test_ground_truth_shape(self):
        gt = load_ground_truth(TASK_DIR)
        assert gt.ndim == 3
        assert gt.shape[0] == 1

    def test_ground_truth_non_negative(self):
        gt = load_ground_truth(TASK_DIR)
        assert np.all(gt >= 0)

    def test_metadata_keys(self):
        meta = load_metadata(TASK_DIR)
        required = ['image_size', 'n_angles', 'n_radial_bins',
                     'count_level', 'noise_model']
        for k in required:
            assert k in meta, f"Missing key: {k}"


class TestPreprocess:
    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "preprocessing_preprocess.npz"))
        result = preprocess_sinogram(fix["input_sinogram"])
        np.testing.assert_allclose(result, fix["output_sinogram"], rtol=1e-10)

    def test_batch_dim_removed(self):
        sino = np.ones((1, 10, 20))
        result = preprocess_sinogram(sino)
        assert result.shape == (10, 20)

    def test_non_negative(self):
        sino = np.array([[[-1.0, 2.0, -0.5]]])
        result = preprocess_sinogram(sino)
        assert np.all(result >= 0)

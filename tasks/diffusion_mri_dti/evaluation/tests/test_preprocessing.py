"""Tests for src/preprocessing.py."""

import numpy as np
import pytest
import sys, os

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_dwi_data, load_ground_truth, load_metadata, preprocess_dwi


class TestLoadData:
    def test_dwi_shape(self, task_dir):
        dwi, bvals, bvecs = load_dwi_data(task_dir)
        assert dwi.ndim == 4
        assert dwi.shape[0] == 1  # batch dim
        assert dwi.shape[3] == len(bvals)
        assert bvecs.shape == (len(bvals), 3)

    def test_dwi_dtype(self, task_dir):
        dwi, bvals, bvecs = load_dwi_data(task_dir)
        assert dwi.dtype == np.float64

    def test_ground_truth_shapes(self, task_dir):
        fa, md, tensor_elems, mask = load_ground_truth(task_dir)
        assert fa.ndim == 3
        assert fa.shape[0] == 1
        assert md.shape == fa.shape
        assert tensor_elems.shape[-1] == 6
        assert mask.dtype == bool

    def test_metadata_keys(self, task_dir):
        meta = load_metadata(task_dir)
        required_keys = ['image_size', 'n_directions', 'b_value_s_per_mm2',
                         'noise_sigma', 'signal_model']
        for key in required_keys:
            assert key in meta, f"Missing key: {key}"


class TestPreprocess:
    def test_batch_dim_removed(self):
        dwi = np.ones((1, 10, 10, 5))
        bvals = np.array([0, 1000, 1000, 1000, 1000], dtype=np.float64)
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float64)
        dwi_2d, S0 = preprocess_dwi(dwi, bvals, bvecs)
        assert dwi_2d.shape == (10, 10, 5)
        assert S0.shape == (10, 10)

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        dwi = rng.normal(0.5, 0.3, (1, 10, 10, 5))  # some negatives
        bvals = np.array([0, 1000, 1000, 1000, 1000], dtype=np.float64)
        bvecs = np.zeros((5, 3))
        dwi_2d, S0 = preprocess_dwi(dwi, bvals, bvecs)
        assert np.all(dwi_2d >= 0)
        assert np.all(S0 > 0)

    def test_S0_from_b0(self):
        """S0 is mean of b=0 volumes."""
        dwi = np.ones((1, 5, 5, 4))
        dwi[0, :, :, 0] = 2.0  # b=0 volume
        bvals = np.array([0, 1000, 1000, 1000], dtype=np.float64)
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        _, S0 = preprocess_dwi(dwi, bvals, bvecs)
        np.testing.assert_allclose(S0, 2.0)

"""Tests for src/preprocessing.py"""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_observation, load_ground_truth, load_metadata, prepare_data


@pytest.fixture
def data_dir():
    return os.path.join(TASK_DIR, 'data')


class TestLoadObservation:
    def test_keys(self, data_dir):
        obs = load_observation(data_dir)
        assert 'undersampled_kspace' in obs
        assert 'undersampling_masks' in obs

    def test_kspace_shape(self, data_dir):
        obs = load_observation(data_dir)
        assert obs['undersampled_kspace'].shape == (20, 128, 128)

    def test_kspace_dtype(self, data_dir):
        obs = load_observation(data_dir)
        assert obs['undersampled_kspace'].dtype == np.complex128

    def test_masks_shape(self, data_dir):
        obs = load_observation(data_dir)
        assert obs['undersampling_masks'].shape == (20, 128, 128)

    def test_masks_binary(self, data_dir):
        obs = load_observation(data_dir)
        masks = obs['undersampling_masks']
        assert set(np.unique(masks)).issubset({0.0, 1.0})


class TestLoadGroundTruth:
    def test_keys(self, data_dir):
        gt = load_ground_truth(data_dir)
        assert 'dynamic_images' in gt
        assert 'time_points' in gt

    def test_images_shape(self, data_dir):
        gt = load_ground_truth(data_dir)
        assert gt['dynamic_images'].shape == (20, 128, 128)

    def test_time_points_shape(self, data_dir):
        gt = load_ground_truth(data_dir)
        assert gt['time_points'].shape == (20,)

    def test_non_negative(self, data_dir):
        gt = load_ground_truth(data_dir)
        assert gt['dynamic_images'].min() >= 0.0

    def test_time_monotonic(self, data_dir):
        gt = load_ground_truth(data_dir)
        assert np.all(np.diff(gt['time_points']) > 0)


class TestLoadMetadata:
    def test_required_keys(self, data_dir):
        meta = load_metadata(data_dir)
        assert 'image_size' in meta
        assert 'num_frames' in meta
        assert 'sampling_rate' in meta

    def test_values(self, data_dir):
        meta = load_metadata(data_dir)
        assert meta['image_size'] == 128
        assert meta['num_frames'] == 20
        assert 0 < meta['sampling_rate'] < 1


class TestPrepareData:
    def test_returns_three(self, data_dir):
        obs, gt, meta = prepare_data(data_dir)
        assert isinstance(obs, dict)
        assert isinstance(gt, dict)
        assert isinstance(meta, dict)
